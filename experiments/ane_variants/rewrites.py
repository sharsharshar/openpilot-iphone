"""Exact graph rewrites of the lean driving model (models/driving_supercombo_lean_fp32.onnx).

Every function takes and returns a `GB` (see gb.py) and changes the graph without changing what it computes
(bit-identical, or float32-rounding-level where a sum is re-ordered or a constant is folded; each docstring says which).
Nothing here quantises, approximates GELU, or changes a weight's value other than by an exact constant fold.

Names of tensors in the lean graph that these rewrites rely on were read from the graph (see RESULTS.md); each
function asserts what it expects, so a different model fails loudly instead of being rewritten wrongly.
"""
import numpy as np
from onnx import TensorProto, helper, shape_inference

NEG_BIG = np.float32(-60000.0)   # stands in for -inf in the causal mask (see rw_A)


# ----------------------------------------------------------------------------------------------------------------
# A. exact replacements for the nodes the ANEForge importer cannot take
# ----------------------------------------------------------------------------------------------------------------
def rw_A(gb):
    """GatherND with constant indices -> Slice; Where(const mask, -inf, x) -> Add(x, const bias);
    Concat with a constant all-zero [1,2] operand -> zeros produced by the preceding Gemm (two extra zero rows).

    * GatherND(x[25,1,512], [[-9]..[-1]]) is rows 16..24: Slice 16:25. Bit-identical (pure data movement).
    * Where(mask, -inf, x) feeding a Softmax: Add(x, bias) with bias = -60000 where masked, else 0. x + 0 == x exactly;
      a masked score becomes about -60000, and exp(-60000 - rowmax) is exactly 0 in float32 (and float16), so the softmax
      output is bit-identical as long as every row has at least one unmasked entry (a causal mask: the diagonal).
      -60000 rather than -inf so that float16 hardware never meets an infinity.
    * Concat([plan 990], [desire_state 8], zeros[1,2]): the final desire_state Gemm gets two extra output rows with weight 0
      and bias 0, whose output is 0*x+0 = 0 for finite x; the Concat then has no constant operand. Bit-identical.
    """
    # 1. GatherND
    for n in gb.find(op="GatherND"):
        idx = gb.init[n.input[1]]
        assert idx.ndim == 2 and idx.shape[1] == 1 and gb.attr(n, "batch_dims", 0) == 0
        shp = gb.shapes()[n.input[0]]
        ix = [int(v) % shp[0] for v in idx[:, 0]]
        assert ix == list(range(ix[0], ix[0] + len(ix)))
        gb.remove(n)
        gb.add("Slice", [n.input[0], gb.ints([ix[0]]), gb.ints([ix[-1] + 1]), gb.ints([0])], out=n.output[0], hint="gathernd_as_slice")
        gb.log.append(f"GatherND {n.name} -> Slice[{ix[0]}:{ix[-1] + 1}]")
    # 2. Where
    for n in gb.find(op="Where"):
        cond, val = gb.init[n.input[0]].astype(bool), gb.init[n.input[1]]
        assert n.input[2] not in gb.init and np.all(np.isneginf(val))
        assert cond.shape[-2:] == (9, 9) and not cond.all(axis=-1).any(), "a fully masked row would not be exact"
        bias = np.where(cond, NEG_BIG, np.float32(0)).astype(np.float32)
        gb.remove(n)
        gb.add("Add", [n.input[2], gb.const(bias, "maskbias")], out=n.output[0], hint="where_as_add")
        gb.log.append(f"Where {n.name} -> Add(const bias {{0,-60000}})")
    # 3. Concat with a constant zero operand
    for n in gb.find(op="Concat"):
        consts = [x for x in n.input if x in gb.init]
        if not consts:
            continue
        assert len(consts) == 1 and not gb.init[consts[0]].any() and gb.init[consts[0]].shape == (1, 2) and gb.attr(n, "axis") == 1
        pos = list(n.input).index(consts[0])
        prev = gb.producer(n.input[pos - 1])
        assert prev.op_type == "Gemm" and gb.attr(prev, "transB") == 1 and gb.attr(prev, "transA", 0) == 0
        w, b = gb.init[prev.input[1]], gb.init[prev.input[2]]
        gb.init[prev.input[1] + "/pad2"] = np.concatenate([w, np.zeros((2, w.shape[1]), w.dtype)], 0)
        gb.init[prev.input[2] + "/pad2"] = np.concatenate([b, np.zeros(2, b.dtype)], 0)
        prev.input[1], prev.input[2] = prev.input[1] + "/pad2", prev.input[2] + "/pad2"
        gb.invalidate_shapes()
        del n.input[pos]
        gb.log.append(f"Concat {n.name}: zero operand moved into Gemm {prev.name} (+2 zero output rows)")
    return gb


# ----------------------------------------------------------------------------------------------------------------
# B. fold (x - 127.5) / 63.75 into the first convolution
# ----------------------------------------------------------------------------------------------------------------
def rw_B(gb):
    """Cast -> Sub(127.5) -> Div(63.75) -> Conv(3x3, stride 2, pad 1) becomes Pad(127.5) -> Conv(pad 0) on the raw values.

    Folding the scale and offset into the weights is exact in real arithmetic, except at the image border: the original
    zero-pads the NORMALISED image, i.e. pads with raw value 127.5, whereas a plain conv with zero padding on the raw image
    would pad with raw 0 (= -2.0 normalised). So the padding is made explicit: Pad(constant 127.5) on the raw input, then the
    conv with pads 0. Weights: w' = w / 63.75, bias: b' = b - sum(w * 127.5 / 63.75) over the 24 channels and 3x3 taps.
    The fold is computed in float64 and rounded to float32 once. Not bit-identical to the original (w / 63.75 is a new
    rounding, and the products are summed in a different order) but equal to float32 rounding level; asserted by verify_variants.py.
    The float->float Cast at the input disappears too.
    """
    sub = gb.one(op="Sub")
    div = gb.one(op="Div", out="model/div")
    assert gb.consumers(sub.output[0])[0] is div
    cast = gb.producer(sub.input[0])
    assert cast.op_type == "Cast" and gb.attr(cast, "to") == TensorProto.FLOAT and cast.input[0] == "img_stack"
    mean, std = gb.init[sub.input[1]], gb.init[div.input[1]]
    assert np.all(mean == mean.ravel()[0]) and np.all(std == std.ravel()[0]), "per-channel statistics differ"
    m, s = float(mean.ravel()[0]), float(std.ravel()[0])
    (conv,) = gb.consumers(div.output[0])
    assert conv.op_type == "Conv" and gb.attr(conv, "pads") == [1, 1, 1, 1] and gb.attr(conv, "group") == 1
    w, b = gb.init[conv.input[1]].astype(np.float64), gb.init[conv.input[2]].astype(np.float64)
    w2 = (w / s).astype(np.float32)
    b2 = (b - (w * (m / s)).sum(axis=(1, 2, 3))).astype(np.float32)
    gb.init[conv.input[1] + "/folded"], gb.init[conv.input[2] + "/folded"] = w2, b2
    conv.input[1], conv.input[2] = conv.input[1] + "/folded", conv.input[2] + "/folded"
    pads = gb.ints([0, 0, 1, 1, 0, 0, 1, 1])
    padded = gb.add("Pad", ["img_stack", pads, gb.const(np.float32(m), "padval")], hint="pad_raw_127_5", mode="constant")
    conv.input[0] = padded
    for a in conv.attribute:
        if a.name == "pads":
            del a.ints[:]
            a.ints.extend([0, 0, 0, 0])
    for n in (cast, sub, div):
        gb.remove(n)
    gb.log.append(f"normalisation folded into stem conv; Pad value {m}")
    return gb


# ----------------------------------------------------------------------------------------------------------------
# D. the queues are kept by the app
# ----------------------------------------------------------------------------------------------------------------
def rw_D(gb, cf):
    """The model no longer shifts the queues. The app passes only what the model reads and the model returns only `outputs`.

    state_feat_q [96,1,512]: the model reads rows 0,4,...,92 (a stride-4 slice), keeps the last 8 of those (rows 64,68,...,92),
    appends the new feature vector (which is also outputs[:, 1064:1576]) and uses these 9 rows. It writes
    next = concat(q[1:], new). New input `feat_hist` = q[64:96:4] ([1,8,512] token-major, or [1,512,1,8] channel-first when
    cf); the app keeps the 96-row ring and takes the new row from `outputs`. next_state_feat_q is no longer an output.

    state_desire_q [100,1,8]: next = concat(q[1:], desire); the model reduces next.view(25,4,8) with max over the group of 4.
    New input `desire_q` = that already-shifted queue, shaped [1,25,4,8] (same bytes as the app's [100,1,8] ring after it
    appended `desire`), or [1,200,4,1] when cf (feature index g*8+c on channels, group member on H: element (g,c,k) at
    (g*8+c)*4+k; needs the app to write the transposed layout). The model does one ReduceMax over the group axis.
    `desire` is no longer an input and next_state_desire_q no longer an output.
    Bit-identical (data selection and max only).
    """
    assert any(i.name == "state_feat_q" for i in gb.inputs)
    shp = gb.shapes()
    # ---- feature queue -> tokens
    tok = gb.one(out="model/p_index")           # Transpose of GatherND result: [1,9,512]
    assert shp["model/p_index"] == [1, 9, 512]
    hidden_t = "model/hidden_t"                 # [1,1,512]
    if cf:
        hist = "feat_hist"
        hidden4 = gb.add("Reshape", [hidden_t, gb.ints([1, 512, 1, 1])], hint="hidden4")
        gb.add("Concat", [hist, hidden4], out="model/tok4", hint="tok4_concat", axis=3)
        hist_info = ("feat_hist", [1, 512, 1, 8])
        gb.tok4 = "model/tok4"
    else:
        gb.remove(tok)
        gb.add("Concat", ["feat_hist", hidden_t], out="model/p_index", hint="tok_concat", axis=1)
        hist_info = ("feat_hist", [1, 8, 512])
    # ---- desire queue
    if cf:
        dq = ("desire_q", [1, 200, 4, 1])
        gb.remove(gb.one(out="model/p_view"))   # the old Reshape that produced p_view
        gb.add("ReduceMax", ["desire_q", gb.ints([2])], out="model/p_view", hint="desire_max", keepdims=1)
        gb.four.add("model/p_view")
    else:
        dq = ("desire_q", [1, 25, 4, 8])
        gb.remove(gb.find(op="Transpose", out="transpose")[0])
        gb.add("ReduceMax", ["desire_q", gb.ints([2])], out="transpose", hint="desire_max", keepdims=0)
    # ---- interface
    from gb import tensor_info
    ins = {i.name: i for i in gb.inputs}
    gb.inputs = [ins["img_stack"], tensor_info(*dq), ins["traffic_convention"], ins["action_t"], tensor_info(*hist_info)]
    gb.outputs = [o for o in gb.outputs if o.name == "outputs"]
    gb.invalidate_shapes()
    gb.log.append(f"queues moved to the app (cf={cf}): inputs {[i.name for i in gb.inputs]}, outputs ['outputs']")
    return gb


# ----------------------------------------------------------------------------------------------------------------
# E. other exact simplifications
# ----------------------------------------------------------------------------------------------------------------
def rw_E_simple(gb, keep_next_states=True):
    """Remove no-op nodes and small redundancies. Bit-identical except E5.

    E1  Cast(float32 -> float32): 6 nodes, removed.
    E2  Identity: 3 nodes, removed.
    E3  Clip(min=0) of a vector 2-norm (a norm is >= 0): removed.
    E4  Expand([1,1] -> [1,512]) feeding a Div: the Div broadcasts instead.
    E5  layer_scale: Conv -> Mul(gamma per channel): gamma folded into that conv's weights and bias (12 Mul nodes).
        Float32 rounding level, not bit-identical (w*gamma is a new rounding).
    E6  next_state_feat_q's new row is sliced out of `outputs` again; use the feature vector that `outputs` was built from.
    """
    for n in list(gb.find(op="Cast")):
        assert gb.attr(n, "to") == TensorProto.FLOAT
        gb.bypass(n)
    for n in list(gb.find(op="Identity")):
        gb.bypass(n)
    for n in list(gb.find(op="Clip")):
        p = gb.producer(n.input[0])
        assert p.op_type == "ReduceL2" and gb.init[n.input[1]].item() == 0.0 and len(n.input) == 2
        gb.bypass(n)
    for n in list(gb.find(op="Expand")):
        (d,) = gb.consumers(n.output[0])
        assert d.op_type == "Div" and d.input[1] == n.output[0]
        d.input[1] = n.input[0]
        gb.remove(n)
    folded = 0
    for n in list(gb.find(op="Mul")):
        if n.input[1] in gb.init and gb.init[n.input[1]].ndim == 3 and gb.init[n.input[1]].shape[1:] == (1, 1):
            conv = gb.producer(n.input[0])
            if conv is None or conv.op_type != "Conv" or len(gb.consumers(conv.output[0])) != 1:
                continue
            g = gb.init[n.input[1]].reshape(-1).astype(np.float64)
            w, b = gb.init[conv.input[1]].astype(np.float64), gb.init[conv.input[2]].astype(np.float64)
            gb.init[conv.input[1] + "/ls"] = (w * g[:, None, None, None]).astype(np.float32)
            gb.init[conv.input[2] + "/ls"] = (b * g).astype(np.float32)
            conv.input[1], conv.input[2] = conv.input[1] + "/ls", conv.input[2] + "/ls"
            gb.replace_uses(n.output[0], conv.output[0])
            gb.remove(n)
            folded += 1
    assert folded == 12, folded
    if any(o.name == "next_state_feat_q" for o in gb.outputs):
        cat = gb.one(out="next_state_feat_q")
        assert cat.input[1] == "unsqueeze_1"
        cat.input[1] = "model/hidden_t"
    gb.invalidate_shapes()
    gb.log.append("E1-E6 no-op/cast/identity/clip/expand removal, layer_scale folded (12), next_state_feat row from hidden")
    return gb


def rw_tokens_shortcut(gb, cf):
    """(no D) The 9 tokens are rows 64,68,...,92 of state_feat_q plus the new feature: 8 rows of a stride-4 slice.
    The original reads the stride-4 slice of all 96 rows (24 rows), transposes, casts, concatenates with the new row (25 rows),
    transposes, GatherND-selects the last 9 and transposes again; this takes Slice(start 64, step 4) of the queue directly.
    Bit-identical (data movement only). cf: the result is made channel-first [1,512,1,9] (tokens along the last axis)."""
    assert any(i.name == "state_feat_q" for i in gb.inputs)
    sl = gb.add("Slice", ["state_feat_q", gb.ints([64]), gb.ints([96]), gb.ints([0]), gb.ints([4])], hint="feat_rows")   # [8,1,512]
    if cf:
        r = gb.add("Reshape", [sl, gb.ints([8, 512, 1, 1])], hint="feat_r4")
        t = gb.add("Transpose", [r], hint="feat_t4", perm=[3, 1, 2, 0])                                                   # [1,512,1,8]
        h4 = gb.add("Reshape", ["model/hidden_t", gb.ints([1, 512, 1, 1])], hint="hidden4")
        gb.add("Concat", [t, h4], out="model/tok4", hint="tok4_concat", axis=3)
        gb.tok4 = "model/tok4"
    else:
        r = gb.add("Reshape", [sl, gb.ints([1, 8, 512])], hint="feat_r")
        gb.remove(gb.one(out="model/p_index"))
        gb.add("Concat", [r, "model/hidden_t"], out="model/p_index", hint="tok_concat", axis=1)
    gb.invalidate_shapes()
    return gb


def rw_tokens_from_chain_cf(gb):
    """(C without D/E) keep the original token path and convert its result to [1,512,1,9]: Transpose + Reshape."""
    t = gb.add("Transpose", ["model/p_index"], hint="tok_t", perm=[0, 2, 1])
    gb.add("Reshape", [t, gb.ints([1, 512, 1, 9])], out="model/tok4", hint="tok4")
    gb.tok4 = "model/tok4"
    return gb


# ----------------------------------------------------------------------------------------------------------------
# transformer block (temporal summariser), rebuilt from the same weights
# ----------------------------------------------------------------------------------------------------------------
def rw_transformer(gb, prune, cf):
    """Rebuild everything between the 9 tokens and `model/p_select` from the original weights.

    prune: only the last token's output is used (Gather index 8 at the end), and that token's causal-mask row is all
      unmasked, so only its query is needed: queries/attention rows/c_proj/LayerNorm 2/MLP of the other 8 tokens are
      dead compute and are not built; keys and values still use all 9 tokens. The mask disappears (row 8 has no masked
      entry). Mathematically identical; float32 rounding level because matrix products over fewer rows may sum in a
      different order. The 1/sqrt(64) = 0.125 score scale is folded into the query weights and bias (a power of two,
      so the fold is exact).
    cf: channel-first (B, C, 1, S) tensors as in Apple's ANE transformer article: every Linear is a 1x1 Conv, tokens lie along
      the last axis, LayerNorm is over channels (written out with ReduceMean as in Apple's LayerNormANE, since ONNX's
      LayerNormalization normalises over trailing axes). The 8 heads are a reshape of the channel axis.
    """
    I = gb.init
    def p(name):
        return I[name]
    # ---- locate weights through the graph
    def linear_after(mm_out_name):
        mm = gb.one(out=mm_out_name)
        assert mm.op_type == "MatMul"
        (add,) = gb.consumers(mm.output[0])
        assert add.op_type == "Add"
        return mm.input[1], [x for x in add.input if x != mm.output[0]][0], add.output[0], mm.input[0]
    W_e1, b_e1, _, _ = linear_after("model/p_val_13")
    W_e2, b_e2, e2_out, _ = linear_after("model/p_val_16")
    assert e2_out == "model/p_linear_1"
    W_qkv, b_qkv, _, _ = linear_after("model/p_val_29")
    W_pr, b_pr, _, _ = linear_after("model/p_val_58")
    W_fc, b_fc, _, _ = linear_after("model/p_val_62")
    W_mp, b_mp, _, _ = linear_after("model/p_val_64")
    ln1, ln2 = gb.one(out="model/p_layer_norm"), gb.one(out="model/p_layer_norm_1")
    eps = gb.attr(ln1, "epsilon")
    assert gb.attr(ln2, "epsilon") == eps and gb.attr(ln1, "axis") == -1
    pos = I["model/p_unsqueeze_2"]               # [1,9,512]
    assert pos.shape == (1, 9, 512)
    assert I["model/p_val_65"].item() == 8 and gb.one(out="model/p_select").op_type == "Gather"
    assert float(I["model/p_scalar_tensor_default"]) == 0.125
    d_vec, t_vec = "model/p_linear_3", "model/p_linear_5"      # [1,512] embeddings of desire / traffic convention
    mask = I["model/p_bitwise_not"].astype(bool) if "model/p_bitwise_not" in I else None
    f32 = np.float32
    Wq, Wk, Wv = (p(W_qkv)[:, i * 512:(i + 1) * 512] for i in range(3))
    bq, bk, bv = (p(b_qkv)[i * 512:(i + 1) * 512] for i in range(3))
    Wq, bq = Wq * f32(0.125), bq * f32(0.125)             # exact: power of two

    if not cf:
        assert prune
        x0 = "model/p_add_2"                                # [1,9,512] (tokens + position + desire + traffic), kept as is
        def lin(x, W, b, hint):
            mm = gb.add("MatMul", [x, gb.const(W, hint + "_w")], hint=hint + "_mm")
            return gb.add("Add", [mm, gb.const(b, hint + "_b")], hint=hint + "_add")
        a = gb.add("LayerNormalization", [x0, ln1.input[1], ln1.input[2]], hint="ln1", axis=-1, epsilon=eps)
        a_last = gb.add("Slice", [a, gb.ints([8]), gb.ints([9]), gb.ints([1])], hint="a_last")
        x0_last = gb.add("Slice", [x0, gb.ints([8]), gb.ints([9]), gb.ints([1])], hint="x0_last")
        q, k, v = lin(a_last, Wq, bq, "q"), lin(a, Wk, bk, "k"), lin(a, Wv, bv, "v")      # [1,1,512], [1,9,512], [1,9,512]
        qh = gb.add("Reshape", [q, gb.ints([1, 8, 1, 64])], hint="qh")
        kh = gb.add("Transpose", [gb.add("Reshape", [k, gb.ints([1, 9, 8, 64])], hint="kr")], hint="kh", perm=[0, 2, 3, 1])
        vh = gb.add("Transpose", [gb.add("Reshape", [v, gb.ints([1, 9, 8, 64])], hint="vr")], hint="vh", perm=[0, 2, 1, 3])
        att = gb.add("Softmax", [gb.add("MatMul", [qh, kh], hint="scores")], hint="softmax", axis=-1)
        ctx_m = gb.add("MatMul", [att, vh], hint="ctx")                       # [1,8,1,64]
        ctx = gb.add("Reshape", [ctx_m, gb.ints([1, 1, 512])], hint="ctx_r")
        x1 = gb.add("Add", [x0_last, lin(ctx, p(W_pr), p(b_pr), "proj")], hint="res1")
        m = gb.add("LayerNormalization", [x1, ln2.input[1], ln2.input[2]], hint="ln2", axis=-1, epsilon=eps)
        f = gb.add("Gelu", [lin(m, p(W_fc), p(b_fc), "fc")], hint="gelu", approximate="tanh")
        x2 = gb.add("Add", [x1, lin(f, p(W_mp), p(b_mp), "mproj")], hint="res2")
        for n in gb.find(op="Gather", out="model/p_select"):      # the old Gather that produced p_select
            gb.remove(n)
        gb.add("Reshape", [x2, gb.ints([1, 512])], out="model/p_select", hint="select")
        gb.invalidate_shapes()
        gb.log.append("transformer: single-query attention (token-major), q/k/v split, mask dropped, scale folded")
        return gb

    # ---------------- channel-first ----------------
    tok4 = gb.tok4

    def conv(x, W, b, hint):                         # W given as [in, out] (x @ W)
        return gb.add("Conv", [x, gb.const(np.ascontiguousarray(W.T)[:, :, None, None], hint + "_w"), gb.const(b, hint + "_b")],
                      hint=hint, kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1, dilations=[1, 1])

    def ln_cf(x, ln, hint):
        g4 = gb.const(p(ln.input[1]).reshape(1, 512, 1, 1), hint + "_g")
        b4 = gb.const(p(ln.input[2]).reshape(1, 512, 1, 1), hint + "_b")
        ax = gb.ints([1])
        mean = gb.add("ReduceMean", [x, ax], hint=hint + "_mean", keepdims=1)
        zm = gb.add("Sub", [x, mean], hint=hint + "_zm")
        var = gb.add("ReduceMean", [gb.add("Mul", [zm, zm], hint=hint + "_sq"), ax], hint=hint + "_var", keepdims=1)
        den = gb.add("Sqrt", [gb.add("Add", [var, gb.const(f32(eps), hint + "_eps")], hint=hint + "_ve")], hint=hint + "_sqrt")
        y = gb.add("Div", [zm, den], hint=hint + "_norm")
        return gb.add("Add", [gb.add("Mul", [y, g4], hint=hint + "_scale"), b4], hint=hint + "_shift")

    def vec4(name):          # [1,512] -> [1,512,1,1]
        return gb.add("Reshape", [name, gb.ints([1, 512, 1, 1])], hint="vec4")

    e = conv(tok4, p(W_e1), p(b_e1), "enc1")
    e = gb.add("Mul", [e, gb.add("Sigmoid", [e], hint="enc_sig")], hint="enc_silu")
    e = conv(e, p(W_e2), p(b_e2), "enc2")
    pos4 = gb.const(np.ascontiguousarray(pos.transpose(0, 2, 1)).reshape(1, 512, 1, 9), "pos4")
    x0 = gb.add("Add", [gb.add("Add", [gb.add("Add", [e, pos4], hint="x0a"), vec4(d_vec)], hint="x0b"), vec4(t_vec)], hint="x0")  # [1,512,1,9]
    a = ln_cf(x0, ln1, "ln1")
    k4, v4 = conv(a, Wk, bk, "k"), conv(a, Wv, bv, "v")
    kh = gb.add("Reshape", [k4, gb.ints([1, 8, 64, 9])], hint="kh")
    vh = gb.add("Reshape", [v4, gb.ints([1, 8, 64, 9])], hint="vh")
    if prune:
        a_last = gb.add("Slice", [a, gb.ints([8]), gb.ints([9]), gb.ints([3])], hint="a_last")
        x_res = gb.add("Slice", [x0, gb.ints([8]), gb.ints([9]), gb.ints([3])], hint="x0_last")
        q4 = conv(a_last, Wq, bq, "q")                                                  # [1,512,1,1]
        qh = gb.add("Reshape", [q4, gb.ints([1, 8, 1, 64])], hint="qh")
        att = gb.add("Softmax", [gb.add("MatMul", [qh, kh], hint="scores")], hint="softmax", axis=-1)     # [1,8,1,9]
        attT = gb.add("Reshape", [att, gb.ints([1, 8, 9, 1])], hint="attT")                              # singleton swap: no data movement
        ctx_m = gb.add("MatMul", [vh, attT], hint="ctx")                                                 # [1,8,64,1]
        ctx = gb.add("Reshape", [ctx_m, gb.ints([1, 512, 1, 1])], hint="ctx_r")
    else:
        q4 = conv(a, Wq, bq, "q")
        qh = gb.add("Reshape", [q4, gb.ints([1, 8, 64, 9])], hint="qh")
        qT = gb.add("Transpose", [qh], hint="qT", perm=[0, 1, 3, 2])                                      # [1,8,9,64]
        sc = gb.add("MatMul", [qT, kh], hint="scores")                                                    # [1,8,9(q),9(k)]
        assert mask is not None and mask.shape == (1, 1, 9, 9)
        sc = gb.add("Add", [sc, gb.const(np.where(mask, NEG_BIG, f32(0)).astype(f32), "maskbias")], hint="masked")
        att = gb.add("Softmax", [sc], hint="softmax", axis=-1)
        attT = gb.add("Transpose", [att], hint="attT", perm=[0, 1, 3, 2])                                 # [1,8,9(k),9(q)]
        ctx_m = gb.add("MatMul", [vh, attT], hint="ctx")                                                  # [1,8,64,9]
        ctx = gb.add("Reshape", [ctx_m, gb.ints([1, 512, 1, 9])], hint="ctx_r")
        x_res = x0
    x1 = gb.add("Add", [x_res, conv(ctx, p(W_pr), p(b_pr), "proj")], hint="res1")
    m = ln_cf(x1, ln2, "ln2")
    f = gb.add("Gelu", [conv(m, p(W_fc), p(b_fc), "fc")], hint="gelu", approximate="tanh")
    x2 = gb.add("Add", [x1, conv(f, p(W_mp), p(b_mp), "mproj")], hint="res2")
    for n in gb.find(op="Gather", out="model/p_select"):          # the old Gather that produced p_select
        gb.remove(n)
    if prune:
        gb.add("Identity", [x2], out="model/p_select", hint="select4")
    else:
        gb.add("Slice", [x2, gb.ints([8]), gb.ints([9]), gb.ints([3])], out="model/p_select", hint="select4")
    gb.four.add("model/p_select")
    gb.invalidate_shapes()
    gb.log.append(f"transformer rebuilt channel-first (prune={prune})")
    return gb


# ----------------------------------------------------------------------------------------------------------------
# C. fully connected layers as 1x1 convolutions on [1, C, 1, 1]
# ----------------------------------------------------------------------------------------------------------------
def rw_C_fc(gb):
    """Every Gemm (y = x W^T + b on [1,K]) becomes Conv(1x1) on [1,K,1,1]; the vector path between them (Relu, Add, Mul, Div,
    Sigmoid, Concat on axis 1, Slice on axis 1, ReduceL2, Expand) runs on 4D channel-first tensors, so the Reshape between
    the pooled feature map and the first Gemm, and all Unsqueezes, disappear; one Reshape returns `outputs` to [1,2576].
    The same sums are computed in the same order per output (a 1x1 conv over C channels is a dot product); equal to float32
    rounding level (ORT may use a different kernel)."""
    gb.sort()
    shp = dict(gb.shapes())
    four = set(gb.four)
    out_nodes = []
    gb.sink = out_nodes
    outs = {o.name for o in gb.outputs}
    n_gemm = 0
    def to4(t):
        if t in four:
            return t
        if t in gb.init:
            a = gb.init[t]
            assert a.ndim <= 2 and (a.ndim == 0 or a.size == a.shape[-1]) or a.ndim == 1, (t, a.shape)
            return t if a.ndim == 0 else gb.const(a.reshape(1, -1, 1, 1), "c4")
        s = shp[t]
        assert len(s) == 2 and s[0] == 1, (t, s)
        r = gb.add("Reshape", [t, gb.ints([1, s[1], 1, 1])], hint="enter4")
        return r
    def back(t):
        s = shp[t]
        return gb.add("Reshape", [t, gb.ints(s)], hint="exit4")
    for n in gb.nodes:
        ins = [x for x in n.input if x]
        any4 = any(x in four for x in ins)
        op = n.op_type
        if op == "Gemm":
            assert gb.attr(n, "transB") == 1 and gb.attr(n, "transA", 0) == 0 and gb.attr(n, "alpha", 1.0) == 1.0 and gb.attr(n, "beta", 1.0) == 1.0
            W, b = gb.init[n.input[1]], gb.init[n.input[2]]
            x = to4(n.input[0])
            Wn = gb.const(W.reshape(W.shape[0], W.shape[1], 1, 1), "fcw")
            gb.add("Conv", [x, Wn, n.input[2]], out=n.output[0], hint="fc_conv", kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1, dilations=[1, 1])
            four.add(n.output[0])
            n_gemm += 1
            continue
        if not any4:
            out_nodes.append(n)
            continue
        if op in ("Relu", "Sigmoid", "Cast", "Identity"):
            out_nodes.append(n)
            four.update(n.output)
        elif op == "Clip":
            out_nodes.append(n)
            four.update(n.output)
        elif op in ("Add", "Mul", "Div", "Sub"):
            n.input[:] = [to4(x) for x in n.input]
            out_nodes.append(n)
            four.update(n.output)
        elif op == "Concat":
            assert gb.attr(n, "axis") == 1
            n.input[:] = [to4(x) for x in n.input]
            out_nodes.append(n)
            four.update(n.output)
        elif op == "Slice":
            ax = gb.init[n.input[3]].tolist()
            assert ax == [1] and n.input[0] in four
            out_nodes.append(n)
            four.update(n.output)
        elif op == "ReduceL2":
            assert gb.init[n.input[1]].tolist() == [-1] and gb.attr(n, "keepdims") == 1
            n.input[1] = gb.ints([1])
            out_nodes.append(n)
            four.update(n.output)
        elif op == "Expand":
            s = gb.init[n.input[1]].tolist()
            assert len(s) == 2
            n.input[1] = gb.ints([s[0], s[1], 1, 1])
            out_nodes.append(n)
            four.update(n.output)
        elif op in ("Reshape", "Unsqueeze") and n.input[0] in four and n.output[0] not in outs:
            tgt = shp[n.output[0]]
            src = shp[n.input[0]]
            nsrc = int(np.prod(src))
            vec = len(tgt) <= 3 and int(np.prod(tgt)) == nsrc and tgt[-1] == nsrc and all(d == 1 for d in tgt[:-1])
            if vec and op == "Reshape" and len(tgt) == 2:
                gb.replace_uses(n.output[0], n.input[0])                 # [1,C,1,1] -> [1,C]: stay 4D
            elif vec and op == "Reshape" and len(tgt) == 4 and tgt[1] == nsrc:
                gb.replace_uses(n.output[0], n.input[0])                 # [1,C] -> [1,C,1,1]: already there
            elif vec and op == "Unsqueeze":
                gb.replace_uses(n.output[0], n.input[0])
            else:
                n.input[0] = back(n.input[0]) if n.input[0] in four and len(shp[n.input[0]]) == 2 else n.input[0]
                out_nodes.append(n)
        else:
            # unknown consumer of a 4D tensor: bring it back to its original 2D shape first
            n.input[:] = [back(x) if x in four and len(shp[x]) == 2 else x for x in n.input]
            out_nodes.append(n)
    gb.nodes = out_nodes
    gb.sink = None
    # graph outputs still in 4D go back to their declared shape
    for o in gb.outputs:
        if o.name in four and len(shp[o.name]) == 2:
            prod = [n for n in gb.nodes if o.name in n.output][0]
            new = o.name + "/4d"
            for k, x in enumerate(prod.output):
                if x == o.name:
                    prod.output[k] = new
            gb.replace_uses(o.name, new)
            gb.nodes.append(helper.make_node("Reshape", [new, gb.ints(shp[o.name])], [o.name], name=gb.uid("out_reshape")))
    gb.invalidate_shapes()
    gb.log.append(f"C: {n_gemm} Gemm -> Conv1x1 on [1,C,1,1]")
    return gb


def rw_cleanup_reshapes(gb):
    """Collapse Reshape(Reshape(x)) and drop Reshapes that do not change the shape. Pure metadata; bit-identical."""
    changed = True
    while changed:
        changed = False
        shp = gb.shapes()
        for n in list(gb.nodes):
            if n.op_type != "Reshape":
                continue
            if shp[n.input[0]] == shp[n.output[0]] and n.output[0] not in {o.name for o in gb.outputs}:
                gb.replace_uses(n.output[0], n.input[0]); gb.remove(n); changed = True; break
            p = gb.producer(n.input[0])
            if p is not None and p.op_type == "Reshape" and len(gb.consumers(p.output[0])) == 1:
                n.input[0] = p.input[0]; gb.remove(p); changed = True; break
        gb.invalidate_shapes()
    return gb
