"""onnx2torch 1.5.15 has no converter for Pad at opset 19+. This adds one (constant mode, constant pads / value), on top of
scripts/onnx2torch_shim.py (imported, not modified). Only what the variants use: Pad with mode=constant, `pads` and
`constant_value` as initializers."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import onnx2torch_shim  # noqa: F401  (registers Gelu, Reduce*, Reshape, ...)
import torch
from torch import nn
import torch.nn.functional as F
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class PadConst(nn.Module, OnnxToTorchModule):
    def __init__(self, pads_onnx, value):
        super().__init__()
        n = len(pads_onnx) // 2
        begin, end = pads_onnx[:n], pads_onnx[n:]
        self.pads = [p for i in reversed(range(n)) for p in (begin[i], end[i])]   # torch: last dim first
        self.value = float(value)
    def forward(self, x):
        return F.pad(x, self.pads, mode="constant", value=self.value)


def _pad(node, graph):
    assert node.attributes.get("mode", "constant") == "constant"
    pads = [int(v) for v in graph.initializers[node.input_values[1]].to_torch().tolist()]
    value = 0.0
    if len(node.input_values) > 2 and node.input_values[2]:
        value = graph.initializers[node.input_values[2]].to_torch().item()
    return OperationConverterResult(torch_module=PadConst(pads, value),
                                    onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values))


for v in (19, 20, 21):
    add_converter(operation_type="Pad", version=v)(_pad)


class Rsqrt(nn.Module, OnnxToTorchModule):
    def forward(self, x):
        return torch.rsqrt(x)


@add_converter(operation_type="Rsqrt", version=1, domain="av")
def _rsqrt(node, graph):
    return OperationConverterResult(torch_module=Rsqrt(), onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values))


def fuse_rsqrt(model):
    """Conversion-time only: Sqrt -> Reciprocal (Sqrt output used once) becomes one node `av::Rsqrt`, so that torch.rsqrt is traced.
    coremltools fuses mean/sub/square/mean/add/rsqrt/mul written this way (Apple's LayerNormANE form) into one `layer_norm` op
    over the channel axis, while sqrt+reciprocal or sqrt+div are not fused. Same arithmetic (1/sqrt(x)); the ONNX that is
    verified with onnxruntime is the unfused one."""
    from onnx import helper
    g = model.graph
    uses = {}
    for n in g.node:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    prod = {o: n for n in g.node for o in n.output}
    drop, new = set(), []
    for n in g.node:
        if n.op_type == "Reciprocal" and n.input[0] in prod and prod[n.input[0]].op_type == "Sqrt" and uses[n.input[0]] == 1:
            sq = prod[n.input[0]]
            drop.add(id(sq))
            new.append(helper.make_node("Rsqrt", list(sq.input), list(n.output), name=n.name + "_rsqrt", domain="av"))
        else:
            new.append(n)
    new = [n for n in new if id(n) not in drop]
    del g.node[:]; g.node.extend(new)
    if not any(o.domain == "av" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("av", 1))
    return len(drop)
