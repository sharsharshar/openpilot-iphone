"""Small ONNX graph-editing helper used by rewrites.py (no dependencies beyond onnx + numpy)."""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference


class GB:
    def __init__(self, model):
        self.m = copy.deepcopy(model)
        g = self.m.graph
        self.init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        self.nodes = list(g.node)
        self.inputs = list(g.input)
        self.outputs = list(g.output)
        self._n = 0
        self.log = []
        self.sink = None          # when a list, add() appends there instead of to self.nodes
        self.four = set()         # tensors known to be [1,C,1,1] vectors (channel-first); used by rewrites.rw_C_fc
        self.tok4 = None          # name of the channel-first token tensor, set by the token builders
        self._shapes = None

    # ---- naming / creation -------------------------------------------------------------------------------
    def uid(self, hint):
        self._n += 1
        return f"av/{hint}_{self._n}"

    def const(self, arr, hint="c"):
        name = self.uid(hint)
        self.init[name] = np.asarray(arr)
        return name

    def ints(self, vals, hint="i"):
        return self.const(np.asarray(vals, np.int64), hint)

    def add(self, op, ins, out=None, n_out=1, hint=None, **attrs):
        outs = [out] if out is not None and n_out == 1 else [self.uid((hint or op.lower()) + "_o") for _ in range(n_out)]
        node = helper.make_node(op, list(ins), outs, name=self.uid((hint or op.lower())), **attrs)
        (self.sink if self.sink is not None else self.nodes).append(node)
        return outs[0] if n_out == 1 else outs

    def invalidate_shapes(self):
        self._shapes = None

    def shapes(self):
        """Static shapes of every tensor (onnx shape inference on a light copy of the graph: weights are given as shapes only)."""
        if self._shapes is None:
            self.prune()
            small = {k: v for k, v in self.init.items() if v.size <= 1024}      # shape-like constants must keep their values
            big = {k: v for k, v in self.init.items() if v.size > 1024}
            ins = list(self.inputs) + [helper.make_tensor_value_info(k, TensorProto.FLOAT if v.dtype.kind == "f" else TensorProto.INT64 if v.dtype.kind == "i" else TensorProto.BOOL, list(v.shape))
                                       for k, v in big.items()]
            g = helper.make_graph(list(self.nodes), "tmp", ins, [], initializer=[numpy_helper.from_array(v, k) for k, v in small.items()])
            m = helper.make_model(g, opset_imports=self.m.opset_import)
            m.ir_version = self.m.ir_version
            inf = shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
            sh = {}
            for v in list(inf.graph.value_info) + list(inf.graph.input):
                if v.type.HasField("tensor_type") and v.type.tensor_type.HasField("shape"):
                    sh[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
            self._shapes = sh
        return self._shapes

    def prune(self):
        """Dead-code elimination from the graph outputs."""
        prod = {o: n for n in self.nodes for o in n.output}
        live, stack, keep = set(), [o.name for o in self.outputs], set()
        while stack:
            t = stack.pop()
            if t in live:
                continue
            live.add(t)
            n = prod.get(t)
            if n is not None and id(n) not in keep:
                keep.add(id(n))
                stack.extend(x for x in n.input if x)
        self.nodes = [n for n in self.nodes if id(n) in keep]

    def sort(self):
        self.prune()
        avail = {i.name for i in self.inputs} | set(self.init)
        ordered, pending = [], list(self.nodes)
        while pending:
            rest = []
            for n in pending:
                if all((not x) or x in avail for x in n.input):
                    ordered.append(n); avail.update(n.output)
                else:
                    rest.append(n)
            assert len(rest) < len(pending), "cycle or dangling input: " + str([(n.name, [x for x in n.input if x and x not in avail]) for n in rest[:3]])
            pending = rest
        self.nodes = ordered

    # ---- queries -------------------------------------------------------------------------------------------
    def producer(self, t):
        for n in self.nodes:
            if t in n.output:
                return n
        return None

    def consumers(self, t):
        return [n for n in self.nodes if t in n.input]

    def find(self, op=None, name=None, out=None):
        r = [n for n in self.nodes if (op is None or n.op_type == op) and (name is None or n.name == name) and (out is None or out in n.output)]
        return r

    def one(self, **kw):
        r = self.find(**kw)
        assert len(r) == 1, (kw, len(r))
        return r[0]

    @staticmethod
    def attr(n, k, default=None):
        for a in n.attribute:
            if a.name == k:
                return helper.get_attribute_value(a)
        return default

    # ---- editing -----------------------------------------------------------------------------------------
    def remove(self, node):
        self.nodes = [n for n in self.nodes if n is not node]

    def replace_uses(self, old, new):
        for n in self.nodes:
            for i, x in enumerate(n.input):
                if x == old:
                    n.input[i] = new

    def bypass(self, node, src_idx=0):
        """Delete a single-input pass-through node (Identity, float->float Cast, ...)."""
        src, dst = node.input[src_idx], node.output[0]
        outs = {o.name for o in self.outputs}
        if dst in outs:
            p = self.producer(src)
            assert p is not None and src not in outs, "cannot bypass into a graph input / double output"
            self.replace_uses(src, dst)
            for k, o in enumerate(p.output):
                if o == src:
                    p.output[k] = dst
        else:
            self.replace_uses(dst, src)
        self.remove(node)

    # ---- finish --------------------------------------------------------------------------------------------
    def finalize(self, keep_inputs=True):
        """Dead-code elimination, topological sort, rebuild the ModelProto, run the ONNX checker."""
        outs = {o.name for o in self.outputs}
        live, stack = set(), list(outs)
        prod = {o: n for n in self.nodes for o in n.output}
        needed_nodes = set()
        while stack:
            t = stack.pop()
            if t in live:
                continue
            live.add(t)
            n = prod.get(t)
            if n is not None and id(n) not in needed_nodes:
                needed_nodes.add(id(n))
                stack.extend(x for x in n.input if x)
        nodes = [n for n in self.nodes if id(n) in needed_nodes]
        # topological sort (stable)
        avail = {i.name for i in self.inputs} | set(self.init)
        ordered, pending = [], list(nodes)
        while pending:
            progressed = False
            rest = []
            for n in pending:
                if all((not x) or x in avail for x in n.input):
                    ordered.append(n)
                    avail.update(n.output)
                    progressed = True
                else:
                    rest.append(n)
            assert progressed, "cycle or dangling input: " + str([(n.name, [x for x in n.input if x and x not in avail]) for n in rest[:3]])
            pending = rest
        g = self.m.graph
        used = {x for n in ordered for x in n.input}
        del g.node[:]
        g.node.extend(ordered)
        del g.initializer[:]
        g.initializer.extend(numpy_helper.from_array(a, k) for k, a in self.init.items() if k in used)
        del g.input[:]
        g.input.extend(self.inputs)
        del g.output[:]
        g.output.extend(self.outputs)
        del g.value_info[:]
        onnx.checker.check_model(self.m)
        return self.m


def tensor_info(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, list(shape))
