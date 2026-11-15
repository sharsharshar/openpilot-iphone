"""Fills the gaps onnx2torch 1.5.15 has for opset-20 models: Gelu (missing entirely), newer
versions of Cast/Identity/Reshape (unchanged semantics, so aliased), and Reduce* v18+
(axes moved from an attribute to an input)."""
import torch
from torch import nn
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY as REG, add_converter, OperationDescription
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, OnnxMapping, onnx_mapping_from_node

for op, new, old in [("Cast", 19, 13), ("Identity", 19, 16)]:
    REG[OperationDescription(domain="", operation_type=op, version=new)] = REG[OperationDescription(domain="", operation_type=op, version=old)]

@add_converter(operation_type="Gelu", version=20)
def _gelu(node, graph):
    return OperationConverterResult(torch_module=nn.GELU(approximate=node.attributes.get("approximate", "none")),
                                    onnx_mapping=onnx_mapping_from_node(node))

class Reduce(nn.Module, OnnxToTorchModule):
    def __init__(self, kind, axes, keepdims):
        super().__init__(); self.kind, self.axes, self.keepdims = kind, axes, keepdims
    def forward(self, x):
        dims = self.axes if self.axes is not None else list(range(x.dim()))
        if self.kind == "ReduceMean": return x.mean(dim=dims, keepdim=self.keepdims)
        if self.kind == "ReduceMax": return torch.amax(x, dim=dims, keepdim=self.keepdims)
        return torch.sqrt((x * x).sum(dim=dims, keepdim=self.keepdims))  # ReduceL2

def _reduce(node, graph):
    axes = None
    if len(node.input_values) > 1 and node.input_values[1]:
        axes = [int(a) for a in graph.initializers[node.input_values[1]].to_torch().tolist()]
    return OperationConverterResult(
        torch_module=Reduce(node.operation_type, axes, bool(node.attributes.get("keepdims", 1))),
        onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values))

for op, v in [("ReduceMean", 18), ("ReduceL2", 18), ("ReduceMax", 20)]:
    add_converter(operation_type=op, version=v)(_reduce)

class Reshape(nn.Module, OnnxToTorchModule):
    """Reshape v19. allowzero=1 means a 0 in the shape is a literal zero, which is what torch does anyway."""
    def __init__(self, allowzero):
        super().__init__(); self.allowzero = allowzero
    def forward(self, x, shape):
        shape = [int(s) for s in shape.tolist()]
        if not self.allowzero:
            shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return torch.reshape(x, shape)

@add_converter(operation_type="Reshape", version=19)
def _reshape(node, graph):
    return OperationConverterResult(torch_module=Reshape(bool(node.attributes.get("allowzero", 0))),
                                    onnx_mapping=onnx_mapping_from_node(node))
