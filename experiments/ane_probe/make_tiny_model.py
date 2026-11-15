"""Build two tiny ML Programs with coremltools' MIL builder (no PyTorch).

  tiny_conv.mlpackage : x [1,64,1,256] fp16 -> 1x1 conv with identity weights
  tiny_gelu.mlpackage : the same conv followed by gelu

Usage: python make_tiny_model.py [output_dir]
Each model is built independently; a failure is printed and the script still exits 0.
"""
import os
import sys
import traceback

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def build(with_gelu, out_dir, name):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 64, 1, 256), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        w = np.eye(64, dtype=np.float16).reshape(64, 64, 1, 1)
        y = mb.conv(x=x, weight=w, name="conv_out")
        if with_gelu:
            y = mb.gelu(x=y, name="gelu_out")
        return y

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    path = os.path.join(out_dir, name + ".mlpackage")
    model.save(path)
    print("saved", path)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    print("coremltools", ct.__version__, "numpy", np.__version__)
    for with_gelu, name in ((False, "tiny_conv"), (True, "tiny_gelu")):
        try:
            build(with_gelu, out_dir, name)
        except Exception:
            print("FAILED to build", name)
            traceback.print_exc()


if __name__ == "__main__":
    main()
