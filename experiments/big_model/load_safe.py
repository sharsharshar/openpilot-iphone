"""Allowlisted loader for openpilot out-of-band tinygrad pickles (shared by run_big.py / struct dumps)."""
import io, pickle, struct, os

ALLOWED = {  # exactly the globals found by inspect_pkl.py in both pickles; all are tinygrad's own classes
  ('tinygrad.device', 'BufferSpec'), ('tinygrad.dtype', 'AddrSpace'), ('tinygrad.dtype', 'DType'),
  ('tinygrad.engine.jit', 'CapturedJit'), ('tinygrad.engine.jit', '_TinyJit'), ('tinygrad.helpers', 'Target'),
  ('tinygrad.renderer', 'Estimates'), ('tinygrad.runtime.support.hcq2', 'HCQInfo'), ('tinygrad.uop', 'Ops'),
  ('tinygrad.uop.ops', 'AxisType'), ('tinygrad.uop.ops', 'CallInfo'), ('tinygrad.uop.ops', 'CustomFunction'),
  ('tinygrad.uop.ops', 'KernelInfo'), ('tinygrad.uop.ops', 'ParamArg'), ('tinygrad.uop.ops', 'ProgramInfo'),
  ('tinygrad.uop.ops', 'UOp'),
}

class SafeUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    if (module, name) not in ALLOWED: raise pickle.UnpicklingError(f"global not in allowlist: {module}.{name}")
    return super().find_class(module, name)

def load_opcodes(path, persistent_load):
  with open(path, 'rb') as f:
    n = struct.unpack('<q', f.read(8))[0]
    ops = f.read(n)
  u = SafeUnpickler(io.BytesIO(ops)); u.persistent_load = persistent_load
  return u.load()

class Placeholder:  # stands in for an out-of-band Buffer view: (nbytes, offset) only, no data read
  def __init__(self, nbytes, off): self.nbytes, self.off = nbytes, off
  def __repr__(self): return f"Buf({self.nbytes}@{self.off})"
