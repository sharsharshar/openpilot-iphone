#!/usr/bin/env python3
"""Safe static inspection of an openpilot out-of-band tinygrad pickle. Never unpickles.
Layout (tinygrad examples/openpilot/helpers.py dump_pickle): <q length><pickle opcodes><256-aligned raw buffer bytes>.
Usage: python -I inspect_pkl.py PATH"""
import sys, struct, pickletools, collections, os

def main(path):
  size = os.path.getsize(path)
  with open(path, 'rb') as f:
    n = struct.unpack('<q', f.read(8))[0]
    ops = f.read(n)
  print(f"file bytes {size}; opcode stream bytes {n}; out-of-band buffer region bytes {size-8-n}")
  memo, strs, globs = {}, [], collections.Counter()
  perm = collections.Counter(); persids = 0; pid_sizes = []; nstr = collections.Counter()
  opcount = collections.Counter(); last_pid = None
  stack_marks = 0; memo_idx = 0
  lastvals = []  # recent pushed values (strings or ('memo',i))
  for op, arg, pos in pickletools.genops(ops):
    opcount[op.name] += 1
    if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE'):
      lastvals.append(arg); nstr[arg] += 1
    elif op.name in ('BINGET', 'LONG_BINGET'):
      lastvals.append(memo.get(arg))
    elif op.name == 'MEMOIZE':
      memo[memo_idx] = lastvals[-1] if lastvals else None; memo_idx += 1
    elif op.name in ('BINPUT', 'LONG_BINPUT'):
      memo[arg] = lastvals[-1] if lastvals else None
    elif op.name == 'GLOBAL':
      globs[tuple(arg.split(' '))] += 1
    elif op.name == 'STACK_GLOBAL':
      globs[(lastvals[-2], lastvals[-1])] += 1
      lastvals.append(None)
    elif op.name == 'BINPERSID':
      persids += 1
    elif op.name in ('REDUCE', 'BUILD', 'NEWOBJ', 'INST', 'OBJ'):
      perm[op.name] += 1
    else:
      lastvals.append(None) if op.name.startswith(('BININT', 'SHORT_BINBYTES', 'BINBYTES')) else None
  print(f"persistent-id references (out-of-band buffer views): {persids}")
  print("\nreferenced globals (module, name): count")
  for (m, nme), c in sorted(globs.items(), key=lambda x: (str(x[0]))):
    print(f"  {m}.{nme}: {c}")
  print("\nopcode counts:", dict(opcount.most_common(12)))
  devs = {s: c for s, c in nstr.items() if any(k in s for k in ('AMD', 'USB', 'CPU', 'QCOM', 'METAL', 'LLVM', 'CLANG', 'GPU', 'NPY', 'DISK'))}
  print("\ndevice-like strings:", devs)
  prog = sorted(s for s in nstr if s.startswith(('E_', 'r_', 'R_', 'cast', 'copy')) )
  print(f"\nkernel-name-like strings: {len(prog)} distinct")
  for s in prog: print(f"  {s} x{nstr[s]}")
  if '--strings' in sys.argv:
    for s, c in nstr.items(): print(repr(s[:100]), c)

if __name__ == '__main__': main(sys.argv[1])
