#!/usr/bin/env python3
"""Safe static inspection of an openpilot out-of-band tinygrad pickle. Never unpickles.
Layout (tinygrad examples/openpilot/helpers.py dump_pickle): <q length><pickle opcodes><256-aligned raw buffer bytes>.
Usage: python -I inspect_pkl.py PATH"""
import sys, struct, pickletools, collections, os

def main(path):
  size = os.path.getsize(path)
  with open(path, 'rb') as f:
    head = f.read(10); f.seek(0)
    n0 = struct.unpack('<q', head[:8])[0]
    if 0 < n0 < size and head[8] == 0x80:   # out-of-band container: <q len> + protocol-N opcodes + raw buffers
      f.read(8); n = n0; ops = f.read(n)
      print(f"file bytes {size}; out-of-band container; opcode stream bytes {n}; raw buffer region bytes {size-8-n}")
    else:                                   # plain pickle (the warp pickles)
      ops = f.read(); n = len(ops)
      print(f"file bytes {size}; plain pickle (no out-of-band region); opcode stream bytes {n}")
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
  import re
  ansi = re.compile(r'\x1b\[[0-9;]*m')
  prog = sorted({ansi.sub('', s) for s in nstr if ansi.search(s) and s[:2] in ('E\x1b', 'r\x1b', 'R\x1b')})
  print(f"\nkernel-name strings (ANSI colour codes stripped), distinct: {len(prog)}")
  print("  (the same name can be reused by several launches; per-launch list is in analyze_pkl.py)")
  if '--strings' in sys.argv:
    for s, c in nstr.items(): print(repr(s[:100]), c)

if __name__ == '__main__': main(sys.argv[1])
