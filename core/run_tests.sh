#!/usr/bin/env bash
# Builds opcore + test driver with -std=c99 -O2 -Wall -Wextra and runs all checks. Non-zero exit on any failure.
# Needs a python with numpy, onnxruntime, tinygrad. Set PY to override (default: the project venv if present, else python3).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
mkdir -p "$HERE/tests/build"
cc -std=c99 -O2 -Wall -Wextra -Werror -o "$HERE/tests/build/test_driver" "$HERE/tests/test_driver.c" "$HERE/opcore.c" -lm
echo "built with: $(cc --version | head -1)"
if command -v clang >/dev/null 2>&1; then
  clang -std=c99 -O2 -Wall -Wextra -Werror -c "$HERE/opcore.c" -o "$HERE/tests/build/opcore_clang.o" && echo "clang syntax/warnings check OK"
fi
DEV=CPU "$PY" -I "$HERE/tests/test_opcore.py"
