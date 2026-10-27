#!/bin/sh
# tests/backend-ops.sh: llama.cpp test-backend-ops, Omega backend against ggml's CPU backend, per op.
# usage: backend-ops.sh TEST_BACKEND_OPS LIBGGML_OMEGA [MUTANT_LIB]
# stdout: one tab-separated row per op: op, cases passed, cases failed, cases not supported (CPU fallback), verdict.
# An op with 0 Omega cases is NOT_RUN, never PASS. With MUTANT_LIB (built OMEGA_GGML_MUTANT=bin-add), ADD
# must fail against it. Full logs: $OUT (default build/backend-ops). Exit 1 on any FAIL or uncaught mutant.
set -u
TBO=$1; LIB=$2; MUT=${3:-}
OUT=${OUT:-build/backend-ops}
mkdir -p "$OUT"
OPS="ADD SUB MUL DIV SCALE RMS_NORM SOFT_MAX ROPE GET_ROWS SET_ROWS CPY CONT DUP SWIGLU SILU MUL_MAT"
fails=0
printf 'op\tpass\tfail\tnot_supported\tverdict\n'
for op in $OPS; do
  GGML_BACKEND_PATH=$LIB "$TBO" test -b Omega -o "$op" > "$OUT/$op.log" 2>&1
  rc=$?
  cases() { grep -a "^  $op(" "$OUT/$op.log" | grep -ac "$1"; }
  p=$(cases '1;32mOK'); f=$(cases '1;31mFAIL'); n=$(cases 'not supported')
  if [ "$f" -gt 0 ] || [ $rc -ne 0 ]; then v=FAIL; fails=$((fails + 1))
  elif [ "$p" -eq 0 ]; then v=NOT_RUN
  else v=PASS; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$op" "$p" "$f" "$n" "$v"
done
if [ -n "$MUT" ]; then
  GGML_BACKEND_PATH=$MUT "$TBO" test -b Omega -o ADD > "$OUT/mutant-ADD.log" 2>&1
  f=$(grep -a '^  ADD(' "$OUT/mutant-ADD.log" | grep -ac '1;31mFAIL')
  if [ "$f" -gt 0 ]; then printf 'mutant-bin-add\t-\t%s\t-\tCAUGHT\n' "$f"
  else printf 'mutant-bin-add\t-\t0\t-\tNOT_CAUGHT\n'; fails=$((fails + 1)); fi
fi
[ $fails -eq 0 ]
