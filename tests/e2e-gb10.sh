#!/bin/sh
# tests/e2e-gb10.sh: greedy decode of the CI model, ggml CPU vs the OmegaGB10 device (MUL_MAT, RMS_NORM, swiglu
# on the GB10 through CUDA++), compared token by token and logit by logit with tests/cmp.c.
# usage: e2e-gb10.sh LLAMA_DIR MODEL.gguf LIBGGML_OMEGA_GB10 [N_TOKENS] [OUT]
# This loads a model's weights onto a chip call path: on the Spark it runs only inside an exclusive GPU
# window (the resident 35B stopped). Outside one it is NOT_RUN; do not run it beside the live server.
set -eu
L=$1; M=$2; LIB=$3; N=${4:-16}; OUT=${5:-build/e2e-gb10}
mkdir -p "$OUT"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -I"$L/include" -I"$L/ggml/include" -o "$OUT/e2e" tests/e2e.c \
  -L"$L/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -o "$OUT/cmp" tests/cmp.c -lm
"$OUT/e2e" "$M" "$N" "$OUT/cpu.logits" 2 > "$OUT/cpu.jsonl"
OMEGA_GGML_DEVICES=gb10 GGML_BACKEND_PATH=$LIB "$OUT/e2e" "$M" "$N" "$OUT/gb10.logits" 2 > "$OUT/gb10.jsonl" 2> "$OUT/gb10.err"
grep -q 'OmegaGB10' "$OUT/gb10.err" || { echo "FAIL OmegaGB10 device not listed" >&2; exit 1; }
NV=$(tail -n1 "$OUT/cpu.jsonl" | sed 's/.*"n_vocab":\([0-9]*\).*/\1/')
"$OUT/cmp" "$NV" "$OUT/cpu.logits" "$OUT/gb10.logits" | tee "$OUT/cmp.json"
