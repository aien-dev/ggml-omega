#!/bin/sh
# Build libggml-omega.so: a ggml backend (llama.cpp backend API v3) whose op kernels are seeded-Omega
# units (kernels/*.om) bound through the omega-bridge native call ABI (src/native.om, docs/NATIVE-ABI.md).
#   OMEGA_GGML_OMC   the seeded compiler, sha256 pinned below (omega-mv seed/out/omc at e5b32567)
#   OMEGA_BRIDGE     an aien-dev/omega-bridge checkout; its src/native.om must match the pinned sha256
#   GGML_SRC         a llama.cpp checkout at the pinned commit (headers: ggml/include, ggml/src)
#   GGML_LIB         the folder holding that checkout's libggml-base.so and libggml-cpu.so
#   OMEGA_GGML_OUT   output folder (default build)
#   OMEGA_GGML_MUTANT negative controls: bin-add (the add lane of bin.om computes x - y; test-backend-ops
#                     must fail) or math-exp (math.om's e^x series keeps 4 terms instead of 13; tests/kernels.c
#                     must fail) or gbmm-k (gbmm.om plans k - 1: the GB10 MUL_MAT drops the last k element;
#                     test-backend-ops -b OmegaGB10 must fail) or bfs-lo (bfs.om writes the hi plane into lo:
#                     the GB10 GEMV route computes about 2 W x; test-backend-ops -b OmegaGB10 must fail)
#   OMEGA_GGML_GB10   1: also build the GB10 device (ggml-omega-gb10.c, migration-pending host glue) against
#                     CUDA++: CUDAPP_SRC (aien-dev/CUDApp src/, needs omega_gpu_gemm_ex.h) and CUDAPP_LIB
#                     (its libomega_gpu.a built with -fPIC). Default 0: the host device only.
set -eu
PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
LLAMA_PIN=7aca0d2ea8d92b56911233b822db01f9fbb1ae50
ROOT=$(cd "$(dirname "$0")" && pwd)
K=$ROOT/kernels
NATIVE_PIN=901684b2dc7cd148eb6bb2fe74dd43a6bce857ff014308b00ebaa9d18029158d
BR=${OMEGA_BRIDGE:?set OMEGA_BRIDGE to an omega-bridge checkout (src/native.om sha256 $NATIVE_PIN)}
OMC=${OMEGA_GGML_OMC:?set OMEGA_GGML_OMC to the seeded omc (sha256 $PIN)}
SRC=${GGML_SRC:?set GGML_SRC to a llama.cpp checkout at $LLAMA_PIN}
LIB=${GGML_LIB:?set GGML_LIB to the folder with libggml-base.so and libggml-cpu.so}
O=${OMEGA_GGML_OUT:-$ROOT/build}
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN" >&2; exit 1; }
got=$(sha256sum < "$BR/src/native.om" | cut -d' ' -f1)
[ "$got" = "$NATIVE_PIN" ] || { echo "FAIL omega-bridge src/native.om sha256 $got, pinned $NATIVE_PIN" >&2; exit 1; }
head=$(git -C "$SRC" rev-parse HEAD)
[ "$head" = "$LLAMA_PIN" ] || { echo "FAIL llama.cpp at $head, pinned $LLAMA_PIN" >&2; exit 1; }
rm -rf "$O"; mkdir -p "$O/units"
"$OMC" < "$BR/src/native.om" > "$O/native"; chmod +x "$O/native"
# file:tile instances. T is the f32/f16 lane count; dq8 takes blocks (B), bytes (W) and floats (X).
INST="bin:32 bin:1 sq:32 sq:1 rmsc:0 scl:32 scl:1 smw:32 smw:1 smwh:32 smwh:1 vmax:32 vmax:1
sexp:32 sexp:1 rcp:0 rtsc:0 rope:0 df32:256 df32:32 df32:1 dh16:256 dh16:32 dh16:1 dq8:8 dq8:1
dq8r:0 h2fv:32 h2fv:1 f2hv:32 f2hv:1 cpy:32 cpy:1 cpyh:32 cpyh:1 swi:32 swi:1 silu:32 silu:1 bfs:256 bfs:1"
GB10=${OMEGA_GGML_GB10:-0}
[ "$GB10" = 1 ] && INST="$INST gbmm:0"
: > "$O/units.in"
for it in $INST; do
  f=${it%%:*}; t=${it#*:}; u=$f; [ "$t" = 0 ] || u=${f}_$t
  { cat "$K/math.om"
    sed -e "s/@T@/$t/g; s/@B@/$t/g; s/@W@/$((34 * t))/g; s/@X@/$((32 * t))/g" "$K/$f.om"; } > "$O/units/$u.om"
  if [ "${OMEGA_GGML_MUTANT:-}" = bin-add ] && [ "$f" = bin ]; then
    sed -i 's/^put d i add x y$/put d i sub x y/' "$O/units/$u.om"
    grep -q '^put d i sub x y$' "$O/units/$u.om" || { echo "FAIL mutant not applied to $u" >&2; exit 1; }
  fi
  if [ "${OMEGA_GGML_MUTANT:-}" = math-exp ]; then
    sed -i 's/^for i ulong 13$/for i ulong 4/; s/^n = f64 sub 13 i$/n = f64 sub 4 i/' "$O/units/$u.om"
    grep -q '^for i ulong 4$' "$O/units/$u.om" || { echo "FAIL mutant not applied to $u" >&2; exit 1; }
  fi
  if [ "${OMEGA_GGML_MUTANT:-}" = gbmm-k ] && [ "$f" = gbmm ]; then
    sed -i 's/^put p 2 get d 0$/put p 2 sub get d 0 1/' "$O/units/$u.om"
    grep -q '^put p 2 sub get d 0 1$' "$O/units/$u.om" || { echo "FAIL mutant not applied to $u" >&2; exit 1; }
  fi
  if [ "${OMEGA_GGML_MUTANT:-}" = bfs-lo ] && [ "$f" = bfs ]; then
    sed -i 's/^put l i ushort shr bitcast uint r 16$/put l i ushort shr u 16/' "$O/units/$u.om"
    grep -q '^put l i ushort shr u 16$' "$O/units/$u.om" || { echo "FAIL mutant not applied to $u" >&2; exit 1; }
  fi
  "$OMC" < "$O/units/$u.om" > "$O/units/$u.bin" || { echo "FAIL omc $u" >&2; exit 1; }
  { cat "$O/units/$u.om"; printf '\0'; printf '%08d' "$(wc -c < "$O/units/$u.bin")"; cat "$O/units/$u.bin"; } >> "$O/units.in"
done
"$O/native" < "$O/units.in" > "$O/omega_ggml_kernels.c"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -fPIC -c -o "$O/omega_ggml_kernels.o" "$O/omega_ggml_kernels.c"
GBC= GBO= GBL=
if [ "$GB10" = 1 ]; then
  CS=${CUDAPP_SRC:?set CUDAPP_SRC to aien-dev/CUDApp src/ (with omega_gpu_gemm_ex.h)}
  CL=${CUDAPP_LIB:?set CUDAPP_LIB to libomega_gpu.a of that checkout, built with -fPIC}
  [ -f "$CS/omega_gpu_gemm_ex.h" ] || { echo "FAIL $CS has no omega_gpu_gemm_ex.h" >&2; exit 1; }
  cc -std=gnu11 -O2 -Wall -Wextra -Werror -fPIC -pthread -I"$SRC/ggml/include" -I"$SRC/ggml/src" -I"$CS" -I"$ROOT" \
    -c -o "$O/ggml-omega-gb10.o" "$ROOT/ggml-omega-gb10.c"
  GBC=-DOMEGA_GGML_GB10; GBO=$O/ggml-omega-gb10.o; GBL="$CL -lm"
fi
cc -std=gnu11 -O2 -Wall -Wextra -Werror -fPIC -pthread $GBC -I"$SRC/ggml/include" -I"$SRC/ggml/src" -I"$O" -I"$ROOT" \
  -c -o "$O/ggml-omega.o" "$ROOT/ggml-omega.c"
cc -shared -pthread -o "$O/libggml-omega.so" "$O/ggml-omega.o" $GBO "$O/omega_ggml_kernels.o" $GBL \
  -L"$LIB" -lggml-base -lggml-cpu -Wl,-rpath,"$LIB"
echo "BUILT $O/libggml-omega.so ($(echo $INST | wc -w) Omega units, GB10 device $GB10, omc $(echo $PIN | cut -c1-12), llama.cpp $(echo $LLAMA_PIN | cut -c1-12))"
