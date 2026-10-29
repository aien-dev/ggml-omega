# ggml-omega

A [ggml](https://github.com/ggml-org/llama.cpp/tree/master/ggml) backend whose op kernels are seeded-Omega units.
llama.cpp loads it as an out-of-tree backend (`GGML_BACKEND_PATH=libggml-omega.so`); the ops it takes run in
Omega-compiled code, every other op falls back to ggml's CPU backend. One job: run ggml ops through Omega.

| Op | Omega kernels (`kernels/`) | Scope |
|---|---|---|
| MUL_MAT | `df32`, `dh16`, `dq8`, `dq8r` (row dots, f64 accumulation) | weights f32, f16, q8_0; activations f32 |
| ADD SUB MUL DIV | `bin` | f32, row-level broadcast |
| RMS_NORM, SCALE | `sq`, `rmsc`, `scl` | f32 rows |
| SOFT_MAX | `smw`, `smwh`, `vmax`, `sexp`, `rcp` | f32, optional f32/f16 mask, no ALiBi bias, no sinks |
| ROPE | `rtsc`, `rope` (own sin/cos/exp/log in `math.om`) | NORMAL and NEOX, no YaRN ramp, no freq factors |
| GET_ROWS, SET_ROWS, CPY/CONT/DUP | `cpy`, `cpyh`, `h2fv`, `f2hv` | f32 and f16 (f16 rounding done in Omega) |
| GLU (swiglu), UNARY (silu) | `swi`, `silu` | f32 |

`ggml-omega.c` is **interim host glue**: tensor addressing, loops, thread split, staging copies and bit moves
only; every arithmetic step and conversion is Omega. Its successor is an `.om` loadable object once the Omega
native object ABI exists (aien-dev/omega-mv #636/#637).

## OmegaGB10 device (opt-in)

Built with `OMEGA_GGML_GB10=1 CUDAPP_SRC=<CUDApp src> CUDAPP_LIB=<CUDApp build-pic/libomega_gpu.a>` and selected at
run time with `OMEGA_GGML_DEVICES=gb10` (or `host,gb10`; default `host`), a second device runs three ops on the
NVIDIA GB10 through CUDA++ (aien-dev/CUDApp) runtime calls whose kernels come from Omega's own code generator:

| Op | Chip call | Scope |
|---|---|---|
| MUL_MAT | `omega_gpu_gemm_ex` (CUDApp `gemm_ex`, aien-dev/CUDApp#139); shapes, leading dimensions and batch strides planned by the Omega unit `kernels/gbmm.om` | weights f32, f16, bf16, or q8_0 dequantised by the Omega unit `dq8r`; activations f32, so every call takes the bf16x3 route (HMMA.BF16, fp32 accumulation; not exact fp32) |
| RMS_NORM | `omega_gpu_rmsnorm_f32` | f32 rows, row length a multiple of 128, at most 16384 |
| GLU swiglu | `omega_gpu_swiglu_f32` | f32 |

Every other op falls back to ggml's CPU backend. `ggml-omega-gb10.c` is interim host glue like `ggml-omega.c`
(same successor). Receipt, 2026-10-27 on the Spark GB10 beside the resident server
([evidence/gb10-20261027T0726Z](evidence/gb10-20261027T0726Z)): `test-backend-ops -b OmegaGB10` against the CPU
backend, MUL_MAT 871 cases OK / 0 FAIL, RMS_NORM 7 / 0, SWIGLU 12 / 0; a mutant `gbmm.om` (`OMEGA_GGML_MUTANT=gbmm-k`)
fails 385 MUL_MAT cases. This is op-level EXERCISED on one chip, not a qualification. NOT_RUN: any model on the
GB10 device (`tests/e2e-gb10.sh`, CPU vs OmegaGB10 token and logit match) waits for an exclusive GPU window; CI
has no GB10 and builds the host device only.

Follow-up on this experimental branch, 2026-10-29: the exclusive model window was
authorized and exercised. Qwen2.5-0.5B Q8 and Qwen3.5-9B BF16/Q8 each matched the
CPU reference's greedy tokens for the fixed prompt over 16 steps. The Q35 runs
recorded nonzero MUL_MAT, RMS_NORM and SWIGLU calls and an after-inference library
snapshot without the forbidden NVIDIA user-space libraries. Logits are not
numerically identical; this is a bounded model check, not full chat-corpus or
performance qualification. The source and file hashes, exclusions and raw-result
locations are in `evidence/gb10-model-20261029.json`.

The model comparison now fails for empty output, partial rows, nonfinite scores,
wrong step counts or a changed greedy token. `e2e-gb10.sh` also requires executed
operations and the inference library snapshot. Its comparator status reaches the
caller without a `tee` pipeline hiding a failure. Existing result files are
preserved; reruns need a fresh output directory.

## Build and test

```sh
OMEGA_GGML_OMC=<omega-mv seed/out/omc> OMEGA_BRIDGE=<omega-bridge checkout> \
GGML_SRC=<llama.cpp checkout> GGML_LIB=<llama.cpp build/bin> ./build.sh
tests/backend-ops.sh <build/bin/test-backend-ops> build/libggml-omega.so [mutant .so]
cc -std=gnu11 -O2 -Wall -Wextra -Werror -o build/cmp tests/cmp.c -lm
node tests/test-cmp.mjs build/cmp
node tests/test-e2e-gb10.mjs
```

The two Node scripts are verification-only failure controls, tested with Node
v24.21.0; they do not add a model-serving or production dependency. The full model
driver remains shell/C and uses the pinned llama.cpp build.

The shared library exports only `ggml_backend_init`. Its Omega kernels, CUDApp
entries and hashing helpers are private to the backend. This prevents a host
library such as llama.cpp's `libmtmd` from replacing a same-named helper with an
incompatible signature. `sh tests/exports.sh build/libggml-omega.so` checks this
boundary. A full-server startup exposed this collision after the bounded model
tests passed; those earlier tests did not load `libmtmd`.

Pins (checked by `build.sh`, refuses on mismatch): omc sha256 `2c824805…` (aien-dev/omega-mv `e5b32567`),
omega-bridge `src/native.om` sha256 `901684b2…` (omega-bridge `d06ec808`), llama.cpp `7aca0d2e`.
`.github/workflows/ci.yml` runs the whole chain on `ubuntu-24.04-arm`: kernels vs libm and `_Float16`,
`test-backend-ops` per op against the CPU backend, an `add -> sub` mutant that must fail, and a Qwen2.5-0.5B
q8_0 end-to-end greedy decode, CPU vs Omega (top-1 agreement, max abs logit difference, timing).

## Connected repos

- Parent: [aien-dev/omega-bridge](https://github.com/aien-dev/omega-bridge) supplies the native call ABI binding
  generator (`src/native.om`, f32/ushort kinds from omega-bridge PR #26), pinned by sha256.
- Compiler: [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) `seed/out/omc`, pinned by sha256.
- Host: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), pinned by commit; nothing here is upstreamed.

## License

Apache-2.0 WITH LLVM-exception, see [LICENSE](LICENSE).
