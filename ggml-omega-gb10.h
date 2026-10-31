// ggml-omega-gb10.h: the GB10 device of ggml-omega (ggml-omega-gb10.c), built when OMEGA_GGML_GB10=1.
#ifndef GGML_OMEGA_GB10_H
#define GGML_OMEGA_GB10_H

#include "ggml-backend-impl.h"

#define OMEGA_GB10_NAME "OmegaGB10"

bool omega_gb10_supports_op(const struct ggml_tensor *op);
ggml_backend_t omega_gb10_init(ggml_backend_dev_t d);

#endif
