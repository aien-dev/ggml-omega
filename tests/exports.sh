#!/bin/sh
# Only the ggml loader entry point is public. Private helpers must not bind to
# same-named functions in the host process (e.g. libmtmd's incompatible SHA ABI).
set -eu
symbols=$(nm -D --defined-only "$1" | awk '{print $NF}')
test "$symbols" = ggml_backend_init || {
    echo 'FAIL unexpected exported backend symbols' >&2
    printf '%s\n' "$symbols" >&2
    exit 1
}
echo 'PASS only ggml_backend_init exported'
