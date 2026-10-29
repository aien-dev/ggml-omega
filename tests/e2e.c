// tests/e2e.c: test infrastructure (not shipped). Greedy decode of a fixed prompt through llama.cpp's public
// API with whatever backends ggml_backend_load_all() finds (GGML_BACKEND_PATH=libggml-omega.so adds Omega).
// usage: e2e MODEL N_TOKENS LOGITS_OUT [THREADS]
// stdout: one JSON line per generated step {"step":i,"token":t}, then {"tok_per_s":x,"prompt_tokens":p}.
// LOGITS_OUT: n_vocab f32 values per step, raw, for tests/cmp.c. Diagnostics on stderr. Exit 0 on success.
#include "llama.h"
#include "ggml-backend.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *PROMPT = "The three primary colors of light are";

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void quiet(enum ggml_log_level l, const char *s, void *u) { (void) u; if (l >= GGML_LOG_LEVEL_WARN) fputs(s, stderr); }

static int positive(const char *s) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    return errno || end == s || *end || v <= 0 || v > INT_MAX ? -1 : (int) v;
}

// Optional verification snapshot after inference, including libraries loaded lazily.
static int save_maps(void) {
    const char *path = getenv("E2E_MAPS_OUT");
    if (!path) return 0;
    FILE *in = fopen("/proc/self/maps", "rb"), *out = fopen(path, "wx");
    if (!in || !out) { if (in) fclose(in); if (out) fclose(out); return 1; }
    char buf[4096];
    size_t n;
    int bad = 0;
    while ((n = fread(buf, 1, sizeof buf, in)) != 0)
        if (fwrite(buf, 1, n, out) != n) { bad = 1; break; }
    if (ferror(in)) bad = 1;
    if (fclose(in) != 0) bad = 1;
    if (fclose(out) != 0) bad = 1;
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: e2e MODEL N_TOKENS LOGITS_OUT [THREADS]\n"); return 2; }
    int n_gen = positive(argv[2]), threads = argc > 4 ? positive(argv[4]) : 2;
    if (n_gen < 1 || threads < 1) { fprintf(stderr, "invalid token or thread count\n"); return 2; }
    llama_log_set(quiet, NULL);
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
        fprintf(stderr, "device %zu: %s\n", i, ggml_backend_dev_name(ggml_backend_dev_get(i)));
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_token toks[64];
    int n_p = llama_tokenize(vocab, PROMPT, (int) strlen(PROMPT), toks, 64, true, false);
    if (n_p <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    if ((uint64_t) n_p + (uint64_t) n_gen - 1 > llama_n_ctx(ctx)) { fprintf(stderr, "generation exceeds context\n"); return 2; }
    FILE *lf = fopen(argv[3], "wx");
    if (!lf) { perror(argv[3]); return 1; }
    struct llama_batch b = llama_batch_get_one(toks, n_p);
    double t0 = now(), t_first = 0;
    llama_token cur;
    for (int i = 0; i < n_gen; i++) {
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at step %d\n", i); return 1; }
        if (i == 0) t_first = now();
        const float *lg = llama_get_logits_ith(ctx, -1);
        if (fwrite(lg, sizeof(float), (size_t) n_vocab, lf) != (size_t) n_vocab) { perror("write"); return 1; }
        cur = 0;
        for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[cur]) cur = v;
        printf("{\"step\":%d,\"token\":%d}\n", i, cur);
        b = llama_batch_get_one(&cur, 1);
    }
    double t1 = now();
    if (fclose(lf) != 0) { perror("close logits"); return 1; }
    if (save_maps() != 0) { fprintf(stderr, "cannot save process maps\n"); return 1; }
    // decode rate over the single-token steps (prompt step excluded)
    printf("{\"n_vocab\":%d,\"prompt_tokens\":%d,\"prompt_s\":%.6f,\"gen_steps\":%d,\"gen_s\":%.6f,\"threads\":%d}\n", n_vocab, n_p, t_first - t0, n_gen - 1, t1 - t_first, threads);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
