// tests/cmp.c: test infrastructure (not shipped). Compares two e2e logits files (n_vocab f32 per step).
// usage: cmp N_VOCAB A.logits B.logits [EXPECTED_STEPS]
// stdout, one JSON line: steps, top1_agree (argmax equal per step), first_divergence (-1 if none),
// max_abs_diff over steps before the first divergence (same context in both runs), and its step.
// Exit 0 only for nonempty, finite, complete rows with identical greedy tokens.
// EXPECTED_STEPS additionally refuses equally truncated files.
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static long positive(const char *s) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    return errno || end == s || *end || v <= 0 ? -1 : v;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) { fprintf(stderr, "usage: cmp N_VOCAB A.logits B.logits [EXPECTED_STEPS]\n"); return 2; }
    long nv = positive(argv[1]), expected = argc == 5 ? positive(argv[4]) : -1;
    if (nv <= 0 || (uintmax_t) nv > SIZE_MAX / sizeof(float) || (argc == 5 && expected <= 0)) {
        fprintf(stderr, "invalid vocabulary or step count\n"); return 2;
    }
    FILE *a = fopen(argv[2], "rb"), *b = fopen(argv[3], "rb");
    if (!a || !b) { fprintf(stderr, "cannot open inputs\n"); if (a) fclose(a); if (b) fclose(b); return 1; }
    float *x = malloc(sizeof(float) * nv), *y = malloc(sizeof(float) * nv);
    if (!x || !y) { fprintf(stderr, "allocation failed\n"); free(x); free(y); fclose(a); fclose(b); return 1; }
    long steps = 0, agree = 0, first_div = -1, worst_step = -1;
    double worst = 0;
    int failed = 0;
    for (;;) {
        size_t bytes = sizeof(float) * (size_t) nv;
        size_t ra = fread(x, 1, bytes, a), rb = fread(y, 1, bytes, b);
        if (ferror(a) || ferror(b)) { fprintf(stderr, "step %ld: read error\n", steps); failed = 1; break; }
        if (ra == 0 && rb == 0) break;
        if (ra != bytes || rb != bytes) { fprintf(stderr, "step %ld: incomplete row\n", steps); failed = 1; break; }
        for (long v = 0; v < nv; v++) if (!isfinite(x[v]) || !isfinite(y[v])) {
            fprintf(stderr, "step %ld: nonfinite logits\n", steps); failed = 1; break;
        }
        if (failed) break;
        long ia = 0, ib = 0;
        for (long v = 1; v < nv; v++) { if (x[v] > x[ia]) ia = v; if (y[v] > y[ib]) ib = v; }
        if (ia == ib) agree++;
        if (first_div < 0) {
            for (long v = 0; v < nv; v++) {
                double d = fabs((double) x[v] - (double) y[v]);
                if (!(d <= worst)) { worst = d; worst_step = steps; }
            }
            if (ia != ib) first_div = steps;
        }
        steps++;
    }
    if (steps == 0 || (expected > 0 && steps != expected)) {
        fprintf(stderr, "unexpected step count: %ld\n", steps); failed = 1;
    }
    if (first_div >= 0) failed = 1;
    printf("{\"steps\":%ld,\"top1_agree\":%ld,\"first_divergence\":%ld,\"max_abs_diff\":%.9g,\"max_abs_diff_step\":%ld}\n",
           steps, agree, first_div, worst, worst_step);
    free(x); free(y);
    if (fclose(a) != 0) failed = 1;
    if (fclose(b) != 0) failed = 1;
    return failed;
}
