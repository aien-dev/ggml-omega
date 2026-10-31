// tests/cmp.c: test infrastructure (not shipped). Compares two e2e logits files (n_vocab f32 per step).
// usage: cmp N_VOCAB A.logits B.logits
// stdout, one JSON line: steps, top1_agree (argmax equal per step), first_divergence (-1 if none),
// max_abs_diff over steps before the first divergence (same context in both runs), and its step.
// Exit 0 when the files have the same step count, 1 otherwise or on read errors.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: cmp N_VOCAB A.logits B.logits\n"); return 2; }
    long nv = atol(argv[1]);
    FILE *a = fopen(argv[2], "rb"), *b = fopen(argv[3], "rb");
    if (!a || !b || nv <= 0) { fprintf(stderr, "cannot open inputs\n"); return 1; }
    float *x = malloc(sizeof(float) * nv), *y = malloc(sizeof(float) * nv);
    long steps = 0, agree = 0, first_div = -1, worst_step = -1;
    double worst = 0;
    for (;;) {
        size_t ra = fread(x, sizeof(float), nv, a), rb = fread(y, sizeof(float), nv, b);
        if (ra == 0 && rb == 0) break;
        if (ra != (size_t) nv || rb != (size_t) nv) { fprintf(stderr, "step %ld: length mismatch\n", steps); return 1; }
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
    printf("{\"steps\":%ld,\"top1_agree\":%ld,\"first_divergence\":%ld,\"max_abs_diff\":%.9g,\"max_abs_diff_step\":%ld}\n",
           steps, agree, first_div, worst, worst_step);
    return 0;
}
