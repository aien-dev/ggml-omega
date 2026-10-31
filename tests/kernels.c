// tests/kernels.c: test infrastructure (not shipped). Checks the Omega math units against independent
// oracles on this host: the C compiler's _Float16 conversions (AArch64 FCVT), glibc libm in double
// precision, and plain C reference loops. Prints one line per check; exit 1 on any failure.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ARR(p) const void *p, uint64_t p##_n
#define OUT(p) void *p, uint64_t p##_n
uint32_t omega_h2fv_1(uint64_t *r, ARR(s), OUT(d));
uint32_t omega_f2hv_1(uint64_t *r, ARR(s), OUT(d));
uint32_t omega_sexp_1(uint64_t *r, uint64_t acc, OUT(w), uint64_t mx);
uint32_t omega_rtsc(uint64_t *r, uint64_t base, uint64_t nd);
uint32_t omega_rope(uint64_t *r, uint64_t xy, uint64_t pos, uint64_t i, uint64_t ts, uint64_t fa);
uint32_t omega_silu_1(uint64_t *r, ARR(a), OUT(d));
uint32_t omega_rmsc(uint64_t *r, uint64_t acc, uint64_t n, uint64_t eps);
uint32_t omega_dq8_1(uint64_t *r, uint64_t acc, ARR(w), ARR(x), uint64_t fin);

static int fails;
static uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
// distance in f32 ulps (ordered integer line), NaN only matches NaN
static int64_t ulps(float a, float b) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b) ? 0 : INT64_MAX;
    int64_t x = (int32_t) fb(a), y = (int32_t) fb(b);
    if (x < 0) x = INT32_MIN - x;
    if (y < 0) y = INT32_MIN - y;
    return x > y ? x - y : y - x;
}
static void report(const char *name, long n, long bad, int64_t worst, int64_t limit) {
    printf("%-28s n=%-9ld bad=%-6ld worst_ulp=%lld limit=%lld %s\n", name, n, bad, (long long) worst, (long long) limit,
           bad ? "FAIL" : "PASS");
    if (bad) fails++;
}
#define OK(c) do { if ((c) != 0) { printf("status fail: %s\n", #c); return 1; } } while (0)

int main(void) {
    uint64_t r;
    long n = 0, bad = 0; int64_t worst = 0;
    // f16 -> f32, all 65536 inputs, exact
    for (uint32_t h = 0; h < 65536; h++) {
        uint16_t hv = (uint16_t) h; _Float16 hf; memcpy(&hf, &hv, 2);
        float o; OK(omega_h2fv_1(&r, &hv, 1, &o, 1));
        float want = (float) hf; n++;
        if (!(isnan(want) ? isnan(o) : fb(o) == fb(want))) bad++;
    }
    report("h2f all f16", n, bad, 0, 0);
    // f32 -> f16 (round to nearest even), every 1021st f32 bit pattern plus all specials, exact (NaN: any NaN)
    n = bad = 0;
    for (uint64_t u = 0; u < 4294967296ull; u += 1021) {
        float x = bf((uint32_t) u); uint16_t o; OK(omega_f2hv_1(&r, &x, 1, &o, 1));
        _Float16 w = (_Float16) x; uint16_t wv; memcpy(&wv, &w, 2); n++;
        if (isnan(x) ? ((o & 0x7c00) != 0x7c00 || (o & 0x3ff) == 0) : o != wv) bad++;
    }
    report("f2h f32 stride 1021", n, bad, 0, 0);
    // exp: the sexp unit returns f32(exp(x - 0)); oracle double exp rounded to f32
    n = bad = 0; worst = 0;
    for (float x = -110.0f; x < 89.0f; x += 0.00731f) {
        float w = x; uint64_t s = 0; OK(omega_sexp_1(&s, 0, &w, 1, 0));
        float want = (float) exp((double) x); int64_t d = ulps(w, want); n++;
        if (d > worst) worst = d;
        if (d > 1) bad++;
    }
    report("exp vs libm (f32 ulp)", n, bad, worst, 1);
    // powf(base, -2/nd) for rope theta scale
    n = bad = 0; worst = 0;
    const float bases[] = { 10000.0f, 1000000.0f, 500000.0f, 10.0f, 1.5f };
    for (int b = 0; b < 5; b++)
        for (uint64_t nd = 2; nd <= 512; nd += 2) {
            uint64_t ts; OK(omega_rtsc(&ts, fb(bases[b]), nd));
            float want = (float) pow((double) bases[b], (double) (-2.0f / (float) nd));
            int64_t d = ulps(bf((uint32_t) ts), want); n++;
            if (d > worst) worst = d;
            if (d > 1) bad++;
        }
    report("pow vs libm (f32 ulp)", n, bad, worst, 1);
    // rope pair: oracle = same f32 theta recurrence, double cos/sin rounded to f32
    n = bad = 0; worst = 0;
    const float ts = 0.8659643f, fs = 1.0f, af = 1.0f, x0 = 0.75f, x1 = -1.25f;
    for (int32_t pos = -3; pos < 4100; pos += 7)
        for (uint64_t i = 0; i < 64; i++) {
            float t = (float) pos;
            for (uint64_t k = 0; k < i; k++) t *= ts;
            t *= fs;
            float c = (float) cos((double) t) * af, s = (float) sin((double) t) * af;
            float w0 = x0 * c - x1 * s, w1 = x0 * s + x1 * c;
            uint64_t o; OK(omega_rope(&o, fb(x0) | ((uint64_t) fb(x1) << 32), (uint32_t) pos, i, fb(ts), fb(fs) | ((uint64_t) fb(af) << 32)));
            int64_t d0 = ulps(bf((uint32_t) o), w0), d1 = ulps(bf((uint32_t) (o >> 32)), w1);
            int64_t d = d0 > d1 ? d0 : d1; n++;
            if (d > worst) worst = d;
            if (d > 0) bad++;
        }
    report("rope vs libm sin/cos", n, bad, worst, 0);
    // silu: x/(1+expf(-x)) with libm-exact expf
    n = bad = 0; worst = 0;
    for (float x = -90.0f; x < 90.0f; x += 0.0137f) {
        float o; OK(omega_silu_1(&r, &x, 1, &o, 1));
        float want = x / (1.0f + (float) exp(-(double) x)); int64_t d = ulps(o, want); n++;
        if (d > worst) worst = d;
        if (d > 1) bad++;
    }
    report("silu vs libm", n, bad, worst, 1);
    // rms scale: 1/sqrtf(f32(sum/n) + eps)
    n = bad = 0;
    for (int k = 1; k < 5000; k++) {
        double sum = k * 0.37; uint64_t sb; memcpy(&sb, &sum, 8); uint64_t o;
        OK(omega_rmsc(&o, sb, 896, fb(1e-6f)));
        float want = 1.0f / sqrtf((float) (sum / 896) + 1e-6f); n++;
        if (fb(bf((uint32_t) o)) != fb(want)) bad++;
    }
    report("rms scale exact", n, bad, 0, 0);
    // q8_0 dot: one block, reference f64 accumulation of f32(q)*x then times f16 d
    n = bad = 0;
    for (int t = 0; t < 2000; t++) {
        unsigned char blk[34]; float x[32]; uint32_t seed = 2654435761u * (uint32_t) (t + 1);
        _Float16 d = (_Float16) (0.001f * (float) (t % 97 + 1)); memcpy(blk, &d, 2);
        for (int j = 0; j < 32; j++) { seed = seed * 1103515245u + 12345u; blk[2 + j] = (unsigned char) (seed >> 16); x[j] = (float) ((int) (seed >> 8) % 2001 - 1000) / 333.0f; }
        float bs = 0.0f; for (int j = 0; j < 32; j++) bs += (float) (int8_t) blk[2 + j] * x[j];
        float want = (float) (double) ((float) d * bs);
        uint64_t o; OK(omega_dq8_1(&o, 0, blk, 34, x, 32, 1)); n++;
        if (fb(bf((uint32_t) o)) != fb(want)) bad++;
    }
    report("q8_0 block dot exact", n, bad, 0, 0);
    printf("%s\n", fails ? "KERNELS FAIL" : "KERNELS PASS");
    return fails ? 1 : 0;
}
