#include "lzgraph/pgen_dist.h"
#include "pgen_mixture.h"
#include "pgen_moments.h"
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

LZGError lzg_pgen_moments(const LZGGraph *g, LZGPgenMoments *out) {
    return lzg_pgen_compute_moments(g, out);
}

LZGError lzg_pgen_analytical(const LZGGraph *g, LZGPgenDist *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    LZGPgenMoments mom;
    LZGError err = lzg_pgen_compute_moments(g, &mom);
    if (err != LZG_OK) return err;

    return lzg_pgen_build_analytical_mixture(g, &mom, out);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Distribution evaluation functions                               */
/* ═══════════════════════════════════════════════════════════════ */

double lzg_pgen_pdf(const LZGPgenDist *dist, double x) {
    if (!dist) return 0.0;
    double p = 0.0;
    for (uint32_t i = 0; i < dist->n_components; i++) {
        double z = (x - dist->means[i]) / fmax(dist->stds[i], 1e-15);
        p += dist->weights[i] * exp(-0.5 * z * z)
             / (dist->stds[i] * sqrt(2.0 * M_PI));
    }
    return p;
}

double lzg_pgen_cdf(const LZGPgenDist *dist, double x) {
    if (!dist) return 0.0;
    double c = 0.0;
    for (uint32_t i = 0; i < dist->n_components; i++) {
        double z = (x - dist->means[i]) / (dist->stds[i] * sqrt(2.0));
        c += dist->weights[i] * 0.5 * erfc(-z);
    }
    return c;
}

LZGError lzg_pgen_sample(const LZGPgenDist *dist, LZGRng *rng,
                           uint32_t n, double *out) {
    if (!dist || !rng || !out) return LZG_ERR_INVALID_ARG;
    if (dist->n_components == 0) return LZG_ERR_EMPTY_INPUT;

    /* Precompute cumulative weights */
    double cum[LZG_PGEN_MAX_COMPONENTS];
    cum[0] = dist->weights[0];
    for (uint32_t i = 1; i < dist->n_components; i++)
        cum[i] = cum[i - 1] + dist->weights[i];

    for (uint32_t i = 0; i < n; i++) {
        /* Pick component */
        double r = lzg_rng_double(rng) * cum[dist->n_components - 1];
        uint32_t c = 0;
        while (c < dist->n_components - 1 && cum[c] < r) c++;

        /* Box-Muller for normal sample */
        double u1 = fmax(lzg_rng_double(rng), 1e-300);
        double u2 = lzg_rng_double(rng);
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        out[i] = dist->means[c] + dist->stds[c] * z;
    }

    return LZG_OK;
}
