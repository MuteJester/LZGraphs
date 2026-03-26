/**
 * @file pgen_dist.c
 * @brief LZPgen moment computation and Gaussian mixture construction.
 *
 * Uses the unconstrained forward DP engine to propagate moment
 * accumulators through the graph topology. For exact LZ-constrained
 * probability computation, use lzg_walk_log_prob() per-sequence.
 *
 * Moments are computed from the graph topology's probability model:
 *   m[0] = total_mass (should be ~1.0)
 *   m[1] = E[log P] (mean log-probability)
 *   m[2] = E[(log P)^2]
 * From which variance, std, skewness, kurtosis are derived.
 */
#include "lzgraph/pgen_dist.h"
#include "lzgraph/forward.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════ */
/* Moments via forward propagation                                 */
/* ═══════════════════════════════════════════════════════════════ */

/*
 * Accumulator: [mass, mass*logP, mass*(logP)^2, mass*(logP)^3, mass*(logP)^4]
 * At terminals, we absorb and accumulate the moments.
 */

static void mom_seed(double *acc, double p, void *ctx) {
    (void)ctx;
    double lp = log(fmax(p, 1e-300));
    acc[0] += p;
    acc[1] += p * lp;
    acc[2] += p * lp * lp;
    acc[3] += p * lp * lp * lp;
    acc[4] += p * lp * lp * lp * lp;
}

static void mom_edge(double *dst, const double *src, double w, double Z, void *ctx) {
    (void)ctx;
    double ratio = w / Z;
    double lr = log(fmax(ratio, 1e-300));
    /* Multiply mass by ratio, shift log-prob by lr */
    dst[0] = src[0] * ratio;
    dst[1] = src[0] * ratio * lr + src[1] * ratio;
    dst[2] = src[0] * ratio * lr * lr + 2.0 * src[1] * ratio * lr + src[2] * ratio;
    dst[3] = src[0] * ratio * lr * lr * lr + 3.0 * src[1] * ratio * lr * lr
             + 3.0 * src[2] * ratio * lr + src[3] * ratio;
    dst[4] = src[0] * ratio * lr * lr * lr * lr + 4.0 * src[1] * ratio * lr * lr * lr
             + 6.0 * src[2] * ratio * lr * lr + 4.0 * src[3] * ratio * lr + src[4] * ratio;
}

static void mom_absorb(double *total, const double *acc, double sp, void *ctx) {
    (void)ctx;
    double lsp = log(fmax(sp, 1e-300));
    /* Absorb: multiply by stop prob, shift log-prob by log(sp) */
    double m = acc[0] * sp;
    total[0] += m;
    total[1] += (acc[1] * sp + acc[0] * sp * lsp);
    total[2] += (acc[2] * sp + 2.0 * acc[1] * sp * lsp + acc[0] * sp * lsp * lsp);
    total[3] += (acc[3] * sp + 3.0 * acc[2] * sp * lsp + 3.0 * acc[1] * sp * lsp * lsp
                 + acc[0] * sp * lsp * lsp * lsp);
    total[4] += (acc[4] * sp + 4.0 * acc[3] * sp * lsp + 6.0 * acc[2] * sp * lsp * lsp
                 + 4.0 * acc[1] * sp * lsp * lsp * lsp + acc[0] * sp * lsp * lsp * lsp * lsp);
}

static void mom_cont(double *co, const double *acc, double sp, void *ctx) {
    (void)ctx;
    double csp = 1.0 - sp;
    double lcsp = log(fmax(csp, 1e-300));
    co[0] = acc[0] * csp;
    co[1] = acc[1] * csp + acc[0] * csp * lcsp;
    co[2] = acc[2] * csp + 2.0 * acc[1] * csp * lcsp + acc[0] * csp * lcsp * lcsp;
    co[3] = acc[3] * csp + 3.0 * acc[2] * csp * lcsp + 3.0 * acc[1] * csp * lcsp * lcsp
            + acc[0] * csp * lcsp * lcsp * lcsp;
    co[4] = acc[4] * csp + 4.0 * acc[3] * csp * lcsp + 6.0 * acc[2] * csp * lcsp * lcsp
            + 4.0 * acc[1] * csp * lcsp * lcsp * lcsp + acc[0] * csp * lcsp * lcsp * lcsp * lcsp;
}

LZGError lzg_pgen_moments(const LZGGraph *g, LZGPgenMoments *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;

    LZGFwdOps ops = {
        .seed = mom_seed, .edge = mom_edge,
        .absorb = mom_absorb, .cont = mom_cont,
        .acc_dim = 5, .ctx = NULL,
    };

    double total[5] = {0};
    LZGError err = lzg_forward_propagate(g, &ops, total);
    if (err != LZG_OK) return err;

    double mass = total[0];
    if (mass < 1e-300) {
        memset(out, 0, sizeof(*out));
        return LZG_OK;
    }

    out->total_mass = mass;
    out->mean = total[1] / mass;
    double var = total[2] / mass - out->mean * out->mean;
    out->variance = fmax(var, 0.0);
    out->std = sqrt(out->variance);

    if (out->std > 1e-15) {
        double m3 = total[3] / mass - 3.0 * out->mean * total[2] / mass
                     + 2.0 * out->mean * out->mean * out->mean;
        out->skewness = m3 / (out->std * out->std * out->std);

        double m4 = total[4] / mass - 4.0 * out->mean * total[3] / mass
                     + 6.0 * out->mean * out->mean * total[2] / mass
                     - 3.0 * out->mean * out->mean * out->mean * out->mean;
        out->kurtosis = m4 / (out->variance * out->variance) - 3.0;
    } else {
        out->skewness = 0.0;
        out->kurtosis = 0.0;
    }

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Analytical Gaussian mixture distribution                        */
/* ═══════════════════════════════════════════════════════════════ */

/*
 * For the analytical distribution, we run the forward DP with
 * length-stratified accumulators. Each "length" (walk depth in tokens)
 * gets its own mean/variance Gaussian component.
 *
 * Simplified approach: use the global moments to construct a single
 * component (or use simulation to build per-length components).
 */

LZGError lzg_pgen_analytical(const LZGGraph *g, LZGPgenDist *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    /* Compute global moments */
    LZGPgenMoments mom;
    LZGError err = lzg_pgen_moments(g, &mom);
    if (err != LZG_OK) return err;

    memset(out, 0, sizeof(*out));
    out->global = mom;

    /* Build per-length components via simulation */
    LZGRng rng;
    lzg_rng_seed(&rng, 12345);

    uint32_t n_sim = 10000;
    LZGSimResult *sims = malloc(n_sim * sizeof(LZGSimResult));
    if (!sims) return LZG_ERR_ALLOC;

    err = lzg_simulate(g, n_sim, &rng, sims);
    if (err != LZG_OK) { free(sims); return err; }

    /* Group by token count (walk length) */
    uint32_t max_len = 0;
    for (uint32_t i = 0; i < n_sim; i++)
        if (sims[i].n_tokens > max_len) max_len = sims[i].n_tokens;

    /* Per-length accumulators: count, sum_logp, sum_logp^2 */
    double *counts = calloc(max_len + 1, sizeof(double));
    double *sum_lp = calloc(max_len + 1, sizeof(double));
    double *sum_lp2 = calloc(max_len + 1, sizeof(double));

    for (uint32_t i = 0; i < n_sim; i++) {
        uint32_t L = sims[i].n_tokens;
        double lp = sims[i].log_prob;
        counts[L] += 1.0;
        sum_lp[L] += lp;
        sum_lp2[L] += lp * lp;
    }

    /* Build Gaussian components for lengths with enough samples */
    uint32_t nc = 0;
    for (uint32_t L = 1; L <= max_len && nc < LZG_PGEN_MAX_COMPONENTS; L++) {
        if (counts[L] < 5) continue;
        double w = counts[L] / n_sim;
        double mu = sum_lp[L] / counts[L];
        double var = sum_lp2[L] / counts[L] - mu * mu;
        double sigma = var > 0 ? sqrt(var) : 0.01;

        out->weights[nc] = w;
        out->means[nc] = mu;
        out->stds[nc] = sigma;
        out->walk_lengths[nc] = L;
        nc++;
    }

    out->n_components = nc;

    /* Cleanup */
    for (uint32_t i = 0; i < n_sim; i++) lzg_sim_result_free(&sims[i]);
    free(sims); free(counts); free(sum_lp); free(sum_lp2);

    return LZG_OK;
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
