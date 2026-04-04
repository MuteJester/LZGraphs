/**
 * @file analytics.c
 * @brief Graph analytics via simulation with exact-probability estimators.
 *
 * Each simulated walk follows the raw constrained process once:
 * it either reaches a sink (absorbed) or gets stranded at a dead end (leaked).
 * Absorbed samples carry exact walk probabilities; leaked samples contribute
 * zero mass to raw power-sum estimates.
 *
 * Estimators:
 *   M(a): (1/N) Σ I_abs * P(s_i)^(a-1)   = raw power sum Σ P(s)^a
 *   D(0): support size over absorbed sequences
 *   D(1): exp(H_cond), where H_cond is Shannon entropy of the
 *         absorbed-conditional distribution
 *   D(a): classical Hill number on the absorbed-conditional distribution:
 *         (M(a) / M(1)^a)^{1/(1-a)}
 *
 * This keeps diagnostics honest on leaky graphs while preserving
 * classical Hill semantics on the normalized completed-sequence law.
 */
#include "lzgraph/analytics.h"
#include "analytics_mc.h"
#include "../simulation/exact_model.h"
#include <math.h>
#include <string.h>

#define MC_PATH_COUNT_SEED         54321ULL
#define MC_EFFECTIVE_DIVERSITY_SEED 11111ULL
#define MC_POWER_SUM_SEED          22222ULL
#define MC_HILL_NUMBER_SEED        33333ULL
#define MC_HILL_NUMBERS_SEED       44444ULL
#define MC_DYNAMIC_RANGE_SEED      12345ULL

/* ═══════════════════════════════════════════════════════════════ */
/* Public API                                                      */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_path_count(const LZGGraph *g, double *out) {
    return lzg_graph_path_count_mc(g, 0, out);
}

LZGError lzg_graph_path_count_mc(const LZGGraph *g, uint32_t n_samples,
                                  double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, n_samples, MC_PATH_COUNT_SEED, &mc);
    if (err != LZG_OK) return err;
    *out = lzg_analytics_mc_support_estimate(&mc);
    lzg_analytics_mc_free(&mc);
    return LZG_OK;
}

LZGError lzg_pgen_diagnostics(const LZGGraph *g, double atol, LZGPgenDiagnostics *out) {
    double absorbed;
    LZGError err;

    if (!g || !out) return LZG_ERR_INVALID_ARG;
    out->initial_prob_sum = 1.0;

    /* Use the raw-walk MC estimate (not the accepted-walk model, which
       would always report 1.0 since it rejection-samples to absorption). */
    err = lzg_exact_model_ensure((LZGGraph *)g);
    if (err != LZG_OK) return err;

    absorbed = lzg_exact_model_root_absorption(g);
    out->total_absorbed = absorbed;
    out->total_leaked = 1.0 - absorbed;
    out->is_proper = fabs(absorbed - 1.0) < atol;
    out->mc_samples = lzg_exact_model_mc_samples(g);
    return LZG_OK;
}

LZGError lzg_effective_diversity(const LZGGraph *g, LZGEffectiveDiversity *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, 0, MC_EFFECTIVE_DIVERSITY_SEED, &mc);
    if (err != LZG_OK) return err;

    /* H_cond = -E_q[log π(S)] + log M(1), where q is the absorbed law. */
    double sum_lp = 0.0;
    uint32_t valid = 0;
    double absorbed = lzg_analytics_mc_absorbed_mass(&mc);
    lzg_analytics_mc_entropy_stats(&mc, &sum_lp, &valid);

    if (valid == 0 || absorbed <= 0.0) {
        memset(out, 0, sizeof(*out));
        lzg_analytics_mc_free(&mc);
        return LZG_OK;
    }

    double support = lzg_analytics_mc_support_estimate(&mc);
    out->entropy_nats = -(sum_lp / valid) + log(absorbed);
    out->entropy_bits = out->entropy_nats / log(2.0);
    out->effective_diversity = exp(out->entropy_nats);
    out->uniformity = support > 0.0
        ? fmin(out->effective_diversity / support, 1.0)
        : 0.0;

    lzg_analytics_mc_free(&mc);
    return LZG_OK;
}

LZGError lzg_power_sum(const LZGGraph *g, double alpha, double *out_m) {
    if (!g || !out_m) return LZG_ERR_INVALID_ARG;

    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, 0, MC_POWER_SUM_SEED, &mc);
    if (err != LZG_OK) return err;

    /* M̂(α) = (1/N) Σ exp((α-1) * log P(s_i)) — unbiased importance sampling */
    *out_m = lzg_analytics_mc_power_mean(&mc, alpha);
    lzg_analytics_mc_free(&mc);
    return LZG_OK;
}

LZGError lzg_hill_number(const LZGGraph *g, double alpha, double *out_d) {
    return lzg_hill_number_mc(g, alpha, 0, out_d);
}

LZGError lzg_hill_number_mc(const LZGGraph *g, double alpha,
                             uint32_t n_samples, double *out_d) {
    if (!g || !out_d) return LZG_ERR_INVALID_ARG;

    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, n_samples, MC_HILL_NUMBER_SEED, &mc);
    if (err != LZG_OK) return err;

    *out_d = lzg_analytics_mc_hill_estimate(&mc, alpha);
    lzg_analytics_mc_free(&mc);
    return LZG_OK;
}

LZGError lzg_hill_numbers(const LZGGraph *g, const double *orders,
                           uint32_t n, double *out) {
    return lzg_hill_numbers_mc(g, orders, n, 0, out);
}

LZGError lzg_hill_numbers_mc(const LZGGraph *g, const double *orders,
                              uint32_t n, uint32_t n_samples, double *out) {
    if (!g || !orders || !out) return LZG_ERR_INVALID_ARG;
    /* Single simulation, compute all orders from it */
    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, n_samples, MC_HILL_NUMBERS_SEED, &mc);
    if (err != LZG_OK) return err;

    for (uint32_t i = 0; i < n; i++) {
        out[i] = lzg_analytics_mc_hill_estimate(&mc, orders[i]);
    }

    lzg_analytics_mc_free(&mc);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* PGEN dynamic range (min/max log P via simulation)               */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_pgen_dynamic_range(const LZGGraph *g, LZGDynamicRange *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    LZGAnalyticsMCResult mc;
    LZGError err = lzg_analytics_mc_run(g, 10000, MC_DYNAMIC_RANGE_SEED, &mc);
    if (err != LZG_OK) return err;

    double min_lp = 0.0, max_lp = -1e300;
    bool first = true;
    for (uint32_t i = 0; i < mc.n; i++) {
        double lp = mc.log_probs[i];
        if (lzg_analytics_mc_is_valid_log_prob(lp)) {
            if (first || lp > max_lp) max_lp = lp;
            if (first || lp < min_lp) min_lp = lp;
            first = false;
        }
    }
    lzg_analytics_mc_free(&mc);

    out->max_log_prob = max_lp;
    out->min_log_prob = min_lp;
    out->dynamic_range_nats = max_lp - min_lp;
    out->dynamic_range_orders = out->dynamic_range_nats / log(10.0);
    return LZG_OK;
}
