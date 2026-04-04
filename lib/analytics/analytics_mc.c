#include "analytics_mc.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"
#include <math.h>
#include <stdlib.h>

static uint32_t normalize_n_samples(uint32_t n_samples) {
    return n_samples > 0 ? n_samples : LZG_ANALYTICS_DEFAULT_MC_SAMPLES;
}

LZGError lzg_analytics_mc_run(const LZGGraph *g, uint32_t n_samples,
                              uint64_t seed, LZGAnalyticsMCResult *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    out->log_probs = NULL;
    out->n = 0;

    n_samples = normalize_n_samples(n_samples);
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;

    LZGRng rng;
    lzg_rng_seed(&rng, seed);

    LZGSimResult *results = calloc(n_samples, sizeof(LZGSimResult));
    if (!results) return LZG_ERR_ALLOC;

    {
        LZGError err = lzg_simulate(g, n_samples, &rng, results);
        if (err != LZG_OK) {
            free(results);
            return err;
        }
    }

    out->log_probs = malloc(n_samples * sizeof(double));
    if (!out->log_probs) {
        for (uint32_t i = 0; i < n_samples; i++)
            lzg_sim_result_free(&results[i]);
        free(results);
        return LZG_ERR_ALLOC;
    }
    out->n = n_samples;

    for (uint32_t i = 0; i < n_samples; i++) {
        out->log_probs[i] = results[i].log_prob;
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return LZG_OK;
}

void lzg_analytics_mc_free(LZGAnalyticsMCResult *mc) {
    free(mc->log_probs);
    mc->log_probs = NULL;
    mc->n = 0;
}

bool lzg_analytics_mc_is_valid_log_prob(double log_prob) {
    return log_prob > LZG_LOG_EPS + 1.0;
}

uint32_t lzg_analytics_mc_valid_count(const LZGAnalyticsMCResult *mc) {
    uint32_t valid = 0;

    for (uint32_t i = 0; i < mc->n; i++) {
        if (lzg_analytics_mc_is_valid_log_prob(mc->log_probs[i]))
            valid++;
    }

    return valid;
}

double lzg_analytics_mc_absorbed_mass(const LZGAnalyticsMCResult *mc) {
    if (!mc || mc->n == 0) return 0.0;
    return (double)lzg_analytics_mc_valid_count(mc) / (double)mc->n;
}

double lzg_analytics_mc_support_estimate(const LZGAnalyticsMCResult *mc) {
    double sum = 0.0;

    for (uint32_t i = 0; i < mc->n; i++) {
        if (!lzg_analytics_mc_is_valid_log_prob(mc->log_probs[i])) continue;
        sum += exp(-mc->log_probs[i]);
    }

    return mc->n > 0 ? sum / mc->n : 0.0;
}

void lzg_analytics_mc_entropy_stats(const LZGAnalyticsMCResult *mc,
                                    double *sum_lp, uint32_t *valid) {
    *sum_lp = 0.0;
    *valid = 0;

    for (uint32_t i = 0; i < mc->n; i++) {
        if (!lzg_analytics_mc_is_valid_log_prob(mc->log_probs[i])) continue;
        *sum_lp += mc->log_probs[i];
        (*valid)++;
    }
}

double lzg_analytics_mc_power_mean(const LZGAnalyticsMCResult *mc, double alpha) {
    double sum = 0.0;

    for (uint32_t i = 0; i < mc->n; i++) {
        if (!lzg_analytics_mc_is_valid_log_prob(mc->log_probs[i])) continue;
        sum += exp((alpha - 1.0) * mc->log_probs[i]);
    }

    return mc->n > 0 ? sum / mc->n : 0.0;
}

double lzg_analytics_mc_hill_estimate(const LZGAnalyticsMCResult *mc, double alpha) {
    double absorbed = lzg_analytics_mc_absorbed_mass(mc);

    if (fabs(alpha - 1.0) < 1e-12) {
        double sum_lp = 0.0;
        uint32_t valid = 0;
        lzg_analytics_mc_entropy_stats(mc, &sum_lp, &valid);
        if (valid == 0 || absorbed <= 0.0) return 0.0;
        return exp(-(sum_lp / valid) + log(absorbed));
    }

    if (fabs(alpha) < 1e-12) {
        return lzg_analytics_mc_support_estimate(mc);
    }

    {
        double m_alpha = lzg_analytics_mc_power_mean(mc, alpha);
        double normalized = absorbed > 0.0
            ? m_alpha / pow(absorbed, alpha)
            : 0.0;
        return normalized > 0.0
            ? pow(normalized, 1.0 / (1.0 - alpha))
            : 0.0;
    }
}
