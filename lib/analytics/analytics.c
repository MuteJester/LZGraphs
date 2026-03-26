/**
 * @file analytics.c
 * @brief Graph analytics via simulation with exact-probability estimators.
 *
 * Each simulated walk carries its exact log-probability from the
 * LZ-constrained walk engine. We use these exact probabilities —
 * not empirical frequencies — for Hill numbers and entropy estimation.
 *
 * Estimators:
 *   D(0): Chao1 lower bound from frequencies (can't do better without enumeration)
 *   D(1): exp(-(1/N) Σ log P(s_i))  — unbiased, negligible Jensen bias at N≥1000
 *   D(2): 1 / ((1/N) Σ P(s_i))  — unbiased for Simpson concentration
 *   D(α): ((1/N) Σ P(s_i)^(α-1))^{1/(1-α)}  — unbiased importance-sampling
 *   H:    -(1/N) Σ log P(s_i)  — unbiased for Shannon entropy
 *
 * Why exact-probability estimators beat frequency-based:
 *   When true diversity >> N (most sequences unique in simulation),
 *   frequency-based entropy saturates at log(N). But each sample carries
 *   its exact P(s), so the probability-based estimator captures the full
 *   diversity even from a small sample.
 */
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Default MC sample size. 10K gives ~1% accuracy for D(1), D(2). */
#define MC_N_SAMPLES 10000

/* ── Shared: simulate N walks, return log-probs ───────────── */

typedef struct {
    double   *log_probs;   /* [n]: exact log P(s_i) per walk            */
    uint32_t  n;           /* number of simulations                     */
    /* Frequency data (for D(0) Chao1 estimation) */
    uint32_t  n_unique;
    uint32_t  f1;          /* singletons                                */
    uint32_t  f2;          /* doubletons                                */
} MCResult;

static LZGError run_mc(const LZGGraph *g, uint32_t n_samples,
                        uint64_t seed, MCResult *out) {
    LZGRng rng;
    lzg_rng_seed(&rng, seed);

    LZGSimResult *results = calloc(n_samples, sizeof(LZGSimResult));
    if (!results) return LZG_ERR_ALLOC;

    LZGError err = lzg_simulate(g, n_samples, &rng, results);
    if (err != LZG_OK) { free(results); return err; }

    /* Extract log-probs and compute frequency stats */
    out->log_probs = malloc(n_samples * sizeof(double));
    out->n = n_samples;

    LZGHashMap *counts = lzg_hm_create(n_samples * 2);
    for (uint32_t i = 0; i < n_samples; i++) {
        out->log_probs[i] = results[i].log_prob;
        uint64_t h = lzg_hash_bytes(results[i].sequence, results[i].seq_len);
        uint64_t *existing = lzg_hm_get(counts, h);
        if (existing) (*existing)++;
        else lzg_hm_put(counts, h, 1);
        lzg_sim_result_free(&results[i]);
    }
    free(results);

    /* Count singletons and doubletons for Chao1 */
    out->n_unique = counts->count;
    out->f1 = 0;
    out->f2 = 0;
    for (uint32_t i = 0; i < counts->capacity; i++) {
        if (counts->keys[i] != LZG_HM_EMPTY &&
            counts->keys[i] != LZG_HM_DELETED) {
            uint32_t f = (uint32_t)counts->values[i];
            if (f == 1) out->f1++;
            if (f == 2) out->f2++;
        }
    }
    lzg_hm_destroy(counts);
    return LZG_OK;
}

static void free_mc(MCResult *r) {
    free(r->log_probs);
    r->log_probs = NULL;
}

/* ── Chao1 (for D(0) only — frequency-based lower bound) ──── */

static double chao1(const MCResult *r) {
    double s = (double)r->n_unique;
    double f1 = (double)r->f1;
    double f2 = (double)r->f2;
    if (f2 > 0) return s + f1 * f1 / (2.0 * f2);
    return s + f1 * (f1 - 1.0) / 2.0;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Public API                                                      */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_path_count(const LZGGraph *g, double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    MCResult mc;
    LZGError err = run_mc(g, MC_N_SAMPLES, 54321, &mc);
    if (err != LZG_OK) return err;
    *out = chao1(&mc);
    free_mc(&mc);
    return LZG_OK;
}

LZGError lzg_pgen_diagnostics(const LZGGraph *g, double atol, LZGPgenDiagnostics *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    out->initial_prob_sum = 1.0;

    MCResult mc;
    LZGError err = run_mc(g, 1000, 99999, &mc);
    if (err != LZG_OK) return err;

    uint32_t valid = 0;
    for (uint32_t i = 0; i < mc.n; i++)
        if (mc.log_probs[i] > LZG_LOG_EPS + 1.0) valid++;

    out->total_absorbed = (double)valid / mc.n;
    out->total_leaked = 1.0 - out->total_absorbed;
    out->is_proper = valid > (uint32_t)(mc.n * 0.95);

    free_mc(&mc);
    return LZG_OK;
}

LZGError lzg_effective_diversity(const LZGGraph *g, LZGEffectiveDiversity *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    MCResult mc;
    LZGError err = run_mc(g, MC_N_SAMPLES, 11111, &mc);
    if (err != LZG_OK) return err;

    /* H = -(1/N) Σ log P(s_i) — unbiased for Shannon entropy */
    double sum_lp = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < mc.n; i++) {
        if (mc.log_probs[i] > LZG_LOG_EPS + 1.0) {
            sum_lp += mc.log_probs[i];
            valid++;
        }
    }

    if (valid == 0) { memset(out, 0, sizeof(*out)); free_mc(&mc); return LZG_OK; }

    out->entropy_nats = -sum_lp / valid;
    out->entropy_bits = out->entropy_nats / log(2.0);
    out->effective_diversity = exp(out->entropy_nats);
    out->uniformity = chao1(&mc) > 0 ? out->effective_diversity / chao1(&mc) : 0.0;

    free_mc(&mc);
    return LZG_OK;
}

LZGError lzg_power_sum(const LZGGraph *g, double alpha, double *out_m) {
    if (!g || !out_m) return LZG_ERR_INVALID_ARG;

    MCResult mc;
    LZGError err = run_mc(g, MC_N_SAMPLES, 22222, &mc);
    if (err != LZG_OK) return err;

    /* M̂(α) = (1/N) Σ exp((α-1) * log P(s_i)) — unbiased importance sampling */
    double sum = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < mc.n; i++) {
        if (mc.log_probs[i] > LZG_LOG_EPS + 1.0) {
            sum += exp((alpha - 1.0) * mc.log_probs[i]);
            valid++;
        }
    }

    *out_m = valid > 0 ? sum / valid : 0.0;
    free_mc(&mc);
    return LZG_OK;
}

LZGError lzg_hill_number(const LZGGraph *g, double alpha, double *out_d) {
    if (!g || !out_d) return LZG_ERR_INVALID_ARG;

    MCResult mc;
    LZGError err = run_mc(g, MC_N_SAMPLES, 33333, &mc);
    if (err != LZG_OK) return err;

    if (fabs(alpha) < 1e-12) {
        /* D(0) = Chao1 richness (lower bound from frequencies) */
        *out_d = chao1(&mc);
    } else if (fabs(alpha - 1.0) < 1e-12) {
        /* D(1) = exp(H) where H = -(1/N) Σ log P(s_i) */
        double sum_lp = 0.0;
        uint32_t valid = 0;
        for (uint32_t i = 0; i < mc.n; i++) {
            if (mc.log_probs[i] > LZG_LOG_EPS + 1.0) {
                sum_lp += mc.log_probs[i];
                valid++;
            }
        }
        *out_d = valid > 0 ? exp(-sum_lp / valid) : 0.0;
    } else {
        /* D(α) = M(α)^{1/(1-α)} where M(α) = (1/N) Σ P(s_i)^(α-1) */
        double sum = 0.0;
        uint32_t valid = 0;
        for (uint32_t i = 0; i < mc.n; i++) {
            if (mc.log_probs[i] > LZG_LOG_EPS + 1.0) {
                sum += exp((alpha - 1.0) * mc.log_probs[i]);
                valid++;
            }
        }
        double m_alpha = valid > 0 ? sum / valid : 0.0;
        *out_d = m_alpha > 0 ? pow(m_alpha, 1.0 / (1.0 - alpha)) : 0.0;
    }

    free_mc(&mc);
    return LZG_OK;
}

LZGError lzg_hill_numbers(const LZGGraph *g, const double *orders,
                           uint32_t n, double *out) {
    /* Single simulation, compute all orders from it */
    MCResult mc;
    LZGError err = run_mc(g, MC_N_SAMPLES, 44444, &mc);
    if (err != LZG_OK) return err;

    for (uint32_t i = 0; i < n; i++) {
        double alpha = orders[i];
        if (fabs(alpha) < 1e-12) {
            out[i] = chao1(&mc);
        } else if (fabs(alpha - 1.0) < 1e-12) {
            double sum_lp = 0.0;
            uint32_t valid = 0;
            for (uint32_t j = 0; j < mc.n; j++) {
                if (mc.log_probs[j] > LZG_LOG_EPS + 1.0) {
                    sum_lp += mc.log_probs[j];
                    valid++;
                }
            }
            out[i] = valid > 0 ? exp(-sum_lp / valid) : 0.0;
        } else {
            double sum = 0.0;
            uint32_t valid = 0;
            for (uint32_t j = 0; j < mc.n; j++) {
                if (mc.log_probs[j] > LZG_LOG_EPS + 1.0) {
                    sum += exp((alpha - 1.0) * mc.log_probs[j]);
                    valid++;
                }
            }
            double m = valid > 0 ? sum / valid : 0.0;
            out[i] = m > 0 ? pow(m, 1.0 / (1.0 - alpha)) : 0.0;
        }
    }

    free_mc(&mc);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* PGEN dynamic range (min/max log P via simulation)               */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_pgen_dynamic_range(const LZGGraph *g, LZGDynamicRange *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    MCResult mc;
    LZGError err = run_mc(g, 10000, 12345, &mc);
    if (err != LZG_OK) return err;

    double min_lp = 0.0, max_lp = -1e300;
    bool first = true;
    for (uint32_t i = 0; i < mc.n; i++) {
        double lp = mc.log_probs[i];
        if (lp > LZG_LOG_EPS + 1.0) {
            if (first || lp > max_lp) max_lp = lp;
            if (first || lp < min_lp) min_lp = lp;
            first = false;
        }
    }
    free_mc(&mc);

    out->max_log_prob = max_lp;
    out->min_log_prob = min_lp;
    out->dynamic_range_nats = max_lp - min_lp;
    out->dynamic_range_orders = out->dynamic_range_nats / log(10.0);
    return LZG_OK;
}
