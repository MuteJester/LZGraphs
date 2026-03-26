/**
 * @file occupancy.c
 * @brief Robust richness and overlap via splitting + Wynn acceleration.
 *
 * For any depth d and any graph size:
 *
 * 1. SPLIT: Simulate N sequences to discover high-probability ones.
 *    Compute their contribution F_large = Σ (1-exp(-d·π(s))) exactly.
 *
 * 2. RESIDUAL: Subtract discovered sequences from the power sums
 *    M_residual(k) = M(k) - Σ_{discovered} π(s)^k.
 *    The residual has tiny p_max, so Taylor converges fast.
 *
 * 3. ACCELERATE: Apply Wynn epsilon to the residual Taylor partial
 *    sums for machine-precision convergence.
 *
 * 4. COMBINE: F(d) = F_large + F_small_residual.
 */
#include "lzgraph/occupancy.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/wynn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Configuration ─────────────────────────────────────── */

#define SPLIT_N_SIM       5000   /* sequences to simulate for splitting */
#define TAYLOR_K_MAX      50     /* max Taylor terms for residual */
#define WYNN_MIN_TERMS    5      /* minimum terms before Wynn */

/* ── Internal: compute richness via splitting + Wynn ───── */

/**
 * Core richness computation.
 *
 * @param g          The graph.
 * @param d          Effective depth.
 * @param M_cache    Pre-computed M(k) values for k=1..K (NULL if not available).
 * @param K          Number of M values in cache (0 if no cache).
 * @param out        Output: F(d).
 */
static LZGError richness_core(const LZGGraph *g, double d,
                               const double *M_cache, uint32_t K,
                               double *out) {
    /* ─── Step 1: Simulate to discover high-probability sequences ─── */

    LZGRng rng;
    lzg_rng_seed(&rng, 42 + (uint64_t)(d * 1000)); /* seed varies with d */

    uint32_t n_sim = SPLIT_N_SIM;
    LZGSimResult *sim = malloc(n_sim * sizeof(LZGSimResult));
    if (!sim) return LZG_ERR_ALLOC;

    LZGError err = lzg_simulate(g, n_sim, &rng, sim);
    if (err != LZG_OK) { free(sim); return err; }

    /* Deduplicate and collect unique (sequence, log_prob) pairs */
    /* Use a hash map: sequence_hash → index in unique array */
    LZGHashMap *seen = lzg_hm_create(n_sim * 2);
    double *unique_probs = malloc(n_sim * sizeof(double));
    uint32_t n_unique = 0;

    for (uint32_t i = 0; i < n_sim; i++) {
        uint64_t h = lzg_hash_bytes(sim[i].sequence, sim[i].seq_len);
        if (!lzg_hm_get(seen, h)) {
            lzg_hm_put(seen, h, n_unique);
            unique_probs[n_unique] = exp(sim[i].log_prob);
            n_unique++;
        }
    }

    /* ─── Step 2: Compute F_large exactly ─── */

    double F_large = 0.0;
    for (uint32_t i = 0; i < n_unique; i++) {
        double p = unique_probs[i];
        F_large += 1.0 - exp(-d * p);
    }

    /* ─── Step 3: Compute residual power sums ─── */

    /* M_residual(k) = M(k) - Σ_{unique} π(s)^k */
    uint32_t K_use = (K > 0 && M_cache) ? K : TAYLOR_K_MAX;
    double *M_resid = malloc((K_use + 1) * sizeof(double));

    for (uint32_t k = 1; k <= K_use; k++) {
        double M_k;
        if (M_cache && k <= K) {
            M_k = M_cache[k];
        } else {
            err = lzg_power_sum(g, (double)k, &M_k);
            if (err != LZG_OK) {
                free(sim); free(seen); free(unique_probs); free(M_resid);
                /* Can't use lzg_hm_destroy here because seen is on stack path */
                return err;
            }
        }

        /* Subtract discovered sequences' contribution */
        double M_found = 0.0;
        for (uint32_t i = 0; i < n_unique; i++)
            M_found += pow(unique_probs[i], (double)k);

        M_resid[k] = M_k - M_found;
        /* Clamp to non-negative (floating point noise) */
        if (M_resid[k] < 0.0) M_resid[k] = 0.0;
    }

    /* ─── Step 4: Taylor series on residual + Wynn acceleration ─── */

    double *partial_sums = malloc((K_use + 1) * sizeof(double));
    double F_resid = 0.0;
    double d_pow = d;
    double factorial = 1.0;
    uint32_t n_terms = 0;

    for (uint32_t k = 1; k <= K_use; k++) {
        if (k > 1) { d_pow *= d; factorial *= k; }

        double sign = (k % 2 == 1) ? 1.0 : -1.0;
        F_resid += sign * d_pow / factorial * M_resid[k];
        partial_sums[n_terms] = F_resid;
        n_terms++;

        /* Early termination if residual terms are negligible */
        double term_mag = fabs(d_pow / factorial * M_resid[k]);
        if (k >= WYNN_MIN_TERMS && term_mag < 1e-15 * fmax(fabs(F_resid), 1e-30))
            break;
    }

    /* Apply Wynn epsilon acceleration */
    double F_small;
    if (n_terms >= 3) {
        F_small = lzg_wynn_epsilon(partial_sums, n_terms);
    } else {
        F_small = F_resid;
    }

    /* ─── Step 5: Combine ─── */

    *out = fmax(F_large + F_small, 0.0);

    /* Cleanup */
    for (uint32_t i = 0; i < n_sim; i++) lzg_sim_result_free(&sim[i]);
    free(sim);
    lzg_hm_destroy(seen);
    free(unique_probs);
    free(M_resid);
    free(partial_sums);

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Public API                                                      */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_predicted_richness(const LZGGraph *g, double d, double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    return richness_core(g, d, NULL, 0, out);
}

LZGError lzg_predicted_overlap(const LZGGraph *g, double d_i, double d_j,
                                double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    double F_i, F_j, F_ij;
    LZGError err;

    err = lzg_predicted_richness(g, d_i, &F_i);
    if (err != LZG_OK) return err;

    err = lzg_predicted_richness(g, d_j, &F_j);
    if (err != LZG_OK) return err;

    err = lzg_predicted_richness(g, d_i + d_j, &F_ij);
    if (err != LZG_OK) return err;

    *out = fmax(F_i + F_j - F_ij, 0.0);
    return LZG_OK;
}

LZGError lzg_richness_curve(const LZGGraph *g, const double *d_values,
                             uint32_t n, double *out) {
    if (!g || !d_values || !out || n == 0) return LZG_ERR_INVALID_ARG;

    /* Find max depth to determine K */
    double d_max = d_values[0];
    for (uint32_t i = 1; i < n; i++)
        if (d_values[i] > d_max) d_max = d_values[i];

    /* Precompute M(k) for k = 1..K_MAX */
    uint32_t K = TAYLOR_K_MAX;
    double *M = malloc((K + 1) * sizeof(double));
    if (!M) return LZG_ERR_ALLOC;

    for (uint32_t k = 1; k <= K; k++) {
        LZGError err = lzg_power_sum(g, (double)k, &M[k]);
        if (err != LZG_OK) { free(M); return err; }
    }

    /* Evaluate at each depth using the shared M cache */
    for (uint32_t i = 0; i < n; i++) {
        LZGError err = richness_core(g, d_values[i], M, K, &out[i]);
        if (err != LZG_OK) { free(M); return err; }
    }

    free(M);
    return LZG_OK;
}
