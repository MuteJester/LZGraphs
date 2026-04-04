/**
 * @file occupancy.c
 * @brief Public occupancy API over internal richness and numeric helpers.
 */
#include "lzgraph/occupancy.h"
#include "occupancy_internal.h"
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════ */
/* Public API                                                      */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_predicted_richness(const LZGGraph *g, double d, double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    return lzg_occupancy_richness_impl(g, d, NULL, 0, out);
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
    double *power_sum_cache = NULL;
    uint32_t n_terms = LZG_OCCUPANCY_TAYLOR_K_MAX;

    if (!g || !d_values || !out || n == 0) return LZG_ERR_INVALID_ARG;

    power_sum_cache = malloc((n_terms + 1) * sizeof(double));
    if (!power_sum_cache) return LZG_ERR_ALLOC;

    {
        LZGError err = lzg_occupancy_build_power_sum_cache(g, n_terms,
                                                           power_sum_cache);
        if (err != LZG_OK) {
            free(power_sum_cache);
            return err;
        }
    }

    for (uint32_t i = 0; i < n; i++) {
        LZGError err = lzg_occupancy_richness_impl(
            g, d_values[i], power_sum_cache, n_terms, &out[i]);
        if (err != LZG_OK) {
            free(power_sum_cache);
            return err;
        }
    }

    free(power_sum_cache);
    return LZG_OK;
}
