/**
 * @file occupancy.c
 * @brief Public occupancy API over internal richness and numeric helpers.
 */
#include <math.h>
#include <float.h>
#include <stdlib.h>

#include "lzgraph/occupancy.h"

#include "occupancy_internal.h"

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

static void compensated_add(long double value, long double *sum,
                            long double *correction) {
    const long double adjusted = value - *correction;
    const long double updated = *sum + adjusted;
    *correction = (updated - *sum) - adjusted;
    *sum = updated;
}

LZGError lzg_pseq_discovery_curve(
    const double *probabilities, const double *multiplicities,
    uint32_t n_atoms, const double *draw_counts, uint32_t n_draw_counts,
    double *richness_out, double *novelty_out) {
    if ((n_atoms && (!probabilities || !multiplicities)) ||
        (n_draw_counts && (!draw_counts || !richness_out || !novelty_out)))
        return LZG_ERR_INVALID_ARG;
    if (n_draw_counts == 0) return LZG_OK;
    for (uint32_t j = 0; j < n_draw_counts; j++) {
        if (!isfinite(draw_counts[j]) || draw_counts[j] < 1.0)
            return LZG_ERR_PARAM_OUT_OF_RANGE;
    }
    for (uint32_t a = 0; a < n_atoms; a++) {
        if (!isfinite(probabilities[a]) || probabilities[a] < 0.0 ||
            probabilities[a] > 1.0 || !isfinite(multiplicities[a]) ||
            multiplicities[a] < 0.0)
            return LZG_ERR_PARAM_OUT_OF_RANGE;
    }
    long double *richness = (long double *)calloc(n_draw_counts,
                                                  sizeof(long double));
    long double *novelty = (long double *)calloc(n_draw_counts,
                                                 sizeof(long double));
    long double *richness_correction = (long double *)calloc(
        n_draw_counts, sizeof(long double));
    long double *novelty_correction = (long double *)calloc(
        n_draw_counts, sizeof(long double));
    if (!richness || !novelty || !richness_correction ||
        !novelty_correction) {
        free(richness); free(novelty);
        free(richness_correction); free(novelty_correction);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t a = 0; a < n_atoms; a++) {
        const double probability = probabilities[a];
        const double multiplicity = multiplicities[a];
        if (probability == 0.0 || multiplicity == 0.0) continue;
        const long double count = (long double)multiplicity;
        if (probability == 1.0) {
            for (uint32_t j = 0; j < n_draw_counts; j++) {
                compensated_add(count, &richness[j],
                                &richness_correction[j]);
                if (draw_counts[j] == 1.0)
                    compensated_add(count, &novelty[j],
                                    &novelty_correction[j]);
            }
            continue;
        }
        const double log_survival = log1p(-probability);
        for (uint32_t j = 0; j < n_draw_counts; j++) {
            const double draws = draw_counts[j];
            const long double observed = (long double)(
                -expm1(draws * log_survival));
            const double exponent = (draws - 1.0) * log_survival;
            long double novelty_contribution;
            if (exponent >= log(DBL_MIN)) {
                novelty_contribution = count *
                    (long double)probability * exp(exponent);
            } else {
                const long double log_term =
                    logl(count) + logl((long double)probability) +
                    (long double)exponent;
                novelty_contribution = log_term >= logl(LDBL_MIN)
                    ? expl(log_term) : 0.0L;
            }
            compensated_add(count * observed, &richness[j],
                            &richness_correction[j]);
            compensated_add(novelty_contribution, &novelty[j],
                            &novelty_correction[j]);
        }
    }

    for (uint32_t j = 0; j < n_draw_counts; j++) {
        richness_out[j] = (double)richness[j];
        novelty_out[j] = (double)novelty[j];
    }
    free(richness); free(novelty);
    free(richness_correction); free(novelty_correction);
    return LZG_OK;
}
