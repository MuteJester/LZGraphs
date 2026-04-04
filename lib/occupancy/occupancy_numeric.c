#include "occupancy_internal.h"
#include "lzgraph/analytics.h"
#include "lzgraph/wynn.h"
#include <math.h>
#include <stdlib.h>

LZGError lzg_occupancy_build_power_sum_cache(const LZGGraph *g,
                                             uint32_t n_terms,
                                             double *out_cache) {
    for (uint32_t k = 1; k <= n_terms; k++) {
        LZGError err = lzg_power_sum(g, (double)k, &out_cache[k]);
        if (err != LZG_OK) return err;
    }

    return LZG_OK;
}

LZGError lzg_occupancy_fill_residual_power_sums(const LZGGraph *g,
                                                const LZGOccupancySplit *split,
                                                const double *power_sum_cache,
                                                uint32_t n_terms,
                                                double *out_residual) {
    for (uint32_t k = 1; k <= n_terms; k++) {
        double power_sum = 0.0;
        double found_mass = 0.0;

        if (power_sum_cache) {
            power_sum = power_sum_cache[k];
        } else {
            LZGError err = lzg_power_sum(g, (double)k, &power_sum);
            if (err != LZG_OK) return err;
        }

        for (uint32_t i = 0; i < split->n_unique; i++)
            found_mass += pow(split->unique_probs[i], (double)k);

        out_residual[k] = power_sum - found_mass;
        if (out_residual[k] < 0.0) out_residual[k] = 0.0;
    }

    return LZG_OK;
}

LZGError lzg_occupancy_accelerated_residual(double d,
                                            const double *residual_power_sums,
                                            uint32_t n_terms,
                                            double *out) {
    double *partial_sums = malloc((n_terms + 1) * sizeof(double));
    double residual = 0.0;
    double d_pow = d;
    double factorial = 1.0;
    uint32_t n_partial = 0;

    if (!partial_sums) return LZG_ERR_ALLOC;

    for (uint32_t k = 1; k <= n_terms; k++) {
        double sign = (k % 2 == 1) ? 1.0 : -1.0;
        double term;

        if (k > 1) {
            d_pow *= d;
            factorial *= k;
        }

        term = sign * d_pow / factorial * residual_power_sums[k];
        residual += term;
        partial_sums[n_partial++] = residual;

        if (k >= LZG_OCCUPANCY_WYNN_MIN_TERMS &&
            fabs(term) < 1e-15 * fmax(fabs(residual), 1e-30)) {
            break;
        }
    }

    if (n_partial >= 3) {
        *out = lzg_wynn_epsilon(partial_sums, n_partial);
    } else {
        *out = residual;
    }

    free(partial_sums);
    return LZG_OK;
}
