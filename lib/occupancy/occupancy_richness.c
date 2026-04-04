#include "occupancy_internal.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/simulate.h"
#include <math.h>
#include <stdlib.h>

void lzg_occupancy_split_destroy(LZGOccupancySplit *split) {
    if (!split) return;

    if (split->samples) {
        for (uint32_t i = 0; i < split->n_samples; i++)
            lzg_sim_result_free(&split->samples[i]);
    }

    free(split->samples);
    lzg_hm_destroy(split->seen);
    free(split->unique_probs);

    split->samples = NULL;
    split->seen = NULL;
    split->unique_probs = NULL;
    split->n_samples = 0;
    split->n_unique = 0;
}

LZGError lzg_occupancy_discover_probabilities(const LZGGraph *g, double d,
                                              LZGOccupancySplit *out) {
    LZGOccupancySplit split = {0};
    LZGRng rng;
    LZGError err;

    split.samples = malloc(LZG_OCCUPANCY_SPLIT_N_SIM * sizeof(LZGSimResult));
    split.seen = lzg_hm_create(LZG_OCCUPANCY_SPLIT_N_SIM * 2u);
    split.unique_probs = malloc(LZG_OCCUPANCY_SPLIT_N_SIM * sizeof(double));
    if (!split.samples || !split.seen || !split.unique_probs) {
        free(split.samples);
        lzg_hm_destroy(split.seen);
        free(split.unique_probs);
        return LZG_ERR_ALLOC;
    }

    lzg_rng_seed(&rng, 42u + (uint64_t)(d * 1000.0));
    err = lzg_simulate(g, LZG_OCCUPANCY_SPLIT_N_SIM, &rng, split.samples);
    if (err != LZG_OK) {
        free(split.samples);
        lzg_hm_destroy(split.seen);
        free(split.unique_probs);
        return err;
    }

    split.n_samples = LZG_OCCUPANCY_SPLIT_N_SIM;
    for (uint32_t i = 0; i < split.n_samples; i++) {
        uint64_t seq_hash = lzg_hash_bytes(split.samples[i].sequence,
                                           split.samples[i].seq_len);
        if (!lzg_hm_get(split.seen, seq_hash)) {
            lzg_hm_put(split.seen, seq_hash, split.n_unique);
            split.unique_probs[split.n_unique] = exp(split.samples[i].log_prob);
            split.n_unique++;
        }
    }

    *out = split;
    return LZG_OK;
}

double lzg_occupancy_large_contribution(const LZGOccupancySplit *split,
                                        double d) {
    double total = 0.0;

    for (uint32_t i = 0; i < split->n_unique; i++)
        total += 1.0 - exp(-d * split->unique_probs[i]);

    return total;
}

LZGError lzg_occupancy_richness_impl(const LZGGraph *g, double d,
                                     const double *power_sum_cache,
                                     uint32_t n_terms,
                                     double *out) {
    LZGOccupancySplit split = {0};
    double *residual_power_sums = NULL;
    double large = 0.0;
    double small = 0.0;
    uint32_t n_use = (power_sum_cache && n_terms > 0)
        ? n_terms : LZG_OCCUPANCY_TAYLOR_K_MAX;
    LZGError err = lzg_occupancy_discover_probabilities(g, d, &split);

    if (err != LZG_OK) return err;

    residual_power_sums = malloc((n_use + 1) * sizeof(double));
    if (!residual_power_sums) {
        lzg_occupancy_split_destroy(&split);
        return LZG_ERR_ALLOC;
    }

    err = lzg_occupancy_fill_residual_power_sums(
        g, &split, power_sum_cache, n_use, residual_power_sums);
    if (err != LZG_OK) {
        free(residual_power_sums);
        lzg_occupancy_split_destroy(&split);
        return err;
    }

    err = lzg_occupancy_accelerated_residual(d, residual_power_sums, n_use, &small);
    if (err != LZG_OK) {
        free(residual_power_sums);
        lzg_occupancy_split_destroy(&split);
        return err;
    }

    large = lzg_occupancy_large_contribution(&split, d);
    if (!isfinite(small) || small < 0.0)
        small = 0.0;
    *out = large + small;

    free(residual_power_sums);
    lzg_occupancy_split_destroy(&split);
    return LZG_OK;
}
