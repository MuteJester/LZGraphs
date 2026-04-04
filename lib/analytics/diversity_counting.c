#include "diversity_internal.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/lz76.h"
#include "lzgraph/string_pool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

LZGError lzg_k_diversity_impl(const char **sequences, uint32_t n,
                              LZGVariant variant,
                              uint32_t sample_size, uint32_t draws,
                              LZGRng *rng, LZGKDiversity *out) {
    LZGStringPool *pool;
    double *counts;
    uint32_t *indices;

    if (!sequences || !rng || !out || n == 0 || sample_size == 0 || draws == 0)
        return LZG_ERR_INVALID_ARG;

    if (sample_size > n) sample_size = n;

    pool = lzg_sp_create(1024);
    counts = malloc(draws * sizeof(double));
    indices = malloc(n * sizeof(uint32_t));
    if (!pool || !counts || !indices) {
        lzg_sp_destroy(pool);
        free(counts);
        free(indices);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t d = 0; d < draws; d++) {
        LZGHashMap *seen;

        for (uint32_t i = 0; i < n; i++) indices[i] = i;

        for (uint32_t i = 0; i < sample_size; i++) {
            uint32_t j = i + lzg_rng_bounded(rng, n - i);
            uint32_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        seen = lzg_hm_create(sample_size * 8u);
        if (!seen) {
            lzg_sp_destroy(pool);
            free(counts);
            free(indices);
            return LZG_ERR_ALLOC;
        }

        for (uint32_t i = 0; i < sample_size; i++) {
            const char *seq = sequences[indices[i]];
            uint32_t seq_len = (uint32_t)strlen(seq);
            uint32_t node_ids[LZG_MAX_TOKENS];
            uint32_t sp_ids[LZG_MAX_TOKENS];
            uint32_t n_tokens = 0;

            lzg_lz76_encode(seq, seq_len, pool, variant,
                            node_ids, sp_ids, &n_tokens);

            for (uint32_t t = 0; t < n_tokens; t++)
                lzg_hm_put(seen, sp_ids[t], 1);
        }

        counts[d] = (double)seen->count;
        lzg_hm_destroy(seen);
    }

    {
        double sum = 0.0;
        double sum_sq = 0.0;
        double variance;
        double standard_error;

        for (uint32_t d = 0; d < draws; d++) {
            sum += counts[d];
            sum_sq += counts[d] * counts[d];
        }

        out->mean = sum / (double)draws;
        variance = (sum_sq / (double)draws) - (out->mean * out->mean);
        out->std = variance > 0.0 ? sqrt(variance) : 0.0;

        standard_error = out->std / sqrt((double)draws);
        out->ci_low = out->mean - 1.96 * standard_error;
        out->ci_high = out->mean + 1.96 * standard_error;
    }

    lzg_sp_destroy(pool);
    free(counts);
    free(indices);
    return LZG_OK;
}

LZGError lzg_saturation_curve_impl(const char **sequences, uint32_t n,
                                   LZGVariant variant,
                                   uint32_t log_every,
                                   LZGSaturationPoint *out,
                                   uint32_t *out_count) {
    LZGStringPool *pool;
    LZGHashMap *node_set;
    LZGHashMap *edge_set;
    uint32_t points = 0;

    if (!sequences || !out || !out_count || n == 0 || log_every == 0)
        return LZG_ERR_INVALID_ARG;

    pool = lzg_sp_create(1024);
    node_set = lzg_hm_create(n * 8u);
    edge_set = lzg_hm_create(n * 8u);
    if (!pool || !node_set || !edge_set) {
        lzg_sp_destroy(pool);
        lzg_hm_destroy(node_set);
        lzg_hm_destroy(edge_set);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t s = 0; s < n; s++) {
        const char *seq = sequences[s];
        uint32_t seq_len = (uint32_t)strlen(seq);
        uint32_t node_ids[LZG_MAX_TOKENS];
        uint32_t sp_ids[LZG_MAX_TOKENS];
        uint32_t n_tokens = 0;

        lzg_lz76_encode(seq, seq_len, pool, variant,
                        node_ids, sp_ids, &n_tokens);

        for (uint32_t t = 0; t < n_tokens; t++)
            lzg_hm_put(node_set, node_ids[t], 1);

        for (uint32_t t = 0; t + 1 < n_tokens; t++) {
            uint64_t edge_key = ((uint64_t)node_ids[t] << 32) | node_ids[t + 1];
            lzg_hm_put(edge_set, edge_key, 1);
        }

        if ((s + 1) % log_every == 0 || s == n - 1) {
            out[points].n_sequences = s + 1;
            out[points].n_nodes = node_set->count;
            out[points].n_edges = edge_set->count;
            points++;
        }
    }

    *out_count = points;
    lzg_sp_destroy(pool);
    lzg_hm_destroy(node_set);
    lzg_hm_destroy(edge_set);
    return LZG_OK;
}
