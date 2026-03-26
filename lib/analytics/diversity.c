/**
 * @file diversity.c
 * @brief Diversity metrics using the LZ-constrained model.
 */
#include "lzgraph/diversity.h"
#include "lzgraph/simulate.h"
#include "lzgraph/lz76.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_LN2
#define M_LN2 0.6931471805599453
#endif

/* ═══════════════════════════════════════════════════════════════ */
/* Perplexity (constrained model)                                  */
/* ═══════════════════════════════════════════════════════════════ */

double lzg_sequence_perplexity(const LZGGraph *g,
                                const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return INFINITY;

    double log_p = lzg_walk_log_prob(g, seq, seq_len);
    if (log_p <= LZG_LOG_EPS + 1.0) return INFINITY;

    /* Encode to count tokens */
    LZGTokens tokens;
    lzg_lz76_decompose(seq, seq_len, (LZGStringPool *)g->pool, &tokens);
    if (tokens.count == 0) return INFINITY;

    /* PP = 2^{-log2(P) / n} = exp(-logP / (n * ln2) * ln2) = exp(-logP / n) */
    /* Actually: PP = 2^{-log2(P)/n} = exp2(-log2(P)/n)
     * log2(P) = log(P) / ln(2)
     * PP = 2^{-log(P)/(n*ln2)} = exp(-log(P)/n)  ... no.
     * PP = 2^{H} where H = -log2(P)/n = -logP/(n*ln2)
     * PP = exp(H * ln2) = exp(-logP/n)
     */
    double n = (double)tokens.count;
    return exp(-log_p / n);
}

double lzg_repertoire_perplexity(const LZGGraph *g,
                                  const char **sequences, uint32_t n) {
    if (!g || !sequences || n == 0) return INFINITY;

    double sum_h = 0.0;
    uint32_t valid = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t slen = (uint32_t)strlen(sequences[i]);
        double log_p = lzg_walk_log_prob(g, sequences[i], slen);
        if (log_p <= LZG_LOG_EPS + 1.0) continue;

        LZGTokens tokens;
        lzg_lz76_decompose(sequences[i], slen, (LZGStringPool *)g->pool, &tokens);
        if (tokens.count == 0) continue;

        sum_h += (-log_p) / (double)tokens.count;
        valid++;
    }

    if (valid == 0) return INFINITY;
    return exp(sum_h / (double)valid);
}

double lzg_path_entropy_rate(const LZGGraph *g,
                              const char **sequences, uint32_t n) {
    if (!g || !sequences || n == 0) return 0.0;

    double sum_h = 0.0;
    uint32_t valid = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t slen = (uint32_t)strlen(sequences[i]);
        double log_p = lzg_walk_log_prob(g, sequences[i], slen);
        if (log_p <= LZG_LOG_EPS + 1.0) continue;

        LZGTokens tokens;
        lzg_lz76_decompose(sequences[i], slen, (LZGStringPool *)g->pool, &tokens);
        if (tokens.count == 0) continue;

        /* Bits per token: -log2(P) / n_tokens = -logP / (n_tokens * ln2) */
        sum_h += (-log_p) / ((double)tokens.count * M_LN2);
        valid++;
    }

    return valid > 0 ? sum_h / (double)valid : 0.0;
}

/* ═══════════════════════════════════════════════════════════════ */
/* K-Diversity (counting, no probability model)                    */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_k_diversity(const char **sequences, uint32_t n,
                          LZGVariant variant,
                          uint32_t sample_size, uint32_t draws,
                          LZGRng *rng, LZGKDiversity *out) {
    if (!sequences || !rng || !out || n == 0 || sample_size == 0 || draws == 0)
        return LZG_ERR_INVALID_ARG;

    if (sample_size > n) sample_size = n;

    LZGStringPool *pool = lzg_sp_create(1024);
    double *counts = malloc(draws * sizeof(double));

    /* Fisher-Yates partial shuffle indices */
    uint32_t *indices = malloc(n * sizeof(uint32_t));

    for (uint32_t d = 0; d < draws; d++) {
        /* Reset indices */
        for (uint32_t i = 0; i < n; i++) indices[i] = i;

        /* Partial shuffle: pick sample_size random elements */
        for (uint32_t i = 0; i < sample_size; i++) {
            uint32_t j = i + lzg_rng_bounded(rng, n - i);
            uint32_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        /* Count unique subpatterns in the sample */
        LZGHashMap *seen = lzg_hm_create(sample_size * 8);
        for (uint32_t i = 0; i < sample_size; i++) {
            const char *seq = sequences[indices[i]];
            uint32_t slen = (uint32_t)strlen(seq);

            uint32_t node_ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
            uint32_t n_tokens;
            lzg_lz76_encode(seq, slen, pool, variant, node_ids, sp_ids, &n_tokens);

            for (uint32_t t = 0; t < n_tokens; t++)
                lzg_hm_put(seen, sp_ids[t], 1);
        }
        counts[d] = (double)seen->count;
        lzg_hm_destroy(seen);
    }

    /* Compute mean and std */
    double sum = 0.0, sum2 = 0.0;
    for (uint32_t d = 0; d < draws; d++) {
        sum += counts[d];
        sum2 += counts[d] * counts[d];
    }
    out->mean = sum / draws;
    double var = (sum2 / draws) - (out->mean * out->mean);
    out->std = var > 0 ? sqrt(var) : 0.0;

    /* 95% CI (normal approximation) */
    double se = out->std / sqrt((double)draws);
    out->ci_low  = out->mean - 1.96 * se;
    out->ci_high = out->mean + 1.96 * se;

    free(counts); free(indices);
    lzg_sp_destroy(pool);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Saturation curve (counting)                                     */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_saturation_curve(const char **sequences, uint32_t n,
                               LZGVariant variant,
                               uint32_t log_every,
                               LZGSaturationPoint *out,
                               uint32_t *out_count) {
    if (!sequences || !out || !out_count || n == 0 || log_every == 0)
        return LZG_ERR_INVALID_ARG;

    LZGStringPool *pool = lzg_sp_create(1024);
    LZGHashMap *node_set = lzg_hm_create(n * 8);
    LZGHashMap *edge_set = lzg_hm_create(n * 8);

    uint32_t pts = 0;

    for (uint32_t s = 0; s < n; s++) {
        const char *seq = sequences[s];
        uint32_t slen = (uint32_t)strlen(seq);

        uint32_t node_ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
        uint32_t n_tokens;
        lzg_lz76_encode(seq, slen, pool, variant, node_ids, sp_ids, &n_tokens);

        for (uint32_t t = 0; t < n_tokens; t++)
            lzg_hm_put(node_set, node_ids[t], 1);

        for (uint32_t t = 0; t + 1 < n_tokens; t++) {
            uint64_t ekey = ((uint64_t)node_ids[t] << 32) | node_ids[t + 1];
            lzg_hm_put(edge_set, ekey, 1);
        }

        if ((s + 1) % log_every == 0 || s == n - 1) {
            out[pts].n_sequences = s + 1;
            out[pts].n_nodes     = node_set->count;
            out[pts].n_edges     = edge_set->count;
            pts++;
        }
    }

    *out_count = pts;
    lzg_hm_destroy(node_set);
    lzg_hm_destroy(edge_set);
    lzg_sp_destroy(pool);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Jensen-Shannon Divergence between two graphs                    */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_jensen_shannon_divergence(const LZGGraph *a,
                                        const LZGGraph *b,
                                        double *out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;

    /* Build unified node frequency vectors.
     * Use a hash map: node_label_string_hash → (freq_a, freq_b) */

    /* Compute node_probability-like distribution from edge counts */
    /* For each node, freq = sum_outgoing_edge_counts / total_edges */

    uint32_t nn_a = a->n_nodes, nn_b = b->n_nodes;

    /* Total outgoing for normalization */
    double total_a = 0, total_b = 0;
    for (uint32_t i = 0; i < nn_a; i++) total_a += a->outgoing_counts[i];
    for (uint32_t i = 0; i < nn_b; i++) total_b += b->outgoing_counts[i];
    if (total_a < 1 || total_b < 1) { *out = 0; return LZG_OK; }

    /* Build label → freq hash maps */
    LZGHashMap *freq_a = lzg_hm_create(nn_a * 2);
    LZGHashMap *freq_b = lzg_hm_create(nn_b * 2);
    LZGHashMap *all_labels = lzg_hm_create((nn_a + nn_b) * 2);

    for (uint32_t i = 0; i < nn_a; i++) {
        uint64_t key = lzg_hash_bytes(lzg_sp_get(a->pool, a->node_sp_id[i]),
                                  lzg_sp_len(a->pool, a->node_sp_id[i]));
        /* Use node_pos to disambiguate */
        key ^= (uint64_t)a->node_pos[i] * 2654435761ULL;
        double freq = a->outgoing_counts[i] / total_a;
        /* Store as uint64 bit pattern */
        uint64_t bits;
        memcpy(&bits, &freq, 8);
        lzg_hm_put(freq_a, key, bits);
        lzg_hm_put(all_labels, key, 1);
    }

    for (uint32_t i = 0; i < nn_b; i++) {
        uint64_t key = lzg_hash_bytes(lzg_sp_get(b->pool, b->node_sp_id[i]),
                                  lzg_sp_len(b->pool, b->node_sp_id[i]));
        key ^= (uint64_t)b->node_pos[i] * 2654435761ULL;
        double freq = b->outgoing_counts[i] / total_b;
        uint64_t bits;
        memcpy(&bits, &freq, 8);
        lzg_hm_put(freq_b, key, bits);
        lzg_hm_put(all_labels, key, 1);
    }

    /* Compute JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q) */
    double jsd = 0.0;
    double eps = 1e-300;

    for (uint32_t i = 0; i < all_labels->capacity; i++) {
        if (all_labels->keys[i] == LZG_HM_EMPTY ||
            all_labels->keys[i] == LZG_HM_DELETED) continue;

        uint64_t key = all_labels->keys[i];
        double pa = 0.0, pb = 0.0;

        uint64_t *va = lzg_hm_get(freq_a, key);
        if (va) memcpy(&pa, va, 8);
        uint64_t *vb = lzg_hm_get(freq_b, key);
        if (vb) memcpy(&pb, vb, 8);

        double m = 0.5 * (pa + pb);
        if (m < eps) continue;

        if (pa > eps) jsd += 0.5 * pa * log(pa / m);
        if (pb > eps) jsd += 0.5 * pb * log(pb / m);
    }

    *out = fmax(jsd, 0.0);

    lzg_hm_destroy(freq_a);
    lzg_hm_destroy(freq_b);
    lzg_hm_destroy(all_labels);
    return LZG_OK;
}
