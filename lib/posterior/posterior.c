/**
 * @file posterior.c
 * @brief Bayesian posterior graph via Dirichlet-Multinomial update.
 *
 * Creates a copy of the prior graph with updated edge weights:
 *   w_post(u→v) = (kappa * w_prior(u→v) + c_ind(u→v)) / (kappa + n_ind(u))
 *
 * where c_ind is the individual's edge count and n_ind is the
 * individual's total outgoing count at node u.
 */
#include "lzgraph/posterior.h"
#include "lzgraph/lz76.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

LZGError lzg_graph_posterior(const LZGGraph *prior,
                        const char **sequences, uint32_t n_seqs,
                        const uint32_t *abundances,
                        double kappa,
                        LZGGraph **out) {
    if (!prior || !sequences || !out) return LZG_ERR_INVALID_ARG;

    /* ─── Step 1: Count individual edge traversals ─── */

    /* Map: pack(src_node_label_id, dst_node_label_id) → count */
    LZGHashMap *ind_edges = lzg_hm_create(n_seqs * 8);
    LZGHashMap *ind_outgoing = lzg_hm_create(1024);

    /* Build label → node index map for the prior */
    LZGHashMap *label_map = lzg_hm_create(prior->n_nodes * 2);
    for (uint32_t i = 0; i < prior->n_nodes; i++) {
        const char *sp = lzg_sp_get(prior->pool, prior->node_sp_id[i]);
        char buf[256];
        int len = snprintf(buf, sizeof(buf), "%s_%u", sp, prior->node_pos[i]);
        uint32_t lid = lzg_sp_intern_n((LZGStringPool *)prior->pool, buf, (uint32_t)len);
        lzg_hm_put(label_map, (uint64_t)lid, (uint64_t)i);
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        uint32_t seq_len = (uint32_t)strlen(seq);
        uint32_t count = abundances ? abundances[s] : 1;

        uint32_t node_ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
        uint32_t n_tokens;
        LZGError err = lzg_lz76_encode(seq, seq_len,
                                    (LZGStringPool *)prior->pool,
                                    prior->variant,
                                    node_ids, sp_ids, &n_tokens);
        if (err != LZG_OK || n_tokens < 2) continue;

        for (uint32_t t = 0; t < n_tokens - 1; t++) {
            /* Map label IDs to graph node indices */
            uint64_t *src_gi = lzg_hm_get(label_map, (uint64_t)node_ids[t]);
            uint64_t *dst_gi = lzg_hm_get(label_map, (uint64_t)node_ids[t + 1]);
            if (!src_gi || !dst_gi) continue;

            uint32_t src = (uint32_t)*src_gi;
            uint32_t dst = (uint32_t)*dst_gi;

            /* Check edge exists in prior */
            uint64_t ekey = ((uint64_t)src << 32) | dst;
            uint64_t *existing = lzg_hm_get(ind_edges, ekey);
            if (existing) *existing += count;
            else lzg_hm_put(ind_edges, ekey, count);

            /* Track outgoing counts */
            uint64_t *oc = lzg_hm_get(ind_outgoing, (uint64_t)src);
            if (oc) *oc += count;
            else lzg_hm_put(ind_outgoing, (uint64_t)src, count);
        }
    }

    /* ─── Step 2: Create posterior graph (deep copy of prior) ─── */

    LZGGraph *post = calloc(1, sizeof(LZGGraph));
    post->variant = prior->variant;
    post->smoothing_alpha = prior->smoothing_alpha;
    post->smoothing_alpha = prior->smoothing_alpha;
    post->n_nodes = prior->n_nodes;
    post->n_edges = prior->n_edges;
    /* n_initial/n_terminal removed — sentinel model */
    post->max_length = prior->max_length;
    post->root_node = prior->root_node;

    /* Copy string pool (share by reference — pool is immutable) */
    /* For simplicity, create a new pool and re-intern all strings */
    post->pool = lzg_sp_create(prior->pool->count + 64);
    for (uint32_t i = 0; i < prior->pool->count; i++)
        lzg_sp_intern_n(post->pool, lzg_sp_get(prior->pool, i),
                          lzg_sp_len(prior->pool, i));

    /* Copy arrays */
    uint32_t nn = prior->n_nodes, ne = prior->n_edges;

    #define COPY_ARR(dst, src, n, type) do { \
        dst = malloc((n) * sizeof(type)); \
        memcpy(dst, src, (n) * sizeof(type)); \
    } while(0)

    COPY_ARR(post->row_offsets, prior->row_offsets, nn + 1, uint32_t);
    COPY_ARR(post->col_indices, prior->col_indices, ne, uint32_t);
    COPY_ARR(post->edge_counts, prior->edge_counts, ne, uint32_t);
    COPY_ARR(post->edge_sp_id, prior->edge_sp_id, ne, uint32_t);
    COPY_ARR(post->edge_sp_len, prior->edge_sp_len, ne, uint8_t);
    COPY_ARR(post->edge_prefix_id, prior->edge_prefix_id, ne, uint32_t);
    COPY_ARR(post->outgoing_counts, prior->outgoing_counts, nn, uint32_t);
    COPY_ARR(post->node_sp_id, prior->node_sp_id, nn, uint32_t);
    COPY_ARR(post->node_sp_len, prior->node_sp_len, nn, uint8_t);
    COPY_ARR(post->node_pos, prior->node_pos, nn, uint32_t);
    /* initial/terminal arrays removed — sentinel model */
    COPY_ARR(post->length_counts, prior->length_counts, prior->max_length + 1, uint32_t);
    COPY_ARR(post->topo_order, prior->topo_order, nn, uint32_t);
    if (prior->node_is_sink) {
        COPY_ARR(post->node_is_sink, prior->node_is_sink, nn, uint8_t);
    }
    post->topo_valid = prior->topo_valid;

    #undef COPY_ARR

    /* Allocate new edge_weights (will be updated) */
    post->edge_weights = malloc(ne * sizeof(double));

    /* ─── Step 3: Update edge weights with Dirichlet-Multinomial ─── */

    for (uint32_t u = 0; u < nn; u++) {
        uint32_t e_start = post->row_offsets[u];
        uint32_t e_end   = post->row_offsets[u + 1];
        if (e_start == e_end) continue;

        /* Individual's total outgoing count at this node */
        uint64_t *ind_n = lzg_hm_get(ind_outgoing, (uint64_t)u);
        double n_ind = ind_n ? (double)*ind_n : 0.0;

        double denom = kappa + n_ind;
        if (denom < 1e-300) denom = 1e-300;

        double wsum = 0.0;
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = post->col_indices[e];
            uint64_t ekey = ((uint64_t)u << 32) | v;

            double w_prior = prior->edge_weights[e];
            uint64_t *ind_c = lzg_hm_get(ind_edges, ekey);
            double c_ind = ind_c ? (double)*ind_c : 0.0;

            post->edge_weights[e] = (kappa * w_prior + c_ind) / denom;
            wsum += post->edge_weights[e];
        }

        /* Renormalize to ensure sum = 1.0 */
        if (wsum > 0.0) {
            for (uint32_t e = e_start; e < e_end; e++)
                post->edge_weights[e] /= wsum;
        }
    }

    /* Cleanup */
    lzg_hm_destroy(ind_edges);
    lzg_hm_destroy(ind_outgoing);
    lzg_hm_destroy(label_map);

    LZGError err = LZG_OK;
    if (err != LZG_OK) {
        lzg_graph_destroy(post);
        return err;
    }

    *out = post;
    return LZG_OK;
}
