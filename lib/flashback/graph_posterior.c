/**
 * @file flashback_posterior.c
 * @brief Bayesian posterior FlashBackGraph via Dirichlet-Multinomial update.
 *
 * Creates a copy of the prior graph with updated edge weights:
 *   w_post(u->v) = (kappa * w_prior(u->v) + c_ind(u->v))
 *                  / (kappa + n_ind(u))
 *
 * where c_ind is the individual's edge count (derived from FlashBack
 * decomposition of `sequences`) and n_ind is the individual's total
 * outgoing count at node u. The output graph has the same topology as
 * the prior — new edges implied by the sequences are ignored.
 *
 * Mirrors lib/posterior/posterior.c (which is LZ76-specific) but uses
 * the FlashBack tokeniser and the flashback node-identity convention
 * (node_pos == UINT32_MAX; nodes are identified solely by sp_id).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lzgraph/flashback_graph.h"
#include "lzgraph/flashback.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"

LZGError lzg_flashback_graph_posterior(const LZGGraph *prior,
                                       const char **sequences,
                                       uint32_t n_seqs,
                                       const uint64_t *abundances,
                                       double kappa,
                                       LZGGraph **out) {
    if (!prior || !out) return LZG_ERR_INVALID_ARG;
    if (n_seqs > 0 && !sequences) return LZG_ERR_INVALID_ARG;
    if (kappa < 0.0) return LZG_ERR_INVALID_ARG;

    LZGHashMap *ind_edges = NULL;
    LZGHashMap *ind_outgoing = NULL;
    LZGHashMap *sp_to_node = NULL;
    LZGGraph *post = NULL;

    /* ─── Step 1: build sp_id -> node index map (flashback convention) ─── */
    sp_to_node = lzg_hm_create(prior->n_nodes > 0 ? prior->n_nodes * 2 : 16);
    if (!sp_to_node) goto oom;
    for (uint32_t i = 0; i < prior->n_nodes; i++) {
        lzg_hm_put(sp_to_node, (uint64_t)prior->node_sp_id[i], (uint64_t)i);
    }

    /* ─── Step 2: tokenise sequences and count individual edge traversals ─── */
    uint32_t init_edge_cap = n_seqs > 0 ? n_seqs * 8 : 64;
    ind_edges = lzg_hm_create(init_edge_cap);
    ind_outgoing = lzg_hm_create(1024);
    if (!ind_edges || !ind_outgoing) goto oom;

    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        uint32_t seq_len = (uint32_t)strlen(seq);
        uint64_t count = abundances ? abundances[s] : 1;
        if (count == 0) continue;

        LZGFlashbackTokens tokens;
        LZGError err = lzg_flashback_decompose(
            seq, seq_len, (LZGStringPool *)prior->pool, &tokens);
        if (err != LZG_OK) continue;
        if (tokens.count < 2) continue;

        for (uint32_t t = 0; t + 1 < tokens.count; t++) {
            uint64_t *src_ni = lzg_hm_get(sp_to_node, (uint64_t)tokens.sp_ids[t]);
            uint64_t *dst_ni = lzg_hm_get(sp_to_node, (uint64_t)tokens.sp_ids[t + 1]);
            if (!src_ni || !dst_ni) continue;

            uint32_t src = (uint32_t)*src_ni;
            uint32_t dst = (uint32_t)*dst_ni;
            uint64_t ekey = ((uint64_t)src << 32) | (uint64_t)dst;

            uint64_t *existing = lzg_hm_get(ind_edges, ekey);
            if (existing) {
                *existing += count;
            } else {
                lzg_hm_put(ind_edges, ekey, count);
            }

            uint64_t *oc = lzg_hm_get(ind_outgoing, (uint64_t)src);
            if (oc) {
                *oc += count;
            } else {
                lzg_hm_put(ind_outgoing, (uint64_t)src, count);
            }
        }
    }

    /* ─── Step 3: deep copy the prior graph ─── */
    post = (LZGGraph *)calloc(1, sizeof(LZGGraph));
    if (!post) goto oom;
    post->variant = prior->variant;
    post->smoothing_alpha = prior->smoothing_alpha;
    post->n_nodes = prior->n_nodes;
    post->n_edges = prior->n_edges;
    post->max_length = prior->max_length;
    post->root_node = prior->root_node;
    post->topo_valid = prior->topo_valid;

    post->pool = lzg_sp_create(prior->pool->count + 64);
    if (!post->pool) goto oom;
    for (uint32_t i = 0; i < prior->pool->count; i++) {
        lzg_sp_intern_n(post->pool,
                        lzg_sp_get(prior->pool, i),
                        lzg_sp_len(prior->pool, i));
    }

    uint32_t nn = prior->n_nodes;
    uint32_t ne = prior->n_edges;

    #define DUP_ARR(dst_field, n, type) do { \
        post->dst_field = (type *)malloc((size_t)(n) * sizeof(type)); \
        if (!post->dst_field) goto oom; \
        memcpy(post->dst_field, prior->dst_field, (size_t)(n) * sizeof(type)); \
    } while (0)

    DUP_ARR(row_offsets,      nn + 1,  uint32_t);
    DUP_ARR(col_indices,      ne,      uint32_t);
    DUP_ARR(edge_counts,      ne,      uint64_t);
    DUP_ARR(edge_sp_id,       ne,      uint32_t);
    DUP_ARR(edge_sp_len,      ne,      uint8_t);
    DUP_ARR(edge_prefix_id,   ne,      uint32_t);
    DUP_ARR(outgoing_counts,  nn,      uint64_t);
    DUP_ARR(node_sp_id,       nn,      uint32_t);
    DUP_ARR(node_sp_len,      nn,      uint8_t);
    DUP_ARR(node_pos,         nn,      uint32_t);
    DUP_ARR(topo_order,       nn,      uint32_t);
    if (prior->node_is_sink) {
        DUP_ARR(node_is_sink, nn,      uint8_t);
    }
    if (prior->length_counts && prior->max_length + 1 > 0) {
        DUP_ARR(length_counts, prior->max_length + 1, uint64_t);
    }

    #undef DUP_ARR

    /* edge_weights is freshly allocated and filled below */
    post->edge_weights = (double *)malloc((size_t)ne * sizeof(double));
    if (!post->edge_weights) goto oom;

    /* ─── Step 4: apply Dirichlet-Multinomial update ─── */
    for (uint32_t u = 0; u < nn; u++) {
        uint32_t e_start = post->row_offsets[u];
        uint32_t e_end   = post->row_offsets[u + 1];
        if (e_start == e_end) continue;

        uint64_t *ind_n = lzg_hm_get(ind_outgoing, (uint64_t)u);
        double n_ind = ind_n ? (double)*ind_n : 0.0;

        double denom = kappa + n_ind;
        if (denom < 1e-300) denom = 1e-300;

        double wsum = 0.0;
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = post->col_indices[e];
            uint64_t ekey = ((uint64_t)u << 32) | (uint64_t)v;

            double w_prior = prior->edge_weights[e];
            uint64_t *ind_c = lzg_hm_get(ind_edges, ekey);
            double c_ind = ind_c ? (double)*ind_c : 0.0;

            post->edge_weights[e] = (kappa * w_prior + c_ind) / denom;
            wsum += post->edge_weights[e];
        }

        /* Renormalise: rows should sum to 1 even at numerical edge cases. */
        if (wsum > 0.0) {
            for (uint32_t e = e_start; e < e_end; e++) {
                post->edge_weights[e] /= wsum;
            }
        }
    }

    lzg_hm_destroy(ind_edges);
    lzg_hm_destroy(ind_outgoing);
    lzg_hm_destroy(sp_to_node);

    *out = post;
    return LZG_OK;

oom:
    lzg_hm_destroy(ind_edges);
    lzg_hm_destroy(ind_outgoing);
    lzg_hm_destroy(sp_to_node);
    if (post) lzg_graph_destroy(post);
    return LZG_ERR_ALLOC;
}
