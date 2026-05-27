/**
 * @file flashback_subtract.c
 * @brief Remove the contribution of a list of sequences from a FlashBackGraph.
 *
 * For each sequence (abundance a), decompose via FlashBack into a token
 * walk and subtract `a` from every edge count on the walk (clamped at 0).
 * After all subtractions:
 *   - edges whose count reached 0 are pruned from the CSR;
 *   - isolated nodes (no in/out edges) are retained for index stability;
 *   - per-source weights are renormalised;
 *   - topo_order and sink flags are rebuilt.
 *
 * Primary use: construct a leave-donor-out foundation graph from an
 * existing graph in seconds, without reingesting the full training data.
 *
 * Returns a new graph; the caller owns it. The input graph is untouched.
 */
#include <stdlib.h>
#include <string.h>

#include "lzgraph/flashback_graph.h"
#include "lzgraph/flashback.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"

LZGError lzg_flashback_graph_subtract(const LZGGraph *g,
                                      const char **sequences,
                                      uint32_t n_seqs,
                                      const uint64_t *abundances,
                                      LZGGraph **out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (n_seqs > 0 && !sequences) return LZG_ERR_INVALID_ARG;

    LZGHashMap *sp_to_node = NULL;
    LZGGraph *sub = NULL;
    uint32_t *new_out_deg = NULL;

    /* Scratch arrays for compaction (freed in all exit paths) */
    uint32_t *new_row_offsets = NULL;
    uint32_t *new_col = NULL;
    uint64_t *new_counts = NULL;
    uint32_t *new_sp_id = NULL;
    uint8_t  *new_sp_len = NULL;
    uint32_t *new_prefix = NULL;
    double   *new_weights = NULL;

    /* ─── Step 1: deep copy graph ─── */
    sub = (LZGGraph *)calloc(1, sizeof(LZGGraph));
    if (!sub) goto oom;
    sub->variant = g->variant;
    sub->smoothing_alpha = g->smoothing_alpha;
    sub->n_nodes = g->n_nodes;
    sub->n_edges = g->n_edges;
    sub->max_length = g->max_length;
    sub->root_node = g->root_node;
    sub->topo_valid = g->topo_valid;

    sub->pool = lzg_sp_create(g->pool->count + 64);
    if (!sub->pool) goto oom;
    for (uint32_t i = 0; i < g->pool->count; i++) {
        lzg_sp_intern_n(sub->pool,
                        lzg_sp_get(g->pool, i),
                        lzg_sp_len(g->pool, i));
    }

    uint32_t nn = g->n_nodes;
    uint32_t ne = g->n_edges;

    #define DUP_ARR(dst_field, n, type) do { \
        sub->dst_field = (type *)malloc((size_t)(n) * sizeof(type)); \
        if (!sub->dst_field) goto oom; \
        memcpy(sub->dst_field, g->dst_field, (size_t)(n) * sizeof(type)); \
    } while (0)

    DUP_ARR(row_offsets,     nn + 1, uint32_t);
    DUP_ARR(col_indices,     ne,     uint32_t);
    DUP_ARR(edge_weights,    ne,     double);
    DUP_ARR(edge_counts,     ne,     uint64_t);
    DUP_ARR(edge_sp_id,      ne,     uint32_t);
    DUP_ARR(edge_sp_len,     ne,     uint8_t);
    DUP_ARR(edge_prefix_id,  ne,     uint32_t);
    DUP_ARR(outgoing_counts, nn,     uint64_t);
    DUP_ARR(node_sp_id,      nn,     uint32_t);
    DUP_ARR(node_sp_len,     nn,     uint8_t);
    DUP_ARR(node_pos,        nn,     uint32_t);
    DUP_ARR(topo_order,      nn,     uint32_t);
    if (g->node_is_sink) {
        DUP_ARR(node_is_sink, nn,    uint8_t);
    }
    if (g->length_counts && g->max_length + 1 > 0) {
        DUP_ARR(length_counts, g->max_length + 1, uint64_t);
    }

    #undef DUP_ARR

    /* ─── Step 2: build sp_id -> node index map (flashback convention) ─── */
    sp_to_node = lzg_hm_create(nn > 0 ? nn * 2 : 16);
    if (!sp_to_node) goto oom;
    for (uint32_t i = 0; i < nn; i++) {
        lzg_hm_put(sp_to_node, (uint64_t)sub->node_sp_id[i], (uint64_t)i);
    }

    /* ─── Step 3: decompose each sequence, subtract along the walk ─── */
    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        uint32_t seq_len = (uint32_t)strlen(seq);
        uint64_t count = abundances ? abundances[s] : 1;
        if (count == 0) continue;

        LZGFlashbackTokens tokens;
        LZGError derr = lzg_flashback_decompose(
            seq, seq_len, (LZGStringPool *)g->pool, &tokens);
        if (derr != LZG_OK) continue;
        if (tokens.count < 2) continue;

        for (uint32_t t = 0; t + 1 < tokens.count; t++) {
            uint64_t *src_ni = lzg_hm_get(sp_to_node, (uint64_t)tokens.sp_ids[t]);
            uint64_t *dst_ni = lzg_hm_get(sp_to_node, (uint64_t)tokens.sp_ids[t + 1]);
            if (!src_ni || !dst_ni) continue;

            uint32_t src = (uint32_t)*src_ni;
            uint32_t dst = (uint32_t)*dst_ni;

            /* Find edge (src, dst) and subtract count (clamped at zero). */
            uint32_t e_start = sub->row_offsets[src];
            uint32_t e_end   = sub->row_offsets[src + 1];
            for (uint32_t e = e_start; e < e_end; e++) {
                if (sub->col_indices[e] == dst) {
                    uint64_t current = sub->edge_counts[e];
                    uint64_t decr = (count > current) ? current : count;
                    sub->edge_counts[e] = current - decr;
                    break;
                }
            }
        }
    }

    lzg_hm_destroy(sp_to_node);
    sp_to_node = NULL;

    /* ─── Step 4: compact CSR — remove zero-count edges ─── */
    new_out_deg = (uint32_t *)calloc(nn, sizeof(uint32_t));
    if (!new_out_deg) goto oom;

    uint32_t new_n_edges = 0;
    for (uint32_t u = 0; u < nn; u++) {
        uint32_t e_start = sub->row_offsets[u];
        uint32_t e_end   = sub->row_offsets[u + 1];
        for (uint32_t e = e_start; e < e_end; e++) {
            if (sub->edge_counts[e] > 0) {
                new_out_deg[u]++;
                new_n_edges++;
            }
        }
    }

    if (new_n_edges < ne) {
        new_row_offsets = (uint32_t *)malloc((size_t)(nn + 1) * sizeof(uint32_t));
        new_col         = (uint32_t *)malloc((size_t)new_n_edges * sizeof(uint32_t));
        new_counts      = (uint64_t *)malloc((size_t)new_n_edges * sizeof(uint64_t));
        new_sp_id       = (uint32_t *)malloc((size_t)new_n_edges * sizeof(uint32_t));
        new_sp_len      = (uint8_t  *)malloc((size_t)new_n_edges * sizeof(uint8_t));
        new_prefix      = (uint32_t *)malloc((size_t)new_n_edges * sizeof(uint32_t));
        new_weights     = (double   *)malloc((size_t)(new_n_edges > 0 ? new_n_edges : 1) * sizeof(double));

        if (!new_row_offsets || !new_col || !new_counts ||
            !new_sp_id || !new_sp_len || !new_prefix || !new_weights) goto oom;

        new_row_offsets[0] = 0;
        for (uint32_t u = 0; u < nn; u++) {
            new_row_offsets[u + 1] = new_row_offsets[u] + new_out_deg[u];
        }

        uint32_t new_e = 0;
        for (uint32_t u = 0; u < nn; u++) {
            uint32_t e_start = sub->row_offsets[u];
            uint32_t e_end   = sub->row_offsets[u + 1];
            for (uint32_t e = e_start; e < e_end; e++) {
                if (sub->edge_counts[e] > 0) {
                    new_col[new_e]     = sub->col_indices[e];
                    new_counts[new_e]  = sub->edge_counts[e];
                    new_sp_id[new_e]   = sub->edge_sp_id[e];
                    new_sp_len[new_e]  = sub->edge_sp_len[e];
                    new_prefix[new_e]  = sub->edge_prefix_id[e];
                    new_weights[new_e] = 0.0; /* recomputed below */
                    new_e++;
                }
            }
        }

        free(sub->row_offsets);     sub->row_offsets    = new_row_offsets;    new_row_offsets = NULL;
        free(sub->col_indices);     sub->col_indices    = new_col;            new_col = NULL;
        free(sub->edge_counts);     sub->edge_counts    = new_counts;         new_counts = NULL;
        free(sub->edge_sp_id);      sub->edge_sp_id     = new_sp_id;          new_sp_id = NULL;
        free(sub->edge_sp_len);     sub->edge_sp_len    = new_sp_len;         new_sp_len = NULL;
        free(sub->edge_prefix_id);  sub->edge_prefix_id = new_prefix;         new_prefix = NULL;
        free(sub->edge_weights);    sub->edge_weights   = new_weights;        new_weights = NULL;
        sub->n_edges = new_n_edges;
    }

    free(new_out_deg);
    new_out_deg = NULL;

    /* ─── Step 5: recompute outgoing_counts + edge_weights ─── */
    LZGError err = lzg_graph_recalculate(sub, LZG_RECALC_WEIGHTS);
    if (err != LZG_OK) { lzg_graph_destroy(sub); return err; }

    /* ─── Step 6: rebuild topological order (graph shape may have changed) ─── */
    sub->topo_valid = false;
    err = lzg_graph_topo_sort(sub);
    if (err != LZG_OK && err != LZG_ERR_HAS_CYCLES) {
        lzg_graph_destroy(sub);
        return err;
    }

    /* ─── Step 7: re-identify root/sinks for flashback conventions ─── */
    lzg_flashback_fix_special_nodes(sub);

    *out = sub;
    return LZG_OK;

oom:
    if (sp_to_node) lzg_hm_destroy(sp_to_node);
    free(new_out_deg);
    free(new_row_offsets);
    free(new_col);
    free(new_counts);
    free(new_sp_id);
    free(new_sp_len);
    free(new_prefix);
    free(new_weights);
    if (sub) lzg_graph_destroy(sub);
    return LZG_ERR_ALLOC;
}
