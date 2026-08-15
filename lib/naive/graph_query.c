/**
 * @file graph_query.c
 * @brief Markov simulation and sequence probability for NaiveGraphs.
 *
 * No LZ constraints: every outgoing edge is always valid, so a walk is a
 * plain weighted random choice and a sequence's probability is the
 * product of the weights along its one possible walk.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzgraph/naive_graph.h"

/* ── Markov random walk ──────────────────────────────────────── */

LZGError lzg_naive_simulate(const LZGGraph *g, uint32_t n,
                            LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "naive simulate: NULL argument");
    if (g->root_node >= g->n_nodes)
        return LZG_FAIL(LZG_ERR_NOT_BUILT, "naive simulate: no root node");

    for (uint32_t seq_idx = 0; seq_idx < n; seq_idx++) {
        uint32_t walk_labels[LZG_NAIVE_MAX_WALK];
        uint32_t walk_len = 0;

        uint32_t cur = g->root_node;
        double log_prob = 0.0;

        while (walk_len < LZG_NAIVE_MAX_WALK) {
            walk_labels[walk_len++] = g->node_sp_id[cur];

            if (g->node_is_sink && g->node_is_sink[cur])
                break;

            uint32_t e_start = g->row_offsets[cur];
            uint32_t e_end   = g->row_offsets[cur + 1];
            if (e_start == e_end) break; /* dead end */

            double u = lzg_rng_double(rng);
            double cumul = 0.0;
            uint32_t chosen = e_start;
            for (uint32_t e = e_start; e < e_end; e++) {
                cumul += g->edge_weights[e];
                if (u < cumul) { chosen = e; break; }
                chosen = e;
            }

            double w = g->edge_weights[chosen];
            if (w > LZG_EPS)
                log_prob += log(w);
            else
                log_prob = LZG_LOG_EPS;

            cur = g->col_indices[chosen];
        }

        char seq_buf[LZG_NAIVE_MAX_WALK];
        uint32_t seq_len = 0;
        LZGError err = lzg_naive_reverse(g->pool, walk_labels, walk_len,
                                         seq_buf, sizeof(seq_buf), &seq_len);
        if (err != LZG_OK) {
            seq_buf[0] = '\0';
            seq_len = 0;
        }

        out[seq_idx].sequence = strdup(seq_buf);
        out[seq_idx].seq_len  = seq_len;
        out[seq_idx].n_tokens = walk_len;
        out[seq_idx].log_prob = log_prob;
    }

    return LZG_OK;
}

/* ── Walk log-probability ────────────────────────────────────── */

/**
 * Lazily build the label-ID -> node-index map used by pgen.
 *
 * The key layout matches what the builder writes (position UINT32_MAX,
 * identity carried by the label), so it is the same key FlashBack and the
 * generic posterior/subtract paths use.
 */
static LZGHashMap *naive_query_map(const LZGGraph *g) {
    LZGGraph *gm = (LZGGraph *)g;
    if (gm->query_node_map) return gm->query_node_map;

    LZGHashMap *map = lzg_hm_create(g->n_nodes * 2);
    if (!map) return NULL;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint64_t key = ((uint64_t)g->node_sp_id[i] << 32) |
                       (uint64_t)g->node_pos[i];
        lzg_hm_put(map, key, (uint64_t)i);
    }
    gm->query_node_map = map;
    return map;
}

/** Resolve one node label to its index, or UINT32_MAX if absent. */
static uint32_t naive_lookup_node(const LZGGraph *g, LZGHashMap *map,
                                  const char *label) {
    uint32_t sp_id = lzg_sp_find(g->pool, label);
    if (sp_id == LZG_SP_NOT_FOUND) return UINT32_MAX;
    uint64_t key = ((uint64_t)sp_id << 32) | (uint64_t)UINT32_MAX;
    uint64_t *slot = lzg_hm_get(map, key);
    return slot ? (uint32_t)*slot : UINT32_MAX;
}

/** Weight of edge prev -> nid, or 0 if there is no such edge. */
static double naive_edge_weight(const LZGGraph *g, uint32_t prev,
                                uint32_t nid) {
    uint32_t e_start = g->row_offsets[prev];
    uint32_t e_end   = g->row_offsets[prev + 1];
    for (uint32_t e = e_start; e < e_end; e++)
        if (g->col_indices[e] == nid)
            return g->edge_weights[e];
    return 0.0;
}

double lzg_naive_pgen(const LZGGraph *g,
                      const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return LZG_LOG_EPS;
    if (g->root_node >= g->n_nodes) return LZG_LOG_EPS;
    if (seq_len + 2u > LZG_NAIVE_MAX_WALK) return LZG_LOG_EPS;

    LZGHashMap *map = naive_query_map(g);
    if (!map) return LZG_LOG_EPS;

    double log_p = 0.0;
    uint32_t prev_nid = g->root_node;
    char label[LZG_NAIVE_LABEL_CAP];

    for (uint32_t i = 0; i < seq_len; i++) {
        int written = snprintf(label, sizeof(label), "%c_%u", seq[i], i + 1u);
        if (written <= 0 || (size_t)written >= sizeof(label))
            return LZG_LOG_EPS;

        uint32_t nid = naive_lookup_node(g, map, label);
        if (nid == UINT32_MAX) return LZG_LOG_EPS;

        double w = naive_edge_weight(g, prev_nid, nid);
        if (w < LZG_EPS) return LZG_LOG_EPS;
        log_p += log(w);
        prev_nid = nid;
    }

    /* Close the walk at the sink; a sequence whose final residue never
     * ended a training sequence is not in the model's support. */
    {
        static const char end_label[2] = {LZG_END_SENTINEL, '\0'};
        uint32_t sink = naive_lookup_node(g, map, end_label);
        if (sink == UINT32_MAX) return LZG_LOG_EPS;
        double w = naive_edge_weight(g, prev_nid, sink);
        if (w < LZG_EPS) return LZG_LOG_EPS;
        log_p += log(w);
    }

    return log_p > LZG_LOG_EPS ? log_p : LZG_LOG_EPS;
}

LZGError lzg_naive_pgen_batch(const LZGGraph *g,
                              const char **sequences,
                              uint32_t n, double *out) {
    if (!g || !sequences || !out) return LZG_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < n; i++)
        out[i] = lzg_naive_pgen(g, sequences[i],
                                (uint32_t)strlen(sequences[i]));
    return LZG_OK;
}
