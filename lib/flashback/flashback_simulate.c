/**
 * @file flashback_simulate.c
 * @brief Markov simulation and walk probability for FlashBack graphs.
 *
 * No LZ constraints — every outgoing edge is always valid.
 * Sequences are reconstructed via lzg_flashback_reverse().
 */
#include "lzgraph/flashback.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FB_MAX_WALK_LEN 4096

/* ── Markov random walk ──────────────────────────────────────── */

LZGError lzg_flashback_simulate(const LZGGraph *g, uint32_t n,
                                LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback simulate: NULL argument");
    if (g->root_node >= g->n_nodes)
        return LZG_FAIL(LZG_ERR_NOT_BUILT, "flashback simulate: no root node");

    for (uint32_t seq_idx = 0; seq_idx < n; seq_idx++) {
        /* Collect token IDs along the walk */
        LZGFlashbackTokens walk_tokens;
        walk_tokens.count = 0;

        uint32_t cur = g->root_node;
        double log_prob = 0.0;

        while (walk_tokens.count < FB_MAX_WALK_LEN) {
            /* Record current node's token */
            if (walk_tokens.count < LZG_FLASHBACK_MAX_TOKENS)
                walk_tokens.sp_ids[walk_tokens.count++] = g->node_sp_id[cur];

            /* Sink check */
            if (g->node_is_sink && g->node_is_sink[cur])
                break;

            uint32_t e_start = g->row_offsets[cur];
            uint32_t e_end   = g->row_offsets[cur + 1];
            if (e_start == e_end) break; /* dead end */

            /* Weighted random choice */
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

        /* Reverse the token walk to get the original sequence */
        char seq_buf[2048];
        uint32_t seq_len = 0;
        LZGError err = lzg_flashback_reverse(g->pool, &walk_tokens,
                                             seq_buf, sizeof(seq_buf),
                                             &seq_len);

        if (err != LZG_OK) {
            /* Fallback: empty sequence */
            seq_buf[0] = '\0';
            seq_len = 0;
        }

        out[seq_idx].sequence = strdup(seq_buf);
        out[seq_idx].seq_len  = seq_len;
        out[seq_idx].n_tokens = walk_tokens.count;
        out[seq_idx].log_prob = log_prob;
    }

    return LZG_OK;
}

/* ── Walk log-probability ────────────────────────────────────── */

double lzg_flashback_pgen(const LZGGraph *g,
                          const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return LZG_LOG_EPS;
    if (g->root_node >= g->n_nodes) return LZG_LOG_EPS;

    /* Decompose query with FlashBack */
    LZGStringPool *query_pool = lzg_sp_create(seq_len + 32u);
    if (!query_pool) return LZG_LOG_EPS;

    LZGFlashbackTokens tokens;
    LZGError err = lzg_flashback_decompose(seq, seq_len, query_pool, &tokens);
    if (err != LZG_OK || tokens.count == 0) {
        lzg_sp_destroy(query_pool);
        return LZG_LOG_EPS;
    }

    /* Build query node map if needed */
    LZGGraph *gm = (LZGGraph *)g;
    if (!gm->query_node_map) {
        LZGHashMap *map = lzg_hm_create(g->n_nodes * 2);
        if (!map) { lzg_sp_destroy(query_pool); return LZG_LOG_EPS; }
        for (uint32_t i = 0; i < g->n_nodes; i++) {
            uint64_t key = ((uint64_t)g->node_sp_id[i] << 32) | (uint64_t)g->node_pos[i];
            lzg_hm_put(map, key, (uint64_t)i);
        }
        gm->query_node_map = map;
    }

    double log_p = 0.0;
    uint32_t prev_nid = UINT32_MAX;

    for (uint32_t t = 0; t < tokens.count; t++) {
        /* Look up token in graph's string pool */
        const char *tok_str = lzg_sp_get(query_pool, tokens.sp_ids[t]);
        uint32_t graph_sp_id = lzg_sp_find(g->pool, tok_str);
        if (graph_sp_id == LZG_SP_NOT_FOUND) {
            log_p = LZG_LOG_EPS;
            break;
        }

        /* Find node index */
        uint64_t key = ((uint64_t)graph_sp_id << 32) | (uint64_t)UINT32_MAX;
        uint64_t *slot = lzg_hm_get(gm->query_node_map, key);
        if (!slot) {
            log_p = LZG_LOG_EPS;
            break;
        }
        uint32_t nid = (uint32_t)*slot;

        if (t == 0) {
            if (nid != g->root_node) { log_p = LZG_LOG_EPS; break; }
            prev_nid = nid;
            continue;
        }

        /* Find edge from prev_nid to nid */
        uint32_t e_start = g->row_offsets[prev_nid];
        uint32_t e_end   = g->row_offsets[prev_nid + 1];
        double edge_w = 0.0;
        for (uint32_t e = e_start; e < e_end; e++) {
            if (g->col_indices[e] == nid) {
                edge_w = g->edge_weights[e];
                break;
            }
        }

        if (edge_w < LZG_EPS) { log_p = LZG_LOG_EPS; break; }
        log_p += log(edge_w);
        prev_nid = nid;
    }

    /* Verify walk ends at a sink */
    if (log_p > LZG_LOG_EPS && tokens.count > 0 && prev_nid < g->n_nodes) {
        if (g->node_is_sink && !g->node_is_sink[prev_nid])
            log_p = LZG_LOG_EPS;
    }

    lzg_sp_destroy(query_pool);
    return log_p;
}

LZGError lzg_flashback_pgen_batch(const LZGGraph *g,
                                  const char **sequences,
                                  uint32_t n, double *out) {
    if (!g || !sequences || !out) return LZG_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < n; i++)
        out[i] = lzg_flashback_pgen(g, sequences[i],
                                    (uint32_t)strlen(sequences[i]));
    return LZG_OK;
}
