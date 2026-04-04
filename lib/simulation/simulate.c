/**
 * @file simulate.c
 * @brief LZ76-constrained simulation and walk probability.
 *
 * With @ / $ sentinels, the model is uniform:
 * - Every walk starts at the @ root node
 * - Every walk ends at a $-suffixed sink node
 * - P(walk) = Π P(edge_t | LZ-valid edges)
 * - No stop probability, no initial distribution — just edge products
 */
#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif
#include "lzgraph/simulate.h"
#include "lzgraph/walk_dict.h"
#include "lzgraph/lz76.h"
#include "lzgraph/hash_map.h"
#include "exact_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Check if a node is a $-sink (its subpattern contains '$') */
static bool is_sink_node(const LZGGraph *g, uint32_t node) {
    return g->node_is_sink ? g->node_is_sink[node] : false;
}

static inline uint64_t query_node_key(const LZGGraph *g,
                                      uint32_t sp_id, uint32_t pos) {
    if (g->variant == LZG_VARIANT_NAIVE)
        return ((uint64_t)sp_id << 32) | (uint64_t)UINT32_MAX;
    return ((uint64_t)sp_id << 32) | (uint64_t)pos;
}

static LZGError ensure_query_node_map(LZGGraph *g) {
    if (g->query_node_map) return LZG_OK;

    LZGHashMap *map = lzg_hm_create(g->n_nodes * 2);
    if (!map) return LZG_FAIL(LZG_ERR_ALLOC, "failed to allocate query node map");

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint64_t key = query_node_key(g, g->node_sp_id[i], g->node_pos[i]);
        lzg_hm_put(map, key, (uint64_t)i);
    }
    g->query_node_map = map;
    return LZG_OK;
}

static char *query_wrap_sentinels(const char *str, uint32_t len, uint32_t *out_len,
                                  char *stack_buf, size_t stack_cap, bool *used_heap) {
    *out_len = len + 2;
    size_t need = (size_t)(*out_len) + 1u;
    char *wrapped = NULL;
    if (need <= stack_cap) {
        wrapped = stack_buf;
        *used_heap = false;
    } else {
        wrapped = malloc(need);
        *used_heap = true;
    }
    if (!wrapped) return NULL;
    wrapped[0] = LZG_START_SENTINEL;
    memcpy(wrapped + 1, str, len);
    wrapped[len + 1] = LZG_END_SENTINEL;
    wrapped[len + 2] = '\0';
    return wrapped;
}

static LZGError query_decompose_sequence(const char *seq, uint32_t seq_len,
                                         LZGTokens *tokens,
                                         LZGStringPool **out_pool) {
    uint32_t wlen = 0;
    char wrapped_stack[512];
    bool wrapped_heap = false;
    LZGStringPool *pool = lzg_sp_create(seq_len + 16u);
    char *wrapped = query_wrap_sentinels(seq, seq_len, &wlen,
                                         wrapped_stack, sizeof(wrapped_stack),
                                         &wrapped_heap);
    if (!pool) return LZG_ERR_ALLOC;
    if (!wrapped) {
        lzg_sp_destroy(pool);
        return LZG_ERR_ALLOC;
    }
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, tokens);
    if (wrapped_heap) free(wrapped);
    if (err != LZG_OK) {
        lzg_sp_destroy(pool);
        return err;
    }
    *out_pool = pool;
    return err;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Simulate                                                        */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_simulate(const LZGGraph *g, uint32_t n,
                       LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph, rng, and output must not be NULL");
    if (!g->topo_valid) return LZG_FAIL(LZG_ERR_NOT_BUILT, "graph not finalized");
    if (lzg_graph_ensure_query_edge_hashes((LZGGraph *)g) != LZG_OK)
        return LZG_FAIL(LZG_ERR_ALLOC, "failed to initialize query edge hash cache");

    LZG_DEBUG("simulate: generating %u sequences from root node %u", n, g->root_node);

    for (uint32_t seq_idx = 0; seq_idx < n; seq_idx++) {
        LZGExactSample sample;
        LZGError err = lzg_exact_model_sample(g, rng, &sample);

        if (err != LZG_OK) return err;

        out[seq_idx].sequence = strdup(sample.sequence);
        out[seq_idx].seq_len = sample.seq_len;
        out[seq_idx].n_tokens = sample.n_tokens;
        out[seq_idx].log_prob = sample.log_prob;
    }

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Walk log-probability                                            */
/* ═══════════════════════════════════════════════════════════════ */

static double lzg_walk_log_prob_raw(const LZGGraph *g,
                                    const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return LZG_LOG_EPS;
    if (!g->topo_valid) return LZG_LOG_EPS;
    LZGGraph *gm = (LZGGraph *)g;
    if (lzg_graph_ensure_query_edge_hashes(gm) != LZG_OK) return LZG_LOG_EPS;
    if (ensure_query_node_map(gm) != LZG_OK) return LZG_LOG_EPS;

    /* Decompose sequence structurally, then resolve tokens to graph nodes. */
    LZGTokens tokens;
    LZGStringPool *query_pool = NULL;
    LZGError err = query_decompose_sequence(seq, seq_len, &tokens, &query_pool);
    uint32_t n_tokens = tokens.count;
    if (err != LZG_OK || n_tokens == 0) {
        if (query_pool) lzg_sp_destroy(query_pool);
        return LZG_LOG_EPS;
    }

    /* Walk dictionary for exact LZ constraint checking */
    LZGWalkDict wd = lzg_wd_create();
    double log_p = 0.0;
    uint32_t prev_nid = UINT32_MAX;

    /*
     * With sentinels, the token sequence is: [@, C, A, S, SL, ..., T$]
     * Token 0 is always @ (the root). No probability factor for it.
     * Tokens 1..n-1 each contribute: log(w(edge) / Z_valid)
     * The last token contains $ (sink). No stop probability.
     */
    for (uint32_t t = 0; t < n_tokens; t++) {
        const char *token = lzg_sp_get(query_pool, tokens.sp_ids[t]);
        uint32_t token_sp_id = lzg_sp_find(g->pool, token);
        uint64_t key;
        if (token_sp_id == LZG_SP_NOT_FOUND) { log_p = LZG_LOG_EPS; break; }
        key = query_node_key(g, token_sp_id, tokens.positions[t]);
        uint64_t *gi = lzg_hm_get(g->query_node_map, key);
        if (!gi) { log_p = LZG_LOG_EPS; break; }
        uint32_t nid = (uint32_t)*gi;

        if (t == 0) {
            /* First token is @: verify it's the root, no probability factor */
            if (nid != g->root_node) { log_p = LZG_LOG_EPS; break; }
            lzg_wd_record_node(&wd, g, nid);
            prev_nid = nid;
            continue;
        }

        /* Compute Z over LZ-valid edges from previous node */
        uint32_t e_start = g->row_offsets[prev_nid];
        uint32_t e_end   = g->row_offsets[prev_nid + 1];
        double Z = 0.0, edge_w = 0.0;

        for (uint32_t e = e_start; e < e_end; e++) {
            if (!lzg_wd_edge_valid(g, e, &wd)) continue;
            Z += g->edge_weights[e];
            if (g->col_indices[e] == nid) edge_w = g->edge_weights[e];
        }

        if (edge_w < LZG_EPS || Z < LZG_EPS) { log_p = LZG_LOG_EPS; break; }
        log_p += log(edge_w / Z);

        /* Record token */
        lzg_wd_record_node(&wd, g, nid);
        prev_nid = nid;
    }

    /* Verify walk ends at a $-sink node */
    if (log_p > LZG_LOG_EPS && n_tokens > 0) {
        if (!is_sink_node(g, prev_nid)) {
            log_p = LZG_LOG_EPS;
        }
    }

    lzg_wd_destroy(&wd);
    lzg_sp_destroy(query_pool);
    return log_p;
}

double lzg_walk_log_prob(const LZGGraph *g,
                          const char *seq, uint32_t seq_len) {
    double raw_log_prob;
    double root_absorption;

    raw_log_prob = lzg_walk_log_prob_raw(g, seq, seq_len);
    if (raw_log_prob <= LZG_LOG_EPS + 1.0)
        return LZG_LOG_EPS;

    if (lzg_exact_model_ensure((LZGGraph *)g) != LZG_OK)
        return LZG_LOG_EPS;

    root_absorption = lzg_exact_model_root_absorption(g);
    if (root_absorption <= LZG_EPS)
        return LZG_LOG_EPS;

    return raw_log_prob - log(root_absorption);
}

/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_walk_log_prob_batch(const LZGGraph *g,
                                  const char **sequences,
                                  uint32_t n, double *out) {
    if (!g || !sequences || !out) return LZG_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < n; i++)
        out[i] = lzg_walk_log_prob(g, sequences[i],
                                    (uint32_t)strlen(sequences[i]));
    return LZG_OK;
}
