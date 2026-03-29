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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Backtracking stack frame ──────────────────────────────── */

#define MAX_WALK_DEPTH 128
#define MAX_BLACKLIST   64

typedef struct {
    uint32_t node;
    uint32_t edge_taken;
    uint8_t  sp_len;
    uint32_t blacklist[MAX_BLACKLIST];
    uint32_t n_blacklisted;
} SimFrame;

static bool is_blacklisted(const SimFrame *f, uint32_t edge) {
    for (uint32_t i = 0; i < f->n_blacklisted; i++)
        if (f->blacklist[i] == edge) return true;
    return false;
}

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
                                         LZGStringPool *pool, LZGTokens *tokens) {
    uint32_t wlen = 0;
    char wrapped_stack[512];
    bool wrapped_heap = false;
    char *wrapped = query_wrap_sentinels(seq, seq_len, &wlen,
                                         wrapped_stack, sizeof(wrapped_stack),
                                         &wrapped_heap);
    if (!wrapped) return LZG_ERR_ALLOC;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, tokens);
    if (wrapped_heap) free(wrapped);
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

    uint32_t root = g->root_node;
    LZG_DEBUG("simulate: generating %u sequences from root node %u", n, root);

    char seq_buf[1024];

    for (uint32_t seq_idx = 0; seq_idx < n; seq_idx++) {
        LZGWalkDict wd = lzg_wd_create();
        SimFrame stack[MAX_WALK_DEPTH];
        uint32_t depth = 0;

        /* Start at root (@) node */
        uint32_t current = root;
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[current]);
        uint8_t sp_len = g->node_sp_len[current];

        /* Don't include @ in the output sequence */
        uint32_t seq_pos = 0;

        /* Record @ token in walk dictionary */
        lzg_wd_record_node(&wd, g, current);

        stack[0].node = current;
        stack[0].edge_taken = UINT32_MAX;
        stack[0].sp_len = 0;  /* @ doesn't contribute to output */
        stack[0].n_blacklisted = 0;
        depth = 1;

        while (depth > 0 && depth < MAX_WALK_DEPTH) {
            SimFrame *top = &stack[depth - 1];
            current = top->node;

            /* If we reached a $-sink, we're done */
            if (is_sink_node(g, current)) {
                break;
            }

            /* Collect LZ-valid edges */
            uint32_t e_start = g->row_offsets[current];
            uint32_t e_end   = g->row_offsets[current + 1];
            uint32_t valid_edges[512];
            double   valid_wts[512];
            uint32_t n_valid = 0;
            double Z = 0.0;

            for (uint32_t e = e_start; e < e_end && n_valid < 512; e++) {
                if (is_blacklisted(top, e)) continue;
                if (!lzg_wd_edge_valid(g, e, &wd)) continue;
                valid_edges[n_valid] = e;
                valid_wts[n_valid]   = g->edge_weights[e];
                Z += g->edge_weights[e];
                n_valid++;
            }

            if (n_valid == 0) {
                /* Dead end — backtrack */
                if (depth <= 1) break;
                seq_pos -= top->sp_len;
                uint32_t dead_edge = top->edge_taken;
                depth--;

                /* Rebuild dictionary from stack */
                lzg_wd_reset(&wd);
                lzg_wd_record_node(&wd, g, stack[0].node);
                for (uint32_t d = 1; d < depth; d++)
                    lzg_wd_record_edge(&wd, g, stack[d].edge_taken);

                SimFrame *parent = &stack[depth - 1];
                if (parent->n_blacklisted < MAX_BLACKLIST)
                    parent->blacklist[parent->n_blacklisted++] = dead_edge;
                continue;
            }

            /* Sample from valid edges */
            double r = lzg_rng_double(rng) * Z;
            double cum = 0.0;
            uint32_t chosen = n_valid - 1;
            for (uint32_t k = 0; k < n_valid; k++) {
                cum += valid_wts[k];
                if (r < cum) { chosen = k; break; }
            }

            uint32_t next_node = g->col_indices[valid_edges[chosen]];
            sp = lzg_sp_get(g->pool, g->node_sp_id[next_node]);
            sp_len = g->node_sp_len[next_node];

            /* Append to sequence buffer (skip $ sentinel in output) */
            uint8_t copy_len = sp_len;
            const char *copy_src = sp;
            /* Strip $ from the last token for output */
            if (sp_len > 0 && sp[sp_len - 1] == LZG_END_SENTINEL) {
                copy_len = sp_len - 1;
            }
            /* Strip @ from the first token for output */
            if (sp_len > 0 && sp[0] == LZG_START_SENTINEL) {
                copy_src = sp + 1;
                copy_len = sp_len - 1;
            }

            if (seq_pos + copy_len < sizeof(seq_buf)) {
                memcpy(seq_buf + seq_pos, copy_src, copy_len);
                seq_pos += copy_len;
            }

            /* Record token and push frame */
            lzg_wd_record_edge(&wd, g, valid_edges[chosen]);

            if (depth < MAX_WALK_DEPTH) {
                stack[depth].node = next_node;
                stack[depth].edge_taken = valid_edges[chosen];
                stack[depth].sp_len = copy_len;
                stack[depth].n_blacklisted = 0;
                depth++;
            }
        }

        seq_buf[seq_pos] = '\0';
        out[seq_idx].sequence = strdup(seq_buf);
        out[seq_idx].seq_len  = seq_pos;
        out[seq_idx].n_tokens = depth > 0 ? depth - 1 : 0; /* exclude @ */
        out[seq_idx].log_prob = lzg_walk_log_prob(g, seq_buf, seq_pos);

        lzg_wd_destroy(&wd);
    }

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Walk log-probability                                            */
/* ═══════════════════════════════════════════════════════════════ */

double lzg_walk_log_prob(const LZGGraph *g,
                          const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return LZG_LOG_EPS;
    if (!g->topo_valid) return LZG_LOG_EPS;
    LZGGraph *gm = (LZGGraph *)g;
    if (lzg_graph_ensure_query_edge_hashes(gm) != LZG_OK) return LZG_LOG_EPS;
    if (ensure_query_node_map(gm) != LZG_OK) return LZG_LOG_EPS;

    /* Decompose sequence structurally, then resolve tokens to graph nodes. */
    LZGTokens tokens;
    LZGError err = query_decompose_sequence(seq, seq_len, (LZGStringPool *)g->pool, &tokens);
    uint32_t n_tokens = tokens.count;
    if (err != LZG_OK || n_tokens == 0) return LZG_LOG_EPS;

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
        uint64_t key = query_node_key(g, tokens.sp_ids[t], tokens.positions[t]);
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
    return log_p;
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
