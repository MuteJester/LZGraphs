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

/* ═══════════════════════════════════════════════════════════════ */
/* Simulate                                                        */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_simulate(const LZGGraph *g, uint32_t n,
                       LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph, rng, and output must not be NULL");
    if (!g->topo_valid) return LZG_FAIL(LZG_ERR_NOT_BUILT, "graph not finalized");

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

        double log_p = 0.0;

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
                lzg_wd_destroy(&wd);
                wd = lzg_wd_create();
                lzg_wd_record_node(&wd, g, stack[0].node);
                for (uint32_t d = 1; d < depth; d++)
                    lzg_wd_record_edge(&wd, g, stack[d].edge_taken);

                SimFrame *parent = &stack[depth - 1];
                if (parent->n_blacklisted < MAX_BLACKLIST)
                    parent->blacklist[parent->n_blacklisted++] = dead_edge;

                log_p = 0.0; /* will be recomputed by walk_log_prob */
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

        /* Compute exact log-probability by retracing */
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

    /* Encode sequence via LZ76 (wraps with @...$ internally) */
    uint32_t node_ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
    uint32_t n_tokens;
    LZGError err = lzg_lz76_encode(seq, seq_len, (LZGStringPool *)g->pool,
                                    g->variant, node_ids, sp_ids, &n_tokens);
    if (err != LZG_OK || n_tokens == 0) return LZG_LOG_EPS;

    /* Build label → graph node index map */
    LZGHashMap *label_map = lzg_hm_create(g->n_nodes * 2);
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *nsp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        uint32_t pos = g->node_pos[i];
        char buf[256];
        int len;
        switch (g->variant) {
            case LZG_VARIANT_AAP:
                len = snprintf(buf, sizeof(buf), "%s_%u", nsp, pos);
                break;
            case LZG_VARIANT_NDP: {
                uint32_t nsp_len = g->node_sp_len[i];
                uint32_t start_pos = pos - nsp_len;
                uint32_t frame = start_pos % 3;
                len = snprintf(buf, sizeof(buf), "%s%u_%u", nsp, frame, pos);
                break;
            }
            case LZG_VARIANT_NAIVE:
                len = snprintf(buf, sizeof(buf), "%s", nsp);
                break;
            default:
                len = snprintf(buf, sizeof(buf), "%s_%u", nsp, pos);
                break;
        }
        uint32_t lid = lzg_sp_intern_n((LZGStringPool *)g->pool, buf, (uint32_t)len);
        lzg_hm_put(label_map, (uint64_t)lid, (uint64_t)i);
    }

    /* Walk dictionary for exact LZ constraint checking */
    LZGWalkDict wd = lzg_wd_create();
    double log_p = 0.0;

    /*
     * With sentinels, the token sequence is: [@, C, A, S, SL, ..., T$]
     * Token 0 is always @ (the root). No probability factor for it.
     * Tokens 1..n-1 each contribute: log(w(edge) / Z_valid)
     * The last token contains $ (sink). No stop probability.
     */
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint64_t *gi = lzg_hm_get(label_map, (uint64_t)node_ids[t]);
        if (!gi) { log_p = LZG_LOG_EPS; break; }
        uint32_t nid = (uint32_t)*gi;

        if (t == 0) {
            /* First token is @: verify it's the root, no probability factor */
            if (nid != g->root_node) { log_p = LZG_LOG_EPS; break; }
            lzg_wd_record_node(&wd, g, nid);
            continue;
        }

        /* Previous node */
        uint64_t *prev_gi = lzg_hm_get(label_map, (uint64_t)node_ids[t - 1]);
        if (!prev_gi) { log_p = LZG_LOG_EPS; break; }
        uint32_t prev_nid = (uint32_t)*prev_gi;

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
    }

    /* Verify walk ends at a $-sink node */
    if (log_p > LZG_LOG_EPS && n_tokens > 0) {
        uint64_t *last_gi = lzg_hm_get(label_map, (uint64_t)node_ids[n_tokens - 1]);
        if (!last_gi || !is_sink_node(g, (uint32_t)*last_gi)) {
            log_p = LZG_LOG_EPS;
        }
    }

    lzg_wd_destroy(&wd);
    lzg_hm_destroy(label_map);
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
