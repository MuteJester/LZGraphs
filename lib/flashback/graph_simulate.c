/**
 * @file flashback_simulate.c
 * @brief Markov simulation and walk probability for FlashBack graphs.
 *
 * No LZ constraints — every outgoing edge is always valid.
 * Sequences are reconstructed via lzg_flashback_reverse().
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lzgraph/flashback.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/rng.h"

#define LZG_FLASHBACK_MAX_WALK_LEN 4096

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

        while (walk_tokens.count < LZG_FLASHBACK_MAX_WALK_LEN) {
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

/* ── FlashBack Anomaly Score (FBAS) ────────────────────────────── */

/* Helper: look up a query token in the graph, returning its node id or
 * UINT32_MAX if the token is not present in the graph at all. */
static uint32_t fbas_lookup_token(const LZGGraph *g, LZGStringPool *qp,
                                  uint32_t token_sp_id) {
    const char *tok_str = lzg_sp_get(qp, token_sp_id);
    uint32_t graph_sp_id = lzg_sp_find(g->pool, tok_str);
    if (graph_sp_id == LZG_SP_NOT_FOUND) return UINT32_MAX;
    uint64_t key = ((uint64_t)graph_sp_id << 32) | (uint64_t)UINT32_MAX;
    uint64_t *slot = lzg_hm_get(((LZGGraph *)g)->query_node_map, key);
    return slot ? (uint32_t)*slot : UINT32_MAX;
}

/* Per-node Shannon entropy of outgoing edge weights, lazily computed and
 * cached in g->node_entropy. Sinks (no outgoing edges) and zero-weight
 * nodes return 0.0.
 *
 * Entropy is computed over RENORMALIZED edge probabilities (in case the
 * stored edge weights don't sum exactly to 1 due to filtering), matching
 * the Python _node_entropy reference implementation. */
double lzg_flashback_node_entropy(const LZGGraph *g, uint32_t node) {
    if (!g || node >= g->n_nodes) return 0.0;
    LZGGraph *gm = (LZGGraph *)g;  /* cast away const for lazy cache fill */

    /* Lazy allocation of the entropy table (one-time O(|E|) cost). */
    if (!gm->node_entropy) {
        double *ent = calloc(gm->n_nodes ? gm->n_nodes : 1, sizeof(double));
        if (!ent) return 0.0;
        for (uint32_t v = 0; v < gm->n_nodes; v++) {
            uint32_t e_start = gm->row_offsets[v];
            uint32_t e_end   = gm->row_offsets[v + 1];
            if (e_end <= e_start) { ent[v] = 0.0; continue; }
            double sum = 0.0;
            for (uint32_t e = e_start; e < e_end; e++) sum += gm->edge_weights[e];
            if (sum <= 0.0) { ent[v] = 0.0; continue; }
            double h = 0.0;
            for (uint32_t e = e_start; e < e_end; e++) {
                double w = gm->edge_weights[e];
                if (w <= 0.0) continue;
                double p = w / sum;
                h -= p * log(p);
            }
            ent[v] = h;
        }
        gm->node_entropy = ent;
    }
    return gm->node_entropy[node];
}

LZGError lzg_flashback_fbas(const LZGGraph *g,
                            const char *seq, uint32_t seq_len,
                            LZGFbasResult *out) {
    if (!g || !seq || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback fbas: NULL argument");

    /* Default-zero result for any early exit path. */
    out->fbas             = 0.0;
    out->log_pgen         = LZG_LOG_EPS;
    out->worst_excess     = 0.0;
    out->n_missing_tokens = 0;
    out->n_missing_edges  = 0;

    if (seq_len == 0) return LZG_OK;
    if (g->root_node >= g->n_nodes) return LZG_OK;

    /* Decompose query with FlashBack. */
    LZGStringPool *query_pool = lzg_sp_create(seq_len + 32u);
    if (!query_pool) return LZG_FAIL(LZG_ERR_ALLOC, "fbas: query pool alloc");

    LZGFlashbackTokens tokens;
    LZGError err = lzg_flashback_decompose(seq, seq_len, query_pool, &tokens);
    if (err != LZG_OK || tokens.count == 0) {
        lzg_sp_destroy(query_pool);
        return LZG_OK;
    }

    /* Build query node map if needed (same lazy cache as fb_pgen). */
    LZGGraph *gm = (LZGGraph *)g;
    if (!gm->query_node_map) {
        LZGHashMap *map = lzg_hm_create(g->n_nodes * 2);
        if (!map) { lzg_sp_destroy(query_pool); return LZG_FAIL(LZG_ERR_ALLOC, "fbas: hm"); }
        for (uint32_t i = 0; i < g->n_nodes; i++) {
            uint64_t key = ((uint64_t)g->node_sp_id[i] << 32) | (uint64_t)g->node_pos[i];
            lzg_hm_put(map, key, (uint64_t)i);
        }
        gm->query_node_map = map;
    }

    /* log_p mirrors lzg_flashback_pgen semantics: a single broken
     * transition or non-sink terminus forces log_pgen = LZG_LOG_EPS. */
    double   log_p        = 0.0;
    bool     log_p_broken = false;
    double   max_weighted = -1e300;
    double   sum_weighted = 0.0;
    uint32_t n_weighted   = 0;
    double   worst_excess = 0.0;

    uint32_t max_depth = tokens.count > 1 ? tokens.count - 1 : 0;
    uint32_t last_nid  = UINT32_MAX;

    /* Walk every transition (src=tokens[i] -> dst=tokens[i+1]) with
     * FRESH per-iteration source lookup. This matches the Python
     * reference, which calls _get_successors(graph, src) each iteration
     * rather than threading a single "current node" through the loop.
     * The practical effect: after a missing source token, the next
     * transition (with a known source) recovers normal classification
     * instead of being chained as another missing-token transition. */
    for (uint32_t i = 0; i + 1 < tokens.count; i++) {
        double depth_weight = (max_depth > 1)
                              ? (1.0 - 0.6 * ((double)i / (double)(max_depth - 1)))
                              : 1.0;

        uint32_t src_nid = fbas_lookup_token(g, query_pool, tokens.sp_ids[i]);
        uint32_t dst_nid = fbas_lookup_token(g, query_pool, tokens.sp_ids[i + 1]);

        double excess;
        if (src_nid == UINT32_MAX) {
            /* Source token unknown to the graph. */
            excess = LZG_FBAS_MISSING_TOKEN_PENALTY;
            out->n_missing_tokens++;
            log_p_broken = true;
        } else {
            /* Source known; look up edge src_nid -> dst_nid. */
            double edge_w = 0.0;
            bool found = false;
            if (dst_nid != UINT32_MAX) {
                uint32_t e_start = g->row_offsets[src_nid];
                uint32_t e_end   = g->row_offsets[src_nid + 1];
                for (uint32_t e = e_start; e < e_end; e++) {
                    if (g->col_indices[e] == dst_nid) {
                        edge_w = g->edge_weights[e];
                        found = true;
                        break;
                    }
                }
            }

            if (found && edge_w > 0.0) {
                double lp = log(edge_w < LZG_EPS ? LZG_EPS : edge_w);
                double ent = lzg_flashback_node_entropy(g, src_nid);
                double raw = -lp - ent;
                excess = raw > 0.0 ? raw : 0.0;
                if (!log_p_broken) log_p += lp;
            } else {
                /* dst either unknown or not a successor of src. */
                excess = LZG_FBAS_MISSING_EDGE_PENALTY;
                out->n_missing_edges++;
                /* Python mirrors: if dst itself is not a graph node,
                 * increment n_missing_tokens too. */
                if (dst_nid == UINT32_MAX) out->n_missing_tokens++;
                log_p_broken = true;
            }
        }

        if (excess > worst_excess) worst_excess = excess;
        double w_excess = depth_weight * excess;
        if (w_excess > max_weighted) max_weighted = w_excess;
        sum_weighted += w_excess;
        n_weighted++;

        last_nid = dst_nid;  /* used for the sink check below */
    }

    if (n_weighted > 0) {
        double mean_weighted = sum_weighted / (double)n_weighted;
        if (max_weighted < 0.0) max_weighted = 0.0;
        out->fbas = max_weighted + mean_weighted;
    } else {
        out->fbas = 0.0;
    }
    out->worst_excess = worst_excess;

    /* Match lzg_flashback_pgen final-state check: walk must end at a sink
     * (and no transition was broken). */
    if (log_p_broken) {
        out->log_pgen = LZG_LOG_EPS;
    } else if (last_nid < g->n_nodes && g->node_is_sink && !g->node_is_sink[last_nid]) {
        out->log_pgen = LZG_LOG_EPS;
    } else {
        out->log_pgen = log_p;
    }

    lzg_sp_destroy(query_pool);
    return LZG_OK;
}

LZGError lzg_flashback_fbas_batch(const LZGGraph *g,
                                  const char **sequences,
                                  uint32_t n,
                                  LZGFbasResult *out) {
    if (!g || !sequences || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback fbas batch: NULL argument");
    /* Trigger entropy precomputation once before the batch. */
    if (g->n_nodes > 0) lzg_flashback_node_entropy(g, 0);
    for (uint32_t i = 0; i < n; i++) {
        const char *s = sequences[i];
        LZGError err = lzg_flashback_fbas(g, s, (uint32_t)strlen(s), &out[i]);
        if (err != LZG_OK) return err;
    }
    return LZG_OK;
}
