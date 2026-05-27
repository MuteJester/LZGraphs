/**
 * @file top_k.c
 * @brief Find the K most (or least) probable walks in a FlashBack DAG.
 *
 * Algorithm: Node-based K-best paths using topological order.
 * For each node in topo order, we maintain a sorted list of the K best
 * partial paths reaching that node. At sink nodes, complete paths are
 * collected into a global result set. Uses predecessor pointers for
 * memory-efficient path reconstruction.
 *
 * Complexity: O(V * K * max_out_degree) time, O(V * K) space.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "lzgraph/graph.h"
#include "lzgraph/flashback.h"
#include "lzgraph/simulate.h"

/* ── Per-node K-best entry ───────────────────────────────────── */

typedef struct {
    double   log_prob;    /* cumulative log probability         */
    uint32_t pred_node;   /* predecessor node (UINT32_MAX=root) */
    uint32_t pred_k;      /* which k-index at predecessor       */
} KEntry;

/* Compare for most-probable (descending log_prob) */
static int cmp_desc(const void *a, const void *b) {
    double da = ((const KEntry *)a)->log_prob;
    double db = ((const KEntry *)b)->log_prob;
    return (db > da) - (db < da);
}

/* Compare for least-probable (ascending log_prob) */
static int cmp_asc(const void *a, const void *b) {
    double da = ((const KEntry *)a)->log_prob;
    double db = ((const KEntry *)b)->log_prob;
    return (da > db) - (da < db);
}

/* ── Reconstruct walk from predecessor chain ─────────────────── */

static uint32_t reconstruct_walk(const KEntry *best, uint32_t k,
                                 const LZGGraph *g,
                                 uint32_t sink_node, uint32_t sink_k,
                                 uint32_t *out_node_ids, uint32_t cap) {
    /* Trace back through predecessors */
    uint32_t stack[LZG_FLASHBACK_MAX_TOKENS];
    uint32_t depth = 0;

    uint32_t cur_node = sink_node;
    uint32_t cur_k    = sink_k;

    while (cur_node != UINT32_MAX && depth < LZG_FLASHBACK_MAX_TOKENS) {
        stack[depth++] = cur_node;
        const KEntry *e = &best[cur_node * k + cur_k];
        uint32_t prev_node = e->pred_node;
        cur_k = e->pred_k;
        cur_node = prev_node;
    }

    /* Reverse into output */
    uint32_t n = 0;
    for (uint32_t i = depth; i > 0 && n < cap; i--)
        out_node_ids[n++] = stack[i - 1];

    return n;
}

/* ── Main entry point ────────────────────────────────────────── */

LZGError lzg_flashback_top_k_walks(const LZGGraph *g, uint32_t k, bool most_probable,
                         LZGSimResult *out, uint32_t *out_count) {
    if (!g || !out || !out_count || k == 0)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "top_k_walks: invalid arguments");

    if (!g->topo_valid)
        return LZG_FAIL(LZG_ERR_NOT_BUILT, "top_k_walks: no topo order");

    const uint32_t n = g->n_nodes;
    int (*cmp_fn)(const void *, const void *) = most_probable ? cmp_desc : cmp_asc;

    /* Allocate per-node K-best arrays */
    KEntry *best = (KEntry *)calloc((size_t)n * k, sizeof(KEntry));
    uint32_t *counts = (uint32_t *)calloc(n, sizeof(uint32_t));
    if (!best || !counts) {
        free(best); free(counts);
        return LZG_FAIL(LZG_ERR_ALLOC, "top_k_walks: out of memory");
    }

    /* Sentinel: mark all entries as invalid */
    for (size_t i = 0; i < (size_t)n * k; i++) {
        best[i].log_prob = most_probable ? -DBL_MAX : DBL_MAX;
        best[i].pred_node = UINT32_MAX;
        best[i].pred_k = 0;
    }

    /* Initialize root */
    uint32_t root = g->root_node;
    best[root * k].log_prob  = 0.0;
    best[root * k].pred_node = UINT32_MAX;
    best[root * k].pred_k    = 0;
    counts[root] = 1;

    /* Forward pass in topological order */
    for (uint32_t ti = 0; ti < n; ti++) {
        uint32_t u = g->topo_order[ti];
        if (counts[u] == 0) continue;

        uint32_t e_start = g->row_offsets[u];
        uint32_t e_end   = g->row_offsets[u + 1];

        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = g->col_indices[e];
            double   w = g->edge_weights[e];
            if (w < 1e-300) continue;
            double log_w = log(w);

            for (uint32_t ki = 0; ki < counts[u]; ki++) {
                double new_lp = best[u * k + ki].log_prob + log_w;

                /* Check if this is better than v's worst entry */
                uint32_t cv = counts[v];
                if (cv < k) {
                    /* Room available — insert */
                    KEntry *slot = &best[v * k + cv];
                    slot->log_prob  = new_lp;
                    slot->pred_node = u;
                    slot->pred_k    = ki;
                    counts[v] = cv + 1;
                    /* Re-sort */
                    qsort(&best[v * k], counts[v], sizeof(KEntry), cmp_fn);
                } else {
                    /* Full — compare to worst (last after sort) */
                    KEntry *worst = &best[v * k + k - 1];
                    bool better = most_probable
                                  ? (new_lp > worst->log_prob)
                                  : (new_lp < worst->log_prob);
                    if (better) {
                        worst->log_prob  = new_lp;
                        worst->pred_node = u;
                        worst->pred_k    = ki;
                        qsort(&best[v * k], k, sizeof(KEntry), cmp_fn);
                    }
                }
            }
        }
    }

    /* Collect complete paths at sink nodes into a temporary buffer */
    uint32_t result_cap = k * 4; /* more than k, will trim later */
    if (result_cap < k) result_cap = k;
    LZGSimResult *candidates = (LZGSimResult *)calloc(result_cap, sizeof(LZGSimResult));
    double *cand_lp = (double *)malloc(result_cap * sizeof(double));
    uint32_t n_cand = 0;

    for (uint32_t i = 0; i < n; i++) {
        if (!g->node_is_sink || !g->node_is_sink[i]) continue;
        for (uint32_t ki = 0; ki < counts[i]; ki++) {
            if (best[i * k + ki].log_prob <= -DBL_MAX + 1.0) continue;
            if (n_cand >= result_cap) {
                result_cap *= 2;
                candidates = realloc(candidates, result_cap * sizeof(LZGSimResult));
                cand_lp = realloc(cand_lp, result_cap * sizeof(double));
            }
            cand_lp[n_cand] = best[i * k + ki].log_prob;

            /* Reconstruct walk */
            uint32_t walk_nodes[LZG_FLASHBACK_MAX_TOKENS];
            uint32_t walk_len = reconstruct_walk(best, k, g, i, ki,
                                                 walk_nodes,
                                                 LZG_FLASHBACK_MAX_TOKENS);

            /* Convert node IDs to token IDs for reversal */
            LZGFlashbackTokens tokens;
            tokens.count = 0;
            for (uint32_t wi = 0; wi < walk_len && tokens.count < LZG_FLASHBACK_MAX_TOKENS; wi++)
                tokens.sp_ids[tokens.count++] = g->node_sp_id[walk_nodes[wi]];

            /* Reverse tokens to CDR3 string */
            char seq_buf[2048];
            uint32_t seq_len = 0;
            LZGError err = lzg_flashback_reverse(g->pool, &tokens,
                                                 seq_buf, sizeof(seq_buf),
                                                 &seq_len);
            if (err != LZG_OK) {
                seq_buf[0] = '\0';
                seq_len = 0;
            }

            candidates[n_cand].sequence = strdup(seq_buf);
            candidates[n_cand].seq_len  = seq_len;
            candidates[n_cand].n_tokens = walk_len;
            candidates[n_cand].log_prob = best[i * k + ki].log_prob;
            n_cand++;
        }
    }

    /* Sort candidates globally and take top K */
    /* Sort by log_prob (use indices to avoid moving structs) */
    uint32_t *order = (uint32_t *)malloc(n_cand * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_cand; i++) order[i] = i;

    /* Simple selection: sort indices by log_prob */
    for (uint32_t i = 0; i < n_cand && i < k; i++) {
        for (uint32_t j = i + 1; j < n_cand; j++) {
            bool swap = most_probable
                        ? (cand_lp[order[j]] > cand_lp[order[i]])
                        : (cand_lp[order[j]] < cand_lp[order[i]]);
            if (swap) {
                uint32_t tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    /* Copy top K to output */
    uint32_t final_k = (n_cand < k) ? n_cand : k;
    for (uint32_t i = 0; i < final_k; i++) {
        out[i] = candidates[order[i]];
        candidates[order[i]].sequence = NULL; /* ownership transferred */
    }
    *out_count = final_k;

    /* Free non-transferred candidates */
    for (uint32_t i = 0; i < n_cand; i++) {
        if (candidates[i].sequence)
            free(candidates[i].sequence);
    }
    free(candidates);
    free(cand_lp);
    free(order);
    free(best);
    free(counts);

    return LZG_OK;
}
