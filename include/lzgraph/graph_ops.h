/**
 * @file graph_ops.h
 * @brief Graph operations: set algebra (union, intersection, difference),
 *        summary statistics.
 */
#ifndef LZGRAPH_GRAPH_OPS_H
#define LZGRAPH_GRAPH_OPS_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/* ── Graph summary ─────────────────────────────────────────── */

typedef struct {
    uint32_t n_nodes;
    uint32_t n_edges;
    uint32_t n_initial;
    uint32_t n_terminal;
    uint32_t max_out_degree;
    uint32_t max_in_degree;
    uint32_t n_isolates;       /* nodes with degree 0 */
    bool     is_dag;
} LZGGraphSummary;

LZGError lzg_graph_summary(const LZGGraph *g, LZGGraphSummary *out);

/* ── Graph union ───────────────────────────────────────────── */

/* ── Graph set operations ──────────────────────────────────── */
/* All operations require both graphs to be the same variant.   */
/* All produce a new graph with recomputed weights and live idx.*/

/**
 * Union: sum edge counts from both graphs.
 * A + B: every edge from either graph, counts summed.
 */
LZGError lzg_graph_union(const LZGGraph *a, const LZGGraph *b,
                          LZGGraph **out);

/**
 * Intersection: keep only edges present in BOTH graphs.
 * Edge count = min(count_A, count_B).
 * Nodes with no remaining edges are dropped.
 */
LZGError lzg_graph_intersection(const LZGGraph *a, const LZGGraph *b,
                                 LZGGraph **out);

/**
 * Difference: edges from A with B's contribution subtracted.
 * Edge count = max(count_A - count_B, 0).
 * Edges that reach zero are dropped.
 * Useful for removing background/population patterns.
 */
LZGError lzg_graph_difference(const LZGGraph *a, const LZGGraph *b,
                               LZGGraph **out);

/**
 * Weighted merge: alpha * A + beta * B.
 * Edge count = round(alpha * count_A + beta * count_B).
 * Generalizes union (alpha=beta=1) and scale (beta=0).
 */
LZGError lzg_graph_weighted_merge(const LZGGraph *a, const LZGGraph *b,
                                   double alpha, double beta,
                                   LZGGraph **out);

#endif /* LZGRAPH_GRAPH_OPS_H */
