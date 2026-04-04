/**
 * @file graph_ops.c
 * @brief Public graph-ops facade over internal graph summary and set-op modules.
 */
#include "lzgraph/graph_ops.h"
#include "graph_ops_internal.h"

LZGError lzg_graph_summary(const LZGGraph *g, LZGGraphSummary *out) {
    return lzg_graph_summary_impl(g, out);
}

LZGError lzg_graph_union(const LZGGraph *a, const LZGGraph *b,
                         LZGGraph **out) {
    return lzg_graph_union_impl(a, b, out);
}

LZGError lzg_graph_intersection(const LZGGraph *a, const LZGGraph *b,
                                LZGGraph **out) {
    return lzg_graph_intersection_impl(a, b, out);
}

LZGError lzg_graph_difference(const LZGGraph *a, const LZGGraph *b,
                              LZGGraph **out) {
    return lzg_graph_difference_impl(a, b, out);
}

LZGError lzg_graph_weighted_merge(const LZGGraph *a, const LZGGraph *b,
                                  double alpha, double beta,
                                  LZGGraph **out) {
    return lzg_graph_weighted_merge_impl(a, b, alpha, beta, out);
}
