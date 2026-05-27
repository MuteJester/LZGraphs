#ifndef LZGRAPH_GRAPH_OPS_INTERNAL_H
#define LZGRAPH_GRAPH_OPS_INTERNAL_H

#include "lzgraph/graph_ops.h"

LZGError lzg_graph_summary_impl(const LZGGraph *g, LZGGraphSummary *out);

LZGError lzg_graph_union_impl(const LZGGraph *a, const LZGGraph *b,
                              LZGGraph **out);
LZGError lzg_graph_intersection_impl(const LZGGraph *a, const LZGGraph *b,
                                     LZGGraph **out);
LZGError lzg_graph_difference_impl(const LZGGraph *a, const LZGGraph *b,
                                   LZGGraph **out);
LZGError lzg_graph_weighted_merge_impl(const LZGGraph *a, const LZGGraph *b,
                                       double alpha, double beta,
                                       LZGGraph **out);

#endif /* LZGRAPH_GRAPH_OPS_INTERNAL_H */
