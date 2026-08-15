/**
 * @file graph_analytics.c
 * @brief Exact DP analytics for NaiveGraphs.
 *
 * Every quantity here is a forward dynamic program over the CSR
 * adjacency: seed at the root, propagate along topological order,
 * absorb at the sinks. Nothing in that recursion depends on what a node
 * *means*, so the implementations in lib/flashback/graph_analytics.c are
 * already correct for this variant and are forwarded to rather than
 * copied. The `lzg_flashback_` prefix on them is historical naming, not
 * a coupling to the FlashBack decomposition — that file includes no
 * tokenizer header and parses no labels.
 *
 * These wrappers exist so callers can stay within one variant's API, and
 * so that if the shared DP ever moves to lib/graph_core there is exactly
 * one place per variant to repoint.
 *
 * Correctness here rests on two properties the naive encoding
 * guarantees: the index strictly increases along a walk, so the graph is
 * acyclic; and the walk/sequence map is a bijection, so a path count is
 * a sequence count and the DP's per-path products are per-sequence
 * probabilities.
 */
#include "lzgraph/flashback_graph.h"
#include "lzgraph/naive_graph.h"

LZGError lzg_naive_path_count(const LZGGraph *g, double *out) {
    return lzg_flashback_path_count(g, out);
}

LZGError lzg_naive_path_count_exact(const LZGGraph *g,
                                    uint32_t **limbs_out,
                                    uint32_t *n_limbs_out) {
    return lzg_flashback_path_count_exact(g, limbs_out, n_limbs_out);
}

LZGError lzg_naive_effective_diversity(const LZGGraph *g,
                                       LZGEffectiveDiversity *out) {
    return lzg_flashback_effective_diversity(g, out);
}

LZGError lzg_naive_power_sum(const LZGGraph *g, double alpha, double *out) {
    return lzg_flashback_power_sum(g, alpha, out);
}

LZGError lzg_naive_hill_number(const LZGGraph *g, double alpha, double *out) {
    return lzg_flashback_hill_number(g, alpha, out);
}

LZGError lzg_naive_hill_numbers(const LZGGraph *g, const double *orders,
                                uint32_t n, double *out) {
    return lzg_flashback_hill_numbers(g, orders, n, out);
}

LZGError lzg_naive_dynamic_range(const LZGGraph *g, LZGDynamicRange *out) {
    return lzg_flashback_dynamic_range(g, out);
}

LZGError lzg_naive_pgen_diagnostics(const LZGGraph *g, double atol,
                                    LZGPgenDiagnostics *out) {
    return lzg_flashback_pgen_diagnostics(g, atol, out);
}
