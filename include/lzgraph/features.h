/**
 * @file features.h
 * @brief ML feature extraction from LZGraphs.
 *
 * Three strategies for extracting fixed-size numerical vectors:
 *
 * A. Reference-aligned: project query graph onto a reference graph's
 *    node space (dimension = reference nodes).
 * B. Mass profile: probability mass by sequence position via forward DP
 *    (dimension = max sequence length, ~25 for CDR3).
 * C. Statistics: scalar features from existing analytics
 *    (dimension = LZG_FEATURE_STATS_DIM = 15).
 */
#ifndef LZGRAPH_FEATURES_H
#define LZGRAPH_FEATURES_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/**
 * Strategy A: Reference-aligned feature vector.
 *
 * For each node in `ref`, look up the matching node in `query`
 * (by label) and store its empirical frequency. Missing = 0.0.
 *
 * @param ref     Reference graph (defines feature dimensions).
 * @param query   Query graph.
 * @param out     Output vector (caller allocates, size ref->n_nodes).
 * @param out_dim Output: ref->n_nodes.
 */
LZGError lzg_feature_aligned(const LZGGraph *ref, const LZGGraph *query,
                              double *out, uint32_t *out_dim);

/**
 * Strategy B: Position-projected mass profile.
 *
 * Runs the LZ-constrained forward DP. Absorbed mass at each terminal
 * is attributed to the terminal's sequence position. The result is
 * a probability distribution over positions.
 *
 * @param g       The graph.
 * @param out     Output vector (caller allocates, size max_pos + 1).
 * @param max_pos Maximum position to track (e.g., 30).
 */
LZGError lzg_feature_mass_profile(const LZGGraph *g,
                                   double *out, uint32_t max_pos);

/**
 * Strategy C: Graph statistics vector (15 features).
 */
#define LZG_FEATURE_STATS_DIM 15

LZGError lzg_feature_stats(const LZGGraph *g, double *out);

#endif /* LZGRAPH_FEATURES_H */
