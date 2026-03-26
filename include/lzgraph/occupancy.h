/**
 * @file occupancy.h
 * @brief Occupancy model: predicted richness, overlap, and richness curves.
 *
 * Under the Poisson occupancy model, each sequence s is observed at
 * least once with probability q(s) = 1 - exp(-d · π(s)), where d is
 * the effective sampling depth and π(s) is the generation probability.
 *
 * F(d) = Σ_s q(s) is computed via Taylor expansion:
 *   F(d) = Σ_{k=1}^K (-1)^{k+1} d^k/k! · M(k)
 * where M(k) = Σ_s π(s)^k is computed exactly by the LZ-constrained DP.
 *
 * G(d_i, d_j) = F(d_i) + F(d_j) - F(d_i + d_j) (Poisson identity).
 */
#ifndef LZGRAPH_OCCUPANCY_H
#define LZGRAPH_OCCUPANCY_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/**
 * Predicted richness F(d) for a single depth value.
 *
 * @param g     The graph (must be finalized).
 * @param d     Effective sampling depth.
 * @param out   Output: expected number of distinct sequences.
 */
LZGError lzg_predicted_richness(const LZGGraph *g, double d, double *out);

/**
 * Predicted overlap between two samples (Poisson model).
 * G(d_i, d_j) = F(d_i) + F(d_j) - F(d_i + d_j).
 *
 * @param g     The graph.
 * @param d_i   Effective depth of sample i.
 * @param d_j   Effective depth of sample j.
 * @param out   Output: expected shared sequences.
 */
LZGError lzg_predicted_overlap(const LZGGraph *g, double d_i, double d_j,
                                double *out);

/**
 * Richness curve F(d) at many depth values efficiently.
 *
 * Precomputes M(k) for k=1..K once, then evaluates the Taylor series
 * at every requested depth — no redundant DP passes.
 *
 * @param g         The graph.
 * @param d_values  Array of depth values.
 * @param n         Number of depth values.
 * @param out       Output array of F(d) values (caller allocates, size n).
 */
LZGError lzg_richness_curve(const LZGGraph *g, const double *d_values,
                             uint32_t n, double *out);

#endif /* LZGRAPH_OCCUPANCY_H */
