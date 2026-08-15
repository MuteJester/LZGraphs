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

/**
 * Binomial discovery and novelty curves over an explicit P-sequence spectrum.
 *
 * Atom `a` represents `multiplicities[a]` distinct sequences, each having
 * probability `probabilities[a]`. For every effective draw count `n >= 1`,
 * this evaluates
 *
 *     R(n) = sum_a c_a [1 - (1 - p_a)^n]
 *     U(n) = sum_a c_a p_a (1 - p_a)^(n - 1).
 *
 * `R(n)` is the expected number of distinct sequences observed by draw `n`;
 * `U(n)` is the probability that draw `n` has not appeared previously.
 * Non-integer draw counts are accepted for smooth analytical curves. The
 * implementation uses `log1p`, `expm1`, long-double accumulation, and
 * compensated summation so probabilities far below float64 epsilon remain
 * effective when multiplied by very large draw counts.
 *
 * Probabilities must be finite in [0, 1], multiplicities finite and
 * non-negative, and draw counts finite and at least one. Empty atom and draw
 * arrays are valid no-ops; corresponding pointers may be NULL when their
 * length is zero.
 */
LZGError lzg_pseq_discovery_curve(
    const double *probabilities, const double *multiplicities,
    uint32_t n_atoms, const double *draw_counts, uint32_t n_draw_counts,
    double *richness_out, double *novelty_out);

#endif /* LZGRAPH_OCCUPANCY_H */
