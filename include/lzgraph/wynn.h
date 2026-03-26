/**
 * @file wynn.h
 * @brief Wynn epsilon algorithm for series acceleration.
 *
 * Given partial sums S_0, S_1, ..., S_n of a slowly converging or
 * alternating series, the Wynn epsilon algorithm computes the Shanks
 * transformation, which often converges much faster than the original.
 *
 * Particularly effective for alternating series like the Taylor
 * expansion of 1 - exp(-x).
 */
#ifndef LZGRAPH_WYNN_H
#define LZGRAPH_WYNN_H

#include "lzgraph/common.h"

/**
 * Apply the Wynn epsilon algorithm to a sequence of partial sums.
 *
 * @param partial_sums  Array of n partial sums S_0, S_1, ..., S_{n-1}.
 * @param n             Number of partial sums (must be >= 3).
 * @return              Best estimate of the series limit.
 */
double lzg_wynn_epsilon(const double *partial_sums, uint32_t n);

#endif /* LZGRAPH_WYNN_H */
