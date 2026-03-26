/**
 * @file pgen_dist.h
 * @brief LZPgen distribution: moments, analytical Gaussian mixture, pdf/cdf.
 *
 * The PGEN distribution characterizes the probability of generation
 * (log P(seq)) across all sequences producible by the graph.
 *
 * lzg_pgen_moments() computes exact mean, variance, skewness, kurtosis
 * via 5-dimensional moment propagation through the LZ-constrained DP.
 *
 * lzg_pgen_analytical_distribution() produces a Gaussian mixture
 * (one component per walk length) for fast pdf/cdf/ppf evaluation.
 */
#ifndef LZGRAPH_PGEN_DIST_H
#define LZGRAPH_PGEN_DIST_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"

/* ── Moment results ────────────────────────────────────────── */

typedef struct {
    double mean;           /* E[log P]                     */
    double variance;       /* Var[log P]                   */
    double std;            /* sqrt(variance)               */
    double skewness;       /* standardized 3rd cumulant    */
    double kurtosis;       /* standardized 4th cumulant    */
    double total_mass;     /* Σ π(s) over valid walks      */
} LZGPgenMoments;

LZGError lzg_pgen_moments(const LZGGraph *g, LZGPgenMoments *out);

/* ── Gaussian mixture distribution ─────────────────────────── */

#define LZG_PGEN_MAX_COMPONENTS 64

typedef struct {
    double weights[LZG_PGEN_MAX_COMPONENTS];   /* mixture weights (sum=1) */
    double means[LZG_PGEN_MAX_COMPONENTS];     /* per-component mean      */
    double stds[LZG_PGEN_MAX_COMPONENTS];      /* per-component std       */
    int32_t walk_lengths[LZG_PGEN_MAX_COMPONENTS]; /* walk length per comp */
    uint32_t n_components;

    /* Global cumulants (from the full moment propagation) */
    LZGPgenMoments global;
} LZGPgenDist;

/**
 * Compute the analytical PGEN distribution as a Gaussian mixture.
 * One component per walk length, with exact per-length (weight, mean, std).
 */
LZGError lzg_pgen_analytical(const LZGGraph *g, LZGPgenDist *out);

/* ── Distribution evaluation ───────────────────────────────── */

/** Evaluate the PDF of the Gaussian mixture at point x (log-probability). */
double lzg_pgen_pdf(const LZGPgenDist *dist, double x);

/** Evaluate the CDF of the Gaussian mixture at point x. */
double lzg_pgen_cdf(const LZGPgenDist *dist, double x);

/** Monte Carlo sampling: draw n log-probability values from the mixture. */
LZGError lzg_pgen_sample(const LZGPgenDist *dist, LZGRng *rng,
                       uint32_t n, double *out);

#endif /* LZGRAPH_PGEN_DIST_H */
