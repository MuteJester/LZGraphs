/**
 * @file publicness.h
 * @brief Analytical publicness: the Poisson-binomial repertoire-occupancy PMF.
 *
 * A sequence with model probability p is present in a repertoire that
 * contributed N distinct sequences with probability
 *
 *   pi = 1 - (1 - p)^N.
 *
 * Sampling depths differ between repertoires, so the number of repertoires
 * containing the sequence is Poisson-binomial, not binomial. Repertoires are
 * collapsed into a few dozen depth groups carrying multiplicities m_b, and the
 * probability generating function of the occupancy count is
 *
 *   G(z) = prod_b (1 - pi_b + pi_b z)^{m_b}.
 *
 * Evaluating G at the K-th roots of unity and taking a forward DFT recovers
 * the PMF exactly, in arithmetic. In floating point it does not, and the
 * failure mode is severe enough that lzg_publicness_accumulate() exists
 * chiefly to contain it. Read the comment on that function before touching it.
 *
 * This differs from sharing.h, which answers a related question by a cruder
 * route: that one integrates a Gaussian-mixture approximation of the PGEN
 * density against a Poisson detection model, whereas this one is driven by an
 * explicit probability spectrum and a binomial detection model.
 */
#ifndef LZGRAPH_PUBLICNESS_H
#define LZGRAPH_PUBLICNESS_H

#include "lzgraph/common.h"

/* ── Truncation defaults ───────────────────────────────────── */

/** Half-width of the retained support, in standard deviations. */
#define LZG_PUBLICNESS_TAIL_SIGMA 40.0

/** Absolute floor on the retained half-width, in occupancy levels. */
#define LZG_PUBLICNESS_TAIL_FLOOR 256.0

/** A truncated PMF must still sum to 1 within this tolerance. */
#define LZG_PUBLICNESS_MASS_TOL 1e-9

/* ── Closed-form moments ───────────────────────────────────── */

/**
 * Mean and variance of the occupancy count, for each of `n_atoms` sequence
 * probabilities, without inverting anything:
 *
 *   mu     = sum_b m_b pi_b
 *   sigma2 = sum_b m_b pi_b (1 - pi_b)
 *
 * These are what makes the truncation in lzg_publicness_accumulate() safe:
 * they locate the mass before the PMF is available, so no decision about
 * which part of the PMF to trust ever has to be taken from the PMF itself.
 *
 * @param p             Sequence probabilities [n_atoms], each in [0, 1].
 * @param n_atoms       Number of probabilities.
 * @param depths        Representative depth per group [n_groups], >= 0.
 * @param multiplicity  Repertoires per group [n_groups], >= 0, integral.
 * @param n_groups      Number of depth groups.
 * @param out_mean      Output [n_atoms].
 * @param out_variance  Output [n_atoms].
 */
LZGError lzg_publicness_moments(const double *p, uint32_t n_atoms,
                                const double *depths,
                                const double *multiplicity,
                                uint32_t n_groups,
                                double *out_mean, double *out_variance);

/* ── Generating function ───────────────────────────────────── */

/**
 * Evaluate G(z) at the `k_fft`-th roots of unity, for each probability.
 *
 * The caller inverts the result with a forward DFT divided by `k_fft` (the
 * forward transform, not the inverse: the inverse returns the PMF reflected
 * about zero). `k_fft` must exceed the total repertoire count so that no mass
 * wraps around, and must be even.
 *
 * @param p              Sequence probabilities [n_atoms], each in [0, 1].
 * @param n_atoms        Number of probabilities.
 * @param depths         Representative depth per group [n_groups].
 * @param multiplicity   Repertoires per group [n_groups], integral.
 * @param n_groups       Number of depth groups.
 * @param k_fft          Number of roots of unity. Even, >= 2.
 * @param out            Output [n_atoms * k_fft * 2], interleaved real and
 *                       imaginary parts, matching a C-contiguous complex128
 *                       array of shape (n_atoms, k_fft).
 */
LZGError lzg_publicness_pgf(const double *p, uint32_t n_atoms,
                            const double *depths,
                            const double *multiplicity,
                            uint32_t n_groups, uint32_t k_fft,
                            double *out);

/* ── Truncation and aggregation ────────────────────────────── */

/**
 * Truncate, clip, verify and bin one batch of inverted PMFs, accumulating
 * `weight[a] * P(count in bin)` into `out_counts`.
 *
 * WHY THE TRUNCATION IS NOT OPTIONAL
 * ----------------------------------
 * Every atom's PMF is scaled by its multiplicity in the probability spectrum.
 * On a foundation-scale graph those multiplicities reach 1.2e29 and sum to a
 * D0 of 1.9e32. Atoms deep in the tail have p of order 1e-80, so their true
 * PMF is a delta at zero: those sequences are detected in no repertoire at
 * all. The DFT returns that delta with relative roundoff of order 1e-16
 * smeared across all 65,536 output bins, and 1e-16 times 1e32 is 1e16.
 *
 * Clipping the PMF at zero, the natural response to the tiny negatives that
 * appear, keeps the positive half of that noise and rectifies it into a
 * floor. Summed over 15,574 atoms it produced roughly 2,000 spurious
 * sequences in EVERY publicness bin, which flattened the predicted tail
 * instead of letting it decay: 2,114 predicted in the top bin where 0 were
 * observed, and where only 135 atoms, carrying 108 sequences between them,
 * could physically reach at all.
 *
 * The fix is to zero everything outside [mu - T*sigma, mu + T*sigma] BEFORE
 * clipping negatives, so the far-tail roundoff is discarded rather than
 * rectified. The window is widened by an absolute floor, which covers the
 * degenerate mu ~ 0 atoms where sigma is also ~ 0 but the first few integers
 * still carry real mass. At T = 40 the Chernoff bound on the discarded mass
 * is below 1e-300, so nothing real is lost.
 *
 * Do not "simplify" this into a plain clip. The result looks plausible and
 * is wrong by fifteen orders of magnitude in the tail.
 *
 * Each atom's retained mass is checked against 1 so that a truncation
 * mistake fails loudly instead of quietly reshaping a tail.
 *
 * @param pmf            Inverted PMFs [n_atoms * k_fft], row-major.
 * @param n_atoms        Number of atoms in this batch.
 * @param k_fft          PMF length per atom.
 * @param mean           Occupancy mean per atom [n_atoms].
 * @param variance       Occupancy variance per atom [n_atoms].
 * @param weight         Spectrum multiplicity per atom [n_atoms].
 * @param edges          Non-decreasing bin edges [n_bins + 1]. Bin i covers
 *                       the half-open level range [edges[i], edges[i+1]).
 * @param n_bins         Number of bins.
 * @param tail_sigma     Half-width of the retained support, in sigma.
 * @param tail_floor     Absolute floor on the retained half-width.
 * @param mass_tol       Tolerance on the retained mass of each atom.
 * @param out_counts     Accumulated into, not overwritten [n_bins].
 * @param out_retained_mass  Optional per-atom retained mass [n_atoms], or NULL.
 * @return LZG_ERR_PARAM_OUT_OF_RANGE if any atom's truncated PMF fails the
 *         mass check, with a message naming the atom and its window.
 */
LZGError lzg_publicness_accumulate(const double *pmf, uint32_t n_atoms,
                                   uint32_t k_fft,
                                   const double *mean, const double *variance,
                                   const double *weight,
                                   const double *edges, uint32_t n_bins,
                                   double tail_sigma, double tail_floor,
                                   double mass_tol,
                                   double *out_counts,
                                   double *out_retained_mass);

#endif /* LZGRAPH_PUBLICNESS_H */
