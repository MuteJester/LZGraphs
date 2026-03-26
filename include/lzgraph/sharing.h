/**
 * @file sharing.h
 * @brief Predict the sharing spectrum: how many sequences are shared
 *        by exactly k out of N donors.
 *
 * Uses the analytical PGEN distribution (Gaussian mixture) and
 * Gauss-Hermite quadrature to evaluate:
 *
 *   f(k) = ∫ (g(x)/e^x) · Pr[k | λ(e^x)] dx
 *
 * where g(x) is the probability-weighted PGEN density, 1/e^x converts
 * to count-weighted, and Pr[k|λ] is the Poisson-Binomial PMF.
 */
#ifndef LZGRAPH_SHARING_H
#define LZGRAPH_SHARING_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

typedef struct {
    double  *spectrum;          /* [max_k]: f(k) for k = 1..max_k     */
    double   expected_total;    /* Σ spectrum                          */
    uint32_t max_k;
    uint32_t n_donors;
    double   total_draws;
} LZGSharingSpectrum;

/**
 * Predict the sharing spectrum for a cohort of donors.
 *
 * @param g             The graph (needs analytical PGEN distribution).
 * @param draw_counts   Effective draw counts per donor [n_donors].
 * @param n_donors      Number of donors.
 * @param max_k         Maximum sharing level (0 = auto: min(n_donors, 500)).
 * @param out           Output: caller allocates the struct, spectrum is
 *                      allocated internally.
 * @return LZG_OK on success.
 */
LZGError lzg_predict_sharing(const LZGGraph *g,
                                       const double *draw_counts,
                                       uint32_t n_donors,
                                       uint32_t max_k,
                                       LZGSharingSpectrum *out);

/** Free the spectrum array inside the result struct. */
void lzg_sharing_spectrum_free(LZGSharingSpectrum *ss);

#endif /* LZGRAPH_SHARING_H */
