/**
 * @file test_publicness.c
 * @brief Tests for the Poisson-binomial repertoire-occupancy PMF.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/publicness.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define N_GROUPS 6
#define K_FFT    32

static const double depths[N_GROUPS] = {10.0, 120.0, 700.0, 4000.0, 63.0, 800.0};
static const double mult[N_GROUPS]   = {1.0, 2.0, 3.0, 1.0, 4.0, 2.0};
static const uint32_t n_repertoires  = 13;

/* ── Independent references ────────────────────────────────── */

/* Poisson-binomial PMF by DP convolution, one repertoire at a time. Nothing
 * here touches the generating function. */
static void reference_pmf(double p, double *out) {
    memset(out, 0, (n_repertoires + 1) * sizeof(double));
    out[0] = 1.0;

    for (uint32_t b = 0; b < N_GROUPS; b++) {
        double pi = (p <= 0.0) ? 0.0
                  : (p >= 1.0) ? 1.0
                               : -expm1(depths[b] * log1p(-p));
        for (uint32_t r = 0; r < (uint32_t)mult[b]; r++) {
            for (uint32_t k = n_repertoires; k > 0; k--)
                out[k] = out[k] * (1.0 - pi) + out[k - 1] * pi;
            out[0] *= 1.0 - pi;
        }
    }
}

/* Forward DFT of the sampled generating function, O(K^2) and written out in
 * full so the fast path has something naive to be checked against. */
static void invert_pgf(const double *g, double *out) {
    for (uint32_t m = 0; m < K_FFT; m++) {
        double acc = 0.0;
        for (uint32_t k = 0; k < K_FFT; k++) {
            double theta = 2.0 * 3.14159265358979323846 *
                           (double)k * (double)m / (double)K_FFT;
            acc += g[2 * k] * cos(theta) + g[2 * k + 1] * sin(theta);
        }
        out[m] = acc / (double)K_FFT;
    }
}

/* ═══════════════════════════════════════════════════════════════ */

static void test_moments_match_the_reference_pmf(void) {
    const double probes[] = {0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.5, 1.0};
    double mean[7], variance[7], pmf[64];

    LZGError err = lzg_publicness_moments(probes, 7, depths, mult,
                                          N_GROUPS, mean, variance);
    ASSERT_MSG(err == LZG_OK, "moments ok");

    for (uint32_t i = 0; i < 7; i++) {
        double m1 = 0.0, m2 = 0.0;
        reference_pmf(probes[i], pmf);
        for (uint32_t k = 0; k <= n_repertoires; k++) {
            m1 += (double)k * pmf[k];
            m2 += (double)k * (double)k * pmf[k];
        }
        ASSERT_MSG(fabs(mean[i] - m1) < 1e-12, "mean matches DP");
        ASSERT_MSG(fabs(variance[i] - (m2 - m1 * m1)) < 1e-12,
                   "variance matches DP");
    }
    PASS();
}

static void test_pgf_inverts_to_the_reference_pmf(void) {
    const double probes[] = {1e-5, 1e-3, 1e-2, 0.25};
    double *g = malloc(4 * K_FFT * 2 * sizeof(double));
    double inverted[K_FFT], pmf[64];
    LZGError err;

    ASSERT_MSG(g != NULL, "alloc");
    err = lzg_publicness_pgf(probes, 4, depths, mult, N_GROUPS, K_FFT, g);
    ASSERT_MSG(err == LZG_OK, "pgf ok");

    for (uint32_t i = 0; i < 4; i++) {
        reference_pmf(probes[i], pmf);
        invert_pgf(g + (size_t)i * 2 * K_FFT, inverted);

        /* G(1) = 1: the coefficients are a probability distribution. */
        ASSERT_MSG(fabs(g[(size_t)i * 2 * K_FFT] - 1.0) < 1e-14, "G(1) = 1");

        for (uint32_t k = 0; k <= n_repertoires; k++)
            ASSERT_MSG(fabs(inverted[k] - pmf[k]) < 1e-13, "PMF matches DP");
        /* No mass above the repertoire count, and none wrapped around. */
        for (uint32_t k = n_repertoires + 1; k < K_FFT; k++)
            ASSERT_MSG(fabs(inverted[k]) < 1e-13, "no mass past the cohort");
    }

    free(g);
    PASS();
}

static void test_accumulate_bins_and_conserves_mass(void) {
    const double probes[] = {1e-4, 1e-2};
    const double weight[] = {1000.0, 3.0};
    double edges[K_FFT + 1], counts[K_FFT], retained[2];
    double g[2 * K_FFT * 2], pmf[2 * K_FFT];
    LZGError err;

    for (uint32_t i = 0; i <= K_FFT; i++) edges[i] = (double)i;
    memset(counts, 0, sizeof(counts));

    err = lzg_publicness_pgf(probes, 2, depths, mult, N_GROUPS, K_FFT, g);
    ASSERT_MSG(err == LZG_OK, "pgf ok");
    invert_pgf(g, pmf);
    invert_pgf(g + 2 * K_FFT, pmf + K_FFT);

    {
        double mean[2], variance[2];
        err = lzg_publicness_moments(probes, 2, depths, mult, N_GROUPS,
                                     mean, variance);
        ASSERT_MSG(err == LZG_OK, "moments ok");
        err = lzg_publicness_accumulate(pmf, 2, K_FFT, mean, variance, weight,
                                        edges, K_FFT,
                                        LZG_PUBLICNESS_TAIL_SIGMA,
                                        LZG_PUBLICNESS_TAIL_FLOOR,
                                        LZG_PUBLICNESS_MASS_TOL,
                                        counts, retained);
        ASSERT_MSG(err == LZG_OK, "accumulate ok");
    }

    ASSERT_MSG(fabs(retained[0] - 1.0) < 1e-12, "atom 0 keeps its mass");
    ASSERT_MSG(fabs(retained[1] - 1.0) < 1e-12, "atom 1 keeps its mass");

    {
        double total = 0.0;
        for (uint32_t k = 0; k < K_FFT; k++) total += counts[k];
        ASSERT_MSG(fabs(total - 1003.0) < 1e-9, "every sequence lands in a bin");
        /* The absolute floor is wider than this whole transform, so levels
         * past the cohort are retained and hold clipped roundoff rather than
         * an exact zero. It has to stay at roundoff scale. */
        for (uint32_t k = n_repertoires + 1; k < K_FFT; k++)
            ASSERT_MSG(counts[k] < 1e-9, "nothing but roundoff past the cohort");
    }
    PASS();
}

static void test_truncation_discards_the_far_tail(void) {
    /* One delta-at-zero atom carrying an astronomical multiplicity: exactly
     * the shape that turns inversion roundoff into a spurious floor. */
    const double p = 1e-12, weight = 1e29;
    double edges[K_FFT + 1], counts[K_FFT], retained[1];
    double g[2 * K_FFT], pmf[K_FFT];
    double mean[1], variance[1];
    LZGError err;

    for (uint32_t i = 0; i <= K_FFT; i++) edges[i] = (double)i;
    memset(counts, 0, sizeof(counts));

    err = lzg_publicness_pgf(&p, 1, depths, mult, N_GROUPS, K_FFT, g);
    ASSERT_MSG(err == LZG_OK, "pgf ok");
    invert_pgf(g, pmf);

    err = lzg_publicness_moments(&p, 1, depths, mult, N_GROUPS,
                                 mean, variance);
    ASSERT_MSG(err == LZG_OK, "moments ok");

    /* A half-width of two levels: everything above must be dropped, not
     * clipped, or the multiplicity turns 1e-16 of roundoff into 1e13. */
    err = lzg_publicness_accumulate(pmf, 1, K_FFT, mean, variance, &weight,
                                    edges, K_FFT, 0.0, 2.0,
                                    LZG_PUBLICNESS_MASS_TOL, counts, retained);
    ASSERT_MSG(err == LZG_OK, "accumulate ok");
    ASSERT_MSG(fabs(counts[0] - weight) < 1e-6 * weight,
               "level 0 holds the atom");
    for (uint32_t k = 3; k < K_FFT; k++)
        ASSERT_MSG(counts[k] == 0.0, "far tail is exactly zero");
    PASS();
}

static void test_narrow_window_fails_loudly(void) {
    const double p = 1e-2, weight = 1.0;
    double edges[K_FFT + 1], counts[K_FFT], retained[1];
    double g[2 * K_FFT], pmf[K_FFT];
    double mean[1], variance[1];
    LZGError err;

    for (uint32_t i = 0; i <= K_FFT; i++) edges[i] = (double)i;
    memset(counts, 0, sizeof(counts));

    lzg_publicness_pgf(&p, 1, depths, mult, N_GROUPS, K_FFT, g);
    invert_pgf(g, pmf);
    lzg_publicness_moments(&p, 1, depths, mult, N_GROUPS, mean, variance);

    err = lzg_publicness_accumulate(pmf, 1, K_FFT, mean, variance, &weight,
                                    edges, K_FFT, 0.0, 0.0,
                                    LZG_PUBLICNESS_MASS_TOL, counts, retained);
    ASSERT_MSG(err == LZG_ERR_PARAM_OUT_OF_RANGE, "truncation error reported");
    ASSERT_MSG(strstr(lzg_error_message(), "occupancy mass") != NULL,
               "message names the retained mass");
    PASS();
}

static void test_rejects_bad_arguments(void) {
    double out_a[2], out_b[2], g[2 * K_FFT];
    const double bad_p[] = {1.5};
    const double bad_mult[N_GROUPS] = {1.0, 2.0, 3.5, 1.0, 4.0, 2.0};
    const double p = 1e-3;

    ASSERT_MSG(lzg_publicness_moments(NULL, 1, depths, mult, N_GROUPS,
                                      out_a, out_b) == LZG_ERR_NULL_ARG,
               "null probabilities rejected");
    ASSERT_MSG(lzg_publicness_moments(bad_p, 1, depths, mult, N_GROUPS,
                                      out_a, out_b)
                   == LZG_ERR_PARAM_OUT_OF_RANGE,
               "probability above one rejected");
    ASSERT_MSG(lzg_publicness_moments(&p, 1, depths, bad_mult, N_GROUPS,
                                      out_a, out_b)
                   == LZG_ERR_PARAM_OUT_OF_RANGE,
               "fractional multiplicity rejected");
    ASSERT_MSG(lzg_publicness_pgf(&p, 1, depths, mult, N_GROUPS, 31, g)
                   == LZG_ERR_PARAM_OUT_OF_RANGE,
               "odd transform length rejected");
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Publicness (Poisson-Binomial Occupancy)\n");
    printf("==============================================================\n\n");

    printf("[publicness]\n");
    RUN_TEST(test_moments_match_the_reference_pmf);
    RUN_TEST(test_pgf_inverts_to_the_reference_pmf);
    RUN_TEST(test_accumulate_bins_and_conserves_mass);
    RUN_TEST(test_truncation_discards_the_far_tail);
    RUN_TEST(test_narrow_window_fails_loudly);
    RUN_TEST(test_rejects_bad_arguments);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
