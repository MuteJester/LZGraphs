/**
 * @file test_pgen_dist.c
 * @brief Tests for PGEN distribution: moments, analytical, pdf/cdf.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/pgen_dist.h"
#include "lzgraph/analytics.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

static LZGGraph *build_graph(void) {
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
        "CASRGGTVYEQYF", "CSVSTSETGDTEQYF", "CASSPPDGILGYTF",
        "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 12, NULL, NULL, NULL, 0.0, 0);
    return g;
}

/* ═══════════════════════════════════════════════════════════════ */

static void test_moments_basic(void) {
    LZGGraph *g = build_graph();
    LZGPgenMoments mom;
    LZGError err = lzg_pgen_moments(g, &mom);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    mean=%.2f std=%.2f skew=%.2f kurt=%.2f mass=%.4f",
           mom.mean, mom.std, mom.skewness, mom.kurtosis, mom.total_mass);

    ASSERT_MSG(mom.mean < 0, "mean log P is negative");
    ASSERT_MSG(mom.std > 0, "positive std");
    ASSERT_MSG(mom.variance > 0, "positive variance");
    ASSERT_MSG(fabs(mom.std * mom.std - mom.variance) < 1e-10, "std² = var");
    ASSERT_MSG(mom.total_mass > 0.5, "significant mass");

    lzg_graph_destroy(g);
    PASS();
}

static void test_moments_are_unconstrained_relative_to_raw_diagnostics(void) {
    /* pgen_moments() stays on the unconstrained forward-DP law. */
    LZGGraph *g = build_graph();

    LZGPgenMoments mom;
    lzg_pgen_moments(g, &mom);

    LZGPgenDiagnostics diag;
    lzg_pgen_diagnostics(g, 1e-12, &diag);

    printf("\n    moments.total_mass=%.4f, raw_absorbed=%.4f",
           mom.total_mass, diag.total_absorbed);

    ASSERT_MSG(fabs(mom.total_mass - 1.0) < 1e-12,
               "forward-DP total mass stays unit-normalized");
    ASSERT_MSG(fabs(diag.total_absorbed - (47.0 / 48.0)) < 0.02,
               "raw absorbed mass tracks leaky-graph value");
    ASSERT_MSG(fabs(mom.total_mass - diag.total_absorbed) > 1e-3,
               "pgen_moments and raw diagnostics describe different laws");

    lzg_graph_destroy(g);
    PASS();
}

static void test_analytical_basic(void) {
    LZGGraph *g = build_graph();
    LZGPgenDist dist;
    LZGError err = lzg_pgen_analytical(g, &dist);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    components=%u, global_mean=%.2f",
           dist.n_components, dist.global.mean);

    ASSERT_MSG(dist.n_components > 0, "has components");
    ASSERT_MSG(dist.n_components <= LZG_PGEN_MAX_COMPONENTS, "within limit");

    /* Weights should sum to ~1 */
    double wsum = 0.0;
    for (uint32_t c = 0; c < dist.n_components; c++) {
        wsum += dist.weights[c];
        ASSERT_MSG(dist.stds[c] > 0, "positive std per component");
    }
    ASSERT_MSG(fabs(wsum - 1.0) < 0.01, "weights sum to ~1");

    lzg_graph_destroy(g);
    PASS();
}

static void test_pdf_cdf(void) {
    LZGGraph *g = build_graph();
    LZGPgenDist dist;
    lzg_pgen_analytical(g, &dist);

    /* PDF should be non-negative and integrate to ~1 */
    double x = dist.global.mean;
    double pdf_val = lzg_pgen_pdf(&dist, x);
    double cdf_val = lzg_pgen_cdf(&dist, x);

    printf("\n    pdf(mean)=%.4f, cdf(mean)=%.4f", pdf_val, cdf_val);

    ASSERT_MSG(pdf_val > 0, "pdf > 0 at mean");
    ASSERT_MSG(cdf_val > 0.3 && cdf_val < 0.7, "cdf near 0.5 at mean");

    /* CDF at extreme values */
    ASSERT_MSG(lzg_pgen_cdf(&dist, -1000.0) < 0.001, "cdf(-1000) ≈ 0");
    ASSERT_MSG(lzg_pgen_cdf(&dist, 0.0) > 0.99, "cdf(0) ≈ 1");

    lzg_graph_destroy(g);
    PASS();
}

static void test_rvs(void) {
    LZGGraph *g = build_graph();
    LZGPgenDist dist;
    lzg_pgen_analytical(g, &dist);

    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    double samples[1000];
    LZGError err = lzg_pgen_sample(&dist, &rng, 1000, samples);
    ASSERT_MSG(err == LZG_OK, "ok");

    /* Sample mean should be close to analytical mean */
    double sum = 0.0;
    for (int i = 0; i < 1000; i++) sum += samples[i];
    double sample_mean = sum / 1000.0;

    printf("\n    sample_mean=%.2f, analytical_mean=%.2f",
           sample_mean, dist.global.mean);

    ASSERT_MSG(fabs(sample_mean - dist.global.mean) < 1.0,
               "sample mean ≈ analytical mean");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 1: PGEN Distribution\n");
    printf("======================================================\n\n");

    printf("[pgen_dist]\n");
    RUN_TEST(test_moments_basic);
    RUN_TEST(test_moments_are_unconstrained_relative_to_raw_diagnostics);
    RUN_TEST(test_analytical_basic);
    RUN_TEST(test_pdf_cdf);
    RUN_TEST(test_rvs);

    printf("\n======================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
