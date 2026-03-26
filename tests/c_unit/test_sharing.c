/**
 * @file test_sharing.c
 * @brief Tests for predict_sharing_spectrum.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/sharing.h"

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

static void test_sharing_basic(void) {
    LZGGraph *g = build_graph();

    double draw_counts[] = {100.0, 200.0, 150.0};
    LZGSharingSpectrum ss;
    LZGError err = lzg_predict_sharing(g, draw_counts, 3, 0, &ss);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    donors=%u max_k=%u total_unique=%.1f total_draws=%.0f",
           ss.n_donors, ss.max_k, ss.expected_total, ss.total_draws);

    ASSERT_MSG(ss.n_donors == 3, "3 donors");
    ASSERT_MSG(ss.max_k == 3, "max_k = 3");
    ASSERT_MSG(ss.expected_total > 0, "positive total");
    ASSERT_MSG(ss.total_draws == 450.0, "total draws = 450");

    /* spectrum[0] = private (k=1), spectrum[1] = shared by 2, spectrum[2] = all 3 */
    printf("\n    k=1: %.2f, k=2: %.2f, k=3: %.2f",
           ss.spectrum[0], ss.spectrum[1], ss.spectrum[2]);

    /* All values should be non-negative */
    for (uint32_t k = 0; k < ss.max_k; k++)
        ASSERT_MSG(ss.spectrum[k] >= 0, "non-negative");

    /* On this tiny graph (12 paths) with depth 100-200 per donor,
     * most sequences are seen by all donors → k=3 dominates. */
    ASSERT_MSG(ss.spectrum[ss.max_k - 1] > 0, "some fully shared");

    lzg_sharing_spectrum_free(&ss);
    lzg_graph_destroy(g);
    PASS();
}

static void test_sharing_monotone_depth(void) {
    /* More depth → more overlap */
    LZGGraph *g = build_graph();

    double low[]  = {10.0, 10.0};
    double high[] = {1000.0, 1000.0};

    LZGSharingSpectrum ss_low, ss_high;
    lzg_predict_sharing(g, low, 2, 2, &ss_low);
    lzg_predict_sharing(g, high, 2, 2, &ss_high);

    printf("\n    low depth: shared=%.2f, high depth: shared=%.2f",
           ss_low.spectrum[1], ss_high.spectrum[1]);

    /* Higher depth → more shared sequences */
    ASSERT_MSG(ss_high.spectrum[1] >= ss_low.spectrum[1],
               "more depth → more overlap");

    lzg_sharing_spectrum_free(&ss_low);
    lzg_sharing_spectrum_free(&ss_high);
    lzg_graph_destroy(g);
    PASS();
}

static void test_sharing_sum_consistent(void) {
    /* sum(spectrum) should equal expected_total */
    LZGGraph *g = build_graph();

    double draw_counts[] = {50.0, 100.0, 75.0, 200.0};
    LZGSharingSpectrum ss;
    lzg_predict_sharing(g, draw_counts, 4, 0, &ss);

    double manual_sum = 0.0;
    for (uint32_t k = 0; k < ss.max_k; k++)
        manual_sum += ss.spectrum[k];

    printf("\n    sum=%.4f expected_total=%.4f", manual_sum, ss.expected_total);
    ASSERT_MSG(fabs(manual_sum - ss.expected_total) < 1e-10,
               "sum == expected_total");

    lzg_sharing_spectrum_free(&ss);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 4: Sharing Spectrum\n");
    printf("=====================================================\n\n");

    printf("[sharing]\n");
    RUN_TEST(test_sharing_basic);
    RUN_TEST(test_sharing_monotone_depth);
    RUN_TEST(test_sharing_sum_consistent);

    printf("\n=====================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
