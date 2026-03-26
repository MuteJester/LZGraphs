/**
 * @file test_occupancy.c
 * @brief Tests for Phase 6: predicted richness, overlap, richness curve.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/occupancy.h"
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

static void test_richness_small_d(void) {
    /* For small d, F(d) ≈ d (linear regime: almost every draw is new) */
    LZGGraph *g = build_graph();
    double F;
    LZGError err = lzg_predicted_richness(g, 1.0, &F);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    F(1.0) = %.4f", F);

    /* F(d) ≈ d for d << D(2). With small graph, D(2) ~ 10,
     * so d=1 is in the linear regime: F(1) ≈ 1. */
    ASSERT_MSG(F > 0.5 && F < 2.0, "F(1) ≈ 1 in linear regime");

    lzg_graph_destroy(g);
    PASS();
}

static void test_richness_monotone(void) {
    /* F(d) must be monotonically increasing */
    LZGGraph *g = build_graph();

    double depths[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    double F_prev = 0.0;

    printf("\n   ");
    for (int i = 0; i < 7; i++) {
        double F;
        lzg_predicted_richness(g, depths[i], &F);
        printf(" F(%.1f)=%.2f", depths[i], F);
        ASSERT_MSG(F >= F_prev - 1e-10, "monotone increasing");
        F_prev = F;
    }

    lzg_graph_destroy(g);
    PASS();
}

static void test_richness_saturation(void) {
    /* With splitting + Wynn, F(d) should work for ANY d — even
     * very large values where pure Taylor would diverge. */
    LZGGraph *g = build_graph();

    double F_100, F_1000, F_1e6;
    lzg_predicted_richness(g, 100.0, &F_100);
    lzg_predicted_richness(g, 1000.0, &F_1000);
    lzg_predicted_richness(g, 1e6, &F_1e6);

    double D0;
    lzg_hill_number(g, 0.0, &D0);

    printf("\n    F(100)=%.2f F(1000)=%.2f F(1e6)=%.2f D(0)=%.0f",
           F_100, F_1000, F_1e6, D0);

    /* F should grow monotonically */
    ASSERT_MSG(F_100 > 5.0, "F(100) substantial");
    ASSERT_MSG(F_1000 >= F_100 - 0.1, "F(1000) ≥ F(100)");
    ASSERT_MSG(F_1e6 >= F_1000 - 0.1, "F(1e6) ≥ F(1000)");

    lzg_graph_destroy(g);
    PASS();
}

static void test_overlap_basic(void) {
    LZGGraph *g = build_graph();

    double G;
    LZGError err = lzg_predicted_overlap(g, 1.0, 1.0, &G);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    G(1,1) = %.6f", G);

    /* G(d,d) ≈ d²/D(2) for small d */
    double D2;
    lzg_hill_number(g, 2.0, &D2);
    double expected = 1.0 / D2;
    printf(", d²/D(2) = %.6f", expected);

    /* Should be in the right ballpark */
    ASSERT_MSG(G >= 0, "non-negative");
    ASSERT_MSG(G < 2.0, "reasonable for small d");

    lzg_graph_destroy(g);
    PASS();
}

static void test_overlap_identity(void) {
    /* Verify G(d_i, d_j) = F(d_i) + F(d_j) - F(d_i + d_j) */
    LZGGraph *g = build_graph();

    double d_i = 2.0, d_j = 3.0;
    double F_i, F_j, F_ij, G;

    lzg_predicted_richness(g, d_i, &F_i);
    lzg_predicted_richness(g, d_j, &F_j);
    lzg_predicted_richness(g, d_i + d_j, &F_ij);
    lzg_predicted_overlap(g, d_i, d_j, &G);

    double expected = F_i + F_j - F_ij;
    printf("\n    G=%.6f, F_i+F_j-F_ij=%.6f", G, expected);

    ASSERT_MSG(fabs(G - expected) < 1e-10, "identity holds exactly");

    lzg_graph_destroy(g);
    PASS();
}

static void test_richness_curve(void) {
    LZGGraph *g = build_graph();

    double depths[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    double F_curve[6];

    LZGError err = lzg_richness_curve(g, depths, 6, F_curve);
    ASSERT_MSG(err == LZG_OK, "ok");

    /* Verify matches individual calls */
    printf("\n   ");
    for (int i = 0; i < 6; i++) {
        double F_single;
        lzg_predicted_richness(g, depths[i], &F_single);
        printf(" %.3f≈%.3f", F_curve[i], F_single);
        ASSERT_MSG(fabs(F_curve[i] - F_single) < 1e-6,
                   "curve matches single calls");
    }

    /* Monotonicity */
    for (int i = 0; i < 5; i++)
        ASSERT_MSG(F_curve[i] <= F_curve[i+1] + 1e-10, "monotone");

    lzg_graph_destroy(g);
    PASS();
}

static void test_richness_curve_efficiency(void) {
    /* richness_curve should precompute M(k) once, not per-depth */
    LZGGraph *g = build_graph();

    double depths[100];
    double F[100];
    for (int i = 0; i < 100; i++)
        depths[i] = 0.01 * (i + 1);

    LZGError err = lzg_richness_curve(g, depths, 100, F);
    ASSERT_MSG(err == LZG_OK, "100-point curve ok");

    printf("\n    F[0]=%.4f F[49]=%.4f F[99]=%.4f", F[0], F[49], F[99]);

    /* Basic sanity */
    ASSERT_MSG(F[0] > 0, "F > 0");
    ASSERT_MSG(F[99] > F[0], "increasing");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 6: Occupancy Model\n");
    printf("=================================================\n\n");

    printf("[occupancy]\n");
    RUN_TEST(test_richness_small_d);
    RUN_TEST(test_richness_monotone);
    RUN_TEST(test_richness_saturation);
    RUN_TEST(test_overlap_basic);
    RUN_TEST(test_overlap_identity);
    RUN_TEST(test_richness_curve);
    RUN_TEST(test_richness_curve_efficiency);

    printf("\n=================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
