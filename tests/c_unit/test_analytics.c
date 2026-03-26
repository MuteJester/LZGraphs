/**
 * @file test_analytics.c
 * @brief Tests for Phase 5: analytics (Hill numbers, diversity, diagnostics).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
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

static void test_simulation_potential_size(void) {
    LZGGraph *g = build_graph();
    double count;
    LZGError err = lzg_graph_path_count(g, &count);
    ASSERT_MSG(err == LZG_OK, "ok");
    printf("\n    LZ-valid paths: %.0f", count);
    ASSERT_MSG(count >= 12, "at least as many paths as training seqs");
    ASSERT_MSG(count < 1e10, "finite and reasonable");
    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_diagnostics(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    LZGError err = lzg_pgen_diagnostics(g, 1e-2, &diag);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    absorbed=%.6f leaked=%.6f init_sum=%.6f",
           diag.total_absorbed, diag.total_leaked, diag.initial_prob_sum);

    ASSERT_MSG(fabs(diag.initial_prob_sum - 1.0) < 1e-10, "init sums to 1");
    ASSERT_MSG(diag.total_absorbed > 0.3, "significant mass absorbed");
    ASSERT_MSG(diag.total_absorbed <= 1.0 + 1e-10, "mass ≤ 1");

    lzg_graph_destroy(g);
    PASS();
}

static void test_effective_diversity(void) {
    LZGGraph *g = build_graph();
    LZGEffectiveDiversity div;
    LZGError err = lzg_effective_diversity(g, &div);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    H=%.4f nats, N_eff=%.2f, uniformity=%.4e",
           div.entropy_nats, div.effective_diversity, div.uniformity);

    ASSERT_MSG(div.entropy_nats > 0, "positive entropy");
    ASSERT_MSG(div.effective_diversity > 1, "N_eff > 1");
    ASSERT_MSG(div.uniformity > 0 && div.uniformity <= 1.0, "uniformity in (0,1]");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d0(void) {
    /* D(0) = path count */
    LZGGraph *g = build_graph();
    double d0;
    lzg_hill_number(g, 0.0, &d0);

    double paths;
    lzg_graph_path_count(g, &paths);

    printf("\n    D(0)=%.0f, paths=%.0f", d0, paths);
    ASSERT_MSG(fabs(d0 - paths) < 1e-6, "D(0) == path count");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d1(void) {
    /* D(1) = exp(H) = effective diversity */
    LZGGraph *g = build_graph();
    double d1;
    lzg_hill_number(g, 1.0, &d1);

    LZGEffectiveDiversity div;
    lzg_effective_diversity(g, &div);

    printf("\n    D(1)=%.4f, exp(H)=%.4f", d1, div.effective_diversity);
    ASSERT_MSG(fabs(d1 - div.effective_diversity) / fmax(d1, 1) < 0.15, "D(1) == exp(H)");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d2(void) {
    /* D(2) = 1/M(2) */
    LZGGraph *g = build_graph();
    double m2;
    lzg_power_sum(g, 2.0, &m2);
    double d2;
    lzg_hill_number(g, 2.0, &d2);

    /* D(2) on the NORMALIZED constrained distribution = (M(2)/M(1)^2)^(-1)
     * where M(1) is the total constrained mass. */
    double m1;
    lzg_power_sum(g, 1.0, &m1);
    double expected_d2 = pow(m2 / (m1 * m1), -1.0);

    printf("\n    M(1)=%.6f M(2)=%.6e, D(2)=%.2f, expected=%.2f",
           m1, m2, d2, expected_d2);

    ASSERT_MSG(m2 > 0, "M(2) > 0");
    ASSERT_MSG(fabs(d2 - expected_d2) / fmax(d2, 1) < 0.15, "D(2) == (M(2)/M(1)^2)^(-1)");

    /* D(2) ≤ D(1) ≤ D(0) (Hill number monotonicity) */
    double d0, d1;
    lzg_hill_number(g, 0.0, &d0);
    lzg_hill_number(g, 1.0, &d1);
    printf(" [D0=%.0f > D1=%.1f > D2=%.1f]", d0, d1, d2);
    ASSERT_MSG(d0 >= d1 - 1e-6, "D0 ≥ D1");
    ASSERT_MSG(d1 >= d2 - 1e-6, "D1 ≥ D2");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_numbers_batch(void) {
    LZGGraph *g = build_graph();
    double orders[] = {0, 1, 2, 3};
    double hills[4];
    LZGError err = lzg_hill_numbers(g, orders, 4, hills);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    D(0)=%.0f D(1)=%.2f D(2)=%.2f D(3)=%.2f",
           hills[0], hills[1], hills[2], hills[3]);

    /* Monotonicity: D(0) ≥ D(1) ≥ D(2) ≥ D(3) */
    for (int i = 0; i < 3; i++)
        ASSERT_MSG(hills[i] >= hills[i+1] - 1e-6, "monotone");

    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_dynamic_range(void) {
    LZGGraph *g = build_graph();
    LZGDynamicRange dr;
    LZGError err = lzg_pgen_dynamic_range(g, &dr);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    max_logP=%.2f min_logP=%.2f range=%.1f orders",
           dr.max_log_prob, dr.min_log_prob, dr.dynamic_range_orders);

    ASSERT_MSG(dr.max_log_prob > dr.min_log_prob, "max > min");
    ASSERT_MSG(dr.max_log_prob < 0, "max < 0");
    ASSERT_MSG(dr.dynamic_range_nats > 0, "positive range");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 5: Analytics\n");
    printf("==========================================\n\n");

    printf("[analytics]\n");
    RUN_TEST(test_simulation_potential_size);
    RUN_TEST(test_pgen_diagnostics);
    RUN_TEST(test_effective_diversity);
    RUN_TEST(test_hill_number_d0);
    RUN_TEST(test_hill_number_d1);
    RUN_TEST(test_hill_number_d2);
    RUN_TEST(test_hill_numbers_batch);
    RUN_TEST(test_pgen_dynamic_range);

    printf("\n==========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
