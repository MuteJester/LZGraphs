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
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"

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

static LZGGraph *build_coinflip_graph(uint32_t a_count, uint32_t b_count) {
    uint32_t n = a_count + b_count;
    const char **seqs = malloc(n * sizeof(*seqs));
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    if (!seqs || !g) {
        free(seqs);
        if (g) lzg_graph_destroy(g);
        return NULL;
    }

    for (uint32_t i = 0; i < a_count; i++) seqs[i] = "A";
    for (uint32_t i = 0; i < b_count; i++) seqs[a_count + i] = "C";

    lzg_graph_build(g, seqs, n, NULL, NULL, NULL, 0.0, 0);
    free(seqs);
    return g;
}

static double direct_mc_hill(const LZGSimResult *sim, uint32_t n, double alpha) {
    if (fabs(alpha - 1.0) < 1e-12) {
        double sum_lp = 0.0;
        uint32_t valid = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (sim[i].log_prob <= LZG_LOG_EPS + 1.0) continue;
            sum_lp += sim[i].log_prob;
            valid++;
        }
        return valid > 0 ? exp(-sum_lp / valid) : 0.0;
    }

    double sum = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (sim[i].log_prob <= LZG_LOG_EPS + 1.0) continue;
        sum += exp((alpha - 1.0) * sim[i].log_prob);
        valid++;
    }

    if (valid == 0) return 0.0;
    double moment = sum / valid;
    return moment > 0.0 ? pow(moment, 1.0 / (1.0 - alpha)) : 0.0;
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
    ASSERT_MSG(fabs(diag.total_absorbed - (47.0 / 48.0)) < 0.02,
               "MC absorbed mass tracks known leaky graph");
    ASSERT_MSG(fabs(diag.total_leaked - (1.0 / 48.0)) < 0.02,
               "MC leaked mass tracks known leaky graph");

    lzg_graph_destroy(g);
    PASS();
}

static void test_leaky_graph_has_exact_raw_diagnostics_but_proper_public_m1(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    double m1;
    LZGError err = lzg_pgen_diagnostics(g, 1e-2, &diag);
    ASSERT_MSG(err == LZG_OK, "diag ok");
    err = lzg_power_sum(g, 1.0, &m1);
    ASSERT_MSG(err == LZG_OK, "power sum ok");

    printf("\n    absorbed=%.6f leaked=%.6f M(1)=%.6f",
           diag.total_absorbed, diag.total_leaked, m1);

    ASSERT_MSG(fabs(diag.total_absorbed - (47.0 / 48.0)) < 0.02,
               "raw diagnostics stay close to known leakage");
    ASSERT_MSG(fabs(m1 - 1.0) < 1e-12,
               "public power_sum(1) is proper under the accepted model");

    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_diagnostics_atol_contract(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    LZGError err = lzg_pgen_diagnostics(g, 0.0, &diag);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    absorbed=%.6f atol=0 => is_proper=%d",
           diag.total_absorbed, (int)diag.is_proper);
    ASSERT_MSG(diag.is_proper == false, "strict atol contract honored");

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
    ASSERT_MSG(d0 > 0 && paths > 0, "positive support estimates");
    ASSERT_MSG(fabs(d0 - paths) / fmax(paths, 1.0) < 0.25,
               "D(0) and path count agree within MC error");

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
    /* D(2) = 1 / M(2) on the accepted-sequence model. */
    LZGGraph *g = build_graph();
    double m2;
    lzg_power_sum(g, 2.0, &m2);
    double d2;
    lzg_hill_number(g, 2.0, &d2);

    double expected_d2 = pow(m2, -1.0);

    printf("\n    M(2)=%.6e, D(2)=%.2f, expected=%.2f",
           m2, d2, expected_d2);

    ASSERT_MSG(m2 > 0, "M(2) > 0");
    ASSERT_MSG(fabs(d2 - expected_d2) / fmax(d2, 1) < 0.15, "D(2) == 1 / M(2)");

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

static void test_hill_number_d0_mc_matches_direct_formula(void) {
    const uint32_t n_samples = 4096;
    LZGGraph *g = build_coinflip_graph(3, 1);
    ASSERT_MSG(g != NULL, "coinflip graph");

    double d0;
    LZGError err = lzg_hill_number_mc(g, 0.0, n_samples, &d0);
    ASSERT_MSG(err == LZG_OK, "ok");

    LZGSimResult *sim = calloc(n_samples, sizeof(*sim));
    ASSERT_MSG(sim != NULL, "simulation alloc");

    LZGRng rng;
    lzg_rng_seed(&rng, 33333ULL);
    err = lzg_simulate(g, n_samples, &rng, sim);
    ASSERT_MSG(err == LZG_OK, "simulate ok");

    double expected = direct_mc_hill(sim, n_samples, 0.0);
    printf("\n    D(0)_mc=%.10f direct=%.10f", d0, expected);
    ASSERT_MSG(fabs(d0 - expected) < 1e-10, "D(0) matches direct MC formula");

    for (uint32_t i = 0; i < n_samples; i++) lzg_sim_result_free(&sim[i]);
    free(sim);
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

static void test_hill_numbers_mc_match_direct_formula(void) {
    const uint32_t n_samples = 4096;
    LZGGraph *g = build_coinflip_graph(3, 1);
    ASSERT_MSG(g != NULL, "coinflip graph");

    double orders[] = {0.0, 1.0, 2.0};
    double hills[3];
    LZGError err = lzg_hill_numbers_mc(g, orders, 3, n_samples, hills);
    ASSERT_MSG(err == LZG_OK, "ok");

    LZGSimResult *sim = calloc(n_samples, sizeof(*sim));
    ASSERT_MSG(sim != NULL, "simulation alloc");

    LZGRng rng;
    lzg_rng_seed(&rng, 44444ULL);
    err = lzg_simulate(g, n_samples, &rng, sim);
    ASSERT_MSG(err == LZG_OK, "simulate ok");

    for (uint32_t i = 0; i < 3; i++) {
        double expected = direct_mc_hill(sim, n_samples, orders[i]);
        printf("\n    D(%.0f)_mc=%.10f direct=%.10f",
               orders[i], hills[i], expected);
        ASSERT_MSG(fabs(hills[i] - expected) < 1e-10,
                   "batch hill matches direct MC formula");
    }

    for (uint32_t i = 0; i < n_samples; i++) lzg_sim_result_free(&sim[i]);
    free(sim);
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
    RUN_TEST(test_leaky_graph_has_exact_raw_diagnostics_but_proper_public_m1);
    RUN_TEST(test_pgen_diagnostics_atol_contract);
    RUN_TEST(test_effective_diversity);
    RUN_TEST(test_hill_number_d0);
    RUN_TEST(test_hill_number_d1);
    RUN_TEST(test_hill_number_d2);
    RUN_TEST(test_hill_number_d0_mc_matches_direct_formula);
    RUN_TEST(test_hill_numbers_batch);
    RUN_TEST(test_hill_numbers_mc_match_direct_formula);
    RUN_TEST(test_pgen_dynamic_range);

    printf("\n==========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
