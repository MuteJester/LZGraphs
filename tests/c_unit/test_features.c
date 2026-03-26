/**
 * @file test_features.c
 * @brief Tests for ML feature extraction.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/features.h"

static int pass_count = 0, fail_count = 0;
#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

static const char *seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
};

static void test_feature_aligned(void) {
    LZGGraph *ref = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(ref, seqs, 6, NULL, NULL, NULL, 0.0, 0);

    /* Query with subset */
    LZGGraph *query = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(query, seqs, 3, NULL, NULL, NULL, 0.0, 0);

    double *vec = malloc(ref->n_nodes * sizeof(double));
    uint32_t dim;
    LZGError err = lzg_feature_aligned(ref, query, vec, &dim);
    ASSERT_MSG(err == LZG_OK, "ok");
    ASSERT_MSG(dim == ref->n_nodes, "dim = ref nodes");

    /* Some features should be > 0 (shared nodes), some 0 (query-missing) */
    uint32_t nonzero = 0;
    for (uint32_t i = 0; i < dim; i++) if (vec[i] > 0) nonzero++;
    printf("\n    dim=%u, nonzero=%u/%u", dim, nonzero, dim);
    ASSERT_MSG(nonzero > 0 && nonzero < dim, "partial overlap");

    free(vec);
    lzg_graph_destroy(ref);
    lzg_graph_destroy(query);
    PASS();
}

static void test_feature_mass_profile(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 6, NULL, NULL, NULL, 0.0, 0);

    double profile[31];
    LZGError err = lzg_feature_mass_profile(g, profile, 30);
    ASSERT_MSG(err == LZG_OK, "ok");

    /* Profile should sum to ~1 (it's a distribution over positions) */
    double sum = 0;
    for (int i = 0; i <= 30; i++) sum += profile[i];
    printf("\n    sum=%.4f", sum);
    ASSERT_MSG(fabs(sum - 1.0) < 0.1, "sums to ~1");

    /* Most mass should be at CDR3-typical positions (10-20) */
    double mass_10_20 = 0;
    for (int i = 10; i <= 20; i++) mass_10_20 += profile[i];
    printf(", mass[10-20]=%.2f", mass_10_20);
    ASSERT_MSG(mass_10_20 > 0.3, "significant mass at CDR3 positions");

    lzg_graph_destroy(g);
    PASS();
}

static void test_feature_stats(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 6, NULL, NULL, NULL, 0.0, 0);

    double stats[LZG_FEATURE_STATS_DIM];
    LZGError err = lzg_feature_stats(g, stats);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    nodes=%.0f edges=%.0f D0=%.0f D1=%.1f D2=%.1f entropy=%.2f",
           stats[0], stats[1], stats[4], stats[6], stats[7], stats[9]);

    ASSERT_MSG(stats[0] > 0, "has nodes");
    ASSERT_MSG(stats[1] > 0, "has edges");
    ASSERT_MSG(stats[4] >= stats[6], "D0 ≥ D1");
    ASSERT_MSG(stats[6] >= stats[7], "D1 ≥ D2");
    ASSERT_MSG(stats[9] > 0, "positive entropy");

    lzg_graph_destroy(g);
    PASS();
}

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 7d: ML Features\n");
    printf("==================================================\n\n");

    RUN_TEST(test_feature_aligned);
    RUN_TEST(test_feature_mass_profile);
    RUN_TEST(test_feature_stats);

    printf("\n==================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
