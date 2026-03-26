/**
 * @file test_diversity.c
 * @brief Tests for Migration 6: perplexity, K-diversity, saturation, JSD.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/diversity.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

static const char *seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
    "CASRGGTVYEQYF", "CSVSTSETGDTEQYF", "CASSPPDGILGYTF",
    "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
};
#define N_SEQS 12

static LZGGraph *build_graph(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, N_SEQS, NULL, NULL, NULL, 0.0, 0);
    return g;
}

/* ═══════════════════════════ Perplexity ═════════════════════ */

static void test_sequence_perplexity(void) {
    LZGGraph *g = build_graph();
    double pp = lzg_sequence_perplexity(g, "CASSLGIRRT", 10);
    printf("\n    PP(CASSLGIRRT) = %.2f", pp);
    ASSERT_MSG(isfinite(pp) && pp > 1.0, "finite perplexity > 1");

    /* Unknown sequence should have higher perplexity */
    double pp_unk = lzg_sequence_perplexity(g, "XXXXXXXXXX", 10);
    printf(", PP(XXXX) = %.2f", pp_unk);
    /* May be INFINITY if path doesn't exist */

    lzg_graph_destroy(g);
    PASS();
}

static void test_repertoire_perplexity(void) {
    LZGGraph *g = build_graph();
    double pp = lzg_repertoire_perplexity(g, seqs, N_SEQS);
    printf("\n    repertoire PP = %.2f", pp);
    ASSERT_MSG(isfinite(pp) && pp > 1.0, "finite > 1");
    lzg_graph_destroy(g);
    PASS();
}

static void test_path_entropy_rate(void) {
    LZGGraph *g = build_graph();
    double h = lzg_path_entropy_rate(g, seqs, N_SEQS);
    printf("\n    entropy rate = %.2f bits/token", h);
    ASSERT_MSG(h > 0.0, "positive entropy rate");
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ K-Diversity ════════════════════ */

static void test_k_diversity(void) {
    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    LZGKDiversity kd;
    LZGError err = lzg_k_diversity(seqs, N_SEQS, LZG_VARIANT_AAP,
                                    5, 30, &rng, &kd);
    ASSERT_MSG(err == LZG_OK, "ok");
    printf("\n    K5 = %.1f ± %.1f [%.1f, %.1f]",
           kd.mean, kd.std, kd.ci_low, kd.ci_high);
    ASSERT_MSG(kd.mean > 0, "positive mean");
    ASSERT_MSG(kd.ci_low <= kd.mean && kd.mean <= kd.ci_high, "CI contains mean");
    PASS();
}

/* ═══════════════════════════ Saturation ═════════════════════ */

static void test_saturation_curve(void) {
    LZGSaturationPoint pts[100];
    uint32_t n_pts;
    LZGError err = lzg_saturation_curve(seqs, N_SEQS, LZG_VARIANT_AAP,
                                          3, pts, &n_pts);
    ASSERT_MSG(err == LZG_OK, "ok");
    ASSERT_MSG(n_pts > 0, "has points");

    printf("\n    points=%u", n_pts);
    printf(", first: n=%u nodes=%u edges=%u",
           pts[0].n_sequences, pts[0].n_nodes, pts[0].n_edges);
    printf(", last: n=%u nodes=%u edges=%u",
           pts[n_pts-1].n_sequences, pts[n_pts-1].n_nodes, pts[n_pts-1].n_edges);

    /* Monotonically increasing */
    for (uint32_t i = 1; i < n_pts; i++) {
        ASSERT_MSG(pts[i].n_nodes >= pts[i-1].n_nodes, "nodes monotone");
        ASSERT_MSG(pts[i].n_edges >= pts[i-1].n_edges, "edges monotone");
    }
    PASS();
}

/* ═══════════════════════════ JSD ════════════════════════════ */

static void test_jsd_self(void) {
    LZGGraph *g = build_graph();
    double jsd;
    lzg_jensen_shannon_divergence(g, g, &jsd);
    printf("\n    JSD(g, g) = %.6f", jsd);
    ASSERT_MSG(jsd < 1e-10, "JSD with self ≈ 0");
    lzg_graph_destroy(g);
    PASS();
}

static void test_jsd_different(void) {
    const char *seqs_a[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    const char *seqs_b[] = { "CASSXYZQYF", "CASSABCDEF" };

    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs_a, 2, NULL, NULL, NULL, 0.0, 0);
    LZGGraph *b = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(b, seqs_b, 2, NULL, NULL, NULL, 0.0, 0);

    double jsd;
    lzg_jensen_shannon_divergence(a, b, &jsd);
    printf("\n    JSD(a, b) = %.4f", jsd);
    ASSERT_MSG(jsd > 0.01, "different graphs have JSD > 0");
    ASSERT_MSG(jsd <= log(2.0) + 0.01, "JSD ≤ ln(2)");

    lzg_graph_destroy(a); lzg_graph_destroy(b);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 6: Diversity Metrics\n");
    printf("======================================================\n\n");

    printf("[perplexity]\n");
    RUN_TEST(test_sequence_perplexity);
    RUN_TEST(test_repertoire_perplexity);
    RUN_TEST(test_path_entropy_rate);

    printf("\n[k_diversity]\n");
    RUN_TEST(test_k_diversity);

    printf("\n[saturation]\n");
    RUN_TEST(test_saturation_curve);

    printf("\n[jsd]\n");
    RUN_TEST(test_jsd_self);
    RUN_TEST(test_jsd_different);

    printf("\n======================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
