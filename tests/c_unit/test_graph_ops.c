/**
 * @file test_graph_ops.c
 * @brief Tests for Migration 5: hill_curve, graph_summary, graph_union.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/graph_ops.h"
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

/* ═══════════════════════════ hill_curve ═════════════════════ */

static void test_hill_curve_default(void) {
    LZGGraph *g = build_graph();
    LZGHillCurve hc;
    LZGError err = lzg_hill_curve(g, NULL, 0, &hc);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    n=%u", hc.n);
    ASSERT_MSG(hc.n == 12, "12 default orders");

    /* Approximately monotone (MC estimation has sampling variance) */
    for (uint32_t i = 0; i + 1 < hc.n; i++)
        ASSERT_MSG(hc.hill_numbers[i] >= hc.hill_numbers[i+1] * 0.8,
                   "approximately monotone (within 20% MC tolerance)");

    printf(" D(0)=%.0f D(1)=%.1f D(2)=%.1f",
           hc.hill_numbers[0], hc.hill_numbers[4], hc.hill_numbers[6]);

    lzg_hill_curve_free(&hc);
    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_curve_custom(void) {
    LZGGraph *g = build_graph();
    double orders[] = {0, 1, 2, 3};
    LZGHillCurve hc;
    lzg_hill_curve(g, orders, 4, &hc);

    ASSERT_MSG(hc.n == 4, "4 orders");
    ASSERT_MSG(hc.orders[0] == 0.0, "first = 0");
    ASSERT_MSG(hc.orders[3] == 3.0, "last = 3");

    lzg_hill_curve_free(&hc);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ graph_summary ══════════════════ */

static void test_graph_summary(void) {
    LZGGraph *g = build_graph();
    LZGGraphSummary sum;
    LZGError err = lzg_graph_summary(g, &sum);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    nodes=%u edges=%u max_out=%u max_in=%u isolates=%u dag=%d",
           sum.n_nodes, sum.n_edges,
           sum.max_out_degree, sum.max_in_degree, sum.n_isolates, sum.is_dag);

    ASSERT_MSG(sum.n_nodes == g->n_nodes, "nodes match");
    ASSERT_MSG(sum.n_edges == g->n_edges, "edges match");
    ASSERT_MSG(sum.max_out_degree > 0, "has out-degree");
    ASSERT_MSG(sum.max_in_degree > 0, "has in-degree");
    ASSERT_MSG(sum.is_dag, "is DAG");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ graph_union ════════════════════ */

static void test_graph_union_basic(void) {
    const char *seqs_a[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    };
    const char *seqs_b[] = {
        "CASSLGIRRT", "CASSDTSGGTDTQYF", "CASSXYZQYF",
    };

    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs_a, 3, NULL, NULL, NULL, 0.0, 0);

    LZGGraph *b = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(b, seqs_b, 3, NULL, NULL, NULL, 0.0, 0);

    printf("\n    a: nodes=%u edges=%u, b: nodes=%u edges=%u",
           a->n_nodes, a->n_edges, b->n_nodes, b->n_edges);

    LZGGraph *merged = NULL;
    LZGError err = lzg_graph_union(a, b, &merged);
    ASSERT_MSG(err == LZG_OK, "union ok");
    ASSERT_MSG(merged != NULL, "merged not null");

    printf("\n    merged: nodes=%u edges=%u",
           merged->n_nodes, merged->n_edges);

    /* Merged should have more or equal nodes/edges than either input */
    ASSERT_MSG(merged->n_nodes >= a->n_nodes, "merged nodes ≥ a");
    ASSERT_MSG(merged->n_nodes >= b->n_nodes, "merged nodes ≥ b");
    ASSERT_MSG(merged->n_edges >= a->n_edges, "merged edges ≥ a");

    /* "CASSLGIRRT" is in both — its edge counts should be summed */
    /* Verify the merged graph is a proper DAG */
    ASSERT_MSG(merged->topo_valid, "merged is DAG");

    /* Analytics should work on merged graph */
    LZGGraphSummary sum;
    lzg_graph_summary(merged, &sum);
    ASSERT_MSG(sum.is_dag, "merged summary says DAG");

    lzg_graph_destroy(a);
    lzg_graph_destroy(b);
    lzg_graph_destroy(merged);
    PASS();
}

static void test_graph_intersection(void) {
    const char *seqs_a[] = { "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF" };
    const char *seqs_b[] = { "CASSLGIRRT", "CASSDTSGGTDTQYF", "CASSXYZQYF" };

    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs_a, 3, NULL, NULL, NULL, 0.0, 0);
    LZGGraph *b = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(b, seqs_b, 3, NULL, NULL, NULL, 0.0, 0);

    LZGGraph *inter = NULL;
    LZGError err = lzg_graph_intersection(a, b, &inter);
    ASSERT_MSG(err == LZG_OK, "intersection ok");
    ASSERT_MSG(inter != NULL, "has result");

    printf("\n    a: %u edges, b: %u edges, inter: %u edges",
           a->n_edges, b->n_edges, inter->n_edges);

    /* Intersection should have fewer edges than either input */
    ASSERT_MSG(inter->n_edges <= a->n_edges, "inter ≤ a edges");
    ASSERT_MSG(inter->n_edges <= b->n_edges, "inter ≤ b edges");
    ASSERT_MSG(inter->n_edges > 0, "some shared edges");

    lzg_graph_destroy(a); lzg_graph_destroy(b); lzg_graph_destroy(inter);
    PASS();
}

static void test_graph_difference(void) {
    const char *seqs_a[] = { "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF" };
    const char *seqs_b[] = { "CASSLGIRRT" };

    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs_a, 3, NULL, NULL, NULL, 0.0, 0);
    LZGGraph *b = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(b, seqs_b, 1, NULL, NULL, NULL, 0.0, 0);

    LZGGraph *diff = NULL;
    LZGError err = lzg_graph_difference(a, b, &diff);
    ASSERT_MSG(err == LZG_OK, "difference ok");
    ASSERT_MSG(diff != NULL, "has result");

    printf("\n    a: %u edges, b: %u edges, diff: %u edges",
           a->n_edges, b->n_edges, diff->n_edges);

    /* Difference should have fewer edges than A (B's contribution removed) */
    ASSERT_MSG(diff->n_edges <= a->n_edges, "diff ≤ a edges");

    lzg_graph_destroy(a); lzg_graph_destroy(b); lzg_graph_destroy(diff);
    PASS();
}

static void test_graph_weighted_merge(void) {
    const char *seqs_a[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    const char *seqs_b[] = { "CASSLGIRRT", "CASSXYZQYF" };

    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs_a, 2, NULL, NULL, NULL, 0.0, 0);
    LZGGraph *b = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(b, seqs_b, 2, NULL, NULL, NULL, 0.0, 0);

    /* alpha=1, beta=1 should be identical to union */
    LZGGraph *wm = NULL, *un = NULL;
    lzg_graph_weighted_merge(a, b, 1.0, 1.0, &wm);
    lzg_graph_union(a, b, &un);

    ASSERT_MSG(wm != NULL && un != NULL, "both ok");
    ASSERT_MSG(wm->n_edges == un->n_edges, "weighted(1,1) == union edges");
    printf("\n    weighted(1,1): %u edges, union: %u edges",
           wm->n_edges, un->n_edges);

    /* alpha=2, beta=0 should double A's counts */
    LZGGraph *scaled = NULL;
    lzg_graph_weighted_merge(a, b, 2.0, 0.0, &scaled);
    ASSERT_MSG(scaled != NULL, "scale ok");
    ASSERT_MSG(scaled->n_edges == a->n_edges, "scaled has same edges as A");

    lzg_graph_destroy(a); lzg_graph_destroy(b);
    lzg_graph_destroy(wm); lzg_graph_destroy(un); lzg_graph_destroy(scaled);
    PASS();
}

static void test_graph_union_variant_mismatch(void) {
    const char *seqs[] = { "CASSLGIRRT" };
    LZGGraph *a = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(a, seqs, 1, NULL, NULL, NULL, 0.0, 0);

    LZGGraph *b = lzg_graph_create(LZG_VARIANT_NDP);
    lzg_graph_build(b, seqs, 1, NULL, NULL, NULL, 0.0, 0);

    LZGGraph *merged = NULL;
    LZGError err = lzg_graph_union(a, b, &merged);
    ASSERT_MSG(err != LZG_OK, "variant mismatch rejected");
    ASSERT_MSG(merged == NULL, "no result on mismatch");

    lzg_graph_destroy(a);
    lzg_graph_destroy(b);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 5: Graph Operations\n");
    printf("=====================================================\n\n");

    printf("[hill_curve]\n");
    RUN_TEST(test_hill_curve_default);
    RUN_TEST(test_hill_curve_custom);

    printf("\n[graph_summary]\n");
    RUN_TEST(test_graph_summary);

    printf("\n[graph_set_operations]\n");
    RUN_TEST(test_graph_union_basic);
    RUN_TEST(test_graph_intersection);
    RUN_TEST(test_graph_difference);
    RUN_TEST(test_graph_weighted_merge);
    RUN_TEST(test_graph_union_variant_mismatch);

    printf("\n=====================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
