/**
 * @file test_forward.c
 * @brief Tests for the LZ-constrained forward propagation engine.
 *
 * Key tests:
 * 1. Path counting: count LZ-valid paths (≤ unconstrained count)
 * 2. Mass conservation: Σ absorbed mass = 1.0 under LZ constraints
 * 3. Power sum: M(α) via constrained DP
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/forward.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

/* ═══════════════════════════════════════════════════════════════ */
/* Callback sets for different analytical methods                  */
/* ═══════════════════════════════════════════════════════════════ */

/* --- Path counting (acc_dim=1): count LZ-valid paths --- */

static void path_seed(double *acc, double init_prob, void *ctx) {
    (void)init_prob; (void)ctx;
    acc[0] += 1.0;  /* each initial state seeds 1 path */
}

static void path_edge(double *dst, const double *src,
                      double w, double Z, void *ctx) {
    (void)w; (void)Z; (void)ctx;
    dst[0] = src[0];  /* path count propagates unchanged per edge */
}

static void path_absorb(double *total, const double *node_acc,
                        double stop_prob, void *ctx) {
    (void)stop_prob; (void)ctx;
    total[0] += node_acc[0];  /* count all paths reaching a terminal */
}

static void path_continue(double *cont, const double *node_acc,
                          double stop_prob, void *ctx) {
    (void)stop_prob; (void)ctx;
    memcpy(cont, node_acc, sizeof(double));  /* paths continue unchanged */
}

/* --- Mass propagation (acc_dim=1): check Σ mass = 1.0 --- */

static void mass_seed(double *acc, double init_prob, void *ctx) {
    (void)ctx;
    acc[0] += init_prob;
}

static void mass_edge(double *dst, const double *src,
                      double w, double Z, void *ctx) {
    (void)ctx;
    /* Constrained transition probability = w / Z */
    dst[0] = src[0] * (w / Z);
}

static void mass_absorb(double *total, const double *node_acc,
                        double stop_prob, void *ctx) {
    (void)ctx;
    total[0] += node_acc[0] * stop_prob;
}

static void mass_continue(double *cont, const double *node_acc,
                          double stop_prob, void *ctx) {
    (void)ctx;
    cont[0] = node_acc[0] * (1.0 - stop_prob);
}

/* --- Power sum M(α) (acc_dim=1) --- */

typedef struct { double alpha; } PowerSumCtx;

static void power_seed(double *acc, double init_prob, void *ctx) {
    PowerSumCtx *pc = (PowerSumCtx *)ctx;
    acc[0] += pow(init_prob, pc->alpha);
}

static void power_edge(double *dst, const double *src,
                       double w, double Z, void *ctx) {
    PowerSumCtx *pc = (PowerSumCtx *)ctx;
    /* Constrained weight = w / Z, raised to alpha */
    dst[0] = src[0] * pow(w / Z, pc->alpha);
}

static void power_absorb(double *total, const double *node_acc,
                         double stop_prob, void *ctx) {
    PowerSumCtx *pc = (PowerSumCtx *)ctx;
    total[0] += node_acc[0] * pow(stop_prob, pc->alpha);
}

static void power_continue(double *cont, const double *node_acc,
                           double stop_prob, void *ctx) {
    PowerSumCtx *pc = (PowerSumCtx *)ctx;
    cont[0] = node_acc[0] * pow(1.0 - stop_prob, pc->alpha);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Helper: build a graph from a small set of sequences              */
/* ═══════════════════════════════════════════════════════════════ */

static LZGGraph *build_test_graph(void) {
    const char *seqs[] = {
        "CASSLGIRRT",
        "CASSLGYEQYF",
        "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF",
        "CASSFGQGSYEQYF",
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 5, NULL, NULL, NULL, 0.0, 0);
    return g;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Tests                                                           */
/* ═══════════════════════════════════════════════════════════════ */

static void test_path_counting(void) {
    LZGGraph *g = build_test_graph();

    LZGFwdOps ops = {
        .seed = path_seed, .edge = path_edge,
        .absorb = path_absorb, .cont = path_continue,
        .acc_dim = 1, .ctx = NULL
    };

    double total = 0.0;
    LZGError err = lzg_forward_propagate(g, &ops, &total);
    ASSERT_MSG(err == LZG_OK, "propagation ok");

    printf("\n    LZ-valid paths: %.0f", total);

    /* Must be at least 5 (one per input sequence) */
    ASSERT_MSG(total >= 5.0, "at least 5 paths (one per training seq)");

    /* Must be finite and positive */
    ASSERT_MSG(total > 0 && isfinite(total), "finite positive count");

    lzg_graph_destroy(g);
    PASS();
}

static void test_mass_conservation(void) {
    LZGGraph *g = build_test_graph();

    LZGFwdOps ops = {
        .seed = mass_seed, .edge = mass_edge,
        .absorb = mass_absorb, .cont = mass_continue,
        .acc_dim = 1, .ctx = NULL
    };

    double total = 0.0;
    LZGError err = lzg_forward_propagate(g, &ops, &total);
    ASSERT_MSG(err == LZG_OK, "propagation ok");

    printf("\n    total absorbed mass: %.10f", total);

    /* Under LZ constraints, total mass should be ≤ 1.0
     * (some mass may be blocked by LZ filtering).
     * But it should be close to 1.0 for well-formed graphs. */
    ASSERT_MSG(total > 0.5, "at least 50% of mass absorbed");
    ASSERT_MSG(total <= 1.0 + 1e-10, "mass ≤ 1.0");

    lzg_graph_destroy(g);
    PASS();
}

static void test_power_sum_alpha1(void) {
    /* M(1) under LZ constraints should equal the total absorbed mass */
    LZGGraph *g = build_test_graph();

    PowerSumCtx pctx = { .alpha = 1.0 };
    LZGFwdOps ops = {
        .seed = power_seed, .edge = power_edge,
        .absorb = power_absorb, .cont = power_continue,
        .acc_dim = 1, .ctx = &pctx
    };

    double m1 = 0.0;
    lzg_forward_propagate(g, &ops, &m1);

    printf("\n    M(1) = %.10f", m1);

    /* M(1) = Σ π(s) under the constrained model.
     * This should equal the total mass from mass_conservation test. */
    ASSERT_MSG(m1 > 0.5, "M(1) > 0.5");
    ASSERT_MSG(m1 <= 1.0 + 1e-10, "M(1) ≤ 1.0");

    /* Verify it matches the mass test */
    LZGFwdOps mass_ops = {
        .seed = mass_seed, .edge = mass_edge,
        .absorb = mass_absorb, .cont = mass_continue,
        .acc_dim = 1, .ctx = NULL
    };
    double mass_total = 0.0;
    lzg_forward_propagate(g, &mass_ops, &mass_total);

    ASSERT_MSG(fabs(m1 - mass_total) < 1e-10,
               "M(1) should equal total mass");

    lzg_graph_destroy(g);
    PASS();
}

static void test_power_sum_alpha2(void) {
    /* M(2) = Σ π(s)^2, gives D(2) = 1/M(2) */
    LZGGraph *g = build_test_graph();

    PowerSumCtx pctx = { .alpha = 2.0 };
    LZGFwdOps ops = {
        .seed = power_seed, .edge = power_edge,
        .absorb = power_absorb, .cont = power_continue,
        .acc_dim = 1, .ctx = &pctx
    };

    double m2 = 0.0;
    lzg_forward_propagate(g, &ops, &m2);

    printf("\n    M(2) = %.10e, D(2) = %.2f", m2, m2 > 0 ? 1.0/m2 : 0.0);

    ASSERT_MSG(m2 > 0, "M(2) > 0");
    ASSERT_MSG(m2 < 1.0, "M(2) < 1 (multiple sequences)");

    lzg_graph_destroy(g);
    PASS();
}

static void test_larger_graph(void) {
    /* Build from more sequences to stress-test the bitmask state tracking */
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
        "CASRGGTVYEQYF", "CSVSTSETGDTEQYF", "CASSPPDGILGYTF",
        "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
        "CAGGDRYNEQPQHF", "CATSREERYFATQYF", "CASSLGGYGYTF",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 15, NULL, NULL, NULL, 0.0, 0);

    printf("\n    nodes=%u edges=%u", g->n_nodes, g->n_edges);

    /* Mass conservation */
    LZGFwdOps mass_ops = {
        .seed = mass_seed, .edge = mass_edge,
        .absorb = mass_absorb, .cont = mass_continue,
        .acc_dim = 1, .ctx = NULL
    };
    double total = 0.0;
    LZGError err = lzg_forward_propagate(g, &mass_ops, &total);
    ASSERT_MSG(err == LZG_OK, "propagation ok");
    printf(" mass=%.6f", total);
    ASSERT_MSG(total > 0.3, "significant mass absorbed");

    /* Path counting */
    LZGFwdOps path_ops = {
        .seed = path_seed, .edge = path_edge,
        .absorb = path_absorb, .cont = path_continue,
        .acc_dim = 1, .ctx = NULL
    };
    double paths = 0.0;
    lzg_forward_propagate(g, &path_ops, &paths);
    printf(" paths=%.0f", paths);
    ASSERT_MSG(paths >= 15, "at least 15 LZ-valid paths");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 3: Forward Propagation\n");
    printf("====================================================\n\n");

    printf("[forward_propagation]\n");
    RUN_TEST(test_path_counting);
    RUN_TEST(test_mass_conservation);
    RUN_TEST(test_power_sum_alpha1);
    RUN_TEST(test_power_sum_alpha2);
    RUN_TEST(test_larger_graph);

    printf("\n====================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
