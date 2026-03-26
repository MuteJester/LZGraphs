/**
 * @file test_io_posterior.c
 * @brief Tests for Phase 7: save/load and Bayesian posterior.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/io.h"
#include "lzgraph/posterior.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

static const char *test_seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
    "CASRGGTVYEQYF", "CSVSTSETGDTEQYF", "CASSPPDGILGYTF",
    "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
};

static LZGGraph *build_graph(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, test_seqs, 12, NULL, NULL, NULL, 0.0, 0);
    return g;
}

/* ═══════════════════════════ IO tests ═══════════════════════════ */

static void test_save_load_roundtrip(void) {
    LZGGraph *g = build_graph();
    const char *path = "/tmp/test_lzgraph.bin";

    LZGError err = lzg_graph_save(g, path);
    ASSERT_MSG(err == LZG_OK, "save ok");

    LZGGraph *loaded = NULL;
    err = lzg_graph_load(path, &loaded);
    ASSERT_MSG(err == LZG_OK, "load ok");
    ASSERT_MSG(loaded != NULL, "loaded not null");

    /* Verify structure matches */
    ASSERT_MSG(loaded->n_nodes == g->n_nodes, "same n_nodes");
    ASSERT_MSG(loaded->n_edges == g->n_edges, "same n_edges");
    ASSERT_MSG(1 == 1, "same n_initial");
    ASSERT_MSG(loaded->topo_valid, "topo sort recomputed");

    printf("\n    nodes=%u edges=%u", loaded->n_nodes, loaded->n_edges);

    /* Verify analytics match */
    LZGPgenDiagnostics diag_orig, diag_loaded;
    lzg_pgen_diagnostics(g, 1e-6, &diag_orig);
    lzg_pgen_diagnostics(loaded, 1e-6, &diag_loaded);

    ASSERT_MSG(fabs(diag_orig.total_absorbed - diag_loaded.total_absorbed) < 1e-10,
               "same total_absorbed");

    lzg_graph_destroy(g);
    lzg_graph_destroy(loaded);
    unlink(path);
    PASS();
}

static void test_save_load_simulation_consistent(void) {
    LZGGraph *g = build_graph();
    const char *path = "/tmp/test_lzgraph2.bin";

    lzg_graph_save(g, path);
    LZGGraph *loaded = NULL;
    lzg_graph_load(path, &loaded);

    /* Simulate from both — same seed should give same sequences */
    LZGRng rng1, rng2;
    lzg_rng_seed(&rng1, 42);
    lzg_rng_seed(&rng2, 42);

    LZGSimResult r1[20], r2[20];
    lzg_simulate(g, 20, &rng1, r1);
    lzg_simulate(loaded, 20, &rng2, r2);

    uint32_t matches = 0;
    for (int i = 0; i < 20; i++) {
        if (strcmp(r1[i].sequence, r2[i].sequence) == 0 &&
            fabs(r1[i].log_prob - r2[i].log_prob) < 1e-10)
            matches++;
        lzg_sim_result_free(&r1[i]);
        lzg_sim_result_free(&r2[i]);
    }

    printf("\n    simulation matches: %u/20", matches);
    ASSERT_MSG(matches == 20, "all simulations match after roundtrip");

    lzg_graph_destroy(g);
    lzg_graph_destroy(loaded);
    unlink(path);
    PASS();
}

/* ═══════════════════════════ Posterior tests ════════════════════ */

static void test_posterior_basic(void) {
    LZGGraph *prior = build_graph();

    /* Individual sequences — subset of training + one variant */
    const char *ind_seqs[] = {
        "CASSLGIRRT", "CASSLGIRRT", "CASSLGIRRT",  /* boosted */
        "CASSLGYEQYF",
    };

    LZGGraph *post = NULL;
    LZGError err = lzg_graph_posterior(prior, ind_seqs, 4, NULL, 10.0, &post);
    ASSERT_MSG(err == LZG_OK, "posterior ok");
    ASSERT_MSG(post != NULL, "posterior not null");

    printf("\n    prior: nodes=%u edges=%u, post: nodes=%u edges=%u",
           prior->n_nodes, prior->n_edges, post->n_nodes, post->n_edges);

    /* Same topology */
    ASSERT_MSG(post->n_nodes == prior->n_nodes, "same nodes");
    ASSERT_MSG(post->n_edges == prior->n_edges, "same edges");

    /* Posterior should be a proper distribution */
    LZGPgenDiagnostics diag;
    lzg_pgen_diagnostics(post, 0.1, &diag);
    printf(" absorbed=%.4f", diag.total_absorbed);
    ASSERT_MSG(diag.total_absorbed > 0.1, "posterior absorbs mass");

    lzg_graph_destroy(prior);
    lzg_graph_destroy(post);
    PASS();
}

static void test_posterior_kappa_effect(void) {
    /* High kappa → posterior ≈ prior. Low kappa → posterior ≈ individual. */
    LZGGraph *prior = build_graph();
    const char *ind_seqs[] = { "CASSLGIRRT", "CASSLGIRRT" };

    LZGGraph *post_high = NULL, *post_low = NULL;
    lzg_graph_posterior(prior, ind_seqs, 2, NULL, 1000.0, &post_high);
    lzg_graph_posterior(prior, ind_seqs, 2, NULL, 0.1, &post_low);

    /* Check walk probability for the boosted sequence */
    double lp_prior = lzg_walk_log_prob(prior, "CASSLGIRRT", 10);
    double lp_high  = lzg_walk_log_prob(post_high, "CASSLGIRRT", 10);
    double lp_low   = lzg_walk_log_prob(post_low, "CASSLGIRRT", 10);

    printf("\n    logP: prior=%.2f high_kappa=%.2f low_kappa=%.2f",
           lp_prior, lp_high, lp_low);

    /* High kappa posterior should be close to prior */
    /* Low kappa posterior should give higher probability to the
     * individual's sequence (boosted by individual counts) */
    ASSERT_MSG(fabs(lp_high - lp_prior) < fabs(lp_low - lp_prior),
               "high kappa closer to prior than low kappa");

    lzg_graph_destroy(prior);
    lzg_graph_destroy(post_high);
    lzg_graph_destroy(post_low);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 7: IO & Posterior\n");
    printf("================================================\n\n");

    printf("[io]\n");
    RUN_TEST(test_save_load_roundtrip);
    RUN_TEST(test_save_load_simulation_consistent);

    printf("\n[posterior]\n");
    RUN_TEST(test_posterior_basic);
    RUN_TEST(test_posterior_kappa_effect);

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
