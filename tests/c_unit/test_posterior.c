/**
 * @file test_posterior.c
 * @brief Tests for Bayesian posterior over a prior graph.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/posterior.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static const char *test_seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
    "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
    "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
};

static LZGGraph *build_graph(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, test_seqs, 12, NULL, NULL, NULL, 0.0, 0);
    return g;
}

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

int main(void) {
    printf("C-LZGraph Unit Tests — Bayesian Posterior\n");
    printf("================================================\n\n");

    printf("[posterior]\n");
    RUN_TEST(test_posterior_basic);
    RUN_TEST(test_posterior_kappa_effect);

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
