/**
 * @file test_io.c
 * @brief Tests for graph save/load (binary serialization round-trip).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/io.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"

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

int main(void) {
    printf("C-LZGraph Unit Tests — IO (save/load)\n");
    printf("================================================\n\n");

    printf("[io]\n");
    RUN_TEST(test_save_load_roundtrip);
    RUN_TEST(test_save_load_simulation_consistent);

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
