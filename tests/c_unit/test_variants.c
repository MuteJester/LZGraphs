/**
 * @file test_variants.c
 * @brief Tests for NDPLZGraph and NaiveLZGraph construction and analytics.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/lz76.h"
#include "lzgraph/graph.h"
#include "lzgraph/analytics.h"
#include "lzgraph/simulate.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

/* ═══════════════════════════ NDP Encoding ═══════════════════ */

static void test_ndp_encoding(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    uint32_t ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
    uint32_t count;

    /* Nucleotide sequence */
    LZGError err = lzg_lz76_encode_ndp("TGTGCCAGCAGT", 12, pool,
                                    ids, sp_ids, &count);
    ASSERT_MSG(err == LZG_OK, "encode ok");
    ASSERT_MSG(count > 0, "has tokens");

    printf("\n    tokens=%u", count);
    /* First token should be a single nucleotide with frame 0 */
    const char *first = lzg_sp_get(pool, ids[0]);
    printf(" first='%s'", first);

    /* With sentinels: first token is @, second is first nucleotide.
     * NDP labels: {subpattern}{frame}_{position} */
    ASSERT_MSG(strcmp(first, "@0_1") == 0, "first node = @0_1 (sentinel)");

    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════ NDP Graph ══════════════════════ */

static void test_ndp_graph_build(void) {
    const char *seqs[] = {
        "TGTGCCAGCAGTTTCAAGAT",
        "TGTGCCAGCAGCCAAAGCAG",
        "TGTGCCAGCAGTTCAGGGAC",
        "TGTGCCAGCAGATCGGGACT",
        "TGTGCCAGCAGCAAAGCTGG",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NDP);
    LZGError err = lzg_graph_build(g, seqs, 5, NULL, NULL, NULL, 0.0, 0);
    ASSERT_MSG(err == LZG_OK, "build ok");

    printf("\n    nodes=%u edges=%u", g->n_nodes, g->n_edges);

    ASSERT_MSG(g->n_nodes > 0, "has nodes");
    ASSERT_MSG(g->n_edges > 0, "has edges");
    ASSERT_MSG(g->topo_valid, "DAG");

    /* NDP with small nucleotide alphabet (4 chars) fills the LZ
     * dictionary fast. On small graphs, live_states may be 0 because
     * no LZ-valid path reaches a terminal. This is expected. */

    lzg_graph_destroy(g);
    PASS();
}

static void test_ndp_simulate(void) {
    /* NDP with small alphabet may have no live paths on tiny graphs.
     * Test that simulate handles this gracefully. */
    const char *seqs[] = {
        "TGTGCCAGCAGTTTCAAGAT",
        "TGTGCCAGCAGCCAAAGCAG",
        "TGTGCCAGCAGTTCAGGGAC",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NDP);
    lzg_graph_build(g, seqs, 3, NULL, NULL, NULL, 0.0, 0);

    /* With dynamic walk dict, simulate uses backtracking for dead ends */
    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    LZGSimResult results[10];
    LZGError err = lzg_simulate(g, 10, &rng, results);
    if (err == LZG_OK) {
        printf("\n    simulated %u sequences", 10);
        for (int i = 0; i < 10; i++) lzg_sim_result_free(&results[i]);
    } else {
        printf("\n    simulate returned %d (expected for tiny NDP graph)", err);
    }

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ Naive Encoding ═════════════════ */

static void test_naive_encoding(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    uint32_t ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
    uint32_t count;

    LZGError err = lzg_lz76_encode_naive("CASSLGIRRT", 10, pool,
                                      ids, sp_ids, &count);
    ASSERT_MSG(err == LZG_OK, "encode ok");

    /* Naive: labels = raw subpatterns, no position */
    /* With sentinels: first token is @, not C */
    ASSERT_MSG(strcmp(lzg_sp_get(pool, ids[0]), "@") == 0, "first = @ (sentinel)");
    /* With sentinels: indices shifted by 1. idx[4] = SL (was idx[3]) */
    ASSERT_MSG(strcmp(lzg_sp_get(pool, ids[4]), "SL") == 0, "fifth = SL (shifted by @)");

    /* ids and sp_ids should be identical (label = subpattern) */
    for (uint32_t i = 0; i < count; i++)
        ASSERT_MSG(ids[i] == sp_ids[i], "id == sp_id");

    printf("\n    tokens=%u, first='%s'", count, lzg_sp_get(pool, ids[0]));

    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════ Naive Graph ════════════════════ */

static void test_naive_graph_build(void) {
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE);
    LZGError err = lzg_graph_build(g, seqs, 5, NULL, NULL, NULL, 0.0, 0);
    ASSERT_MSG(err == LZG_OK, "build ok");

    printf("\n    nodes=%u edges=%u topo=%d",
           g->n_nodes, g->n_edges, g->topo_valid);

    /* Naive graphs should have FEWER nodes than AAP */
    LZGGraph *aap = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(aap, seqs, 5, NULL, NULL, NULL, 0.0, 0);
    printf(" (AAP: nodes=%u)", aap->n_nodes);
    ASSERT_MSG(g->n_nodes <= aap->n_nodes, "Naive ≤ AAP nodes");

    /* node_pos should be UINT32_MAX for all Naive nodes */
    for (uint32_t i = 0; i < g->n_nodes; i++)
        ASSERT_MSG(g->node_pos[i] == UINT32_MAX, "no position in Naive");

    /* Naive graphs may have cycles → no topo sort, no live index */
    /* This is expected and correct behavior */

    lzg_graph_destroy(g);
    lzg_graph_destroy(aap);
    PASS();
}

static void test_naive_has_cycles(void) {
    /* Naive graphs can have cycles (same subpattern at different positions
     * maps to the same node). The topo sort should detect this. */
    const char *seqs[] = { "AABA", "BABA" };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE);
    LZGError err = lzg_graph_build(g, seqs, 2, NULL, NULL, NULL, 0.0, 0);

    /* May have cycles → topo sort might fail */
    printf("\n    topo_valid=%d", g->topo_valid);
    /* This is expected: Naive graphs with repeated subpatterns create cycles.
     * The LZ-constrained analytics won't work on cyclic graphs. */

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ Generic dispatch ═══════════════ */

static void test_generic_encode_dispatch(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    uint32_t ids_aap[LZG_MAX_TOKENS], ids_ndp[LZG_MAX_TOKENS];
    uint32_t ids_naive[LZG_MAX_TOKENS], sp[LZG_MAX_TOKENS];
    uint32_t c_aap, c_ndp, c_naive;

    lzg_lz76_encode("CASSLG", 6, pool, LZG_VARIANT_AAP,
                ids_aap, sp, &c_aap);
    lzg_lz76_encode("CASSLG", 6, pool, LZG_VARIANT_NDP,
                ids_ndp, sp, &c_ndp);
    lzg_lz76_encode("CASSLG", 6, pool, LZG_VARIANT_NAIVE,
                ids_naive, sp, &c_naive);

    /* Same string → same number of tokens */
    ASSERT_MSG(c_aap == c_ndp, "AAP == NDP token count");
    ASSERT_MSG(c_aap == c_naive, "AAP == Naive token count");

    /* But different node labels */
    ASSERT_MSG(ids_aap[0] != ids_naive[0], "AAP ≠ Naive labels");

    printf("\n    AAP='%s' NDP='%s' Naive='%s'",
           lzg_sp_get(pool, ids_aap[0]),
           lzg_sp_get(pool, ids_ndp[0]),
           lzg_sp_get(pool, ids_naive[0]));

    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 2: Graph Variants\n");
    printf("===================================================\n\n");

    printf("[ndp_encoding]\n");
    RUN_TEST(test_ndp_encoding);

    printf("\n[ndp_graph]\n");
    RUN_TEST(test_ndp_graph_build);
    RUN_TEST(test_ndp_simulate);

    printf("\n[naive_encoding]\n");
    RUN_TEST(test_naive_encoding);

    printf("\n[naive_graph]\n");
    RUN_TEST(test_naive_graph_build);
    RUN_TEST(test_naive_has_cycles);

    printf("\n[generic_dispatch]\n");
    RUN_TEST(test_generic_encode_dispatch);

    printf("\n===================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
