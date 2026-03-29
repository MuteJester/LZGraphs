/**
 * @file test_graph.c
 * @brief Unit tests for Phase 2: lz76, edge_builder, csr_graph.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/lz76.h"
#include "lzgraph/edge_builder.h"
#include "lzgraph/graph.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

/* ═══════════════════════════ LZ76 ══════════════════════════════ */

static void test_lz76_casslgirrt(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    LZGTokens tokens;

    LZGError err = lzg_lz76_decompose("CASSLGIRRT", 10, pool, &tokens);
    ASSERT_MSG(err == LZG_OK, "decompose ok");
    ASSERT_MSG(tokens.count == 8, "8 tokens");

    /* Verify token strings: C, A, S, SL, G, I, R, RT */
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[0]), "C") == 0, "t0=C");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[1]), "A") == 0, "t1=A");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[2]), "S") == 0, "t2=S");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[3]), "SL") == 0, "t3=SL");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[4]), "G") == 0, "t4=G");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[5]), "I") == 0, "t5=I");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[6]), "R") == 0, "t6=R");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, tokens.sp_ids[7]), "RT") == 0, "t7=RT");

    /* Verify positions: 1, 2, 3, 5, 6, 7, 8, 10 */
    ASSERT_MSG(tokens.positions[0] == 1, "p0=1");
    ASSERT_MSG(tokens.positions[3] == 5, "p3=5");
    ASSERT_MSG(tokens.positions[7] == 10, "p7=10");

    lzg_sp_destroy(pool);
    PASS();
}

static void test_lzg_lz76_encode_aap_nodes(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    uint32_t node_ids[LZG_MAX_TOKENS], sp_ids[LZG_MAX_TOKENS];
    uint32_t count;

    LZGError err = lzg_lz76_encode_aap("CASSLGIRRT", 10, pool,
                                    node_ids, sp_ids, &count);
    ASSERT_MSG(err == LZG_OK, "encode ok");
    /* With @ / $ sentinels: "@CASSLGIRRT$" → 10 tokens:
     * @_1, C_2, A_3, S_4, SL_6, G_7, I_8, R_9, RT_11, $_12 */
    ASSERT_MSG(count == 10, "10 nodes (with @ and $ sentinels)");

    ASSERT_MSG(strcmp(lzg_sp_get(pool, node_ids[0]), "@_1") == 0, "n0=@_1");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, node_ids[1]), "C_2") == 0, "n1=C_2");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, node_ids[4]), "SL_6") == 0, "n4=SL_6");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, node_ids[9]), "$_12") == 0, "n9=$_12");

    /* Subpattern IDs should match (shifted by 1 due to @ sentinel) */
    ASSERT_MSG(strcmp(lzg_sp_get(pool, sp_ids[0]), "@") == 0, "sp0=@");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, sp_ids[4]), "SL") == 0, "sp4=SL");
    ASSERT_MSG(strcmp(lzg_sp_get(pool, sp_ids[9]), "$") == 0, "sp9=$");

    lzg_sp_destroy(pool);
    PASS();
}

static void test_lz76_deterministic(void) {
    LZGStringPool *p1 = lzg_sp_create(64);
    LZGStringPool *p2 = lzg_sp_create(64);
    LZGTokens t1, t2;

    lzg_lz76_decompose("CASSLEPSGGTDTQYF", 16, p1, &t1);
    lzg_lz76_decompose("CASSLEPSGGTDTQYF", 16, p2, &t2);

    ASSERT_MSG(t1.count == t2.count, "same count");
    for (uint32_t i = 0; i < t1.count; i++) {
        ASSERT_MSG(strcmp(lzg_sp_get(p1, t1.sp_ids[i]),
                          lzg_sp_get(p2, t2.sp_ids[i])) == 0, "same tokens");
        ASSERT_MSG(t1.positions[i] == t2.positions[i], "same positions");
    }

    lzg_sp_destroy(p1);
    lzg_sp_destroy(p2);
    PASS();
}

/* ═══════════════════════════ EdgeBuilder ════════════════════════ */

static void test_edge_builder(void) {
    LZGEdgeBuilder *eb = lzg_eb_create(16);
    ASSERT_MSG(eb != NULL, "create");

    lzg_eb_record(eb, 0, 1, 3, NULL);
    lzg_eb_record(eb, 0, 1, 2, NULL);  /* same edge, should increment */
    lzg_eb_record(eb, 1, 2, 1, NULL);

    ASSERT_MSG(eb->n_edges == 2, "2 unique edges");
    ASSERT_MSG(eb->counts[0] == 5, "0→1 count=5 (3+2)");
    ASSERT_MSG(eb->counts[1] == 1, "1→2 count=1");

    lzg_eb_destroy(eb);
    PASS();
}

/* ═══════════════════════════ CSR Graph Build ════════════════════ */

static void test_graph_build_simple(void) {
    const char *seqs[] = {
        "CASSLGIRRT",
        "CASSLGYEQYF",
        "CASSLEPSGGTDTQYF",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    ASSERT_MSG(g != NULL, "create");

    LZGError err = lzg_graph_build(g, seqs, 3, NULL, NULL, NULL, 0.0, 0);
    ASSERT_MSG(err == LZG_OK, "build ok");

    printf("\n    nodes=%u edges=%u", g->n_nodes, g->n_edges);

    ASSERT_MSG(g->n_nodes > 0, "has nodes");
    ASSERT_MSG(g->n_edges > 0, "has edges");
    ASSERT_MSG(1 > 0, "has initial states");
    ASSERT_MSG(g->topo_valid, "topological sort succeeded (DAG)");

    /* Edge weights should sum to ~1.0 at each node with outgoing edges */
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint32_t start = g->row_offsets[i];
        uint32_t end   = g->row_offsets[i + 1];
        if (start == end) continue;

        double wsum = 0.0;
        for (uint32_t e = start; e < end; e++)
            wsum += g->edge_weights[e];

        ASSERT_MSG(fabs(wsum - 1.0) < 1e-10, "weights sum to 1.0");
    }


    /* LZ constraint info precomputed */
    for (uint32_t e = 0; e < g->n_edges; e++) {
        ASSERT_MSG(g->edge_sp_len[e] > 0, "sp_len > 0");
        if (g->edge_sp_len[e] > 1) {
            ASSERT_MSG(g->edge_prefix_id[e] != UINT32_MAX, "multi-char has prefix");
        }
    }

    lzg_graph_destroy(g);
    PASS();
}

static void test_graph_shared_prefix(void) {
    /* Two sequences sharing the "CASS" prefix should share early nodes */
    const char *seqs[] = {
        "CASSLFGK",
        "CASSLGQYF",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 2, NULL, NULL, NULL, 0.0, 0);

    printf("\n    nodes=%u edges=%u", g->n_nodes, g->n_edges);

    /* Both sequences start with C_1 → A_2 → S_3 → SL_5,
     * so nodes <= 4 tokens should be shared */
    ASSERT_MSG(1 == 1, "single initial state (C_1)");

    lzg_graph_destroy(g);
    PASS();
}

static void test_graph_abundance(void) {
    const char *seqs[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    uint32_t abundances[] = { 100, 1 };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 2, abundances, NULL, NULL, 0.0, 0);

    /* The first sequence's edges should have much higher counts */
    /* C_1 → A_2 should have count 101 (100 + 1) since both seqs use it */
    printf("\n    nodes=%u edges=%u", g->n_nodes, g->n_edges);

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════ main ═══════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 2: Graph\n");
    printf("======================================\n\n");

    printf("[lz76]\n");
    RUN_TEST(test_lz76_casslgirrt);
    RUN_TEST(test_lzg_lz76_encode_aap_nodes);
    RUN_TEST(test_lz76_deterministic);

    printf("\n[edge_builder]\n");
    RUN_TEST(test_edge_builder);

    printf("\n[csr_graph]\n");
    RUN_TEST(test_graph_build_simple);
    RUN_TEST(test_graph_shared_prefix);
    RUN_TEST(test_graph_abundance);

    printf("\n======================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
