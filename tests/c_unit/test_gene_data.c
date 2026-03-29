/**
 * @file test_gene_data.c
 * @brief Tests for V/J gene data support during graph construction.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

/* ═══════════════════════════════════════════════════════════════ */

static void test_build_with_genes(void) {
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF",
    };
    const char *v_genes[] = {
        "TRBV5-1", "TRBV5-1", "TRBV12-3",
        "TRBV12-3", "TRBV5-1",
    };
    const char *j_genes[] = {
        "TRBJ1-1", "TRBJ2-7", "TRBJ1-1",
        "TRBJ1-1", "TRBJ2-7",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    LZGError err = lzg_graph_build(g, seqs, 5, NULL, v_genes, j_genes, 0.0, 0);
    ASSERT_MSG(err == LZG_OK, "build ok");
    ASSERT_MSG(g->gene_data != NULL, "has gene data");

    LZGGeneData *gd = (LZGGeneData *)g->gene_data;
    printf("\n    v_genes=%u j_genes=%u vj_pairs=%u",
           gd->n_v_genes, gd->n_j_genes, gd->n_vj_pairs);

    ASSERT_MSG(gd->n_v_genes == 2, "2 V genes (TRBV5-1, TRBV12-3)");
    ASSERT_MSG(gd->n_j_genes == 2, "2 J genes (TRBJ1-1, TRBJ2-7)");
    ASSERT_MSG(gd->n_vj_pairs >= 2, "at least 2 VJ pairs");

    /* Check V marginals sum to 1 */
    double v_sum = 0.0;
    for (uint32_t i = 0; i < gd->n_v_genes; i++)
        v_sum += gd->v_marginal_probs[i];
    ASSERT_MSG(fabs(v_sum - 1.0) < 1e-10, "V marginals sum to 1");

    /* Check J marginals sum to 1 */
    double j_sum = 0.0;
    for (uint32_t i = 0; i < gd->n_j_genes; i++)
        j_sum += gd->j_marginal_probs[i];
    ASSERT_MSG(fabs(j_sum - 1.0) < 1e-10, "J marginals sum to 1");

    /* Check VJ probs sum to 1 */
    double vj_sum = 0.0;
    for (uint32_t i = 0; i < gd->n_vj_pairs; i++)
        vj_sum += gd->vj_probs[i];
    ASSERT_MSG(fabs(vj_sum - 1.0) < 1e-10, "VJ probs sum to 1");

    /* TRBV5-1 should be 3/5 = 0.6 */
    for (uint32_t i = 0; i < gd->n_v_genes; i++) {
        const char *name = lzg_sp_get(gd->gene_pool, gd->v_marginal_ids[i]);
        if (strcmp(name, "TRBV5-1") == 0) {
            printf("\n    P(TRBV5-1) = %.2f", gd->v_marginal_probs[i]);
            ASSERT_MSG(fabs(gd->v_marginal_probs[i] - 0.6) < 0.01,
                       "P(TRBV5-1) ≈ 0.6");
        }
    }

    lzg_graph_destroy(g);
    PASS();
}

static void test_build_without_genes(void) {
    const char *seqs[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 2, NULL, NULL, NULL, 0.0, 0);

    ASSERT_MSG(g->gene_data == NULL, "no gene data when not provided");

    lzg_graph_destroy(g);
    PASS();
}

static void test_per_edge_gene_csr(void) {
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGIRRT", "CASSLGYEQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF",
    };
    const char *v_genes[] = {
        "TRBV5-1", "TRBV12-3", "TRBV5-1",
        "TRBV12-3", "TRBV5-1",
    };
    const char *j_genes[] = {
        "TRBJ1-1", "TRBJ1-1", "TRBJ2-7",
        "TRBJ1-1", "TRBJ2-7",
    };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 5, NULL, v_genes, j_genes, 0.0, 0);

    LZGGeneData *gd = (LZGGeneData *)g->gene_data;
    ASSERT_MSG(gd != NULL, "has gene data");

    printf("\n    total V entries: %u, total J entries: %u",
           gd->total_v_entries, gd->total_j_entries);

    ASSERT_MSG(gd->total_v_entries > 0, "V gene CSR populated");
    ASSERT_MSG(gd->total_j_entries > 0, "J gene CSR populated");

    /* Check that some edges have V/J genes */
    uint32_t edges_with_v = 0, edges_with_j = 0;
    for (uint32_t e = 0; e < g->n_edges; e++) {
        if (gd->v_offsets[e + 1] > gd->v_offsets[e]) edges_with_v++;
        if (gd->j_offsets[e + 1] > gd->j_offsets[e]) edges_with_j++;
    }
    printf(", edges with V: %u, edges with J: %u", edges_with_v, edges_with_j);
    ASSERT_MSG(edges_with_v > 0, "some edges have V genes");
    ASSERT_MSG(edges_with_j > 0, "some edges have J genes");

    /* Test the query functions */
    uint32_t v5_id = lzg_sp_find(gd->gene_pool, "TRBV5-1");
    uint32_t j1_id = lzg_sp_find(gd->gene_pool, "TRBJ1-1");
    ASSERT_MSG(v5_id != LZG_SP_NOT_FOUND, "TRBV5-1 in gene pool");

    /* At least one edge should have TRBV5-1 */
    bool found_v = false;
    for (uint32_t e = 0; e < g->n_edges && !found_v; e++)
        if (lzg_edge_has_v(gd, e, v5_id)) found_v = true;
    ASSERT_MSG(found_v, "TRBV5-1 found on at least one edge");

    bool found_j = false;
    for (uint32_t e = 0; e < g->n_edges && !found_j; e++)
        if (lzg_edge_has_j(gd, e, j1_id)) found_j = true;
    ASSERT_MSG(found_j, "TRBJ1-1 found on at least one edge");

    for (uint32_t e = 0; e < g->n_edges; e++) {
        uint32_t vlo = gd->v_offsets[e], vhi = gd->v_offsets[e + 1];
        uint32_t jlo = gd->j_offsets[e], jhi = gd->j_offsets[e + 1];
        for (uint32_t i = vlo + 1; i < vhi; i++)
            ASSERT_MSG(gd->v_gene_ids[i - 1] < gd->v_gene_ids[i],
                       "V gene ids sorted and unique per edge");
        for (uint32_t i = jlo + 1; i < jhi; i++)
            ASSERT_MSG(gd->j_gene_ids[i - 1] < gd->j_gene_ids[i],
                       "J gene ids sorted and unique per edge");
    }

    lzg_graph_destroy(g);
    PASS();
}

static void test_gene_names_interned(void) {
    const char *seqs[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    const char *v[] = { "TRBV5-1", "TRBV5-1" };
    const char *j[] = { "TRBJ1-1", "TRBJ2-7" };

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 2, NULL, v, j, 0.0, 0);

    LZGGeneData *gd = (LZGGeneData *)g->gene_data;
    ASSERT_MSG(gd != NULL, "has gene data");

    /* Gene names should be retrievable from the pool */
    for (uint32_t i = 0; i < gd->n_v_genes; i++) {
        const char *name = lzg_sp_get(gd->gene_pool, gd->v_marginal_ids[i]);
        ASSERT_MSG(name != NULL && strlen(name) > 0, "valid V gene name");
    }
    for (uint32_t i = 0; i < gd->n_j_genes; i++) {
        const char *name = lzg_sp_get(gd->gene_pool, gd->j_marginal_ids[i]);
        ASSERT_MSG(name != NULL && strlen(name) > 0, "valid J gene name");
    }

    printf("\n    V genes:");
    for (uint32_t i = 0; i < gd->n_v_genes; i++)
        printf(" %s(%.2f)", lzg_sp_get(gd->gene_pool, gd->v_marginal_ids[i]),
               gd->v_marginal_probs[i]);
    printf("\n    J genes:");
    for (uint32_t i = 0; i < gd->n_j_genes; i++)
        printf(" %s(%.2f)", lzg_sp_get(gd->gene_pool, gd->j_marginal_ids[i]),
               gd->j_marginal_probs[i]);

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 3: Gene Data\n");
    printf("===============================================\n\n");

    printf("[gene_data]\n");
    RUN_TEST(test_build_with_genes);
    RUN_TEST(test_build_without_genes);
    RUN_TEST(test_per_edge_gene_csr);
    RUN_TEST(test_gene_names_interned);

    printf("\n===============================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
