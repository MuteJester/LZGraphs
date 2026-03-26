/**
 * @file test_genomic_simulate.c
 * @brief Tests for gene-constrained simulation.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/simulate.h"
#include "lzgraph/simulate.h"

static int pass_count = 0, fail_count = 0;
#define RUN_TEST(fn) do { printf("  %-50s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

/* Shared test data — 10 sequences with known V/J genes */
static const char *seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF",
    "CASSLGIRRT", "CASSQETQYF", "CASSLGYEQYF",
    "CASSFGQGSYEQYF", "CASSDTSGGTDTQYF",
};
static const char *v_genes[] = {
    "TRBV5-1", "TRBV5-1", "TRBV12-3",
    "TRBV12-3", "TRBV5-1",
    "TRBV12-3", "TRBV5-1", "TRBV12-3",
    "TRBV5-1", "TRBV5-1",
};
static const char *j_genes[] = {
    "TRBJ1-1", "TRBJ2-7", "TRBJ1-1",
    "TRBJ1-1", "TRBJ2-7",
    "TRBJ1-1", "TRBJ2-7", "TRBJ2-7",
    "TRBJ1-1", "TRBJ1-1",
};

static LZGGraph *build_test_graph(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 10, NULL, v_genes, j_genes, 0.0, 0);
    return g;
}

/* ═══════════════════════════════════════════════════════════════ */

static void test_genomic_simulate_basic(void) {
    LZGGraph *g = build_test_graph();
    ASSERT_MSG(g->gene_data != NULL, "has gene data");

    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    uint32_t N = 20;
    LZGGeneSimResult *results = malloc(N * sizeof(LZGGeneSimResult));
    LZGError err = lzg_gene_simulate(g, N, &rng, results);
    ASSERT_MSG(err == LZG_OK, "simulate ok");

    uint32_t nonempty = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (results[i].base.seq_len > 0) nonempty++;
    }
    printf("\n    generated %u/%u non-empty sequences", nonempty, N);
    ASSERT_MSG(nonempty > 0, "at least some sequences generated");

    /* Print a few samples */
    for (uint32_t i = 0; i < N && i < 3; i++) {
        const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;
        const char *vname = results[i].v_gene_id != LZG_SP_NOT_FOUND
            ? lzg_sp_get(gd->gene_pool, results[i].v_gene_id) : "?";
        const char *jname = results[i].j_gene_id != LZG_SP_NOT_FOUND
            ? lzg_sp_get(gd->gene_pool, results[i].j_gene_id) : "?";
        printf("\n    [%u] '%s' V=%s J=%s logP=%.2f",
               i, results[i].base.sequence, vname, jname,
               results[i].base.log_prob);
    }

    for (uint32_t i = 0; i < N; i++) lzg_gene_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

static void test_genomic_simulate_specific_vj(void) {
    LZGGraph *g = build_test_graph();
    const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;

    uint32_t v5_id = lzg_sp_find(gd->gene_pool, "TRBV5-1");
    uint32_t j1_id = lzg_sp_find(gd->gene_pool, "TRBJ1-1");
    ASSERT_MSG(v5_id != LZG_SP_NOT_FOUND, "TRBV5-1 found");
    ASSERT_MSG(j1_id != LZG_SP_NOT_FOUND, "TRBJ1-1 found");

    LZGRng rng;
    lzg_rng_seed(&rng, 123);

    uint32_t N = 30;
    LZGGeneSimResult *results = malloc(N * sizeof(LZGGeneSimResult));
    LZGError err = lzg_gene_simulate_vj(g, N, &rng, v5_id, j1_id, results);
    ASSERT_MSG(err == LZG_OK, "ok");

    uint32_t nonempty = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (results[i].base.seq_len > 0) {
            nonempty++;
            ASSERT_MSG(results[i].v_gene_id == v5_id, "V gene matches");
            ASSERT_MSG(results[i].j_gene_id == j1_id, "J gene matches");
        }
    }
    printf("\n    TRBV5-1/TRBJ1-1: %u/%u non-empty", nonempty, N);
    ASSERT_MSG(nonempty > 0, "at least some generated");

    for (uint32_t i = 0; i < N; i++) lzg_gene_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

static void test_genomic_simulate_v_only(void) {
    LZGGraph *g = build_test_graph();
    const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;

    uint32_t v12_id = lzg_sp_find(gd->gene_pool, "TRBV12-3");
    ASSERT_MSG(v12_id != LZG_SP_NOT_FOUND, "TRBV12-3 found");

    LZGRng rng;
    lzg_rng_seed(&rng, 999);

    uint32_t N = 20;
    LZGGeneSimResult *results = malloc(N * sizeof(LZGGeneSimResult));
    LZGError err = lzg_gene_simulate_vj(g, N, &rng, v12_id,
                                            LZG_SP_NOT_FOUND, results);
    ASSERT_MSG(err == LZG_OK, "ok");

    uint32_t nonempty = 0;
    for (uint32_t i = 0; i < N; i++)
        if (results[i].base.seq_len > 0) nonempty++;
    printf("\n    TRBV12-3 only: %u/%u non-empty", nonempty, N);
    ASSERT_MSG(nonempty > 0, "some generated");

    for (uint32_t i = 0; i < N; i++) lzg_gene_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

static void test_genomic_simulate_deterministic(void) {
    LZGGraph *g = build_test_graph();

    LZGRng rng1, rng2;
    lzg_rng_seed(&rng1, 77);
    lzg_rng_seed(&rng2, 77);

    uint32_t N = 5;
    LZGGeneSimResult *r1 = malloc(N * sizeof(LZGGeneSimResult));
    LZGGeneSimResult *r2 = malloc(N * sizeof(LZGGeneSimResult));

    lzg_gene_simulate(g, N, &rng1, r1);
    lzg_gene_simulate(g, N, &rng2, r2);

    for (uint32_t i = 0; i < N; i++) {
        if (r1[i].base.seq_len > 0 && r2[i].base.seq_len > 0) {
            ASSERT_MSG(strcmp(r1[i].base.sequence, r2[i].base.sequence) == 0,
                       "deterministic: same sequence");
        }
    }

    for (uint32_t i = 0; i < N; i++) {
        lzg_gene_sim_result_free(&r1[i]);
        lzg_gene_sim_result_free(&r2[i]);
    }
    free(r1); free(r2);
    lzg_graph_destroy(g);
    PASS();
}

static void test_genomic_simulate_lz_validity(void) {
    LZGGraph *g = build_test_graph();

    LZGRng rng;
    lzg_rng_seed(&rng, 314);

    uint32_t N = 50;
    LZGGeneSimResult *results = malloc(N * sizeof(LZGGeneSimResult));
    lzg_gene_simulate(g, N, &rng, results);

    /* Verify each generated sequence has positive walk probability
     * (which implies LZ-valid) */
    uint32_t violations = 0, checked = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (results[i].base.seq_len == 0) continue;
        checked++;
        double lp = lzg_walk_log_prob(g, results[i].base.sequence,
                                              results[i].base.seq_len);
        if (lp <= LZG_LOG_EPS + 1.0) violations++;
    }
    printf("\n    LZ violations: %u/%u", violations, checked);
    ASSERT_MSG(violations == 0, "all sequences are LZ-valid");

    for (uint32_t i = 0; i < N; i++) lzg_gene_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

static void test_genomic_no_gene_data(void) {
    const char *s[] = { "CASSLGIRRT", "CASSLGYEQYF" };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, s, 2, NULL, NULL, NULL, 0.0, 0);

    LZGRng rng;
    lzg_rng_seed(&rng, 1);
    LZGGeneSimResult r;
    LZGError err = lzg_gene_simulate(g, 1, &rng, &r);
    ASSERT_MSG(err == LZG_ERR_NO_GENE_DATA, "rejects graph without gene data");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Migration 7c: Genomic Simulation\n");
    printf("========================================================\n\n");

    printf("[genomic_simulate]\n");
    RUN_TEST(test_genomic_simulate_basic);
    RUN_TEST(test_genomic_simulate_specific_vj);
    RUN_TEST(test_genomic_simulate_v_only);
    RUN_TEST(test_genomic_simulate_deterministic);
    RUN_TEST(test_genomic_simulate_lz_validity);
    RUN_TEST(test_genomic_no_gene_data);

    printf("\n========================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
