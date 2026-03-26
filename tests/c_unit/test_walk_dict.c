/**
 * @file test_walk_dict.c
 * @brief Comprehensive validation of LZ76 constraint correctness.
 *
 * Tests every invariant of the LZGraph model:
 * - LZ76 decomposition correctness
 * - Walk dictionary constraint enforcement (all 3 LZ76 cases)
 * - Simulation validity and consistency
 * - LZPGEN correctness
 * - Probability model properties
 * - Edge cases
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/lz76.h"
#include "lzgraph/simulate.h"
#include "lzgraph/walk_dict.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/rng.h"

static int pass_count = 0, fail_count = 0;
#define RUN_TEST(fn) do { printf("  %-55s ", #fn); fn(); } while(0)
#define ASSERT_MSG(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } } while(0)
#define PASS() do { printf("PASS\n"); pass_count++; } while(0)

static const char *train_seqs[] = {
    "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
    "CASSLGIRRT", "CASSLGYEQYF", "CASSFGQGSYEQYF",
    "CASSDTSGGTDTQYF", "CASSQETQYF", "CASSLEPSGGTDTQYF",
};
#define N_TRAIN 12

static LZGGraph *build_graph(void) {
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, train_seqs, N_TRAIN, NULL, NULL, NULL, 0.0, 0);
    return g;
}

/* ═══════════════════════════════════════════════════════════════ */
/* 1. LZ76 round-trip                                             */
/* ═══════════════════════════════════════════════════════════════ */

static void test_lz76_roundtrip(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    for (int i = 0; i < 6; i++) {
        LZGTokens tokens;
        uint32_t len = (uint32_t)strlen(train_seqs[i]);
        lzg_lz76_decompose(train_seqs[i], len, pool, &tokens);

        char buf[256] = "";
        uint32_t pos = 0;
        for (uint32_t t = 0; t < tokens.count; t++) {
            const char *tok = lzg_sp_get(pool, tokens.sp_ids[t]);
            uint32_t tlen = lzg_sp_len(pool, tokens.sp_ids[t]);
            memcpy(buf + pos, tok, tlen);
            pos += tlen;
        }
        buf[pos] = '\0';
        ASSERT_MSG(strcmp(train_seqs[i], buf) == 0, "round-trip matches");
    }
    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 2. LZ76 dictionary rules on raw decomposition                 */
/* ═══════════════════════════════════════════════════════════════ */

static void test_lz76_dict_rules(void) {
    LZGStringPool *pool = lzg_sp_create(64);
    uint32_t violations = 0;

    for (int s = 0; s < 6; s++) {
        LZGTokens tokens;
        lzg_lz76_decompose(train_seqs[s], (uint32_t)strlen(train_seqs[s]),
                            pool, &tokens);

        LZGHashMap *dict = lzg_hm_create(32);
        for (uint32_t t = 0; t < tokens.count; t++) {
            const char *tok = lzg_sp_get(pool, tokens.sp_ids[t]);
            uint32_t tlen = lzg_sp_len(pool, tokens.sp_ids[t]);
            uint64_t h = lzg_hash_bytes(tok, tlen);
            bool is_last = (t == tokens.count - 1);

            if (tlen == 1) {
                /* Single char: must be novel (unless last token = case 3) */
                if (!is_last && lzg_hm_get(dict, h)) violations++;
            } else {
                /* Multi char: prefix in dict */
                uint64_t ph = lzg_hash_bytes(tok, tlen - 1);
                if (!lzg_hm_get(dict, ph)) violations++;
                /* Full token NOT in dict (unless last token = case 3) */
                if (!is_last && lzg_hm_get(dict, h)) violations++;
            }
            lzg_hm_put(dict, h, 1);
        }
        lzg_hm_destroy(dict);
    }
    printf("\n    dict rule violations: %u", violations);
    ASSERT_MSG(violations == 0, "all decompositions follow LZ76 rules");
    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 3. Simulated sequences are valid LZ76                          */
/* ═══════════════════════════════════════════════════════════════ */

static void test_simulate_lz_valid(void) {
    LZGGraph *g = build_graph();
    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    uint32_t N = 500;
    LZGSimResult *results = calloc(N, sizeof(LZGSimResult));
    lzg_simulate(g, N, &rng, results);

    LZGStringPool *pool = lzg_sp_create(256);
    uint32_t violations = 0;

    for (uint32_t i = 0; i < N; i++) {
        if (results[i].seq_len == 0) continue;

        LZGTokens tokens;
        lzg_lz76_decompose(results[i].sequence, results[i].seq_len,
                            pool, &tokens);

        LZGHashMap *dict = lzg_hm_create(32);
        for (uint32_t t = 0; t < tokens.count; t++) {
            const char *tok = lzg_sp_get(pool, tokens.sp_ids[t]);
            uint32_t tlen = lzg_sp_len(pool, tokens.sp_ids[t]);
            uint64_t h = lzg_hash_bytes(tok, tlen);
            bool is_last = (t == tokens.count - 1);

            if (tlen == 1) {
                if (!is_last && lzg_hm_get(dict, h)) { violations++; break; }
            } else {
                uint64_t ph = lzg_hash_bytes(tok, tlen - 1);
                if (!lzg_hm_get(dict, ph)) { violations++; break; }
                if (!is_last && lzg_hm_get(dict, h)) { violations++; break; }
            }
            lzg_hm_put(dict, h, 1);
        }
        lzg_hm_destroy(dict);
    }

    printf("\n    LZ violations: %u/%u", violations, N);
    ASSERT_MSG(violations == 0, "all simulated seqs are LZ-valid");

    lzg_sp_destroy(pool);
    for (uint32_t i = 0; i < N; i++) lzg_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 4. LZPGEN consistency: simulate vs walk_log_prob               */
/* ═══════════════════════════════════════════════════════════════ */

static void test_lzpgen_consistency(void) {
    LZGGraph *g = build_graph();
    LZGRng rng;
    lzg_rng_seed(&rng, 123);

    uint32_t N = 200;
    LZGSimResult *results = calloc(N, sizeof(LZGSimResult));
    lzg_simulate(g, N, &rng, results);

    uint32_t mismatches = 0;
    double max_delta = 0.0;

    for (uint32_t i = 0; i < N; i++) {
        if (results[i].seq_len == 0) continue;
        double lp = lzg_walk_log_prob(g, results[i].sequence, results[i].seq_len);
        double delta = fabs(lp - results[i].log_prob);
        if (delta > max_delta) max_delta = delta;
        if (delta > 0.01) mismatches++;
    }

    printf("\n    max |Δ|: %.2e, mismatches: %u/%u", max_delta, mismatches, N);
    ASSERT_MSG(mismatches == 0, "simulate log_prob == walk_log_prob");

    for (uint32_t i = 0; i < N; i++) lzg_sim_result_free(&results[i]);
    free(results);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 5. ALL training sequences have positive LZPGEN                 */
/* ═══════════════════════════════════════════════════════════════ */

static void test_all_training_positive(void) {
    LZGGraph *g = build_graph();
    uint32_t zero_count = 0;

    for (int i = 0; i < 6; i++) {
        double lp = lzg_walk_log_prob(g, train_seqs[i],
                                       (uint32_t)strlen(train_seqs[i]));
        if (lp <= LZG_LOG_EPS + 1.0) {
            printf("\n    ZERO: '%s' logP=%.2f", train_seqs[i], lp);
            zero_count++;
        }
    }
    printf("\n    positive: %u/6", 6 - zero_count);
    ASSERT_MSG(zero_count == 0, "ALL training sequences have P > 0");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 6. Random/unseen sequences have near-zero LZPGEN               */
/* ═══════════════════════════════════════════════════════════════ */

static void test_unseen_near_zero(void) {
    LZGGraph *g = build_graph();

    const char *unseen[] = {"XXXXXXXXXX", "AAAAAAAA", "QQQQQQQQQQ"};
    for (int i = 0; i < 3; i++) {
        double lp = lzg_walk_log_prob(g, unseen[i], (uint32_t)strlen(unseen[i]));
        ASSERT_MSG(lp < -600.0, "unseen seq has near-zero prob");
    }

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 7. All log_probs are finite and negative                       */
/* ═══════════════════════════════════════════════════════════════ */

static void test_log_probs_valid(void) {
    LZGGraph *g = build_graph();
    LZGRng rng;
    lzg_rng_seed(&rng, 42);

    LZGSimResult results[200];
    lzg_simulate(g, 200, &rng, results);

    uint32_t invalid = 0;
    for (int i = 0; i < 200; i++) {
        if (!isfinite(results[i].log_prob)) invalid++;
        if (results[i].log_prob >= 0.0) invalid++;
    }
    printf("\n    invalid: %u/200", invalid);
    ASSERT_MSG(invalid == 0, "all log_probs finite and negative");

    for (int i = 0; i < 200; i++) lzg_sim_result_free(&results[i]);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 8. Determinism                                                  */
/* ═══════════════════════════════════════════════════════════════ */

static void test_deterministic(void) {
    LZGGraph *g = build_graph();
    LZGRng rng1, rng2;
    lzg_rng_seed(&rng1, 77);
    lzg_rng_seed(&rng2, 77);

    LZGSimResult r1[10], r2[10];
    lzg_simulate(g, 10, &rng1, r1);
    lzg_simulate(g, 10, &rng2, r2);

    for (int i = 0; i < 10; i++) {
        ASSERT_MSG(strcmp(r1[i].sequence, r2[i].sequence) == 0, "same sequence");
        ASSERT_MSG(fabs(r1[i].log_prob - r2[i].log_prob) < 1e-10, "same log_prob");
    }

    for (int i = 0; i < 10; i++) {
        lzg_sim_result_free(&r1[i]);
        lzg_sim_result_free(&r2[i]);
    }
    lzg_graph_destroy(g);
    PASS();
}

/* Tests 9 (initial probs) and 11 (stop probs) removed — sentinel model */

/* ═══════════════════════════════════════════════════════════════ */
/* 10. Edge weights at each node sum to 1                         */
/* ═══════════════════════════════════════════════════════════════ */

static void test_edge_weights_normalized(void) {
    LZGGraph *g = build_graph();
    uint32_t bad_nodes = 0;

    for (uint32_t u = 0; u < g->n_nodes; u++) {
        uint32_t e_start = g->row_offsets[u];
        uint32_t e_end = g->row_offsets[u + 1];
        if (e_start == e_end) continue;

        double sum = 0.0;
        for (uint32_t e = e_start; e < e_end; e++)
            sum += g->edge_weights[e];

        if (fabs(sum - 1.0) > 0.01) bad_nodes++;
    }
    printf("\n    unnormalized nodes: %u/%u", bad_nodes, g->n_nodes);
    ASSERT_MSG(bad_nodes == 0, "all nodes have edge weights summing to ~1");

    lzg_graph_destroy(g);
    PASS();
}

/* Test 11 (stop probs) removed — sentinel model uses $-sinks */
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 12. Higher-frequency seqs have higher LZPGEN                   */
/* ═══════════════════════════════════════════════════════════════ */

static void test_frequency_ordering(void) {
    /* Build with abundances: CASSLGIRRT ×5, CASSLGYEQYF ×1 */
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGIRRT", "CASSLGIRRT",
        "CASSLGIRRT", "CASSLGIRRT", "CASSLGYEQYF",
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 6, NULL, NULL, NULL, 0.0, 0);

    double lp_high = lzg_walk_log_prob(g, "CASSLGIRRT", 10);
    double lp_low = lzg_walk_log_prob(g, "CASSLGYEQYF", 11);

    printf("\n    high-freq: %.4f, low-freq: %.4f", lp_high, lp_low);
    ASSERT_MSG(lp_high > lp_low, "more frequent seq has higher prob");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 13. Walk dict constraint manual test                           */
/* ═══════════════════════════════════════════════════════════════ */

static void test_walk_dict_manual(void) {
    LZGWalkDict wd = lzg_wd_create();
    LZGStringPool *pool = lzg_sp_create(32);

    /* Record tokens: C, A, S, SL */
    lzg_wd_record(&wd, pool, lzg_sp_intern(pool, "C"), 1);
    lzg_wd_record(&wd, pool, lzg_sp_intern(pool, "A"), 1);
    lzg_wd_record(&wd, pool, lzg_sp_intern(pool, "S"), 1);
    lzg_wd_record(&wd, pool, lzg_sp_intern(pool, "SL"), 2);

    /* C: single-char, already in dict → blocked */
    ASSERT_MSG(wd.single_char_bits & (1u << lzg_aa_to_bit('C')), "C in bits");
    /* G: single-char, novel → allowed */
    ASSERT_MSG(!(wd.single_char_bits & (1u << lzg_aa_to_bit('G'))), "G not in bits");
    /* SL: already in dict → blocked for continuing */
    uint64_t sl_h = lzg_hash_bytes("SL", 2);
    ASSERT_MSG(lzg_hm_get(wd.tokens, sl_h) != NULL, "SL in dict");
    /* SG: prefix S in dict, SG not in dict → allowed */
    uint64_t sg_h = lzg_hash_bytes("SG", 2);
    ASSERT_MSG(lzg_hm_get(wd.tokens, sg_h) == NULL, "SG not in dict");
    uint64_t s_h = lzg_hash_bytes("S", 1);
    ASSERT_MSG(lzg_hm_get(wd.tokens, s_h) != NULL, "S in dict");
    /* XY: prefix X not in dict → blocked */
    uint64_t x_h = lzg_hash_bytes("X", 1);
    ASSERT_MSG(lzg_hm_get(wd.tokens, x_h) == NULL, "X not in dict");

    lzg_wd_destroy(&wd);
    lzg_sp_destroy(pool);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 14. Save/load preserves LZPGEN                                 */
/* ═══════════════════════════════════════════════════════════════ */

static void test_save_load_lzpgen(void) {
    LZGGraph *g = build_graph();
    lzg_graph_save(g, "/tmp/test_audit.lzg");

    LZGGraph *g2 = NULL;
    lzg_graph_load("/tmp/test_audit.lzg", &g2);

    for (int i = 0; i < 6; i++) {
        double lp1 = lzg_walk_log_prob(g, train_seqs[i],
                                        (uint32_t)strlen(train_seqs[i]));
        double lp2 = lzg_walk_log_prob(g2, train_seqs[i],
                                        (uint32_t)strlen(train_seqs[i]));
        ASSERT_MSG(fabs(lp1 - lp2) < 1e-10, "save/load preserves LZPGEN");
    }

    lzg_graph_destroy(g);
    lzg_graph_destroy(g2);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 15. Simulated seqs contain only valid amino acids              */
/* ═══════════════════════════════════════════════════════════════ */

static void test_simulate_valid_chars(void) {
    LZGGraph *g = build_graph();
    LZGRng rng;
    lzg_rng_seed(&rng, 99);

    LZGSimResult results[100];
    lzg_simulate(g, 100, &rng, results);

    const char *valid_aa = "ACDEFGHIKLMNPQRSTVWY";
    uint32_t bad = 0;
    for (int i = 0; i < 100; i++) {
        for (uint32_t j = 0; j < results[i].seq_len; j++) {
            if (!strchr(valid_aa, results[i].sequence[j])) {
                bad++;
                break;
            }
        }
    }
    printf("\n    seqs with invalid chars: %u/100", bad);
    ASSERT_MSG(bad == 0, "all simulated seqs have valid amino acids");

    for (int i = 0; i < 100; i++) lzg_sim_result_free(&results[i]);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 16. Sequence ending with repeated token (case 3)               */
/* ═══════════════════════════════════════════════════════════════ */

static void test_case3_repeated_final_token(void) {
    /* Build from sequences ending with FF (double phenylalanine).
     * F appears as a single-char token early, then again at the end. */
    const char *seqs[] = {
        "CASSLGIRRTFF", "CASSLGYEQYFF", "CASSLEPSGGTDTQYFF",
        "CASSDTSGGTDTQYFF", "CASSFGQGSYEQYFF", "CASSQETQYFF",
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 6, NULL, NULL, NULL, 0.0, 0);

    /* Every training sequence should have positive LZPGEN */
    uint32_t zero = 0;
    for (int i = 0; i < 6; i++) {
        double lp = lzg_walk_log_prob(g, seqs[i], (uint32_t)strlen(seqs[i]));
        if (lp <= LZG_LOG_EPS + 1.0) {
            printf("\n    ZERO: '%s' logP=%.2f", seqs[i], lp);
            zero++;
        }
    }
    printf("\n    FF seqs with P>0: %u/6", 6 - zero);
    ASSERT_MSG(zero == 0, "all FF-ending sequences have positive prob");

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */
/* 17. Single-sequence graph                                      */
/* ═══════════════════════════════════════════════════════════════ */

static void test_single_sequence_graph(void) {
    const char *seqs[] = {"CASSLGIRRT"};
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 1, NULL, NULL, NULL, 0.0, 0);

    double lp = lzg_walk_log_prob(g, "CASSLGIRRT", 10);
    printf("\n    single-seq LZPGEN: %.4f", lp);
    ASSERT_MSG(lp > -100.0, "single seq has positive prob");

    /* Simulate should produce the same (or similar) sequence */
    LZGRng rng;
    lzg_rng_seed(&rng, 42);
    LZGSimResult r;
    lzg_simulate(g, 1, &rng, &r);
    printf(", sim: '%s'", r.sequence);
    ASSERT_MSG(r.seq_len > 0, "simulation produces non-empty");

    lzg_sim_result_free(&r);
    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Validation — Walk Dictionary & Model Correctness\n");
    printf("============================================================\n\n");

    RUN_TEST(test_lz76_roundtrip);
    RUN_TEST(test_lz76_dict_rules);
    RUN_TEST(test_simulate_lz_valid);
    RUN_TEST(test_lzpgen_consistency);
    RUN_TEST(test_all_training_positive);
    RUN_TEST(test_unseen_near_zero);
    RUN_TEST(test_log_probs_valid);
    RUN_TEST(test_deterministic);
    RUN_TEST(test_edge_weights_normalized);
    RUN_TEST(test_frequency_ordering);
    RUN_TEST(test_walk_dict_manual);
    RUN_TEST(test_save_load_lzpgen);
    RUN_TEST(test_simulate_valid_chars);
    RUN_TEST(test_case3_repeated_final_token);
    RUN_TEST(test_single_sequence_graph);

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
