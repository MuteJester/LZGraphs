/**
 * @file test_core.c
 * @brief Unit tests for lib/core: rng, hash_map, string_pool, common.
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "lzgraph/common.h"
#include "lzgraph/rng.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"

static int pass_count = 0, fail_count = 0;

#define RUN_TEST(fn) do { \
    printf("  %-44s ", #fn); \
    fn(); \
} while (0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } \
} while (0)

#define PASS() do { printf("PASS\n"); pass_count++; } while (0)

/* ═══════════════════════════ common.h ═══════════════════════════ */

static void test_aa_bitmask(void) {
    ASSERT_MSG(lzg_aa_to_bit('A') == 0,  "A → 0");
    ASSERT_MSG(lzg_aa_to_bit('Y') == 19, "Y → 19");
    ASSERT_MSG(lzg_aa_to_bit('*') == 31, "* → 31 (invalid)");

    uint32_t mask = lzg_aa_bitmask("CASS", 4);
    ASSERT_MSG(mask & (1u << 1),  "C bit set");
    ASSERT_MSG(mask & (1u << 0),  "A bit set");
    ASSERT_MSG(mask & (1u << 15), "S bit set");
    ASSERT_MSG(!(mask & (1u << 9)), "L bit NOT set");
    PASS();
}

/* ═══════════════════════════ rng.h ══════════════════════════════ */

static void test_rng_deterministic(void) {
    LZGRng r1, r2;
    lzg_rng_seed(&r1, 42);
    lzg_rng_seed(&r2, 42);
    for (int i = 0; i < 100; i++)
        ASSERT_MSG(lzg_rng_next(&r1) == lzg_rng_next(&r2), "same seed → same seq");
    PASS();
}

static void test_rng_double_range(void) {
    LZGRng rng;
    lzg_rng_seed(&rng, 123);
    for (int i = 0; i < 10000; i++) {
        double d = lzg_rng_double(&rng);
        ASSERT_MSG(d >= 0.0 && d < 1.0, "in [0,1)");
    }
    PASS();
}

/* ═══════════════════════════ hash_map.h ═════════════════════════ */

static void test_hm_basic(void) {
    LZGHashMap *m = lzg_hm_create(16);
    ASSERT_MSG(m != NULL, "create");

    ASSERT_MSG(lzg_hm_put(m, 100, 42) == true,  "new key");
    ASSERT_MSG(lzg_hm_put(m, 100, 43) == false, "existing key");

    uint64_t *v = lzg_hm_get(m, 100);
    ASSERT_MSG(v && *v == 43, "updated value");
    ASSERT_MSG(lzg_hm_get(m, 999) == NULL, "missing key");

    lzg_hm_destroy(m);
    PASS();
}

static void test_hm_many(void) {
    LZGHashMap *m = lzg_hm_create(16);
    for (uint64_t i = 0; i < 10000; i++)
        lzg_hm_put(m, i * 7 + 3, i);
    for (uint64_t i = 0; i < 10000; i++) {
        uint64_t *v = lzg_hm_get(m, i * 7 + 3);
        ASSERT_MSG(v && *v == i, "value match");
    }
    ASSERT_MSG(m->count == 10000, "count");
    lzg_hm_destroy(m);
    PASS();
}

static void test_hm_delete(void) {
    LZGHashMap *m = lzg_hm_create(16);
    lzg_hm_put(m, 1, 10);
    lzg_hm_put(m, 2, 20);
    ASSERT_MSG(lzg_hm_delete(m, 1) == true, "delete existing");
    ASSERT_MSG(lzg_hm_get(m, 1) == NULL, "deleted gone");
    ASSERT_MSG(lzg_hm_get(m, 2) != NULL, "other still there");
    ASSERT_MSG(lzg_hm_delete(m, 99) == false, "delete missing");
    lzg_hm_destroy(m);
    PASS();
}

/* ═══════════════════════════ string_pool.h ══════════════════════ */

static void test_sp_intern(void) {
    LZGStringPool *p = lzg_sp_create(64);
    ASSERT_MSG(p != NULL, "create");

    uint32_t a = lzg_sp_intern(p, "hello");
    uint32_t b = lzg_sp_intern(p, "world");
    uint32_t c = lzg_sp_intern(p, "hello");

    ASSERT_MSG(a != b, "different strings → different IDs");
    ASSERT_MSG(a == c, "same string → same ID");
    ASSERT_MSG(strcmp(lzg_sp_get(p, a), "hello") == 0, "get");
    ASSERT_MSG(lzg_sp_len(p, b) == 5, "len");
    ASSERT_MSG(p->count == 2, "count");

    lzg_sp_destroy(p);
    PASS();
}

static void test_sp_find(void) {
    LZGStringPool *p = lzg_sp_create(64);
    lzg_sp_intern(p, "abc");
    ASSERT_MSG(lzg_sp_find(p, "abc") != LZG_SP_NOT_FOUND, "found");
    ASSERT_MSG(lzg_sp_find(p, "xyz") == LZG_SP_NOT_FOUND, "not found");
    lzg_sp_destroy(p);
    PASS();
}

static void test_sp_many(void) {
    LZGStringPool *p = lzg_sp_create(16);
    char buf[32];
    for (int i = 0; i < 5000; i++) {
        snprintf(buf, sizeof(buf), "node_%d", i);
        lzg_sp_intern(p, buf);
    }
    ASSERT_MSG(p->count == 5000, "5000 unique strings");
    /* Verify deduplication */
    uint32_t id1 = lzg_sp_intern(p, "node_42");
    uint32_t id2 = lzg_sp_intern(p, "node_42");
    ASSERT_MSG(id1 == id2, "dedup after many");
    ASSERT_MSG(p->count == 5000, "still 5000");
    lzg_sp_destroy(p);
    PASS();
}

/* ═══════════════════════════ FNV-1a ═════════════════════════════ */

static void test_fnv1a(void) {
    uint64_t h1 = lzg_hash_str("CASSLG");
    uint64_t h2 = lzg_hash_str("CASSLG");
    uint64_t h3 = lzg_hash_str("CASSLE");
    ASSERT_MSG(h1 == h2, "same string → same hash");
    ASSERT_MSG(h1 != h3, "different strings → different hash");
    PASS();
}

/* ═══════════════════════════ main ═══════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 1: Core\n");
    printf("=====================================\n\n");

    printf("[common]\n");
    RUN_TEST(test_aa_bitmask);

    printf("\n[rng]\n");
    RUN_TEST(test_rng_deterministic);
    RUN_TEST(test_rng_double_range);

    printf("\n[hash_map]\n");
    RUN_TEST(test_hm_basic);
    RUN_TEST(test_hm_many);
    RUN_TEST(test_hm_delete);

    printf("\n[string_pool]\n");
    RUN_TEST(test_sp_intern);
    RUN_TEST(test_sp_find);
    RUN_TEST(test_sp_many);

    printf("\n[fnv1a]\n");
    RUN_TEST(test_fnv1a);

    printf("\n=====================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
