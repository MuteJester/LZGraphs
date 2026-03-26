/**
 * @file string_pool.c
 * @brief String interning with FNV-1a hashed open-addressing table.
 */
#include "lzgraph/string_pool.h"
#include "lzgraph/hash_map.h"   /* for lzg_hash_bytes */
#include <stdlib.h>
#include <string.h>

static uint32_t next_pow2(uint32_t v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; return v + 1;
}

LZGStringPool *lzg_sp_create(uint32_t initial_capacity) {
    if (initial_capacity < 64) initial_capacity = 64;
    LZGStringPool *p = calloc(1, sizeof(LZGStringPool));
    if (!p) return NULL;

    p->capacity = initial_capacity;
    p->strings  = calloc(initial_capacity, sizeof(char *));
    p->str_lens = calloc(initial_capacity, sizeof(uint32_t));

    p->ht_cap    = next_pow2(initial_capacity * 2);
    p->ht_hashes = malloc(p->ht_cap * sizeof(uint64_t));
    p->ht_ids    = malloc(p->ht_cap * sizeof(uint32_t));
    memset(p->ht_hashes, 0, p->ht_cap * sizeof(uint64_t));
    for (uint32_t i = 0; i < p->ht_cap; i++)
        p->ht_ids[i] = LZG_SP_NOT_FOUND;

    return p;
}

void lzg_sp_destroy(LZGStringPool *p) {
    if (!p) return;
    for (uint32_t i = 0; i < p->count; i++) free(p->strings[i]);
    free(p->strings); free(p->str_lens);
    free(p->ht_hashes); free(p->ht_ids);
    free(p);
}

static void sp_ht_resize(LZGStringPool *p) {
    uint32_t nc = p->ht_cap * 2;
    uint64_t *nh = malloc(nc * sizeof(uint64_t));
    uint32_t *ni = malloc(nc * sizeof(uint32_t));
    memset(nh, 0, nc * sizeof(uint64_t));
    for (uint32_t i = 0; i < nc; i++) ni[i] = LZG_SP_NOT_FOUND;

    uint32_t mask = nc - 1;
    for (uint32_t i = 0; i < p->ht_cap; i++) {
        if (p->ht_ids[i] != LZG_SP_NOT_FOUND) {
            uint32_t idx = (uint32_t)(p->ht_hashes[i] & mask);
            while (ni[idx] != LZG_SP_NOT_FOUND) idx = (idx + 1) & mask;
            nh[idx] = p->ht_hashes[i];
            ni[idx] = p->ht_ids[i];
        }
    }
    free(p->ht_hashes); free(p->ht_ids);
    p->ht_hashes = nh; p->ht_ids = ni; p->ht_cap = nc;
}

uint32_t lzg_sp_intern_n(LZGStringPool *p, const char *str, uint32_t len) {
    uint64_t h = lzg_hash_bytes(str, len);
    uint32_t mask = p->ht_cap - 1;
    uint32_t idx = (uint32_t)(h & mask);

    while (p->ht_ids[idx] != LZG_SP_NOT_FOUND) {
        if (p->ht_hashes[idx] == h) {
            uint32_t id = p->ht_ids[idx];
            if (p->str_lens[id] == len && memcmp(p->strings[id], str, len) == 0)
                return id;
        }
        idx = (idx + 1) & mask;
    }

    /* New entry */
    uint32_t new_id = p->count;
    if (new_id >= p->capacity) {
        p->capacity *= 2;
        p->strings  = realloc(p->strings,  p->capacity * sizeof(char *));
        p->str_lens = realloc(p->str_lens, p->capacity * sizeof(uint32_t));
    }
    char *copy = malloc(len + 1);
    memcpy(copy, str, len);
    copy[len] = '\0';

    p->strings[new_id]  = copy;
    p->str_lens[new_id] = len;
    p->count++;

    p->ht_hashes[idx] = h;
    p->ht_ids[idx]    = new_id;

    if (p->count * 5 > p->ht_cap * 3) sp_ht_resize(p);
    return new_id;
}

uint32_t lzg_sp_intern(LZGStringPool *p, const char *str) {
    return lzg_sp_intern_n(p, str, (uint32_t)strlen(str));
}

uint32_t lzg_sp_find(const LZGStringPool *p, const char *str) {
    uint32_t len = (uint32_t)strlen(str);
    uint64_t h = lzg_hash_bytes(str, len);
    uint32_t mask = p->ht_cap - 1;
    uint32_t idx = (uint32_t)(h & mask);

    while (p->ht_ids[idx] != LZG_SP_NOT_FOUND) {
        if (p->ht_hashes[idx] == h) {
            uint32_t id = p->ht_ids[idx];
            if (p->str_lens[id] == len && memcmp(p->strings[id], str, len) == 0)
                return id;
        }
        idx = (idx + 1) & mask;
    }
    return LZG_SP_NOT_FOUND;
}
