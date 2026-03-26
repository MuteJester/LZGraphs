/**
 * @file hash_map.c
 * @brief Open-addressing hash map with linear probing.
 */
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>

static uint32_t next_pow2(uint32_t v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; return v + 1;
}

LZGHashMap *lzg_hm_create(uint32_t initial_capacity) {
    if (initial_capacity < 16) initial_capacity = 16;
    initial_capacity = next_pow2(initial_capacity);

    LZGHashMap *m = calloc(1, sizeof(LZGHashMap));
    if (!m) return NULL;

    m->keys   = malloc(initial_capacity * sizeof(uint64_t));
    m->values = malloc(initial_capacity * sizeof(uint64_t));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values); free(m);
        return NULL;
    }
    m->capacity = initial_capacity;
    memset(m->keys, 0xFF, initial_capacity * sizeof(uint64_t)); /* LZG_HM_EMPTY */
    return m;
}

void lzg_hm_destroy(LZGHashMap *m) {
    if (!m) return;
    free(m->keys); free(m->values); free(m);
}

static void hm_resize(LZGHashMap *m, uint32_t new_cap) {
    uint64_t *ok = m->keys, *ov = m->values;
    uint32_t oc = m->capacity;
    m->keys   = malloc(new_cap * sizeof(uint64_t));
    m->values = malloc(new_cap * sizeof(uint64_t));
    m->capacity = new_cap;
    m->count = m->tombstones = 0;
    memset(m->keys, 0xFF, new_cap * sizeof(uint64_t));
    uint32_t mask = new_cap - 1;
    for (uint32_t i = 0; i < oc; i++) {
        if (ok[i] != LZG_HM_EMPTY && ok[i] != LZG_HM_DELETED) {
            uint32_t idx = (uint32_t)(ok[i] & mask);
            while (m->keys[idx] != LZG_HM_EMPTY) idx = (idx + 1) & mask;
            m->keys[idx] = ok[i];
            m->values[idx] = ov[i];
            m->count++;
        }
    }
    free(ok); free(ov);
}

bool lzg_hm_put(LZGHashMap *m, uint64_t key, uint64_t value) {
    if ((m->count + m->tombstones) * 5 > m->capacity * 3)
        hm_resize(m, m->capacity * 2);
    uint32_t mask = m->capacity - 1;
    uint32_t idx = (uint32_t)(key & mask);
    uint32_t tomb = UINT32_MAX;
    while (1) {
        uint64_t k = m->keys[idx];
        if (k == key)       { m->values[idx] = value; return false; }
        if (k == LZG_HM_EMPTY) {
            if (tomb != UINT32_MAX) { idx = tomb; m->tombstones--; }
            m->keys[idx] = key; m->values[idx] = value; m->count++;
            return true;
        }
        if (k == LZG_HM_DELETED && tomb == UINT32_MAX) tomb = idx;
        idx = (idx + 1) & mask;
    }
}

uint64_t *lzg_hm_get(const LZGHashMap *m, uint64_t key) {
    uint32_t mask = m->capacity - 1;
    uint32_t idx = (uint32_t)(key & mask);
    while (1) {
        uint64_t k = m->keys[idx];
        if (k == key)       return (uint64_t *)&m->values[idx];
        if (k == LZG_HM_EMPTY) return NULL;
        idx = (idx + 1) & mask;
    }
}

bool lzg_hm_delete(LZGHashMap *m, uint64_t key) {
    uint32_t mask = m->capacity - 1;
    uint32_t idx = (uint32_t)(key & mask);
    while (1) {
        uint64_t k = m->keys[idx];
        if (k == key) {
            m->keys[idx] = LZG_HM_DELETED;
            m->count--; m->tombstones++;
            return true;
        }
        if (k == LZG_HM_EMPTY) return false;
        idx = (idx + 1) & mask;
    }
}

void lzg_hm_clear(LZGHashMap *m) {
    memset(m->keys, 0xFF, m->capacity * sizeof(uint64_t));
    m->count = m->tombstones = 0;
}

uint64_t lzg_hash_bytes(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

uint64_t lzg_hash_str(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ULL; }
    return h;
}
