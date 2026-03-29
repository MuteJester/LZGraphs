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

static inline uint64_t hm_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static uint32_t hm_find_existing(const LZGHashMap *m, uint64_t key) {
    uint32_t mask = m->capacity - 1;
    uint32_t idx = (uint32_t)(hm_mix64(key) & mask);
    while (1) {
        uint64_t k = m->keys[idx];
        if (k == key) return idx;
        if (k == LZG_HM_EMPTY) return UINT32_MAX;
        idx = (idx + 1) & mask;
    }
}

static uint32_t hm_find_insert_slot(const LZGHashMap *m, uint64_t key, bool *found) {
    uint32_t mask = m->capacity - 1;
    uint32_t idx = (uint32_t)(hm_mix64(key) & mask);
    uint32_t tomb = UINT32_MAX;
    while (1) {
        uint64_t k = m->keys[idx];
        if (k == key) {
            *found = true;
            return idx;
        }
        if (k == LZG_HM_EMPTY) {
            *found = false;
            return (tomb != UINT32_MAX) ? tomb : idx;
        }
        if (k == LZG_HM_DELETED && tomb == UINT32_MAX) tomb = idx;
        idx = (idx + 1) & mask;
    }
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
    for (uint32_t i = 0; i < oc; i++) {
        if (ok[i] != LZG_HM_EMPTY && ok[i] != LZG_HM_DELETED) {
            uint32_t mask = new_cap - 1;
            uint32_t idx = (uint32_t)(hm_mix64(ok[i]) & mask);
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
    bool found = false;
    uint32_t idx = hm_find_insert_slot(m, key, &found);
    if (found) {
        m->values[idx] = value;
        return false;
    }
    if (m->keys[idx] == LZG_HM_DELETED) m->tombstones--;
    m->keys[idx] = key;
    m->values[idx] = value;
    m->count++;
    return true;
}

uint64_t *lzg_hm_get(const LZGHashMap *m, uint64_t key) {
    uint32_t idx = hm_find_existing(m, key);
    if (idx == UINT32_MAX) return NULL;
    return (uint64_t *)&m->values[idx];
}

uint64_t *lzg_hm_get_or_insert(LZGHashMap *m, uint64_t key,
                               uint64_t initial_value, bool *inserted) {
    if ((m->count + m->tombstones) * 5 > m->capacity * 3)
        hm_resize(m, m->capacity * 2);

    bool found = false;
    uint32_t idx = hm_find_insert_slot(m, key, &found);
    if (found) {
        if (inserted) *inserted = false;
        return &m->values[idx];
    }

    if (m->keys[idx] == LZG_HM_DELETED) m->tombstones--;
    m->keys[idx] = key;
    m->values[idx] = initial_value;
    m->count++;
    if (inserted) *inserted = true;
    return &m->values[idx];
}

uint64_t *lzg_hm_add_u64(LZGHashMap *m, uint64_t key,
                         uint64_t delta, bool *inserted) {
    bool created = false;
    uint64_t *slot = lzg_hm_get_or_insert(m, key, delta, &created);
    if (!created) *slot += delta;
    if (inserted) *inserted = created;
    return slot;
}

bool lzg_hm_delete(LZGHashMap *m, uint64_t key) {
    uint32_t idx = hm_find_existing(m, key);
    if (idx == UINT32_MAX) return false;
    m->keys[idx] = LZG_HM_DELETED;
    m->count--;
    m->tombstones++;
    return true;
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
