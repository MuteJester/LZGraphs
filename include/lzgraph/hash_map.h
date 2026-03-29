/**
 * @file hash_map.h
 * @brief Open-addressing hash map (uint64 keys → uint64 values).
 */
#ifndef LZGRAPH_HASH_MAP_H
#define LZGRAPH_HASH_MAP_H

#include "lzgraph/common.h"

#define LZG_HM_EMPTY    UINT64_MAX
#define LZG_HM_DELETED  (UINT64_MAX - 1)

typedef struct {
    uint64_t *keys;
    uint64_t *values;
    uint32_t  capacity;   /* always a power of 2 */
    uint32_t  count;
    uint32_t  tombstones;
} LZGHashMap;

LZGHashMap *lzg_hm_create(uint32_t initial_capacity);
void        lzg_hm_destroy(LZGHashMap *map);

/** Insert or update. Returns true if new key. */
bool        lzg_hm_put(LZGHashMap *map, uint64_t key, uint64_t value);

/** Lookup. Returns pointer to value or NULL. */
uint64_t   *lzg_hm_get(const LZGHashMap *map, uint64_t key);

/**
 * Lookup or insert in a single probe walk. Returns a writable value slot.
 * If the key is absent, inserts it with `initial_value`.
 * If `inserted` is non-NULL, it is set to true iff a new entry was created.
 */
uint64_t   *lzg_hm_get_or_insert(LZGHashMap *map, uint64_t key,
                                 uint64_t initial_value, bool *inserted);

/**
 * Add `delta` to the value stored at `key`, inserting `delta` if absent.
 * Returns a writable value slot after the update.
 * If `inserted` is non-NULL, it is set to true iff a new entry was created.
 */
uint64_t   *lzg_hm_add_u64(LZGHashMap *map, uint64_t key,
                           uint64_t delta, bool *inserted);

/** Delete. Returns true if found. */
bool        lzg_hm_delete(LZGHashMap *map, uint64_t key);

/** Clear all entries (keeps allocated memory). */
void        lzg_hm_clear(LZGHashMap *map);

/** FNV-1a hash for byte arrays. */
uint64_t    lzg_hash_bytes(const void *data, size_t len);

/** FNV-1a hash for null-terminated strings. */
uint64_t    lzg_hash_str(const char *str);

#endif /* LZGRAPH_HASH_MAP_H */
