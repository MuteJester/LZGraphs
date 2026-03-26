/**
 * @file string_pool.h
 * @brief String interning: maps unique strings to sequential uint32 IDs.
 */
#ifndef LZGRAPH_STRING_POOL_H
#define LZGRAPH_STRING_POOL_H

#include "lzgraph/common.h"

#define LZG_SP_NOT_FOUND UINT32_MAX

typedef struct {
    char     **strings;    /* owned copies, indexed by ID       */
    uint32_t  *str_lens;   /* length of each string             */
    uint32_t   count;      /* number of interned strings        */
    uint32_t   capacity;   /* allocated slots                   */

    /* Hash table: open addressing, linear probing */
    uint64_t  *ht_hashes;
    uint32_t  *ht_ids;     /* LZG_SP_NOT_FOUND = empty slot     */
    uint32_t   ht_cap;     /* always a power of 2               */
} LZGStringPool;

LZGStringPool *lzg_sp_create(uint32_t initial_capacity);
void           lzg_sp_destroy(LZGStringPool *pool);

/** Intern a null-terminated string. Returns its ID (existing or new). */
uint32_t       lzg_sp_intern(LZGStringPool *pool, const char *str);

/** Intern a string with explicit length. */
uint32_t       lzg_sp_intern_n(LZGStringPool *pool, const char *str, uint32_t len);

/** Find without interning. Returns LZG_SP_NOT_FOUND if absent. */
uint32_t       lzg_sp_find(const LZGStringPool *pool, const char *str);

/** Get string by ID. */
static inline const char *lzg_sp_get(const LZGStringPool *pool, uint32_t id) {
    return pool->strings[id];
}

/** Get string length by ID. */
static inline uint32_t lzg_sp_len(const LZGStringPool *pool, uint32_t id) {
    return pool->str_lens[id];
}

#endif /* LZGRAPH_STRING_POOL_H */
