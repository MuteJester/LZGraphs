/**
 * @file walk_dict.h
 * @brief Per-walk LZ76 dictionary for dynamic constraint checking.
 *
 * Replaces the precomputed bitmask live index for simulation and walk
 * probability. Each walk carries a lightweight hash dictionary that
 * tracks which tokens have been emitted, enabling EXACT LZ76 constraint
 * enforcement at O(degree) per step with no global precomputation.
 *
 * Correctness: The bitmask approach was an approximation —
 *   - Single-char: tracked "character in ANY token" (too restrictive)
 *   - Multi-char: tracked "first char of prefix seen" (too permissive)
 *
 * The walk dictionary checks the exact LZ76 conditions:
 *   - Single-char token "X": X was NOT a previous standalone token
 *   - Multi-char token "XY...Z": prefix "XY..." WAS a previous token
 */
#ifndef LZGRAPH_WALK_DICT_H
#define LZGRAPH_WALK_DICT_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"
#include <string.h>

#define LZG_WD_INLINE_CAP 256u

/**
 * Per-walk LZ76 dictionary state.
 *
 * Uses a small exact fixed-capacity token set in the hot path.
 * If an unexpected walk exceeds that bound, we promote once into an
 * overflow hash map and keep exact behavior.
 */
typedef struct {
    uint64_t    inline_keys[LZG_WD_INLINE_CAP];
    uint8_t     inline_used[LZG_WD_INLINE_CAP];
    uint16_t    inline_count;
    uint32_t    single_char_bits; /* bitmask: chars emitted as single-char */
    LZGHashMap *overflow;         /* exact fallback if inline set fills     */
} LZGWalkDict;

static inline bool lzg_wd_inline_contains(const LZGWalkDict *wd, uint64_t key) {
    uint32_t mask = LZG_WD_INLINE_CAP - 1u;
    uint32_t idx = (uint32_t)(key & mask);
    while (wd->inline_used[idx]) {
        if (wd->inline_keys[idx] == key) return true;
        idx = (idx + 1u) & mask;
    }
    return false;
}

static inline bool lzg_wd_inline_insert(LZGWalkDict *wd, uint64_t key) {
    /* Keep probe chains short; normal walks never approach this bound. */
    if ((uint32_t)wd->inline_count * 4u >= LZG_WD_INLINE_CAP * 3u)
        return false;

    uint32_t mask = LZG_WD_INLINE_CAP - 1u;
    uint32_t idx = (uint32_t)(key & mask);
    while (wd->inline_used[idx]) {
        if (wd->inline_keys[idx] == key) return true;
        idx = (idx + 1u) & mask;
    }
    wd->inline_used[idx] = 1u;
    wd->inline_keys[idx] = key;
    wd->inline_count++;
    return true;
}

static inline bool lzg_wd_promote_overflow(LZGWalkDict *wd) {
    if (!wd->overflow) {
        wd->overflow = lzg_hm_create(LZG_WD_INLINE_CAP * 2u);
        if (!wd->overflow) return false;
    } else {
        lzg_hm_clear(wd->overflow);
    }

    for (uint32_t i = 0; i < LZG_WD_INLINE_CAP; i++) {
        if (!wd->inline_used[i]) continue;
        lzg_hm_put(wd->overflow, wd->inline_keys[i], 1u);
    }

    memset(wd->inline_used, 0, sizeof(wd->inline_used));
    wd->inline_count = 0;
    return true;
}

static inline bool lzg_wd_contains_hash(const LZGWalkDict *wd, uint64_t key) {
    if (wd->overflow) return lzg_hm_get(wd->overflow, key) != NULL;
    return lzg_wd_inline_contains(wd, key);
}

static inline bool lzg_wd_record_hash(LZGWalkDict *wd, uint64_t key) {
    if (wd->overflow) {
        lzg_hm_put(wd->overflow, key, 1u);
        return true;
    }
    if (lzg_wd_inline_insert(wd, key)) return true;
    if (!lzg_wd_promote_overflow(wd)) return false;
    lzg_hm_put(wd->overflow, key, 1u);
    return true;
}

static inline bool lzg_wd_record_prehashed(LZGWalkDict *wd, uint64_t key,
                                           uint8_t single_char_idx) {
    if (!lzg_wd_record_hash(wd, key)) return false;
    if (single_char_idx != UINT8_MAX)
        wd->single_char_bits |= (1u << single_char_idx);
    return true;
}

/** Create a walk dictionary. */
static inline LZGWalkDict lzg_wd_create(void) {
    LZGWalkDict wd;
    memset(&wd, 0, sizeof(wd));
    return wd;
}

/** Reset a walk dictionary for reuse. */
static inline void lzg_wd_reset(LZGWalkDict *wd) {
    memset(wd->inline_used, 0, sizeof(wd->inline_used));
    wd->inline_count = 0;
    wd->single_char_bits = 0;
    if (wd->overflow) lzg_hm_clear(wd->overflow);
}

/** Destroy a walk dictionary. */
static inline void lzg_wd_destroy(LZGWalkDict *wd) {
    if (wd->overflow) {
        lzg_hm_destroy(wd->overflow);
        wd->overflow = NULL;
    }
}

/**
 * Record a token in the walk dictionary.
 *
 * @param wd      Walk dictionary.
 * @param pool    String pool containing the token.
 * @param sp_id   Interned ID of the token string.
 * @param sp_len  Length of the token.
 */
static inline void lzg_wd_record(LZGWalkDict *wd, const LZGStringPool *pool,
                                  uint32_t sp_id, uint8_t sp_len) {
    const char *sp = lzg_sp_get(pool, sp_id);
    uint64_t h = lzg_hash_bytes(sp, sp_len);
    uint8_t single_char_idx = (sp_len == 1) ? lzg_aa_to_bit(sp[0]) : UINT8_MAX;
    (void)lzg_wd_record_prehashed(wd, h, single_char_idx);
}

/**
 * Check if an edge is LZ76-valid given the current walk dictionary.
 *
 * @param g     The graph.
 * @param edge  Edge index.
 * @param wd    Current walk dictionary state.
 * @return true if the edge's token is a valid next LZ76 token.
 */
/**
 * Check if an edge is LZ76-valid for a CONTINUING walk (not the last step).
 * Rules:
 *   - Single-char "c": c must NOT be a previous standalone token
 *   - Multi-char "wc": prefix "w" IN dictionary AND "wc" NOT in dictionary
 */
static inline bool lzg_wd_edge_valid(const LZGGraph *g, uint32_t edge,
                                      const LZGWalkDict *wd) {
    uint8_t sp_len = g->edge_sp_len[edge];

    if (sp_len == 1) {
        uint8_t idx = g->edge_single_char_idx ? g->edge_single_char_idx[edge] : UINT8_MAX;
        if (idx != UINT8_MAX)
            return !(wd->single_char_bits & (1u << idx));
        const char *sp = lzg_sp_get(g->pool, g->edge_sp_id[edge]);
        return !(wd->single_char_bits & (1u << lzg_aa_to_bit(sp[0])));
    } else {
        uint64_t prefix_h;
        if (g->edge_prefix_hash) {
            prefix_h = g->edge_prefix_hash[edge];
        } else {
            const char *prefix = lzg_sp_get(g->pool, g->edge_prefix_id[edge]);
            uint32_t prefix_len = sp_len - 1;
            prefix_h = lzg_hash_bytes(prefix, prefix_len);
        }
        if (!lzg_wd_contains_hash(wd, prefix_h))
            return false;

        uint64_t token_h;
        if (g->edge_sp_hash) {
            token_h = g->edge_sp_hash[edge];
        } else {
            const char *sp = lzg_sp_get(g->pool, g->edge_sp_id[edge]);
            token_h = lzg_hash_bytes(sp, sp_len);
        }
        if (lzg_wd_contains_hash(wd, token_h))
            return false;

        return true;
    }
}

/* lzg_wd_edge_valid_terminal removed: with $ sentinels, every last token
 * is guaranteed novel (contains $), so case-3 exception is eliminated. */

/**
 * Record a node's subpattern into the walk dictionary.
 * Convenience for recording the initial node of a walk.
 */
static inline void lzg_wd_record_node(LZGWalkDict *wd, const LZGGraph *g,
                                       uint32_t node_id) {
    if (g->node_sp_hash && g->node_single_char_idx) {
        (void)lzg_wd_record_prehashed(wd, g->node_sp_hash[node_id],
                                      g->node_single_char_idx[node_id]);
    } else {
        lzg_wd_record(wd, g->pool, g->node_sp_id[node_id], g->node_sp_len[node_id]);
    }
}

/**
 * Record an edge's destination subpattern into the walk dictionary.
 */
static inline void lzg_wd_record_edge(LZGWalkDict *wd, const LZGGraph *g,
                                       uint32_t edge) {
    if (g->edge_sp_hash && g->edge_single_char_idx) {
        (void)lzg_wd_record_prehashed(wd, g->edge_sp_hash[edge],
                                      g->edge_single_char_idx[edge]);
    } else {
        lzg_wd_record(wd, g->pool, g->edge_sp_id[edge], g->edge_sp_len[edge]);
    }
}

#endif /* LZGRAPH_WALK_DICT_H */
