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

/**
 * Per-walk LZ76 dictionary state.
 *
 * Created at walk start, updated O(1) per step, destroyed at walk end.
 * Memory: O(walk_length) — typically 5-15 entries.
 */
typedef struct {
    LZGHashMap *tokens;          /* hash set of emitted token hashes      */
    uint32_t    single_char_bits;/* bitmask: chars emitted as single-char */
} LZGWalkDict;

/** Create a walk dictionary. */
static inline LZGWalkDict lzg_wd_create(void) {
    LZGWalkDict wd;
    wd.tokens = lzg_hm_create(32);
    wd.single_char_bits = 0;
    return wd;
}

/** Destroy a walk dictionary. */
static inline void lzg_wd_destroy(LZGWalkDict *wd) {
    if (wd->tokens) { lzg_hm_destroy(wd->tokens); wd->tokens = NULL; }
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
    lzg_hm_put(wd->tokens, h, 1);
    if (sp_len == 1)
        wd->single_char_bits |= (1u << lzg_aa_to_bit(sp[0]));
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
    const char *sp = lzg_sp_get(g->pool, g->edge_sp_id[edge]);

    if (sp_len == 1) {
        return !(wd->single_char_bits & (1u << lzg_aa_to_bit(sp[0])));
    } else {
        const char *prefix = lzg_sp_get(g->pool, g->edge_prefix_id[edge]);
        uint32_t prefix_len = sp_len - 1;
        uint64_t prefix_h = lzg_hash_bytes(prefix, prefix_len);
        if (!lzg_hm_get(wd->tokens, prefix_h))
            return false;

        uint64_t token_h = lzg_hash_bytes(sp, sp_len);
        if (lzg_hm_get(wd->tokens, token_h))
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
    lzg_wd_record(wd, g->pool, g->node_sp_id[node_id], g->node_sp_len[node_id]);
}

/**
 * Record an edge's destination subpattern into the walk dictionary.
 */
static inline void lzg_wd_record_edge(LZGWalkDict *wd, const LZGGraph *g,
                                       uint32_t edge) {
    lzg_wd_record(wd, g->pool, g->edge_sp_id[edge], g->edge_sp_len[edge]);
}

#endif /* LZGRAPH_WALK_DICT_H */
