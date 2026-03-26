/**
 * @file lz76.h
 * @brief LZ76 (Lempel-Ziv 1976) sequence decomposition.
 *
 * Decomposes a string into the shortest sequence of subpatterns where
 * each new subpattern extends a previously seen one by exactly one
 * character.
 *
 * Example: "CASSLGIRRT" → ["C","A","S","SL","G","I","R","RT"]
 *          positions:       [1,  2,  3,  5,   6,  7,  8,  10]
 */
#ifndef LZGRAPH_LZ76_H
#define LZGRAPH_LZ76_H

#include "lzgraph/common.h"
#include "lzgraph/string_pool.h"

/* Maximum tokens per decomposition (CDR3 seqs are ≤100 chars) */
#define LZG_MAX_TOKENS 128

/**
 * Result of an LZ76 decomposition (stack-friendly, fixed-capacity).
 *
 * Tokens are stored as interned string IDs in `sp_ids[]`.
 * Cumulative character positions (1-indexed) are in `positions[]`.
 */
typedef struct {
    uint32_t sp_ids[LZG_MAX_TOKENS];    /* interned subpattern IDs    */
    uint32_t positions[LZG_MAX_TOKENS]; /* cumulative char positions  */
    uint32_t count;                      /* number of tokens           */
} LZGTokens;

/**
 * Decompose a string into LZ76 tokens.
 *
 * @param str   Input string.
 * @param len   Length of the string.
 * @param pool  String pool for interning subpattern strings.
 * @param out   Output: filled with interned token IDs and positions.
 * @return LZG_OK on success.
 */
LZGError lzg_lz76_decompose(const char *str, uint32_t len,
                         LZGStringPool *pool, LZGTokens *out);

/**
 * Encode a sequence into AAPLZGraph node labels.
 *
 * Each node label is "{subpattern}_{cumulative_position}".
 * Node labels are interned in `pool`.
 *
 * @param str       Amino acid sequence.
 * @param len       Length.
 * @param pool      String pool for interning node labels.
 * @param out_ids   Output: interned node label IDs (caller provides,
 *                  must have room for LZG_MAX_TOKENS entries).
 * @param out_sp_ids Output: interned subpattern IDs (parallel to out_ids).
 * @param out_count Output: number of nodes.
 * @return LZG_OK on success.
 */
LZGError lzg_lz76_encode_aap(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids,
                          uint32_t *out_sp_ids,
                          uint32_t *out_count);

/**
 * Encode a sequence into NDPLZGraph node labels.
 *
 * Each node label is "{subpattern}{reading_frame}_{cumulative_position}".
 * Reading frame = (start_position_of_token) % 3.
 * Example: "ATG" at position 3 with frame 0 → "ATG0_3"
 */
LZGError lzg_lz76_encode_ndp(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids,
                          uint32_t *out_sp_ids,
                          uint32_t *out_count);

/**
 * Encode a sequence into NaiveLZGraph node labels.
 *
 * Each node label is just the raw subpattern (no position, no frame).
 * Example: "ATG" → node "ATG"
 */
LZGError lzg_lz76_encode_naive(const char *str, uint32_t len,
                            LZGStringPool *pool,
                            uint32_t *out_ids,
                            uint32_t *out_sp_ids,
                            uint32_t *out_count);

/**
 * Generic encode: dispatches to AAP, NDP, or Naive based on variant.
 */
LZGError lzg_lz76_encode(const char *str, uint32_t len,
                      LZGStringPool *pool, LZGVariant variant,
                      uint32_t *out_ids, uint32_t *out_sp_ids,
                      uint32_t *out_count);

#endif /* LZGRAPH_LZ76_H */
