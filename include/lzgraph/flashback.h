/**
 * @file flashback.h
 * @brief FlashBack sequence decomposition.
 *
 * Decomposes a string by recursively peeling matched runs from both
 * ends of the sentinel-wrapped sequence "@<seq>$". Each token captures
 * the leading run and trailing run together with a split position.
 *
 * Token format: "{front}{back}_{split_pos}{discovery_order}"
 *
 * Example: "CASSAYFF" →
 *   ["@$_1{0}", "CFF_1{1}", "AY_1{2}", "SSA_0{3}"]
 *
 * The decomposition is fully reversible via rev_flashback().
 */
#ifndef LZGRAPH_FLASHBACK_H
#define LZGRAPH_FLASHBACK_H

#include "lzgraph/common.h"
#include "lzgraph/string_pool.h"

/* Max tokens per decomposition */
#define LZG_FLASHBACK_MAX_TOKENS 128

/**
 * Result of a FlashBack decomposition (stack-friendly, fixed-capacity).
 *
 * Tokens are stored as interned string IDs in `sp_ids[]`.
 */
typedef struct {
    uint32_t sp_ids[LZG_FLASHBACK_MAX_TOKENS]; /* interned token IDs */
    uint32_t count;                             /* number of tokens   */
} LZGFlashbackTokens;

/**
 * Decompose a string using the FlashBack method.
 *
 * The string is wrapped with sentinels ("@<str>$") internally.
 *
 * @param str   Input string (without sentinels).
 * @param len   Length of the string.
 * @param pool  String pool for interning token strings.
 * @param out   Output: filled with interned token IDs.
 * @return LZG_OK on success.
 */
LZGError lzg_flashback_decompose(const char *str, uint32_t len,
                                 LZGStringPool *pool,
                                 LZGFlashbackTokens *out);

/**
 * Reverse a FlashBack decomposition back to the original string.
 *
 * @param pool      String pool containing the token strings.
 * @param tokens    Token IDs from a prior decomposition.
 * @param out_buf   Output buffer (caller provides, must be large enough).
 * @param buf_cap   Capacity of out_buf.
 * @param out_len   Output: length of reconstructed string.
 * @return LZG_OK on success.
 */
LZGError lzg_flashback_reverse(const LZGStringPool *pool,
                               const LZGFlashbackTokens *tokens,
                               char *out_buf, uint32_t buf_cap,
                               uint32_t *out_len);

#endif /* LZGRAPH_FLASHBACK_H */
