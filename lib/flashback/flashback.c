/**
 * @file flashback.c
 * @brief FlashBack decomposition and reversal.
 *
 * Decompose: recursively peel the leading and trailing runs of identical
 * characters from "@<seq>$", emitting one token per recursion level.
 * Zero-copy — operates on indices into the wrapped string.
 *
 * Reverse: recursively nest each token's symbols around the inner
 * result, reconstructing the original sequence.
 */
#include <stdlib.h>
#include <string.h>

#include "lzgraph/flashback.h"

#define LZG_FLASHBACK_WRAP_STACK_CAP 512u
#define LZG_FLASHBACK_TOKEN_BUF_CAP  256u

static uint32_t write_u32(char *dst, uint32_t v) {
    char tmp[10];
    uint32_t n = 0;
    do {
        tmp[n++] = (char)('0' + (v % 10));
        v /= 10;
    } while (v > 0);
    for (uint32_t i = 0; i < n; i++)
        dst[i] = tmp[n - 1 - i];
    return n;
}

/**
 * Emit one token: front[lo..lo+flen) + back[bstart..hi) + "_" + split + "{" + order + "}"
 * If `is_last` is true, split is forced to 0.
 */
static LZGError emit_token(const char *seq, uint32_t lo, uint32_t flen,
                           uint32_t bstart, uint32_t hi, uint32_t order,
                           bool is_last,
                           LZGStringPool *pool, LZGFlashbackTokens *out) {
    if (out->count >= LZG_FLASHBACK_MAX_TOKENS)
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "FlashBack: >%d tokens", LZG_FLASHBACK_MAX_TOKENS);

    char buf[LZG_FLASHBACK_TOKEN_BUF_CAP];
    uint32_t p = 0;
    uint32_t back_len = hi - bstart;
    uint32_t tok_chars = flen + back_len;

    if (tok_chars + 20 > LZG_FLASHBACK_TOKEN_BUF_CAP)
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "FlashBack: token too long (%u)", tok_chars);

    memcpy(buf + p, seq + lo, flen);
    p += flen;
    memcpy(buf + p, seq + bstart, back_len);
    p += back_len;
    buf[p++] = '_';
    p += write_u32(buf + p, is_last ? 0 : flen);
    buf[p++] = '{';
    p += write_u32(buf + p, order);
    buf[p++] = '}';
    buf[p] = '\0';

    out->sp_ids[out->count++] = lzg_sp_intern_n(pool, buf, p);
    return LZG_OK;
}

/**
 * Recursive decomposition on seq[lo..hi).
 *
 * At each level:
 *   - Find the leading run of seq[lo] (length flen)
 *   - Find the trailing run of seq[hi-1] (starts at bstart)
 *   - Emit token combining front + back
 *   - Recurse on the middle: seq[lo+flen .. bstart)
 */
static LZGError decompose_rec(const char *seq, uint32_t lo, uint32_t hi,
                               uint32_t order, LZGStringPool *pool,
                               LZGFlashbackTokens *out) {
    uint32_t span = hi - lo;
    if (span == 0) return LZG_OK;

    /* Single character */
    if (span == 1)
        return emit_token(seq, lo, 1, hi, hi, order, true, pool, out);

    /* Leading run */
    uint32_t flen = 1;
    while (lo + flen < hi && seq[lo + flen] == seq[lo])
        flen++;

    /* Entire span is one repeated character */
    if (flen == span)
        return emit_token(seq, lo, flen, hi, hi, order, true, pool, out);

    /* Trailing run */
    uint32_t bstart = hi - 1;
    while (bstart > lo && seq[bstart - 1] == seq[hi - 1])
        bstart--;

    /* Middle bounds */
    uint32_t mid_lo = lo + flen;
    uint32_t mid_hi = bstart;
    bool is_last = (mid_lo >= mid_hi);  /* no middle left → this is the last token */

    LZGError err = emit_token(seq, lo, flen, bstart, hi, order,
                              is_last, pool, out);
    if (err != LZG_OK) return err;

    if (is_last) return LZG_OK;

    return decompose_rec(seq, mid_lo, mid_hi, order + 1, pool, out);
}

LZGError lzg_flashback_decompose(const char *str, uint32_t len,
                                 LZGStringPool *pool,
                                 LZGFlashbackTokens *out) {
    if (!str || !pool || !out) return LZG_ERR_INVALID_ARG;
    out->count = 0;

    /* Wrap: "@" + str + "$" */
    uint32_t wlen = len + 2;
    char wrap_stack[LZG_FLASHBACK_WRAP_STACK_CAP];
    char *seq;
    bool wrap_heap = false;

    if ((size_t)(wlen + 1) <= sizeof(wrap_stack)) {
        seq = wrap_stack;
    } else {
        seq = malloc((size_t)(wlen + 1));
        if (!seq) return LZG_ERR_ALLOC;
        wrap_heap = true;
    }
    seq[0] = '@';
    memcpy(seq + 1, str, len);
    seq[len + 1] = '$';
    seq[wlen] = '\0';

    LZGError err = decompose_rec(seq, 0, wlen, 0, pool, out);

    if (wrap_heap) free(seq);
    return err;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Reverse: recursive nesting                                      */
/* ═══════════════════════════════════════════════════════════════ */

/**
 * Parse a token string "SYMBOLS_POS{ORDER}" into its parts.
 * Returns the symbol length and split position.
 */
static void parse_token(const char *tok, uint32_t *sym_len, uint32_t *split_pos) {
    const char *underscore = NULL;
    for (const char *p = tok; *p; p++) {
        if (*p == '_') underscore = p;
        if (*p == '{') break;
    }
    *sym_len = (uint32_t)(underscore - tok);
    *split_pos = 0;
    for (const char *p = underscore + 1; *p && *p != '{'; p++)
        *split_pos = *split_pos * 10 + (uint32_t)(*p - '0');
}

/**
 * Recursive reverse: process tokens[idx] wrapping around inner,
 * writing result to buf. Returns length written.
 *
 * reverse(idx) = symbols[0..pos) + reverse(idx+1) + symbols[pos..sym_len)
 * reverse(last) = symbols
 */
static LZGError reverse_rec(const LZGStringPool *pool,
                             const LZGFlashbackTokens *tokens,
                             uint32_t idx, char *buf, uint32_t buf_cap,
                             uint32_t *out_len) {
    const char *tok = lzg_sp_get(pool, tokens->sp_ids[idx]);
    uint32_t sym_len, split_pos;
    parse_token(tok, &sym_len, &split_pos);

    /* Base case: innermost token (last in array) */
    if (idx == tokens->count - 1) {
        if (sym_len >= buf_cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "FlashBack reverse: buffer too small");
        memcpy(buf, tok, sym_len);
        *out_len = sym_len;
        return LZG_OK;
    }

    /* Write front part: symbols[0..split_pos) */
    if (split_pos > 0)
        memcpy(buf, tok, split_pos);

    /* Recurse for the inner part */
    uint32_t inner_len = 0;
    LZGError err = reverse_rec(pool, tokens, idx + 1,
                               buf + split_pos, buf_cap - split_pos,
                               &inner_len);
    if (err != LZG_OK) return err;

    /* Write back part: symbols[split_pos..sym_len) */
    uint32_t back_len = sym_len - split_pos;
    uint32_t total = split_pos + inner_len + back_len;
    if (total >= buf_cap)
        return LZG_FAIL(LZG_ERR_OVERFLOW, "FlashBack reverse: buffer too small");

    if (back_len > 0)
        memcpy(buf + split_pos + inner_len, tok + split_pos, back_len);

    *out_len = total;
    return LZG_OK;
}

LZGError lzg_flashback_reverse(const LZGStringPool *pool,
                               const LZGFlashbackTokens *tokens,
                               char *out_buf, uint32_t buf_cap,
                               uint32_t *out_len) {
    if (!pool || !tokens || !out_buf || !out_len)
        return LZG_ERR_INVALID_ARG;

    if (tokens->count == 0) {
        *out_len = 0;
        out_buf[0] = '\0';
        return LZG_OK;
    }

    /* Reverse into a temp buffer (includes @ and $) */
    char *tmp = malloc(buf_cap);
    if (!tmp) return LZG_ERR_ALLOC;

    uint32_t raw_len = 0;
    LZGError err = reverse_rec(pool, tokens, 0, tmp, buf_cap, &raw_len);
    if (err != LZG_OK) {
        free(tmp);
        return err;
    }

    /* Strip sentinels @ and $ */
    uint32_t start = 0, end = raw_len;
    if (raw_len > 0 && tmp[0] == '@') start++;
    if (end > start && tmp[end - 1] == '$') end--;

    uint32_t result_len = end - start;
    if (result_len >= buf_cap) {
        free(tmp);
        return LZG_FAIL(LZG_ERR_OVERFLOW, "FlashBack reverse: buffer too small");
    }

    memcpy(out_buf, tmp + start, result_len);
    out_buf[result_len] = '\0';
    *out_len = result_len;

    free(tmp);
    return LZG_OK;
}
