/**
 * @file lz76.c
 * @brief LZ76 decomposition and node encoding with @ / $ sentinels.
 *
 * Every sequence is wrapped as "@<sequence>$" before decomposition.
 * This ensures:
 *   - Single root node (@) for all walks
 *   - Every last token contains $ (always novel, no case-3 ambiguity)
 *   - No separate initial/terminal probability machinery needed
 */
#include "lzgraph/lz76.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>

#define LZG_WRAP_STACK_CAP 512u
#define LZG_LABEL_STACK_CAP 256u
/**
 * Wrap a sequence with start/end sentinels: "@" + seq + "$"
 * Uses caller-provided stack storage when possible and heap only as fallback.
 */
static char *wrap_sentinels(const char *str, uint32_t len, uint32_t *out_len,
                            char *stack_buf, size_t stack_cap, bool *used_heap) {
    *out_len = len + 2;
    size_t need = (size_t)(*out_len) + 1u;
    char *wrapped = NULL;
    if (need <= stack_cap) {
        wrapped = stack_buf;
        *used_heap = false;
    } else {
        wrapped = malloc(need);
        *used_heap = true;
    }
    if (!wrapped) return NULL;
    wrapped[0] = LZG_START_SENTINEL;
    memcpy(wrapped + 1, str, len);
    wrapped[len + 1] = LZG_END_SENTINEL;
    wrapped[len + 2] = '\0';
    return wrapped;
}

static inline uint32_t u32_decimal_len(uint32_t v) {
    uint32_t n = 1;
    while (v >= 10) {
        v /= 10;
        n++;
    }
    return n;
}

static uint32_t write_u32_decimal(char *dst, uint32_t v) {
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

static char *acquire_label_buf(uint32_t need, char *stack_buf, size_t stack_cap,
                               bool *used_heap) {
    size_t total = (size_t)need + 1u;
    if (total <= stack_cap) {
        *used_heap = false;
        return stack_buf;
    }
    *used_heap = true;
    return malloc(total);
}

LZGError lzg_lz76_decompose(const char *str, uint32_t len,
                         LZGStringPool *pool, LZGTokens *out) {
    if (!str || !pool || !out) return LZG_ERR_INVALID_ARG;
    out->count = 0;
    if (len == 0) return LZG_OK;

    /*
     * Dictionary tracks which substrings have been emitted as tokens.
     * We store FNV-1a hashes in a hash set (value unused, just presence).
     *
     * The LZ76 algorithm:
     *   1. At current position, find the longest prefix in the dictionary.
     *   2. If no match: emit the single character (new to dictionary).
     *   3. If match of length m and more chars remain: emit match + next char.
     *   4. If match but at end of string: emit just the match.
     *   5. Add the new token to the dictionary.
     */
    uint64_t dict_hashes[LZG_MAX_TOKENS];
    uint32_t dict_count = 0;

    uint32_t pos = 0;

    while (pos < len) {
        /* Find longest match in dictionary */
        uint32_t match_len = 0;
        for (uint32_t try_len = 1; try_len <= len - pos; try_len++) {
            uint64_t h = lzg_hash_bytes(str + pos, try_len);
            bool found = false;
            for (uint32_t i = 0; i < dict_count; i++) {
                if (dict_hashes[i] == h) {
                    found = true;
                    break;
                }
            }
            if (found) {
                match_len = try_len;
            } else {
                break; /* dictionary is prefix-closed by construction */
            }
        }

        /* Determine token length */
        uint32_t tok_len;
        if (match_len == 0) {
            tok_len = 1;                         /* new single character */
        } else if (pos + match_len < len) {
            tok_len = match_len + 1;             /* extend match by 1 */
        } else {
            tok_len = match_len;                 /* end of string */
        }

        /* Intern the token and record it */
        if (out->count >= LZG_MAX_TOKENS) {
            return LZG_FAIL(LZG_ERR_OVERFLOW, "sequence produces >%d LZ76 tokens (max %d)", out->count, LZG_MAX_TOKENS);
        }

        uint32_t sp_id = lzg_sp_intern_n(pool, str + pos, tok_len);
        pos += tok_len;

        out->sp_ids[out->count]    = sp_id;
        out->positions[out->count] = pos; /* cumulative, 1-indexed */
        out->count++;

        /* Add token to dictionary */
        dict_hashes[dict_count++] = lzg_hash_bytes(lzg_sp_get(pool, sp_id), tok_len);
    }

    return LZG_OK;
}

LZGError lzg_lz76_encode_aap(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids,
                          uint32_t *out_sp_ids,
                          uint32_t *out_count) {
    if (!str || !pool || !out_ids || !out_count) return LZG_ERR_INVALID_ARG;

    /* Wrap with sentinels: "@<sequence>$" */
    uint32_t wlen;
    char wrapped_stack[LZG_WRAP_STACK_CAP];
    bool wrapped_heap = false;
    char *wrapped = wrap_sentinels(str, len, &wlen,
                                   wrapped_stack, sizeof(wrapped_stack),
                                   &wrapped_heap);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    if (wrapped_heap) free(wrapped);
    if (err != LZG_OK) return err;

    /* Build node labels: "{subpattern}_{position}" */
    char label_stack[LZG_LABEL_STACK_CAP];
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        uint32_t sp_len = lzg_sp_len(pool, tokens.sp_ids[i]);
        uint32_t pos_len = u32_decimal_len(tokens.positions[i]);
        uint32_t label_len = sp_len + 1u + pos_len;
        bool label_heap = false;
        char *label_buf = acquire_label_buf(label_len, label_stack,
                                            sizeof(label_stack), &label_heap);
        if (!label_buf) return LZG_ERR_ALLOC;
        memcpy(label_buf, sp, sp_len);
        label_buf[sp_len] = '_';
        (void)write_u32_decimal(label_buf + sp_len + 1u, tokens.positions[i]);
        label_buf[label_len] = '\0';
        out_ids[i] = lzg_sp_intern_n(pool, label_buf, label_len);
        if (label_heap) free(label_buf);
        if (out_sp_ids)
            out_sp_ids[i] = tokens.sp_ids[i];
    }
    *out_count = tokens.count;
    return LZG_OK;
}

/*
 * NDP encoding: "{subpattern}{reading_frame}_{cumulative_position}"
 * Reading frame = start_position_of_token % 3
 * Start position = cumulative_position - token_length
 */
LZGError lzg_lz76_encode_ndp(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids,
                          uint32_t *out_sp_ids,
                          uint32_t *out_count) {
    if (!str || !pool || !out_ids || !out_count) return LZG_ERR_INVALID_ARG;

    uint32_t wlen;
    char wrapped_stack[LZG_WRAP_STACK_CAP];
    bool wrapped_heap = false;
    char *wrapped = wrap_sentinels(str, len, &wlen,
                                   wrapped_stack, sizeof(wrapped_stack),
                                   &wrapped_heap);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    if (wrapped_heap) free(wrapped);
    if (err != LZG_OK) return err;

    char label_stack[LZG_LABEL_STACK_CAP];
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        uint32_t sp_len = lzg_sp_len(pool, tokens.sp_ids[i]);
        uint32_t end_pos = tokens.positions[i];
        uint32_t start_pos = end_pos - sp_len;
        uint32_t frame = start_pos % 3;
        uint32_t end_pos_len = u32_decimal_len(end_pos);
        uint32_t label_len = sp_len + 1u + 1u + end_pos_len;
        bool label_heap = false;
        char *label_buf = acquire_label_buf(label_len, label_stack,
                                            sizeof(label_stack), &label_heap);
        if (!label_buf) return LZG_ERR_ALLOC;
        memcpy(label_buf, sp, sp_len);
        label_buf[sp_len] = (char)('0' + frame);
        label_buf[sp_len + 1u] = '_';
        (void)write_u32_decimal(label_buf + sp_len + 2u, end_pos);
        label_buf[label_len] = '\0';
        out_ids[i] = lzg_sp_intern_n(pool, label_buf, label_len);
        if (label_heap) free(label_buf);
        if (out_sp_ids)
            out_sp_ids[i] = tokens.sp_ids[i];
    }
    *out_count = tokens.count;
    return LZG_OK;
}

/*
 * Naive encoding: just the raw subpattern, no position or frame.
 */
LZGError lzg_lz76_encode_naive(const char *str, uint32_t len,
                            LZGStringPool *pool,
                            uint32_t *out_ids,
                            uint32_t *out_sp_ids,
                            uint32_t *out_count) {
    if (!str || !pool || !out_ids || !out_count) return LZG_ERR_INVALID_ARG;

    uint32_t wlen;
    char wrapped_stack[LZG_WRAP_STACK_CAP];
    bool wrapped_heap = false;
    char *wrapped = wrap_sentinels(str, len, &wlen,
                                   wrapped_stack, sizeof(wrapped_stack),
                                   &wrapped_heap);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    if (wrapped_heap) free(wrapped);
    if (err != LZG_OK) return err;

    /* Node labels ARE the subpatterns — no formatting needed */
    for (uint32_t i = 0; i < tokens.count; i++) {
        out_ids[i] = tokens.sp_ids[i]; /* label = subpattern */
        if (out_sp_ids)
            out_sp_ids[i] = tokens.sp_ids[i];
    }
    *out_count = tokens.count;
    return LZG_OK;
}

/*
 * Generic dispatcher based on variant.
 */
LZGError lzg_lz76_encode(const char *str, uint32_t len,
                      LZGStringPool *pool, LZGVariant variant,
                      uint32_t *out_ids, uint32_t *out_sp_ids,
                      uint32_t *out_count) {
    switch (variant) {
        case LZG_VARIANT_AAP:
            return lzg_lz76_encode_aap(str, len, pool, out_ids, out_sp_ids, out_count);
        case LZG_VARIANT_NDP:
            return lzg_lz76_encode_ndp(str, len, pool, out_ids, out_sp_ids, out_count);
        default: /* LZG_VARIANT_NAIVE */
            return lzg_lz76_encode_naive(str, len, pool, out_ids, out_sp_ids, out_count);
    }
}
