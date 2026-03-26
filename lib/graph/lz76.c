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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Wrap a sequence with start/end sentinels: "@" + seq + "$"
 * Caller must free the returned buffer.
 */
static char *wrap_sentinels(const char *str, uint32_t len, uint32_t *out_len) {
    *out_len = len + 2;
    char *wrapped = malloc(*out_len + 1);
    if (!wrapped) return NULL;
    wrapped[0] = LZG_START_SENTINEL;
    memcpy(wrapped + 1, str, len);
    wrapped[len + 1] = LZG_END_SENTINEL;
    wrapped[len + 2] = '\0';
    return wrapped;
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
    LZGHashMap *dict = lzg_hm_create(64);
    if (!dict) return LZG_ERR_ALLOC;

    uint32_t pos = 0;

    while (pos < len) {
        /* Find longest match in dictionary */
        uint32_t match_len = 0;
        for (uint32_t try_len = 1; try_len <= len - pos; try_len++) {
            uint64_t h = lzg_hash_bytes(str + pos, try_len);
            if (lzg_hm_get(dict, h)) {
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
            lzg_hm_destroy(dict);
            return LZG_FAIL(LZG_ERR_OVERFLOW, "sequence produces >%d LZ76 tokens (max %d)", out->count, LZG_MAX_TOKENS);
        }

        uint32_t sp_id = lzg_sp_intern_n(pool, str + pos, tok_len);
        pos += tok_len;

        out->sp_ids[out->count]    = sp_id;
        out->positions[out->count] = pos; /* cumulative, 1-indexed */
        out->count++;

        /* Add token to dictionary */
        uint64_t h = lzg_hash_bytes(lzg_sp_get(pool, sp_id), tok_len);
        lzg_hm_put(dict, h, 1);
    }

    lzg_hm_destroy(dict);
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
    char *wrapped = wrap_sentinels(str, len, &wlen);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    free(wrapped);
    if (err != LZG_OK) return err;

    /* Build node labels: "{subpattern}_{position}" */
    char label_buf[256];
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        int label_len = snprintf(label_buf, sizeof(label_buf),
                                 "%s_%u", sp, tokens.positions[i]);
        out_ids[i] = lzg_sp_intern_n(pool, label_buf, (uint32_t)label_len);
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
    char *wrapped = wrap_sentinels(str, len, &wlen);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    free(wrapped);
    if (err != LZG_OK) return err;

    char label_buf[256];
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        uint32_t sp_len = lzg_sp_len(pool, tokens.sp_ids[i]);
        uint32_t end_pos = tokens.positions[i];
        uint32_t start_pos = end_pos - sp_len;
        uint32_t frame = start_pos % 3;

        int label_len = snprintf(label_buf, sizeof(label_buf),
                                 "%s%u_%u", sp, frame, end_pos);
        out_ids[i] = lzg_sp_intern_n(pool, label_buf, (uint32_t)label_len);
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
    char *wrapped = wrap_sentinels(str, len, &wlen);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, pool, &tokens);
    free(wrapped);
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
