/**
 * @file naive.c
 * @brief Naive positional encoding: "{residue}_{1-based index}".
 *
 * The whole decomposition is this file. There is no dictionary, no
 * back-reference, no run peeling — a character's node identity is the
 * character plus how far into the sequence it sits. That is the point:
 * every structural claim FlashBack makes has to earn its keep against
 * a model that knows nothing but position.
 */
#include <stdio.h>
#include <string.h>

#include "lzgraph/naive_graph.h"

LZGError lzg_naive_encode(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids, uint32_t *out_count) {
    if (!str || !pool || !out_ids || !out_count)
        return LZG_ERR_NULL_ARG;
    if (len == 0) {
        *out_count = 0;
        return LZG_OK;
    }
    /* len residues plus the two sentinels. */
    if (len + 2u > LZG_NAIVE_MAX_WALK)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                        "naive encode: sequence length %u exceeds max walk %u",
                        len, (uint32_t)LZG_NAIVE_MAX_WALK - 2u);

    static const char start_label[2] = {LZG_START_SENTINEL, '\0'};
    static const char end_label[2]   = {LZG_END_SENTINEL, '\0'};

    uint32_t n = 0;
    out_ids[n++] = lzg_sp_intern(pool, start_label);

    char label[LZG_NAIVE_LABEL_CAP];
    for (uint32_t i = 0; i < len; i++) {
        int written = snprintf(label, sizeof(label), "%c_%u", str[i], i + 1u);
        if (written <= 0 || (size_t)written >= sizeof(label))
            return LZG_FAIL(LZG_ERR_INTERNAL,
                            "naive encode: label overflow at index %u", i);
        out_ids[n++] = lzg_sp_intern_n(pool, label, (uint32_t)written);
    }

    out_ids[n++] = lzg_sp_intern(pool, end_label);
    *out_count = n;
    return LZG_OK;
}

LZGError lzg_naive_reverse(const LZGStringPool *pool,
                           const uint32_t *label_ids, uint32_t count,
                           char *out_buf, uint32_t buf_cap,
                           uint32_t *out_len) {
    if (!pool || !label_ids || !out_buf || !out_len)
        return LZG_ERR_NULL_ARG;
    if (buf_cap == 0) return LZG_ERR_PARAM_OUT_OF_RANGE;

    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++) {
        const char *label = lzg_sp_get(pool, label_ids[i]);
        if (!label || label[0] == '\0') continue;
        if (label[0] == LZG_START_SENTINEL || label[0] == LZG_END_SENTINEL)
            continue;
        if (n + 1u >= buf_cap)
            return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                            "naive reverse: output buffer too small");
        out_buf[n++] = label[0];
    }
    out_buf[n] = '\0';
    *out_len = n;
    return LZG_OK;
}
