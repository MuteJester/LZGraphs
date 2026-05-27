/**
 * @file decompose.c
 * @brief Canonical FlashBack decomposition into rule-struct steps.
 *
 * This is a fresh implementation — it does NOT share code with
 * lzg_flashback_decompose() because that routine bakes {order} into
 * interned strings we don't want. Here we emit LZGFlashbackGrammarStep structs
 * directly, mapping chars to the grammar's alphabet index space.
 *
 * Reference: .private/flashback_grammar_2026-04-22/flashback_grammar_prototype.py
 * functions `decompose` and `_rec`.
 */
#include <string.h>
#include <stdlib.h>

#include "flashback_grammar_internal.h"

/* ── Alphabet helpers ────────────────────────────────────────── */

void lzg_flashback_grammar_alphabet_init(uint8_t char_to_idx[256],
                           uint8_t idx_to_char[LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE],
                           uint8_t *alphabet_size) {
    memset(char_to_idx, LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN, 256);
    char_to_idx[(uint8_t)LZG_START_SENTINEL] = LZG_FLASHBACK_GRAMMAR_IDX_AT;
    char_to_idx[(uint8_t)LZG_END_SENTINEL]   = LZG_FLASHBACK_GRAMMAR_IDX_DOLL;
    idx_to_char[LZG_FLASHBACK_GRAMMAR_IDX_AT]   = (uint8_t)LZG_START_SENTINEL;
    idx_to_char[LZG_FLASHBACK_GRAMMAR_IDX_DOLL] = (uint8_t)LZG_END_SENTINEL;
    *alphabet_size = 2;
}

LZGError lzg_flashback_grammar_alphabet_observe(uint8_t char_to_idx[256],
                                  uint8_t idx_to_char[LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE],
                                  uint8_t *alphabet_size,
                                  char ch) {
    uint8_t uc = (uint8_t)ch;
    if (char_to_idx[uc] != LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN)
        return LZG_OK;
    if (*alphabet_size >= LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE)
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "fbg: alphabet full (%u), cannot add %c (0x%02x)",
                        LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE, ch, uc);
    char_to_idx[uc] = *alphabet_size;
    idx_to_char[*alphabet_size] = uc;
    (*alphabet_size)++;
    return LZG_OK;
}

/* ── Recursive decomposition ─────────────────────────────────── */

/* Recursive step on seq[lo..hi). `is_root` means this is the S-rule level. */
static LZGError decompose_rec(const uint8_t char_to_idx[256],
                              const char *seq, uint32_t lo, uint32_t hi,
                              bool is_root,
                              LZGFlashbackGrammarStep *out, uint32_t *n_out, uint32_t cap) {
    uint32_t span = hi - lo;
    if (span == 0) return LZG_OK;
    if (*n_out >= cap)
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "fbg decompose: >%u steps", cap);

    uint8_t a_idx = char_to_idx[(uint8_t)seq[lo]];
    uint8_t z_idx = char_to_idx[(uint8_t)seq[hi - 1]];
    if (a_idx == LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN || z_idx == LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN)
        return LZG_FAIL(LZG_ERR_INVALID_SEQUENCE,
                        "fbg decompose: char out of alphabet at [%u,%u)", lo, hi);

    /* Case 1: single char */
    if (span == 1) {
        LZGFlashbackGrammarStep *s = &out[(*n_out)++];
        s->kind       = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE;
        s->a_char     = a_idx;
        s->z_char     = a_idx;
        s->a_run_len  = 1;
        s->z_run_len  = 0;
        s->dst_a      = 0;
        s->dst_z      = 0;
        s->is_start   = is_root ? 1 : 0;
        return LZG_OK;
    }

    /* Leading run of seq[lo] */
    uint32_t flen = 1;
    while (lo + flen < hi && seq[lo + flen] == seq[lo]) flen++;

    /* Case 2: entire span one char */
    if (flen == span) {
        if (flen > 255)
            return LZG_FAIL(LZG_ERR_OVERFLOW,
                            "fbg decompose: leaf_run length %u > 255", flen);
        LZGFlashbackGrammarStep *s = &out[(*n_out)++];
        s->kind       = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN;
        s->a_char     = a_idx;
        s->z_char     = a_idx;
        s->a_run_len  = (uint8_t)flen;
        s->z_run_len  = 0;
        s->dst_a      = 0;
        s->dst_z      = 0;
        s->is_start   = is_root ? 1 : 0;
        return LZG_OK;
    }

    /* Trailing run of seq[hi-1] */
    uint32_t bstart = hi - 1;
    while (bstart > lo && seq[bstart - 1] == seq[hi - 1]) bstart--;

    uint32_t a_run = flen;
    uint32_t z_run = hi - bstart;
    uint32_t mid_lo = lo + flen;
    uint32_t mid_hi = bstart;

    /* Case 3: no middle after peeling (two-char different) */
    if (mid_lo >= mid_hi) {
        /* Canonicity invariant — different chars on the two sides. */
        if (a_idx == z_idx)
            return LZG_FAIL(LZG_ERR_INTERNAL,
                            "fbg decompose: leaf_pair with a==z (bug)");
        if (a_run > 255 || z_run > 255)
            return LZG_FAIL(LZG_ERR_OVERFLOW,
                            "fbg decompose: leaf_pair run length > 255");
        LZGFlashbackGrammarStep *s = &out[(*n_out)++];
        s->kind       = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR;
        s->a_char     = a_idx;
        s->z_char     = z_idx;
        s->a_run_len  = (uint8_t)a_run;
        s->z_run_len  = (uint8_t)z_run;
        s->dst_a      = 0;
        s->dst_z      = 0;
        s->is_start   = is_root ? 1 : 0;
        return LZG_OK;
    }

    /* Internal rule — recurse on middle */
    uint8_t dst_a = char_to_idx[(uint8_t)seq[mid_lo]];
    uint8_t dst_z = char_to_idx[(uint8_t)seq[mid_hi - 1]];
    if (dst_a == LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN || dst_z == LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN)
        return LZG_FAIL(LZG_ERR_INVALID_SEQUENCE,
                        "fbg decompose: inner-middle char out of alphabet");

    /* Canonicity invariants — these hold by FlashBack's maximal-run rule. */
    if (dst_a == a_idx)
        return LZG_FAIL(LZG_ERR_INTERNAL,
                        "fbg decompose: canonicity violated (dst_a == a)");
    if (dst_z == z_idx)
        return LZG_FAIL(LZG_ERR_INTERNAL,
                        "fbg decompose: canonicity violated (dst_z == z)");

    if (a_run > 255 || z_run > 255)
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "fbg decompose: internal run length > 255");

    LZGFlashbackGrammarStep *s = &out[(*n_out)++];
    s->kind       = LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL;
    s->a_char     = a_idx;
    s->z_char     = z_idx;
    s->a_run_len  = (uint8_t)a_run;
    s->z_run_len  = (uint8_t)z_run;
    s->dst_a      = dst_a;
    s->dst_z      = dst_z;
    s->is_start   = is_root ? 1 : 0;

    return decompose_rec(char_to_idx, seq, mid_lo, mid_hi,
                         /*is_root=*/false, out, n_out, cap);
}

LZGError lzg_flashback_grammar_decompose(const LZGFlashbackGrammar *g,
                           const char *seq, uint32_t len,
                           LZGFlashbackGrammarStep *out, uint32_t *n_out) {
    if (!g || !seq || !out || !n_out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_decompose: NULL argument");
    if (len == 0)
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "flashback_grammar_decompose: empty sequence");

    /* Wrap with sentinels internally. For large sequences fall back to heap. */
    enum { STACK_CAP = 512 };
    char stack_buf[STACK_CAP];
    char *wrap;
    bool heap = false;
    uint32_t wlen = len + 2;

    if ((size_t)(wlen + 1) <= sizeof(stack_buf)) {
        wrap = stack_buf;
    } else {
        wrap = (char *)malloc((size_t)(wlen + 1));
        if (!wrap) return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_decompose: alloc");
        heap = true;
    }
    wrap[0] = LZG_START_SENTINEL;
    memcpy(wrap + 1, seq, len);
    wrap[len + 1] = LZG_END_SENTINEL;
    wrap[wlen] = '\0';

    *n_out = 0;
    LZGError err = decompose_rec(g->char_to_idx, wrap, 0, wlen,
                                 /*is_root=*/true, out, n_out,
                                 LZG_FLASHBACK_GRAMMAR_MAX_STEPS);

    if (heap) free(wrap);
    return err;
}

/* ── Reverse: reconstruct sequence from step tree ────────────── */

static LZGError reverse_rec(const uint8_t idx_to_char[LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE],
                            const LZGFlashbackGrammarStep *steps, uint32_t n,
                            uint32_t *i,
                            char *out, uint32_t cap, uint32_t *pos) {
    if (*i >= n)
        return LZG_FAIL(LZG_ERR_INTERNAL, "fbg reverse: step index overflow");
    const LZGFlashbackGrammarStep *s = &steps[*i];
    (*i)++;

    char a_ch = (char)idx_to_char[s->a_char];
    char z_ch = (char)idx_to_char[s->z_char];

    switch ((LZGFlashbackGrammarRuleKind)s->kind) {
    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE:
        if (*pos + 1 > cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        out[(*pos)++] = a_ch;
        return LZG_OK;

    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN:
        if (*pos + s->a_run_len > cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        for (uint32_t k = 0; k < s->a_run_len; k++)
            out[(*pos)++] = a_ch;
        return LZG_OK;

    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR:
        if (*pos + (uint32_t)s->a_run_len + (uint32_t)s->z_run_len > cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        for (uint32_t k = 0; k < s->a_run_len; k++) out[(*pos)++] = a_ch;
        for (uint32_t k = 0; k < s->z_run_len; k++) out[(*pos)++] = z_ch;
        return LZG_OK;

    case LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL: {
        if (*pos + s->a_run_len > cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        for (uint32_t k = 0; k < s->a_run_len; k++) out[(*pos)++] = a_ch;
        LZGError err = reverse_rec(idx_to_char, steps, n, i, out, cap, pos);
        if (err != LZG_OK) return err;
        if (*pos + s->z_run_len > cap)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        for (uint32_t k = 0; k < s->z_run_len; k++) out[(*pos)++] = z_ch;
        return LZG_OK;
    }}

    return LZG_FAIL(LZG_ERR_INTERNAL, "fbg reverse: unknown rule kind %u", s->kind);
}

LZGError lzg_flashback_grammar_tree_to_string(const LZGFlashbackGrammar *g,
                                const LZGFlashbackGrammarStep *steps, uint32_t n,
                                char *out, uint32_t cap, uint32_t *out_len) {
    if (!g || !steps || !out || !out_len)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_tree_to_string: NULL argument");
    if (n == 0) {
        if (cap == 0)
            return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
        out[0] = '\0';
        *out_len = 0;
        return LZG_OK;
    }

    uint32_t pos = 0;
    uint32_t i = 0;
    LZGError err = reverse_rec(g->idx_to_char, steps, n, &i, out, cap, &pos);
    if (err != LZG_OK) return err;

    /* Strip leading '@' and trailing '$' if present. */
    uint32_t start = 0, end = pos;
    if (end > 0 && out[0] == LZG_START_SENTINEL) start = 1;
    if (end > start && out[end - 1] == LZG_END_SENTINEL) end--;

    uint32_t result_len = end - start;
    if (start > 0)
        memmove(out, out + start, result_len);
    if (result_len >= cap)
        return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg reverse: buffer too small");
    out[result_len] = '\0';
    *out_len = result_len;
    return LZG_OK;
}

/* ── NT lookup table ─────────────────────────────────────────── */

void lzg_flashback_grammar_nt_lookup_init(LZGFlashbackGrammarNTLookup *lut) {
    for (uint32_t i = 0; i < LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE; i++)
        for (uint32_t j = 0; j < LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE; j++)
            lut->table[i][j] = UINT32_MAX;
    lut->start_nt = 0;
}

void lzg_flashback_grammar_nt_lookup_set(LZGFlashbackGrammarNTLookup *lut, uint8_t a, uint8_t z, uint32_t nt_idx) {
    if (a < LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE && z < LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE)
        lut->table[a][z] = nt_idx;
}

/* ── Rule / NT lookup helpers shared across pgen, posterior, subtract ── */

int32_t lzg_flashback_grammar_find_nt_for_step(const LZGFlashbackGrammar *g, const LZGFlashbackGrammarStep *step) {
    if (step->is_start) return (int32_t)g->start_nt;
    uint8_t a = step->a_char, z = step->z_char;
    if (a >= LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE || z >= LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE) return -1;
    return g->nt_by_az[(uint32_t)a * LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE + (uint32_t)z];
}

int32_t lzg_flashback_grammar_find_rule_in_nt(const LZGFlashbackGrammar *g, uint32_t nt_idx,
                                 const LZGFlashbackGrammarStep *step) {
    const LZGFlashbackGrammarNT *nt = &g->nts[nt_idx];
    const LZGFlashbackGrammarRule *rules = &g->rules[nt->rule_offset];
    uint8_t kind = step->kind;
    for (uint32_t r = 0; r < nt->n_rules; r++) {
        const LZGFlashbackGrammarRule *rule = &rules[r];
        if (rule->kind != kind) continue;
        switch ((LZGFlashbackGrammarRuleKind)kind) {
        case LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL:
            if (rule->a_run_len == step->a_run_len
                && rule->z_run_len == step->z_run_len
                && rule->dst_a == step->dst_a
                && rule->dst_z == step->dst_z)
                return (int32_t)r;
            break;
        case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE:
            return (int32_t)r;
        case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN:
            if (rule->a_run_len == step->a_run_len) return (int32_t)r;
            break;
        case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR:
            if (rule->a_run_len == step->a_run_len
                && rule->z_run_len == step->z_run_len)
                return (int32_t)r;
            break;
        }
    }
    return -1;
}
