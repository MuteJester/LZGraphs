/**
 * @file fbg_internal.h
 * @brief Shared helpers for the FlashBackGrammar implementation.
 *
 * Not part of the public API. Do not include from outside lib/fbg/.
 */
#ifndef LZG_FBG_INTERNAL_H
#define LZG_FBG_INTERNAL_H

#include "lzgraph/fbg.h"

#define LZG_FBG_CHAR_UNKNOWN 0xFFu

/* Reserved alphabet indices for sentinels. */
#define LZG_FBG_IDX_AT    0u   /* '@' */
#define LZG_FBG_IDX_DOLL  1u   /* '$' */

/* ── Alphabet helpers ────────────────────────────────────────── */

/**
 * Build char→idx map from an array of raw ASCII chars seen in training.
 * Sentinels @ and $ are always included; other chars are assigned in order
 * of first appearance.
 */
void lzg_fbg_alphabet_init(uint8_t char_to_idx[256],
                           uint8_t idx_to_char[LZG_FBG_ALPHABET_SIZE],
                           uint8_t *alphabet_size);

/** Ensure `ch` has an alphabet index; extend the map if room remains. */
LZGError lzg_fbg_alphabet_observe(uint8_t char_to_idx[256],
                                  uint8_t idx_to_char[LZG_FBG_ALPHABET_SIZE],
                                  uint8_t *alphabet_size,
                                  char ch);

/* ── Rule identity keys ──────────────────────────────────────── */

/**
 * Pack a rule's identity (excluding source NT a_char, z_char for INTERNAL,
 * but including them for leaves since they *are* the leaf content) into
 * a single 64-bit key suitable for marginal lookup across non-terminals.
 *
 * Two rules at different source NTs collide under this key iff they
 * represent the same grammar production in the prototype's sense.
 */
static inline uint64_t lzg_fbg_rule_marginal_key(uint8_t kind,
                                                 uint8_t a_char, uint8_t z_char,
                                                 uint8_t a_run_len, uint8_t z_run_len,
                                                 uint8_t dst_a, uint8_t dst_z) {
    uint64_t k = (uint64_t)kind;
    switch ((LZGFbgRuleKind)kind) {
    case LZG_FBG_RULE_INTERNAL:
        k |= ((uint64_t)a_run_len) << 8;
        k |= ((uint64_t)z_run_len) << 16;
        k |= ((uint64_t)dst_a)     << 24;
        k |= ((uint64_t)dst_z)     << 32;
        break;
    case LZG_FBG_RULE_LEAF_SINGLE:
        k |= ((uint64_t)a_char) << 8;
        break;
    case LZG_FBG_RULE_LEAF_RUN:
        k |= ((uint64_t)a_char)    << 8;
        k |= ((uint64_t)a_run_len) << 16;
        break;
    case LZG_FBG_RULE_LEAF_PAIR:
        k |= ((uint64_t)a_char)    << 8;
        k |= ((uint64_t)a_run_len) << 16;
        k |= ((uint64_t)z_char)    << 24;
        k |= ((uint64_t)z_run_len) << 32;
        break;
    }
    return k;
}

/**
 * Within-NT rule identity key for lookup during build/decompose.
 * Excludes source NT's a_char, z_char (they're constant within an NT).
 */
static inline uint64_t lzg_fbg_rule_local_key(uint8_t kind,
                                              uint8_t a_run_len, uint8_t z_run_len,
                                              uint8_t dst_a, uint8_t dst_z,
                                              uint8_t leaf_a_char, uint8_t leaf_z_char) {
    uint64_t k = (uint64_t)kind;
    switch ((LZGFbgRuleKind)kind) {
    case LZG_FBG_RULE_INTERNAL:
        k |= ((uint64_t)a_run_len) << 8;
        k |= ((uint64_t)z_run_len) << 16;
        k |= ((uint64_t)dst_a)     << 24;
        k |= ((uint64_t)dst_z)     << 32;
        break;
    case LZG_FBG_RULE_LEAF_SINGLE:
        /* At M(c,c), leaf_single(c) — no extra fields needed. */
        break;
    case LZG_FBG_RULE_LEAF_RUN:
        k |= ((uint64_t)a_run_len) << 8;
        break;
    case LZG_FBG_RULE_LEAF_PAIR:
        k |= ((uint64_t)a_run_len) << 8;
        k |= ((uint64_t)z_run_len) << 16;
        break;
    }
    /* Leaf a/z chars are implicit in the source NT (a,z), so we don't
     * re-encode them here. The leaf_a_char/leaf_z_char params are accepted
     * for consistency with call sites that may want to cross-check. */
    (void)leaf_a_char;
    (void)leaf_z_char;
    return k;
}

/* ── Non-terminal lookup ─────────────────────────────────────── */

/** Flat (a, z) → NT-index lookup table. UINT32_MAX means "not present". */
typedef struct {
    uint32_t table[LZG_FBG_ALPHABET_SIZE][LZG_FBG_ALPHABET_SIZE];
    uint32_t start_nt;  /* always 0 — S is the start NT */
} LZGFbgNTLookup;

void lzg_fbg_nt_lookup_init(LZGFbgNTLookup *lut);
void lzg_fbg_nt_lookup_set(LZGFbgNTLookup *lut, uint8_t a, uint8_t z, uint32_t nt_idx);
static inline uint32_t lzg_fbg_nt_lookup_get(const LZGFbgNTLookup *lut,
                                              uint8_t a, uint8_t z) {
    if (a >= LZG_FBG_ALPHABET_SIZE || z >= LZG_FBG_ALPHABET_SIZE)
        return UINT32_MAX;
    return lut->table[a][z];
}

/* ── Abundance weight ────────────────────────────────────────── */

static inline double lzg_fbg_abund_weight(LZGFbgAbundanceMode mode, uint64_t a) {
    switch (mode) {
    case LZG_FBG_ABUND_NONE:   return 1.0;
    case LZG_FBG_ABUND_LINEAR: return (double)a;
    case LZG_FBG_ABUND_LOG:    return log1p((double)a);
    }
    return 1.0;
}

/* ── Internal finalize (shared by build, posterior, subtract) ── */

/**
 * After `g->rules` and `g->nts` have been filled with counts (kind, chars,
 * etc. set; weight left at 0), compute:
 *   - weight for each rule (MLE with Laplace smoothing)
 *   - nt->total_count, nt->unseen_mass (if backoff == GT)
 *   - spectral radius ρ(T) and is_consistent
 *   - rule_marginal hash map (if backoff == GT)
 */
LZGError lzg_fbg_finalize(LZGFbg *g);

/* ── Rule / NT lookup helpers (shared by pgen, posterior, subtract) ── */

/** Find NT index for a decomposed step. Returns -1 if not present. */
int32_t lzg_fbg_find_nt_for_step(const LZGFbg *g, const LZGFbgStep *step);

/** Linear scan for matching rule within a source NT. Returns relative index
 *  (0..n_rules-1) or -1 if not found. */
int32_t lzg_fbg_find_rule_in_nt(const LZGFbg *g, uint32_t nt_idx,
                                 const LZGFbgStep *step);

/** Power-iteration estimate of the spectral radius of a dense n×n
 *  non-negative matrix (row-major). Uses ≤ 200 iterations. */
double lzg_fbg_spectral_radius_dense(const double *T, uint32_t n);

#endif /* LZG_FBG_INTERNAL_H */
