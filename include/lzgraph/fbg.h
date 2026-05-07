/**
 * @file fbg.h
 * @brief FlashBackGrammar — boundary-indexed PCFG over FlashBack decomposition trees.
 *
 * Unlike FlashBackGraph (a Markov chain on linearised tokens), FlashBackGrammar
 * models the *tree* of FlashBack decomposition directly as a weighted probabilistic
 * context-free grammar with non-terminals indexed by the boundary characters of
 * the current middle. Rules are shared across all recursion depths that share a
 * boundary pair, giving cross-depth motif reuse while still guaranteeing every
 * generated tree is a canonical FlashBack decomposition.
 *
 * All analytics (path count series, entropy, Hill numbers, dynamic range) are
 * computed exactly via small linear systems over the non-terminal space.
 *
 * Reference implementation: `.private/flashback_grammar_2026-04-22/fbg_prototype.py`.
 */
#ifndef LZGRAPH_FBG_H
#define LZGRAPH_FBG_H

#include "lzgraph/common.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/rng.h"
#include "lzgraph/simulate.h"     /* LZGSimResult                        */
#include "lzgraph/analytics.h"    /* LZGEffectiveDiversity, LZGDynamicRange */

/* ── Configuration enums ─────────────────────────────────────── */

typedef enum {
    LZG_FBG_ABUND_NONE   = 0,  /* each sequence contributes 1 regardless of abundance */
    LZG_FBG_ABUND_LINEAR = 1,  /* rule counts scaled by abundance (default)           */
    LZG_FBG_ABUND_LOG    = 2,  /* rule counts scaled by log1p(abundance)              */
} LZGFbgAbundanceMode;

typedef enum {
    LZG_FBG_BACKOFF_NONE = 0,  /* strict MLE: unseen rule ⇒ pgen collapses to ε       */
    LZG_FBG_BACKOFF_GT   = 1,  /* Good-Turing reserved-mass + marginal-frequency      */
} LZGFbgBackoff;

/* ── Rule representation ─────────────────────────────────────── */

/* Rule kinds. Must fit in 8 bits. */
typedef enum {
    LZG_FBG_RULE_INTERNAL    = 0,
    LZG_FBG_RULE_LEAF_SINGLE = 1,
    LZG_FBG_RULE_LEAF_RUN    = 2,
    LZG_FBG_RULE_LEAF_PAIR   = 3,
} LZGFbgRuleKind;

/**
 * A single rule at some non-terminal. Identity comes from the struct fields,
 * not any string. Field semantics depend on `kind`:
 *
 *   INTERNAL:     peel(a_run_len copies of a_char, z_run_len copies of z_char)
 *                 then transit to M(dst_a, dst_z).  a_char,z_char come from
 *                 the source NT; a_run_len,z_run_len,dst_a,dst_z identify the rule.
 *   LEAF_SINGLE:  one char a_char (= z_char). Only valid at M(a_char, a_char).
 *   LEAF_RUN:     a_run_len copies of a_char (a_run_len ≥ 2). Only at M(a_char, a_char).
 *   LEAF_PAIR:    a_run_len copies of a_char + z_run_len copies of z_char,
 *                 a_char ≠ z_char. Valid at M(a_char, z_char).
 *
 * Chars are stored as alphabet indices, not ASCII. See `LZGFbg.idx_to_char`.
 */
typedef struct {
    uint8_t  kind;
    uint8_t  a_char;
    uint8_t  z_char;
    uint8_t  a_run_len;
    uint8_t  z_run_len;
    uint8_t  dst_a;       /* INTERNAL only */
    uint8_t  dst_z;       /* INTERNAL only */
    uint8_t  _pad;
    uint64_t count;
    double   weight;      /* MLE probability = count / total_at_nt (post-smoothing) */
} LZGFbgRule;

/** A non-terminal: contiguous slice of rules[] sharing a source (a, z) pair. */
typedef struct {
    uint8_t  a;              /* left boundary char index (0 used for S's inner) */
    uint8_t  z;              /* right boundary char index                        */
    uint8_t  is_start;       /* 1 for S, 0 otherwise                             */
    uint8_t  _pad;
    uint32_t rule_offset;    /* start index into LZGFbg.rules[]                  */
    uint32_t n_rules;
    double   total_count;    /* Σ counts at this NT (may be fractional)          */
    double   unseen_mass;    /* δ_nt for backoff='gt'; 0 otherwise                */
} LZGFbgNT;

/* A single rule observation emitted by the decomposer (no counts, no weights). */
typedef struct {
    uint8_t kind;
    uint8_t a_char;
    uint8_t z_char;
    uint8_t a_run_len;
    uint8_t z_run_len;
    uint8_t dst_a;
    uint8_t dst_z;
    uint8_t is_start;     /* 1 iff this step is the S-rule (the first step) */
} LZGFbgStep;

/** Max canonical decomposition depth — same as existing FlashBack cap. */
#define LZG_FBG_MAX_STEPS 128

/** Chars we admit: 20 AAs + 2 sentinels. Index 0..21. 22..255 unused. */
#define LZG_FBG_ALPHABET_SIZE 22

/* ── Main grammar struct ─────────────────────────────────────── */

typedef struct {
    /* Alphabet mapping. char_to_idx[ch] = index into rules' a_char/z_char.
     * idx_to_char[i] recovers the ASCII char. Index 0xFF = unknown/absent. */
    uint8_t char_to_idx[256];
    uint8_t idx_to_char[LZG_FBG_ALPHABET_SIZE];
    uint8_t alphabet_size;   /* count of known chars (incl. sentinels)  */

    /* Non-terminals. nts[0] is always S. nts[1..] are M(a, z) pairs. */
    LZGFbgNT *nts;
    uint32_t  n_nts;
    uint32_t  start_nt;      /* always 0 after finalize                 */

    /* All rules, grouped contiguously by source NT per `nts[i].rule_offset`. */
    LZGFbgRule *rules;
    uint32_t    n_rules;
    uint32_t    n_internal;  /* count of rules with kind==INTERNAL       */

    /* Training length histogram. length_counts[L] = (weighted) training seqs
     * of output length L. max_length is the highest L with a nonzero count. */
    uint64_t *length_counts;
    uint32_t  max_length;

    /* Configuration (copied from build args; not mutated). */
    LZGFbgAbundanceMode abundance_mode;
    double              smoothing;
    LZGFbgBackoff       backoff;

    /* Derived at finalize. */
    double  spectral_radius;   /* ρ(T) — consistency test               */
    bool    is_consistent;     /* spectral_radius < 1.0                  */

    /* Backoff table: marginal(rule) = Σ count(rule, nt) / Σ total_counts.
     * Keyed by packed rule identity (see fbg_internal.h). Present iff
     * backoff == LZG_FBG_BACKOFF_GT. Values are doubles stored as uint64
     * bit-patterns in the hash map. */
    LZGHashMap *rule_marginal;  /* key: packed rule id → double bits     */

    /* Fast (a_idx, z_idx) → nt index lookup. -1 means "no NT for this
     * boundary pair". Populated by lzg_fbg_finalize(). Does NOT include S
     * (the start NT is always at index 0 via start_nt). */
    int32_t nt_by_az[LZG_FBG_ALPHABET_SIZE * LZG_FBG_ALPHABET_SIZE];
} LZGFbg;

/* ── Lifecycle ───────────────────────────────────────────────── */

/** Allocate a zero-initialised grammar struct. Caller owns. */
LZGFbg *lzg_fbg_new(void);

/** Destroy a grammar and all its owned memory. NULL-safe. */
void lzg_fbg_destroy(LZGFbg *g);

/**
 * Build a grammar from sequences with optional abundances.
 *
 * @param g           Freshly allocated (via lzg_fbg_new) grammar.
 * @param sequences   Array of n null-terminated CDR3 strings.
 * @param n           Number of sequences.
 * @param abundances  Per-sequence counts. NULL = all 1.
 * @param mode        Abundance weighting mode.
 * @param smoothing   Laplace α over observed rules per NT. 0 = pure MLE.
 * @param backoff     Unseen-rule policy for pgen queries.
 */
LZGError lzg_fbg_build(LZGFbg *g,
                       const char **sequences, uint32_t n,
                       const uint64_t *abundances,
                       LZGFbgAbundanceMode mode,
                       double smoothing,
                       LZGFbgBackoff backoff);

/** Build from a plain text file (one seq per line, or seq\tabundance). */
LZGError lzg_fbg_build_file(LZGFbg *g, const char *path,
                            LZGFbgAbundanceMode mode,
                            double smoothing,
                            LZGFbgBackoff backoff);

/* ── Decomposer (fresh — does NOT reuse lzg_flashback_decompose) ── */

/**
 * Decompose a CDR3 into canonical FlashBack steps using the grammar's alphabet.
 *
 * @param g         Source grammar (for alphabet mapping). If a char in `seq` is
 *                  outside g's alphabet, returns LZG_ERR_INVALID_SEQUENCE.
 * @param seq       Input CDR3 (no sentinels).
 * @param len       Length of seq.
 * @param out       Output buffer of at least LZG_FBG_MAX_STEPS steps.
 * @param n_out     On success, filled with the number of steps emitted.
 */
LZGError lzg_fbg_decompose(const LZGFbg *g,
                           const char *seq, uint32_t len,
                           LZGFbgStep *out, uint32_t *n_out);

/**
 * Reconstruct the sequence (without sentinels) from a step tree.
 *
 * Expected length = Σ output chars emitted by the steps (O(L) given n steps).
 * Returns LZG_ERR_OVERFLOW if the output buffer is too small.
 */
LZGError lzg_fbg_tree_to_string(const LZGFbg *g,
                                const LZGFbgStep *steps, uint32_t n,
                                char *out, uint32_t cap, uint32_t *out_len);

/* ── pgen ────────────────────────────────────────────────────── */

/** pgen under g's configured backoff policy. Returns log-probability. */
double lzg_fbg_pgen(const LZGFbg *g, const char *seq, uint32_t len);

/** pgen under pure MLE — ignores g->backoff. Used for MC-entropy checks. */
double lzg_fbg_pgen_mle(const LZGFbg *g, const char *seq, uint32_t len);

/** Batch pgen. out[] must have space for n doubles. */
LZGError lzg_fbg_pgen_batch(const LZGFbg *g,
                            const char **seqs, uint32_t n,
                            double *out);

/* ── Analytics ───────────────────────────────────────────────── */

LZGError lzg_fbg_path_count_series(const LZGFbg *g, uint32_t L_max, double *out);
LZGError lzg_fbg_length_distribution(const LZGFbg *g, uint32_t L_max, double *out);
LZGError lzg_fbg_entropy(const LZGFbg *g, double *H_nats);
LZGError lzg_fbg_effective_diversity(const LZGFbg *g, LZGEffectiveDiversity *out);
LZGError lzg_fbg_power_sum(const LZGFbg *g, double alpha, double *out);
LZGError lzg_fbg_hill_number(const LZGFbg *g, double alpha, double *out);
LZGError lzg_fbg_hill_numbers(const LZGFbg *g,
                              const double *alphas, uint32_t n,
                              double *out);
LZGError lzg_fbg_dynamic_range(const LZGFbg *g, uint32_t L_max,
                               LZGDynamicRange *out);

/* ── Sampling ────────────────────────────────────────────────── */

LZGError lzg_fbg_simulate(const LZGFbg *g, uint32_t n,
                          LZGRng *rng, LZGSimResult *out);

LZGError lzg_fbg_top_k_sequences(const LZGFbg *g, uint32_t k,
                                 bool most_probable, uint32_t L_max,
                                 LZGSimResult *out, uint32_t *n_out);

/* ── Posterior / subtract ────────────────────────────────────── */

LZGError lzg_fbg_posterior(const LZGFbg *prior,
                           const char **sequences, uint32_t n,
                           const uint64_t *abundances,
                           double kappa,
                           LZGFbg **out);

LZGError lzg_fbg_subtract(const LZGFbg *g,
                          const char **sequences, uint32_t n,
                          const uint64_t *abundances,
                          LZGFbg **out);

/* ── I/O ─────────────────────────────────────────────────────── */

LZGError lzg_fbg_save(const LZGFbg *g, const char *path);
LZGError lzg_fbg_load(const char *path, LZGFbg **out);

#endif /* LZGRAPH_FBG_H */
