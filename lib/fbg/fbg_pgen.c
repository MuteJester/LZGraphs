/**
 * @file fbg_pgen.c
 * @brief pgen computation for FlashBackGrammar.
 *
 * Reference: fbg_prototype.py `_pgen`.
 *
 * Mutation hygiene note: this function does NOT intern into g->pool or any
 * other shared structure. Decomposition is stack-allocated LZGFbgStep[].
 */
#include "fbg_internal.h"
#include <string.h>

/* Reinterpret the uint64 value stored in the rule_marginal hash map as double. */
static inline double bits_to_double(uint64_t bits) {
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

/* NT + rule lookup helpers live in fbg_decompose.c — see fbg_internal.h. */
#define find_rule_in_nt lzg_fbg_find_rule_in_nt
#define find_nt          lzg_fbg_find_nt_for_step

/* Core pgen routine — returns log-probability. Uses backoff iff use_backoff. */
static double pgen_inner(const LZGFbg *g,
                          const char *seq, uint32_t len,
                          bool use_backoff) {
    if (!g || !seq || len == 0) return LZG_LOG_EPS;

    LZGFbgStep steps[LZG_FBG_MAX_STEPS];
    uint32_t n_steps = 0;
    LZGError err = lzg_fbg_decompose(g, seq, len, steps, &n_steps);
    if (err != LZG_OK) return LZG_LOG_EPS;  /* out-of-alphabet → ε */

    double log_p = 0.0;

    for (uint32_t t = 0; t < n_steps; t++) {
        const LZGFbgStep *s = &steps[t];

        int32_t nt_idx = find_nt(g, s);
        if (nt_idx < 0) {
            /* NT itself is unseen — no backoff for NTs. */
            return LZG_LOG_EPS;
        }

        int32_t r_idx = find_rule_in_nt(g, (uint32_t)nt_idx, s);

        if (r_idx >= 0) {
            double w = g->rules[g->nts[nt_idx].rule_offset + r_idx].weight;
            if (use_backoff) {
                double delta = g->nts[nt_idx].unseen_mass;
                w = (1.0 - delta) * w;
            }
            if (w <= 0.0) return LZG_LOG_EPS;
            log_p += log(w);
        } else {
            /* Rule unseen at this NT. */
            if (!use_backoff) return LZG_LOG_EPS;

            double delta = g->nts[nt_idx].unseen_mass;
            if (delta <= 0.0) return LZG_LOG_EPS;

            uint64_t key = lzg_fbg_rule_marginal_key(
                s->kind,
                s->a_char, s->z_char,
                s->a_run_len, s->z_run_len,
                s->dst_a, s->dst_z);
            uint64_t *slot = g->rule_marginal
                             ? lzg_hm_get(g->rule_marginal, key) : NULL;
            if (!slot) return LZG_LOG_EPS;
            double marg = bits_to_double(*slot);
            if (marg <= 0.0) return LZG_LOG_EPS;

            double w = delta * marg;
            if (w <= 0.0) return LZG_LOG_EPS;
            log_p += log(w);
        }
    }

    return log_p;
}

/* ── Public API ──────────────────────────────────────────────── */

double lzg_fbg_pgen(const LZGFbg *g, const char *seq, uint32_t len) {
    return pgen_inner(g, seq, len, g && g->backoff == LZG_FBG_BACKOFF_GT);
}

double lzg_fbg_pgen_mle(const LZGFbg *g, const char *seq, uint32_t len) {
    return pgen_inner(g, seq, len, false);
}

LZGError lzg_fbg_pgen_batch(const LZGFbg *g,
                            const char **seqs, uint32_t n,
                            double *out) {
    if (!g || !seqs || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_pgen_batch: NULL argument");
    for (uint32_t i = 0; i < n; i++) {
        const char *s = seqs[i];
        out[i] = s ? lzg_fbg_pgen(g, s, (uint32_t)strlen(s)) : LZG_LOG_EPS;
    }
    return LZG_OK;
}
