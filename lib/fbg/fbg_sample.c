/**
 * @file fbg_sample.c
 * @brief Top-down grammar sampler and K-best sequences for FlashBackGrammar.
 *
 * Canonicity is a hard invariant: every tree emitted by the sampler is a
 * canonical FlashBack decomposition by construction (rules enforce
 * a' ≠ a, z' ≠ z; leaves only exist at NTs that can accept their shape).
 *
 * Reference: fbg_prototype.py `sample` and `top_k_sequences`.
 */
#include "fbg_internal.h"
#include <stdlib.h>
#include <string.h>

/* ═════════════════════════════════════════════════════════════ */
/* Top-down sampling                                             */
/* ═════════════════════════════════════════════════════════════ */

/* Sample one rule at NT nt_idx via weighted choice. Returns rule index
 * relative to the NT's rule slab (0..nt->n_rules-1). */
static uint32_t sample_rule_at(const LZGFbg *g, uint32_t nt_idx, LZGRng *rng) {
    const LZGFbgNT *nt = &g->nts[nt_idx];
    const LZGFbgRule *rules = &g->rules[nt->rule_offset];

    double total = 0.0;
    for (uint32_t r = 0; r < nt->n_rules; r++) total += rules[r].weight;

    double u = lzg_rng_double(rng) * total;
    double acc = 0.0;
    for (uint32_t r = 0; r < nt->n_rules; r++) {
        acc += rules[r].weight;
        if (u < acc) return r;
    }
    return nt->n_rules - 1;  /* fall-through due to FP slop */
}

/* Recursive top-down expansion. Appends steps and accumulates log-prob.
 * Returns LZG_OK or LZG_ERR_OVERFLOW if step buffer is exhausted. */
static LZGError sample_rec(const LZGFbg *g, uint32_t nt_idx,
                            LZGRng *rng,
                            LZGFbgStep *steps, uint32_t *n_steps,
                            double *log_p) {
    if (*n_steps >= LZG_FBG_MAX_STEPS)
        return LZG_FAIL(LZG_ERR_OVERFLOW, "fbg sample: >%u steps",
                        LZG_FBG_MAX_STEPS);

    uint32_t rel = sample_rule_at(g, nt_idx, rng);
    const LZGFbgNT *nt = &g->nts[nt_idx];
    const LZGFbgRule *rule = &g->rules[nt->rule_offset + rel];

    double w = rule->weight;
    if (w > 0.0) *log_p += log(w);
    else         *log_p += LZG_LOG_EPS;

    LZGFbgStep *s = &steps[(*n_steps)++];
    s->kind       = rule->kind;
    s->a_char     = rule->a_char;
    s->z_char     = rule->z_char;
    s->a_run_len  = rule->a_run_len;
    s->z_run_len  = rule->z_run_len;
    s->dst_a      = rule->dst_a;
    s->dst_z      = rule->dst_z;
    s->is_start   = nt->is_start;

    if (rule->kind == LZG_FBG_RULE_INTERNAL) {
        int32_t dst = g->nt_by_az[
            (uint32_t)rule->dst_a * LZG_FBG_ALPHABET_SIZE
            + (uint32_t)rule->dst_z];
        if (dst < 0)
            return LZG_FAIL(LZG_ERR_INTERNAL,
                            "fbg sample: destination NT not found");
        return sample_rec(g, (uint32_t)dst, rng, steps, n_steps, log_p);
    }
    return LZG_OK;
}

LZGError lzg_fbg_simulate(const LZGFbg *g, uint32_t n,
                           LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_simulate: NULL");
    if (g->n_nts == 0)
        return LZG_FAIL(LZG_ERR_NOT_BUILT, "fbg_simulate: empty grammar");

    for (uint32_t i = 0; i < n; i++) {
        LZGFbgStep steps[LZG_FBG_MAX_STEPS];
        uint32_t n_steps = 0;
        double log_p = 0.0;

        LZGError err = sample_rec(g, g->start_nt, rng, steps, &n_steps, &log_p);
        if (err != LZG_OK) {
            /* On overflow, emit empty and continue — sampling shouldn't block. */
            out[i].sequence = strdup("");
            out[i].seq_len  = 0;
            out[i].n_tokens = 0;
            out[i].log_prob = LZG_LOG_EPS;
            continue;
        }

        char buf[2048];
        uint32_t out_len = 0;
        err = lzg_fbg_tree_to_string(g, steps, n_steps, buf, sizeof(buf), &out_len);
        if (err != LZG_OK) {
            out[i].sequence = strdup("");
            out[i].seq_len  = 0;
            out[i].n_tokens = n_steps;
            out[i].log_prob = log_p;
            continue;
        }

        out[i].sequence = strdup(buf);
        out[i].seq_len  = out_len;
        out[i].n_tokens = n_steps;   /* tree depth */
        out[i].log_prob = log_p;
    }
    return LZG_OK;
}

/* ═════════════════════════════════════════════════════════════ */
/* Dynamic range: length-bounded max/min log-prob per NT          */
/* ═════════════════════════════════════════════════════════════ */

LZGError lzg_fbg_dynamic_range(const LZGFbg *g, uint32_t L_max,
                                LZGDynamicRange *out) {
    if (!g || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_dynamic_range: NULL");
    memset(out, 0, sizeof(*out));

    uint32_t n = g->n_nts;
    size_t cells = (size_t)n * (L_max + 1);
    double *max_lp = (double *)malloc(cells * sizeof(double));
    double *min_lp = (double *)malloc(cells * sizeof(double));
    if (!max_lp || !min_lp) {
        free(max_lp); free(min_lp);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_dynamic_range: alloc");
    }
    /* Sentinel: unreachable. */
    for (size_t k = 0; k < cells; k++) {
        max_lp[k] = -1e300;
        min_lp[k] =  1e300;
    }

    /* Build: for each L (ascending), compute non-S NTs, then S. */
    for (uint32_t L = 0; L <= L_max; L++) {
        /* non-S */
        for (uint32_t i = 1; i < n; i++) {
            const LZGFbgNT *nt = &g->nts[i];
            size_t slot = (size_t)i * (L_max + 1) + L;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                if (rule->weight <= 0.0) continue;
                double lw = log(rule->weight);

                switch ((LZGFbgRuleKind)rule->kind) {
                case LZG_FBG_RULE_LEAF_SINGLE:
                    if (L == 1) {
                        if (lw > max_lp[slot]) max_lp[slot] = lw;
                        if (lw < min_lp[slot]) min_lp[slot] = lw;
                    }
                    break;
                case LZG_FBG_RULE_LEAF_RUN:
                    if (L == rule->a_run_len) {
                        if (lw > max_lp[slot]) max_lp[slot] = lw;
                        if (lw < min_lp[slot]) min_lp[slot] = lw;
                    }
                    break;
                case LZG_FBG_RULE_LEAF_PAIR: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    if (L == ell) {
                        if (lw > max_lp[slot]) max_lp[slot] = lw;
                        if (lw < min_lp[slot]) min_lp[slot] = lw;
                    }
                    break;
                }
                case LZG_FBG_RULE_INTERNAL: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    if (L < ell) break;
                    int32_t dst = g->nt_by_az[
                        (uint32_t)rule->dst_a * LZG_FBG_ALPHABET_SIZE
                        + (uint32_t)rule->dst_z];
                    if (dst < 0) break;
                    size_t dst_slot = (size_t)dst * (L_max + 1) + (L - ell);
                    if (max_lp[dst_slot] > -1e299) {
                        double v = lw + max_lp[dst_slot];
                        if (v > max_lp[slot]) max_lp[slot] = v;
                    }
                    if (min_lp[dst_slot] <  1e299) {
                        double v = lw + min_lp[dst_slot];
                        if (v < min_lp[slot]) min_lp[slot] = v;
                    }
                    break;
                }}
            }
        }
        /* S: zero-consumption peel(@,$) */
        if (g->nts[0].is_start) {
            const LZGFbgNT *nt = &g->nts[0];
            size_t slot = L;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                if (rule->kind != LZG_FBG_RULE_INTERNAL) continue;
                if (rule->weight <= 0.0) continue;
                double lw = log(rule->weight);
                int32_t dst = g->nt_by_az[
                    (uint32_t)rule->dst_a * LZG_FBG_ALPHABET_SIZE
                    + (uint32_t)rule->dst_z];
                if (dst < 0) continue;
                size_t dst_slot = (size_t)dst * (L_max + 1) + L;
                if (max_lp[dst_slot] > -1e299) {
                    double v = lw + max_lp[dst_slot];
                    if (v > max_lp[slot]) max_lp[slot] = v;
                }
                if (min_lp[dst_slot] <  1e299) {
                    double v = lw + min_lp[dst_slot];
                    if (v < min_lp[slot]) min_lp[slot] = v;
                }
            }
        }
    }

    /* Collect over S row for all reachable L. */
    double g_max = -1e300, g_min = 1e300;
    bool found = false;
    for (uint32_t L = 1; L <= L_max; L++) {
        size_t slot = L;
        if (max_lp[slot] > -1e299) { found = true; if (max_lp[slot] > g_max) g_max = max_lp[slot]; }
        if (min_lp[slot] <  1e299) {               if (min_lp[slot] < g_min) g_min = min_lp[slot]; }
    }
    free(max_lp);
    free(min_lp);
    if (!found) {
        memset(out, 0, sizeof(*out));
        return LZG_OK;
    }
    out->max_log_prob = g_max;
    out->min_log_prob = g_min;
    out->dynamic_range_nats = g_max - g_min;
    out->dynamic_range_orders = out->dynamic_range_nats / log(10.0);
    return LZG_OK;
}

/* ═════════════════════════════════════════════════════════════ */
/* Top-K sequences via per-NT K-best DP                           */
/* ═════════════════════════════════════════════════════════════ */

/* K-best entry at a (NT, L) cell: log-prob and back-pointer info to
 * reconstruct the tree. */
typedef struct {
    double   log_prob;
    uint8_t  rule_idx;       /* relative index within NT's rule slab */
    /* For INTERNAL rules: the child's (nt, L-ell, k-index). */
    int32_t  child_nt;
    uint32_t child_L;
    uint32_t child_k;
} KBest;

/* Insert entry into a sorted-descending K-best list (compact insertion).
 * `arr` holds up to `cap` entries in descending log_prob order; `count`
 * tracks current size. */
static void kbest_insert(KBest *arr, uint32_t cap, uint32_t *count,
                          const KBest *e, bool most_probable) {
    /* Find insertion index. */
    uint32_t i;
    for (i = 0; i < *count; i++) {
        bool better = most_probable
            ? (e->log_prob > arr[i].log_prob)
            : (e->log_prob < arr[i].log_prob);
        if (better) break;
    }
    if (i >= cap) return;  /* not good enough */
    /* Shift right. */
    uint32_t end = (*count < cap) ? *count : cap - 1;
    for (uint32_t j = end; j > i; j--) arr[j] = arr[j - 1];
    arr[i] = *e;
    if (*count < cap) (*count)++;
}

/* Reconstruct the walk for the k-th best tree rooted at (nt, L).
 * Emits steps[0..n_steps) in grammar traversal order. */
static LZGError reconstruct(const LZGFbg *g, const KBest *kb,
                             uint32_t K, uint32_t L_max,
                             uint32_t nt_idx, uint32_t L, uint32_t k_idx,
                             LZGFbgStep *steps, uint32_t *n_steps) {
    if (*n_steps >= LZG_FBG_MAX_STEPS)
        return LZG_FAIL(LZG_ERR_OVERFLOW, "top_k: too deep");

    size_t cell = ((size_t)nt_idx * (L_max + 1) + L) * K + k_idx;
    const KBest *e = &kb[cell];
    const LZGFbgNT *nt = &g->nts[nt_idx];
    const LZGFbgRule *rule = &g->rules[nt->rule_offset + e->rule_idx];

    LZGFbgStep *s = &steps[(*n_steps)++];
    s->kind      = rule->kind;
    s->a_char    = rule->a_char;
    s->z_char    = rule->z_char;
    s->a_run_len = rule->a_run_len;
    s->z_run_len = rule->z_run_len;
    s->dst_a     = rule->dst_a;
    s->dst_z     = rule->dst_z;
    s->is_start  = nt->is_start;

    if (rule->kind == LZG_FBG_RULE_INTERNAL) {
        return reconstruct(g, kb, K, L_max,
                           (uint32_t)e->child_nt, e->child_L, e->child_k,
                           steps, n_steps);
    }
    return LZG_OK;
}

LZGError lzg_fbg_top_k_sequences(const LZGFbg *g, uint32_t K,
                                  bool most_probable, uint32_t L_max,
                                  LZGSimResult *out, uint32_t *n_out) {
    if (!g || !out || !n_out || K == 0)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_top_k: bad args");

    uint32_t n = g->n_nts;
    /* kb[nt_idx, L, k] — contiguous K-best slots. */
    size_t total_cells = (size_t)n * (L_max + 1) * K;
    KBest *kb = (KBest *)calloc(total_cells, sizeof(KBest));
    uint32_t *counts = (uint32_t *)calloc((size_t)n * (L_max + 1),
                                           sizeof(uint32_t));
    if (!kb || !counts) {
        free(kb); free(counts);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_top_k: alloc");
    }

    for (uint32_t L = 0; L <= L_max; L++) {
        /* non-S first */
        for (uint32_t i = 1; i < n; i++) {
            size_t cell = (size_t)i * (L_max + 1) + L;
            uint32_t *cnt = &counts[cell];
            KBest *arr = &kb[cell * K];
            const LZGFbgNT *nt = &g->nts[i];
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                if (rule->weight <= 0.0) continue;
                double lw = log(rule->weight);

                switch ((LZGFbgRuleKind)rule->kind) {
                case LZG_FBG_RULE_LEAF_SINGLE:
                    if (L == 1) {
                        KBest e = {lw, (uint8_t)r, -1, 0, 0};
                        kbest_insert(arr, K, cnt, &e, most_probable);
                    }
                    break;
                case LZG_FBG_RULE_LEAF_RUN:
                    if (L == rule->a_run_len) {
                        KBest e = {lw, (uint8_t)r, -1, 0, 0};
                        kbest_insert(arr, K, cnt, &e, most_probable);
                    }
                    break;
                case LZG_FBG_RULE_LEAF_PAIR: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    if (L == ell) {
                        KBest e = {lw, (uint8_t)r, -1, 0, 0};
                        kbest_insert(arr, K, cnt, &e, most_probable);
                    }
                    break;
                }
                case LZG_FBG_RULE_INTERNAL: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    if (L < ell) break;
                    int32_t dst = g->nt_by_az[
                        (uint32_t)rule->dst_a * LZG_FBG_ALPHABET_SIZE
                        + (uint32_t)rule->dst_z];
                    if (dst < 0) break;
                    size_t dst_cell = (size_t)dst * (L_max + 1) + (L - ell);
                    uint32_t dst_cnt = counts[dst_cell];
                    const KBest *dst_arr = &kb[dst_cell * K];
                    for (uint32_t k = 0; k < dst_cnt; k++) {
                        KBest e = {
                            lw + dst_arr[k].log_prob,
                            (uint8_t)r, dst, L - ell, k
                        };
                        kbest_insert(arr, K, cnt, &e, most_probable);
                    }
                    break;
                }}
            }
        }
        /* S (zero-consumption) */
        if (g->nts[0].is_start) {
            size_t cell = L;
            uint32_t *cnt = &counts[cell];
            KBest *arr = &kb[cell * K];
            const LZGFbgNT *nt = &g->nts[0];
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                if (rule->kind != LZG_FBG_RULE_INTERNAL) continue;
                if (rule->weight <= 0.0) continue;
                double lw = log(rule->weight);
                int32_t dst = g->nt_by_az[
                    (uint32_t)rule->dst_a * LZG_FBG_ALPHABET_SIZE
                    + (uint32_t)rule->dst_z];
                if (dst < 0) continue;
                size_t dst_cell = (size_t)dst * (L_max + 1) + L;
                uint32_t dst_cnt = counts[dst_cell];
                const KBest *dst_arr = &kb[dst_cell * K];
                for (uint32_t k = 0; k < dst_cnt; k++) {
                    KBest e = {
                        lw + dst_arr[k].log_prob,
                        (uint8_t)r, dst, L, k
                    };
                    kbest_insert(arr, K, cnt, &e, most_probable);
                }
            }
        }
    }

    /* Collect K-best across all lengths at S. */
    KBest *final = (KBest *)calloc(K, sizeof(KBest));
    uint32_t final_cnt = 0;
    uint32_t *from_L = (uint32_t *)calloc(K, sizeof(uint32_t));
    uint32_t *from_k = (uint32_t *)calloc(K, sizeof(uint32_t));
    if (!final || !from_L || !from_k) {
        free(final); free(from_L); free(from_k);
        free(kb); free(counts);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_top_k: final alloc");
    }

    for (uint32_t L = 1; L <= L_max; L++) {
        size_t cell = L;
        uint32_t cnt = counts[cell];
        const KBest *arr = &kb[cell * K];
        for (uint32_t k = 0; k < cnt; k++) {
            /* Insert into final K-best with (L, k) bookkeeping. */
            uint32_t i;
            for (i = 0; i < final_cnt; i++) {
                bool better = most_probable
                    ? (arr[k].log_prob > final[i].log_prob)
                    : (arr[k].log_prob < final[i].log_prob);
                if (better) break;
            }
            if (i >= K) continue;
            uint32_t end = (final_cnt < K) ? final_cnt : K - 1;
            for (uint32_t j = end; j > i; j--) {
                final[j]  = final[j - 1];
                from_L[j] = from_L[j - 1];
                from_k[j] = from_k[j - 1];
            }
            final[i]  = arr[k];
            from_L[i] = L;
            from_k[i] = k;
            if (final_cnt < K) final_cnt++;
        }
    }

    /* Reconstruct each of the final_cnt best trees. */
    for (uint32_t i = 0; i < final_cnt; i++) {
        LZGFbgStep steps[LZG_FBG_MAX_STEPS];
        uint32_t n_steps = 0;
        LZGError err = reconstruct(g, kb, K, L_max,
                                    g->start_nt, from_L[i], from_k[i],
                                    steps, &n_steps);
        char buf[2048];
        uint32_t out_len = 0;
        if (err == LZG_OK) {
            err = lzg_fbg_tree_to_string(g, steps, n_steps, buf, sizeof(buf), &out_len);
        }
        if (err != LZG_OK) { buf[0] = '\0'; out_len = 0; }
        out[i].sequence = strdup(buf);
        out[i].seq_len  = out_len;
        out[i].n_tokens = n_steps;
        out[i].log_prob = final[i].log_prob;
    }
    *n_out = final_cnt;

    free(final); free(from_L); free(from_k);
    free(kb); free(counts);
    return LZG_OK;
}
