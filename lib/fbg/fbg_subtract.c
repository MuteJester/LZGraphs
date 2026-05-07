/**
 * @file fbg_subtract.c
 * @brief Remove the contribution of a list of sequences from a grammar.
 *
 * Reference: fbg_prototype.py `without`.
 *
 * For each sequence (abundance a), decompose canonically, decrement the
 * matching rule's count by a's abundance-mode weight (clamped at 0).
 * Prune rules whose count hits 0, and NTs whose rule count becomes 0.
 * Raises if the result is empty or if the start NT is removed.
 */
#include "fbg_internal.h"
#include <stdlib.h>
#include <string.h>

/* Recover the pre-normalization fractional count of a rule:
 *   frac_count = weight · (total + smoothing · n_rules) − smoothing
 * With smoothing=0 this simplifies to weight · total_count. */
static inline double recover_frac_count(const LZGFbg *g,
                                         const LZGFbgNT *nt,
                                         const LZGFbgRule *rule) {
    double denom = nt->total_count + g->smoothing * (double)nt->n_rules;
    double c = rule->weight * denom - g->smoothing;
    return c < 0.0 ? 0.0 : c;
}

LZGError lzg_fbg_subtract(const LZGFbg *g,
                           const char **sequences, uint32_t n_seqs,
                           const uint64_t *abundances,
                           LZGFbg **out) {
    if (!g || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_subtract: NULL argument");
    if (n_seqs > 0 && !sequences)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_subtract: sequences NULL");

    /* Step 1: accumulate subtraction amounts per absolute rule index. */
    double *sub = (double *)calloc(g->n_rules, sizeof(double));
    if (!sub) return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: sub alloc");

    LZGFbgStep steps[LZG_FBG_MAX_STEPS];
    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        uint32_t len = (uint32_t)strlen(seq);
        if (len == 0) continue;
        uint64_t a = abundances ? abundances[s] : 1;
        double w = lzg_fbg_abund_weight(g->abundance_mode, a);
        if (w <= 0.0) continue;

        uint32_t n_steps = 0;
        LZGError err = lzg_fbg_decompose(g, seq, len, steps, &n_steps);
        if (err != LZG_OK) continue;

        for (uint32_t t = 0; t < n_steps; t++) {
            int32_t nt_idx = lzg_fbg_find_nt_for_step(g, &steps[t]);
            if (nt_idx < 0) continue;
            int32_t rel = lzg_fbg_find_rule_in_nt(g, (uint32_t)nt_idx, &steps[t]);
            if (rel < 0) continue;
            uint32_t abs_idx = g->nts[nt_idx].rule_offset + (uint32_t)rel;
            sub[abs_idx] += w;
        }
    }

    /* Step 2: compute surviving fractional count per rule, mark kept rules. */
    double  *surv      = (double *)calloc(g->n_rules, sizeof(double));
    uint8_t *keep_rule = (uint8_t *)calloc(g->n_rules, sizeof(uint8_t));
    uint8_t *keep_nt   = (uint8_t *)calloc(g->n_nts, sizeof(uint8_t));
    if (!surv || !keep_rule || !keep_nt) {
        free(sub); free(surv); free(keep_rule); free(keep_nt);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: scratch alloc");
    }

    uint32_t n_keep_rules = 0;
    uint32_t n_keep_nts = 0;
    for (uint32_t i = 0; i < g->n_nts; i++) {
        const LZGFbgNT *nt = &g->nts[i];
        uint32_t kept_at_nt = 0;
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            uint32_t abs_idx = nt->rule_offset + r;
            double frac = recover_frac_count(g, nt, &g->rules[abs_idx]);
            double new_c = frac - sub[abs_idx];
            if (new_c > 1e-9) {
                surv[abs_idx] = new_c;
                keep_rule[abs_idx] = 1;
                kept_at_nt++;
                n_keep_rules++;
            }
        }
        if (kept_at_nt > 0) {
            keep_nt[i] = 1;
            n_keep_nts++;
        }
    }
    free(sub);

    if (n_keep_nts == 0) {
        free(surv); free(keep_rule); free(keep_nt);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "fbg_subtract: would leave an empty grammar");
    }
    if (!keep_nt[g->start_nt]) {
        free(surv); free(keep_rule); free(keep_nt);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "fbg_subtract: would remove the start non-terminal");
    }

    /* Step 3: build old→new NT mapping (S first, others in original order). */
    int32_t *old_to_new = (int32_t *)malloc((size_t)g->n_nts * sizeof(int32_t));
    if (!old_to_new) {
        free(surv); free(keep_rule); free(keep_nt);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: map alloc");
    }
    for (uint32_t i = 0; i < g->n_nts; i++) old_to_new[i] = -1;
    old_to_new[g->start_nt] = 0;
    uint32_t next_new = 1;
    for (uint32_t i = 0; i < g->n_nts; i++) {
        if (i == g->start_nt) continue;
        if (keep_nt[i]) old_to_new[i] = (int32_t)(next_new++);
    }

    /* Step 4: allocate the new grammar. */
    LZGFbg *sub_g = lzg_fbg_new();
    if (!sub_g) {
        free(surv); free(keep_rule); free(keep_nt); free(old_to_new);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: grammar alloc");
    }

    memcpy(sub_g->char_to_idx, g->char_to_idx, 256);
    memcpy(sub_g->idx_to_char, g->idx_to_char, LZG_FBG_ALPHABET_SIZE);
    sub_g->alphabet_size = g->alphabet_size;
    sub_g->n_nts = n_keep_nts;
    sub_g->start_nt = 0;
    sub_g->n_rules = n_keep_rules;

    sub_g->nts = (LZGFbgNT *)calloc(n_keep_nts, sizeof(LZGFbgNT));
    sub_g->rules = (LZGFbgRule *)calloc(n_keep_rules, sizeof(LZGFbgRule));
    if (!sub_g->nts || !sub_g->rules) {
        lzg_fbg_destroy(sub_g);
        free(surv); free(keep_rule); free(keep_nt); free(old_to_new);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: nts/rules alloc");
    }

    /* Step 5: copy surviving rules in new ordering. */
    uint32_t rule_cursor = 0;
    uint32_t n_internal = 0;

    for (uint32_t new_i = 0; new_i < n_keep_nts; new_i++) {
        /* Find the old NT that maps to new_i. */
        uint32_t old_i = UINT32_MAX;
        if (new_i == 0) {
            old_i = g->start_nt;
        } else {
            for (uint32_t k = 0; k < g->n_nts; k++) {
                if (old_to_new[k] == (int32_t)new_i) { old_i = k; break; }
            }
        }
        const LZGFbgNT *old_nt = &g->nts[old_i];
        LZGFbgNT *new_nt = &sub_g->nts[new_i];
        *new_nt = *old_nt;
        new_nt->rule_offset = rule_cursor;
        new_nt->n_rules = 0;

        double total = 0.0;
        for (uint32_t r = 0; r < old_nt->n_rules; r++) {
            uint32_t abs_old = old_nt->rule_offset + r;
            if (!keep_rule[abs_old]) continue;
            LZGFbgRule *new_rule = &sub_g->rules[rule_cursor];
            *new_rule = g->rules[abs_old];
            /* Stash fractional count in weight for now; MLE below normalises. */
            new_rule->weight = surv[abs_old];
            total += surv[abs_old];
            if (new_rule->kind == LZG_FBG_RULE_INTERNAL) n_internal++;
            rule_cursor++;
            new_nt->n_rules++;
        }
        new_nt->total_count = total;
        new_nt->unseen_mass = 0.0;
    }
    sub_g->n_internal = n_internal;

    /* Step 6: apply MLE with the parent's smoothing. */
    for (uint32_t i = 0; i < sub_g->n_nts; i++) {
        LZGFbgNT *nt = &sub_g->nts[i];
        double denom = nt->total_count + g->smoothing * (double)nt->n_rules;
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            uint32_t abs_idx = nt->rule_offset + r;
            double frac = sub_g->rules[abs_idx].weight;  /* stashed count */
            sub_g->rules[abs_idx].count = (uint64_t)(frac + 0.5);
            sub_g->rules[abs_idx].weight = denom > 0.0
                ? (frac + g->smoothing) / denom
                : 0.0;
        }
    }

    /* Step 7: copy length counts (inherited; subtract doesn't recompute). */
    if (g->max_length > 0 && g->length_counts) {
        size_t n = (size_t)g->max_length + 1;
        sub_g->length_counts = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!sub_g->length_counts) {
            lzg_fbg_destroy(sub_g);
            free(surv); free(keep_rule); free(keep_nt); free(old_to_new);
            return LZG_FAIL(LZG_ERR_ALLOC, "fbg_subtract: length_counts");
        }
        memcpy(sub_g->length_counts, g->length_counts, n * sizeof(uint64_t));
        sub_g->max_length = g->max_length;
    }

    /* Step 8: config + finalize. */
    sub_g->abundance_mode = g->abundance_mode;
    sub_g->smoothing = g->smoothing;
    sub_g->backoff = g->backoff;

    free(surv); free(keep_rule); free(keep_nt); free(old_to_new);

    LZGError err = lzg_fbg_finalize(sub_g);
    if (err != LZG_OK) { lzg_fbg_destroy(sub_g); return err; }

    *out = sub_g;
    return LZG_OK;
}
