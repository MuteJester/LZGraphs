/**
 * @file posterior.c
 * @brief Dirichlet-Multinomial posterior FlashBackGrammar.
 *
 * Reference: flashback_grammar_prototype.py `posterior`.
 *
 *   w_post(r | nt) = (κ · w_prior(r | nt) + c_ind(r | nt)) / (κ + n_ind(nt))
 *
 * κ = 0       → pure individual MLE (restricted to prior topology)
 * κ → ∞       → posterior → prior
 * Individual rules not present in prior are ignored.
 *
 * Mutation hygiene: decomposition uses stack LZGFlashbackGrammarStep[], so prior is never
 * mutated. Alphabet is copied into the new grammar.
 */
#include <stdlib.h>
#include <string.h>

#include "flashback_grammar_internal.h"

LZGError lzg_flashback_grammar_posterior(const LZGFlashbackGrammar *prior,
                            const char **sequences, uint32_t n_seqs,
                            const uint64_t *abundances,
                            double kappa,
                            LZGFlashbackGrammar **out) {
    if (!prior || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_posterior: NULL argument");
    if (kappa < 0.0)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                        "flashback_grammar_posterior: kappa must be >= 0, got %g", kappa);
    if (n_seqs > 0 && !sequences)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_posterior: sequences NULL");

    /* Step 1: accumulate individual counts, restricted to prior topology. */
    double *ind_per_rule = (double *)calloc(prior->n_rules, sizeof(double));
    double *ind_per_nt   = (double *)calloc(prior->n_nts, sizeof(double));
    if (!ind_per_rule || !ind_per_nt) {
        free(ind_per_rule); free(ind_per_nt);
        return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_posterior: alloc");
    }

    LZGFlashbackGrammarStep steps[LZG_FLASHBACK_GRAMMAR_MAX_STEPS];
    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        uint32_t len = (uint32_t)strlen(seq);
        if (len == 0) continue;
        uint64_t a = abundances ? abundances[s] : 1;
        double w = lzg_flashback_grammar_abund_weight(prior->abundance_mode, a);
        if (w <= 0.0) continue;

        uint32_t n_steps = 0;
        LZGError err = lzg_flashback_grammar_decompose(prior, seq, len, steps, &n_steps);
        if (err != LZG_OK) continue;  /* out-of-alphabet seq → skip */

        for (uint32_t t = 0; t < n_steps; t++) {
            int32_t nt_idx = lzg_flashback_grammar_find_nt_for_step(prior, &steps[t]);
            if (nt_idx < 0) continue;
            int32_t rel = lzg_flashback_grammar_find_rule_in_nt(prior, (uint32_t)nt_idx, &steps[t]);
            if (rel < 0) continue;
            uint32_t abs_idx = prior->nts[nt_idx].rule_offset + (uint32_t)rel;
            ind_per_rule[abs_idx] += w;
            ind_per_nt[nt_idx] += w;
        }
    }

    /* Step 2: allocate posterior and copy structure from prior. */
    LZGFlashbackGrammar *post = lzg_flashback_grammar_new();
    if (!post) {
        free(ind_per_rule); free(ind_per_nt);
        return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_posterior: grammar alloc");
    }

    memcpy(post->char_to_idx, prior->char_to_idx, 256);
    memcpy(post->idx_to_char, prior->idx_to_char, LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE);
    post->alphabet_size = prior->alphabet_size;

    post->n_nts = prior->n_nts;
    post->start_nt = prior->start_nt;
    post->n_rules = prior->n_rules;
    post->n_internal = prior->n_internal;

    post->nts = (LZGFlashbackGrammarNT *)malloc((size_t)post->n_nts * sizeof(LZGFlashbackGrammarNT));
    post->rules = (LZGFlashbackGrammarRule *)malloc((size_t)post->n_rules * sizeof(LZGFlashbackGrammarRule));
    if (!post->nts || !post->rules) {
        lzg_flashback_grammar_destroy(post);
        free(ind_per_rule); free(ind_per_nt);
        return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_posterior: copy alloc");
    }
    memcpy(post->nts, prior->nts, (size_t)post->n_nts * sizeof(LZGFlashbackGrammarNT));
    memcpy(post->rules, prior->rules, (size_t)post->n_rules * sizeof(LZGFlashbackGrammarRule));

    /* Step 3: DM update per NT. */
    for (uint32_t i = 0; i < post->n_nts; i++) {
        LZGFlashbackGrammarNT *nt = &post->nts[i];
        double n_ind = ind_per_nt[i];
        double denom = kappa + n_ind;

        if (denom <= 0.0) {
            /* No information — preserve prior weights. Use a nominal
             * total_count = 1.0 so that derivative operations (backoff,
             * subtract) have a sensible denominator. Rule counts and
             * weights are already the prior's. */
            nt->total_count = 1.0;
            nt->unseen_mass = 0.0;
            continue;
        }

        double total_cnt = 0.0;
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            uint32_t abs_idx = nt->rule_offset + r;
            double w_prior = prior->rules[abs_idx].weight;
            double c_ind   = ind_per_rule[abs_idx];
            double post_count = kappa * w_prior + c_ind;
            post->rules[abs_idx].count  = (uint64_t)(post_count + 0.5);
            post->rules[abs_idx].weight = post_count / denom;
            total_cnt += post_count;
        }
        nt->total_count = total_cnt;
        nt->unseen_mass = 0.0;   /* refilled below if backoff == GT */
    }

    /* Step 4: copy length counts verbatim. Posterior inherits prior's
     * training-length histogram — it's a read-only artifact. */
    if (prior->max_length > 0 && prior->length_counts) {
        size_t n = (size_t)prior->max_length + 1;
        post->length_counts = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!post->length_counts) {
            lzg_flashback_grammar_destroy(post);
            free(ind_per_rule); free(ind_per_nt);
            return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_posterior: length_counts alloc");
        }
        memcpy(post->length_counts, prior->length_counts, n * sizeof(uint64_t));
        post->max_length = prior->max_length;
    }

    /* Step 5: config + finalize. */
    post->abundance_mode = prior->abundance_mode;
    post->smoothing = 0.0;  /* DM is the smoother — don't double-smooth */
    post->backoff = prior->backoff;

    LZGError err = lzg_flashback_grammar_finalize(post);
    free(ind_per_rule); free(ind_per_nt);
    if (err != LZG_OK) { lzg_flashback_grammar_destroy(post); return err; }

    *out = post;
    return LZG_OK;
}
