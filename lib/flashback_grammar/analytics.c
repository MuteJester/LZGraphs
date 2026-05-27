/**
 * @file analytics.c
 * @brief Exact analytics for FlashBackGrammar via small linear systems.
 *
 * Closed forms (reference: flashback_grammar_prototype.py):
 *   - path_count_series(L_max) / length_distribution(L_max):
 *       Z_i(x) = ℓ_i(x) + Σ_j q_ij(x) Z_j(x)
 *     Truncated polynomial iteration in L.
 *   - entropy:     H = (I - T)^-1 · h,  where h_i = -Σ p(r) log p(r)
 *   - power_sum(α): M_α = (I - T_α)^-1 · ℓ_α(1)  with shadow weights p(r)^α
 *   - hill(α):     D(α) = M_α[S]^{1/(1-α)}
 *
 * For n_nts ≤ a few hundred, dense Gaussian elimination is fast enough;
 * we don't need LAPACK.
 */
#include <stdlib.h>
#include <string.h>

#include "flashback_grammar_internal.h"

/* ── Gaussian elimination with partial pivoting ──────────────
 * Solves A · x = b in place. A is n×n row-major; b is length n.
 * After return, b contains x. A is destroyed.
 * Returns LZG_ERR_CONVERGENCE if A is singular (shouldn't happen for
 * (I − T) when ρ(T) < 1). */
static LZGError solve_linear(double *A, double *b, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) {
        /* Partial pivot: find the row with the largest |A[i,k]| in i >= k. */
        uint32_t piv = k;
        double max_abs = fabs(A[(size_t)k * n + k]);
        for (uint32_t i = k + 1; i < n; i++) {
            double a = fabs(A[(size_t)i * n + k]);
            if (a > max_abs) { max_abs = a; piv = i; }
        }
        if (max_abs < 1e-300)
            return LZG_FAIL(LZG_ERR_CONVERGENCE,
                            "fbg: singular matrix at column %u", k);
        if (piv != k) {
            /* Swap rows k and piv. */
            for (uint32_t j = 0; j < n; j++) {
                double t = A[(size_t)k * n + j];
                A[(size_t)k * n + j] = A[(size_t)piv * n + j];
                A[(size_t)piv * n + j] = t;
            }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double pivot = A[(size_t)k * n + k];
        for (uint32_t i = k + 1; i < n; i++) {
            double factor = A[(size_t)i * n + k] / pivot;
            if (factor == 0.0) continue;
            for (uint32_t j = k; j < n; j++)
                A[(size_t)i * n + j] -= factor * A[(size_t)k * n + j];
            b[i] -= factor * b[k];
        }
    }
    /* Back-substitute. */
    for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
        double s = b[i];
        for (uint32_t j = (uint32_t)i + 1; j < n; j++)
            s -= A[(size_t)i * n + j] * b[j];
        b[i] = s / A[(size_t)i * n + (uint32_t)i];
    }
    return LZG_OK;
}

/* ── Shared helpers ──────────────────────────────────────────── */

/* Build T[n*n] where T[i][j] = Σ_{internal r at Mi → Mj} p(r)^pow_α.
 * pow_α == 1.0 gives the plain transition matrix. */
static void build_T_pow(const LZGFlashbackGrammar *g, double alpha, double *T) {
    uint32_t n = g->n_nts;
    memset(T, 0, (size_t)n * n * sizeof(double));
    for (uint32_t i = 0; i < n; i++) {
        const LZGFlashbackGrammarNT *nt = &g->nts[i];
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            const LZGFlashbackGrammarRule *rule = &g->rules[nt->rule_offset + r];
            if (rule->kind != LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) continue;
            int32_t j = g->nt_by_az[
                (uint32_t)rule->dst_a * LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE
                + (uint32_t)rule->dst_z];
            if (j < 0) continue;
            double w = rule->weight;
            double w_alpha = (alpha == 1.0) ? w : pow(w, alpha);
            T[(size_t)i * n + (uint32_t)j] += w_alpha;
        }
    }
}

/* Build leaf mass vector ℓ_i(1) with weights p(r)^α. */
static void build_leaf_pow(const LZGFlashbackGrammar *g, double alpha, double *l) {
    uint32_t n = g->n_nts;
    memset(l, 0, (size_t)n * sizeof(double));
    for (uint32_t i = 0; i < n; i++) {
        const LZGFlashbackGrammarNT *nt = &g->nts[i];
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            const LZGFlashbackGrammarRule *rule = &g->rules[nt->rule_offset + r];
            if (rule->kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) continue;
            double w = rule->weight;
            double w_alpha = (alpha == 1.0) ? w : pow(w, alpha);
            l[i] += w_alpha;
        }
    }
}

/* Local entropy h_i = -Σ p(r) log p(r). */
static void build_local_entropy(const LZGFlashbackGrammar *g, double *h) {
    for (uint32_t i = 0; i < g->n_nts; i++) {
        const LZGFlashbackGrammarNT *nt = &g->nts[i];
        double hi = 0.0;
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            double w = g->rules[nt->rule_offset + r].weight;
            if (w > 0.0) hi -= w * log(w);
        }
        h[i] = hi;
    }
}

/* Form (I - A) in place (A is n×n row-major). */
static void form_I_minus_A(double *A, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double v = -A[(size_t)i * n + j];
            if (i == j) v += 1.0;
            A[(size_t)i * n + j] = v;
        }
    }
}

/* ── Path count / length distribution ────────────────────────── */

/* Fill out[L] for L = 0..L_max with the S-row of Z(x) coefficients.
 *   weighted=true  -> probability of generating a length-L sequence
 *   weighted=false -> count of distinct length-L sequences
 *
 * Because S's single rule consumes 0 output chars, S must be computed
 * AFTER all non-S NTs at each L. Our index convention puts S at index 0
 * and non-S at 1..n-1.
 */
static LZGError length_polynomial(const LZGFlashbackGrammar *g, uint32_t L_max,
                                   bool weighted, double *out_S) {
    uint32_t n = g->n_nts;
    if (n == 0) return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "fbg: empty grammar");

    /* Z[i * (L_max+1) + L] */
    size_t total = (size_t)n * (L_max + 1);
    double *Z = (double *)calloc(total, sizeof(double));
    if (!Z) return LZG_FAIL(LZG_ERR_ALLOC, "fbg: Z alloc");

    for (uint32_t L = 0; L <= L_max; L++) {
        /* Pass 1: non-S NTs (their rules all consume ≥1 char). */
        for (uint32_t i = 1; i < n; i++) {
            const LZGFlashbackGrammarNT *nt = &g->nts[i];
            double sum = 0.0;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFlashbackGrammarRule *rule = &g->rules[nt->rule_offset + r];
                double contrib = weighted ? rule->weight : 1.0;
                switch ((LZGFlashbackGrammarRuleKind)rule->kind) {
                case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE:
                    if (L == 1) sum += contrib;
                    break;
                case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN:
                    if (L == rule->a_run_len) sum += contrib;
                    break;
                case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    if (L == ell) sum += contrib;
                    break;
                }
                case LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL: {
                    uint32_t ell = (uint32_t)rule->a_run_len
                                 + (uint32_t)rule->z_run_len;
                    int32_t dst = g->nt_by_az[
                        (uint32_t)rule->dst_a * LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE
                        + (uint32_t)rule->dst_z];
                    if (dst >= 0 && L >= ell)
                        sum += contrib * Z[(size_t)dst * (L_max + 1) + (L - ell)];
                    break;
                }}
            }
            Z[(size_t)i * (L_max + 1) + L] = sum;
        }

        /* Pass 2: S — its peel(@, $) rules are zero output-consumption. */
        if (g->nts[0].is_start) {
            const LZGFlashbackGrammarNT *nt = &g->nts[0];
            double sum = 0.0;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFlashbackGrammarRule *rule = &g->rules[nt->rule_offset + r];
                if (rule->kind != LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) continue;
                double contrib = weighted ? rule->weight : 1.0;
                int32_t dst = g->nt_by_az[
                    (uint32_t)rule->dst_a * LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE
                    + (uint32_t)rule->dst_z];
                if (dst >= 0)
                    sum += contrib * Z[(size_t)dst * (L_max + 1) + L];
            }
            Z[L] = sum;
        }
    }

    /* Copy S row (index 0) to output. */
    for (uint32_t L = 0; L <= L_max; L++) out_S[L] = Z[L];
    free(Z);
    return LZG_OK;
}

LZGError lzg_flashback_grammar_path_count_series(const LZGFlashbackGrammar *g, uint32_t L_max, double *out) {
    if (!g || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_path_count_series");
    return length_polynomial(g, L_max, /*weighted=*/false, out);
}

LZGError lzg_flashback_grammar_length_distribution(const LZGFlashbackGrammar *g, uint32_t L_max, double *out) {
    if (!g || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_length_distribution");
    return length_polynomial(g, L_max, /*weighted=*/true, out);
}

/* ── Entropy ─────────────────────────────────────────────────── */

/* Fill H[i] for every NT via (I - T) H = h. Returns H_S in *H_start. */
LZGError lzg_flashback_grammar_entropy(const LZGFlashbackGrammar *g, double *H_nats) {
    if (!g || !H_nats) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_entropy");
    uint32_t n = g->n_nts;
    if (n == 0) return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "flashback_grammar_entropy: empty");

    double *T = (double *)malloc((size_t)n * n * sizeof(double));
    double *h = (double *)malloc((size_t)n * sizeof(double));
    if (!T || !h) { free(T); free(h); return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_entropy"); }

    build_T_pow(g, 1.0, T);
    build_local_entropy(g, h);
    form_I_minus_A(T, n);

    LZGError err = solve_linear(T, h, n);
    if (err != LZG_OK) { free(T); free(h); return err; }

    *H_nats = h[g->start_nt];
    free(T); free(h);
    return LZG_OK;
}

LZGError lzg_flashback_grammar_effective_diversity(const LZGFlashbackGrammar *g, LZGEffectiveDiversity *out) {
    if (!g || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_effective_diversity");
    memset(out, 0, sizeof(*out));

    double H;
    LZGError err = lzg_flashback_grammar_entropy(g, &H);
    if (err != LZG_OK) return err;

    out->entropy_nats = H;
    out->entropy_bits = H / log(2.0);
    out->effective_diversity = exp(H);

    /* uniformity = exp(H) / total_path_count. Use a reasonable L cap:
     * 4 × the training max length should contain nearly all mass. */
    uint32_t L_cap = g->max_length * 4;
    if (L_cap < 64) L_cap = 64;
    double *pc = (double *)calloc((size_t)L_cap + 1, sizeof(double));
    if (pc) {
        err = lzg_flashback_grammar_path_count_series(g, L_cap, pc);
        if (err == LZG_OK) {
            double total = 0.0;
            for (uint32_t L = 0; L <= L_cap; L++) total += pc[L];
            if (total > 0.0) {
                double u = out->effective_diversity / total;
                out->uniformity = (u > 1.0) ? 1.0 : u;
            }
        }
        free(pc);
    }
    return LZG_OK;
}

/* ── Power sum / Hill ────────────────────────────────────────── */

LZGError lzg_flashback_grammar_power_sum(const LZGFlashbackGrammar *g, double alpha, double *out) {
    if (!g || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_power_sum");
    if (fabs(alpha - 1.0) < 1e-15) {
        *out = 1.0;
        return LZG_OK;
    }
    /* α = 0 : shadow T_α has unit weights, ρ is almost always ≫ 1 for any
     * recursive grammar. The "power sum" = |support| which is infinite when
     * the grammar is recursive. Report as +∞ without attempting the solve. */
    if (alpha == 0.0) {
        *out = INFINITY;
        return LZG_OK;
    }

    uint32_t n = g->n_nts;
    double *T_a = (double *)malloc((size_t)n * n * sizeof(double));
    double *l_a = (double *)malloc((size_t)n * sizeof(double));
    if (!T_a || !l_a) { free(T_a); free(l_a); return LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_power_sum"); }

    build_T_pow(g, alpha, T_a);
    build_leaf_pow(g, alpha, l_a);

    /* Guard: shadow grammar is only stable when ρ(T_α) < 1. Under α < 1,
     * rule weights p^α > p, so T_α can grow super-stochastic. Without this
     * check the solve returns garbage (negative or huge values). */
    double rho = lzg_flashback_grammar_spectral_radius_dense(T_a, n);
    if (rho >= 1.0 - 1e-9) {
        free(T_a); free(l_a);
        *out = INFINITY;
        LZG_WARN("flashback_grammar_power_sum: shadow spectral radius ρ(T_α)=%.6f ≥ 1 at "
                 "α=%.4f; returning +∞", rho, alpha);
        return LZG_OK;
    }

    form_I_minus_A(T_a, n);
    LZGError err = solve_linear(T_a, l_a, n);
    if (err != LZG_OK) { free(T_a); free(l_a); return err; }

    double v = l_a[g->start_nt];
    /* Defence-in-depth: if the solve somehow produced negative (ill-conditioned
     * matrix near the stability boundary), report +∞ rather than a nonsense
     * number that would poison downstream Hill computation. */
    if (!(v > 0.0) || !isfinite(v)) {
        v = INFINITY;
        LZG_WARN("flashback_grammar_power_sum: solve gave non-positive %.4g at α=%.4f; "
                 "returning +∞", l_a[g->start_nt], alpha);
    }
    *out = v;
    free(T_a); free(l_a);
    return LZG_OK;
}

LZGError lzg_flashback_grammar_hill_number(const LZGFlashbackGrammar *g, double alpha, double *out) {
    if (!g || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_hill_number");
    if (fabs(alpha - 1.0) < 1e-15) {
        double H;
        LZGError err = lzg_flashback_grammar_entropy(g, &H);
        if (err != LZG_OK) return err;
        *out = exp(H);
        return LZG_OK;
    }
    double M_a;
    LZGError err = lzg_flashback_grammar_power_sum(g, alpha, &M_a);
    if (err != LZG_OK) return err;
    if (!isfinite(M_a) || M_a <= 0.0) {
        /* Unstable shadow grammar (ρ(T_α) ≥ 1) → D(α) is +∞. For
         * α < 1 this means the grammar has an unbounded support, which is
         * the mathematically correct answer for a recursive PCFG at α=0. */
        *out = INFINITY;
        return LZG_OK;
    }
    *out = pow(M_a, 1.0 / (1.0 - alpha));
    return LZG_OK;
}

LZGError lzg_flashback_grammar_hill_numbers(const LZGFlashbackGrammar *g,
                              const double *alphas, uint32_t n,
                              double *out) {
    if (!g || !alphas || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_hill_numbers");
    for (uint32_t i = 0; i < n; i++) {
        LZGError err = lzg_flashback_grammar_hill_number(g, alphas[i], &out[i]);
        if (err != LZG_OK) return err;
    }
    return LZG_OK;
}
