/**
 * @file flashback_analytics.c
 * @brief Exact analytics for FlashBack graphs via forward DP.
 *
 * All computations are exact — no Monte Carlo. Uses the generic
 * lzg_forward_propagate() engine with appropriate callback sets.
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "lzgraph/flashback_graph.h"
#include "lzgraph/forward.h"

/* ═══════════════════════════════════════════════════════════════ */
/* Path count (exact): count distinct root-to-sink walks          */
/* ═══════════════════════════════════════════════════════════════ */

static void pc_seed(double *acc, double p, void *ctx) {
    (void)ctx; (void)p;
    acc[0] = 1.0;
}

static void pc_edge(double *dst, const double *src,
                    double w, double z, void *ctx) {
    (void)ctx; (void)w; (void)z;
    dst[0] = src[0]; /* each path continues through each edge */
}

static void pc_absorb(double *total, const double *node,
                      double sp, void *ctx) {
    (void)ctx; (void)sp;
    total[0] += node[0];
}

LZGError lzg_flashback_path_count(const LZGGraph *g, double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    LZGFwdOps ops = { pc_seed, pc_edge, pc_absorb, NULL, 1, NULL };
    double total = 0.0;
    LZGError err = lzg_forward_propagate(g, &ops, &total);
    if (err != LZG_OK) return err;
    *out = total;
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Path count (exact, arbitrary precision)                        */
/*                                                                 */
/* The double-accumulator version above saturates at 2^53 and      */
/* overflows to +inf past ~1.8e308. Real repertoire graphs exceed  */
/* 2^53 easily, so this variant carries the count in base-2^32     */
/* limbs and returns every digit. Only addition is required: the   */
/* DP is counts[v] += counts[u] over a topological order.          */
/* ═══════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *limbs;  /* little-endian base 2^32; limbs[>= n] are zero */
    uint32_t  n;      /* significant limbs; 0 means the value is zero  */
    uint32_t  cap;
} LZGBigU;

/* Grow so that at least `need` limbs are addressable, zero-filling. */
static bool bigu_reserve(LZGBigU *a, uint32_t need) {
    if (a->cap >= need) return true;
    uint32_t ncap = a->cap ? a->cap : 2;
    while (ncap < need) {
        if (ncap > UINT32_MAX / 2) return false;
        ncap *= 2;
    }
    uint32_t *p = (uint32_t *)realloc(a->limbs, (size_t)ncap * sizeof(uint32_t));
    if (!p) return false;
    memset(p + a->cap, 0, (size_t)(ncap - a->cap) * sizeof(uint32_t));
    a->limbs = p;
    a->cap = ncap;
    return true;
}

/* dst += src */
static bool bigu_add(LZGBigU *dst, const LZGBigU *src) {
    if (src->n == 0) return true;
    uint32_t maxn = dst->n > src->n ? dst->n : src->n;
    if (!bigu_reserve(dst, maxn + 1)) return false;
    uint64_t carry = 0;
    for (uint32_t i = 0; i < maxn; i++) {
        uint64_t s = (uint64_t)dst->limbs[i] + carry;
        if (i < src->n) s += (uint64_t)src->limbs[i];
        dst->limbs[i] = (uint32_t)s;
        carry = s >> 32;
    }
    dst->n = maxn;
    if (carry) {
        dst->limbs[maxn] = (uint32_t)carry;
        dst->n = maxn + 1;
    }
    return true;
}

static void bigu_free(LZGBigU *a) {
    free(a->limbs);
    a->limbs = NULL;
    a->n = 0;
    a->cap = 0;
}

LZGError lzg_flashback_path_count_exact(const LZGGraph *g,
                                        uint32_t **limbs_out,
                                        uint32_t *n_limbs_out) {
    if (!g || !limbs_out || !n_limbs_out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;
    *limbs_out = NULL;
    *n_limbs_out = 0;

    const uint32_t n_nodes = g->n_nodes;
    if (n_nodes == 0 || g->root_node >= n_nodes) return LZG_ERR_NOT_BUILT;

    LZGBigU *acc = (LZGBigU *)calloc(n_nodes, sizeof(LZGBigU));
    if (!acc) return LZG_ERR_ALLOC;

    LZGBigU total = {NULL, 0, 0};
    LZGError err = LZG_OK;

    /* Seed the root with exactly one path. */
    if (!bigu_reserve(&acc[g->root_node], 1)) {
        err = LZG_ERR_ALLOC;
        goto done;
    }
    acc[g->root_node].limbs[0] = 1;
    acc[g->root_node].n = 1;

    for (uint32_t t = 0; t < n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        LZGBigU *src = &acc[u];
        if (src->n == 0) continue; /* unreachable from the root */

        if (g->node_is_sink && g->node_is_sink[u]) {
            if (!bigu_add(&total, src)) { err = LZG_ERR_ALLOC; goto done; }
        } else {
            uint32_t e_end = g->row_offsets[u + 1];
            for (uint32_t e = g->row_offsets[u]; e < e_end; e++) {
                if (!bigu_add(&acc[g->col_indices[e]], src)) {
                    err = LZG_ERR_ALLOC;
                    goto done;
                }
            }
        }
        /* u is never revisited in topological order: release it now so
           peak memory tracks the DAG frontier rather than the whole graph. */
        bigu_free(src);
    }

    *limbs_out = total.limbs;
    *n_limbs_out = total.n;
    total.limbs = NULL; /* ownership transferred to the caller */

done:
    for (uint32_t i = 0; i < n_nodes; i++) bigu_free(&acc[i]);
    free(acc);
    if (err != LZG_OK) bigu_free(&total);
    return err;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shannon entropy (exact)                                        */
/* acc[0] = probability mass, acc[1] = entropy accumulator        */
/* ═══════════════════════════════════════════════════════════════ */

static void ent_seed(double *acc, double p, void *ctx) {
    (void)ctx;
    acc[0] = p;
    acc[1] = 0.0;
}

static void ent_edge(double *dst, const double *src,
                     double w, double z, void *ctx) {
    (void)ctx;
    double p = w / z;
    double lp = log(p);
    dst[0] = src[0] * p;
    dst[1] = src[1] * p + src[0] * p * (-lp);
}

static void ent_absorb(double *total, const double *node,
                       double sp, void *ctx) {
    (void)ctx; (void)sp;
    total[0] += node[0];
    total[1] += node[1];
}

LZGError lzg_flashback_effective_diversity(const LZGGraph *g,
                                           LZGEffectiveDiversity *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    LZGFwdOps ops = { ent_seed, ent_edge, ent_absorb, NULL, 2, NULL };
    double total[2] = {0.0, 0.0};
    LZGError err = lzg_forward_propagate(g, &ops, total);
    if (err != LZG_OK) return err;

    double absorbed = total[0];
    if (absorbed < LZG_EPS) {
        memset(out, 0, sizeof(*out));
        return LZG_OK;
    }

    double H = total[1] / absorbed;
    out->entropy_nats = H;
    out->entropy_bits = H / log(2.0);
    out->effective_diversity = exp(H);

    double path_count = 0.0;
    (void)lzg_flashback_path_count(g, &path_count);
    out->uniformity = path_count > 0.0
        ? fmin(out->effective_diversity / path_count, 1.0)
        : 0.0;

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Power sum M(α) = Σ P(s)^α                                     */
/* ═══════════════════════════════════════════════════════════════ */

static void pow_seed(double *acc, double p, void *ctx) {
    double alpha = *(double *)ctx;
    acc[0] = pow(p, alpha);
}

static void pow_edge(double *dst, const double *src,
                     double w, double z, void *ctx) {
    double alpha = *(double *)ctx;
    dst[0] = src[0] * pow(w / z, alpha);
}

static void pow_absorb(double *total, const double *node,
                       double sp, void *ctx) {
    (void)ctx; (void)sp;
    total[0] += node[0];
}

LZGError lzg_flashback_power_sum(const LZGGraph *g, double alpha,
                                  double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (fabs(alpha) < 1e-15)
        return lzg_flashback_path_count(g, out);
    if (fabs(alpha - 1.0) < 1e-15) {
        *out = 1.0;
        return LZG_OK;
    }
    LZGFwdOps ops = { pow_seed, pow_edge, pow_absorb, NULL, 1, &alpha };
    double total = 0.0;
    LZGError err = lzg_forward_propagate(g, &ops, &total);
    if (err != LZG_OK) return err;
    *out = total;
    return LZG_OK;
}

LZGError lzg_flashback_hill_number(const LZGGraph *g, double alpha,
                                    double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (fabs(alpha) < 1e-15)
        return lzg_flashback_path_count(g, out);
    if (fabs(alpha - 1.0) < 1e-15) {
        LZGEffectiveDiversity ed;
        LZGError err = lzg_flashback_effective_diversity(g, &ed);
        if (err != LZG_OK) return err;
        *out = ed.effective_diversity;
        return LZG_OK;
    }
    double m;
    LZGError err = lzg_flashback_power_sum(g, alpha, &m);
    if (err != LZG_OK) return err;
    *out = m < LZG_EPS ? 0.0 : pow(m, 1.0 / (1.0 - alpha));
    return LZG_OK;
}

LZGError lzg_flashback_hill_numbers(const LZGGraph *g,
                                     const double *orders,
                                     uint32_t n, double *out) {
    if (!g || !orders || !out) return LZG_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < n; i++) {
        LZGError err = lzg_flashback_hill_number(g, orders[i], &out[i]);
        if (err != LZG_OK) return err;
    }
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Dynamic range: min/max log-probability via topo-order DP       */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_flashback_dynamic_range(const LZGGraph *g,
                                      LZGDynamicRange *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;

    uint32_t nn = g->n_nodes;
    double *max_lp = malloc(nn * sizeof(double));
    double *min_lp = malloc(nn * sizeof(double));
    if (!max_lp || !min_lp) {
        free(max_lp); free(min_lp);
        return LZG_ERR_ALLOC;
    }

    /* Initialize: -inf for max, +inf for min, except root = 0 */
    for (uint32_t i = 0; i < nn; i++) {
        max_lp[i] = -1e300;
        min_lp[i] =  1e300;
    }
    if (g->root_node < nn) {
        max_lp[g->root_node] = 0.0;
        min_lp[g->root_node] = 0.0;
    }

    /* Forward pass in topo order */
    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        if (max_lp[u] < -1e299) continue; /* unreachable */

        uint32_t e_start = g->row_offsets[u];
        uint32_t e_end   = g->row_offsets[u + 1];
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = g->col_indices[e];
            double w = g->edge_weights[e];
            if (w < LZG_EPS) continue;
            double lw = log(w);

            double candidate_max = max_lp[u] + lw;
            double candidate_min = min_lp[u] + lw;
            if (candidate_max > max_lp[v]) max_lp[v] = candidate_max;
            if (candidate_min < min_lp[v]) min_lp[v] = candidate_min;
        }
    }

    /* Collect over sinks */
    double global_max = -1e300, global_min = 1e300;
    bool found = false;
    for (uint32_t i = 0; i < nn; i++) {
        if (!g->node_is_sink || !g->node_is_sink[i]) continue;
        if (max_lp[i] < -1e299) continue;
        found = true;
        if (max_lp[i] > global_max) global_max = max_lp[i];
        if (min_lp[i] < global_min) global_min = min_lp[i];
    }

    free(max_lp);
    free(min_lp);

    if (!found) {
        memset(out, 0, sizeof(*out));
        return LZG_OK;
    }

    out->max_log_prob = global_max;
    out->min_log_prob = global_min;
    out->dynamic_range_nats = global_max - global_min;
    out->dynamic_range_orders = out->dynamic_range_nats / log(10.0);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* PGEN diagnostics (exact for Markovian)                         */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_flashback_pgen_diagnostics(const LZGGraph *g, double atol,
                                         LZGPgenDiagnostics *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    LZGFwdOps ops = { ent_seed, ent_edge, ent_absorb, NULL, 2, NULL };
    double total[2] = {0.0, 0.0};
    LZGError err = lzg_forward_propagate(g, &ops, total);
    if (err != LZG_OK) return err;

    out->total_absorbed = total[0];
    out->total_leaked = 1.0 - total[0];
    out->initial_prob_sum = 1.0;
    out->is_proper = fabs(total[0] - 1.0) < atol;
    out->mc_samples = 0;
    return LZG_OK;
}
