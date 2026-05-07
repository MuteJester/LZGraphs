/**
 * @file flashback_analytics.c
 * @brief Exact analytics for FlashBack graphs via forward DP.
 *
 * All computations are exact — no Monte Carlo. Uses the generic
 * lzg_forward_propagate() engine with appropriate callback sets.
 */
#include "lzgraph/flashback_graph.h"
#include "lzgraph/forward.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

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
