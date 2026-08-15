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

/* Count residues contributed by a node using the same token rule as the
 * Python p-sequence analysis: inspect the base before the final underscore
 * and exclude the @/$ sentinels. */
static uint32_t pc_symbol_length(const LZGGraph *g, uint32_t node) {
    const char *label = lzg_sp_get(g->pool, g->node_sp_id[node]);
    const char *last_underscore = strrchr(label, '_');
    const char *end = last_underscore ? last_underscore : label + strlen(label);
    uint32_t length = 0;
    for (const char *p = label; p < end; p++)
        if (*p != '@' && *p != '$') length++;
    return length;
}

LZGError lzg_flashback_pseq_init(const LZGGraph *g,
                                 uint8_t *symbol_lengths_out,
                                 double *min_surprisal_out,
                                 double *max_surprisal_out,
                                 uint32_t *max_edges_out) {
    if (!g || !symbol_lengths_out || !min_surprisal_out ||
        !max_surprisal_out || !max_edges_out)
        return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    const uint32_t nn = g->n_nodes;
    double *min_x = (double *)malloc((size_t)nn * sizeof(double));
    double *max_x = (double *)malloc((size_t)nn * sizeof(double));
    uint32_t *edge_depth = (uint32_t *)calloc(nn, sizeof(uint32_t));
    uint8_t *reachable = (uint8_t *)calloc(nn, sizeof(uint8_t));
    if (!min_x || !max_x || !edge_depth || !reachable) {
        free(min_x); free(max_x); free(edge_depth); free(reachable);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t u = 0; u < nn; u++) {
        uint32_t length = pc_symbol_length(g, u);
        if (length > UINT8_MAX) {
            free(min_x); free(max_x); free(edge_depth); free(reachable);
            return LZG_ERR_INVALID_ARG;
        }
        symbol_lengths_out[u] = (uint8_t)length;
        min_x[u] = INFINITY;
        max_x[u] = -INFINITY;
    }
    min_x[g->root_node] = 0.0;
    max_x[g->root_node] = 0.0;
    reachable[g->root_node] = 1;

    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        if (!reachable[u]) continue;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            double increment = -log(g->edge_weights[e]);
            double candidate_min = min_x[u] + increment;
            double candidate_max = max_x[u] + increment;
            uint32_t candidate_depth = edge_depth[u] + 1;
            if (candidate_min < min_x[v]) min_x[v] = candidate_min;
            if (candidate_max > max_x[v]) max_x[v] = candidate_max;
            if (!reachable[v] || candidate_depth > edge_depth[v])
                edge_depth[v] = candidate_depth;
            reachable[v] = 1;
        }
    }

    double true_min = INFINITY;
    double true_max = -INFINITY;
    uint32_t true_max_edges = 0;
    bool found_sink = false;
    for (uint32_t u = 0; u < nn; u++) {
        if (!reachable[u] || g->row_offsets[u] != g->row_offsets[u + 1]) continue;
        if (min_x[u] < true_min) true_min = min_x[u];
        if (max_x[u] > true_max) true_max = max_x[u];
        if (edge_depth[u] > true_max_edges) true_max_edges = edge_depth[u];
        found_sink = true;
    }

    free(min_x); free(max_x); free(edge_depth); free(reachable);
    if (!found_sink) return LZG_ERR_NOT_BUILT;
    *min_surprisal_out = true_min;
    *max_surprisal_out = true_max;
    *max_edges_out = true_max_edges;
    return LZG_OK;
}

LZGError lzg_flashback_path_count_by_length(const LZGGraph *g,
                                            double **counts_out,
                                            uint32_t *max_length_out) {
    if (!g || !counts_out || !max_length_out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;
    *counts_out = NULL;
    *max_length_out = 0;

    const uint32_t nn = g->n_nodes;
    if (nn == 0 || g->root_node >= nn) return LZG_ERR_NOT_BUILT;

    uint32_t *symbol_length = (uint32_t *)malloc((size_t)nn * sizeof(uint32_t));
    uint32_t *min_length = (uint32_t *)malloc((size_t)nn * sizeof(uint32_t));
    uint32_t *max_length = (uint32_t *)calloc(nn, sizeof(uint32_t));
    if (!symbol_length || !min_length || !max_length) {
        free(symbol_length); free(min_length); free(max_length);
        return LZG_ERR_ALLOC;
    }
    for (uint32_t u = 0; u < nn; u++) {
        symbol_length[u] = pc_symbol_length(g, u);
        min_length[u] = UINT32_MAX;
    }

    const uint32_t root_length = symbol_length[g->root_node];
    min_length[g->root_node] = root_length;
    max_length[g->root_node] = root_length;

    /* A scalar first pass establishes the smallest allocation that can hold
     * every generated length, including recombined paths longer than any
     * training sequence. */
    uint32_t generated_max = 0;
    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        if (min_length[u] == UINT32_MAX) continue;
        if (g->node_is_sink && g->node_is_sink[u] &&
            max_length[u] > generated_max)
            generated_max = max_length[u];
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            uint32_t candidate_min = min_length[u] + symbol_length[v];
            uint32_t candidate_max = max_length[u] + symbol_length[v];
            if (candidate_min < min_length[v]) min_length[v] = candidate_min;
            if (candidate_max > max_length[v]) max_length[v] = candidate_max;
        }
    }

    size_t width = (size_t)generated_max + 1;
    if (width > SIZE_MAX / (size_t)nn / sizeof(double)) {
        free(symbol_length); free(min_length); free(max_length);
        return LZG_ERR_ALLOC;
    }
    double *state = (double *)calloc((size_t)nn * width, sizeof(double));
    double *total = (double *)calloc(width, sizeof(double));
    if (!state || !total) {
        free(state); free(total);
        free(symbol_length); free(min_length); free(max_length);
        return LZG_ERR_ALLOC;
    }
    state[(size_t)g->root_node * width + root_length] = 1.0;

    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        if (min_length[u] == UINT32_MAX) continue;
        double *source = state + (size_t)u * width;
        if (g->node_is_sink && g->node_is_sink[u]) {
            for (uint32_t length = min_length[u]; length <= max_length[u]; length++)
                total[length] += source[length];
            continue;
        }
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            uint32_t shift = symbol_length[v];
            double *destination = state + (size_t)v * width + shift;
            for (uint32_t length = min_length[u]; length <= max_length[u]; length++)
                destination[length] += source[length];
        }
    }

    free(state);
    free(symbol_length); free(min_length); free(max_length);
    *counts_out = total;
    *max_length_out = generated_max;
    return LZG_OK;
}

LZGError lzg_flashback_pseq_length_derivatives(
    const LZGGraph *g, double q, uint32_t order,
    double **derivatives_out, uint8_t **present_out,
    uint32_t *max_length_out) {
    if (!g || !derivatives_out || !present_out || !max_length_out)
        return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid || order > 8) return LZG_ERR_INVALID_ARG;
    *derivatives_out = NULL;
    *present_out = NULL;
    *max_length_out = 0;

    const uint32_t nn = g->n_nodes;
    if (nn == 0 || g->root_node >= nn) return LZG_ERR_NOT_BUILT;

    uint32_t *symbol_length = (uint32_t *)malloc((size_t)nn * sizeof(uint32_t));
    uint32_t *min_length = (uint32_t *)malloc((size_t)nn * sizeof(uint32_t));
    uint32_t *max_length = (uint32_t *)calloc(nn, sizeof(uint32_t));
    long double **state = (long double **)calloc(nn, sizeof(long double *));
    if (!symbol_length || !min_length || !max_length || !state) {
        free(symbol_length); free(min_length); free(max_length); free(state);
        return LZG_ERR_ALLOC;
    }
    for (uint32_t u = 0; u < nn; u++) {
        symbol_length[u] = pc_symbol_length(g, u);
        min_length[u] = UINT32_MAX;
    }

    const uint32_t root_length = symbol_length[g->root_node];
    min_length[g->root_node] = root_length;
    max_length[g->root_node] = root_length;
    uint32_t generated_max = 0;
    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        if (min_length[u] == UINT32_MAX) continue;
        if (g->row_offsets[u] == g->row_offsets[u + 1] &&
            max_length[u] > generated_max)
            generated_max = max_length[u];
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            uint32_t candidate_min = min_length[u] + symbol_length[v];
            uint32_t candidate_max = max_length[u] + symbol_length[v];
            if (candidate_min < min_length[v]) min_length[v] = candidate_min;
            if (candidate_max > max_length[v]) max_length[v] = candidate_max;
        }
    }

    const size_t dim = (size_t)order + 1;
    const size_t output_width = (size_t)generated_max + 1;
    uint32_t binomial[9][9] = {{0}};
    for (uint32_t r = 0; r <= order; r++) {
        binomial[r][0] = binomial[r][r] = 1;
        for (uint32_t j = 1; j < r; j++)
            binomial[r][j] = binomial[r - 1][j - 1] + binomial[r - 1][j];
    }
    if (output_width > SIZE_MAX / dim / sizeof(long double)) {
        free(symbol_length); free(min_length); free(max_length); free(state);
        return LZG_ERR_ALLOC;
    }
    long double *total = (long double *)calloc(
        output_width * dim, sizeof(long double));
    double *result = (double *)malloc(output_width * dim * sizeof(double));
    uint8_t *present = (uint8_t *)calloc(output_width, sizeof(uint8_t));
    if (!total || !result || !present) {
        free(total); free(result); free(present);
        free(symbol_length); free(min_length); free(max_length); free(state);
        return LZG_ERR_ALLOC;
    }

    state[g->root_node] = (long double *)calloc(dim, sizeof(long double));
    if (!state[g->root_node]) {
        free(total); free(result); free(present);
        free(symbol_length); free(min_length); free(max_length); free(state);
        return LZG_ERR_ALLOC;
    }
    state[g->root_node][0] = 1.0L;

    LZGError err = LZG_OK;
    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        long double *source = state[u];
        if (!source) continue;
        const uint32_t source_min = min_length[u];
        const uint32_t source_max = max_length[u];
        const size_t source_width = (size_t)(source_max - source_min) + 1;

        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            for (size_t offset = 0; offset < source_width; offset++) {
                uint32_t length = source_min + (uint32_t)offset;
                long double *destination = total + (size_t)length * dim;
                const long double *source_jet = source + offset * dim;
                if (source_jet[0] != 0.0L) present[length] = 1;
                for (uint32_t r = 0; r <= order; r++)
                    destination[r] += source_jet[r];
            }
            free(source);
            state[u] = NULL;
            continue;
        }

        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            const size_t destination_width =
                (size_t)(max_length[v] - min_length[v]) + 1;
            if (!state[v]) {
                if (destination_width > SIZE_MAX / dim / sizeof(long double)) {
                    err = LZG_ERR_ALLOC;
                    goto length_derivatives_done;
                }
                state[v] = (long double *)calloc(
                    destination_width * dim, sizeof(long double));
                if (!state[v]) {
                    err = LZG_ERR_ALLOC;
                    goto length_derivatives_done;
                }
            }

            const long double log_probability = logl((long double)g->edge_weights[e]);
            const long double factor = expl((long double)q * log_probability);
            long double powers[9];
            powers[0] = 1.0L;
            for (uint32_t r = 1; r <= order; r++)
                powers[r] = powers[r - 1] * log_probability;

            const uint32_t shift = symbol_length[v];
            const size_t destination_start =
                (size_t)(source_min + shift - min_length[v]);
            if (order == 0) {
                for (size_t offset = 0; offset < source_width; offset++)
                    state[v][destination_start + offset] += factor * source[offset];
                continue;
            }
            if (order == 1) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                for (size_t offset = 0; offset < source_width; offset++) {
                    const long double *s = source + offset * 2;
                    long double *d = state[v] + (destination_start + offset) * 2;
                    d[0] += c0 * s[0];
                    d[1] += c0 * s[1] + c1 * s[0];
                }
                continue;
            }
            if (order == 2) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                const long double c2 = factor * powers[2];
                for (size_t offset = 0; offset < source_width; offset++) {
                    const long double *s = source + offset * 3;
                    long double *d = state[v] + (destination_start + offset) * 3;
                    d[0] += c0 * s[0];
                    d[1] += c0 * s[1] + c1 * s[0];
                    d[2] += c0 * s[2] + 2.0L * c1 * s[1] + c2 * s[0];
                }
                continue;
            }
            if (order == 4) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                const long double c2 = factor * powers[2];
                const long double c3 = factor * powers[3];
                const long double c4 = factor * powers[4];
                for (size_t offset = 0; offset < source_width; offset++) {
                    const long double *s = source + offset * 5;
                    long double *d = state[v] + (destination_start + offset) * 5;
                    d[0] += c0 * s[0];
                    d[1] += c0 * s[1] + c1 * s[0];
                    d[2] += c0 * s[2] + 2.0L * c1 * s[1] + c2 * s[0];
                    d[3] += c0 * s[3] + 3.0L * c1 * s[2] +
                            3.0L * c2 * s[1] + c3 * s[0];
                    d[4] += c0 * s[4] + 4.0L * c1 * s[3] +
                            6.0L * c2 * s[2] + 4.0L * c3 * s[1] + c4 * s[0];
                }
                continue;
            }
            for (size_t offset = 0; offset < source_width; offset++) {
                const long double *source_jet = source + offset * dim;
                long double *destination_jet =
                    state[v] + (destination_start + offset) * dim;
                for (uint32_t r = 0; r <= order; r++) {
                    long double value = 0.0L;
                    for (uint32_t j = 0; j <= r; j++)
                        value += (long double)binomial[r][j] *
                                 source_jet[j] * powers[r - j];
                    destination_jet[r] += factor * value;
                }
            }
        }
        free(source);
        state[u] = NULL;
    }

    for (size_t i = 0; i < output_width * dim; i++)
        result[i] = (double)total[i];

length_derivatives_done:
    for (uint32_t u = 0; u < nn; u++) free(state[u]);
    free(state);
    free(total);
    free(symbol_length); free(min_length); free(max_length);
    if (err != LZG_OK) {
        free(result); free(present);
        return err;
    }
    *derivatives_out = result;
    *present_out = present;
    *max_length_out = generated_max;
    return LZG_OK;
}

LZGError lzg_flashback_pseq_derivatives(const LZGGraph *g, double q,
                                        uint32_t order,
                                        double *derivatives_out) {
    if (!g || !derivatives_out || order > 8) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    const uint32_t nn = g->n_nodes;
    const size_t dim = (size_t)order + 1;
    if ((size_t)nn > SIZE_MAX / dim / sizeof(long double))
        return LZG_ERR_ALLOC;
    long double *state = (long double *)calloc(
        (size_t)nn * dim, sizeof(long double));
    if (!state) return LZG_ERR_ALLOC;
    long double total[9] = {0.0L};
    state[(size_t)g->root_node * dim] = 1.0L;

    uint32_t binomial[9][9] = {{0}};
    for (uint32_t r = 0; r <= order; r++) {
        binomial[r][0] = binomial[r][r] = 1;
        for (uint32_t j = 1; j < r; j++)
            binomial[r][j] = binomial[r - 1][j - 1] + binomial[r - 1][j];
    }

    for (uint32_t t = 0; t < nn; t++) {
        uint32_t u = g->topo_order[t];
        long double *source = state + (size_t)u * dim;
        if (source[0] == 0.0L) continue;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            for (uint32_t r = 0; r <= order; r++) total[r] += source[r];
            continue;
        }

        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            long double *destination = state + (size_t)v * dim;
            const long double log_probability = logl((long double)g->edge_weights[e]);
            const long double factor = expl((long double)q * log_probability);
            long double powers[9];
            powers[0] = 1.0L;
            for (uint32_t r = 1; r <= order; r++)
                powers[r] = powers[r - 1] * log_probability;

            if (order == 0) {
                destination[0] += factor * source[0];
                continue;
            }
            if (order == 1) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                destination[0] += c0 * source[0];
                destination[1] += c0 * source[1] + c1 * source[0];
                continue;
            }
            if (order == 2) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                const long double c2 = factor * powers[2];
                destination[0] += c0 * source[0];
                destination[1] += c0 * source[1] + c1 * source[0];
                destination[2] += c0 * source[2] +
                                  2.0L * c1 * source[1] + c2 * source[0];
                continue;
            }
            if (order == 4) {
                const long double c0 = factor;
                const long double c1 = factor * powers[1];
                const long double c2 = factor * powers[2];
                const long double c3 = factor * powers[3];
                const long double c4 = factor * powers[4];
                destination[0] += c0 * source[0];
                destination[1] += c0 * source[1] + c1 * source[0];
                destination[2] += c0 * source[2] +
                                  2.0L * c1 * source[1] + c2 * source[0];
                destination[3] += c0 * source[3] +
                                  3.0L * c1 * source[2] +
                                  3.0L * c2 * source[1] + c3 * source[0];
                destination[4] += c0 * source[4] +
                                  4.0L * c1 * source[3] +
                                  6.0L * c2 * source[2] +
                                  4.0L * c3 * source[1] + c4 * source[0];
                continue;
            }
            for (uint32_t r = 0; r <= order; r++) {
                long double value = 0.0L;
                for (uint32_t j = 0; j <= r; j++)
                    value += (long double)binomial[r][j] *
                             source[j] * powers[r - j];
                destination[r] += factor * value;
            }
        }
    }

    for (uint32_t r = 0; r <= order; r++)
        derivatives_out[r] = (double)total[r];
    free(state);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Log-normalized Mellin tilts and saddlepoint inversion           */
/* ═══════════════════════════════════════════════════════════════ */

static const uint32_t pseq_binomial[9][9] = {
    {1},
    {1, 1},
    {1, 2, 1},
    {1, 3, 3, 1},
    {1, 4, 6, 4, 1},
    {1, 5, 10, 10, 5, 1},
    {1, 6, 15, 20, 15, 6, 1},
    {1, 7, 21, 35, 35, 21, 7, 1},
    {1, 8, 28, 56, 70, 56, 28, 8, 1},
};

/* `shape[1]` stores the mean; entries >=2 are normalized central
 * moments. Entry zero is always one. This representation makes an edge
 * traversal exact and cheap: adding log(weight) shifts only the mean. */
static void pseq_merge_normalized(
    long double *destination, long double *destination_log_mass,
    const long double *source, long double source_log_mass,
    long double source_mean, uint32_t order) {
    if (!isfinite(*destination_log_mass)) {
        *destination_log_mass = source_log_mass;
        destination[0] = 1.0L;
        if (order >= 1) destination[1] = source_mean;
        for (uint32_t r = 2; r <= order; r++) destination[r] = source[r];
        return;
    }

    const long double old_log_mass = *destination_log_mass;
    long double alpha, beta, merged_log_mass;
    if (old_log_mass >= source_log_mass) {
        const long double ratio = expl(source_log_mass - old_log_mass);
        const long double denominator = 1.0L + ratio;
        alpha = 1.0L / denominator;
        beta = ratio / denominator;
        merged_log_mass = old_log_mass + log1pl(ratio);
    } else {
        const long double ratio = expl(old_log_mass - source_log_mass);
        const long double denominator = 1.0L + ratio;
        alpha = ratio / denominator;
        beta = 1.0L / denominator;
        merged_log_mass = source_log_mass + log1pl(ratio);
    }
    *destination_log_mass = merged_log_mass;
    if (order == 0) return;

    long double old[9];
    for (uint32_t r = 0; r <= order; r++) old[r] = destination[r];
    const long double old_mean = old[1];
    const long double merged_mean = alpha * old_mean + beta * source_mean;
    destination[0] = 1.0L;
    destination[1] = merged_mean;
    if (order == 1) return;

    const long double shift_a = old_mean - merged_mean;
    const long double shift_b = source_mean - merged_mean;
    long double power_a[9], power_b[9];
    power_a[0] = power_b[0] = 1.0L;
    for (uint32_t r = 1; r <= order; r++) {
        power_a[r] = power_a[r - 1] * shift_a;
        power_b[r] = power_b[r - 1] * shift_b;
    }
    for (uint32_t r = 2; r <= order; r++) {
        long double around_a = power_a[r];
        long double around_b = power_b[r];
        for (uint32_t j = 2; j <= r; j++) {
            around_a += (long double)pseq_binomial[r][j] *
                        old[j] * power_a[r - j];
            around_b += (long double)pseq_binomial[r][j] *
                        source[j] * power_b[r - j];
        }
        destination[r] = alpha * around_a + beta * around_b;
    }
}

static LZGError pseq_tilted_moments_core(
    const LZGGraph *g, long double q, uint32_t order,
    long double *log_mass_out, long double *raw_out,
    long double *central_out) {
    if (!g || !log_mass_out || !raw_out || !central_out || order > 8 ||
        !isfinite(q))
        return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    const uint32_t nn = g->n_nodes;
    const size_t dim = (size_t)order + 1;
    if ((size_t)nn > SIZE_MAX / dim / sizeof(long double))
        return LZG_ERR_ALLOC;
    long double *log_mass = (long double *)malloc(
        (size_t)nn * sizeof(long double));
    long double *shape = (long double *)calloc(
        (size_t)nn * dim, sizeof(long double));
    if (!log_mass || !shape) {
        free(log_mass); free(shape);
        return LZG_ERR_ALLOC;
    }
    for (uint32_t u = 0; u < nn; u++) log_mass[u] = -INFINITY;
    log_mass[g->root_node] = 0.0L;
    shape[(size_t)g->root_node * dim] = 1.0L;

    long double sink_log_mass = -INFINITY;
    long double sink_shape[9] = {0.0L};
    for (uint32_t t = 0; t < nn; t++) {
        const uint32_t u = g->topo_order[t];
        if (!isfinite(log_mass[u])) continue;
        const long double *source = shape + (size_t)u * dim;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            const long double source_mean = order >= 1 ? source[1] : 0.0L;
            pseq_merge_normalized(
                sink_shape, &sink_log_mass, source, log_mass[u],
                source_mean, order);
            continue;
        }
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            const double weight = g->edge_weights[e];
            if (!(weight > 0.0) || !isfinite(weight)) {
                free(log_mass); free(shape);
                return LZG_ERR_INVALID_ARG;
            }
            const uint32_t v = g->col_indices[e];
            const long double log_probability = logl((long double)weight);
            const long double candidate_log_mass =
                log_mass[u] + q * log_probability;
            const long double candidate_mean =
                (order >= 1 ? source[1] : 0.0L) + log_probability;
            pseq_merge_normalized(
                shape + (size_t)v * dim, &log_mass[v], source,
                candidate_log_mass, candidate_mean, order);
        }
    }
    free(log_mass); free(shape);
    if (!isfinite(sink_log_mass)) return LZG_ERR_NO_LIVE_PATHS;

    *log_mass_out = sink_log_mass;
    central_out[0] = 1.0L;
    raw_out[0] = 1.0L;
    if (order >= 1) {
        central_out[1] = 0.0L;
        raw_out[1] = sink_shape[1];
    }
    long double mean_powers[9];
    mean_powers[0] = 1.0L;
    for (uint32_t r = 1; r <= order; r++)
        mean_powers[r] = mean_powers[r - 1] * sink_shape[1];
    for (uint32_t r = 2; r <= order; r++) {
        central_out[r] = sink_shape[r];
        long double raw = mean_powers[r];
        for (uint32_t j = 2; j <= r; j++)
            raw += (long double)pseq_binomial[r][j] *
                   sink_shape[j] * mean_powers[r - j];
        raw_out[r] = raw;
    }
    return LZG_OK;
}

LZGError lzg_flashback_pseq_tilted_moments(
    const LZGGraph *g, double q, uint32_t order,
    double *log_mass_out, double *raw_moments_out,
    double *central_moments_out) {
    if (!log_mass_out || !raw_moments_out || !central_moments_out)
        return LZG_ERR_INVALID_ARG;
    long double log_mass, raw[9] = {0.0L}, central[9] = {0.0L};
    LZGError err = pseq_tilted_moments_core(
        g, (long double)q, order, &log_mass, raw, central);
    if (err != LZG_OK) return err;
    *log_mass_out = (double)log_mass;
    for (uint32_t r = 0; r <= order; r++) {
        raw_moments_out[r] = (double)raw[r];
        central_moments_out[r] = (double)central[r];
    }
    return LZG_OK;
}

static long double pseq_logadd(long double a, long double b) {
    if (!isfinite(a)) return b;
    if (!isfinite(b)) return a;
    if (a < b) {
        const long double temporary = a;
        a = b;
        b = temporary;
    }
    return a + log1pl(expl(b - a));
}

static double pseq_probability_from_log(long double log_probability) {
    if (!isfinite(log_probability)) return 0.0;
    if (log_probability >= 0.0L) return 1.0;
    return (double)expl(log_probability);
}

void lzg_flashback_pseq_attribution_destroy(LZGPseqAttribution *result) {
    if (!result) return;
    free(result->node_probability);
    free(result->edge_probability);
    memset(result, 0, sizeof(*result));
}

LZGError lzg_flashback_pseq_attribution(
    const LZGGraph *g, double q, LZGPseqAttribution *out) {
    if (!g || !out || !isfinite(q))
        return LZG_FAIL(LZG_ERR_INVALID_ARG,
                        "pseq_attribution: graph, output, and finite q required");
    memset(out, 0, sizeof(*out));
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_FAIL(LZG_ERR_NOT_BUILT,
                        "pseq_attribution: a built DAG is required");

    const uint32_t nn = g->n_nodes;
    const uint32_t ne = g->n_edges;
    long double *forward = (long double *)malloc(
        (size_t)nn * sizeof(long double));
    long double *backward = (long double *)malloc(
        (size_t)nn * sizeof(long double));
    double *node_probability = (double *)calloc(nn, sizeof(double));
    double *edge_probability = ne ? (double *)calloc(ne, sizeof(double)) : NULL;
    if (!forward || !backward || !node_probability || (ne && !edge_probability)) {
        free(forward); free(backward);
        free(node_probability); free(edge_probability);
        return LZG_FAIL(LZG_ERR_ALLOC,
                        "pseq_attribution: unable to allocate forward/backward state");
    }
    for (uint32_t u = 0; u < nn; u++) {
        forward[u] = -INFINITY;
        backward[u] = -INFINITY;
    }
    forward[g->root_node] = 0.0L;

    const long double tilt = (long double)q;
    for (uint32_t t = 0; t < nn; t++) {
        const uint32_t u = g->topo_order[t];
        if (!isfinite(forward[u])) continue;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            const double weight = g->edge_weights[e];
            if (!(weight > 0.0) || !isfinite(weight)) {
                free(forward); free(backward);
                free(node_probability); free(edge_probability);
                return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                                "pseq_attribution: edge weights must be finite and positive");
            }
            const uint32_t v = g->col_indices[e];
            const long double candidate = forward[u] +
                tilt * logl((long double)weight);
            forward[v] = pseq_logadd(forward[v], candidate);
        }
    }

    long double log_mass = -INFINITY;
    for (uint32_t u = 0; u < nn; u++) {
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            log_mass = pseq_logadd(log_mass, forward[u]);
            backward[u] = 0.0L;
        }
    }
    if (!isfinite(log_mass)) {
        free(forward); free(backward);
        free(node_probability); free(edge_probability);
        return LZG_FAIL(LZG_ERR_NO_LIVE_PATHS,
                        "pseq_attribution: root cannot reach a sink");
    }

    for (uint32_t t = nn; t > 0; t--) {
        const uint32_t u = g->topo_order[t - 1];
        if (g->row_offsets[u] == g->row_offsets[u + 1]) continue;
        long double completion = -INFINITY;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            const uint32_t v = g->col_indices[e];
            if (!isfinite(backward[v])) continue;
            const long double candidate =
                tilt * logl((long double)g->edge_weights[e]) + backward[v];
            completion = pseq_logadd(completion, candidate);
        }
        backward[u] = completion;
    }

    for (uint32_t u = 0; u < nn; u++) {
        node_probability[u] = pseq_probability_from_log(
            forward[u] + backward[u] - log_mass);
        if (!isfinite(forward[u])) continue;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            const uint32_t v = g->col_indices[e];
            const long double log_probability = forward[u] +
                tilt * logl((long double)g->edge_weights[e]) +
                backward[v] - log_mass;
            edge_probability[e] = pseq_probability_from_log(log_probability);
        }
    }

    free(forward); free(backward);
    out->n_nodes = nn;
    out->n_edges = ne;
    out->q = q;
    out->log_mass = (double)log_mass;
    out->node_probability = node_probability;
    out->edge_probability = edge_probability;
    return LZG_OK;
}

/* Threshold lanes are processed in cache-sized batches. This bounds working
 * memory independently of a sweep's resolution while retaining contiguous
 * per-node lanes and scanning each CSR edge only once per batch. */
#define PSEQ_THRESHOLD_BATCH 8U

static uint32_t pseq_thresholds_below(
    const double *thresholds, uint32_t n, double weight) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (thresholds[mid] < weight)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

LZGError lzg_flashback_edge_threshold_diversity(
    const LZGGraph *g, const double *thresholds, uint32_t n_thresholds,
    double *log_d0_out, double *log_d1_out, double *log_d2_out,
    double *surviving_mass_out, uint64_t *kept_edges_out) {
    if (!g) return LZG_ERR_INVALID_ARG;
    if (n_thresholds == 0) return LZG_OK;
    if (!thresholds || !log_d0_out || !log_d1_out || !log_d2_out ||
        !surviving_mass_out || !kept_edges_out)
        return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;
    for (uint32_t j = 0; j < n_thresholds; j++) {
        if (isnan(thresholds[j]) ||
            (j > 0 && thresholds[j] < thresholds[j - 1]))
            return LZG_ERR_INVALID_ARG;
        log_d0_out[j] = -INFINITY;
        log_d1_out[j] = -INFINITY;
        log_d2_out[j] = -INFINITY;
        surviving_mass_out[j] = 0.0;
        kept_edges_out[j] = 0;
    }
    for (uint32_t e = 0; e < g->n_edges; e++) {
        const double weight = g->edge_weights[e];
        if (!(weight > 0.0) || !isfinite(weight))
            return LZG_ERR_PARAM_OUT_OF_RANGE;
    }

    const uint32_t nn = g->n_nodes;
    for (uint32_t first = 0; first < n_thresholds;
         first += PSEQ_THRESHOLD_BATCH) {
        const uint32_t lanes = n_thresholds - first < PSEQ_THRESHOLD_BATCH
            ? n_thresholds - first : PSEQ_THRESHOLD_BATCH;
        if ((size_t)nn > SIZE_MAX / lanes / sizeof(long double))
            return LZG_ERR_ALLOC;
        const size_t state_size = (size_t)nn * lanes;
        long double *count = (long double *)calloc(
            state_size, sizeof(long double));
        long double *mass = (long double *)calloc(
            state_size, sizeof(long double));
        long double *square_mass = (long double *)calloc(
            state_size, sizeof(long double));
        long double *surprisal = (long double *)calloc(
            state_size, sizeof(long double));
        if (!count || !mass || !square_mass || !surprisal) {
            free(count); free(mass); free(square_mass); free(surprisal);
            return LZG_ERR_ALLOC;
        }
        const size_t root = (size_t)g->root_node * lanes;
        for (uint32_t j = 0; j < lanes; j++) {
            count[root + j] = 1.0L;
            mass[root + j] = 1.0L;
            square_mass[root + j] = 1.0L;
        }

        long double total_count[PSEQ_THRESHOLD_BATCH] = {0};
        long double total_mass[PSEQ_THRESHOLD_BATCH] = {0};
        long double total_square[PSEQ_THRESHOLD_BATCH] = {0};
        long double total_surprisal[PSEQ_THRESHOLD_BATCH] = {0};
        const double *batch_thresholds = thresholds + first;

        for (uint32_t t = 0; t < nn; t++) {
            const uint32_t u = g->topo_order[t];
            const size_t source = (size_t)u * lanes;
            if (g->row_offsets[u] == g->row_offsets[u + 1]) {
                for (uint32_t j = 0; j < lanes; j++) {
                    total_count[j] += count[source + j];
                    total_mass[j] += mass[source + j];
                    total_square[j] += square_mass[source + j];
                    total_surprisal[j] += surprisal[source + j];
                }
                continue;
            }
            for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
                const double weight = g->edge_weights[e];
                const uint32_t active = pseq_thresholds_below(
                    batch_thresholds, lanes, weight);
                for (uint32_t j = 0; j < active; j++)
                    kept_edges_out[first + j]++;
                if (active == 0) continue;
                const uint32_t v = g->col_indices[e];
                const size_t destination = (size_t)v * lanes;
                const long double w = (long double)weight;
                const long double w2 = w * w;
                const long double negative_log_w = -logl(w);
                for (uint32_t j = 0; j < active; j++) {
                    const long double source_count = count[source + j];
                    if (source_count == 0.0L) continue;
                    const long double source_mass = mass[source + j];
                    count[destination + j] += source_count;
                    mass[destination + j] += source_mass * w;
                    square_mass[destination + j] +=
                        square_mass[source + j] * w2;
                    surprisal[destination + j] +=
                        w * surprisal[source + j] +
                        negative_log_w * w * source_mass;
                }
            }
        }

        for (uint32_t j = 0; j < lanes; j++) {
            const uint32_t output = first + j;
            if (total_count[j] > 0.0L && total_mass[j] > 0.0L &&
                total_square[j] > 0.0L) {
                const long double log_z = logl(total_mass[j]);
                log_d0_out[output] = (double)logl(total_count[j]);
                log_d1_out[output] = (double)(
                    log_z + total_surprisal[j] / total_mass[j]);
                log_d2_out[output] = (double)(
                    2.0L * log_z - logl(total_square[j]));
                surviving_mass_out[output] = (double)total_mass[j];
            }
        }
        free(count); free(mass); free(square_mass); free(surprisal);
    }
    return LZG_OK;
}

static LZGError pseq_saddle_eval(const LZGGraph *g, long double t,
                                 uint32_t order, long double *k,
                                 long double *mean, long double *variance,
                                 long double *third, long double *fourth) {
    long double raw[9] = {0.0L}, central[9] = {0.0L}, log_mass;
    LZGError err = pseq_tilted_moments_core(
        g, 1.0L - t, order, &log_mass, raw, central);
    if (err != LZG_OK) return err;
    *k = log_mass;
    *mean = order >= 1 ? -raw[1] : 0.0L;
    *variance = order >= 2 ? central[2] : 0.0L;
    *third = order >= 3 ? -central[3] : 0.0L;
    *fourth = order >= 4 ? central[4] - 3.0L * central[2] * central[2] : 0.0L;
    return LZG_OK;
}

static long double pseq_normal_cdf(long double z) {
    return 0.5L * erfcl(-z / sqrtl(2.0L));
}

LZGError lzg_flashback_pseq_saddlepoint_batch(
    const LZGGraph *g, const double *x, uint32_t n,
    double *pdf_out, double *cdf_out, double *saddle_out,
    uint32_t *iterations_out) {
    if (!g || (!x && n) || (!pdf_out && n) || (!cdf_out && n) ||
        (!saddle_out && n))
        return LZG_ERR_INVALID_ARG;
    if (n == 0) return LZG_OK;

    uint8_t *symbol_lengths = (uint8_t *)malloc(g->n_nodes);
    if (!symbol_lengths) return LZG_ERR_ALLOC;
    double minimum_double, maximum_double;
    uint32_t max_edges;
    LZGError err = lzg_flashback_pseq_init(
        g, symbol_lengths, &minimum_double, &maximum_double, &max_edges);
    free(symbol_lengths);
    (void)max_edges;
    if (err != LZG_OK) return err;
    const long double minimum = (long double)minimum_double;
    const long double maximum = (long double)maximum_double;

    long double log_mass_zero, mean_zero, variance_zero, third, fourth;
    err = pseq_saddle_eval(g, 0.0L, 4, &log_mass_zero, &mean_zero,
                           &variance_zero, &third, &fourth);
    if (err != LZG_OK) return err;
    const long double range = fmaxl(1.0L, maximum - minimum);
    const long double x_tolerance = 8.0e-15L * range;

    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) return LZG_ERR_INVALID_ARG;
        const long double target = (long double)x[i];
        if (maximum == minimum) {
            pdf_out[i] = 0.0;
            cdf_out[i] = target < minimum ? 0.0 : 1.0;
            saddle_out[i] = target < minimum ? -INFINITY : INFINITY;
            if (iterations_out) iterations_out[i] = 0;
            continue;
        }
        if (target <= minimum) {
            pdf_out[i] = 0.0;
            cdf_out[i] = 0.0;
            saddle_out[i] = -INFINITY;
            if (iterations_out) iterations_out[i] = 0;
            continue;
        }
        if (target >= maximum) {
            pdf_out[i] = 0.0;
            cdf_out[i] = 1.0;
            saddle_out[i] = INFINITY;
            if (iterations_out) iterations_out[i] = 0;
            continue;
        }

        long double lo, hi, t, k = log_mass_zero;
        long double mean = mean_zero, variance = variance_zero;
        uint32_t evaluations = 0;
        if (fabsl(target - mean_zero) <= x_tolerance) {
            lo = hi = t = 0.0L;
        } else if (target < mean_zero) {
            hi = 0.0L;
            lo = -1.0L;
            for (uint32_t expansion = 0; ; expansion++) {
                err = pseq_saddle_eval(
                    g, lo, 2, &k, &mean, &variance, &third, &fourth);
                if (err != LZG_OK) return err;
                evaluations++;
                if (mean <= target || expansion == 63) break;
                lo *= 2.0L;
            }
            if (mean > target) return LZG_ERR_CONVERGENCE;
            t = (target - mean_zero) / fmaxl(variance_zero, 1.0e-300L);
            if (!isfinite(t) || t <= lo || t >= hi) t = 0.5L * (lo + hi);
        } else {
            lo = 0.0L;
            hi = 1.0L;
            for (uint32_t expansion = 0; ; expansion++) {
                err = pseq_saddle_eval(
                    g, hi, 2, &k, &mean, &variance, &third, &fourth);
                if (err != LZG_OK) return err;
                evaluations++;
                if (mean >= target || expansion == 63) break;
                hi *= 2.0L;
            }
            if (mean < target) return LZG_ERR_CONVERGENCE;
            t = (target - mean_zero) / fmaxl(variance_zero, 1.0e-300L);
            if (!isfinite(t) || t <= lo || t >= hi) t = 0.5L * (lo + hi);
        }

        if (lo != hi) {
            for (uint32_t iteration = 0; iteration < 48; iteration++) {
                err = pseq_saddle_eval(
                    g, t, 2, &k, &mean, &variance, &third, &fourth);
                if (err != LZG_OK) return err;
                evaluations++;
                const long double residual = mean - target;
                if (fabsl(residual) <= x_tolerance) break;
                if (residual < 0.0L) lo = t;
                else hi = t;
                long double candidate = NAN;
                if (variance > 0.0L)
                    candidate = t - residual / variance;
                if (!isfinite(candidate) || candidate <= lo || candidate >= hi)
                    candidate = 0.5L * (lo + hi);
                t = candidate;
            }
        }

        err = pseq_saddle_eval(
            g, t, 4, &k, &mean, &variance, &third, &fourth);
        if (err != LZG_OK) return err;
        evaluations++;
        const long double normalized_k = k - log_mass_zero;
        long double pdf = 0.0L, cdf;
        if (variance > 0.0L) {
            const long double log_pdf = normalized_k - t * target -
                0.5L * logl(2.0L * acosl(-1.0L) * variance);
            pdf = expl(log_pdf);
        }
        if (fabsl(t) < 1.0e-8L || variance <= 0.0L) {
            const long double z = (target - mean_zero) /
                sqrtl(fmaxl(variance_zero, 1.0e-300L));
            cdf = pseq_normal_cdf(z);
        } else {
            const long double argument = fmaxl(
                2.0L * (t * target - normalized_k), 0.0L);
            const long double w = copysignl(sqrtl(argument), t);
            const long double u = t * sqrtl(fmaxl(variance, 1.0e-300L));
            if (fabsl(w) < 1.0e-10L || fabsl(u) < 1.0e-10L) {
                const long double z = (target - mean_zero) /
                    sqrtl(fmaxl(variance_zero, 1.0e-300L));
                cdf = pseq_normal_cdf(z);
            } else {
                const long double normal_pdf =
                    expl(-0.5L * w * w) / sqrtl(2.0L * acosl(-1.0L));
                cdf = pseq_normal_cdf(w) +
                      normal_pdf * (1.0L / w - 1.0L / u);
            }
        }
        if (cdf < 0.0L) cdf = 0.0L;
        if (cdf > 1.0L) cdf = 1.0L;
        pdf_out[i] = (double)pdf;
        cdf_out[i] = (double)cdf;
        saddle_out[i] = (double)t;
        if (iterations_out) iterations_out[i] = evaluations;
    }
    return LZG_OK;
}

typedef struct {
    long double *mass;
    uint32_t lo;
    uint32_t hi;
} LZGPseqGridState;

static void pseq_grid_transport(long double *destination, uint32_t destination_lo,
                                const LZGPseqGridState *source,
                                uint32_t lower, long double fraction,
                                long double factor, uint32_t bins) {
    const long double lower_weight = 1.0L - fraction;
    const size_t source_width = (size_t)(source->hi - source->lo) + 1;
    for (size_t offset = 0; offset < source_width; offset++) {
        long double value = source->mass[offset];
        if (value == 0.0L) continue;
        uint64_t base = (uint64_t)source->lo + offset + lower;
        if (base < bins && lower_weight != 0.0L)
            destination[(size_t)base - destination_lo] +=
                value * lower_weight * factor;
        if (fraction != 0.0L && base + 1 < bins)
            destination[(size_t)(base + 1) - destination_lo] +=
                value * fraction * factor;
    }
}

static LZGError pseq_histogram_grid_bounds(
    const LZGGraph *g, uint32_t bins, double spacing,
    uint32_t *grid_lo, uint32_t *grid_hi) {
    for (uint32_t u = 0; u < g->n_nodes; u++) grid_lo[u] = UINT32_MAX;
    grid_lo[g->root_node] = 0;
    grid_hi[g->root_node] = 0;
    for (uint32_t t = 0; t < g->n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        if (grid_lo[u] == UINT32_MAX) continue;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            double shift = -log(g->edge_weights[e]) / spacing;
            if (!isfinite(shift) || shift < 0.0 || shift > UINT32_MAX)
                return LZG_ERR_INVALID_ARG;
            uint32_t lower = (uint32_t)floor(shift);
            bool has_upper = shift - (double)lower > 0.0;
            uint64_t candidate_lo = (uint64_t)grid_lo[u] + lower;
            uint64_t candidate_hi = (uint64_t)grid_hi[u] + lower + has_upper;
            if (candidate_lo >= bins) candidate_lo = bins - 1;
            if (candidate_hi >= bins) candidate_hi = bins - 1;
            if ((uint32_t)candidate_lo < grid_lo[v])
                grid_lo[v] = (uint32_t)candidate_lo;
            if ((uint32_t)candidate_hi > grid_hi[v])
                grid_hi[v] = (uint32_t)candidate_hi;
        }
    }
    return LZG_OK;
}

static LZGError pseq_histogram_global(
    const LZGGraph *g, uint32_t bins, double q, double spacing,
    const uint32_t *grid_lo, const uint32_t *grid_hi,
    long double *total) {
    LZGPseqGridState *state = (LZGPseqGridState *)calloc(
        g->n_nodes, sizeof(LZGPseqGridState));
    if (!state) return LZG_ERR_ALLOC;
    state[g->root_node].mass = (long double *)calloc(1, sizeof(long double));
    if (!state[g->root_node].mass) {
        free(state);
        return LZG_ERR_ALLOC;
    }
    state[g->root_node].mass[0] = 1.0L;
    state[g->root_node].lo = state[g->root_node].hi = 0;
    LZGError err = LZG_OK;

    for (uint32_t t = 0; t < g->n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        LZGPseqGridState *source = &state[u];
        if (!source->mass) continue;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            size_t width = (size_t)(source->hi - source->lo) + 1;
            for (size_t i = 0; i < width; i++)
                total[(size_t)source->lo + i] += source->mass[i];
        } else {
            for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
                uint32_t v = g->col_indices[e];
                LZGPseqGridState *destination = &state[v];
                if (!destination->mass) {
                    destination->lo = grid_lo[v];
                    destination->hi = grid_hi[v];
                    size_t width = (size_t)(destination->hi - destination->lo) + 1;
                    if (width > SIZE_MAX / sizeof(long double)) {
                        err = LZG_ERR_ALLOC;
                        goto histogram_global_done;
                    }
                    destination->mass = (long double *)calloc(
                        width, sizeof(long double));
                    if (!destination->mass) {
                        err = LZG_ERR_ALLOC;
                        goto histogram_global_done;
                    }
                }
                double shift = -log(g->edge_weights[e]) / spacing;
                uint32_t lower = (uint32_t)floor(shift);
                long double fraction = (long double)(shift - (double)lower);
                long double factor = expl(
                    (long double)q * logl((long double)g->edge_weights[e]));
                pseq_grid_transport(
                    destination->mass, destination->lo, source,
                    lower, fraction, factor, bins);
            }
        }
        free(source->mass);
        source->mass = NULL;
    }

histogram_global_done:
    for (uint32_t u = 0; u < g->n_nodes; u++) free(state[u].mass);
    free(state);
    return err;
}

typedef struct {
    double *counting;
    double *generated;
    uint32_t lo;
    uint32_t hi;
} LZGPseqGridPairState;

static LZGError pseq_histogram_global_pair(
    const LZGGraph *g, uint32_t bins, double spacing,
    const uint32_t *grid_lo, const uint32_t *grid_hi,
    long double *counting_total, long double *generated_total) {
    LZGPseqGridPairState *state = (LZGPseqGridPairState *)calloc(
        g->n_nodes, sizeof(LZGPseqGridPairState));
    if (!state) return LZG_ERR_ALLOC;
    LZGPseqGridPairState *root = &state[g->root_node];
    root->counting = (double *)calloc(1, sizeof(double));
    root->generated = (double *)calloc(1, sizeof(double));
    if (!root->counting || !root->generated) {
        free(root->counting); free(root->generated); free(state);
        return LZG_ERR_ALLOC;
    }
    root->counting[0] = 1.0L;
    root->generated[0] = 1.0L;
    root->lo = root->hi = 0;
    LZGError err = LZG_OK;

    for (uint32_t t = 0; t < g->n_nodes; t++) {
        const uint32_t u = g->topo_order[t];
        LZGPseqGridPairState *source = &state[u];
        if (!source->counting) continue;
        const size_t source_width = (size_t)(source->hi - source->lo) + 1;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            for (size_t i = 0; i < source_width; i++) {
                counting_total[(size_t)source->lo + i] += source->counting[i];
                generated_total[(size_t)source->lo + i] += source->generated[i];
            }
        } else {
            for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
                const uint32_t v = g->col_indices[e];
                LZGPseqGridPairState *destination = &state[v];
                if (!destination->counting) {
                    destination->lo = grid_lo[v];
                    destination->hi = grid_hi[v];
                    const size_t width =
                        (size_t)(destination->hi - destination->lo) + 1;
                    if (width > SIZE_MAX / sizeof(double)) {
                        err = LZG_ERR_ALLOC;
                        goto histogram_pair_done;
                    }
                    destination->counting = (double *)calloc(
                        width, sizeof(double));
                    destination->generated = (double *)calloc(
                        width, sizeof(double));
                    if (!destination->counting || !destination->generated) {
                        err = LZG_ERR_ALLOC;
                        goto histogram_pair_done;
                    }
                }
                const double weight = g->edge_weights[e];
                const double shift = -log(weight) / spacing;
                const uint32_t lower = (uint32_t)floor(shift);
                const long double fraction =
                    (long double)(shift - (double)lower);
                const long double lower_weight = 1.0L - fraction;
                for (size_t offset = 0; offset < source_width; offset++) {
                    if (source->counting[offset] == 0.0L &&
                        source->generated[offset] == 0.0L)
                        continue;
                    const uint64_t base =
                        (uint64_t)source->lo + offset + lower;
                    const size_t destination_offset =
                        (size_t)base - destination->lo;
                    if (base < bins && lower_weight != 0.0L) {
                        destination->counting[destination_offset] +=
                            source->counting[offset] * (double)lower_weight;
                        destination->generated[destination_offset] +=
                            source->generated[offset] *
                            (double)lower_weight * weight;
                    }
                    if (fraction != 0.0L && base + 1 < bins) {
                        destination->counting[destination_offset + 1] +=
                            source->counting[offset] * (double)fraction;
                        destination->generated[destination_offset + 1] +=
                            source->generated[offset] *
                            (double)fraction * weight;
                    }
                }
            }
        }
        free(source->counting); free(source->generated);
        source->counting = NULL;
        source->generated = NULL;
    }

histogram_pair_done:
    for (uint32_t u = 0; u < g->n_nodes; u++) {
        free(state[u].counting); free(state[u].generated);
    }
    free(state);
    return err;
}

static void pseq_bitset_shift_or(uint64_t *destination, const uint64_t *source,
                                 size_t words, uint32_t shift,
                                 uint32_t max_bit) {
    const size_t word_shift = shift / 64;
    const uint32_t bit_shift = shift % 64;
    for (size_t i = 0; i < words; i++) {
        uint64_t value = source[i];
        if (!value || i + word_shift >= words) continue;
        destination[i + word_shift] |= value << bit_shift;
        if (bit_shift && i + word_shift + 1 < words)
            destination[i + word_shift + 1] |= value >> (64 - bit_shift);
    }
    const uint32_t used = max_bit % 64 + 1;
    if (used < 64) destination[words - 1] &= (UINT64_C(1) << used) - 1;
}

static bool pseq_bitset_get(const uint64_t *bits, uint32_t bit) {
    return (bits[bit / 64] >> (bit % 64)) & UINT64_C(1);
}

static LZGError pseq_histogram_length(
    const LZGGraph *g, uint32_t bins, double q, double spacing,
    uint32_t target_length, const uint8_t *symbol_length,
    const uint32_t *grid_lo, const uint32_t *grid_hi,
    long double *total) {
    uint32_t *max_length = (uint32_t *)calloc(
        g->n_nodes, sizeof(uint32_t));
    uint8_t *reachable = (uint8_t *)calloc(g->n_nodes, sizeof(uint8_t));
    if (!max_length || !reachable) {
        free(max_length); free(reachable);
        return LZG_ERR_ALLOC;
    }
    max_length[g->root_node] = symbol_length[g->root_node];
    reachable[g->root_node] = 1;
    uint32_t generated_max = 0;
    for (uint32_t t = 0; t < g->n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        if (!reachable[u]) continue;
        if (g->row_offsets[u] == g->row_offsets[u + 1] &&
            max_length[u] > generated_max)
            generated_max = max_length[u];
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            uint32_t candidate = max_length[u] + symbol_length[v];
            if (!reachable[v] || candidate > max_length[v]) max_length[v] = candidate;
            reachable[v] = 1;
        }
    }
    free(max_length); free(reachable);
    if (target_length > generated_max) return LZG_OK;

    const size_t words = (size_t)target_length / 64 + 1;
    if ((size_t)g->n_nodes > SIZE_MAX / words / sizeof(uint64_t))
        return LZG_ERR_ALLOC;
    uint64_t *can_finish = (uint64_t *)calloc(
        (size_t)g->n_nodes * words, sizeof(uint64_t));
    if (!can_finish) return LZG_ERR_ALLOC;

    for (uint32_t t = g->n_nodes; t > 0; t--) {
        uint32_t u = g->topo_order[t - 1];
        uint64_t *destination = can_finish + (size_t)u * words;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) destination[0] |= 1;
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            uint32_t v = g->col_indices[e];
            pseq_bitset_shift_or(
                destination, can_finish + (size_t)v * words,
                words, symbol_length[v], target_length);
        }
    }

    const size_t length_dim = (size_t)target_length + 1;
    if ((size_t)g->n_nodes > SIZE_MAX / length_dim /
        sizeof(LZGPseqGridState)) {
        free(can_finish);
        return LZG_ERR_ALLOC;
    }
    LZGPseqGridState *state = (LZGPseqGridState *)calloc(
        (size_t)g->n_nodes * length_dim, sizeof(LZGPseqGridState));
    if (!state) {
        free(can_finish);
        return LZG_ERR_ALLOC;
    }

    uint32_t root_length = symbol_length[g->root_node];
    if (root_length <= target_length && pseq_bitset_get(
        can_finish + (size_t)g->root_node * words,
        target_length - root_length)) {
        LZGPseqGridState *root =
            &state[(size_t)g->root_node * length_dim + root_length];
        root->mass = (long double *)calloc(1, sizeof(long double));
        if (!root->mass) {
            free(state); free(can_finish);
            return LZG_ERR_ALLOC;
        }
        root->mass[0] = 1.0L;
        root->lo = root->hi = 0;
    }

    LZGError err = LZG_OK;
    for (uint32_t t = 0; t < g->n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        LZGPseqGridState *node_state = state + (size_t)u * length_dim;
        if (g->row_offsets[u] == g->row_offsets[u + 1]) {
            LZGPseqGridState *source = &node_state[target_length];
            if (source->mass) {
                size_t width = (size_t)(source->hi - source->lo) + 1;
                for (size_t i = 0; i < width; i++)
                    total[(size_t)source->lo + i] += source->mass[i];
            }
        } else {
            for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
                uint32_t v = g->col_indices[e];
                uint32_t increment = symbol_length[v];
                if (increment > target_length) continue;
                double shift = -log(g->edge_weights[e]) / spacing;
                uint32_t lower = (uint32_t)floor(shift);
                long double fraction = (long double)(shift - (double)lower);
                long double factor = expl(
                    (long double)q * logl((long double)g->edge_weights[e]));
                for (uint32_t length = 0;
                     length + increment <= target_length; length++) {
                    LZGPseqGridState *source = &node_state[length];
                    if (!source->mass) continue;
                    uint32_t new_length = length + increment;
                    if (!pseq_bitset_get(
                        can_finish + (size_t)v * words,
                        target_length - new_length))
                        continue;
                    LZGPseqGridState *destination =
                        &state[(size_t)v * length_dim + new_length];
                    if (!destination->mass) {
                        destination->lo = grid_lo[v];
                        destination->hi = grid_hi[v];
                        size_t width =
                            (size_t)(destination->hi - destination->lo) + 1;
                        if (width > SIZE_MAX / sizeof(long double)) {
                            err = LZG_ERR_ALLOC;
                            goto histogram_length_done;
                        }
                        destination->mass = (long double *)calloc(
                            width, sizeof(long double));
                        if (!destination->mass) {
                            err = LZG_ERR_ALLOC;
                            goto histogram_length_done;
                        }
                    }
                    pseq_grid_transport(
                        destination->mass, destination->lo, source,
                        lower, fraction, factor, bins);
                }
            }
        }
        for (uint32_t length = 0; length <= target_length; length++) {
            free(node_state[length].mass);
            node_state[length].mass = NULL;
        }
    }

histogram_length_done:
    for (size_t i = 0; i < (size_t)g->n_nodes * length_dim; i++)
        free(state[i].mass);
    free(state);
    free(can_finish);
    return err;
}

LZGError lzg_flashback_pseq_histogram(const LZGGraph *g, uint32_t bins,
                                      double q, int64_t length,
                                      double **weights_out,
                                      double *spacing_out,
                                      double *true_max_surprisal_out,
                                      uint32_t *max_edges_out) {
    if (!g || !weights_out || !spacing_out || !true_max_surprisal_out ||
        !max_edges_out || length < -1)
        return LZG_ERR_INVALID_ARG;
    *weights_out = NULL;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    uint8_t *symbol_length = (uint8_t *)malloc(g->n_nodes);
    uint32_t *grid_lo = (uint32_t *)malloc(
        (size_t)g->n_nodes * sizeof(uint32_t));
    uint32_t *grid_hi = (uint32_t *)calloc(g->n_nodes, sizeof(uint32_t));
    if (!symbol_length || !grid_lo || !grid_hi) {
        free(symbol_length); free(grid_lo); free(grid_hi);
        return LZG_ERR_ALLOC;
    }
    double true_min, true_max;
    uint32_t max_edges;
    LZGError err = lzg_flashback_pseq_init(
        g, symbol_length, &true_min, &true_max, &max_edges);
    (void)true_min;
    if (err != LZG_OK) goto histogram_done;
    if (true_max <= 0.0 || bins <= max_edges + 1) {
        err = LZG_ERR_INVALID_ARG;
        goto histogram_done;
    }
    double spacing = true_max / (double)(bins - 1 - max_edges);
    err = pseq_histogram_grid_bounds(
        g, bins, spacing, grid_lo, grid_hi);
    if (err != LZG_OK) goto histogram_done;

    long double *total = (long double *)calloc(bins, sizeof(long double));
    double *weights = (double *)malloc((size_t)bins * sizeof(double));
    if (!total || !weights) {
        free(total); free(weights);
        err = LZG_ERR_ALLOC;
        goto histogram_done;
    }
    if (length < 0) {
        err = pseq_histogram_global(
            g, bins, q, spacing, grid_lo, grid_hi, total);
    } else if ((uint64_t)length <= UINT32_MAX) {
        err = pseq_histogram_length(
            g, bins, q, spacing, (uint32_t)length, symbol_length,
            grid_lo, grid_hi, total);
    } else {
        err = LZG_ERR_INVALID_ARG;
    }
    if (err != LZG_OK) {
        free(total); free(weights);
        goto histogram_done;
    }
    for (uint32_t i = 0; i < bins; i++) weights[i] = (double)total[i];
    free(total);
    *weights_out = weights;
    *spacing_out = spacing;
    *true_max_surprisal_out = true_max;
    *max_edges_out = max_edges;

histogram_done:
    free(symbol_length); free(grid_lo); free(grid_hi);
    return err;
}

LZGError lzg_flashback_pseq_histogram_pair(
    const LZGGraph *g, uint32_t bins,
    double **counting_weights_out, double **generated_weights_out,
    double *spacing_out, double *true_max_surprisal_out,
    uint32_t *max_edges_out) {
    if (!g || !counting_weights_out || !generated_weights_out ||
        !spacing_out || !true_max_surprisal_out || !max_edges_out)
        return LZG_ERR_INVALID_ARG;
    *counting_weights_out = NULL;
    *generated_weights_out = NULL;
    if (!g->topo_valid || g->n_nodes == 0 || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    uint8_t *symbol_length = (uint8_t *)malloc(g->n_nodes);
    uint32_t *grid_lo = (uint32_t *)malloc(
        (size_t)g->n_nodes * sizeof(uint32_t));
    uint32_t *grid_hi = (uint32_t *)calloc(g->n_nodes, sizeof(uint32_t));
    if (!symbol_length || !grid_lo || !grid_hi) {
        free(symbol_length); free(grid_lo); free(grid_hi);
        return LZG_ERR_ALLOC;
    }
    double true_min, true_max;
    uint32_t max_edges;
    LZGError err = lzg_flashback_pseq_init(
        g, symbol_length, &true_min, &true_max, &max_edges);
    (void)true_min;
    if (err != LZG_OK) goto histogram_pair_wrapper_done;
    if (true_max <= 0.0 || bins <= max_edges + 1) {
        err = LZG_ERR_INVALID_ARG;
        goto histogram_pair_wrapper_done;
    }
    const double spacing = true_max / (double)(bins - 1 - max_edges);
    err = pseq_histogram_grid_bounds(
        g, bins, spacing, grid_lo, grid_hi);
    if (err != LZG_OK) goto histogram_pair_wrapper_done;

    long double *counting_total = (long double *)calloc(
        bins, sizeof(long double));
    long double *generated_total = (long double *)calloc(
        bins, sizeof(long double));
    double *counting_weights = (double *)malloc(
        (size_t)bins * sizeof(double));
    double *generated_weights = (double *)malloc(
        (size_t)bins * sizeof(double));
    if (!counting_total || !generated_total ||
        !counting_weights || !generated_weights) {
        free(counting_total); free(generated_total);
        free(counting_weights); free(generated_weights);
        err = LZG_ERR_ALLOC;
        goto histogram_pair_wrapper_done;
    }
    err = pseq_histogram_global_pair(
        g, bins, spacing, grid_lo, grid_hi,
        counting_total, generated_total);
    if (err != LZG_OK) {
        free(counting_total); free(generated_total);
        free(counting_weights); free(generated_weights);
        goto histogram_pair_wrapper_done;
    }
    for (uint32_t i = 0; i < bins; i++) {
        counting_weights[i] = (double)counting_total[i];
        generated_weights[i] = (double)generated_total[i];
    }
    free(counting_total); free(generated_total);
    *counting_weights_out = counting_weights;
    *generated_weights_out = generated_weights;
    *spacing_out = spacing;
    *true_max_surprisal_out = true_max;
    *max_edges_out = max_edges;

histogram_pair_wrapper_done:
    free(symbol_length); free(grid_lo); free(grid_hi);
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
