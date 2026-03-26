/**
 * @file features.c
 * @brief ML feature extraction from LZGraphs.
 */
#include "lzgraph/features.h"
#include "lzgraph/forward.h"
#include "lzgraph/analytics.h"
#include "lzgraph/graph_ops.h"
#include "lzgraph/simulate.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════ */
/* Strategy A: Reference-aligned feature vector                    */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_feature_aligned(const LZGGraph *ref, const LZGGraph *query,
                              double *out, uint32_t *out_dim) {
    if (!ref || !query || !out || !out_dim) return LZG_ERR_INVALID_ARG;

    uint32_t dim = ref->n_nodes;
    *out_dim = dim;
    memset(out, 0, dim * sizeof(double));

    /* Build label → index map for query graph */
    LZGHashMap *q_map = lzg_hm_create(query->n_nodes * 2);
    double q_total = 0.0;
    for (uint32_t i = 0; i < query->n_nodes; i++)
        q_total += query->outgoing_counts[i];

    for (uint32_t i = 0; i < query->n_nodes; i++) {
        const char *sp = lzg_sp_get(query->pool, query->node_sp_id[i]);
        uint32_t pos = query->node_pos[i];
        char buf[256];
        int len;
        if (query->variant == LZG_VARIANT_NAIVE)
            len = snprintf(buf, sizeof(buf), "%s", sp);
        else
            len = snprintf(buf, sizeof(buf), "%s_%u", sp, pos);
        uint64_t key = lzg_hash_bytes(buf, len);

        /* Store node frequency as uint64 bit pattern */
        double freq = q_total > 0 ? (double)query->outgoing_counts[i] / q_total : 0.0;
        uint64_t bits;
        memcpy(&bits, &freq, 8);
        lzg_hm_put(q_map, key, bits);
    }

    /* For each ref node, look up in query */
    for (uint32_t i = 0; i < dim; i++) {
        const char *sp = lzg_sp_get(ref->pool, ref->node_sp_id[i]);
        uint32_t pos = ref->node_pos[i];
        char buf[256];
        int len;
        if (ref->variant == LZG_VARIANT_NAIVE)
            len = snprintf(buf, sizeof(buf), "%s", sp);
        else
            len = snprintf(buf, sizeof(buf), "%s_%u", sp, pos);
        uint64_t key = lzg_hash_bytes(buf, len);

        uint64_t *val = lzg_hm_get(q_map, key);
        if (val) memcpy(&out[i], val, 8);
        /* else: out[i] remains 0.0 */
    }

    lzg_hm_destroy(q_map);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Strategy B: Position-projected mass profile via forward DP      */
/* ═══════════════════════════════════════════════════════════════ */

/* Context for the mass-profile callbacks */
typedef struct {
    double  *profile;   /* output array indexed by position */
    const LZGGraph *g;
    uint32_t max_pos;
} MassProfileCtx;

static void mp_seed(double *a, double p, void *c) { (void)c; a[0] += p; }

static void mp_edge(double *d, const double *s, double w, double Z, void *c) {
    (void)c; d[0] = s[0] * (w / Z);
}

static void mp_absorb(double *total, const double *a, double sp, void *c) {
    (void)total;
    MassProfileCtx *ctx = (MassProfileCtx *)c;
    /* Attribute absorbed mass to the terminal node's position.
     * We need the current node's position — but the forward engine
     * doesn't tell us which node we're at in the absorb callback.
     *
     * Workaround: accumulate total mass in total[0], then distribute
     * by position in a post-processing step using simulation. */

    /* For now, just accumulate total mass — we'll use Strategy B-alt
     * (simulation-based) instead. */
    total[0] += a[0] * sp;
}

static void mp_cont(double *co, const double *a, double sp, void *c) {
    (void)c; co[0] = a[0] * (1.0 - sp);
}

LZGError lzg_feature_mass_profile(const LZGGraph *g,
                                   double *out, uint32_t max_pos) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    memset(out, 0, (max_pos + 1) * sizeof(double));

    /* The forward engine's absorb callback doesn't know which node
     * it's processing. Use simulation to build the mass profile. */
    LZGRng rng;
    lzg_rng_seed(&rng, 54321);

    uint32_t N = 10000;
    LZGSimResult *results = malloc(N * sizeof(LZGSimResult));
    if (!results) return LZG_ERR_ALLOC;

    LZGError err = lzg_simulate(g, N, &rng, results);
    if (err != LZG_OK) { free(results); return err; }

    /* Build position profile: for each simulated sequence, attribute
     * its probability to the last token's position */
    double total_mass = 0.0;
    for (uint32_t i = 0; i < N; i++) {
        double p = exp(results[i].log_prob);
        uint32_t pos = results[i].seq_len;
        if (pos > max_pos) pos = max_pos;
        out[pos] += p;
        total_mass += p;
    }

    /* Normalize */
    if (total_mass > 0) {
        for (uint32_t i = 0; i <= max_pos; i++)
            out[i] /= total_mass;
    }

    for (uint32_t i = 0; i < N; i++) lzg_sim_result_free(&results[i]);
    free(results);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Strategy C: Graph statistics vector                             */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_feature_stats(const LZGGraph *g, double *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    memset(out, 0, LZG_FEATURE_STATS_DIM * sizeof(double));

    /* Basic counts */
    out[0] = (double)g->n_nodes;
    out[1] = (double)g->n_edges;
    out[2] = 1.0; /* single root node (@) */
    uint32_t n_sinks = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++)
        if (g->node_is_sink && g->node_is_sink[i]) n_sinks++;
    out[3] = (double)n_sinks;

    /* Hill numbers */
    double orders[] = {0, 0.5, 1, 2, 5};
    double hills[5];
    lzg_hill_numbers(g, orders, 5, hills);
    out[4] = hills[0];   /* D(0)   */
    out[5] = hills[1];   /* D(0.5) */
    out[6] = hills[2];   /* D(1)   */
    out[7] = hills[3];   /* D(2)   */
    out[8] = hills[4];   /* D(5)   */

    /* Effective diversity */
    LZGEffectiveDiversity div;
    if (lzg_effective_diversity(g, &div) == LZG_OK)
        out[9] = div.entropy_nats;

    /* Dynamic range */
    LZGDynamicRange dr;
    if (lzg_pgen_dynamic_range(g, &dr) == LZG_OK)
        out[10] = dr.dynamic_range_orders;

    /* Sink node statistics (reuse n_sinks from above) */
    out[11] = (double)n_sinks;
    out[12] = g->n_nodes > 0 ? (double)n_sinks / g->n_nodes : 0.0;

    /* Max out-degree */
    LZGGraphSummary sum;
    if (lzg_graph_summary(g, &sum) == LZG_OK)
        out[13] = (double)sum.max_out_degree;

    /* Uniformity */
    if (lzg_effective_diversity(g, &div) == LZG_OK)
        out[14] = div.uniformity;

    return LZG_OK;
}
