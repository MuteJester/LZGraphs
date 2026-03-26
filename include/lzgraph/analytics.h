/**
 * @file analytics.h
 * @brief Graph-level analytics computed via LZ-constrained forward DP.
 *
 * Every function here uses lzg_forward_propagate() internally, meaning
 * all results respect LZ76 dictionary constraints — only valid walks
 * contribute to the computed quantities.
 */
#ifndef LZGRAPH_ANALYTICS_H
#define LZGRAPH_ANALYTICS_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/* ── Simulation potential size ──────────────────────────────── */

/**
 * Count the number of distinct LZ-valid walks (sequences) the graph
 * can produce from initial states to terminal states.
 */
LZGError lzg_graph_path_count(const LZGGraph *g, double *out_count);

/* ── PGEN diagnostics ───────────────────────────────────────── */

typedef struct {
    double total_absorbed;          /* should be ≤ 1.0               */
    double total_leaked;            /* mass at dead-end non-terminals */
    double initial_prob_sum;        /* should be 1.0                 */
    bool   is_proper;               /* |total_absorbed - 1.0| < atol */
} LZGPgenDiagnostics;

LZGError lzg_pgen_diagnostics(const LZGGraph *g, double atol,
                               LZGPgenDiagnostics *out);

/* ── Effective diversity ────────────────────────────────────── */

typedef struct {
    double entropy_nats;
    double entropy_bits;
    double effective_diversity;      /* exp(H)                       */
    double uniformity;               /* N_eff / path_count           */
} LZGEffectiveDiversity;

LZGError lzg_effective_diversity(const LZGGraph *g, LZGEffectiveDiversity *out);

/* ── Hill numbers ───────────────────────────────────────────── */

/**
 * Compute M(α) = Σ_{valid walks} π(s)^α via LZ-constrained DP.
 * The Hill number is D(α) = M(α)^{1/(1-α)}.
 */
LZGError lzg_power_sum(const LZGGraph *g, double alpha, double *out_m);

/**
 * Compute D(α) = M(α)^{1/(1-α)} for a single order.
 * For α = 0: returns simulation_potential_size.
 * For α = 1: returns exp(Shannon entropy).
 */
LZGError lzg_hill_number(const LZGGraph *g, double alpha, double *out_d);

/**
 * Compute Hill numbers for multiple orders at once.
 *
 * @param g       The graph.
 * @param orders  Array of α values.
 * @param n       Number of orders.
 * @param out     Output array of D(α) values (caller allocates, size n).
 */
LZGError lzg_hill_numbers(const LZGGraph *g, const double *orders,
                           uint32_t n, double *out);

/* ── PGEN dynamic range ─────────────────────────────────────── */

typedef struct {
    double max_log_prob;             /* log P of most probable walk  */
    double min_log_prob;             /* log P of least probable walk */
    double dynamic_range_nats;
    double dynamic_range_orders;     /* range / ln(10)               */
} LZGDynamicRange;

LZGError lzg_pgen_dynamic_range(const LZGGraph *g, LZGDynamicRange *out);

/* ── Hill curve ────────────────────────────────────────────── */

typedef struct {
    double  *orders;       /* [n]: α values                        */
    double  *hill_numbers; /* [n]: D(α) values                     */
    uint32_t n;            /* number of points                     */
} LZGHillCurve;

/**
 * Compute the Hill diversity curve with default orders.
 * Default: [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10].
 *
 * @param g       The graph.
 * @param orders  Custom orders (NULL = use defaults).
 * @param n       Number of custom orders (0 = use defaults).
 * @param out     Output: caller allocates struct, arrays allocated internally.
 */
LZGError lzg_hill_curve(const LZGGraph *g, const double *orders,
                         uint32_t n, LZGHillCurve *out);

void lzg_hill_curve_free(LZGHillCurve *hc);

#endif /* LZGRAPH_ANALYTICS_H */
