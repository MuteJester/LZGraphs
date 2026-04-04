/**
 * @file analytics.h
 * @brief Graph-level analytics computed via LZ-constrained Monte Carlo.
 *
 * All public analytics are Monte Carlo estimates, not exact graph invariants.
 * They sample the accepted-walk model (same as simulate()/lzpgen()) and
 * compute diversity/entropy/support statistics from the sampled log-probs.
 *
 * Default sample count is LZG_ANALYTICS_DEFAULT_MC_SAMPLES (10000).
 * Functions with _mc suffixes accept a custom sample count.
 *
 * pgen_diagnostics() estimates absorbed/leaked mass of the raw structural
 * walk law via a separate MC run — see its doc for caveats.
 */
#ifndef LZGRAPH_ANALYTICS_H
#define LZGRAPH_ANALYTICS_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

#define LZG_ANALYTICS_DEFAULT_MC_SAMPLES 10000u

/* ── Simulation potential size ──────────────────────────────── */

/**
 * Estimate the number of distinct completed LZ-valid walks (support size).
 */
LZGError lzg_graph_path_count(const LZGGraph *g, double *out_count);

/**
 * Monte Carlo version of lzg_graph_path_count().
 *
 * @param n_samples Number of simulated walks. Use 0 for the default.
 */
LZGError lzg_graph_path_count_mc(const LZGGraph *g, uint32_t n_samples,
                                  double *out_count);

/* ── PGEN diagnostics ───────────────────────────────────────── */

typedef struct {
    double total_absorbed;          /* MC estimate of absorbed mass (≤ 1.0) */
    double total_leaked;            /* 1 - total_absorbed                   */
    double initial_prob_sum;        /* should be 1.0                        */
    bool   is_proper;               /* |total_absorbed - 1.0| < atol       */
    uint32_t mc_samples;            /* number of MC samples used            */
} LZGPgenDiagnostics;

/**
 * Estimate absorption/leakage of the raw structural walk law via Monte Carlo.
 *
 * total_absorbed is a Monte Carlo estimate (not an exact proof): it reports
 * the fraction of raw constrained walks that reached a sink node out of
 * mc_samples trials. A value of 1.0 means "no leaks were observed," not
 * "the graph is mathematically proven leak-free." Rare leakage events may
 * be missed if mc_samples is too small relative to the leakage probability.
 *
 * @param g    The graph.
 * @param atol Tolerance for is_proper check.
 * @param out  Output diagnostics.
 */
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
 * Estimate the power sum M(α) = Σ_s π(s)^α of the public accepted-walk model.
 * In particular, M(1) = 1 for every graph with at least one live completion.
 */
LZGError lzg_power_sum(const LZGGraph *g, double alpha, double *out_m);

/**
 * Estimate the classical Hill number of the public accepted-walk law.
 * For α = 0: returns support size.
 * For α = 1: returns exp(Shannon entropy of the absorbed law).
 */
LZGError lzg_hill_number(const LZGGraph *g, double alpha, double *out_d);

/**
 * Monte Carlo version of lzg_hill_number().
 *
 * @param n_samples Number of simulated walks. Use 0 for the default.
 */
LZGError lzg_hill_number_mc(const LZGGraph *g, double alpha,
                             uint32_t n_samples, double *out_d);

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

/**
 * Monte Carlo version of lzg_hill_numbers().
 *
 * @param n_samples Number of simulated walks. Use 0 for the default.
 */
LZGError lzg_hill_numbers_mc(const LZGGraph *g, const double *orders,
                              uint32_t n, uint32_t n_samples, double *out);

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
