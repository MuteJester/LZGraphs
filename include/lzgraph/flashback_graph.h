/**
 * @file flashback_graph.h
 * @brief FlashBackGraph — Markovian graph from FlashBack decomposition.
 *
 * All functions operate on the standard LZGGraph struct but use FlashBack
 * tokenization. The resulting graph is a DAG with no LZ constraints,
 * enabling exact DP analytics and simple Markov simulation.
 */
#ifndef LZGRAPH_FLASHBACK_GRAPH_H
#define LZGRAPH_FLASHBACK_GRAPH_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/simulate.h"
#include "lzgraph/analytics.h"
#include "lzgraph/rng.h"

/* ── Graph construction ───────────────────────────────────── */

LZGError lzg_flashback_graph_build(LZGGraph *g,
                                   const char **sequences,
                                   uint32_t n_seqs,
                                   const uint64_t *abundances,
                                   double smoothing);

LZGError lzg_flashback_graph_build_file(LZGGraph *g,
                                        const char *path,
                                        double smoothing);

/* ── Streaming construction ───────────────────────────────────
 *
 * The streaming builder exposes the inner per-sequence accumulator that
 * the list and file builders use, letting callers feed sequences in
 * batches and inspect the running node/edge counts before deciding to
 * stop. Designed for "build a graph from an open-ended source" — e.g.,
 * a generator that produces sequences in real time, or a corpus too
 * large to materialise into a list.
 *
 * Lifecycle:
 *
 *   stream = lzg_flashback_stream_open(smoothing);
 *   while (more sequences):
 *       lzg_flashback_stream_add(stream, batch, n, counts);
 *       lzg_flashback_stream_peek(stream, &n_nodes, &n_edges);
 *       (decide to stop based on RAM, plateau, etc.)
 *   lzg_flashback_stream_finalize(stream, &graph);   // graph owned by caller
 *   // OR: lzg_flashback_stream_abort(stream);       // discard everything
 *
 * After finalize, the stream object is freed and `graph` is owned by
 * the caller (destroy with lzg_graph_destroy when done). After abort
 * the stream is freed and no graph is produced. Calling add/peek/
 * finalize on a finalized or aborted stream is undefined.
 */

typedef struct LZGFlashbackStream LZGFlashbackStream;

/** Open a new streaming FlashBack builder. Returns NULL on allocation
 *  failure. The stream allocates its own LZGGraph internally. */
LZGFlashbackStream *lzg_flashback_stream_open(double smoothing);

/** Append `n_seqs` sequences to the stream. `counts` may be NULL
 *  (treated as all 1). Empty / null sequences and zero counts are
 *  silently skipped. Returns LZG_ERR_INVALID_ARG if the stream is
 *  already finalized. */
LZGError lzg_flashback_stream_add(LZGFlashbackStream *s,
                                   const char *const *sequences,
                                   uint32_t n_seqs,
                                   const uint64_t *counts);

/** Read current node and edge counts (cheap; no work done).
 *  Either pointer may be NULL. Returns 0/0 if stream is finalized. */
void lzg_flashback_stream_peek(const LZGFlashbackStream *s,
                                uint32_t *out_n_nodes,
                                uint32_t *out_n_edges);

/** Finalize: build CSR from the accumulator, fix special nodes, and
 *  hand back the graph in *out (caller takes ownership). On error
 *  *out is set to NULL. The stream itself is freed regardless of
 *  success — do not call any further functions on it. */
LZGError lzg_flashback_stream_finalize(LZGFlashbackStream *s,
                                        LZGGraph **out);

/** Snapshot: build a CSR graph from the current accumulator state
 *  WITHOUT consuming the stream. The returned graph in *out owns its
 *  own CSR storage but borrows (via refcount) the stream's string
 *  pool, so the snapshot must be destroyed before the stream itself
 *  is. After this call the stream remains live and may continue to
 *  accept add/peek/snapshot/finalize/abort. On error *out is NULL.
 *  Returns LZG_ERR_INVALID_ARG if the stream is already finalized or
 *  aborted. */
LZGError lzg_flashback_stream_snapshot(LZGFlashbackStream *s,
                                        LZGGraph **out);

/** Abort without producing a graph. Releases the graph and the
 *  accumulator but does NOT free the stream struct itself — call
 *  ``lzg_flashback_stream_destroy`` for that. Safe to call on NULL,
 *  and idempotent (a second call is a no-op). */
void lzg_flashback_stream_abort(LZGFlashbackStream *s);

/** Free the stream struct itself. Implicitly aborts if the stream
 *  was not already finalized or aborted. Safe to call on NULL.
 *  After this call the pointer is invalid. */
void lzg_flashback_stream_destroy(LZGFlashbackStream *s);

/** Re-identify root and sink nodes for FlashBack token conventions.
 *  Call after loading a saved graph to restore correct root/sink state. */
void lzg_flashback_fix_special_nodes(LZGGraph *g);

/* ── Bayesian posterior ─────────────────────────────────────── */

/**
 * Create a Bayesian posterior FlashBackGraph: same topology as `prior`
 * but edge weights updated via Dirichlet-Multinomial:
 *
 *   w_post(u->v) = (kappa * w_prior(u->v) + c_ind(u->v))
 *                  / (kappa + n_ind(u))
 *
 * where c_ind is the individual's count of edge (u,v) and n_ind(u) is
 * their total outgoing count at node u, both derived from FlashBack
 * decomposition of `sequences`. kappa=0 -> pure individual; kappa->inf
 * -> pure prior.
 *
 * Edges not present in `prior` are ignored; only the prior's topology is
 * retained. The output is a new graph; the caller owns it.
 */
LZGError lzg_flashback_graph_posterior(const LZGGraph *prior,
                                       const char **sequences,
                                       uint32_t n_seqs,
                                       const uint64_t *abundances,
                                       double kappa,
                                       LZGGraph **out);

/* ── Repertoire subtraction ────────────────────────────────── */

/**
 * Return a new FlashBackGraph with the contribution of `sequences`
 * removed: for each sequence, decompose via FlashBack and subtract its
 * abundance from every edge count on its token walk. Edges whose count
 * reaches 0 after subtraction are physically pruned from the CSR.
 * Isolated nodes (no in- or out-edges after pruning) are retained for
 * node-index stability; Python-side accessors filter them.
 *
 * Subtraction is clamped at zero (never goes negative). Edges not
 * present in the graph are silently ignored. Per-node weights are
 * renormalised after pruning; topo_order and sink flags are rebuilt.
 *
 * Used primarily to construct a leave-donor-out foundation from an
 * existing graph without rebuilding from source repertoires.
 */
LZGError lzg_flashback_graph_subtract(const LZGGraph *g,
                                      const char **sequences,
                                      uint32_t n_seqs,
                                      const uint64_t *abundances,
                                      LZGGraph **out);

/* ── Markov simulation ────────────────────────────────────── */

LZGError lzg_flashback_simulate(const LZGGraph *g, uint32_t n,
                                LZGRng *rng, LZGSimResult *out);

/* ── Sequence probability ─────────────────────────────────── */

double lzg_flashback_pgen(const LZGGraph *g,
                          const char *seq, uint32_t seq_len);

LZGError lzg_flashback_pgen_batch(const LZGGraph *g,
                                  const char **sequences,
                                  uint32_t n, double *out);

/* ── Exact DP analytics ───────────────────────────────────── */

LZGError lzg_flashback_path_count(const LZGGraph *g, double *out);

/**
 * Exact root-to-sink path count in arbitrary precision.
 *
 * The double-returning variant above loses precision above 2^53 and
 * overflows to +inf past ~1.8e308; real repertoire graphs exceed 2^53.
 * This variant returns the exact integer as little-endian base-2^32
 * limbs. On success `*limbs_out` is a malloc'd array the caller must
 * free(); `*n_limbs_out` is 0 (with `*limbs_out` NULL) when the count
 * is zero.
 */
LZGError lzg_flashback_path_count_exact(const LZGGraph *g,
                                        uint32_t **limbs_out,
                                        uint32_t *n_limbs_out);

/**
 * Root-to-sink path counts grouped by reconstructed sequence length.
 *
 * Counts are accumulated in double precision. This preserves roughly 16
 * significant decimal digits while keeping the dense length state compact;
 * unlike the arbitrary-precision scalar counter, low-order integer digits may
 * therefore be rounded once counts exceed 2^53. On success `*counts_out` is a
 * calloc'd array indexed from zero through `*max_length_out`, inclusive, and
 * must be freed by the caller.
 */
LZGError lzg_flashback_path_count_by_length(const LZGGraph *g,
                                            double **counts_out,
                                            uint32_t *max_length_out);

/**
 * Structural metadata needed to initialize p-sequence analysis.
 *
 * `symbol_lengths_out` must address `g->n_nodes` bytes and receives the
 * reconstructed amino-acid contribution of each node (sentinels and token
 * metadata excluded). The remaining outputs describe the exact reachable
 * root-to-sink surprisal range and maximum number of edges in such a path.
 */
LZGError lzg_flashback_pseq_init(const LZGGraph *g,
                                 uint8_t *symbol_lengths_out,
                                 double *min_surprisal_out,
                                 double *max_surprisal_out,
                                 uint32_t *max_edges_out);

/**
 * Mellin-transform derivatives grouped by reconstructed sequence length.
 *
 * Computes derivatives zero through `order` of
 * `sum_s P(s)^q` independently for every generated amino-acid length. The
 * returned row-major array has (`*max_length_out + 1`) rows and
 * (`order + 1`) columns. `*present_out[length]` distinguishes structurally
 * reachable lengths from numerical zero after float64 conversion. Both
 * output arrays are malloc'd and must be freed by the caller.
 */
LZGError lzg_flashback_pseq_length_derivatives(
    const LZGGraph *g, double q, uint32_t order,
    double **derivatives_out, uint8_t **present_out,
    uint32_t *max_length_out);

/**
 * Paired counting and generated mass grouped by amino-acid length.
 *
 * This is the zeroth-order result of
 * `lzg_flashback_pseq_length_derivatives()` at `q=0` and `q=1`, computed in
 * one traversal. Counting states use double precision; generated-mass states
 * use long-double accumulation. The two returned arrays are indexed from zero
 * through `*max_length_out`, inclusive. `*present_out[length]` identifies
 * structurally reachable root-to-sink lengths. All three arrays are malloc'd
 * and must be freed by the caller.
 */
LZGError lzg_flashback_pseq_length_marginals(
    const LZGGraph *g, double **counting_out, double **generated_out,
    uint8_t **present_out, uint32_t *max_length_out);

/**
 * Global Mellin-transform derivatives for the p-sequence distribution.
 *
 * Computes derivatives zero through `order` of `sum_s P(s)^q` using
 * long-double internal accumulation. `derivatives_out` must address
 * `order + 1` doubles supplied by the caller.
 */
LZGError lzg_flashback_pseq_derivatives(const LZGGraph *g, double q,
                                        uint32_t order,
                                        double *derivatives_out);

/**
 * Stable normalized log-probability moments under Mellin tilt `q`.
 *
 * For path weights proportional to `P(s)^q`, returns `log(sum_s P(s)^q)`
 * together with normalized raw and central moments of `log P(s)` through
 * `order`. Log-sum-exp normalization is maintained at every node, so this
 * interface remains finite when the unnormalized Mellin derivatives overflow
 * or underflow. `raw_moments_out` and `central_moments_out` must each address
 * `order + 1` doubles. Orders zero through eight are supported.
 */
LZGError lzg_flashback_pseq_tilted_moments(
    const LZGGraph *g, double q, uint32_t order,
    double *log_mass_out, double *raw_moments_out,
    double *central_moments_out);

/**
 * Batched saddlepoint approximation for generated-sequence surprisal.
 *
 * Solves `K'(t)=x` with safeguarded Newton iterations, where
 * `K(t)=log(M(1-t)/M(1))`, then evaluates the saddlepoint density and the
 * Lugannani-Rice CDF. Inputs outside the exact reachable surprisal interval
 * receive their limiting CDF and zero density. Every output array must address
 * `n` elements; `iterations_out` may be NULL.
 */
LZGError lzg_flashback_pseq_saddlepoint_batch(
    const LZGGraph *g, const double *x, uint32_t n,
    double *pdf_out, double *cdf_out, double *saddle_out,
    uint32_t *iterations_out);

/** Tilted node and edge usage probabilities for a FlashBack DAG. */
typedef struct {
    uint32_t n_nodes;
    uint32_t n_edges;
    double q;
    double log_mass;
    double *node_probability;  /**< `n_nodes` normalized marginals. */
    double *edge_probability;  /**< `n_edges` normalized marginals. */
} LZGPseqAttribution;

/**
 * Compute exact-DP structural attribution under path weights `P(s)^q`.
 *
 * If `pi_q(s) = P(s)^q / sum_r P(r)^q`, each node or edge probability is
 * the probability that a path drawn from `pi_q` visits that object. The
 * calculation uses log-domain forward and backward dynamic programs. The
 * returned arrays are owned by `out` and must be released with
 * `lzg_flashback_pseq_attribution_destroy`.
 *
 * For an edge `e`, `q * edge_probability[e]` is
 * `d log M(q) / d log w_e` when edge weights are treated as independent.
 */
LZGError lzg_flashback_pseq_attribution(
    const LZGGraph *g, double q, LZGPseqAttribution *out);

/** Release arrays owned by a p-sequence attribution result. Safe on NULL. */
void lzg_flashback_pseq_attribution_destroy(LZGPseqAttribution *result);

/**
 * Diversity conditioned on retaining edges whose weight exceeds a threshold.
 *
 * For every non-decreasing value `thresholds[j]`, paths may traverse exactly
 * those original graph edges with `weight > thresholds[j]`. Terminal states
 * remain the sinks of the unmodified graph: an internal node stranded by the
 * threshold is a dead end, not a newly invented sequence termination.
 *
 * If the surviving paths have their original probabilities `P(s)`, the
 * kernel computes `N = sum 1`, `Z = sum P`, `A = sum P log(P)`, and
 * `Q = sum P^2`, then returns natural logarithms of the conditioned Hill
 * diversities
 *
 *     log D0 = log N
 *     log D1 = log Z - A/Z
 *     log D2 = 2 log Z - log Q.
 *
 * Empty surviving supports receive `-INFINITY` for all three logarithms and
 * zero mass. `kept_edges_out` counts all retained CSR edges, including edges
 * not reachable from the root. All output arrays must address `n_thresholds`
 * elements. `n_thresholds == 0` is a no-op and permits NULL array pointers.
 */
LZGError lzg_flashback_edge_threshold_diversity(
    const LZGGraph *g, const double *thresholds, uint32_t n_thresholds,
    double *log_d0_out, double *log_d1_out, double *log_d2_out,
    double *surviving_mass_out, uint64_t *kept_edges_out);

/**
 * Deterministic linear-grid reconstruction of a p-sequence measure.
 *
 * `length` is -1 for the global spectrum or a non-negative literal
 * reconstructed amino-acid length; sentinels and token metadata do not
 * contribute. On success `*weights_out` contains `bins` float64 grid masses
 * and must be freed by the caller. The grid coordinate for index `i` is
 * `i * *spacing_out`.
 */
LZGError lzg_flashback_pseq_histogram(const LZGGraph *g, uint32_t bins,
                                      double q, int64_t length,
                                      double **weights_out,
                                      double *spacing_out,
                                      double *true_max_surprisal_out,
                                      uint32_t *max_edges_out);

/**
 * Fused global counting and generated p-sequence histograms.
 *
 * This is mathematically equivalent to two global calls to
 * `lzg_flashback_pseq_histogram()` with `q=0` and `q=1`, respectively, but
 * computes grid bounds, edge shifts, and graph traversal once. Both measures
 * use the same surprisal grid and linear transport rule. On success the two
 * `bins`-element float64 arrays are owned by the caller and must be freed.
 */
LZGError lzg_flashback_pseq_histogram_pair(
    const LZGGraph *g, uint32_t bins,
    double **counting_weights_out, double **generated_weights_out,
    double *spacing_out, double *true_max_surprisal_out,
    uint32_t *max_edges_out);

LZGError lzg_flashback_effective_diversity(const LZGGraph *g,
                                           LZGEffectiveDiversity *out);

LZGError lzg_flashback_power_sum(const LZGGraph *g, double alpha,
                                  double *out);

LZGError lzg_flashback_hill_number(const LZGGraph *g, double alpha,
                                    double *out);

LZGError lzg_flashback_hill_numbers(const LZGGraph *g,
                                     const double *orders,
                                     uint32_t n, double *out);

LZGError lzg_flashback_dynamic_range(const LZGGraph *g,
                                      LZGDynamicRange *out);

LZGError lzg_flashback_pgen_diagnostics(const LZGGraph *g, double atol,
                                         LZGPgenDiagnostics *out);

/* ── Top-K walks ─────────────────────────────────────────────── */

/**
 * Find the K most (or least) probable complete walks through the DAG.
 *
 * Uses a forward DP on topological order, maintaining K-best partial paths
 * per node. Sequences are reconstructed via lzg_flashback_reverse().
 *
 * @param g              The graph (must have valid topo order).
 * @param k              Number of walks to return.
 * @param most_probable  If true, return the K highest-probability walks.
 *                       If false, return the K lowest-probability walks.
 * @param out            Output array of LZGSimResult[k] (caller allocates).
 * @param out_count      Output: actual number of results (<= k).
 * @return LZG_OK on success.
 */
LZGError lzg_flashback_top_k_walks(const LZGGraph *g, uint32_t k, bool most_probable,
                         LZGSimResult *out, uint32_t *out_count);

#endif /* LZGRAPH_FLASHBACK_GRAPH_H */
