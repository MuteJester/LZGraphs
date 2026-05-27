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

/* ── FlashBack Anomaly Score (FBAS) ───────────────────────── */

/**
 * Per-sequence anomaly diagnosis result.
 *
 * FBAS = max(depth_weighted_excess_surprise) + mean(depth_weighted_excess_surprise)
 *
 * where for each transition (token_d -> token_{d+1}):
 *   excess_surprise = max(0, -log(edge_weight) - node_entropy(src))
 *   depth_weight    = 1.0 - 0.6 * (d / (max_depth - 1))   if max_depth > 1
 *                   = 1.0                                  otherwise
 *
 * Missing tokens or edges in the graph yield fixed-penalty contributions
 * (50.0 and 30.0 respectively) and increment the corresponding counters.
 */
typedef struct LZGFbasResult_ {
    double   fbas;             /* depth-weighted max + mean of excess-surprise */
    double   log_pgen;         /* natural log P_gen of the full walk          */
    double   worst_excess;     /* max raw (unweighted) excess-surprise         */
    uint32_t n_missing_tokens; /* count of transitions with unknown source     */
    uint32_t n_missing_edges;  /* count of transitions with known src, no edge */
} LZGFbasResult;

/* Missing-element penalties used in FBAS scoring. */
#define LZG_FBAS_MISSING_TOKEN_PENALTY 50.0
#define LZG_FBAS_MISSING_EDGE_PENALTY  30.0

/* ── FBAS helper ─────────────────────────────────────────────── */

/**
 * Per-node Shannon entropy of outgoing edge probabilities.
 *
 * Lazily computed and cached in the graph's transient `node_entropy`
 * array on first call; subsequent calls are O(1). Returns 0.0 for sink
 * nodes and graphs without outgoing edges.
 *
 * Used internally by lzg_flashback_fbas; exposed for callers that want
 * to pre-warm the cache before a batch.
 */
double lzg_flashback_node_entropy(const LZGGraph *g, uint32_t node);

/**
 * Compute FBAS / log_pgen / worst_excess for a single sequence via a
 * single graph walk. Roughly mirrors the Python diagnose_sequence path
 * but ~50-100x faster (no per-call entropy recomputation, no Python
 * loop overhead).
 *
 * @param g       The graph.
 * @param seq     Null-terminated CDR3 amino acid sequence.
 * @param seq_len Sequence length (excluding null terminator).
 * @param out     Output result struct (caller allocates).
 * @return LZG_OK on success.
 */
LZGError lzg_flashback_fbas(const LZGGraph *g,
                            const char *seq, uint32_t seq_len,
                            LZGFbasResult *out);

/**
 * Batch FBAS for `n` sequences. Equivalent to calling lzg_flashback_fbas
 * `n` times but with shared setup and tight C-side iteration.
 */
LZGError lzg_flashback_fbas_batch(const LZGGraph *g,
                                  const char **sequences,
                                  uint32_t n,
                                  LZGFbasResult *out);

/* ── Exact DP analytics ───────────────────────────────────── */

LZGError lzg_flashback_path_count(const LZGGraph *g, double *out);

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
