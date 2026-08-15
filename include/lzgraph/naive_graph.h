/**
 * @file naive_graph.h
 * @brief NaiveGraph — one node per residue per position.
 *
 * The deliberately simple control for FlashBack. A node is a single
 * character together with its 1-based index in the sequence, so the walk
 * for "CASS" is
 *
 *   @ -> C_1 -> A_2 -> S_3 -> S_4 -> $
 *
 * Nothing about repeat structure enters the node alphabet: the encoding
 * knows only "which residue" and "how far along". Comparing this graph
 * with a FlashBackGraph trained on the same sequences isolates the value
 * of the FlashBack decomposition itself, since the two models are
 * identical in every other respect (same CSR engine, same MLE edge
 * weights, same exact forward-DP analytics).
 *
 * Because the index strictly increases along a walk, the graph is a DAG
 * and the walk/sequence map is a bijection. That is what makes the path
 * count equal the support size, and what lets every analytic below be
 * exact rather than sampled.
 *
 * Sentinels are the bare strings "@" and "$", which is exactly what the
 * generic `lzg_graph_identify_special_nodes` expects — unlike FlashBack,
 * this variant needs no root/sink fixup after finalize or load.
 */
#ifndef LZGRAPH_NAIVE_GRAPH_H
#define LZGRAPH_NAIVE_GRAPH_H

#include "lzgraph/analytics.h"
#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"
#include "lzgraph/simulate.h"
#include "lzgraph/string_pool.h"

/** Max nodes in one walk, i.e. longest sequence + 2 sentinels. */
#define LZG_NAIVE_MAX_WALK 1024

/** Longest node label: one char, '_', up to 10 digits, NUL. */
#define LZG_NAIVE_LABEL_CAP 16

/* ── Encoding ─────────────────────────────────────────────── */

/**
 * Encode a sequence into naive positional node labels.
 *
 * Produces `len + 2` labels: "@", then "{c}_{i}" for each character at
 * 1-based index i, then "$". Labels are interned into `pool`.
 *
 * @param str        Input sequence (no sentinels).
 * @param len        Length of the sequence.
 * @param pool       String pool for interning labels.
 * @param out_ids    Output: interned label IDs, room for len + 2.
 * @param out_count  Output: number of labels written.
 * @return LZG_OK, or LZG_ERR_PARAM_OUT_OF_RANGE if the walk would exceed
 *         LZG_NAIVE_MAX_WALK.
 */
LZGError lzg_naive_encode(const char *str, uint32_t len,
                          LZGStringPool *pool,
                          uint32_t *out_ids, uint32_t *out_count);

/**
 * Reverse a naive walk back into its sequence.
 *
 * Each non-sentinel label contributes its first character; "@" and "$"
 * contribute nothing. Exact inverse of lzg_naive_encode.
 */
LZGError lzg_naive_reverse(const LZGStringPool *pool,
                           const uint32_t *label_ids, uint32_t count,
                           char *out_buf, uint32_t buf_cap,
                           uint32_t *out_len);

/* ── Graph construction ───────────────────────────────────── */

/**
 * Build from an in-memory sequence array.
 *
 * @param max_length Skip sequences longer than this; 0 means no limit.
 *                   Capping keeps the node set bounded at 20 * max_length
 *                   and, more importantly, makes a comparison against
 *                   another model well defined by fixing what both see.
 */
LZGError lzg_naive_graph_build(LZGGraph *g,
                               const char **sequences,
                               uint32_t n_seqs,
                               const uint64_t *abundances,
                               uint32_t max_length,
                               double smoothing);

/**
 * Build by streaming a plain text file, in constant memory.
 *
 * Accepts one sequence per line or "sequence<TAB>abundance". Gene columns
 * and headered tabular formats are not handled here.
 */
LZGError lzg_naive_graph_build_file(LZGGraph *g,
                                    const char *path,
                                    uint32_t max_length,
                                    double smoothing);

/**
 * Re-identify root and sink nodes.
 *
 * The bare "@"/"$" labels already satisfy the generic finalize pass, so
 * this exists only for symmetry with FlashBack and to repair a graph
 * whose special-node flags were dropped by a generic set operation.
 */
void lzg_naive_fix_special_nodes(LZGGraph *g);

/* ── Simulation and probability ───────────────────────────── */

LZGError lzg_naive_simulate(const LZGGraph *g, uint32_t n,
                            LZGRng *rng, LZGSimResult *out);

/** Exact log P(seq); returns LZG_LOG_EPS for unsupported sequences. */
double lzg_naive_pgen(const LZGGraph *g,
                      const char *seq, uint32_t seq_len);

LZGError lzg_naive_pgen_batch(const LZGGraph *g,
                              const char **sequences,
                              uint32_t n, double *out);

/* ── Exact DP analytics ───────────────────────────────────────
 *
 * These forward to the shared forward-DP engine, which is generic over
 * the CSR graph. They are declared here so callers need not reach into
 * another variant's header for them.
 */

LZGError lzg_naive_path_count(const LZGGraph *g, double *out);

/** Exact path count in arbitrary precision (little-endian base-2^32
 *  limbs, malloc'd; caller frees). Real graphs exceed 2^53, where the
 *  double-returning variant above starts losing low-order digits. */
LZGError lzg_naive_path_count_exact(const LZGGraph *g,
                                    uint32_t **limbs_out,
                                    uint32_t *n_limbs_out);

LZGError lzg_naive_effective_diversity(const LZGGraph *g,
                                       LZGEffectiveDiversity *out);

LZGError lzg_naive_power_sum(const LZGGraph *g, double alpha, double *out);

LZGError lzg_naive_hill_number(const LZGGraph *g, double alpha, double *out);

LZGError lzg_naive_hill_numbers(const LZGGraph *g, const double *orders,
                                uint32_t n, double *out);

LZGError lzg_naive_dynamic_range(const LZGGraph *g, LZGDynamicRange *out);

LZGError lzg_naive_pgen_diagnostics(const LZGGraph *g, double atol,
                                    LZGPgenDiagnostics *out);

#endif /* LZGRAPH_NAIVE_GRAPH_H */
