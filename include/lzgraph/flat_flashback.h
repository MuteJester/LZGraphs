/**
 * @file flat_flashback.h
 * @brief FlattenedFlashBack — FlashBack's bilateral scan without run compression.
 *
 * FlashBack peels matching *runs* from both ends of a sequence, so one token
 * can absorb several identical residues. This variant makes the same bilateral
 * pass but takes exactly one residue from each end per step:
 *
 *   CASSAYFF  ->  @ -> CF_1 -> AF_2 -> SY_3 -> SA_4 -> $
 *   CASSLGQ   ->  @ -> CQ_1 -> AG_2 -> SL_3 -> S$_4 -> $
 *
 * A token is "{front}{back}_{step}". An odd-length sequence leaves one
 * unpaired residue in the middle, which is emitted as "{residue}$_{step}".
 *
 * It sits deliberately between the other two encodings and isolates one
 * variable. Against FlashBack it holds the bilateral scan constant and
 * removes only run compression, so any gap between them is what compressing
 * repeats is worth. Against NaiveGraph it holds "no compression" constant and
 * adds only the bilateral scan, so that gap is what reading from both ends is
 * worth. Neither comparison is available from FlashBack and naive alone.
 *
 * Walks end at an explicit "$" sink. That is not decoration: a pair token
 * cannot signal "stop" by itself, because the same label is terminal in one
 * sequence and internal in a longer one (AA_2 ends CAAF but continues in
 * CAAAAF). Without the sink the model could not place probability on
 * stopping, and would not be a distribution over sequences.
 */
#ifndef LZGRAPH_FLAT_FLASHBACK_H
#define LZGRAPH_FLAT_FLASHBACK_H

#include "lzgraph/analytics.h"
#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"
#include "lzgraph/simulate.h"
#include "lzgraph/string_pool.h"

/** Max nodes in one walk: ceil(len/2) tokens plus the two sentinels. */
#define LZG_FLAT_MAX_WALK 1024

/** Longest label: two residues, '_', up to 10 digits, NUL. */
#define LZG_FLAT_LABEL_CAP 16

/* ── Encoding ─────────────────────────────────────────────── */

/**
 * Encode a sequence into FlattenedFlashBack node labels.
 *
 * Emits "@", then one token per bilateral step, then "$".
 *
 * @param out_ids   Output: interned label IDs, room for len/2 + 3.
 * @param out_count Output: number of labels written.
 */
LZGError lzg_flat_encode(const char *str, uint32_t len,
                         LZGStringPool *pool,
                         uint32_t *out_ids, uint32_t *out_count);

/** Reverse a flattened walk back into its sequence. Exact inverse of encode. */
LZGError lzg_flat_reverse(const LZGStringPool *pool,
                          const uint32_t *label_ids, uint32_t count,
                          char *out_buf, uint32_t buf_cap,
                          uint32_t *out_len);

/* ── Graph construction ───────────────────────────────────── */

/** @param max_length Skip sequences longer than this; 0 means no limit. */
LZGError lzg_flat_graph_build(LZGGraph *g,
                              const char **sequences,
                              uint32_t n_seqs,
                              const uint64_t *abundances,
                              uint32_t max_length,
                              double smoothing);

LZGError lzg_flat_graph_build_file(LZGGraph *g,
                                   const char *path,
                                   uint32_t max_length,
                                   double smoothing);

/** Re-identify root and sink. Only the bare "@" and "$" qualify: a middle
 *  token like "S$_4" contains a sentinel but is an ordinary node. */
void lzg_flat_fix_special_nodes(LZGGraph *g);

/* ── Simulation and probability ───────────────────────────── */

LZGError lzg_flat_simulate(const LZGGraph *g, uint32_t n,
                           LZGRng *rng, LZGSimResult *out);

/** Exact log Pseq; returns LZG_LOG_EPS for unsupported sequences. */
double lzg_flat_pseq(const LZGGraph *g, const char *seq, uint32_t seq_len);

LZGError lzg_flat_pseq_batch(const LZGGraph *g, const char **sequences,
                             uint32_t n, double *out);

/* ── Exact DP analytics ───────────────────────────────────────
 * Forwarded to the shared forward-DP engine, which is generic over the
 * CSR graph. See lib/naive/graph_analytics.c for the same argument.
 */

LZGError lzg_flat_path_count(const LZGGraph *g, double *out);

LZGError lzg_flat_path_count_exact(const LZGGraph *g,
                                   uint32_t **limbs_out,
                                   uint32_t *n_limbs_out);

LZGError lzg_flat_effective_diversity(const LZGGraph *g,
                                      LZGEffectiveDiversity *out);

LZGError lzg_flat_power_sum(const LZGGraph *g, double alpha, double *out);

LZGError lzg_flat_hill_number(const LZGGraph *g, double alpha, double *out);

LZGError lzg_flat_hill_numbers(const LZGGraph *g, const double *orders,
                               uint32_t n, double *out);

LZGError lzg_flat_dynamic_range(const LZGGraph *g, LZGDynamicRange *out);

LZGError lzg_flat_pseq_diagnostics(const LZGGraph *g, double atol,
                                   LZGPgenDiagnostics *out);

#endif /* LZGRAPH_FLAT_FLASHBACK_H */
