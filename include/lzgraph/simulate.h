/**
 * @file simulate.h
 * @brief LZ76-constrained sequence simulation and walk probability.
 *
 * simulate() generates sequences from a scalable accepted-sequence
 * approximation: raw constrained walks are sampled forward until one
 * reaches a terminal sink, and probabilities are normalized by a cached
 * Monte Carlo estimate of the graph's absorption mass.
 *
 * walk_log_probability() uses that same cached normalization estimate, so
 * its output stays consistent with simulate() without materializing the full
 * history-expanded exact state space.
 */
#ifndef LZGRAPH_SIMULATE_H
#define LZGRAPH_SIMULATE_H

#include <stdlib.h>
#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/rng.h"

/**
 * Result of a single simulated walk.
 */
typedef struct {
    char     *sequence;       /* generated string (owned, null-terminated) */
    double    log_prob;       /* log P(sequence) under LZ-constrained model */
    uint32_t  seq_len;        /* length of sequence */
    uint32_t  n_tokens;       /* number of LZ tokens in the walk */
} LZGSimResult;

/**
 * Simulate n sequences from the public accepted-sequence model.
 *
 * Each trial walk enforces LZ76 dictionary constraints exactly, stops on
 * dead ends, and retries until a completed sink-reaching walk is obtained.
 * This keeps generation scalable on large graphs while still returning only
 * completed accepted walks.
 *
 * @param g          The graph (must be finalized + topo-sorted).
 * @param n          Number of sequences to generate.
 * @param rng        RNG state (modified in place).
 * @param out        Output array of n LZGSimResult structs (caller allocates).
 * @return LZG_OK on success.
 */
LZGError lzg_simulate(const LZGGraph *g, uint32_t n,
                       LZGRng *rng, LZGSimResult *out);

/**
 * Free the sequence string in a simulation result.
 */
static inline void lzg_sim_result_free(LZGSimResult *r) {
    if (r && r->sequence) { free(r->sequence); r->sequence = NULL; }
}

/**
 * Compute the log-probability of a sequence under the public accepted model.
 *
 * The sequence is re-encoded via LZ76, producing its canonical raw
 * constrained walk, and then normalized by the same cached Monte Carlo
 * absorption estimate used by simulate(). This keeps lzpgen() consistent
 * with public generation semantics without the exact state explosion.
 *
 * @param g       The graph.
 * @param seq     Null-terminated amino acid sequence.
 * @param seq_len Length of the sequence.
 * @return Log-probability (negative value), or LZG_LOG_EPS on error.
 */
double lzg_walk_log_prob(const LZGGraph *g,
                                 const char *seq, uint32_t seq_len);

/**
 * Compute log-probabilities for a batch of sequences.
 *
 * @param g         The graph.
 * @param sequences Array of null-terminated strings.
 * @param n         Number of sequences.
 * @param out_logps Output array of n doubles (caller allocates).
 * @return LZG_OK on success.
 */
LZGError lzg_walk_log_prob_batch(const LZGGraph *g,
                                         const char **sequences,
                                         uint32_t n,
                                         double *out_logps);

/* ── Gene-constrained simulation ──────────────────────────── */

/**
 * Result of a single gene-constrained simulation.
 */
typedef struct {
    LZGSimResult base;       /* sequence, log_prob, seq_len, n_tokens */
    uint32_t     v_gene_id;  /* V gene used (interned ID in gene_pool) */
    uint32_t     j_gene_id;  /* J gene used (interned ID in gene_pool) */
} LZGGeneSimResult;

/**
 * Simulate n sequences constrained to specific V/J genes.
 *
 * For each sequence:
 *   1. Sample a (V,J) pair from the joint VJ distribution
 *   2. Walk with edges filtered by LZ + live + V/J gene presence
 *   3. Backtrack on dead ends (stack with blacklist)
 *
 * The reported log_prob is the gene-conditional walk probability: the
 * product of transition probabilities over gene-filtered, LZ-valid edges.
 * This is a different probability law from the unconditional model returned
 * by simulate() / lzg_walk_log_prob(). The two are not directly comparable.
 *
 * @param g     Graph with gene_data != NULL.
 * @param n     Number of sequences to generate.
 * @param rng   RNG state.
 * @param out   Output array (caller allocates n elements).
 * @return LZG_OK on success, LZG_ERR_INVALID_ARG if gene_data is NULL.
 */
LZGError lzg_gene_simulate(const LZGGraph *g, uint32_t n,
                             LZGRng *rng, LZGGeneSimResult *out);

/**
 * Simulate n sequences constrained to a specific V and J gene.
 *
 * @param g         Graph with gene_data != NULL.
 * @param n         Number of sequences to generate.
 * @param rng       RNG state.
 * @param v_gene_id V gene to constrain to (LZG_SP_NOT_FOUND = no V filter).
 * @param j_gene_id J gene to constrain to (LZG_SP_NOT_FOUND = no J filter).
 * @param out       Output array (caller allocates n elements).
 * @return LZG_OK on success.
 */
LZGError lzg_gene_simulate_vj(const LZGGraph *g, uint32_t n,
                                LZGRng *rng,
                                uint32_t v_gene_id, uint32_t j_gene_id,
                                LZGGeneSimResult *out);

/**
 * Free a gene simulation result (inner resources only).
 */
static inline void lzg_gene_sim_result_free(LZGGeneSimResult *r) {
    if (r) lzg_sim_result_free(&r->base);
}

#endif /* LZGRAPH_SIMULATE_H */
