/**
 * @file diversity.h
 * @brief Diversity metrics: perplexity, entropy rate, K-diversity, saturation.
 *
 * All probability-based metrics use the public constrained walk model
 * exposed through lzg_walk_log_prob().
 *
 * Counting-based metrics (K-diversity, saturation) operate on
 * raw sequences without touching the probability model.
 */
#ifndef LZGRAPH_DIVERSITY_H
#define LZGRAPH_DIVERSITY_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"

/* ── Perplexity (probability-based, uses constrained model) ── */

/**
 * Perplexity of a single sequence under the LZ-constrained model.
 * PP(s) = 2^{-log2 P(s) / n_tokens}
 *
 * Lower perplexity = sequence fits the model better.
 * Returns INFINITY if sequence has P = 0 (missing edges).
 */
double lzg_sequence_perplexity(const LZGGraph *g,
                                const char *seq, uint32_t seq_len);

/**
 * Repertoire perplexity: geometric mean of per-sequence perplexities.
 * Equivalent to 2^{mean of per-sequence cross-entropies}.
 *
 * @param g          The graph.
 * @param sequences  Array of sequences.
 * @param n          Number of sequences.
 * @return           Geometric mean perplexity (INFINITY if any sequence has P=0).
 */
double lzg_repertoire_perplexity(const LZGGraph *g,
                                  const char **sequences, uint32_t n);

/**
 * Path entropy rate: average bits per token across sequences.
 * h = (1/N) Σ [-log2 P(s_i) / |s_i|]
 *
 * This is the correct information-theoretic summary of model fit
 * under LZ constraints.
 */
double lzg_path_entropy_rate(const LZGGraph *g,
                              const char **sequences, uint32_t n);

/* ── K-Diversity (counting-based, no probability model) ────── */

typedef struct {
    double mean;     /* mean unique subpatterns across draws */
    double std;      /* standard deviation */
    double ci_low;   /* 95% CI lower bound */
    double ci_high;  /* 95% CI upper bound */
} LZGKDiversity;

/**
 * K-diversity: expected number of unique LZ subpatterns in a
 * random subsample of `sample_size` sequences.
 *
 * @param sequences   Array of sequences.
 * @param n           Total number of sequences.
 * @param variant     Graph variant (determines encoding).
 * @param sample_size Number of sequences per subsample.
 * @param draws       Number of resampling draws.
 * @param rng         RNG state.
 * @param out         Output.
 */
LZGError lzg_k_diversity(const char **sequences, uint32_t n,
                          LZGVariant variant,
                          uint32_t sample_size, uint32_t draws,
                          LZGRng *rng, LZGKDiversity *out);

/* ── Saturation curve (counting-based) ─────────────────────── */

typedef struct {
    uint32_t n_sequences;  /* sequences processed so far */
    uint32_t n_nodes;      /* unique nodes seen */
    uint32_t n_edges;      /* unique edges seen */
} LZGSaturationPoint;

/**
 * Compute a saturation curve: unique nodes/edges as sequences accumulate.
 *
 * @param sequences  Array of sequences (processed in order).
 * @param n          Number of sequences.
 * @param variant    Graph variant.
 * @param log_every  Record a point every `log_every` sequences.
 * @param out        Output array (caller allocates, size n/log_every + 1).
 * @param out_count  Output: number of points written.
 */
LZGError lzg_saturation_curve(const char **sequences, uint32_t n,
                               LZGVariant variant,
                               uint32_t log_every,
                               LZGSaturationPoint *out,
                               uint32_t *out_count);

/* ── Jensen-Shannon Divergence between two graphs ──────────── */

/**
 * JSD between the node-frequency distributions of two graphs.
 * Returns a value in [0, ln(2)] (nats) or [0, 1] (bits).
 *
 * @param a, b  Two graphs (same variant).
 * @param out   Output: JSD in nats.
 */
LZGError lzg_jensen_shannon_divergence(const LZGGraph *a,
                                        const LZGGraph *b,
                                        double *out);

#endif /* LZGRAPH_DIVERSITY_H */
