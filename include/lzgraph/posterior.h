/**
 * @file posterior.h
 * @brief Bayesian posterior graph: blend population prior with individual data.
 *
 * Given a population graph (prior) and an individual's sequences,
 * creates a new graph with Dirichlet-Multinomial updated edge weights:
 *
 *   w_post(u→v) = (kappa * w_prior(u→v) + count_individual(u→v))
 *                 / (kappa + total_individual_outgoing(u))
 *
 * The concentration parameter kappa controls prior strength:
 * kappa=0 → pure individual data, kappa=∞ → pure population prior.
 */
#ifndef LZGRAPH_POSTERIOR_H
#define LZGRAPH_POSTERIOR_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/**
 * Create a posterior graph by updating a prior with individual sequences.
 *
 * The posterior graph has the same topology as the prior (no new edges
 * are added from the individual data — this keeps the graph structure
 * stable while personalizing the weights).
 *
 * @param prior       The population graph (prior).
 * @param sequences   Individual's sequences.
 * @param n_seqs      Number of individual sequences.
 * @param abundances  Optional abundance per sequence (NULL → all 1).
 * @param kappa       Dirichlet concentration parameter (prior strength).
 * @param out         Output: new posterior graph (caller frees).
 * @return LZG_OK on success.
 */
LZGError lzg_graph_posterior(const LZGGraph *prior,
                        const char **sequences, uint32_t n_seqs,
                        const uint64_t *abundances,
                        double kappa,
                        LZGGraph **out);

#endif /* LZGRAPH_POSTERIOR_H */
