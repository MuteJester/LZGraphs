/**
 * @file gene_data.h
 * @brief V/J gene annotations on edges and graph-level marginals.
 *
 * Gene data is OPTIONAL — graphs built without V/J columns have
 * gene_data == NULL.
 *
 * Per-edge gene counts are stored in CSR-within-CSR format:
 * - gene_offsets[e] .. gene_offsets[e+1] indexes into gene_ids/gene_counts
 * - Sorted by gene_id for binary search
 *
 * Graph-level data:
 * - marginal_v/j: normalized frequency of each V/J gene across all sequences
 * - vj_probs: joint V×J co-occurrence probabilities
 */
#ifndef LZGRAPH_GENE_DATA_H
#define LZGRAPH_GENE_DATA_H

#include "lzgraph/common.h"
#include "lzgraph/string_pool.h"

typedef struct LZGGeneData_ {
    /* ── Gene string pool (separate namespace from node labels) ── */
    LZGStringPool *gene_pool;

    /* ── Per-edge V gene counts (CSR-within-CSR) ── */
    uint32_t *v_offsets;      /* [n_edges + 1]: range per edge         */
    uint32_t *v_gene_ids;     /* [total_v_entries]: interned gene IDs  */
    uint32_t *v_gene_counts;  /* [total_v_entries]: raw counts         */
    uint32_t  total_v_entries;

    /* ── Per-edge J gene counts (CSR-within-CSR) ── */
    uint32_t *j_offsets;
    uint32_t *j_gene_ids;
    uint32_t *j_gene_counts;
    uint32_t  total_j_entries;

    /* ── Graph-level marginal distributions ── */
    uint32_t  n_v_genes;       /* number of distinct V genes           */
    uint32_t *v_marginal_ids;  /* [n_v_genes]: gene IDs                */
    double   *v_marginal_probs;/* [n_v_genes]: normalized probabilities */

    uint32_t  n_j_genes;
    uint32_t *j_marginal_ids;
    double   *j_marginal_probs;

    /* ── VJ joint distribution ── */
    uint32_t  n_vj_pairs;
    uint32_t *vj_v_ids;        /* [n_vj_pairs]: V gene ID per pair     */
    uint32_t *vj_j_ids;        /* [n_vj_pairs]: J gene ID per pair     */
    double   *vj_probs;        /* [n_vj_pairs]: joint probability      */
} LZGGeneData;

/** Create an empty gene data container. */
LZGGeneData *lzg_gene_data_create(void);

/** Free all gene data. */
void lzg_gene_data_destroy(LZGGeneData *gd);

/**
 * Query: probability of V gene on edge e.
 * Returns 0.0 if gene not found.
 */
double lzg_edge_v_prob(const LZGGeneData *gd, uint32_t edge_idx,
                        uint32_t v_gene_id);

/**
 * Query: probability of J gene on edge e.
 */
double lzg_edge_j_prob(const LZGGeneData *gd, uint32_t edge_idx,
                        uint32_t j_gene_id);

/**
 * Check: does edge e have V gene v_gene_id?
 */
bool lzg_edge_has_v(const LZGGeneData *gd, uint32_t edge_idx,
                     uint32_t v_gene_id);

/**
 * Check: does edge e have J gene j_gene_id?
 */
bool lzg_edge_has_j(const LZGGeneData *gd, uint32_t edge_idx,
                     uint32_t j_gene_id);

#endif /* LZGRAPH_GENE_DATA_H */
