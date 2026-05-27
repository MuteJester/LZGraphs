#ifndef LZGRAPH_GRAPH_FINALIZE_H
#define LZGRAPH_GRAPH_FINALIZE_H

#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"

typedef struct {
    LZGStringPool *gene_pool;
    LZGHashMap *v_marginal_counts;
    LZGHashMap *j_marginal_counts;
    LZGHashMap *vj_pair_counts;
    LZGHashMap *edge_v_genes;
    LZGHashMap *edge_j_genes;
} LZGFinalizeGeneInputs;

void lzg_graph_parse_node_label(const LZGStringPool *pool, uint32_t node_label_id,
                                LZGVariant variant,
                                uint32_t *out_sp_id, uint32_t *out_position,
                                LZGStringPool *sp_pool);

void lzg_graph_alloc_csr_storage(LZGGraph *g, uint32_t n_nodes, uint32_t n_edges);

LZGError lzg_graph_finalize_derived_state(
    LZGGraph *g,
    uint64_t *len_counts, uint32_t max_len,
    const LZGEdgeBuilder *eb,
    const uint32_t *builder_to_csr,
    const LZGFinalizeGeneInputs *gene_inputs);

#endif /* LZGRAPH_GRAPH_FINALIZE_H */
