#include "graph_ops_internal.h"
#include <stdlib.h>

LZGError lzg_graph_summary_impl(const LZGGraph *g, LZGGraphSummary *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;

    out->n_nodes = g->n_nodes;
    out->n_edges = g->n_edges;
    out->n_initial = 1; /* single root @ */
    out->n_terminal = 0;
    out->max_out_degree = 0;
    out->is_dag = g->topo_valid;

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint32_t out_degree = g->row_offsets[i + 1] - g->row_offsets[i];
        if (g->node_is_sink && g->node_is_sink[i]) out->n_terminal++;
        if (out_degree > out->max_out_degree) out->max_out_degree = out_degree;
    }

    uint32_t *in_degree = calloc(g->n_nodes, sizeof(uint32_t));
    if (!in_degree) return LZG_ERR_ALLOC;

    for (uint32_t e = 0; e < g->n_edges; e++)
        in_degree[g->col_indices[e]]++;

    out->max_in_degree = 0;
    out->n_isolates = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint32_t out_degree = g->row_offsets[i + 1] - g->row_offsets[i];
        if (in_degree[i] > out->max_in_degree) out->max_in_degree = in_degree[i];
        if (in_degree[i] == 0 && out_degree == 0) out->n_isolates++;
    }

    free(in_degree);
    return LZG_OK;
}
