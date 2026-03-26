/**
 * @file forward.c
 * @brief Forward propagation engine on graph topology.
 *
 * Propagates accumulators through the DAG in topological order.
 * At each node, probability mass is absorbed at terminals and
 * propagated through outgoing edges with renormalized weights.
 *
 * This is the unconstrained forward DP — edge weights are used
 * directly without LZ dictionary constraint filtering. For exact
 * LZ-constrained probability computation, use lzg_walk_log_prob()
 * which enforces constraints dynamically per-walk.
 */
#include "lzgraph/forward.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_forward_propagate(const LZGGraph *g,
                                const LZGFwdOps *ops,
                                double *total_out) {
    if (!g || !ops || !total_out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;

    const uint32_t dim = ops->acc_dim;
    const uint32_t n_nodes = g->n_nodes;
    memset(total_out, 0, dim * sizeof(double));

    double *node_acc = calloc(n_nodes * dim, sizeof(double));
    double *cont_buf = malloc(dim * sizeof(double));
    double *edge_buf = malloc(dim * sizeof(double));
    if (!node_acc || !cont_buf || !edge_buf) {
        free(node_acc); free(cont_buf); free(edge_buf);
        return LZG_ERR_ALLOC;
    }

    /* ── Seed from root node (@) with probability 1.0 ── */
    if (g->root_node < n_nodes) {
        ops->seed(&node_acc[g->root_node * dim], 1.0, ops->ctx);
    }

    /* ── Forward pass in topological order ── */
    for (uint32_t t = 0; t < n_nodes; t++) {
        uint32_t u = g->topo_order[t];
        double *u_acc = &node_acc[u * dim];

        /* Skip nodes with no mass */
        double mass = 0.0;
        for (uint32_t d = 0; d < dim; d++) mass += fabs(u_acc[d]);
        if (mass < LZG_EPS) continue;

        /* Sink nodes ($-terminal): absorb all mass, stop probability = 1.0 */
        bool is_sink = g->node_is_sink && g->node_is_sink[u];
        if (is_sink) {
            ops->absorb(total_out, u_acc, 1.0, ops->ctx);
            continue; /* no outgoing edges from sinks */
        }

        /* Non-sink node: all mass continues to successors */
        memcpy(cont_buf, u_acc, dim * sizeof(double));

        /* Renormalize over outgoing edges */
        uint32_t e_start = g->row_offsets[u];
        uint32_t e_end   = g->row_offsets[u + 1];
        double Z = 0.0;
        for (uint32_t e = e_start; e < e_end; e++)
            Z += g->edge_weights[e];
        if (Z < LZG_EPS) continue;

        /* Propagate mass to successors */
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = g->col_indices[e];
            double w = g->edge_weights[e];
            memset(edge_buf, 0, dim * sizeof(double));
            ops->edge(edge_buf, cont_buf, w, Z, ops->ctx);
            for (uint32_t d = 0; d < dim; d++)
                node_acc[v * dim + d] += edge_buf[d];
        }
    }

    free(node_acc);
    free(cont_buf);
    free(edge_buf);
    return LZG_OK;
}
