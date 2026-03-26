/**
 * @file forward.h
 * @brief Generic LZ-constrained forward propagation engine.
 *
 * This is the computational backbone of all analytical methods. It
 * traverses the DAG in topological order, propagating user-defined
 * accumulator state through edges while enforcing LZ76 dictionary
 * constraints via character bitmasks.
 *
 * The bitmask tracks which single characters have been emitted as
 * tokens. At each edge:
 *   - Length-1 successor: valid only if its char bit is NOT set
 *   - Length-2+ successor: valid only if its prefix char bit IS set
 *
 * The engine maintains a hash map from bitmask → accumulator at each
 * node. Different analytics plug in callbacks for seeding, edge
 * propagation, and terminal absorption.
 *
 * Design: the accumulator is a fixed-size block of doubles. The caller
 * specifies `acc_dim` (number of doubles per accumulator). This avoids
 * virtual dispatch while supporting scalar (dim=1), entropy (dim=2),
 * and moment (dim=5) accumulators with the same engine.
 */
#ifndef LZGRAPH_FORWARD_H
#define LZGRAPH_FORWARD_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/* Maximum accumulator dimension (m0..m4 = 5 doubles) */
#define LZG_FWD_MAX_DIM 8

/**
 * Per-bitmask accumulator entry stored in the hash map at each node.
 * Key = uint32 bitmask packed into uint64.
 * Value index → position in a parallel accumulator array.
 */

/**
 * Callback: seed an initial state.
 *
 * @param acc        Output accumulator (array of acc_dim doubles).
 * @param init_prob  P(start at this node).
 * @param ctx        User context.
 */
typedef void (*LZGFwdSeedFn)(double *acc, double init_prob, void *ctx);

/**
 * Callback: propagate accumulator from node u through an edge to node v.
 *
 * @param dst_acc    Output: accumulator to ADD to at node v.
 * @param src_acc    Input: accumulator at node u (after continue split).
 * @param edge_weight Original edge weight w(u→v).
 * @param renorm_z   Sum of weights of LZ-valid successors at u for this bitmask.
 *                   The constrained transition probability is edge_weight / renorm_z.
 * @param ctx        User context.
 */
typedef void (*LZGFwdEdgeFn)(double *dst_acc, const double *src_acc,
                              double edge_weight, double renorm_z, void *ctx);

/**
 * Callback: absorb mass at a terminal node.
 *
 * @param total_acc  Running total accumulator (add absorbed mass here).
 * @param node_acc   Accumulator at this node.
 * @param stop_prob  P(stop | this node).
 * @param ctx        User context.
 */
typedef void (*LZGFwdAbsorbFn)(double *total_acc, const double *node_acc,
                                double stop_prob, void *ctx);

/**
 * Callback: compute the "continue" accumulator after partial stop absorption.
 *
 * @param cont_acc   Output: accumulator for the continuing fraction.
 * @param node_acc   Input: full accumulator at this node.
 * @param stop_prob  P(stop | this node). Continue factor = 1 - stop_prob.
 * @param ctx        User context.
 */
typedef void (*LZGFwdContinueFn)(double *cont_acc, const double *node_acc,
                                  double stop_prob, void *ctx);

/**
 * Callback set for a specific analytical method.
 */
typedef struct {
    LZGFwdSeedFn      seed;
    LZGFwdEdgeFn      edge;
    LZGFwdAbsorbFn    absorb;
    LZGFwdContinueFn  cont;
    uint32_t          acc_dim;   /* number of doubles per accumulator */
    void             *ctx;       /* user context passed to all callbacks */
} LZGFwdOps;

/**
 * Run the LZ-constrained forward propagation.
 *
 * @param g          The graph (must be topologically sorted).
 * @param ops        Callback set defining the analytical method.
 * @param total_out  Output: final accumulated result (acc_dim doubles).
 * @return LZG_OK on success.
 */
LZGError lzg_forward_propagate(const LZGGraph *g,
                                const LZGFwdOps *ops,
                                double *total_out);

#endif /* LZGRAPH_FORWARD_H */
