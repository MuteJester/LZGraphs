/**
 * @file graph.h
 * @brief CSR (Compressed Sparse Row) graph for finalized LZGraph.
 *
 * After construction via EdgeBuilder, the graph is packed into CSR
 * format for cache-friendly traversal. Nodes are renumbered to match
 * topological order so all DP traversals become linear memory sweeps.
 *
 * All per-edge arrays are parallel to `col_indices` (indexed by edge ID).
 * All per-node arrays are indexed by node ID.
 */
#ifndef LZGRAPH_GRAPH_H
#define LZGRAPH_GRAPH_H

#include "lzgraph/common.h"
#include "lzgraph/string_pool.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/edge_builder.h"

struct LZGExactModel_;

/**
 * LZGraph — directed acyclic graph with sentinel-bounded walks.
 *
 * Every walk starts at the root node (@) and ends at a $-suffixed sink.
 * The probability of a walk is the product of renormalized edge weights:
 *   P(walk) = Π P(edge_t | LZ-valid edges at step t)
 *
 * No separate initial/terminal/stop probability — everything is edge weights.
 */
typedef struct LZGGraph_ {
    /* ── Dimensions ── */
    uint32_t n_nodes;
    uint32_t n_edges;

    /* ── CSR adjacency ── */
    uint32_t *row_offsets;     /* [n_nodes + 1]: edge range per node       */
    uint32_t *col_indices;     /* [n_edges]: destination node per edge     */

    /* ── Per-edge data (parallel to col_indices) ── */
    double   *edge_weights;    /* [n_edges]: normalized P(dst|src)         */
    uint64_t *edge_counts;     /* [n_edges]: raw transition counts         */

    /* ── Per-edge LZ constraint info ── */
    uint32_t *edge_sp_id;      /* [n_edges]: interned subpattern of dst    */
    uint8_t  *edge_sp_len;     /* [n_edges]: length of dst subpattern      */
    uint32_t *edge_prefix_id;  /* [n_edges]: interned prefix of dst sp     */

    /* ── Per-node data ── */
    uint64_t *outgoing_counts; /* [n_nodes]: total raw outgoing count      */
    uint32_t *node_sp_id;      /* [n_nodes]: interned subpattern string    */
    uint8_t  *node_sp_len;     /* [n_nodes]: subpattern character length   */
    uint32_t *node_pos;        /* [n_nodes]: cumulative position integer   */
    uint8_t  *node_is_sink;    /* [n_nodes]: 1 if this is a $-terminal     */

    /* ── Root node ── */
    uint32_t  root_node;       /* index of the @ root node                 */

    /* ── Topological order ── */
    uint32_t *topo_order;      /* [n_nodes]: node IDs in topological order */
    bool      topo_valid;

    /* ── Length distribution ── */
    uint64_t *length_counts;   /* indexed by sequence length               */
    uint32_t  max_length;      /* largest observed sequence length         */

    /* ── String pool (owns all string data) ── */
    LZGStringPool *pool;

    /* ── Configuration ── */
    LZGVariant variant;
    double     smoothing_alpha;

    /* ── Gene data (optional, NULL if no gene columns provided) ── */
    struct LZGGeneData_ *gene_data;

    /* ── Transient query cache (not serialized) ── */
    LZGHashMap *query_node_map; /* structural key -> node index */
    uint64_t   *edge_sp_hash;   /* [n_edges]: hash of destination token     */
    uint64_t   *edge_prefix_hash; /* [n_edges]: hash of prefix token or 0   */
    uint64_t   *node_sp_hash;   /* [n_nodes]: hash of node token            */
    uint8_t    *edge_single_char_idx; /* [n_edges]: aa bit index or UINT8_MAX */
    uint8_t    *node_single_char_idx; /* [n_nodes]: aa bit index or UINT8_MAX */
    struct LZGExactModel_ *exact_model_cache; /* accepted-model normalizer cache */
} LZGGraph;

/** Allocate and initialize an empty graph. */
LZGGraph *lzg_graph_create(LZGVariant variant);

/** Free all graph memory. */
void lzg_graph_destroy(LZGGraph *g);

/**
 * Build a graph from sequences.
 *
 * @param g           An empty graph (from lzg_graph_create).
 * @param sequences   Array of null-terminated sequence strings.
 * @param n_seqs      Number of sequences.
 * @param abundances  Optional abundance per sequence (NULL → all 1).
 * @param v_genes     Optional V gene per sequence (NULL → no gene data).
 * @param j_genes     Optional J gene per sequence (NULL → no gene data).
 * @param smoothing   Laplace smoothing alpha (0 = no smoothing).
 * @param min_init    Deprecated (ignored).
 * @return LZG_OK on success.
 */
LZGError lzg_graph_build(LZGGraph *g,
                          const char **sequences,
                          uint32_t n_seqs,
                          const uint64_t *abundances,
                          const char **v_genes,
                          const char **j_genes,
                          double smoothing,
                          uint32_t min_init);

/**
 * Build a graph from a plain text file without materializing all sequences in Python.
 *
 * Supported input formats:
 * - one sequence per line
 * - sequence<TAB>abundance
 *
 * Gene columns and headered tabular formats are not supported by this path.
 */
LZGError lzg_graph_build_plain_file(LZGGraph *g,
                                     const char *path,
                                     double smoothing);

/** Compute topological sort (cached, invalidated on structural changes). */
LZGError lzg_graph_topo_sort(LZGGraph *g);

/**
 * Internal: finalize a graph from pre-collected edge data.
 * Used by lzg_graph_build and lzg_graph_union.
 * All hash maps are consumed (destroyed) by this function.
 */
LZGError lzg_graph_finalize_from_edges(
    LZGGraph *g,
    LZGEdgeBuilder *eb,
    LZGHashMap *node_set,
    LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts,
    LZGHashMap *outgoing_counts,
    uint64_t *len_counts, uint32_t max_len);

/* ── Recalculation flags ───────────────────────────────────── */

typedef enum {
    LZG_RECALC_WEIGHTS    = 0x01,  /* edge_weights from edge_counts    */
    LZG_RECALC_ALL        = 0x01,
} LZGRecalcFlags;

/**
 * Recompute derived quantities from raw counts in place.
 * Call after modifying edge_counts.
 * Topology (row_offsets, col_indices) must remain unchanged.
 */
LZGError lzg_graph_recalculate(LZGGraph *g, uint32_t flags);

/**
 * Internal: ensure transient query-side edge hash caches exist.
 * These caches are derived from immutable edge metadata and are not serialized.
 */
LZGError lzg_graph_ensure_query_edge_hashes(LZGGraph *g);

/** Get the number of successors for node `node_id`. */
static inline uint32_t lzg_graph_out_degree(const LZGGraph *g, uint32_t node_id) {
    return g->row_offsets[node_id + 1] - g->row_offsets[node_id];
}

#endif /* LZGRAPH_GRAPH_H */
