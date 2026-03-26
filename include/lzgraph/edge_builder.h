/**
 * @file edge_builder.h
 * @brief Dynamic edge accumulator for graph construction.
 *
 * During construction, sequences are processed one at a time. Each
 * sequence produces edges between consecutive LZ76 tokens. The
 * EdgeBuilder accumulates (src, dst) → count using a hash map for
 * fast insert-or-increment.
 *
 * After all sequences are processed, `lzg_eb_finalize()` packs the
 * edges into CSR format for efficient querying.
 */
#ifndef LZGRAPH_EDGE_BUILDER_H
#define LZGRAPH_EDGE_BUILDER_H

#include "lzgraph/common.h"
#include "lzgraph/hash_map.h"

typedef struct LZGEdgeBuilder_ {
    LZGHashMap *edge_map;      /* key: pack(src,dst), value: edge index  */
    uint32_t   *src_ids;       /* [n_edges] source node IDs             */
    uint32_t   *dst_ids;       /* [n_edges] destination node IDs        */
    uint32_t   *counts;        /* [n_edges] raw transition counts       */
    uint32_t    n_edges;
    uint32_t    capacity;
} LZGEdgeBuilder;

/** Create an edge builder with initial capacity. */
LZGEdgeBuilder *lzg_eb_create(uint32_t initial_capacity);

/** Destroy the edge builder. */
void lzg_eb_destroy(LZGEdgeBuilder *eb);

/**
 * Record a transition from src → dst with the given count.
 * If the edge already exists, increments its count.
 */
LZGError lzg_eb_record(LZGEdgeBuilder *eb,
                        uint32_t src_id, uint32_t dst_id,
                        uint32_t count);

/** Pack key from (src, dst) pair. */
static inline uint64_t lzg_eb_pack_key(uint32_t src, uint32_t dst) {
    return ((uint64_t)src << 32) | (uint64_t)dst;
}

#endif /* LZGRAPH_EDGE_BUILDER_H */
