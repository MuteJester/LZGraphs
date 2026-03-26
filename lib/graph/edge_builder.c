/**
 * @file edge_builder.c
 * @brief Dynamic edge accumulator: insert-or-increment with hash map.
 */
#include "lzgraph/edge_builder.h"
#include <stdlib.h>

LZGEdgeBuilder *lzg_eb_create(uint32_t initial_capacity) {
    if (initial_capacity < 256) initial_capacity = 256;

    LZGEdgeBuilder *eb = calloc(1, sizeof(LZGEdgeBuilder));
    if (!eb) return NULL;

    eb->edge_map = lzg_hm_create(initial_capacity * 2);
    eb->src_ids  = malloc(initial_capacity * sizeof(uint32_t));
    eb->dst_ids  = malloc(initial_capacity * sizeof(uint32_t));
    eb->counts   = malloc(initial_capacity * sizeof(uint32_t));
    eb->capacity = initial_capacity;
    eb->n_edges  = 0;

    if (!eb->edge_map || !eb->src_ids || !eb->dst_ids || !eb->counts) {
        lzg_eb_destroy(eb);
        return NULL;
    }
    return eb;
}

void lzg_eb_destroy(LZGEdgeBuilder *eb) {
    if (!eb) return;
    lzg_hm_destroy(eb->edge_map);
    free(eb->src_ids);
    free(eb->dst_ids);
    free(eb->counts);
    free(eb);
}

LZGError lzg_eb_record(LZGEdgeBuilder *eb,
                        uint32_t src_id, uint32_t dst_id,
                        uint32_t count) {
    uint64_t key = lzg_eb_pack_key(src_id, dst_id);
    uint64_t *existing = lzg_hm_get(eb->edge_map, key);

    if (existing) {
        /* Edge exists — increment count */
        uint32_t idx = (uint32_t)*existing;
        eb->counts[idx] += count;
        return LZG_OK;
    }

    /* New edge — grow arrays if needed */
    if (eb->n_edges >= eb->capacity) {
        uint32_t new_cap = eb->capacity * 2;
        eb->src_ids = realloc(eb->src_ids, new_cap * sizeof(uint32_t));
        eb->dst_ids = realloc(eb->dst_ids, new_cap * sizeof(uint32_t));
        eb->counts  = realloc(eb->counts,  new_cap * sizeof(uint32_t));
        eb->capacity = new_cap;
        if (!eb->src_ids || !eb->dst_ids || !eb->counts)
            return LZG_ERR_ALLOC;
    }

    uint32_t idx = eb->n_edges;
    eb->src_ids[idx] = src_id;
    eb->dst_ids[idx] = dst_id;
    eb->counts[idx]  = count;
    eb->n_edges++;

    lzg_hm_put(eb->edge_map, key, (uint64_t)idx);
    return LZG_OK;
}
