#include "graph_ops_internal.h"
#include "lzgraph/edge_builder.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    LZGStringPool *pool;
    LZGHashMap *edge_counts;
    LZGHashMap *edge_counts_a;
    LZGHashMap *edge_counts_b;
    LZGHashMap *node_set;
    uint64_t *length_counts;
} LZGSetOpScratch;

static void lzg_set_op_scratch_destroy(LZGSetOpScratch *scratch) {
    if (!scratch) return;
    if (scratch->pool) lzg_sp_destroy(scratch->pool);
    lzg_hm_destroy(scratch->edge_counts);
    lzg_hm_destroy(scratch->edge_counts_a);
    lzg_hm_destroy(scratch->edge_counts_b);
    lzg_hm_destroy(scratch->node_set);
    free(scratch->length_counts);
}

static LZGError lzg_validate_set_op_inputs(const LZGGraph *a, const LZGGraph *b,
                                           LZGGraph **out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;
    *out = NULL;
    if (a->variant != b->variant) {
        return LZG_FAIL(
            LZG_ERR_VARIANT_MISMATCH,
            "cannot combine graphs with different variants (a=%d, b=%d)",
            a->variant, b->variant);
    }
    return LZG_OK;
}

static uint32_t intern_node_label(const LZGGraph *src, uint32_t node_idx,
                                  LZGStringPool *dst_pool) {
    const char *sp = lzg_sp_get(src->pool, src->node_sp_id[node_idx]);
    uint32_t pos = src->node_pos[node_idx];
    char buf[256];
    int len;

    if (src->variant == LZG_VARIANT_NAIVE) {
        len = snprintf(buf, sizeof(buf), "%s", sp);
    } else {
        len = snprintf(buf, sizeof(buf), "%s_%u", sp, pos);
    }

    return lzg_sp_intern_n(dst_pool, buf, (uint32_t)len);
}

static void include_edge_nodes(LZGHashMap *node_set, uint64_t edge_key) {
    uint32_t src = (uint32_t)(edge_key >> 32);
    uint32_t dst = (uint32_t)(edge_key & 0xFFFFFFFFu);

    (void)lzg_hm_get_or_insert(node_set, src, 1, NULL);
    (void)lzg_hm_get_or_insert(node_set, dst, 1, NULL);
}

static void collect_graph_edges(const LZGGraph *src,
                                LZGStringPool *dst_pool,
                                LZGHashMap *edge_counts,
                                LZGHashMap *node_set) {
    for (uint32_t u = 0; u < src->n_nodes; u++) {
        uint32_t src_id = intern_node_label(src, u, dst_pool);
        if (node_set) (void)lzg_hm_get_or_insert(node_set, src_id, 1, NULL);

        uint32_t e_start = src->row_offsets[u];
        uint32_t e_end = src->row_offsets[u + 1];
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = src->col_indices[e];
            uint32_t dst_id = intern_node_label(src, v, dst_pool);
            if (node_set) (void)lzg_hm_get_or_insert(node_set, dst_id, 1, NULL);

            uint64_t edge_key = lzg_eb_pack_key(src_id, dst_id);
            (void)lzg_hm_add_u64(edge_counts, edge_key, src->edge_counts[e], NULL);
        }
    }
}

static uint64_t *merge_lengths(const LZGGraph *a, const LZGGraph *b,
                               uint32_t *out_max) {
    uint32_t max_len = (a->max_length > b->max_length)
        ? a->max_length : b->max_length;
    uint64_t *length_counts = calloc(max_len + 1, sizeof(uint64_t));

    if (!length_counts) return NULL;

    for (uint32_t i = 0; i <= a->max_length; i++)
        length_counts[i] += a->length_counts[i];
    for (uint32_t i = 0; i <= b->max_length; i++)
        length_counts[i] += b->length_counts[i];

    *out_max = max_len;
    return length_counts;
}

static uint64_t *copy_lengths(const LZGGraph *src, uint32_t *out_max) {
    uint64_t *length_counts = calloc(src->max_length + 1, sizeof(uint64_t));

    if (!length_counts) return NULL;

    memcpy(length_counts, src->length_counts,
           (src->max_length + 1) * sizeof(uint64_t));
    *out_max = src->max_length;
    return length_counts;
}

static LZGError build_from_edge_map(LZGVariant variant,
                                    double smoothing,
                                    uint32_t max_len,
                                    LZGSetOpScratch *scratch,
                                    LZGGraph **out) {
    LZGGraph *graph = calloc(1, sizeof(LZGGraph));
    LZGEdgeBuilder *builder = NULL;
    LZGHashMap *initial_counts = NULL;
    LZGHashMap *terminal_counts = NULL;
    LZGHashMap *outgoing_counts = NULL;
    LZGError err = LZG_OK;

    if (!graph) return LZG_ERR_ALLOC;

    graph->variant = variant;
    graph->smoothing_alpha = smoothing;
    graph->pool = scratch->pool;
    scratch->pool = NULL;

    builder = lzg_eb_create((scratch->edge_counts ? scratch->edge_counts->count : 0u) * 2u + 1u);
    initial_counts = lzg_hm_create(16);
    terminal_counts = lzg_hm_create(16);
    outgoing_counts = lzg_hm_create(4096);
    if (!builder || !initial_counts || !terminal_counts || !outgoing_counts) {
        lzg_eb_destroy(builder);
        lzg_hm_destroy(initial_counts);
        lzg_hm_destroy(terminal_counts);
        lzg_hm_destroy(outgoing_counts);
        lzg_graph_destroy(graph);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t i = 0; i < scratch->edge_counts->capacity; i++) {
        uint64_t edge_key = scratch->edge_counts->keys[i];
        uint64_t count;
        uint32_t src;
        uint32_t dst;

        if (edge_key == LZG_HM_EMPTY || edge_key == LZG_HM_DELETED) continue;

        count = scratch->edge_counts->values[i];
        if (count == 0) continue;

        src = (uint32_t)(edge_key >> 32);
        dst = (uint32_t)(edge_key & 0xFFFFFFFFu);
        err = lzg_eb_record(builder, src, dst, count, NULL);
        if (err != LZG_OK) {
            lzg_eb_destroy(builder);
            lzg_hm_destroy(initial_counts);
            lzg_hm_destroy(terminal_counts);
            lzg_hm_destroy(outgoing_counts);
            lzg_graph_destroy(graph);
            return err;
        }
    }

    lzg_hm_destroy(scratch->edge_counts);
    scratch->edge_counts = NULL;

    for (uint32_t i = 0; i < builder->n_edges; i++)
        (void)lzg_hm_add_u64(outgoing_counts, builder->src_ids[i], builder->counts[i], NULL);

    err = lzg_graph_finalize_from_edges(
        graph, builder, scratch->node_set,
        initial_counts, terminal_counts, outgoing_counts,
        scratch->length_counts, max_len);
    scratch->node_set = NULL;
    scratch->length_counts = NULL;

    if (err != LZG_OK) {
        lzg_graph_destroy(graph);
        return err;
    }

    *out = graph;
    return LZG_OK;
}

LZGError lzg_graph_union_impl(const LZGGraph *a, const LZGGraph *b,
                              LZGGraph **out) {
    LZGSetOpScratch scratch = {0};
    uint32_t max_len = 0;
    LZGError err = lzg_validate_set_op_inputs(a, b, out);

    if (err != LZG_OK) return err;

    scratch.pool = lzg_sp_create(4096);
    scratch.edge_counts = lzg_hm_create((a->n_edges + b->n_edges) * 2u);
    scratch.node_set = lzg_hm_create(a->n_nodes + b->n_nodes);
    scratch.length_counts = merge_lengths(a, b, &max_len);
    if (!scratch.pool || !scratch.edge_counts || !scratch.node_set ||
        !scratch.length_counts) {
        lzg_set_op_scratch_destroy(&scratch);
        return LZG_ERR_ALLOC;
    }

    collect_graph_edges(a, scratch.pool, scratch.edge_counts, scratch.node_set);
    collect_graph_edges(b, scratch.pool, scratch.edge_counts, scratch.node_set);

    err = build_from_edge_map(a->variant, a->smoothing_alpha, max_len,
                              &scratch, out);
    lzg_set_op_scratch_destroy(&scratch);
    return err;
}

LZGError lzg_graph_intersection_impl(const LZGGraph *a, const LZGGraph *b,
                                     LZGGraph **out) {
    LZGSetOpScratch scratch = {0};
    uint32_t max_len = 0;
    LZGError err = lzg_validate_set_op_inputs(a, b, out);

    if (err != LZG_OK) return err;

    scratch.pool = lzg_sp_create(4096);
    scratch.edge_counts_a = lzg_hm_create(a->n_edges * 2u);
    scratch.edge_counts_b = lzg_hm_create(b->n_edges * 2u);
    scratch.edge_counts = lzg_hm_create(a->n_edges * 2u);
    scratch.node_set = lzg_hm_create(a->n_nodes + b->n_nodes);
    scratch.length_counts = merge_lengths(a, b, &max_len);
    if (!scratch.pool || !scratch.edge_counts_a || !scratch.edge_counts_b ||
        !scratch.edge_counts || !scratch.node_set || !scratch.length_counts) {
        lzg_set_op_scratch_destroy(&scratch);
        return LZG_ERR_ALLOC;
    }

    collect_graph_edges(a, scratch.pool, scratch.edge_counts_a, NULL);
    collect_graph_edges(b, scratch.pool, scratch.edge_counts_b, NULL);

    for (uint32_t i = 0; i < scratch.edge_counts_a->capacity; i++) {
        uint64_t edge_key = scratch.edge_counts_a->keys[i];
        uint64_t *count_b;
        uint64_t count;

        if (edge_key == LZG_HM_EMPTY || edge_key == LZG_HM_DELETED) continue;

        count_b = lzg_hm_get(scratch.edge_counts_b, edge_key);
        if (!count_b) continue;

        count = scratch.edge_counts_a->values[i] < *count_b
            ? scratch.edge_counts_a->values[i] : *count_b;
        lzg_hm_put(scratch.edge_counts, edge_key, count);
        include_edge_nodes(scratch.node_set, edge_key);
    }

    err = build_from_edge_map(a->variant, a->smoothing_alpha, max_len,
                              &scratch, out);
    lzg_set_op_scratch_destroy(&scratch);
    return err;
}

LZGError lzg_graph_difference_impl(const LZGGraph *a, const LZGGraph *b,
                                   LZGGraph **out) {
    LZGSetOpScratch scratch = {0};
    uint32_t max_len = 0;
    LZGError err = lzg_validate_set_op_inputs(a, b, out);

    if (err != LZG_OK) return err;

    scratch.pool = lzg_sp_create(4096);
    scratch.edge_counts_a = lzg_hm_create(a->n_edges * 2u);
    scratch.edge_counts_b = lzg_hm_create(b->n_edges * 2u);
    scratch.edge_counts = lzg_hm_create(a->n_edges * 2u);
    scratch.node_set = lzg_hm_create(a->n_nodes);
    scratch.length_counts = copy_lengths(a, &max_len);
    if (!scratch.pool || !scratch.edge_counts_a || !scratch.edge_counts_b ||
        !scratch.edge_counts || !scratch.node_set || !scratch.length_counts) {
        lzg_set_op_scratch_destroy(&scratch);
        return LZG_ERR_ALLOC;
    }

    collect_graph_edges(a, scratch.pool, scratch.edge_counts_a, NULL);
    collect_graph_edges(b, scratch.pool, scratch.edge_counts_b, NULL);

    for (uint32_t i = 0; i < scratch.edge_counts_a->capacity; i++) {
        uint64_t edge_key = scratch.edge_counts_a->keys[i];
        uint64_t *count_b;
        uint64_t count_a;

        if (edge_key == LZG_HM_EMPTY || edge_key == LZG_HM_DELETED) continue;

        count_a = scratch.edge_counts_a->values[i];
        count_b = lzg_hm_get(scratch.edge_counts_b, edge_key);
        if (count_b && count_a <= *count_b) continue;

        lzg_hm_put(scratch.edge_counts, edge_key,
                   count_b ? (count_a - *count_b) : count_a);
        include_edge_nodes(scratch.node_set, edge_key);
    }

    err = build_from_edge_map(a->variant, a->smoothing_alpha, max_len,
                              &scratch, out);
    lzg_set_op_scratch_destroy(&scratch);
    return err;
}

LZGError lzg_graph_weighted_merge_impl(const LZGGraph *a, const LZGGraph *b,
                                       double alpha, double beta,
                                       LZGGraph **out) {
    LZGSetOpScratch scratch = {0};
    uint32_t max_len = 0;
    LZGError err = lzg_validate_set_op_inputs(a, b, out);

    if (err != LZG_OK) return err;

    scratch.pool = lzg_sp_create(4096);
    scratch.edge_counts_a = lzg_hm_create(a->n_edges * 2u);
    scratch.edge_counts_b = lzg_hm_create(b->n_edges * 2u);
    scratch.edge_counts = lzg_hm_create((a->n_edges + b->n_edges) * 2u);
    scratch.node_set = lzg_hm_create(a->n_nodes + b->n_nodes);
    scratch.length_counts = merge_lengths(a, b, &max_len);
    if (!scratch.pool || !scratch.edge_counts_a || !scratch.edge_counts_b ||
        !scratch.edge_counts || !scratch.node_set || !scratch.length_counts) {
        lzg_set_op_scratch_destroy(&scratch);
        return LZG_ERR_ALLOC;
    }

    collect_graph_edges(a, scratch.pool, scratch.edge_counts_a, NULL);
    collect_graph_edges(b, scratch.pool, scratch.edge_counts_b, NULL);

    for (uint32_t i = 0; i < scratch.edge_counts_a->capacity; i++) {
        uint64_t edge_key = scratch.edge_counts_a->keys[i];
        uint64_t scaled_count;

        if (edge_key == LZG_HM_EMPTY || edge_key == LZG_HM_DELETED) continue;

        scaled_count = (uint64_t)round(alpha * scratch.edge_counts_a->values[i]);
        if (scaled_count == 0) continue;

        lzg_hm_put(scratch.edge_counts, edge_key, scaled_count);
        include_edge_nodes(scratch.node_set, edge_key);
    }

    for (uint32_t i = 0; i < scratch.edge_counts_b->capacity; i++) {
        uint64_t edge_key = scratch.edge_counts_b->keys[i];
        uint64_t scaled_count;

        if (edge_key == LZG_HM_EMPTY || edge_key == LZG_HM_DELETED) continue;

        scaled_count = (uint64_t)round(beta * scratch.edge_counts_b->values[i]);
        if (scaled_count == 0) continue;

        (void)lzg_hm_add_u64(scratch.edge_counts, edge_key, scaled_count, NULL);
        include_edge_nodes(scratch.node_set, edge_key);
    }

    err = build_from_edge_map(a->variant, a->smoothing_alpha, max_len,
                              &scratch, out);
    lzg_set_op_scratch_destroy(&scratch);
    return err;
}
