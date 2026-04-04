#include "graph_finalize.h"
#include "../simulation/exact_model.h"
#include <stdlib.h>
#include <string.h>

static void sort_compact_gene_segment(uint32_t lo, uint32_t hi,
                                      uint32_t *gene_ids,
                                      uint64_t *gene_counts) {
    if (hi - lo < 2) return;

    for (uint32_t i = lo + 1; i < hi; i++) {
        uint32_t key_id = gene_ids[i];
        uint64_t key_count = gene_counts[i];
        uint32_t j = i;
        while (j > lo && gene_ids[j - 1] > key_id) {
            gene_ids[j] = gene_ids[j - 1];
            gene_counts[j] = gene_counts[j - 1];
            j--;
        }
        gene_ids[j] = key_id;
        gene_counts[j] = key_count;
    }
}

static uint32_t sort_compact_gene_csr(uint32_t n_edges,
                                      uint32_t *offsets,
                                      uint32_t *gene_ids,
                                      uint64_t *gene_counts) {
    uint32_t old_lo = 0;
    uint32_t write = 0;

    offsets[0] = 0;
    for (uint32_t e = 0; e < n_edges; e++) {
        uint32_t old_hi = offsets[e + 1];
        uint32_t seg_start = write;

        sort_compact_gene_segment(old_lo, old_hi, gene_ids, gene_counts);

        for (uint32_t i = old_lo; i < old_hi; i++) {
            if (write > seg_start && gene_ids[write - 1] == gene_ids[i]) {
                gene_counts[write - 1] += gene_counts[i];
            } else {
                gene_ids[write] = gene_ids[i];
                gene_counts[write] = gene_counts[i];
                write++;
            }
        }

        offsets[e] = seg_start;
        offsets[e + 1] = write;
        old_lo = old_hi;
    }

    return write;
}

static void lzg_graph_identify_special_nodes(LZGGraph *g) {
    g->node_is_sink = calloc(g->n_nodes, sizeof(uint8_t));
    g->root_node = UINT32_MAX;

    if (!g->node_is_sink) return;

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        uint8_t sp_len = g->node_sp_len[i];
        if (sp_len == 1 && sp[0] == LZG_START_SENTINEL)
            g->root_node = i;
        if (sp_len > 0 && sp[sp_len - 1] == LZG_END_SENTINEL)
            g->node_is_sink[i] = 1;
    }

    if (g->root_node == UINT32_MAX)
        LZG_WARN("no @ root node found — graph may not have sentinel encoding");
}

static void lzg_graph_normalize_edge_weights(LZGGraph *g) {
    double smoothing = g->smoothing_alpha;

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint32_t start = g->row_offsets[i];
        uint32_t end = g->row_offsets[i + 1];
        uint64_t total = g->outgoing_counts[i];

        for (uint32_t e = start; e < end; e++) {
            if (smoothing > 0.0) {
                uint32_t k = end - start;
                g->edge_weights[e] = (g->edge_counts[e] + smoothing) /
                                     (total + smoothing * k);
            } else {
                g->edge_weights[e] = total > 0
                    ? (double)g->edge_counts[e] / total
                    : 0.0;
            }
        }
    }
}

static void lzg_graph_populate_edge_lz_metadata(LZGGraph *g) {
    for (uint32_t e = 0; e < g->n_edges; e++) {
        uint32_t dst = g->col_indices[e];
        g->edge_sp_id[e] = g->node_sp_id[dst];
        g->edge_sp_len[e] = g->node_sp_len[dst];
        if (g->node_sp_len[dst] > 1) {
            const char *sp = lzg_sp_get(g->pool, g->node_sp_id[dst]);
            uint32_t plen = g->node_sp_len[dst] - 1;
            g->edge_prefix_id[e] = lzg_sp_intern_n(g->pool, sp, plen);
        } else {
            g->edge_prefix_id[e] = UINT32_MAX;
        }
    }
    (void)lzg_graph_ensure_query_edge_hashes(g);
}

static void fill_marginal_distribution(const LZGHashMap *counts,
                                       uint32_t *out_n,
                                       uint32_t **out_ids,
                                       double **out_probs) {
    *out_n = counts->count;
    *out_ids = malloc((*out_n) * sizeof(uint32_t));
    *out_probs = malloc((*out_n) * sizeof(double));

    uint64_t total = 0;
    uint32_t j = 0;
    for (uint32_t i = 0; i < counts->capacity; i++) {
        if (counts->keys[i] == LZG_HM_EMPTY || counts->keys[i] == LZG_HM_DELETED)
            continue;
        (*out_ids)[j] = (uint32_t)counts->keys[i];
        (*out_probs)[j] = (double)counts->values[i];
        total += counts->values[i];
        j++;
    }

    for (uint32_t i = 0; i < *out_n; i++)
        (*out_probs)[i] /= (double)(total > 0 ? total : 1);
}

static void fill_joint_distribution(const LZGHashMap *counts,
                                    uint32_t *out_n,
                                    uint32_t **out_v_ids,
                                    uint32_t **out_j_ids,
                                    double **out_probs) {
    *out_n = counts->count;
    *out_v_ids = malloc((*out_n) * sizeof(uint32_t));
    *out_j_ids = malloc((*out_n) * sizeof(uint32_t));
    *out_probs = malloc((*out_n) * sizeof(double));

    uint64_t total = 0;
    uint32_t j = 0;
    for (uint32_t i = 0; i < counts->capacity; i++) {
        if (counts->keys[i] == LZG_HM_EMPTY || counts->keys[i] == LZG_HM_DELETED)
            continue;
        (*out_v_ids)[j] = (uint32_t)(counts->keys[i] >> 32);
        (*out_j_ids)[j] = (uint32_t)(counts->keys[i] & 0xFFFFFFFF);
        (*out_probs)[j] = (double)counts->values[i];
        total += counts->values[i];
        j++;
    }

    for (uint32_t i = 0; i < *out_n; i++)
        (*out_probs)[i] /= (double)(total > 0 ? total : 1);
}

static void build_edge_gene_csr(const LZGHashMap *edge_genes,
                                const LZGEdgeBuilder *eb,
                                const uint32_t *builder_to_csr,
                                uint32_t n_edges,
                                uint32_t **out_offsets,
                                uint32_t **out_gene_ids,
                                uint64_t **out_gene_counts,
                                uint32_t *out_total_entries) {
    *out_offsets = calloc(n_edges + 1, sizeof(uint32_t));

    if (edge_genes && builder_to_csr) {
        for (uint32_t i = 0; i < edge_genes->capacity; i++) {
            if (edge_genes->keys[i] == LZG_HM_EMPTY ||
                edge_genes->keys[i] == LZG_HM_DELETED) continue;
            uint32_t builder_idx = (uint32_t)(edge_genes->keys[i] >> 32);
            if (builder_idx < eb->n_edges) {
                uint32_t csr_idx = builder_to_csr[builder_idx];
                (*out_offsets)[csr_idx + 1]++;
            }
        }
    }

    for (uint32_t e = 0; e < n_edges; e++)
        (*out_offsets)[e + 1] += (*out_offsets)[e];
    *out_total_entries = (*out_offsets)[n_edges];

    *out_gene_ids = malloc((*out_total_entries) * sizeof(uint32_t));
    *out_gene_counts = malloc((*out_total_entries) * sizeof(uint64_t));
    uint32_t *cursor = calloc(n_edges, sizeof(uint32_t));

    if (edge_genes && builder_to_csr) {
        for (uint32_t i = 0; i < edge_genes->capacity; i++) {
            if (edge_genes->keys[i] == LZG_HM_EMPTY ||
                edge_genes->keys[i] == LZG_HM_DELETED) continue;
            uint32_t builder_idx = (uint32_t)(edge_genes->keys[i] >> 32);
            uint32_t gene_id = (uint32_t)(edge_genes->keys[i] & 0xFFFFFFFF);
            if (builder_idx >= eb->n_edges) continue;
            uint32_t csr_idx = builder_to_csr[builder_idx];
            uint32_t pos = (*out_offsets)[csr_idx] + cursor[csr_idx];
            (*out_gene_ids)[pos] = gene_id;
            (*out_gene_counts)[pos] = edge_genes->values[i];
            cursor[csr_idx]++;
        }
    }

    free(cursor);
    *out_total_entries = sort_compact_gene_csr(n_edges, *out_offsets,
                                               *out_gene_ids,
                                               *out_gene_counts);
}

static void lzg_graph_attach_gene_data(LZGGraph *g,
                                       const LZGEdgeBuilder *eb,
                                       const uint32_t *builder_to_csr,
                                       const LZGFinalizeGeneInputs *gene_inputs) {
    if (!gene_inputs || !gene_inputs->gene_pool) return;

    LZGGeneData *gd = lzg_gene_data_create();
    lzg_sp_destroy(gd->gene_pool);
    gd->gene_pool = gene_inputs->gene_pool;

    fill_marginal_distribution(gene_inputs->v_marginal_counts,
                               &gd->n_v_genes,
                               &gd->v_marginal_ids,
                               &gd->v_marginal_probs);
    fill_marginal_distribution(gene_inputs->j_marginal_counts,
                               &gd->n_j_genes,
                               &gd->j_marginal_ids,
                               &gd->j_marginal_probs);
    fill_joint_distribution(gene_inputs->vj_pair_counts,
                            &gd->n_vj_pairs,
                            &gd->vj_v_ids,
                            &gd->vj_j_ids,
                            &gd->vj_probs);

    build_edge_gene_csr(gene_inputs->edge_v_genes, eb, builder_to_csr, g->n_edges,
                        &gd->v_offsets, &gd->v_gene_ids, &gd->v_gene_counts,
                        &gd->total_v_entries);
    build_edge_gene_csr(gene_inputs->edge_j_genes, eb, builder_to_csr, g->n_edges,
                        &gd->j_offsets, &gd->j_gene_ids, &gd->j_gene_counts,
                        &gd->total_j_entries);

    g->gene_data = gd;
}

static LZGError topo_sort_internal(LZGGraph *g) {
    uint32_t n = g->n_nodes;
    uint32_t *in_degree = calloc(n, sizeof(uint32_t));
    if (!in_degree) return LZG_ERR_ALLOC;

    for (uint32_t e = 0; e < g->n_edges; e++)
        in_degree[g->col_indices[e]]++;

    uint32_t *queue = malloc(n * sizeof(uint32_t));
    uint32_t head = 0, tail = 0;
    for (uint32_t i = 0; i < n; i++)
        if (in_degree[i] == 0) queue[tail++] = i;

    g->topo_order = malloc(n * sizeof(uint32_t));
    uint32_t count = 0;

    while (head < tail) {
        uint32_t u = queue[head++];
        g->topo_order[count++] = u;

        uint32_t start = g->row_offsets[u];
        uint32_t end = g->row_offsets[u + 1];
        for (uint32_t e = start; e < end; e++) {
            uint32_t v = g->col_indices[e];
            if (--in_degree[v] == 0)
                queue[tail++] = v;
        }
    }

    free(in_degree);
    free(queue);

    if (count != n) return LZG_ERR_HAS_CYCLES;
    g->topo_valid = true;
    return LZG_OK;
}

void lzg_graph_parse_node_label(const LZGStringPool *pool, uint32_t node_label_id,
                                LZGVariant variant,
                                uint32_t *out_sp_id, uint32_t *out_position,
                                LZGStringPool *sp_pool) {
    const char *label = lzg_sp_get(pool, node_label_id);
    uint32_t label_len = lzg_sp_len(pool, node_label_id);

    if (variant == LZG_VARIANT_AAP) {
        uint32_t sep = label_len;
        while (sep > 0 && label[sep - 1] != '_') sep--;
        uint32_t sp_len = (sep > 0) ? sep - 1 : label_len;
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, sp_len);
        *out_position = 0;
        for (uint32_t i = sep; i < label_len; i++)
            *out_position = *out_position * 10 + (label[i] - '0');
    } else if (variant == LZG_VARIANT_NDP) {
        uint32_t sep = 0;
        while (sep < label_len && label[sep] != '_') sep++;
        uint32_t sp_len = (sep > 1) ? sep - 1 : sep;
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, sp_len);
        *out_position = 0;
        for (uint32_t i = sep + 1; i < label_len; i++)
            *out_position = *out_position * 10 + (label[i] - '0');
    } else {
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, label_len);
        *out_position = UINT32_MAX;
    }
}

void lzg_graph_alloc_csr_storage(LZGGraph *g, uint32_t n_nodes, uint32_t n_edges) {
    g->n_nodes = n_nodes;
    g->n_edges = n_edges;

    g->row_offsets = calloc(n_nodes + 1, sizeof(uint32_t));
    g->col_indices = malloc(n_edges * sizeof(uint32_t));
    g->edge_weights = malloc(n_edges * sizeof(double));
    g->edge_counts = malloc(n_edges * sizeof(uint64_t));
    g->edge_sp_id = malloc(n_edges * sizeof(uint32_t));
    g->edge_sp_len = malloc(n_edges * sizeof(uint8_t));
    g->edge_prefix_id = malloc(n_edges * sizeof(uint32_t));

    g->outgoing_counts = calloc(n_nodes, sizeof(uint64_t));
    g->node_sp_id = malloc(n_nodes * sizeof(uint32_t));
    g->node_sp_len = malloc(n_nodes * sizeof(uint8_t));
    g->node_pos = malloc(n_nodes * sizeof(uint32_t));
}

LZGError lzg_graph_finalize_derived_state(
    LZGGraph *g,
    uint64_t *len_counts, uint32_t max_len,
    const LZGEdgeBuilder *eb,
    const uint32_t *builder_to_csr,
    const LZGFinalizeGeneInputs *gene_inputs) {
    lzg_graph_identify_special_nodes(g);
    if (!g->node_is_sink) return LZG_ERR_ALLOC;

    lzg_graph_normalize_edge_weights(g);
    lzg_graph_populate_edge_lz_metadata(g);

    g->length_counts = len_counts;
    g->max_length = max_len;

    LZGError topo_err = topo_sort_internal(g);

    if (gene_inputs && gene_inputs->gene_pool)
        lzg_graph_attach_gene_data(g, eb, builder_to_csr, gene_inputs);

    return topo_err;
}

LZGError lzg_graph_topo_sort(LZGGraph *g) {
    if (!g) return LZG_ERR_INVALID_ARG;
    if (g->topo_valid) return LZG_OK;
    return topo_sort_internal(g);
}

LZGError lzg_graph_recalculate(LZGGraph *g, uint32_t flags) {
    if (!g) return LZG_ERR_INVALID_ARG;
    lzg_exact_model_invalidate(g);

    if (flags & LZG_RECALC_WEIGHTS) {
        memset(g->outgoing_counts, 0, g->n_nodes * sizeof(uint64_t));
        for (uint32_t u = 0; u < g->n_nodes; u++) {
            uint32_t e_start = g->row_offsets[u];
            uint32_t e_end = g->row_offsets[u + 1];
            for (uint32_t e = e_start; e < e_end; e++)
                g->outgoing_counts[u] += g->edge_counts[e];
        }
    }

    if (flags & LZG_RECALC_WEIGHTS) {
        double alpha = g->smoothing_alpha;
        for (uint32_t u = 0; u < g->n_nodes; u++) {
            uint32_t e_start = g->row_offsets[u];
            uint32_t e_end = g->row_offsets[u + 1];
            uint64_t total = g->outgoing_counts[u];
            uint32_t k = e_end - e_start;

            for (uint32_t e = e_start; e < e_end; e++) {
                if (alpha > 0.0) {
                    g->edge_weights[e] = (g->edge_counts[e] + alpha) /
                                         (total + alpha * k);
                } else {
                    g->edge_weights[e] = total > 0
                        ? (double)g->edge_counts[e] / total : 0.0;
                }
            }
        }
    }

    return LZG_OK;
}
