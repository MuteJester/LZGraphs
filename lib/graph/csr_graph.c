/**
 * @file csr_graph.c
 * @brief Graph construction: sequences → LZ76 → EdgeBuilder → CSR.
 */
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/lz76.h"
#include "lzgraph/edge_builder.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Create / Destroy ──────────────────────────────────────── */

LZGGraph *lzg_graph_create(LZGVariant variant) {
    LZGGraph *g = calloc(1, sizeof(LZGGraph));
    if (!g) return NULL;
    g->variant = variant;
    g->pool = lzg_sp_create(4096);
    g->smoothing_alpha = 0.0;
    g->root_node = UINT32_MAX;
    return g;
}

void lzg_graph_destroy(LZGGraph *g) {
    if (!g) return;
    free(g->row_offsets);    free(g->col_indices);
    free(g->edge_weights);   free(g->edge_counts);
    free(g->edge_sp_id);     free(g->edge_sp_len);
    free(g->edge_prefix_id);
    free(g->outgoing_counts);
    free(g->node_sp_id);     free(g->node_sp_len);
    free(g->node_pos);       free(g->node_is_sink);
    free(g->topo_order);
    free(g->length_counts);
    lzg_sp_destroy(g->pool);
    if (g->gene_data) lzg_gene_data_destroy(g->gene_data);
    free(g);
}

/* ── Internal helpers ──────────────────────────────────────── */

/**
 * Parse a node label into subpattern ID and position, variant-aware.
 *
 * AAP:   "SL_5"    → sp="SL", pos=5
 * NDP:   "ATG0_3"  → sp="ATG", pos=3  (strip frame digit before '_')
 * Naive: "SL"      → sp="SL", pos=UINT32_MAX
 */
static void parse_node_label(const LZGStringPool *pool, uint32_t node_label_id,
                             LZGVariant variant,
                             uint32_t *out_sp_id, uint32_t *out_position,
                             LZGStringPool *sp_pool) {
    const char *label = lzg_sp_get(pool, node_label_id);
    uint32_t label_len = lzg_sp_len(pool, node_label_id);

    if (variant == LZG_VARIANT_AAP) {
        /* Find last '_' — everything before it is the subpattern */
        uint32_t sep = label_len;
        while (sep > 0 && label[sep - 1] != '_') sep--;
        uint32_t sp_len = (sep > 0) ? sep - 1 : label_len;
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, sp_len);
        *out_position = 0;
        for (uint32_t i = sep; i < label_len; i++)
            *out_position = *out_position * 10 + (label[i] - '0');

    } else if (variant == LZG_VARIANT_NDP) {
        /* Find first '_' — subpattern is everything before it minus the
         * last character (the reading frame digit) */
        uint32_t sep = 0;
        while (sep < label_len && label[sep] != '_') sep++;
        uint32_t sp_len = (sep > 1) ? sep - 1 : sep; /* strip frame digit */
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, sp_len);
        *out_position = 0;
        for (uint32_t i = sep + 1; i < label_len; i++)
            *out_position = *out_position * 10 + (label[i] - '0');

    } else {
        /* Naive: label IS the subpattern, no position */
        *out_sp_id = lzg_sp_intern_n(sp_pool, label, label_len);
        *out_position = UINT32_MAX;
    }
}

/* Forward declarations */
static LZGError finalize_from_edges(
    LZGGraph *g, LZGEdgeBuilder *eb,
    LZGHashMap *node_set, LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts, LZGHashMap *outgoing_counts,
    uint32_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts, LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes, LZGHashMap *edge_j_genes);

/* Public wrapper without gene data (used by graph_union) */
LZGError lzg_graph_finalize_from_edges(
    LZGGraph *g, LZGEdgeBuilder *eb,
    LZGHashMap *node_set, LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts, LZGHashMap *outgoing_counts,
    uint32_t *len_counts, uint32_t max_len) {
    return finalize_from_edges(g, eb, node_set, initial_counts,
                                terminal_counts, outgoing_counts,
                                len_counts, max_len,
                                NULL, NULL, NULL, NULL, NULL, NULL);
}

/**
 * Kahn's algorithm for topological sort on the CSR graph.
 */
static LZGError topo_sort_internal(LZGGraph *g) {
    uint32_t n = g->n_nodes;
    uint32_t *in_degree = calloc(n, sizeof(uint32_t));
    if (!in_degree) return LZG_ERR_ALLOC;

    /* Count in-degrees */
    for (uint32_t e = 0; e < g->n_edges; e++)
        in_degree[g->col_indices[e]]++;

    /* Queue of nodes with in-degree 0 */
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
        uint32_t end   = g->row_offsets[u + 1];
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

/* ── Main build function ──────────────────────────────────── */

LZGError lzg_graph_build(LZGGraph *g,
                          const char **sequences,
                          uint32_t n_seqs,
                          const uint32_t *abundances,
                          const char **v_genes,
                          const char **j_genes,
                          double smoothing,
                          uint32_t min_init) {
    if (!g || !sequences || n_seqs == 0) return LZG_ERR_INVALID_ARG;

    g->smoothing_alpha = smoothing;
    (void)min_init; /* deprecated — sentinels make single root */
    bool has_genes = (v_genes != NULL && j_genes != NULL);

    LZG_INFO("building graph: %u sequences, variant=%d, genes=%s",
             n_seqs, (int)g->variant, has_genes ? "yes" : "no");

    LZGEdgeBuilder *eb = lzg_eb_create(n_seqs * 8);
    if (!eb) return LZG_ERR_ALLOC;

    /* Gene string pool and marginal accumulators */
    LZGStringPool *gene_pool = has_genes ? lzg_sp_create(256) : NULL;
    LZGHashMap *v_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    LZGHashMap *j_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    LZGHashMap *vj_pair_counts    = has_genes ? lzg_hm_create(256) : NULL;
    /* Per-edge gene: keyed by pack(edge_idx, gene_id) → count */
    LZGHashMap *edge_v_genes = has_genes ? lzg_hm_create(n_seqs * 4) : NULL;
    LZGHashMap *edge_j_genes = has_genes ? lzg_hm_create(n_seqs * 4) : NULL;

    /* Track initial/terminal states and lengths via hash maps */
    LZGHashMap *initial_counts  = lzg_hm_create(256);
    LZGHashMap *terminal_counts = lzg_hm_create(256);
    LZGHashMap *outgoing_counts = lzg_hm_create(4096);
    LZGHashMap *node_set        = lzg_hm_create(4096);
    uint32_t max_len = 0;

    /* Temporary length count array (grow as needed) */
    uint32_t len_cap = 128;
    uint32_t *len_counts = calloc(len_cap, sizeof(uint32_t));

    /* ── Process each sequence ── */
    for (uint32_t s = 0; s < n_seqs; s++) {
        const char *seq = sequences[s];
        uint32_t seq_len = (uint32_t)strlen(seq);
        uint32_t count = abundances ? abundances[s] : 1;

        /* Encode to node labels */
        uint32_t node_ids[LZG_MAX_TOKENS];
        uint32_t sp_ids[LZG_MAX_TOKENS];
        uint32_t n_tokens;

        LZGError err = lzg_lz76_encode(seq, seq_len, g->pool, g->variant,
                                    node_ids, sp_ids, &n_tokens);
        if (err != LZG_OK || n_tokens == 0) continue;

        /* Register all nodes */
        for (uint32_t i = 0; i < n_tokens; i++)
            lzg_hm_put(node_set, node_ids[i], 1);

        /* Initial state: first node */
        uint64_t *ic = lzg_hm_get(initial_counts, node_ids[0]);
        if (ic) *ic += count; else lzg_hm_put(initial_counts, node_ids[0], count);

        /* Terminal state: last node */
        uint64_t *tc = lzg_hm_get(terminal_counts, node_ids[n_tokens - 1]);
        if (tc) *tc += count; else lzg_hm_put(terminal_counts, node_ids[n_tokens - 1], count);

        /* Gene marginals (before edges, to get gene IDs) */
        uint32_t v_gene_id = 0, j_gene_id = 0;
        if (has_genes) {
            v_gene_id = lzg_sp_intern(gene_pool, v_genes[s]);
            j_gene_id = lzg_sp_intern(gene_pool, j_genes[s]);

            /* Accumulate marginal counts */
            uint64_t *vc = lzg_hm_get(v_marginal_counts, v_gene_id);
            if (vc) *vc += count; else lzg_hm_put(v_marginal_counts, v_gene_id, count);
            uint64_t *jc = lzg_hm_get(j_marginal_counts, j_gene_id);
            if (jc) *jc += count; else lzg_hm_put(j_marginal_counts, j_gene_id, count);

            /* VJ pair: pack(v_gene_id, j_gene_id) */
            uint64_t vj_key = ((uint64_t)v_gene_id << 32) | j_gene_id;
            uint64_t *vjc = lzg_hm_get(vj_pair_counts, vj_key);
            if (vjc) *vjc += count; else lzg_hm_put(vj_pair_counts, vj_key, count);
        }

        /* Edges between consecutive nodes */
        for (uint32_t i = 0; i < n_tokens - 1; i++) {
            lzg_eb_record(eb, node_ids[i], node_ids[i + 1], count);

            /* Track outgoing counts */
            uint64_t *oc = lzg_hm_get(outgoing_counts, node_ids[i]);
            if (oc) *oc += count; else lzg_hm_put(outgoing_counts, node_ids[i], count);

            /* Per-edge gene counts */
            if (has_genes) {
                /* Find the edge index in the builder */
                uint64_t ekey = lzg_eb_pack_key(node_ids[i], node_ids[i + 1]);
                uint64_t *eidx = lzg_hm_get(eb->edge_map, ekey);
                if (eidx) {
                    uint32_t ei = (uint32_t)*eidx;
                    uint64_t vk = ((uint64_t)ei << 32) | v_gene_id;
                    uint64_t jk = ((uint64_t)ei << 32) | j_gene_id;
                    uint64_t *ev = lzg_hm_get(edge_v_genes, vk);
                    if (ev) *ev += count; else lzg_hm_put(edge_v_genes, vk, count);
                    uint64_t *ej = lzg_hm_get(edge_j_genes, jk);
                    if (ej) *ej += count; else lzg_hm_put(edge_j_genes, jk, count);
                }
            }
        }
        /* Ensure last node has an outgoing entry (possibly 0) */
        if (!lzg_hm_get(outgoing_counts, node_ids[n_tokens - 1]))
            lzg_hm_put(outgoing_counts, node_ids[n_tokens - 1], 0);

        /* Length distribution */
        if (seq_len >= len_cap) {
            uint32_t new_cap = seq_len + 64;
            len_counts = realloc(len_counts, new_cap * sizeof(uint32_t));
            memset(len_counts + len_cap, 0, (new_cap - len_cap) * sizeof(uint32_t));
            len_cap = new_cap;
        }
        len_counts[seq_len] += count;
        if (seq_len > max_len) max_len = seq_len;
    }

    /* Delegate to shared finalization pipeline */
    LZGError final_err = finalize_from_edges(
        g, eb, node_set, initial_counts, terminal_counts,
        outgoing_counts, len_counts, max_len,
        has_genes ? gene_pool : NULL,
        v_marginal_counts, j_marginal_counts,
        vj_pair_counts, edge_v_genes, edge_j_genes);
    return final_err;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shared finalization: EdgeBuilder → CSR + normalization + topo   */
/* ═══════════════════════════════════════════════════════════════ */

static LZGError finalize_from_edges(
    LZGGraph *g,
    LZGEdgeBuilder *eb,
    LZGHashMap *node_set,
    LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts,
    LZGHashMap *outgoing_counts,
    uint32_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts,
    LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes,
    LZGHashMap *edge_j_genes)
{
    double smoothing = g->smoothing_alpha;
    uint32_t min_init = 0; /* deprecated — sentinels use single root */
    bool has_genes = (gene_pool != NULL);

    /* ── Build node ID mapping ── */
    uint32_t n_nodes = node_set->count;
    g->n_nodes = n_nodes;
    g->n_edges = eb->n_edges;

    /* Collect all unique node IDs and assign sequential indices */
    uint32_t *label_ids = malloc(n_nodes * sizeof(uint32_t));
    LZGHashMap *label_to_idx = lzg_hm_create(n_nodes * 2);
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < node_set->capacity; i++) {
            if (node_set->keys[i] != LZG_HM_EMPTY &&
                node_set->keys[i] != LZG_HM_DELETED) {
                uint32_t label_id = (uint32_t)node_set->keys[i];
                label_ids[idx] = label_id;
                lzg_hm_put(label_to_idx, label_id, idx);
                idx++;
            }
        }
    }

    /* ── Pack into CSR ── */
    g->row_offsets     = calloc(n_nodes + 1, sizeof(uint32_t));
    g->col_indices     = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_weights    = malloc(eb->n_edges * sizeof(double));
    g->edge_counts     = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_sp_id      = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_sp_len     = malloc(eb->n_edges * sizeof(uint8_t));
    g->edge_prefix_id  = malloc(eb->n_edges * sizeof(uint32_t));

    g->outgoing_counts = calloc(n_nodes, sizeof(uint32_t));
    g->node_sp_id      = malloc(n_nodes * sizeof(uint32_t));
    g->node_sp_len     = malloc(n_nodes * sizeof(uint8_t));
    g->node_pos        = malloc(n_nodes * sizeof(uint32_t));

    /* Count edges per source node */
    uint32_t *edge_deg = calloc(n_nodes, sizeof(uint32_t));
    for (uint32_t e = 0; e < eb->n_edges; e++) {
        uint32_t src_idx = (uint32_t)*lzg_hm_get(label_to_idx, eb->src_ids[e]);
        edge_deg[src_idx]++;
    }

    /* Build row_offsets (prefix sum) */
    g->row_offsets[0] = 0;
    for (uint32_t i = 0; i < n_nodes; i++)
        g->row_offsets[i + 1] = g->row_offsets[i] + edge_deg[i];

    /* Fill edges (use edge_deg as write cursor).
     * Also build builder_to_csr mapping for gene data unpacking. */
    uint32_t *builder_to_csr = has_genes ? malloc(eb->n_edges * sizeof(uint32_t)) : NULL;
    memset(edge_deg, 0, n_nodes * sizeof(uint32_t));
    for (uint32_t e = 0; e < eb->n_edges; e++) {
        uint32_t src_idx = (uint32_t)*lzg_hm_get(label_to_idx, eb->src_ids[e]);
        uint32_t dst_idx = (uint32_t)*lzg_hm_get(label_to_idx, eb->dst_ids[e]);
        uint32_t pos = g->row_offsets[src_idx] + edge_deg[src_idx];

        g->col_indices[pos] = dst_idx;
        g->edge_counts[pos] = eb->counts[e];
        if (builder_to_csr) builder_to_csr[e] = pos;
        edge_deg[src_idx]++;
    }
    free(edge_deg);

    /* ── Parse node metadata and compute derived quantities ── */
    LZGStringPool *sp_pool = g->pool; /* reuse the same pool for subpatterns */

    for (uint32_t i = 0; i < n_nodes; i++) {
        parse_node_label(g->pool, label_ids[i], g->variant,
                         &g->node_sp_id[i], &g->node_pos[i], sp_pool);
        g->node_sp_len[i] = (uint8_t)lzg_sp_len(sp_pool, g->node_sp_id[i]);

        /* Outgoing count */
        uint64_t *oc = lzg_hm_get(outgoing_counts, label_ids[i]);
        g->outgoing_counts[i] = oc ? (uint32_t)*oc : 0;

    }

    /* ── Identify root (@) and sink ($) nodes ── */
    g->node_is_sink = calloc(n_nodes, sizeof(uint8_t));
    g->root_node = UINT32_MAX;
    for (uint32_t i = 0; i < n_nodes; i++) {
        const char *sp = lzg_sp_get(sp_pool, g->node_sp_id[i]);
        uint8_t sp_len = g->node_sp_len[i];
        if (sp_len == 1 && sp[0] == LZG_START_SENTINEL) {
            g->root_node = i;
        }
        if (sp_len > 0 && sp[sp_len - 1] == LZG_END_SENTINEL) {
            g->node_is_sink[i] = 1;
        }
    }
    if (g->root_node == UINT32_MAX) {
        LZG_WARN("no @ root node found — graph may not have sentinel encoding");
    }

    /* ── Normalize edge weights ── */
    for (uint32_t i = 0; i < n_nodes; i++) {
        uint32_t start = g->row_offsets[i];
        uint32_t end   = g->row_offsets[i + 1];
        uint32_t total = g->outgoing_counts[i];

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

    /* ── Precompute per-edge LZ constraint info ── */
    for (uint32_t e = 0; e < g->n_edges; e++) {
        uint32_t dst = g->col_indices[e];
        g->edge_sp_id[e]  = g->node_sp_id[dst];
        g->edge_sp_len[e] = g->node_sp_len[dst];

        if (g->node_sp_len[dst] > 1) {
            /* Prefix = subpattern[:-1] */
            const char *sp = lzg_sp_get(sp_pool, g->node_sp_id[dst]);
            uint32_t plen = g->node_sp_len[dst] - 1;
            g->edge_prefix_id[e] = lzg_sp_intern_n(sp_pool, sp, plen);
        } else {
            g->edge_prefix_id[e] = UINT32_MAX; /* no prefix for single char */
        }
    }

    /* ── Length distribution ── */
    g->length_counts = len_counts;
    g->max_length = max_len;

    /* ── Topological sort ── */
    LZGError topo_err = topo_sort_internal(g);

    /* ── Gene data finalization ── */
    if (has_genes) {
        LZGGeneData *gd = lzg_gene_data_create();
        lzg_sp_destroy(gd->gene_pool);
        gd->gene_pool = gene_pool; /* transfer ownership */

        /* Build marginal V gene distribution */
        gd->n_v_genes = v_marginal_counts->count;
        gd->v_marginal_ids   = malloc(gd->n_v_genes * sizeof(uint32_t));
        gd->v_marginal_probs = malloc(gd->n_v_genes * sizeof(double));
        {
            uint64_t total = 0;
            uint32_t j = 0;
            for (uint32_t i = 0; i < v_marginal_counts->capacity; i++) {
                if (v_marginal_counts->keys[i] != LZG_HM_EMPTY &&
                    v_marginal_counts->keys[i] != LZG_HM_DELETED) {
                    gd->v_marginal_ids[j] = (uint32_t)v_marginal_counts->keys[i];
                    gd->v_marginal_probs[j] = (double)v_marginal_counts->values[i];
                    total += v_marginal_counts->values[i];
                    j++;
                }
            }
            for (uint32_t i = 0; i < gd->n_v_genes; i++)
                gd->v_marginal_probs[i] /= (double)(total > 0 ? total : 1);
        }

        /* Build marginal J gene distribution */
        gd->n_j_genes = j_marginal_counts->count;
        gd->j_marginal_ids   = malloc(gd->n_j_genes * sizeof(uint32_t));
        gd->j_marginal_probs = malloc(gd->n_j_genes * sizeof(double));
        {
            uint64_t total = 0;
            uint32_t j = 0;
            for (uint32_t i = 0; i < j_marginal_counts->capacity; i++) {
                if (j_marginal_counts->keys[i] != LZG_HM_EMPTY &&
                    j_marginal_counts->keys[i] != LZG_HM_DELETED) {
                    gd->j_marginal_ids[j] = (uint32_t)j_marginal_counts->keys[i];
                    gd->j_marginal_probs[j] = (double)j_marginal_counts->values[i];
                    total += j_marginal_counts->values[i];
                    j++;
                }
            }
            for (uint32_t i = 0; i < gd->n_j_genes; i++)
                gd->j_marginal_probs[i] /= (double)(total > 0 ? total : 1);
        }

        /* Build VJ joint distribution */
        gd->n_vj_pairs = vj_pair_counts->count;
        gd->vj_v_ids = malloc(gd->n_vj_pairs * sizeof(uint32_t));
        gd->vj_j_ids = malloc(gd->n_vj_pairs * sizeof(uint32_t));
        gd->vj_probs = malloc(gd->n_vj_pairs * sizeof(double));
        {
            uint64_t total = 0;
            uint32_t j = 0;
            for (uint32_t i = 0; i < vj_pair_counts->capacity; i++) {
                if (vj_pair_counts->keys[i] != LZG_HM_EMPTY &&
                    vj_pair_counts->keys[i] != LZG_HM_DELETED) {
                    gd->vj_v_ids[j] = (uint32_t)(vj_pair_counts->keys[i] >> 32);
                    gd->vj_j_ids[j] = (uint32_t)(vj_pair_counts->keys[i] & 0xFFFFFFFF);
                    gd->vj_probs[j] = (double)vj_pair_counts->values[i];
                    total += vj_pair_counts->values[i];
                    j++;
                }
            }
            for (uint32_t i = 0; i < gd->n_vj_pairs; i++)
                gd->vj_probs[i] /= (double)(total > 0 ? total : 1);
        }

        /* Build per-edge V/J gene CSR-within-CSR.
         * Unpack (builder_edge_idx, gene_id) → count entries from the
         * hash maps into sorted per-CSR-edge gene arrays. */
        {
            uint32_t ne = g->n_edges;
            gd->v_offsets = calloc(ne + 1, sizeof(uint32_t));
            gd->j_offsets = calloc(ne + 1, sizeof(uint32_t));

            /* Count entries per CSR edge for V genes */
            if (edge_v_genes && builder_to_csr) {
                for (uint32_t i = 0; i < edge_v_genes->capacity; i++) {
                    if (edge_v_genes->keys[i] == LZG_HM_EMPTY ||
                        edge_v_genes->keys[i] == LZG_HM_DELETED) continue;
                    uint32_t builder_idx = (uint32_t)(edge_v_genes->keys[i] >> 32);
                    if (builder_idx < eb->n_edges) {
                        uint32_t csr_idx = builder_to_csr[builder_idx];
                        gd->v_offsets[csr_idx + 1]++;
                    }
                }
            }
            /* Prefix sum for V offsets */
            for (uint32_t e = 0; e < ne; e++)
                gd->v_offsets[e + 1] += gd->v_offsets[e];
            gd->total_v_entries = gd->v_offsets[ne];

            /* Allocate and fill V gene arrays */
            gd->v_gene_ids    = malloc(gd->total_v_entries * sizeof(uint32_t));
            gd->v_gene_counts = malloc(gd->total_v_entries * sizeof(uint32_t));
            uint32_t *v_cursor = calloc(ne, sizeof(uint32_t)); /* write cursor per edge */

            if (edge_v_genes && builder_to_csr) {
                for (uint32_t i = 0; i < edge_v_genes->capacity; i++) {
                    if (edge_v_genes->keys[i] == LZG_HM_EMPTY ||
                        edge_v_genes->keys[i] == LZG_HM_DELETED) continue;
                    uint32_t builder_idx = (uint32_t)(edge_v_genes->keys[i] >> 32);
                    uint32_t gene_id = (uint32_t)(edge_v_genes->keys[i] & 0xFFFFFFFF);
                    if (builder_idx >= eb->n_edges) continue;
                    uint32_t csr_idx = builder_to_csr[builder_idx];
                    uint32_t pos = gd->v_offsets[csr_idx] + v_cursor[csr_idx];
                    gd->v_gene_ids[pos]    = gene_id;
                    gd->v_gene_counts[pos] = (uint32_t)edge_v_genes->values[i];
                    v_cursor[csr_idx]++;
                }
            }
            free(v_cursor);

            /* Same for J genes */
            if (edge_j_genes && builder_to_csr) {
                for (uint32_t i = 0; i < edge_j_genes->capacity; i++) {
                    if (edge_j_genes->keys[i] == LZG_HM_EMPTY ||
                        edge_j_genes->keys[i] == LZG_HM_DELETED) continue;
                    uint32_t builder_idx = (uint32_t)(edge_j_genes->keys[i] >> 32);
                    if (builder_idx < eb->n_edges) {
                        uint32_t csr_idx = builder_to_csr[builder_idx];
                        gd->j_offsets[csr_idx + 1]++;
                    }
                }
            }
            for (uint32_t e = 0; e < ne; e++)
                gd->j_offsets[e + 1] += gd->j_offsets[e];
            gd->total_j_entries = gd->j_offsets[ne];

            gd->j_gene_ids    = malloc(gd->total_j_entries * sizeof(uint32_t));
            gd->j_gene_counts = malloc(gd->total_j_entries * sizeof(uint32_t));
            uint32_t *j_cursor = calloc(ne, sizeof(uint32_t));

            if (edge_j_genes && builder_to_csr) {
                for (uint32_t i = 0; i < edge_j_genes->capacity; i++) {
                    if (edge_j_genes->keys[i] == LZG_HM_EMPTY ||
                        edge_j_genes->keys[i] == LZG_HM_DELETED) continue;
                    uint32_t builder_idx = (uint32_t)(edge_j_genes->keys[i] >> 32);
                    uint32_t gene_id = (uint32_t)(edge_j_genes->keys[i] & 0xFFFFFFFF);
                    if (builder_idx >= eb->n_edges) continue;
                    uint32_t csr_idx = builder_to_csr[builder_idx];
                    uint32_t pos = gd->j_offsets[csr_idx] + j_cursor[csr_idx];
                    gd->j_gene_ids[pos]    = gene_id;
                    gd->j_gene_counts[pos] = (uint32_t)edge_j_genes->values[i];
                    j_cursor[csr_idx]++;
                }
            }
            free(j_cursor);
        }

        g->gene_data = gd;
        lzg_hm_destroy(v_marginal_counts);
        lzg_hm_destroy(j_marginal_counts);
        lzg_hm_destroy(vj_pair_counts);
        lzg_hm_destroy(edge_v_genes);
        lzg_hm_destroy(edge_j_genes);
    }

    /* ── Cleanup temporaries ── */
    free(builder_to_csr);
    lzg_eb_destroy(eb);
    lzg_hm_destroy(initial_counts);
    lzg_hm_destroy(terminal_counts);
    lzg_hm_destroy(outgoing_counts);
    lzg_hm_destroy(node_set);
    lzg_hm_destroy(label_to_idx);
    free(label_ids);

    if (topo_err == LZG_ERR_HAS_CYCLES) {
        g->topo_valid = false;
        LZG_INFO("graph ready: %u nodes, %u edges (has cycles)",
                 g->n_nodes, g->n_edges);
        return LZG_OK;
    }
    if (topo_err != LZG_OK) return topo_err;

    LZG_INFO("graph ready: %u nodes, %u edges, root=%u",
             g->n_nodes, g->n_edges, g->root_node);
    return LZG_OK;
}

LZGError lzg_graph_topo_sort(LZGGraph *g) {
    if (!g) return LZG_ERR_INVALID_ARG;
    if (g->topo_valid) return LZG_OK;
    return topo_sort_internal(g);
}

LZGError lzg_graph_recalculate(LZGGraph *g, uint32_t flags) {
    if (!g) return LZG_ERR_INVALID_ARG;

    uint32_t nn = g->n_nodes, ne = g->n_edges;

    /* ── Recompute outgoing_counts from edge_counts ── */
    if (flags & LZG_RECALC_WEIGHTS) {
        memset(g->outgoing_counts, 0, nn * sizeof(uint32_t));
        for (uint32_t u = 0; u < nn; u++) {
            uint32_t e_start = g->row_offsets[u];
            uint32_t e_end   = g->row_offsets[u + 1];
            for (uint32_t e = e_start; e < e_end; e++)
                g->outgoing_counts[u] += g->edge_counts[e];
        }
    }

    /* ── Recompute edge_weights ── */
    if (flags & LZG_RECALC_WEIGHTS) {
        double alpha = g->smoothing_alpha;
        for (uint32_t u = 0; u < nn; u++) {
            uint32_t e_start = g->row_offsets[u];
            uint32_t e_end   = g->row_offsets[u + 1];
            uint32_t total   = g->outgoing_counts[u];
            uint32_t k       = e_end - e_start;

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

    /* stop_probs and initial_probs removed — sentinel model */

    return LZG_OK;
}
