#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif
/**
 * @file csr_graph.c
 * @brief Graph construction: sequences → LZ76 → EdgeBuilder → CSR.
 */
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"
#include "graph_finalize.h"
#include "graph_build_ingest.h"
#include "../simulation/exact_model.h"
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LZG_BUILD_INIT_CAP_MAX (1u << 20)

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
    lzg_exact_model_invalidate(g);
    free(g->row_offsets);    free(g->col_indices);
    free(g->edge_weights);   free(g->edge_counts);
    free(g->edge_sp_id);     free(g->edge_sp_len);
    free(g->edge_prefix_id);
    free(g->edge_sp_hash);   free(g->edge_prefix_hash);
    free(g->node_sp_hash);
    free(g->edge_single_char_idx);
    free(g->node_single_char_idx);
    free(g->outgoing_counts);
    free(g->node_sp_id);     free(g->node_sp_len);
    free(g->node_pos);       free(g->node_is_sink);
    free(g->topo_order);
    free(g->length_counts);
    lzg_hm_destroy(g->query_node_map);
    lzg_sp_destroy(g->pool);
    if (g->gene_data) lzg_gene_data_destroy(g->gene_data);
    free(g);
}

LZGError lzg_graph_ensure_query_edge_hashes(LZGGraph *g) {
    if (!g) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph pointer is NULL");
    if (g->edge_sp_hash && g->edge_prefix_hash &&
        g->node_sp_hash && g->edge_single_char_idx &&
        g->node_single_char_idx) return LZG_OK;

    uint64_t *sp_hash = calloc(g->n_edges ? g->n_edges : 1, sizeof(uint64_t));
    uint64_t *prefix_hash = calloc(g->n_edges ? g->n_edges : 1, sizeof(uint64_t));
    uint64_t *node_hash = calloc(g->n_nodes ? g->n_nodes : 1, sizeof(uint64_t));
    uint8_t *edge_single = malloc(g->n_edges ? g->n_edges : 1);
    uint8_t *node_single = malloc(g->n_nodes ? g->n_nodes : 1);
    if (!sp_hash || !prefix_hash || !node_hash || !edge_single || !node_single) {
        free(sp_hash);
        free(prefix_hash);
        free(node_hash);
        free(edge_single);
        free(node_single);
        return LZG_FAIL(LZG_ERR_ALLOC, "failed to allocate query edge hash cache");
    }

    memset(edge_single, 0xFF, g->n_edges ? g->n_edges : 1);
    memset(node_single, 0xFF, g->n_nodes ? g->n_nodes : 1);

    for (uint32_t e = 0; e < g->n_edges; e++) {
        const char *sp = lzg_sp_get(g->pool, g->edge_sp_id[e]);
        uint32_t sp_len = g->edge_sp_len[e];
        sp_hash[e] = lzg_hash_bytes(sp, sp_len);
        if (sp_len == 1)
            edge_single[e] = lzg_aa_to_bit(sp[0]);
        if (sp_len > 1 && g->edge_prefix_id[e] != UINT32_MAX) {
            const char *prefix = lzg_sp_get(g->pool, g->edge_prefix_id[e]);
            prefix_hash[e] = lzg_hash_bytes(prefix, (uint32_t)(sp_len - 1));
        }
    }

    for (uint32_t n = 0; n < g->n_nodes; n++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[n]);
        uint32_t sp_len = g->node_sp_len[n];
        node_hash[n] = lzg_hash_bytes(sp, sp_len);
        if (sp_len == 1)
            node_single[n] = lzg_aa_to_bit(sp[0]);
    }

    free(g->edge_sp_hash);
    free(g->edge_prefix_hash);
    free(g->node_sp_hash);
    free(g->edge_single_char_idx);
    free(g->node_single_char_idx);
    g->edge_sp_hash = sp_hash;
    g->edge_prefix_hash = prefix_hash;
    g->node_sp_hash = node_hash;
    g->edge_single_char_idx = edge_single;
    g->node_single_char_idx = node_single;
    return LZG_OK;
}

/* ── Internal helpers ──────────────────────────────────────── */

static uint32_t bounded_capacity_hint(uint32_t n_items,
                                      uint32_t multiplier,
                                      uint32_t min_cap,
                                      uint32_t max_cap) {
    uint64_t estimate = (uint64_t)n_items * (uint64_t)multiplier;
    if (estimate < (uint64_t)min_cap) return min_cap;
    if (estimate > (uint64_t)max_cap) return max_cap;
    return (uint32_t)estimate;
}

static void cleanup_structural_finalize_inputs(
    LZGEdgeBuilder *eb,
    LZGNodeBuilder *build_nodes,
    uint64_t *len_counts,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts,
    LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes,
    LZGHashMap *edge_j_genes,
    bool destroy_gene_pool) {
    LZGBuildResources res = {
        .edge_builder = eb,
        .build_nodes = build_nodes,
        .gene_pool = destroy_gene_pool ? gene_pool : NULL,
        .v_marginal_counts = v_marginal_counts,
        .j_marginal_counts = j_marginal_counts,
        .vj_pair_counts = vj_pair_counts,
        .edge_v_genes = edge_v_genes,
        .edge_j_genes = edge_j_genes,
        .len_counts = len_counts,
    };
    lzg_build_resources_destroy(&res);
}

/* Forward declarations */
static LZGError finalize_from_edges(
    LZGGraph *g, LZGEdgeBuilder *eb,
    LZGHashMap *node_set, LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts, LZGHashMap *outgoing_counts,
    uint64_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts, LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes, LZGHashMap *edge_j_genes);

static LZGError finalize_from_structural_edges(
    LZGGraph *g, LZGEdgeBuilder *eb,
    LZGNodeBuilder *build_nodes,
    uint64_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts, LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes, LZGHashMap *edge_j_genes);

/* Public wrapper without gene data (used by graph_union) */
LZGError lzg_graph_finalize_from_edges(
    LZGGraph *g, LZGEdgeBuilder *eb,
    LZGHashMap *node_set, LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts, LZGHashMap *outgoing_counts,
    uint64_t *len_counts, uint32_t max_len) {
    return finalize_from_edges(g, eb, node_set, initial_counts,
                                terminal_counts, outgoing_counts,
                                len_counts, max_len,
                                NULL, NULL, NULL, NULL, NULL, NULL);
}

/* ── Main build function ──────────────────────────────────── */

LZGError lzg_graph_build(LZGGraph *g,
                          const char **sequences,
                          uint32_t n_seqs,
                          const uint64_t *abundances,
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

    uint32_t eb_cap = bounded_capacity_hint(n_seqs, 8u, 256u, LZG_BUILD_INIT_CAP_MAX);
    LZGBuildResources res = {0};
    res.edge_builder = lzg_eb_create(eb_cap);
    if (!res.edge_builder) return LZG_ERR_ALLOC;

    res.gene_pool = has_genes ? lzg_sp_create(256) : NULL;
    res.v_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    res.j_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    res.vj_pair_counts = has_genes ? lzg_hm_create(256) : NULL;
    uint32_t edge_gene_cap = bounded_capacity_hint(n_seqs, 4u, 256u, LZG_BUILD_INIT_CAP_MAX);
    res.edge_v_genes = has_genes ? lzg_hm_create(edge_gene_cap) : NULL;
    res.edge_j_genes = has_genes ? lzg_hm_create(edge_gene_cap) : NULL;
    res.build_nodes = lzg_node_builder_create(4096);
    uint32_t max_len = 0;

    res.len_cap = 128;
    res.len_counts = calloc(res.len_cap, sizeof(uint64_t));
    if (!res.build_nodes || !res.len_counts) {
        lzg_build_resources_destroy(&res);
        return LZG_ERR_ALLOC;
    }

    /* ── Process each sequence ── */
    for (uint32_t s = 0; s < n_seqs; s++) {
        uint64_t count = abundances ? abundances[s] : 1;
        LZGError err = lzg_accumulate_sequence_record(
            g, &res, sequences[s], count,
            has_genes ? v_genes[s] : NULL, has_genes ? j_genes[s] : NULL, &max_len);
        if (err != LZG_OK) {
            lzg_build_resources_destroy(&res);
            return err;
        }
    }

    /* Delegate to structural finalization pipeline */
    LZGError final_err = finalize_from_structural_edges(
        g, res.edge_builder, res.build_nodes, res.len_counts, max_len,
        has_genes ? res.gene_pool : NULL,
        res.v_marginal_counts, res.j_marginal_counts,
        res.vj_pair_counts, res.edge_v_genes, res.edge_j_genes);
    return final_err;
}

LZGError lzg_graph_build_plain_file(LZGGraph *g,
                                     const char *path,
                                     double smoothing) {
    if (!g || !path || path[0] == '\0') return LZG_ERR_INVALID_ARG;

    FILE *fh = fopen(path, "r");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN, "could not open input file '%s'", path);

    g->smoothing_alpha = smoothing;
    LZGStreamBuildStats stats = {0};
    stats.file_size_bytes = lzg_detect_regular_file_size(path);
    stats.start_time = lzg_build_monotonic_seconds();
    stats.last_log_time = stats.start_time;
    stats.peak_rss_kb = lzg_build_current_rss_kb();
    LZG_INFO("stream build: start phase=ingest file=%s variant=%s size=%.1fMB progress_every_lines=%llu progress_every_sec=%.1f",
             path,
             lzg_build_variant_name(g->variant),
             (double)stats.file_size_bytes / (1024.0 * 1024.0),
             (unsigned long long)LZG_STREAM_PROGRESS_EVERY,
             LZG_STREAM_PROGRESS_MIN_SEC);

    LZGBuildResources res = {0};
    res.edge_builder = lzg_eb_create(256);
    if (!res.edge_builder) { fclose(fh); return LZG_ERR_ALLOC; }

    res.build_nodes = lzg_node_builder_create(4096);
    uint32_t max_len = 0;
    char *line = NULL;
    res.len_cap = 128;
    res.len_counts = calloc(res.len_cap, sizeof(uint64_t));
    if (!res.build_nodes || !res.len_counts) {
        free(line);
        fclose(fh);
        lzg_build_resources_destroy(&res);
        return LZG_ERR_ALLOC;
    }

    size_t line_cap = 0;
    ptrdiff_t nread;
    uint64_t lines_seen = 0;
    uint64_t sequences_seen = 0;
    errno = 0;
    while ((nread = lzg_getline_portable(&line, &line_cap, fh)) != -1) {
        lines_seen++;
        stats.bytes_seen += (uint64_t)nread;
        char *seq = NULL;
        uint64_t count = 0;
        LZGParsedLineKind line_kind = LZG_LINE_EMPTY;
        LZGError err = lzg_parse_plain_sequence_line(line, &seq, &count, &line_kind);
        if (err != LZG_OK) {
            free(line);
            fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_update_stream_mode(&stats, line_kind, path, lines_seen);
        if (!seq || count == 0) continue;
        sequences_seen++;

        err = lzg_accumulate_sequence_record(g, &res, seq, count, NULL, NULL, &max_len);
        if (err != LZG_OK) {
            free(line);
            fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_maybe_log_stream_progress(path, lines_seen, sequences_seen, &res, &stats);
    }

    if (ferror(fh) || (!feof(fh) && errno != 0)) {
        int saved_errno = errno;
        free(line);
        fclose(fh);
        lzg_build_resources_destroy(&res);
        if (saved_errno == ENOMEM)
            return LZG_FAIL(LZG_ERR_ALLOC, "stream build ran out of memory reading '%s'", path);
        return LZG_FAIL(LZG_ERR_IO_READ, "failed while reading '%s'", path);
    }

    free(line);
    fclose(fh);

    {
        double end_time = lzg_build_monotonic_seconds();
        double elapsed = (stats.start_time > 0.0 && end_time > stats.start_time) ? (end_time - stats.start_time) : 0.0;
        double rate = elapsed > 0.0 ? (double)lines_seen / elapsed : 0.0;
        double mbps = elapsed > 0.0 ? ((double)stats.bytes_seen / (1024.0 * 1024.0)) / elapsed : 0.0;
        long long rss_kb = lzg_build_current_rss_kb();
        if (rss_kb > stats.peak_rss_kb) stats.peak_rss_kb = rss_kb;
        char elapsed_buf[32];
        lzg_build_format_duration(elapsed, elapsed_buf, sizeof(elapsed_buf));
        if (rss_kb >= 0) {
            LZG_INFO("stream build: done phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu plain_records=%llu seqcount_records=%llu bytes=%.1fMB nodes=%u edges=%u rss=%.1fMB peak_rss=%.1fMB elapsed=%s rate=%.0f lines/s avg_mbps=%.1f",
                     path,
                     lzg_build_stream_mode_name(stats.mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats.blank_lines,
                     (unsigned long long)stats.plain_records,
                     (unsigned long long)stats.seqcount_records,
                     (double)stats.bytes_seen / (1024.0 * 1024.0),
                     res.build_nodes ? res.build_nodes->count : 0u,
                     res.edge_builder->n_edges,
                     (double)rss_kb / 1024.0,
                     stats.peak_rss_kb >= 0 ? (double)stats.peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0,
                     elapsed_buf,
                     rate,
                     mbps);
        } else {
            LZG_INFO("stream build: done phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu plain_records=%llu seqcount_records=%llu bytes=%.1fMB nodes=%u edges=%u elapsed=%s rate=%.0f lines/s avg_mbps=%.1f",
                     path,
                     lzg_build_stream_mode_name(stats.mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats.blank_lines,
                     (unsigned long long)stats.plain_records,
                     (unsigned long long)stats.seqcount_records,
                     (double)stats.bytes_seen / (1024.0 * 1024.0),
                     res.build_nodes ? res.build_nodes->count : 0u,
                     res.edge_builder->n_edges,
                     elapsed_buf,
                     rate,
                     mbps);
        }
    }

    {
        double finalize_start = lzg_build_monotonic_seconds();
        long long rss_kb = lzg_build_current_rss_kb();
        if (rss_kb > stats.peak_rss_kb) stats.peak_rss_kb = rss_kb;
        if (rss_kb >= 0) {
            LZG_INFO("stream build: start phase=finalize file=%s nodes=%u edges=%u rss=%.1fMB peak_rss=%.1fMB",
                     path,
                     res.build_nodes ? res.build_nodes->count : 0u,
                     res.edge_builder->n_edges,
                     (double)rss_kb / 1024.0,
                     stats.peak_rss_kb >= 0 ? (double)stats.peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0);
        } else {
            LZG_INFO("stream build: start phase=finalize file=%s nodes=%u edges=%u",
                     path,
                     res.build_nodes ? res.build_nodes->count : 0u,
                     res.edge_builder->n_edges);
        }
        LZGError final_err = finalize_from_structural_edges(
        g, res.edge_builder, res.build_nodes, res.len_counts, max_len,
        NULL, NULL, NULL, NULL, NULL, NULL);
        if (final_err == LZG_OK) {
            double finalize_end = lzg_build_monotonic_seconds();
            double finalize_elapsed = (finalize_end > finalize_start) ? (finalize_end - finalize_start) : 0.0;
            char finalize_buf[32];
            lzg_build_format_duration(finalize_elapsed, finalize_buf, sizeof(finalize_buf));
            rss_kb = lzg_build_current_rss_kb();
            if (rss_kb > stats.peak_rss_kb) stats.peak_rss_kb = rss_kb;
            if (rss_kb >= 0) {
                LZG_INFO("stream build: done phase=finalize file=%s nodes=%u edges=%u rss=%.1fMB peak_rss=%.1fMB elapsed=%s",
                         path,
                         g->n_nodes,
                         g->n_edges,
                         (double)rss_kb / 1024.0,
                         stats.peak_rss_kb >= 0 ? (double)stats.peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0,
                         finalize_buf);
            } else {
                LZG_INFO("stream build: done phase=finalize file=%s nodes=%u edges=%u elapsed=%s",
                         path,
                         g->n_nodes,
                         g->n_edges,
                         finalize_buf);
            }
        }
        return final_err;
    }
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shared finalization: EdgeBuilder → CSR + normalization + topo   */
/* ═══════════════════════════════════════════════════════════════ */

static LZGError finalize_from_structural_edges(
    LZGGraph *g,
    LZGEdgeBuilder *eb,
    LZGNodeBuilder *build_nodes,
    uint64_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts,
    LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes,
    LZGHashMap *edge_j_genes)
{
    bool has_genes = (gene_pool != NULL);
    uint32_t n_nodes = build_nodes ? build_nodes->count : 0;

    lzg_graph_alloc_csr_storage(g, n_nodes, eb->n_edges);

    if (!g->row_offsets || !g->col_indices || !g->edge_weights || !g->edge_counts ||
        !g->edge_sp_id || !g->edge_sp_len || !g->edge_prefix_id ||
        !g->outgoing_counts || !g->node_sp_id || !g->node_sp_len || !g->node_pos) {
        cleanup_structural_finalize_inputs(
            eb, build_nodes, len_counts, gene_pool,
            v_marginal_counts, j_marginal_counts, vj_pair_counts,
            edge_v_genes, edge_j_genes, true);
        return LZG_ERR_ALLOC;
    }

    uint32_t *edge_deg = calloc(n_nodes, sizeof(uint32_t));
    if (!edge_deg) {
        cleanup_structural_finalize_inputs(
            eb, build_nodes, len_counts, gene_pool,
            v_marginal_counts, j_marginal_counts, vj_pair_counts,
            edge_v_genes, edge_j_genes, true);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t e = 0; e < eb->n_edges; e++)
        edge_deg[eb->src_ids[e]]++;

    g->row_offsets[0] = 0;
    for (uint32_t i = 0; i < n_nodes; i++)
        g->row_offsets[i + 1] = g->row_offsets[i] + edge_deg[i];

    uint32_t *builder_to_csr = has_genes ? malloc(eb->n_edges * sizeof(uint32_t)) : NULL;
    memset(edge_deg, 0, n_nodes * sizeof(uint32_t));
    for (uint32_t e = 0; e < eb->n_edges; e++) {
        uint32_t src_idx = eb->src_ids[e];
        uint32_t dst_idx = eb->dst_ids[e];
        uint32_t pos = g->row_offsets[src_idx] + edge_deg[src_idx];

        g->col_indices[pos] = dst_idx;
        g->edge_counts[pos] = eb->counts[e];
        g->outgoing_counts[src_idx] += eb->counts[e];
        if (builder_to_csr) builder_to_csr[e] = pos;
        edge_deg[src_idx]++;
    }
    free(edge_deg);

    LZGStringPool *sp_pool = g->pool;
    for (uint32_t i = 0; i < n_nodes; i++) {
        g->node_sp_id[i] = build_nodes->sp_ids[i];
        g->node_pos[i] = build_nodes->positions[i];
        g->node_sp_len[i] = (uint8_t)lzg_sp_len(sp_pool, g->node_sp_id[i]);
    }

    LZGFinalizeGeneInputs gene_inputs = {
        .gene_pool = gene_pool,
        .v_marginal_counts = v_marginal_counts,
        .j_marginal_counts = j_marginal_counts,
        .vj_pair_counts = vj_pair_counts,
        .edge_v_genes = edge_v_genes,
        .edge_j_genes = edge_j_genes,
    };
    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, len_counts, max_len, eb, builder_to_csr, has_genes ? &gene_inputs : NULL);
    if (topo_err == LZG_ERR_ALLOC) {
        free(builder_to_csr);
        cleanup_structural_finalize_inputs(
            eb, build_nodes, len_counts, gene_pool,
            v_marginal_counts, j_marginal_counts, vj_pair_counts,
            edge_v_genes, edge_j_genes,
            !(g->gene_data && has_genes && g->gene_data->gene_pool == gene_pool));
        return topo_err;
    }

    if (has_genes) {
        lzg_hm_destroy(v_marginal_counts);
        lzg_hm_destroy(j_marginal_counts);
        lzg_hm_destroy(vj_pair_counts);
        lzg_hm_destroy(edge_v_genes);
        lzg_hm_destroy(edge_j_genes);
    }

    free(builder_to_csr);
    lzg_eb_destroy(eb);
    lzg_node_builder_destroy(build_nodes);

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

static LZGError finalize_from_edges(
    LZGGraph *g,
    LZGEdgeBuilder *eb,
    LZGHashMap *node_set,
    LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts,
    LZGHashMap *outgoing_counts,
    uint64_t *len_counts, uint32_t max_len,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts,
    LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes,
    LZGHashMap *edge_j_genes)
{
    bool has_genes = (gene_pool != NULL);

    /* ── Build node ID mapping ── */
    uint32_t n_nodes = node_set->count;
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

    lzg_graph_alloc_csr_storage(g, n_nodes, eb->n_edges);

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
        g->outgoing_counts[src_idx] += eb->counts[e];
        if (builder_to_csr) builder_to_csr[e] = pos;
        edge_deg[src_idx]++;
    }
    free(edge_deg);

    /* ── Parse node metadata and compute derived quantities ── */
    LZGStringPool *sp_pool = g->pool; /* reuse the same pool for subpatterns */

    for (uint32_t i = 0; i < n_nodes; i++) {
        lzg_graph_parse_node_label(g->pool, label_ids[i], g->variant,
                                   &g->node_sp_id[i], &g->node_pos[i], sp_pool);
        g->node_sp_len[i] = (uint8_t)lzg_sp_len(sp_pool, g->node_sp_id[i]);
    }

    LZGFinalizeGeneInputs gene_inputs = {
        .gene_pool = gene_pool,
        .v_marginal_counts = v_marginal_counts,
        .j_marginal_counts = j_marginal_counts,
        .vj_pair_counts = vj_pair_counts,
        .edge_v_genes = edge_v_genes,
        .edge_j_genes = edge_j_genes,
    };
    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, len_counts, max_len, eb, builder_to_csr, has_genes ? &gene_inputs : NULL);

    if (has_genes) {
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
