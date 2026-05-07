/**
 * @file flashback_graph_build.c
 * @brief FlashBackGraph construction from sequences (list or file).
 *
 * Uses FlashBack decomposition to tokenize sequences, then builds a
 * standard CSR graph via the shared EdgeBuilder → finalize pipeline.
 * After finalization, root/sink nodes are corrected for FlashBack
 * token conventions (root = @$_1{0}, sinks = zero out-degree).
 */
#include "lzgraph/flashback.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/graph.h"
#include "../graph/graph_finalize.h"
#include "../graph/graph_build_ingest.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FB_BUILD_INIT_CAP_MAX (1u << 20)

static inline uint32_t fb_bounded_cap(uint32_t n, uint32_t mul,
                                      uint32_t lo, uint32_t hi) {
    uint64_t est = (uint64_t)n * (uint64_t)mul;
    if (est < lo) return lo;
    if (est > hi) return hi;
    return (uint32_t)est;
}

/**
 * Accumulate one sequence into the build resources using FlashBack.
 */
static LZGError fb_accumulate(LZGGraph *g, LZGBuildResources *res,
                              const char *seq, uint64_t count,
                              uint32_t *max_len) {
    uint32_t seq_len = (uint32_t)strlen(seq);

    LZGFlashbackTokens tokens;
    LZGError err = lzg_flashback_decompose(seq, seq_len, g->pool, &tokens);
    if (err != LZG_OK) return err;
    if (tokens.count == 0) return LZG_OK;

    /* Intern each token as a node via LZGNodeBuilder.
     * Key = (sp_id, discovery_order) to ensure uniqueness. */
    uint32_t node_ids[LZG_FLASHBACK_MAX_TOKENS];
    for (uint32_t i = 0; i < tokens.count; i++) {
        LZGNodeBuilder *nb = res->build_nodes;
        if (nb->count >= nb->capacity) {
            uint32_t new_cap = nb->capacity * 2;
            uint32_t *new_sp = realloc(nb->sp_ids, new_cap * sizeof(uint32_t));
            if (!new_sp) return LZG_ERR_ALLOC;
            uint32_t *new_pos = realloc(nb->positions, new_cap * sizeof(uint32_t));
            if (!new_pos) { nb->sp_ids = new_sp; return LZG_ERR_ALLOC; }
            nb->sp_ids = new_sp;
            nb->positions = new_pos;
            nb->capacity = new_cap;
        }
        {
            /* Pack key: sp_id in high 32 bits, discovery order in low 32 bits.
             * The {order} is already embedded in the token string, and the
             * sp_id is unique per token string, so sp_id alone suffices.
             * Use UINT32_MAX as position to distinguish from LZ76 keys. */
            uint64_t key = ((uint64_t)tokens.sp_ids[i] << 32) | (uint64_t)UINT32_MAX;
            bool inserted = false;
            uint64_t *slot = lzg_hm_get_or_insert(nb->key_to_id, key,
                                                  (uint64_t)nb->count, &inserted);
            uint32_t node_id = (uint32_t)*slot;
            if (inserted) {
                nb->sp_ids[node_id] = tokens.sp_ids[i];
                nb->positions[node_id] = UINT32_MAX;
                nb->count++;
            }
            node_ids[i] = node_id;
        }
    }

    /* Record edges between consecutive tokens */
    for (uint32_t i = 0; i < tokens.count - 1; i++) {
        uint32_t edge_idx = UINT32_MAX;
        err = lzg_eb_record(res->edge_builder, node_ids[i], node_ids[i + 1],
                            count, &edge_idx);
        if (err != LZG_OK) return err;
    }

    /* Update length distribution */
    if (seq_len >= res->len_cap) {
        uint32_t new_cap = seq_len + 64;
        uint64_t *new_counts = realloc(res->len_counts, new_cap * sizeof(uint64_t));
        if (!new_counts) return LZG_ERR_ALLOC;
        memset(new_counts + res->len_cap, 0, (new_cap - res->len_cap) * sizeof(uint64_t));
        res->len_counts = new_counts;
        res->len_cap = new_cap;
    }
    res->len_counts[seq_len] += count;
    if (seq_len > *max_len) *max_len = seq_len;

    return LZG_OK;
}

/**
 * Common finalization: CSR packing + FlashBack root/sink fixup.
 */
static LZGError fb_finalize(LZGGraph *g, LZGBuildResources *res,
                            uint32_t max_len) {
    uint32_t n_nodes = res->build_nodes->count;
    uint32_t n_edges = res->edge_builder->n_edges;

    lzg_graph_alloc_csr_storage(g, n_nodes, n_edges);
    if (!g->row_offsets || !g->col_indices || !g->edge_weights ||
        !g->edge_counts || !g->edge_sp_id || !g->edge_sp_len ||
        !g->edge_prefix_id || !g->outgoing_counts ||
        !g->node_sp_id || !g->node_sp_len || !g->node_pos)
        return LZG_ERR_ALLOC;

    uint32_t *edge_deg = calloc(n_nodes, sizeof(uint32_t));
    if (!edge_deg) return LZG_ERR_ALLOC;

    LZGEdgeBuilder *eb = res->edge_builder;
    for (uint32_t e = 0; e < n_edges; e++)
        edge_deg[eb->src_ids[e]]++;

    g->row_offsets[0] = 0;
    for (uint32_t i = 0; i < n_nodes; i++)
        g->row_offsets[i + 1] = g->row_offsets[i] + edge_deg[i];

    memset(edge_deg, 0, n_nodes * sizeof(uint32_t));
    for (uint32_t e = 0; e < n_edges; e++) {
        uint32_t src = eb->src_ids[e];
        uint32_t pos = g->row_offsets[src] + edge_deg[src];
        g->col_indices[pos] = eb->dst_ids[e];
        g->edge_counts[pos] = eb->counts[e];
        g->outgoing_counts[src] += eb->counts[e];
        edge_deg[src]++;
    }
    free(edge_deg);

    LZGNodeBuilder *nb = res->build_nodes;
    for (uint32_t i = 0; i < n_nodes; i++) {
        g->node_sp_id[i] = nb->sp_ids[i];
        g->node_pos[i] = nb->positions[i];
        g->node_sp_len[i] = (uint8_t)lzg_sp_len(g->pool, nb->sp_ids[i]);
    }

    /* Pre-set root_node to suppress the generic LZ76 "no @ root" warning.
     * The real root is identified by lzg_flashback_fix_special_nodes below. */
    g->root_node = 0;

    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, res->len_counts, max_len, eb, NULL, NULL);

    lzg_eb_destroy(res->edge_builder);
    lzg_node_builder_destroy(res->build_nodes);
    res->edge_builder = NULL;
    res->build_nodes = NULL;
    res->len_counts = NULL;

    lzg_flashback_fix_special_nodes(g);

    if (topo_err == LZG_ERR_HAS_CYCLES) {
        g->topo_valid = false;
        LZG_INFO("flashback graph ready: %u nodes, %u edges (has cycles)",
                 g->n_nodes, g->n_edges);
        return LZG_OK;
    }
    if (topo_err != LZG_OK) return topo_err;

    LZG_INFO("flashback graph ready: %u nodes, %u edges, root=%u",
             g->n_nodes, g->n_edges, g->root_node);
    return LZG_OK;
}

/* ── Build from in-memory sequence array ─────────────────────── */

LZGError lzg_flashback_graph_build(LZGGraph *g,
                                   const char **sequences,
                                   uint32_t n_seqs,
                                   const uint64_t *abundances,
                                   double smoothing) {
    if (!g || !sequences || n_seqs == 0) return LZG_ERR_INVALID_ARG;
    g->smoothing_alpha = smoothing;

    LZG_INFO("flashback graph: building from %u sequences", n_seqs);

    uint32_t eb_cap = fb_bounded_cap(n_seqs, 8u, 256u, FB_BUILD_INIT_CAP_MAX);
    LZGBuildResources res = {0};
    res.edge_builder = lzg_eb_create(eb_cap);
    if (!res.edge_builder) return LZG_ERR_ALLOC;

    res.build_nodes = lzg_node_builder_create(4096);
    res.len_cap = 128;
    res.len_counts = calloc(res.len_cap, sizeof(uint64_t));
    if (!res.build_nodes || !res.len_counts) {
        lzg_build_resources_destroy(&res);
        return LZG_ERR_ALLOC;
    }

    uint32_t max_len = 0;
    for (uint32_t s = 0; s < n_seqs; s++) {
        uint64_t count = abundances ? abundances[s] : 1;
        LZGError err = fb_accumulate(g, &res, sequences[s], count, &max_len);
        if (err != LZG_OK) {
            lzg_build_resources_destroy(&res);
            return err;
        }
    }

    LZGError err = fb_finalize(g, &res, max_len);
    if (err != LZG_OK) lzg_build_resources_destroy(&res);
    return err;
}

/* ── Build from plain text file (streaming) ──────────────────── */

LZGError lzg_flashback_graph_build_file(LZGGraph *g,
                                        const char *path,
                                        double smoothing) {
    if (!g || !path || path[0] == '\0') return LZG_ERR_INVALID_ARG;

    FILE *fh = fopen(path, "r");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN,
                             "flashback: could not open '%s'", path);

    g->smoothing_alpha = smoothing;

    LZGStreamBuildStats stats = {0};
    stats.file_size_bytes = lzg_detect_regular_file_size(path);
    stats.start_time = lzg_build_monotonic_seconds();
    stats.last_log_time = stats.start_time;
    stats.peak_rss_kb = lzg_build_current_rss_kb();
    LZG_INFO("flashback stream build: start file=%s size=%.1fMB",
             path, (double)stats.file_size_bytes / (1024.0 * 1024.0));

    LZGBuildResources res = {0};
    res.edge_builder = lzg_eb_create(256);
    if (!res.edge_builder) { fclose(fh); return LZG_ERR_ALLOC; }

    res.build_nodes = lzg_node_builder_create(4096);
    res.len_cap = 128;
    res.len_counts = calloc(res.len_cap, sizeof(uint64_t));
    if (!res.build_nodes || !res.len_counts) {
        fclose(fh);
        lzg_build_resources_destroy(&res);
        return LZG_ERR_ALLOC;
    }

    uint32_t max_len = 0;
    char *line = NULL;
    size_t line_cap = 0;
    ptrdiff_t nread;
    uint64_t lines_seen = 0, sequences_seen = 0;
    errno = 0;

    while ((nread = lzg_getline_portable(&line, &line_cap, fh)) != -1) {
        lines_seen++;
        stats.bytes_seen += (uint64_t)nread;

        char *seq = NULL;
        uint64_t count = 0;
        LZGParsedLineKind line_kind = LZG_LINE_EMPTY;
        LZGError err = lzg_parse_plain_sequence_line(line, &seq, &count, &line_kind);
        if (err != LZG_OK) {
            free(line); fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_update_stream_mode(&stats, line_kind, path, lines_seen);
        if (!seq || count == 0) continue;
        sequences_seen++;

        err = fb_accumulate(g, &res, seq, count, &max_len);
        if (err != LZG_OK) {
            free(line); fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_maybe_log_stream_progress(path, lines_seen, sequences_seen,
                                      &res, &stats);
    }

    if (ferror(fh) || (!feof(fh) && errno != 0)) {
        int saved = errno;
        free(line); fclose(fh);
        lzg_build_resources_destroy(&res);
        if (saved == ENOMEM)
            return LZG_FAIL(LZG_ERR_ALLOC, "flashback stream OOM reading '%s'", path);
        return LZG_FAIL(LZG_ERR_IO_READ, "flashback stream failed reading '%s'", path);
    }

    free(line);
    fclose(fh);

    {
        double end = lzg_build_monotonic_seconds();
        double elapsed = (end > stats.start_time) ? (end - stats.start_time) : 0;
        LZG_INFO("flashback stream build: ingest done file=%s lines=%llu "
                 "sequences=%llu nodes=%u edges=%u elapsed=%.1fs rate=%.0f/s",
                 path, (unsigned long long)lines_seen,
                 (unsigned long long)sequences_seen,
                 res.build_nodes ? res.build_nodes->count : 0u,
                 res.edge_builder->n_edges, elapsed,
                 elapsed > 0 ? (double)lines_seen / elapsed : 0);
    }

    LZGError final_err = fb_finalize(g, &res, max_len);
    if (final_err != LZG_OK) lzg_build_resources_destroy(&res);
    return final_err;
}

/* ── Public fixup: re-identify root/sinks for FlashBack tokens ── */

void lzg_flashback_fix_special_nodes(LZGGraph *g) {
    if (!g || g->n_nodes == 0) return;

    g->root_node = UINT32_MAX;
    if (g->node_is_sink)
        memset(g->node_is_sink, 0, g->n_nodes * sizeof(uint8_t));

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        if (sp[0] == '@')
            g->root_node = i;
        uint32_t out_deg = g->row_offsets[i + 1] - g->row_offsets[i];
        if (out_deg == 0 && g->node_is_sink)
            g->node_is_sink[i] = 1;
    }
}
