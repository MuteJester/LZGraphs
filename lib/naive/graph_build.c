/**
 * @file graph_build.c
 * @brief NaiveGraph construction from sequences (list or file).
 *
 * Tokenizes with the naive positional encoding, then hands off to the
 * shared EdgeBuilder -> finalize pipeline, exactly as the FlashBack
 * builder does. Because the sentinels are the bare strings "@" and "$",
 * the generic special-node pass inside lzg_graph_finalize_derived_state
 * already identifies the root and the sink correctly, so there is no
 * post-finalize fixup in this path.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzgraph/graph.h"
#include "lzgraph/naive_graph.h"

#include "../graph_core/graph_build_ingest_internal.h"
#include "../graph_core/graph_finalize_internal.h"


static inline uint32_t naive_bounded_cap(uint32_t n, uint32_t mul,
                                         uint32_t lo, uint32_t hi) {
    uint64_t est = (uint64_t)n * (uint64_t)mul;
    if (est < lo) return lo;
    if (est > hi) return hi;
    return (uint32_t)est;
}

/**
 * Accumulate one sequence into the build resources.
 *
 * Sequences longer than `max_length` (when non-zero) are skipped and
 * counted in *skipped, leaving the length histogram untouched — they
 * must not contribute to a distribution the model cannot represent.
 */
static LZGError naive_accumulate(LZGGraph *g, LZGBuildResources *res,
                                 const char *seq, uint64_t count,
                                 uint32_t max_length,
                                 uint32_t *max_len, uint64_t *skipped) {
    uint32_t seq_len = (uint32_t)strlen(seq);
    if (seq_len == 0) return LZG_OK;
    if (max_length > 0 && seq_len > max_length) {
        if (skipped) (*skipped)++;
        return LZG_OK;
    }
    /* Even uncapped, a sequence longer than the walk buffer cannot be
     * encoded. Skip it like an over-cap sequence rather than failing the
     * whole build for one pathological record. */
    if (seq_len + 2u > LZG_NAIVE_MAX_WALK) {
        if (skipped) (*skipped)++;
        return LZG_OK;
    }

    uint32_t label_ids[LZG_NAIVE_MAX_WALK];
    uint32_t n_labels = 0;
    LZGError err = lzg_naive_encode(seq, seq_len, g->pool,
                                    label_ids, &n_labels);
    if (err != LZG_OK) return err;
    if (n_labels < 2) return LZG_OK;

    /* Intern each label as a node. The label string already carries the
     * position, so the interned ID alone identifies the node; position is
     * stored as UINT32_MAX to mark "identity lives in the label", the same
     * convention FlashBack uses and the one the query-side node map and
     * reconstruct_node_label both key on. */
    uint32_t node_ids[LZG_NAIVE_MAX_WALK];
    for (uint32_t i = 0; i < n_labels; i++) {
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
        uint64_t key = ((uint64_t)label_ids[i] << 32) | (uint64_t)UINT32_MAX;
        bool inserted = false;
        uint64_t *slot = lzg_hm_get_or_insert(nb->key_to_id, key,
                                              (uint64_t)nb->count, &inserted);
        uint32_t node_id = (uint32_t)*slot;
        if (inserted) {
            nb->sp_ids[node_id] = label_ids[i];
            nb->positions[node_id] = UINT32_MAX;
            nb->count++;
        }
        node_ids[i] = node_id;
    }

    for (uint32_t i = 0; i + 1 < n_labels; i++) {
        uint32_t edge_idx = UINT32_MAX;
        err = lzg_eb_record(res->edge_builder, node_ids[i], node_ids[i + 1],
                            count, &edge_idx);
        if (err != LZG_OK) return err;
    }

    if (seq_len >= res->len_cap) {
        uint32_t new_cap = seq_len + 64;
        uint64_t *new_counts = realloc(res->len_counts,
                                       new_cap * sizeof(uint64_t));
        if (!new_counts) return LZG_ERR_ALLOC;
        memset(new_counts + res->len_cap, 0,
               (new_cap - res->len_cap) * sizeof(uint64_t));
        res->len_counts = new_counts;
        res->len_cap = new_cap;
    }
    res->len_counts[seq_len] += count;
    if (seq_len > *max_len) *max_len = seq_len;

    return LZG_OK;
}

/**
 * Pack CSR + node metadata into the target graph.
 *
 * Takes ownership of `len_counts`: on success it becomes
 * g->length_counts; on early failure it is freed here.
 */
static LZGError naive_pack_into_graph(LZGGraph *g,
                                      const LZGEdgeBuilder *eb,
                                      const LZGNodeBuilder *nb,
                                      uint64_t *len_counts,
                                      uint32_t max_len) {
    uint32_t n_nodes = nb->count;
    uint32_t n_edges = eb->n_edges;

    lzg_graph_alloc_csr_storage(g, n_nodes, n_edges);
    if (!g->row_offsets || !g->col_indices || !g->edge_weights ||
        !g->edge_counts || !g->edge_sp_id || !g->edge_sp_len ||
        !g->edge_prefix_id || !g->outgoing_counts ||
        !g->node_sp_id || !g->node_sp_len || !g->node_pos) {
        free(len_counts);
        return LZG_ERR_ALLOC;
    }

    uint32_t *edge_deg = calloc(n_nodes, sizeof(uint32_t));
    if (!edge_deg) { free(len_counts); return LZG_ERR_ALLOC; }

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

    for (uint32_t i = 0; i < n_nodes; i++) {
        g->node_sp_id[i] = nb->sp_ids[i];
        g->node_pos[i] = nb->positions[i];
        g->node_sp_len[i] = (uint8_t)lzg_sp_len(g->pool, nb->sp_ids[i]);
    }

    /* The bare "@" / "$" labels are what the generic pass inside
     * finalize expects, so root and sinks come out correct with no
     * variant-specific fixup. */
    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, len_counts, max_len, eb, NULL, NULL);
    /* len_counts is owned by g from here on. */

    if (topo_err == LZG_ERR_HAS_CYCLES) {
        g->topo_valid = false;
        return LZG_OK;
    }
    return topo_err;
}

/** Common finalization; consumes the build resources. */
static LZGError naive_finalize(LZGGraph *g, LZGBuildResources *res,
                               uint32_t max_len) {
    uint64_t *len_counts = res->len_counts;
    res->len_counts = NULL;  /* ownership moves into naive_pack_into_graph */

    LZGError err = naive_pack_into_graph(g, res->edge_builder,
                                         res->build_nodes, len_counts, max_len);

    lzg_eb_destroy(res->edge_builder);
    lzg_node_builder_destroy(res->build_nodes);
    res->edge_builder = NULL;
    res->build_nodes = NULL;

    if (err == LZG_OK) {
        LZG_INFO("naive graph ready: %u nodes, %u edges, root=%u",
                 g->n_nodes, g->n_edges, g->root_node);
    }
    return err;
}

static LZGError naive_resources_init(LZGBuildResources *res, uint32_t eb_cap) {
    res->edge_builder = lzg_eb_create(eb_cap);
    if (!res->edge_builder) return LZG_ERR_ALLOC;
    res->build_nodes = lzg_node_builder_create(4096);
    res->len_cap = 128;
    res->len_counts = calloc(res->len_cap, sizeof(uint64_t));
    if (!res->build_nodes || !res->len_counts) {
        lzg_build_resources_destroy(res);
        return LZG_ERR_ALLOC;
    }
    return LZG_OK;
}

/* ── Build from in-memory sequence array ─────────────────────── */

LZGError lzg_naive_graph_build(LZGGraph *g,
                               const char **sequences,
                               uint32_t n_seqs,
                               const uint64_t *abundances,
                               uint32_t max_length,
                               double smoothing) {
    if (!g || !sequences || n_seqs == 0) return LZG_ERR_INVALID_ARG;
    g->smoothing_alpha = smoothing;

    LZG_INFO("naive graph: building from %u sequences (max_length=%u)",
             n_seqs, max_length);

    LZGBuildResources res = {0};
    uint32_t eb_cap = naive_bounded_cap(n_seqs, 8u, 256u,
                                        LZG_BUILD_INIT_CAP_MAX);
    LZGError err = naive_resources_init(&res, eb_cap);
    if (err != LZG_OK) return err;

    uint32_t max_len = 0;
    uint64_t skipped = 0;
    for (uint32_t s = 0; s < n_seqs; s++) {
        if (!sequences[s]) continue;
        uint64_t count = abundances ? abundances[s] : 1;
        if (count == 0) continue;
        err = naive_accumulate(g, &res, sequences[s], count, max_length,
                               &max_len, &skipped);
        if (err != LZG_OK) {
            lzg_build_resources_destroy(&res);
            return err;
        }
    }

    if (skipped > 0)
        LZG_INFO("naive graph: skipped %llu sequences longer than %u",
                 (unsigned long long)skipped, max_length);

    if (res.build_nodes->count == 0) {
        lzg_build_resources_destroy(&res);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "naive graph: no sequence survived filtering "
                        "(max_length=%u)", max_length);
    }

    err = naive_finalize(g, &res, max_len);
    if (err != LZG_OK) lzg_build_resources_destroy(&res);
    return err;
}

/* ── Build from plain text file (streaming) ──────────────────── */

LZGError lzg_naive_graph_build_file(LZGGraph *g,
                                    const char *path,
                                    uint32_t max_length,
                                    double smoothing) {
    if (!g || !path || path[0] == '\0') return LZG_ERR_INVALID_ARG;

    FILE *fh = fopen(path, "r");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN,
                             "naive: could not open '%s'", path);

    g->smoothing_alpha = smoothing;

    LZGStreamBuildStats stats = {0};
    stats.file_size_bytes = lzg_detect_regular_file_size(path);
    stats.start_time = lzg_build_monotonic_seconds();
    stats.last_log_time = stats.start_time;
    stats.peak_rss_kb = lzg_build_current_rss_kb();
    LZG_INFO("naive stream build: start file=%s size=%.1fMB max_length=%u",
             path, (double)stats.file_size_bytes / (1024.0 * 1024.0),
             max_length);

    LZGBuildResources res = {0};
    LZGError err = naive_resources_init(&res, 256);
    if (err != LZG_OK) { fclose(fh); return err; }

    uint32_t max_len = 0;
    uint64_t skipped = 0;
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
        err = lzg_parse_plain_sequence_line(line, &seq, &count, &line_kind);
        if (err != LZG_OK) {
            free(line); fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_update_stream_mode(&stats, line_kind, path, lines_seen);
        if (!seq || count == 0) continue;
        sequences_seen++;

        err = naive_accumulate(g, &res, seq, count, max_length,
                               &max_len, &skipped);
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
            return LZG_FAIL(LZG_ERR_ALLOC, "naive stream OOM reading '%s'", path);
        return LZG_FAIL(LZG_ERR_IO_READ, "naive stream failed reading '%s'", path);
    }

    free(line);
    fclose(fh);

    {
        double end = lzg_build_monotonic_seconds();
        double elapsed = (end > stats.start_time) ? (end - stats.start_time) : 0;
        LZG_INFO("naive stream build: ingest done file=%s lines=%llu "
                 "sequences=%llu skipped=%llu nodes=%u edges=%u "
                 "elapsed=%.1fs rate=%.0f/s",
                 path, (unsigned long long)lines_seen,
                 (unsigned long long)sequences_seen,
                 (unsigned long long)skipped,
                 res.build_nodes ? res.build_nodes->count : 0u,
                 res.edge_builder->n_edges, elapsed,
                 elapsed > 0 ? (double)lines_seen / elapsed : 0);
    }

    if (res.build_nodes->count == 0) {
        lzg_build_resources_destroy(&res);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "naive graph: no sequence in '%s' survived filtering "
                        "(max_length=%u)", path, max_length);
    }

    LZGError final_err = naive_finalize(g, &res, max_len);
    if (final_err != LZG_OK) lzg_build_resources_destroy(&res);
    return final_err;
}

/* ── Root/sink repair ────────────────────────────────────────── */

void lzg_naive_fix_special_nodes(LZGGraph *g) {
    if (!g || g->n_nodes == 0) return;

    g->root_node = UINT32_MAX;
    if (g->node_is_sink)
        memset(g->node_is_sink, 0, g->n_nodes * sizeof(uint8_t));

    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        if (sp[0] == LZG_START_SENTINEL && sp[1] == '\0')
            g->root_node = i;
        if (sp[0] == LZG_END_SENTINEL && sp[1] == '\0' && g->node_is_sink)
            g->node_is_sink[i] = 1;
    }
}
