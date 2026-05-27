/**
 * @file flashback_graph_build.c
 * @brief FlashBackGraph construction from sequences (list or file).
 *
 * Uses FlashBack decomposition to tokenize sequences, then builds a
 * standard CSR graph via the shared EdgeBuilder → finalize pipeline.
 * After finalization, root/sink nodes are corrected for FlashBack
 * token conventions (root = @$_1{0}, sinks = zero out-degree).
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzgraph/flashback.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/graph.h"

#include "../graph_core/graph_finalize_internal.h"
#include "../graph_core/graph_build_ingest_internal.h"


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
 * Pack CSR + node metadata from read-only resources into a target graph.
 *
 * Takes ownership of ``len_counts``: on success it is assigned to
 * ``g->length_counts`` by ``lzg_graph_finalize_derived_state``; on early
 * failure it is freed here so the caller does not need to track it.
 * ``eb`` and ``nb`` are read-only — the caller decides whether to free
 * them after this returns. Used by both ``fb_finalize`` (which then
 * destroys eb/nb) and ``lzg_flashback_stream_snapshot`` (which keeps
 * them alive for further accumulation).
 */
static LZGError fb_pack_into_graph(LZGGraph *g,
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

    /* Pre-set root_node to suppress the generic LZ76 "no @ root" warning.
     * The real root is identified by lzg_flashback_fix_special_nodes below. */
    g->root_node = 0;

    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, len_counts, max_len, eb, NULL, NULL);
    /* From here on len_counts is owned by g (assigned to g->length_counts). */

    lzg_flashback_fix_special_nodes(g);

    if (topo_err == LZG_ERR_HAS_CYCLES) {
        g->topo_valid = false;
        return LZG_OK;
    }
    return topo_err;
}

/**
 * Common finalization: CSR packing + FlashBack root/sink fixup.
 * Consumes the resources (eb and nb are destroyed).
 */
static LZGError fb_finalize(LZGGraph *g, LZGBuildResources *res,
                            uint32_t max_len) {
    uint64_t *len_counts = res->len_counts;
    res->len_counts = NULL;  /* ownership transferred via fb_pack_into_graph */

    LZGError err = fb_pack_into_graph(g, res->edge_builder, res->build_nodes,
                                       len_counts, max_len);

    lzg_eb_destroy(res->edge_builder);
    lzg_node_builder_destroy(res->build_nodes);
    res->edge_builder = NULL;
    res->build_nodes = NULL;

    if (err == LZG_OK) {
        LZG_INFO("flashback graph ready: %u nodes, %u edges, root=%u",
                 g->n_nodes, g->n_edges, g->root_node);
    } else if (err == LZG_ERR_HAS_CYCLES) {
        LZG_INFO("flashback graph ready: %u nodes, %u edges (has cycles)",
                 g->n_nodes, g->n_edges);
    }
    return err;
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

    uint32_t eb_cap = fb_bounded_cap(n_seqs, 8u, 256u, LZG_BUILD_INIT_CAP_MAX);
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

/* ══════════════════════════════════════════════════════════════ */
/* Streaming builder — incremental sequence ingestion              */
/* ══════════════════════════════════════════════════════════════ */

struct LZGFlashbackStream {
    LZGGraph         *g;
    LZGBuildResources res;
    uint32_t          max_len;
    bool              consumed;     /* true after finalize or abort */
};

/*
 * The list and file builders both follow the same pattern:
 *   create resources -> fb_accumulate per sequence -> fb_finalize.
 * The streaming API exposes those three steps to callers so they can
 * monitor the in-progress state (current node/edge count) and decide
 * dynamically when to stop. Internally it shares fb_accumulate and
 * fb_finalize so behaviour is identical to the batch builders.
 *
 * The opaque LZGFlashbackStream owns:
 *   - a fresh LZGGraph (allocated on open, released on
 *     finalize/abort),
 *   - the LZGBuildResources accumulator (edges + nodes + len counts),
 *   - the running max_len.
 *
 * After finalize the graph ownership transfers to the caller (handed
 * back via *out) and the stream itself is freed; further calls on the
 * stream pointer are undefined.
 *
 * Lifecycle note: open() returns a heap-allocated stream that owns
 * a graph + accumulator. finalize() transfers the graph to the caller
 * and releases the accumulator (the struct itself stays alive, in a
 * "consumed" state, so a Python capsule destructor can safely call
 * destroy() on it). abort() releases everything but keeps the struct.
 * The struct is finally freed by destroy(), which is the only call
 * that frees the LZGFlashbackStream * itself.
 *
 * State table:
 *   open  : g != NULL, resources live, consumed=false
 *   add   : permitted iff consumed==false
 *   peek  : permitted iff consumed==false; returns 0/0 otherwise
 *   final : consumed=true; g transferred to caller; resources freed
 *   abort : consumed=true; g and resources freed
 *   destr : frees the struct itself; safe to call after any state.
 */

LZGFlashbackStream *lzg_flashback_stream_open(double smoothing) {
    LZGFlashbackStream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;

    s->g = lzg_graph_create(LZG_VARIANT_NAIVE);
    if (!s->g) { free(s); return NULL; }
    s->g->smoothing_alpha = smoothing;

    s->res.edge_builder = lzg_eb_create(256);
    if (!s->res.edge_builder) {
        lzg_graph_destroy(s->g); free(s); return NULL;
    }

    s->res.build_nodes = lzg_node_builder_create(4096);
    s->res.len_cap = 128;
    s->res.len_counts = calloc(s->res.len_cap, sizeof(uint64_t));
    if (!s->res.build_nodes || !s->res.len_counts) {
        lzg_build_resources_destroy(&s->res);
        lzg_graph_destroy(s->g);
        free(s);
        return NULL;
    }

    s->max_len = 0;
    s->consumed = false;
    return s;
}

LZGError lzg_flashback_stream_add(LZGFlashbackStream *s,
                                   const char *const *sequences,
                                   uint32_t n_seqs,
                                   const uint64_t *counts) {
    if (!s) return LZG_ERR_NULL_ARG;
    if (s->consumed) return LZG_ERR_INVALID_ARG;
    if (n_seqs == 0) return LZG_OK;
    if (!sequences) return LZG_ERR_NULL_ARG;

    for (uint32_t i = 0; i < n_seqs; i++) {
        if (!sequences[i] || sequences[i][0] == '\0') continue;
        uint64_t c = counts ? counts[i] : 1;
        if (c == 0) continue;
        LZGError err = fb_accumulate(s->g, &s->res, sequences[i], c, &s->max_len);
        if (err != LZG_OK) return err;
    }
    return LZG_OK;
}

void lzg_flashback_stream_peek(const LZGFlashbackStream *s,
                                uint32_t *out_n_nodes,
                                uint32_t *out_n_edges) {
    if (out_n_nodes) *out_n_nodes = 0;
    if (out_n_edges) *out_n_edges = 0;
    if (!s || s->consumed) return;
    if (out_n_nodes && s->res.build_nodes)
        *out_n_nodes = s->res.build_nodes->count;
    if (out_n_edges && s->res.edge_builder)
        *out_n_edges = s->res.edge_builder->n_edges;
}

LZGError lzg_flashback_stream_finalize(LZGFlashbackStream *s,
                                        LZGGraph **out) {
    if (!s || !out) return LZG_ERR_NULL_ARG;
    *out = NULL;
    if (s->consumed) return LZG_ERR_INVALID_ARG;

    LZGError err = fb_finalize(s->g, &s->res, s->max_len);
    s->consumed = true;
    if (err != LZG_OK) {
        /* fb_finalize cleans up resources internally on success but
         * not on failure — release here. */
        lzg_build_resources_destroy(&s->res);
        lzg_graph_destroy(s->g);
        s->g = NULL;
        return err;
    }
    /* Transfer graph ownership to caller; struct stays alive but
     * holds no graph pointer (so destroy() is a clean free). */
    *out = s->g;
    s->g = NULL;
    return LZG_OK;
}

LZGError lzg_flashback_stream_snapshot(LZGFlashbackStream *s,
                                        LZGGraph **out) {
    if (!s || !out) return LZG_ERR_NULL_ARG;
    *out = NULL;
    if (s->consumed) return LZG_ERR_INVALID_ARG;
    if (!s->res.edge_builder || !s->res.build_nodes) return LZG_ERR_INVALID_ARG;

    /* Build a fresh graph that borrows the source pool. */
    LZGGraph *g = lzg_graph_create(s->g->variant);
    if (!g) return LZG_ERR_ALLOC;
    g->smoothing_alpha = s->g->smoothing_alpha;
    /* Drop the auto-allocated empty pool; retain the source's instead.
     * The borrowed pool will outlive the snapshot as long as the stream
     * does, and refcounting ensures it survives the snapshot's destroy. */
    lzg_sp_destroy(g->pool);
    g->pool = lzg_sp_retain(s->g->pool);

    /* Copy len_counts (will be owned by the snapshot graph after pack). */
    uint64_t *len_counts_copy = NULL;
    if (s->res.len_counts && s->res.len_cap > 0) {
        len_counts_copy = malloc(s->res.len_cap * sizeof(uint64_t));
        if (!len_counts_copy) { lzg_graph_destroy(g); return LZG_ERR_ALLOC; }
        memcpy(len_counts_copy, s->res.len_counts,
               s->res.len_cap * sizeof(uint64_t));
    }

    LZGError err = fb_pack_into_graph(g, s->res.edge_builder, s->res.build_nodes,
                                       len_counts_copy, s->max_len);
    /* Source resources are untouched: stream remains live for further adds. */

    if (err != LZG_OK && err != LZG_ERR_HAS_CYCLES) {
        lzg_graph_destroy(g);  /* frees len_counts_copy via g->length_counts */
        return err;
    }
    *out = g;
    return LZG_OK;
}

void lzg_flashback_stream_abort(LZGFlashbackStream *s) {
    if (!s) return;
    if (s->consumed) return;
    lzg_build_resources_destroy(&s->res);
    if (s->g) {
        lzg_graph_destroy(s->g);
        s->g = NULL;
    }
    s->consumed = true;
}

void lzg_flashback_stream_destroy(LZGFlashbackStream *s) {
    if (!s) return;
    if (!s->consumed) {
        lzg_build_resources_destroy(&s->res);
        if (s->g) lzg_graph_destroy(s->g);
    }
    free(s);
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
