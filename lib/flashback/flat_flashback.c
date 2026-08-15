/**
 * @file flat_flashback.c
 * @brief FlattenedFlashBack encoding, construction, and queries.
 *
 * The decomposition is the whole idea and it is four lines: walk inward from
 * both ends, emit one token per step holding the two residues met there, and
 * emit the odd middle residue on its own if the length is odd. FlashBack's
 * run compression is exactly what has been removed; everything else about
 * the bilateral scan is kept.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzgraph/flashback_graph.h"
#include "lzgraph/flat_flashback.h"
#include "lzgraph/graph.h"

#include "../graph_core/graph_build_ingest_internal.h"
#include "../graph_core/graph_finalize_internal.h"

/* ── Encoding ─────────────────────────────────────────────── */

LZGError lzg_flat_encode(const char *str, uint32_t len,
                         LZGStringPool *pool,
                         uint32_t *out_ids, uint32_t *out_count) {
    if (!str || !pool || !out_ids || !out_count) return LZG_ERR_NULL_ARG;
    if (len == 0) { *out_count = 0; return LZG_OK; }
    /* ceil(len/2) tokens plus the two sentinels. */
    if ((len + 1u) / 2u + 2u > LZG_FLAT_MAX_WALK)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                        "flat encode: sequence length %u exceeds max walk", len);

    static const char start_label[2] = {LZG_START_SENTINEL, '\0'};
    static const char end_label[2]   = {LZG_END_SENTINEL, '\0'};

    uint32_t n = 0;
    out_ids[n++] = lzg_sp_intern(pool, start_label);

    char label[LZG_FLAT_LABEL_CAP];
    uint32_t i = 0, j = len - 1, step = 1;
    while (i < j) {
        int w = snprintf(label, sizeof(label), "%c%c_%u", str[i], str[j], step);
        if (w <= 0 || (size_t)w >= sizeof(label))
            return LZG_FAIL(LZG_ERR_INTERNAL, "flat encode: label overflow");
        out_ids[n++] = lzg_sp_intern_n(pool, label, (uint32_t)w);
        i++; j--; step++;
    }
    if (i == j) {
        /* Odd length: one residue is left unpaired in the middle. */
        int w = snprintf(label, sizeof(label), "%c%c_%u",
                         str[i], LZG_END_SENTINEL, step);
        if (w <= 0 || (size_t)w >= sizeof(label))
            return LZG_FAIL(LZG_ERR_INTERNAL, "flat encode: label overflow");
        out_ids[n++] = lzg_sp_intern_n(pool, label, (uint32_t)w);
    }

    out_ids[n++] = lzg_sp_intern(pool, end_label);
    *out_count = n;
    return LZG_OK;
}

LZGError lzg_flat_reverse(const LZGStringPool *pool,
                          const uint32_t *label_ids, uint32_t count,
                          char *out_buf, uint32_t buf_cap,
                          uint32_t *out_len) {
    if (!pool || !label_ids || !out_buf || !out_len) return LZG_ERR_NULL_ARG;
    if (buf_cap == 0) return LZG_ERR_PARAM_OUT_OF_RANGE;

    /* Front residues accumulate left to right; back residues were met in
     * reverse order, so they are laid down from the right end inward. */
    uint32_t n_front = 0, n_back = 0;
    char middle = '\0';
    char back[LZG_FLAT_MAX_WALK];

    for (uint32_t t = 0; t < count; t++) {
        const char *lab = lzg_sp_get(pool, label_ids[t]);
        if (!lab || lab[0] == '\0') continue;
        if (lab[1] == '\0' &&
            (lab[0] == LZG_START_SENTINEL || lab[0] == LZG_END_SENTINEL))
            continue;                        /* bare sentinel node */
        if (lab[1] == LZG_END_SENTINEL) {    /* unpaired middle residue */
            middle = lab[0];
            continue;
        }
        if (n_front + 1u >= buf_cap || n_back >= LZG_FLAT_MAX_WALK)
            return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                            "flat reverse: output buffer too small");
        out_buf[n_front++] = lab[0];
        back[n_back++] = lab[1];
    }

    uint32_t total = n_front + (middle ? 1u : 0u) + n_back;
    if (total + 1u > buf_cap)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                        "flat reverse: output buffer too small");
    uint32_t w = n_front;
    if (middle) out_buf[w++] = middle;
    for (uint32_t k = n_back; k > 0; k--) out_buf[w++] = back[k - 1];
    out_buf[w] = '\0';
    *out_len = w;
    return LZG_OK;
}

/* ── Construction ─────────────────────────────────────────── */

static inline uint32_t flat_bounded_cap(uint32_t n, uint32_t mul,
                                        uint32_t lo, uint32_t hi) {
    uint64_t est = (uint64_t)n * (uint64_t)mul;
    if (est < lo) return lo;
    if (est > hi) return hi;
    return (uint32_t)est;
}

static LZGError flat_accumulate(LZGGraph *g, LZGBuildResources *res,
                                const char *seq, uint64_t count,
                                uint32_t max_length,
                                uint32_t *max_len, uint64_t *skipped) {
    uint32_t seq_len = (uint32_t)strlen(seq);
    if (seq_len == 0) return LZG_OK;
    if (max_length > 0 && seq_len > max_length) {
        if (skipped) (*skipped)++;
        return LZG_OK;
    }
    if ((seq_len + 1u) / 2u + 2u > LZG_FLAT_MAX_WALK) {
        if (skipped) (*skipped)++;
        return LZG_OK;
    }

    uint32_t label_ids[LZG_FLAT_MAX_WALK];
    uint32_t n_labels = 0;
    LZGError err = lzg_flat_encode(seq, seq_len, g->pool, label_ids, &n_labels);
    if (err != LZG_OK) return err;
    if (n_labels < 2) return LZG_OK;

    uint32_t node_ids[LZG_FLAT_MAX_WALK];
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
        /* The label carries the step, so the interned ID identifies the node;
         * UINT32_MAX position marks "identity lives in the label", the same
         * convention FlashBack and the naive positional graph use. */
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
        uint64_t *nc = realloc(res->len_counts, new_cap * sizeof(uint64_t));
        if (!nc) return LZG_ERR_ALLOC;
        memset(nc + res->len_cap, 0,
               (new_cap - res->len_cap) * sizeof(uint64_t));
        res->len_counts = nc;
        res->len_cap = new_cap;
    }
    res->len_counts[seq_len] += count;
    if (seq_len > *max_len) *max_len = seq_len;
    return LZG_OK;
}

static LZGError flat_pack_into_graph(LZGGraph *g,
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

    for (uint32_t e = 0; e < n_edges; e++) edge_deg[eb->src_ids[e]]++;
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

    g->root_node = 0;   /* suppress the generic "no @ root" warning */
    LZGError topo_err = lzg_graph_finalize_derived_state(
        g, len_counts, max_len, eb, NULL, NULL);
    /* The generic pass would mark "S$_4" a sink, since it only looks at the
     * last character; this variant needs its own rule. */
    lzg_flat_fix_special_nodes(g);

    if (topo_err == LZG_ERR_HAS_CYCLES) {
        g->topo_valid = false;
        return LZG_OK;
    }
    return topo_err;
}

static LZGError flat_finalize(LZGGraph *g, LZGBuildResources *res,
                              uint32_t max_len) {
    uint64_t *len_counts = res->len_counts;
    res->len_counts = NULL;
    LZGError err = flat_pack_into_graph(g, res->edge_builder,
                                        res->build_nodes, len_counts, max_len);
    lzg_eb_destroy(res->edge_builder);
    lzg_node_builder_destroy(res->build_nodes);
    res->edge_builder = NULL;
    res->build_nodes = NULL;
    if (err == LZG_OK)
        LZG_INFO("flat flashback graph ready: %u nodes, %u edges, root=%u",
                 g->n_nodes, g->n_edges, g->root_node);
    return err;
}

static LZGError flat_resources_init(LZGBuildResources *res, uint32_t eb_cap) {
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

LZGError lzg_flat_graph_build(LZGGraph *g,
                              const char **sequences,
                              uint32_t n_seqs,
                              const uint64_t *abundances,
                              uint32_t max_length,
                              double smoothing) {
    if (!g || !sequences || n_seqs == 0) return LZG_ERR_INVALID_ARG;
    g->smoothing_alpha = smoothing;
    LZG_INFO("flat flashback graph: building from %u sequences", n_seqs);

    LZGBuildResources res = {0};
    LZGError err = flat_resources_init(
        &res, flat_bounded_cap(n_seqs, 8u, 256u, LZG_BUILD_INIT_CAP_MAX));
    if (err != LZG_OK) return err;

    uint32_t max_len = 0;
    uint64_t skipped = 0;
    for (uint32_t s = 0; s < n_seqs; s++) {
        if (!sequences[s]) continue;
        uint64_t count = abundances ? abundances[s] : 1;
        if (count == 0) continue;
        err = flat_accumulate(g, &res, sequences[s], count, max_length,
                              &max_len, &skipped);
        if (err != LZG_OK) { lzg_build_resources_destroy(&res); return err; }
    }
    if (skipped > 0)
        LZG_INFO("flat flashback: skipped %llu sequences over the cap",
                 (unsigned long long)skipped);
    if (res.build_nodes->count == 0) {
        lzg_build_resources_destroy(&res);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "flat flashback: no sequence survived filtering");
    }
    err = flat_finalize(g, &res, max_len);
    if (err != LZG_OK) lzg_build_resources_destroy(&res);
    return err;
}

LZGError lzg_flat_graph_build_file(LZGGraph *g, const char *path,
                                   uint32_t max_length, double smoothing) {
    if (!g || !path || path[0] == '\0') return LZG_ERR_INVALID_ARG;
    FILE *fh = fopen(path, "r");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN,
                             "flat flashback: could not open '%s'", path);
    g->smoothing_alpha = smoothing;

    LZGStreamBuildStats stats = {0};
    stats.file_size_bytes = lzg_detect_regular_file_size(path);
    stats.start_time = lzg_build_monotonic_seconds();
    stats.last_log_time = stats.start_time;
    stats.peak_rss_kb = lzg_build_current_rss_kb();
    LZG_INFO("flat flashback stream build: start file=%s size=%.1fMB",
             path, (double)stats.file_size_bytes / (1024.0 * 1024.0));

    LZGBuildResources res = {0};
    LZGError err = flat_resources_init(&res, 256);
    if (err != LZG_OK) { fclose(fh); return err; }

    uint32_t max_len = 0;
    uint64_t skipped = 0, lines_seen = 0, sequences_seen = 0;
    char *line = NULL;
    size_t line_cap = 0;
    ptrdiff_t nread;
    errno = 0;

    while ((nread = lzg_getline_portable(&line, &line_cap, fh)) != -1) {
        lines_seen++;
        stats.bytes_seen += (uint64_t)nread;
        char *seq = NULL;
        uint64_t count = 0;
        LZGParsedLineKind kind = LZG_LINE_EMPTY;
        err = lzg_parse_plain_sequence_line(line, &seq, &count, &kind);
        if (err != LZG_OK) {
            free(line); fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_update_stream_mode(&stats, kind, path, lines_seen);
        if (!seq || count == 0) continue;
        sequences_seen++;
        err = flat_accumulate(g, &res, seq, count, max_length,
                              &max_len, &skipped);
        if (err != LZG_OK) {
            free(line); fclose(fh);
            lzg_build_resources_destroy(&res);
            return err;
        }
        lzg_maybe_log_stream_progress(path, lines_seen, sequences_seen,
                                      &res, &stats);
    }

    if (ferror(fh)) {
        free(line); fclose(fh);
        lzg_build_resources_destroy(&res);
        return LZG_FAIL(LZG_ERR_IO_READ,
                        "flat flashback stream failed reading '%s'", path);
    }
    free(line);
    fclose(fh);
    LZG_INFO("flat flashback stream: %llu lines, %llu sequences, %llu skipped",
             (unsigned long long)lines_seen,
             (unsigned long long)sequences_seen,
             (unsigned long long)skipped);

    if (res.build_nodes->count == 0) {
        lzg_build_resources_destroy(&res);
        return LZG_FAIL(LZG_ERR_EMPTY_INPUT,
                        "flat flashback: nothing in '%s' survived", path);
    }
    err = flat_finalize(g, &res, max_len);
    if (err != LZG_OK) lzg_build_resources_destroy(&res);
    return err;
}

void lzg_flat_fix_special_nodes(LZGGraph *g) {
    if (!g || g->n_nodes == 0) return;
    g->root_node = UINT32_MAX;
    if (g->node_is_sink)
        memset(g->node_is_sink, 0, g->n_nodes * sizeof(uint8_t));
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        if (sp[0] == LZG_START_SENTINEL && sp[1] == '\0')
            g->root_node = i;
        /* Only the BARE "$" is the sink. "S$_4" ends a sequence's residues
         * but is a normal node with an edge on to the sink. */
        if (sp[0] == LZG_END_SENTINEL && sp[1] == '\0' && g->node_is_sink)
            g->node_is_sink[i] = 1;
    }
}

/* ── Simulation and probability ───────────────────────────── */

LZGError lzg_flat_simulate(const LZGGraph *g, uint32_t n,
                           LZGRng *rng, LZGSimResult *out) {
    if (!g || !rng || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flat simulate: NULL argument");
    if (g->root_node >= g->n_nodes)
        return LZG_FAIL(LZG_ERR_NOT_BUILT, "flat simulate: no root node");

    for (uint32_t s = 0; s < n; s++) {
        uint32_t walk[LZG_FLAT_MAX_WALK];
        uint32_t walk_len = 0;
        uint32_t cur = g->root_node;
        double log_prob = 0.0;

        while (walk_len < LZG_FLAT_MAX_WALK) {
            walk[walk_len++] = g->node_sp_id[cur];
            if (g->node_is_sink && g->node_is_sink[cur]) break;
            uint32_t e0 = g->row_offsets[cur], e1 = g->row_offsets[cur + 1];
            if (e0 == e1) break;
            double u = lzg_rng_double(rng), cumul = 0.0;
            uint32_t chosen = e0;
            for (uint32_t e = e0; e < e1; e++) {
                cumul += g->edge_weights[e];
                if (u < cumul) { chosen = e; break; }
                chosen = e;
            }
            double w = g->edge_weights[chosen];
            log_prob = (w > LZG_EPS) ? log_prob + log(w) : LZG_LOG_EPS;
            cur = g->col_indices[chosen];
        }

        char buf[LZG_FLAT_MAX_WALK];
        uint32_t seq_len = 0;
        if (lzg_flat_reverse(g->pool, walk, walk_len,
                             buf, sizeof(buf), &seq_len) != LZG_OK) {
            buf[0] = '\0';
            seq_len = 0;
        }
        out[s].sequence = strdup(buf);
        out[s].seq_len = seq_len;
        out[s].n_tokens = walk_len;
        out[s].log_prob = log_prob;
    }
    return LZG_OK;
}

static LZGHashMap *flat_query_map(const LZGGraph *g) {
    LZGGraph *gm = (LZGGraph *)g;
    if (gm->query_node_map) return gm->query_node_map;
    LZGHashMap *map = lzg_hm_create(g->n_nodes * 2);
    if (!map) return NULL;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint64_t key = ((uint64_t)g->node_sp_id[i] << 32) |
                       (uint64_t)g->node_pos[i];
        lzg_hm_put(map, key, (uint64_t)i);
    }
    gm->query_node_map = map;
    return map;
}

static uint32_t flat_lookup(const LZGGraph *g, LZGHashMap *map,
                            const char *label) {
    uint32_t sp_id = lzg_sp_find(g->pool, label);
    if (sp_id == LZG_SP_NOT_FOUND) return UINT32_MAX;
    uint64_t key = ((uint64_t)sp_id << 32) | (uint64_t)UINT32_MAX;
    uint64_t *slot = lzg_hm_get(map, key);
    return slot ? (uint32_t)*slot : UINT32_MAX;
}

static double flat_edge_weight(const LZGGraph *g, uint32_t u, uint32_t v) {
    for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++)
        if (g->col_indices[e] == v) return g->edge_weights[e];
    return 0.0;
}

double lzg_flat_pseq(const LZGGraph *g, const char *seq, uint32_t seq_len) {
    if (!g || !seq || seq_len == 0) return LZG_LOG_EPS;
    if (g->root_node >= g->n_nodes) return LZG_LOG_EPS;
    if ((seq_len + 1u) / 2u + 2u > LZG_FLAT_MAX_WALK) return LZG_LOG_EPS;

    LZGHashMap *map = flat_query_map(g);
    if (!map) return LZG_LOG_EPS;

    double log_p = 0.0;
    uint32_t prev = g->root_node;
    char label[LZG_FLAT_LABEL_CAP];
    uint32_t i = 0, j = seq_len - 1, step = 1;

    while (i < j) {
        if (snprintf(label, sizeof(label), "%c%c_%u",
                     seq[i], seq[j], step) >= (int)sizeof(label))
            return LZG_LOG_EPS;
        uint32_t nid = flat_lookup(g, map, label);
        if (nid == UINT32_MAX) return LZG_LOG_EPS;
        double w = flat_edge_weight(g, prev, nid);
        if (w < LZG_EPS) return LZG_LOG_EPS;
        log_p += log(w);
        prev = nid;
        i++; j--; step++;
    }
    if (i == j) {
        if (snprintf(label, sizeof(label), "%c%c_%u",
                     seq[i], LZG_END_SENTINEL, step) >= (int)sizeof(label))
            return LZG_LOG_EPS;
        uint32_t nid = flat_lookup(g, map, label);
        if (nid == UINT32_MAX) return LZG_LOG_EPS;
        double w = flat_edge_weight(g, prev, nid);
        if (w < LZG_EPS) return LZG_LOG_EPS;
        log_p += log(w);
        prev = nid;
    }

    {
        static const char end_label[2] = {LZG_END_SENTINEL, '\0'};
        uint32_t sink = flat_lookup(g, map, end_label);
        if (sink == UINT32_MAX) return LZG_LOG_EPS;
        double w = flat_edge_weight(g, prev, sink);
        if (w < LZG_EPS) return LZG_LOG_EPS;
        log_p += log(w);
    }
    return log_p > LZG_LOG_EPS ? log_p : LZG_LOG_EPS;
}

LZGError lzg_flat_pseq_batch(const LZGGraph *g, const char **sequences,
                             uint32_t n, double *out) {
    if (!g || !sequences || !out) return LZG_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < n; i++)
        out[i] = lzg_flat_pseq(g, sequences[i],
                               (uint32_t)strlen(sequences[i]));
    return LZG_OK;
}

/* ── Exact DP analytics (generic over the CSR; see naive/graph_analytics.c) ── */

LZGError lzg_flat_path_count(const LZGGraph *g, double *out) {
    return lzg_flashback_path_count(g, out);
}
LZGError lzg_flat_path_count_exact(const LZGGraph *g, uint32_t **limbs,
                                   uint32_t *n_limbs) {
    return lzg_flashback_path_count_exact(g, limbs, n_limbs);
}
LZGError lzg_flat_effective_diversity(const LZGGraph *g,
                                      LZGEffectiveDiversity *out) {
    return lzg_flashback_effective_diversity(g, out);
}
LZGError lzg_flat_power_sum(const LZGGraph *g, double alpha, double *out) {
    return lzg_flashback_power_sum(g, alpha, out);
}
LZGError lzg_flat_hill_number(const LZGGraph *g, double alpha, double *out) {
    return lzg_flashback_hill_number(g, alpha, out);
}
LZGError lzg_flat_hill_numbers(const LZGGraph *g, const double *orders,
                               uint32_t n, double *out) {
    return lzg_flashback_hill_numbers(g, orders, n, out);
}
LZGError lzg_flat_dynamic_range(const LZGGraph *g, LZGDynamicRange *out) {
    return lzg_flashback_dynamic_range(g, out);
}
LZGError lzg_flat_pseq_diagnostics(const LZGGraph *g, double atol,
                                   LZGPgenDiagnostics *out) {
    return lzg_flashback_pgen_diagnostics(g, atol, out);
}
