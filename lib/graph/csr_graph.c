#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif
/**
 * @file csr_graph.c
 * @brief Graph construction: sequences → LZ76 → EdgeBuilder → CSR.
 */
#include "lzgraph/graph.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/lz76.h"
#include "lzgraph/edge_builder.h"
#include "lzgraph/hash_map.h"
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <unistd.h>
#endif

#define LZG_BUILD_INIT_CAP_MAX (1u << 20)
#define LZG_STREAM_PROGRESS_EVERY 1000000ULL
#define LZG_STREAM_PROGRESS_MIN_SEC 5.0
#define LZG_WRAP_STACK_CAP 512u

typedef struct {
    LZGHashMap *key_to_id;   /* key: pack(sp_id, pos) -> build node id */
    uint32_t   *sp_ids;      /* [count] subpattern ids                  */
    uint32_t   *positions;   /* [count] cumulative positions            */
    uint32_t    count;
    uint32_t    capacity;
} LZGNodeBuilder;

typedef enum {
    LZG_STREAM_MODE_UNKNOWN = 0,
    LZG_STREAM_MODE_PLAIN = 1,
    LZG_STREAM_MODE_SEQCOUNT = 2,
    LZG_STREAM_MODE_MIXED = 3,
} LZGStreamInputMode;

typedef enum {
    LZG_LINE_EMPTY = 0,
    LZG_LINE_PLAIN = 1,
    LZG_LINE_SEQCOUNT = 2,
} LZGParsedLineKind;

typedef struct {
    uint64_t file_size_bytes;
    uint64_t bytes_seen;
    uint64_t blank_lines;
    uint64_t plain_records;
    uint64_t seqcount_records;
    uint64_t last_lines_seen;
    uint64_t last_bytes_seen;
    uint32_t last_nodes_seen;
    uint32_t last_edges_seen;
    double start_time;
    double last_log_time;
    long long peak_rss_kb;
    LZGStreamInputMode mode;
    bool warned_mixed_mode;
} LZGStreamBuildStats;

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

static uint32_t bounded_capacity_hint(uint32_t n_items,
                                      uint32_t multiplier,
                                      uint32_t min_cap,
                                      uint32_t max_cap) {
    uint64_t estimate = (uint64_t)n_items * (uint64_t)multiplier;
    if (estimate < (uint64_t)min_cap) return min_cap;
    if (estimate > (uint64_t)max_cap) return max_cap;
    return (uint32_t)estimate;
}

static inline uint64_t pack_build_node_key(uint32_t sp_id, uint32_t pos) {
    return ((uint64_t)sp_id << 32) | (uint64_t)pos;
}

static LZGNodeBuilder *node_builder_create(uint32_t initial_capacity) {
    if (initial_capacity < 256) initial_capacity = 256;
    LZGNodeBuilder *nb = calloc(1, sizeof(LZGNodeBuilder));
    if (!nb) return NULL;
    nb->key_to_id = lzg_hm_create(initial_capacity * 2);
    nb->sp_ids = malloc(initial_capacity * sizeof(uint32_t));
    nb->positions = malloc(initial_capacity * sizeof(uint32_t));
    nb->capacity = initial_capacity;
    if (!nb->key_to_id || !nb->sp_ids || !nb->positions) {
        lzg_hm_destroy(nb->key_to_id);
        free(nb->sp_ids);
        free(nb->positions);
        free(nb);
        return NULL;
    }
    return nb;
}

static void node_builder_destroy(LZGNodeBuilder *nb) {
    if (!nb) return;
    lzg_hm_destroy(nb->key_to_id);
    free(nb->sp_ids);
    free(nb->positions);
    free(nb);
}

static LZGError node_builder_intern(LZGNodeBuilder *nb,
                                    uint32_t sp_id,
                                    uint32_t pos,
                                    uint32_t *out_id) {
    if (nb->count >= nb->capacity) {
        uint32_t new_cap = nb->capacity * 2;
        uint32_t *new_sp_ids = realloc(nb->sp_ids, new_cap * sizeof(uint32_t));
        if (!new_sp_ids) return LZG_ERR_ALLOC;
        uint32_t *new_positions = realloc(nb->positions, new_cap * sizeof(uint32_t));
        if (!new_positions) {
            nb->sp_ids = new_sp_ids;
            return LZG_ERR_ALLOC;
        }
        nb->sp_ids = new_sp_ids;
        nb->positions = new_positions;
        nb->capacity = new_cap;
    }

    bool inserted = false;
    uint64_t key = pack_build_node_key(sp_id, pos);
    uint64_t *slot = lzg_hm_get_or_insert(nb->key_to_id, key,
                                          (uint64_t)nb->count,
                                          &inserted);
    uint32_t id = (uint32_t)*slot;
    if (inserted) {
        nb->sp_ids[id] = sp_id;
        nb->positions[id] = pos;
        nb->count++;
    }
    *out_id = id;
    return LZG_OK;
}

static char *wrap_sentinels_local(const char *str, uint32_t len, uint32_t *out_len,
                                  char *stack_buf, size_t stack_cap, bool *used_heap) {
    *out_len = len + 2;
    size_t need = (size_t)(*out_len) + 1u;
    char *wrapped = NULL;
    if (need <= stack_cap) {
        wrapped = stack_buf;
        *used_heap = false;
    } else {
        wrapped = malloc(need);
        *used_heap = true;
    }
    if (!wrapped) return NULL;
    wrapped[0] = LZG_START_SENTINEL;
    memcpy(wrapped + 1, str, len);
    wrapped[len + 1] = LZG_END_SENTINEL;
    wrapped[len + 2] = '\0';
    return wrapped;
}

static LZGError encode_sequence_structural(const LZGGraph *g,
                                           LZGNodeBuilder *build_nodes,
                                           const char *seq,
                                           uint32_t seq_len,
                                           uint32_t *out_node_ids,
                                           uint32_t *out_count) {
    uint32_t wlen = 0;
    char wrapped_stack[LZG_WRAP_STACK_CAP];
    bool wrapped_heap = false;
    char *wrapped = wrap_sentinels_local(seq, seq_len, &wlen,
                                         wrapped_stack, sizeof(wrapped_stack),
                                         &wrapped_heap);
    if (!wrapped) return LZG_ERR_ALLOC;

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(wrapped, wlen, g->pool, &tokens);
    if (wrapped_heap) free(wrapped);
    if (err != LZG_OK) return err;

    for (uint32_t i = 0; i < tokens.count; i++) {
        uint32_t pos = (g->variant == LZG_VARIANT_NAIVE) ? UINT32_MAX : tokens.positions[i];
        err = node_builder_intern(build_nodes, tokens.sp_ids[i], pos, &out_node_ids[i]);
        if (err != LZG_OK) return err;
    }
    *out_count = tokens.count;
    return LZG_OK;
}

static double monotonic_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
    return 0.0;
}

static const char *variant_name(LZGVariant variant) {
    switch (variant) {
        case LZG_VARIANT_AAP: return "aap";
        case LZG_VARIANT_NDP: return "ndp";
        case LZG_VARIANT_NAIVE: return "naive";
        default: return "unknown";
    }
}

static long long current_rss_kb(void) {
#if defined(__linux__)
    FILE *fh = fopen("/proc/self/statm", "r");
    if (!fh) return -1;
    unsigned long total_pages = 0, resident_pages = 0;
    int ok = fscanf(fh, "%lu %lu", &total_pages, &resident_pages);
    fclose(fh);
    if (ok != 2) return -1;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    return (long long)resident_pages * (long long)page_size / 1024LL;
#else
    return -1;
#endif
}

static const char *stream_mode_name(LZGStreamInputMode mode) {
    switch (mode) {
        case LZG_STREAM_MODE_PLAIN: return "plain";
        case LZG_STREAM_MODE_SEQCOUNT: return "plain_seqcount";
        case LZG_STREAM_MODE_MIXED: return "mixed";
        default: return "pending";
    }
}

static uint64_t detect_regular_file_size(const char *path) {
    struct stat st;
    if (!path || stat(path, &st) != 0) return 0;
#if defined(_WIN32)
    if ((st.st_mode & _S_IFREG) == 0 || st.st_size <= 0) return 0;
#else
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) return 0;
#endif
    return (uint64_t)st.st_size;
}

static ptrdiff_t lzg_getline_portable(char **lineptr, size_t *cap, FILE *fh) {
    if (!lineptr || !cap || !fh) {
        errno = EINVAL;
        return -1;
    }

    if (!*lineptr || *cap == 0) {
        *cap = 256;
        *lineptr = malloc(*cap);
        if (!*lineptr) {
            errno = ENOMEM;
            return -1;
        }
    }

    size_t len = 0;
    int ch = 0;
    while ((ch = fgetc(fh)) != EOF) {
        if (len + 2 > *cap) {
            size_t new_cap = (*cap < (SIZE_MAX / 2)) ? (*cap * 2) : 0;
            if (new_cap < len + 2) new_cap = len + 2;
            if (new_cap == 0) {
                errno = ERANGE;
                return -1;
            }
            char *tmp = realloc(*lineptr, new_cap);
            if (!tmp) {
                errno = ENOMEM;
                return -1;
            }
            *lineptr = tmp;
            *cap = new_cap;
        }
        (*lineptr)[len++] = (char)ch;
        if (ch == '\n') break;
    }

    if (len == 0 && ch == EOF) {
        errno = 0;
        return -1;
    }
    (*lineptr)[len] = '\0';
    errno = 0;
    return (ptrdiff_t)len;
}

static void format_duration(double seconds, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    if (seconds < 0.0 || !isfinite(seconds)) {
        snprintf(buf, cap, "unknown");
        return;
    }
    unsigned long long total = (unsigned long long)(seconds + 0.5);
    unsigned long long h = total / 3600ULL;
    unsigned long long m = (total % 3600ULL) / 60ULL;
    unsigned long long s = total % 60ULL;
    snprintf(buf, cap, "%02llu:%02llu:%02llu", h, m, s);
}

static void update_stream_mode(LZGStreamBuildStats *stats,
                               LZGParsedLineKind kind,
                               const char *path,
                               uint64_t line_no) {
    if (!stats) return;
    if (kind == LZG_LINE_EMPTY) {
        stats->blank_lines++;
        return;
    }

    LZGStreamInputMode current =
        (kind == LZG_LINE_SEQCOUNT) ? LZG_STREAM_MODE_SEQCOUNT : LZG_STREAM_MODE_PLAIN;
    if (current == LZG_STREAM_MODE_SEQCOUNT) stats->seqcount_records++;
    else stats->plain_records++;

    if (stats->mode == LZG_STREAM_MODE_UNKNOWN) {
        stats->mode = current;
        return;
    }
    if (stats->mode == current || stats->mode == LZG_STREAM_MODE_MIXED) return;

    if (!stats->warned_mixed_mode) {
        LZG_WARN("stream build: warning phase=ingest file=%s line=%llu issue=mixed_input_format previous_mode=%s current_record=%s continuing=true",
                 path,
                 (unsigned long long)line_no,
                 stream_mode_name(stats->mode),
                 stream_mode_name(current));
        stats->warned_mixed_mode = true;
    }
    stats->mode = LZG_STREAM_MODE_MIXED;
}

static void maybe_log_stream_progress(const char *path,
                                      uint64_t lines_seen,
                                      uint64_t sequences_seen,
                                      uint32_t node_count,
                                      const LZGEdgeBuilder *eb,
                                      LZGStreamBuildStats *stats) {
    if (!stats || lines_seen == 0) return;
    bool hit_line_checkpoint = (lines_seen % LZG_STREAM_PROGRESS_EVERY) == 0;

    double now = monotonic_seconds();
    bool hit_time_checkpoint = (stats->last_log_time > 0.0) &&
                               (now - stats->last_log_time >= LZG_STREAM_PROGRESS_MIN_SEC);
    if (!hit_line_checkpoint && !hit_time_checkpoint) return;

    double elapsed = (stats->start_time > 0.0 && now > stats->start_time) ? (now - stats->start_time) : 0.0;
    double window_elapsed = (stats->last_log_time > 0.0 && now > stats->last_log_time)
        ? (now - stats->last_log_time) : elapsed;
    double rate = elapsed > 0.0 ? (double)lines_seen / elapsed : 0.0;
    uint64_t line_delta = lines_seen - stats->last_lines_seen;
    uint64_t byte_delta = stats->bytes_seen - stats->last_bytes_seen;
    uint32_t node_delta = node_count - stats->last_nodes_seen;
    uint32_t edge_count = eb ? eb->n_edges : 0u;
    uint32_t edge_delta = edge_count - stats->last_edges_seen;
    double inst_rate = window_elapsed > 0.0 ? (double)line_delta / window_elapsed : rate;
    double avg_mbps = elapsed > 0.0 ? ((double)stats->bytes_seen / (1024.0 * 1024.0)) / elapsed : 0.0;
    double inst_mbps = window_elapsed > 0.0 ? ((double)byte_delta / (1024.0 * 1024.0)) / window_elapsed : avg_mbps;
    long long rss_kb = current_rss_kb();
    if (rss_kb > stats->peak_rss_kb) stats->peak_rss_kb = rss_kb;
    double pct = 0.0;
    double eta_sec = -1.0;
    if (stats->file_size_bytes > 0) {
        pct = 100.0 * (double)stats->bytes_seen / (double)stats->file_size_bytes;
        double bytes_per_sec = window_elapsed > 0.0 ? (double)byte_delta / window_elapsed : 0.0;
        if (bytes_per_sec <= 0.0 && elapsed > 0.0)
            bytes_per_sec = (double)stats->bytes_seen / elapsed;
        if (bytes_per_sec > 0.0 && stats->bytes_seen < stats->file_size_bytes)
            eta_sec = (double)(stats->file_size_bytes - stats->bytes_seen) / bytes_per_sec;
    }
    char eta_buf[32];
    format_duration(eta_sec, eta_buf, sizeof(eta_buf));

    if (rss_kb >= 0) {
        if (stats->file_size_bytes > 0) {
            LZG_INFO("stream build: phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu pct=%.2f bytes=%.1f/%.1fMB nodes=%u edges=%u d_nodes=%u d_edges=%u rss=%.1fMB peak_rss=%.1fMB rate=%.0f inst_rate=%.0f avg_mbps=%.1f inst_mbps=%.1f eta=%s",
                     path,
                     stream_mode_name(stats->mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats->blank_lines,
                     pct,
                     (double)stats->bytes_seen / (1024.0 * 1024.0),
                     (double)stats->file_size_bytes / (1024.0 * 1024.0),
                     node_count,
                     edge_count,
                     node_delta,
                     edge_delta,
                     (double)rss_kb / 1024.0,
                     stats->peak_rss_kb >= 0 ? (double)stats->peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0,
                     rate,
                     inst_rate,
                     avg_mbps,
                     inst_mbps,
                     eta_buf);
        } else {
            LZG_INFO("stream build: phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu bytes=%.1fMB nodes=%u edges=%u d_nodes=%u d_edges=%u rss=%.1fMB peak_rss=%.1fMB rate=%.0f inst_rate=%.0f avg_mbps=%.1f inst_mbps=%.1f eta=%s",
                     path,
                     stream_mode_name(stats->mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats->blank_lines,
                     (double)stats->bytes_seen / (1024.0 * 1024.0),
                     node_count,
                     edge_count,
                     node_delta,
                     edge_delta,
                     (double)rss_kb / 1024.0,
                     stats->peak_rss_kb >= 0 ? (double)stats->peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0,
                     rate,
                     inst_rate,
                     avg_mbps,
                     inst_mbps,
                     eta_buf);
        }
    } else if (stats->file_size_bytes > 0) {
        LZG_INFO("stream build: phase=ingest_progress file=%s mode=%s lines=%llu sequences=%llu blank=%llu pct=%.2f bytes=%.1f/%.1fMB nodes=%u edges=%u d_nodes=%u d_edges=%u rate=%.0f inst_rate=%.0f avg_mbps=%.1f inst_mbps=%.1f eta=%s",
                 path,
                 stream_mode_name(stats->mode),
                 (unsigned long long)lines_seen,
                 (unsigned long long)sequences_seen,
                 (unsigned long long)stats->blank_lines,
                 pct,
                 (double)stats->bytes_seen / (1024.0 * 1024.0),
                 (double)stats->file_size_bytes / (1024.0 * 1024.0),
                 node_count,
                 edge_count,
                 node_delta,
                 edge_delta,
                 rate,
                 inst_rate,
                 avg_mbps,
                 inst_mbps,
                 eta_buf);
    } else {
        LZG_INFO("stream build: phase=ingest_progress file=%s mode=%s lines=%llu sequences=%llu blank=%llu bytes=%.1fMB nodes=%u edges=%u d_nodes=%u d_edges=%u rate=%.0f inst_rate=%.0f avg_mbps=%.1f inst_mbps=%.1f eta=%s",
                 path,
                 stream_mode_name(stats->mode),
                 (unsigned long long)lines_seen,
                 (unsigned long long)sequences_seen,
                 (unsigned long long)stats->blank_lines,
                 (double)stats->bytes_seen / (1024.0 * 1024.0),
                 node_count,
                 edge_count,
                 node_delta,
                 edge_delta,
                 rate,
                 inst_rate,
                 avg_mbps,
                 inst_mbps,
                 eta_buf);
    }

    stats->last_log_time = now;
    stats->last_lines_seen = lines_seen;
    stats->last_bytes_seen = stats->bytes_seen;
    stats->last_nodes_seen = node_count;
    stats->last_edges_seen = edge_count;
}

static void destroy_build_state(LZGEdgeBuilder *eb,
                                LZGNodeBuilder *build_nodes,
                                LZGStringPool *gene_pool,
                                LZGHashMap *v_marginal_counts,
                                LZGHashMap *j_marginal_counts,
                                LZGHashMap *vj_pair_counts,
                                LZGHashMap *edge_v_genes,
                                LZGHashMap *edge_j_genes,
                                LZGHashMap *initial_counts,
                                LZGHashMap *terminal_counts,
                                LZGHashMap *outgoing_counts,
                                uint64_t *len_counts) {
    lzg_eb_destroy(eb);
    node_builder_destroy(build_nodes);
    if (gene_pool) lzg_sp_destroy(gene_pool);
    if (v_marginal_counts) lzg_hm_destroy(v_marginal_counts);
    if (j_marginal_counts) lzg_hm_destroy(j_marginal_counts);
    if (vj_pair_counts) lzg_hm_destroy(vj_pair_counts);
    if (edge_v_genes) lzg_hm_destroy(edge_v_genes);
    if (edge_j_genes) lzg_hm_destroy(edge_j_genes);
    lzg_hm_destroy(initial_counts);
    lzg_hm_destroy(terminal_counts);
    lzg_hm_destroy(outgoing_counts);
    free(len_counts);
}

static LZGError grow_length_counts(uint32_t seq_len,
                                   uint64_t **len_counts,
                                   uint32_t *len_cap) {
    if (seq_len < *len_cap) return LZG_OK;

    uint32_t new_cap = seq_len + 64;
    uint64_t *new_counts = realloc(*len_counts, new_cap * sizeof(uint64_t));
    if (!new_counts) return LZG_ERR_ALLOC;
    memset(new_counts + *len_cap, 0, (new_cap - *len_cap) * sizeof(uint64_t));
    *len_counts = new_counts;
    *len_cap = new_cap;
    return LZG_OK;
}

static LZGError accumulate_sequence_record(
    LZGGraph *g,
    LZGNodeBuilder *build_nodes,
    const char *seq,
    uint64_t count,
    bool has_genes,
    const char *v_gene,
    const char *j_gene,
    LZGEdgeBuilder *eb,
    LZGStringPool *gene_pool,
    LZGHashMap *v_marginal_counts,
    LZGHashMap *j_marginal_counts,
    LZGHashMap *vj_pair_counts,
    LZGHashMap *edge_v_genes,
    LZGHashMap *edge_j_genes,
    LZGHashMap *initial_counts,
    LZGHashMap *terminal_counts,
    LZGHashMap *outgoing_counts,
    uint64_t **len_counts,
    uint32_t *len_cap,
    uint32_t *max_len) {
    uint32_t seq_len = (uint32_t)strlen(seq);

    uint32_t node_ids[LZG_MAX_TOKENS];
    uint32_t n_tokens;

    LZGError err = encode_sequence_structural(g, build_nodes, seq, seq_len,
                                              node_ids, &n_tokens);
    if (err != LZG_OK || n_tokens == 0) return LZG_OK;

    if (initial_counts)
        (void)lzg_hm_add_u64(initial_counts, node_ids[0], count, NULL);
    if (terminal_counts)
        (void)lzg_hm_add_u64(terminal_counts, node_ids[n_tokens - 1], count, NULL);

    uint32_t v_gene_id = 0, j_gene_id = 0;
    if (has_genes) {
        v_gene_id = lzg_sp_intern(gene_pool, v_gene);
        j_gene_id = lzg_sp_intern(gene_pool, j_gene);

        (void)lzg_hm_add_u64(v_marginal_counts, v_gene_id, count, NULL);
        (void)lzg_hm_add_u64(j_marginal_counts, j_gene_id, count, NULL);

        uint64_t vj_key = ((uint64_t)v_gene_id << 32) | j_gene_id;
        (void)lzg_hm_add_u64(vj_pair_counts, vj_key, count, NULL);
    }

    for (uint32_t i = 0; i < n_tokens - 1; i++) {
        uint32_t edge_idx = UINT32_MAX;
        err = lzg_eb_record(eb, node_ids[i], node_ids[i + 1], count, &edge_idx);
        if (err != LZG_OK) return err;

        if (outgoing_counts)
            (void)lzg_hm_add_u64(outgoing_counts, node_ids[i], count, NULL);

        if (has_genes) {
            uint64_t vk = ((uint64_t)edge_idx << 32) | v_gene_id;
            uint64_t jk = ((uint64_t)edge_idx << 32) | j_gene_id;
            (void)lzg_hm_add_u64(edge_v_genes, vk, count, NULL);
            (void)lzg_hm_add_u64(edge_j_genes, jk, count, NULL);
        }
    }

    if (outgoing_counts)
        (void)lzg_hm_get_or_insert(outgoing_counts, node_ids[n_tokens - 1], 0, NULL);

    err = grow_length_counts(seq_len, len_counts, len_cap);
    if (err != LZG_OK) return err;
    (*len_counts)[seq_len] += count;
    if (seq_len > *max_len) *max_len = seq_len;

    return LZG_OK;
}

static void trim_eol(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[n - 1] = '\0';
        n--;
    }
}

static LZGError parse_plain_sequence_line(char *line,
                                          char **out_seq,
                                          uint64_t *out_count,
                                          LZGParsedLineKind *out_kind) {
    trim_eol(line);
    if (line[0] == '\0') {
        *out_seq = NULL;
        *out_count = 0;
        if (out_kind) *out_kind = LZG_LINE_EMPTY;
        return LZG_OK;
    }

    char *tab = strchr(line, '\t');
    if (!tab) {
        *out_seq = line;
        *out_count = 1;
        if (out_kind) *out_kind = LZG_LINE_PLAIN;
        return LZG_OK;
    }

    *tab = '\0';
    char *seq = line;
    char *count_str = tab + 1;
    if (*seq == '\0') {
        return LZG_FAIL(LZG_ERR_INVALID_SEQUENCE, "empty sequence in plain text input");
    }
    if (*count_str == '\0') {
        *out_seq = seq;
        *out_count = 1;
        if (out_kind) *out_kind = LZG_LINE_SEQCOUNT;
        return LZG_OK;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long raw = strtoull(count_str, &end, 10);
    if (errno != 0 || end == count_str || *end != '\0') {
        return LZG_FAIL(LZG_ERR_INVALID_ARG,
                        "invalid abundance '%s' in plain text input", count_str);
    }
    if (raw > (unsigned long long)UINT64_MAX) {
        return LZG_FAIL(LZG_ERR_OVERFLOW,
                        "abundance exceeds uint64 limit in plain text input");
    }

    *out_seq = seq;
    *out_count = (uint64_t)raw;
    if (out_kind) *out_kind = LZG_LINE_SEQCOUNT;
    return LZG_OK;
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
    LZGEdgeBuilder *eb = lzg_eb_create(eb_cap);
    if (!eb) return LZG_ERR_ALLOC;

    /* Gene string pool and marginal accumulators */
    LZGStringPool *gene_pool = has_genes ? lzg_sp_create(256) : NULL;
    LZGHashMap *v_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    LZGHashMap *j_marginal_counts = has_genes ? lzg_hm_create(128) : NULL;
    LZGHashMap *vj_pair_counts    = has_genes ? lzg_hm_create(256) : NULL;
    /* Per-edge gene: keyed by pack(edge_idx, gene_id) → count */
    uint32_t edge_gene_cap = bounded_capacity_hint(n_seqs, 4u, 256u, LZG_BUILD_INIT_CAP_MAX);
    LZGHashMap *edge_v_genes = has_genes ? lzg_hm_create(edge_gene_cap) : NULL;
    LZGHashMap *edge_j_genes = has_genes ? lzg_hm_create(edge_gene_cap) : NULL;

    /* Initial/terminal state maps are unused in the sentinel finalization path. */
    LZGHashMap *initial_counts  = NULL;
    LZGHashMap *terminal_counts = NULL;
    LZGHashMap *outgoing_counts = NULL;
    LZGNodeBuilder *build_nodes = node_builder_create(4096);
    uint32_t max_len = 0;

    /* Temporary length count array (grow as needed) */
    uint32_t len_cap = 128;
    uint64_t *len_counts = calloc(len_cap, sizeof(uint64_t));
    if (!build_nodes || !len_counts) {
        destroy_build_state(eb, build_nodes, gene_pool, v_marginal_counts, j_marginal_counts,
                            vj_pair_counts, edge_v_genes, edge_j_genes,
                            initial_counts, terminal_counts, outgoing_counts,
                            len_counts);
        return LZG_ERR_ALLOC;
    }

    /* ── Process each sequence ── */
    for (uint32_t s = 0; s < n_seqs; s++) {
        uint64_t count = abundances ? abundances[s] : 1;
        LZGError err = accumulate_sequence_record(
            g, build_nodes, sequences[s], count,
            has_genes, has_genes ? v_genes[s] : NULL, has_genes ? j_genes[s] : NULL,
            eb, gene_pool, v_marginal_counts, j_marginal_counts, vj_pair_counts,
            edge_v_genes, edge_j_genes, initial_counts, terminal_counts,
            outgoing_counts, &len_counts, &len_cap, &max_len);
        if (err != LZG_OK) {
            destroy_build_state(eb, build_nodes, gene_pool, v_marginal_counts, j_marginal_counts,
                                vj_pair_counts, edge_v_genes, edge_j_genes,
                                initial_counts, terminal_counts, outgoing_counts,
                                len_counts);
            return err;
        }
    }

    /* Delegate to structural finalization pipeline */
    LZGError final_err = finalize_from_structural_edges(
        g, eb, build_nodes, len_counts, max_len,
        has_genes ? gene_pool : NULL,
        v_marginal_counts, j_marginal_counts,
        vj_pair_counts, edge_v_genes, edge_j_genes);
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
    stats.file_size_bytes = detect_regular_file_size(path);
    stats.start_time = monotonic_seconds();
    stats.last_log_time = stats.start_time;
    stats.peak_rss_kb = current_rss_kb();
    LZG_INFO("stream build: start phase=ingest file=%s variant=%s size=%.1fMB progress_every_lines=%llu progress_every_sec=%.1f",
             path,
             variant_name(g->variant),
             (double)stats.file_size_bytes / (1024.0 * 1024.0),
             (unsigned long long)LZG_STREAM_PROGRESS_EVERY,
             LZG_STREAM_PROGRESS_MIN_SEC);

    LZGEdgeBuilder *eb = lzg_eb_create(256);
    if (!eb) { fclose(fh); return LZG_ERR_ALLOC; }

    LZGHashMap *initial_counts  = NULL;
    LZGHashMap *terminal_counts = NULL;
    LZGHashMap *outgoing_counts = NULL;
    LZGNodeBuilder *build_nodes = node_builder_create(4096);
    uint32_t max_len = 0;
    char *line = NULL;
    uint32_t len_cap = 128;
    uint64_t *len_counts = calloc(len_cap, sizeof(uint64_t));
    if (!build_nodes || !len_counts) {
        free(line);
        fclose(fh);
        destroy_build_state(eb, build_nodes, NULL, NULL, NULL, NULL, NULL, NULL,
                            initial_counts, terminal_counts, outgoing_counts,
                            len_counts);
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
        LZGError err = parse_plain_sequence_line(line, &seq, &count, &line_kind);
        if (err != LZG_OK) {
            free(line);
            fclose(fh);
            destroy_build_state(eb, build_nodes, NULL, NULL, NULL, NULL, NULL, NULL,
                                initial_counts, terminal_counts, outgoing_counts,
                                len_counts);
            return err;
        }
        update_stream_mode(&stats, line_kind, path, lines_seen);
        if (!seq || count == 0) continue;
        sequences_seen++;

        err = accumulate_sequence_record(
            g, build_nodes, seq, count, false, NULL, NULL,
            eb, NULL, NULL, NULL, NULL, NULL, NULL,
            initial_counts, terminal_counts, outgoing_counts,
            &len_counts, &len_cap, &max_len);
        if (err != LZG_OK) {
            free(line);
            fclose(fh);
            destroy_build_state(eb, build_nodes, NULL, NULL, NULL, NULL, NULL, NULL,
                                initial_counts, terminal_counts, outgoing_counts,
                                len_counts);
            return err;
        }
        maybe_log_stream_progress(path, lines_seen, sequences_seen,
                                  build_nodes ? build_nodes->count : 0u, eb, &stats);
    }

    if (ferror(fh) || (!feof(fh) && errno != 0)) {
        int saved_errno = errno;
        free(line);
        fclose(fh);
        destroy_build_state(eb, build_nodes, NULL, NULL, NULL, NULL, NULL, NULL,
                            initial_counts, terminal_counts, outgoing_counts,
                            len_counts);
        if (saved_errno == ENOMEM)
            return LZG_FAIL(LZG_ERR_ALLOC, "stream build ran out of memory reading '%s'", path);
        return LZG_FAIL(LZG_ERR_IO_READ, "failed while reading '%s'", path);
    }

    free(line);
    fclose(fh);

    {
        double end_time = monotonic_seconds();
        double elapsed = (stats.start_time > 0.0 && end_time > stats.start_time) ? (end_time - stats.start_time) : 0.0;
        double rate = elapsed > 0.0 ? (double)lines_seen / elapsed : 0.0;
        double mbps = elapsed > 0.0 ? ((double)stats.bytes_seen / (1024.0 * 1024.0)) / elapsed : 0.0;
        long long rss_kb = current_rss_kb();
        if (rss_kb > stats.peak_rss_kb) stats.peak_rss_kb = rss_kb;
        char elapsed_buf[32];
        format_duration(elapsed, elapsed_buf, sizeof(elapsed_buf));
        if (rss_kb >= 0) {
            LZG_INFO("stream build: done phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu plain_records=%llu seqcount_records=%llu bytes=%.1fMB nodes=%u edges=%u rss=%.1fMB peak_rss=%.1fMB elapsed=%s rate=%.0f lines/s avg_mbps=%.1f",
                     path,
                     stream_mode_name(stats.mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats.blank_lines,
                     (unsigned long long)stats.plain_records,
                     (unsigned long long)stats.seqcount_records,
                     (double)stats.bytes_seen / (1024.0 * 1024.0),
                     build_nodes ? build_nodes->count : 0u,
                     eb->n_edges,
                     (double)rss_kb / 1024.0,
                     stats.peak_rss_kb >= 0 ? (double)stats.peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0,
                     elapsed_buf,
                     rate,
                     mbps);
        } else {
            LZG_INFO("stream build: done phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu plain_records=%llu seqcount_records=%llu bytes=%.1fMB nodes=%u edges=%u elapsed=%s rate=%.0f lines/s avg_mbps=%.1f",
                     path,
                     stream_mode_name(stats.mode),
                     (unsigned long long)lines_seen,
                     (unsigned long long)sequences_seen,
                     (unsigned long long)stats.blank_lines,
                     (unsigned long long)stats.plain_records,
                     (unsigned long long)stats.seqcount_records,
                     (double)stats.bytes_seen / (1024.0 * 1024.0),
                     build_nodes ? build_nodes->count : 0u,
                     eb->n_edges,
                     elapsed_buf,
                     rate,
                     mbps);
        }
    }

    {
        double finalize_start = monotonic_seconds();
        long long rss_kb = current_rss_kb();
        if (rss_kb > stats.peak_rss_kb) stats.peak_rss_kb = rss_kb;
        if (rss_kb >= 0) {
            LZG_INFO("stream build: start phase=finalize file=%s nodes=%u edges=%u rss=%.1fMB peak_rss=%.1fMB",
                     path,
                     build_nodes ? build_nodes->count : 0u,
                     eb->n_edges,
                     (double)rss_kb / 1024.0,
                     stats.peak_rss_kb >= 0 ? (double)stats.peak_rss_kb / 1024.0 : (double)rss_kb / 1024.0);
        } else {
            LZG_INFO("stream build: start phase=finalize file=%s nodes=%u edges=%u",
                     path,
                     build_nodes ? build_nodes->count : 0u,
                     eb->n_edges);
        }
        LZGError final_err = finalize_from_structural_edges(
        g, eb, build_nodes, len_counts, max_len,
        NULL, NULL, NULL, NULL, NULL, NULL);
        if (final_err == LZG_OK) {
            double finalize_end = monotonic_seconds();
            double finalize_elapsed = (finalize_end > finalize_start) ? (finalize_end - finalize_start) : 0.0;
            char finalize_buf[32];
            format_duration(finalize_elapsed, finalize_buf, sizeof(finalize_buf));
            rss_kb = current_rss_kb();
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
    double smoothing = g->smoothing_alpha;
    bool has_genes = (gene_pool != NULL);
    uint32_t n_nodes = build_nodes ? build_nodes->count : 0;

    g->n_nodes = n_nodes;
    g->n_edges = eb->n_edges;

    g->row_offsets     = calloc(n_nodes + 1, sizeof(uint32_t));
    g->col_indices     = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_weights    = malloc(eb->n_edges * sizeof(double));
    g->edge_counts     = malloc(eb->n_edges * sizeof(uint64_t));
    g->edge_sp_id      = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_sp_len     = malloc(eb->n_edges * sizeof(uint8_t));
    g->edge_prefix_id  = malloc(eb->n_edges * sizeof(uint32_t));

    g->outgoing_counts = calloc(n_nodes, sizeof(uint64_t));
    g->node_sp_id      = malloc(n_nodes * sizeof(uint32_t));
    g->node_sp_len     = malloc(n_nodes * sizeof(uint8_t));
    g->node_pos        = malloc(n_nodes * sizeof(uint32_t));

    if (!g->row_offsets || !g->col_indices || !g->edge_weights || !g->edge_counts ||
        !g->edge_sp_id || !g->edge_sp_len || !g->edge_prefix_id ||
        !g->outgoing_counts || !g->node_sp_id || !g->node_sp_len || !g->node_pos) {
        destroy_build_state(eb, build_nodes, gene_pool, v_marginal_counts, j_marginal_counts,
                            vj_pair_counts, edge_v_genes, edge_j_genes,
                            NULL, NULL, NULL, len_counts);
        return LZG_ERR_ALLOC;
    }

    uint32_t *edge_deg = calloc(n_nodes, sizeof(uint32_t));
    if (!edge_deg) {
        destroy_build_state(eb, build_nodes, gene_pool, v_marginal_counts, j_marginal_counts,
                            vj_pair_counts, edge_v_genes, edge_j_genes,
                            NULL, NULL, NULL, len_counts);
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

    g->node_is_sink = calloc(n_nodes, sizeof(uint8_t));
    g->root_node = UINT32_MAX;
    if (!g->node_is_sink) {
        free(builder_to_csr);
        destroy_build_state(eb, build_nodes, gene_pool, v_marginal_counts, j_marginal_counts,
                            vj_pair_counts, edge_v_genes, edge_j_genes,
                            NULL, NULL, NULL, len_counts);
        return LZG_ERR_ALLOC;
    }
    for (uint32_t i = 0; i < n_nodes; i++) {
        const char *sp = lzg_sp_get(sp_pool, g->node_sp_id[i]);
        uint8_t sp_len = g->node_sp_len[i];
        if (sp_len == 1 && sp[0] == LZG_START_SENTINEL)
            g->root_node = i;
        if (sp_len > 0 && sp[sp_len - 1] == LZG_END_SENTINEL)
            g->node_is_sink[i] = 1;
    }
    if (g->root_node == UINT32_MAX)
        LZG_WARN("no @ root node found — graph may not have sentinel encoding");

    for (uint32_t i = 0; i < n_nodes; i++) {
        uint32_t start = g->row_offsets[i];
        uint32_t end   = g->row_offsets[i + 1];
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

    for (uint32_t e = 0; e < g->n_edges; e++) {
        uint32_t dst = g->col_indices[e];
        g->edge_sp_id[e]  = g->node_sp_id[dst];
        g->edge_sp_len[e] = g->node_sp_len[dst];
        if (g->node_sp_len[dst] > 1) {
            const char *sp = lzg_sp_get(sp_pool, g->node_sp_id[dst]);
            uint32_t plen = g->node_sp_len[dst] - 1;
            g->edge_prefix_id[e] = lzg_sp_intern_n(sp_pool, sp, plen);
        } else {
            g->edge_prefix_id[e] = UINT32_MAX;
        }
    }
    (void)lzg_graph_ensure_query_edge_hashes(g);

    g->length_counts = len_counts;
    g->max_length = max_len;

    LZGError topo_err = topo_sort_internal(g);

    if (has_genes) {
        LZGGeneData *gd = lzg_gene_data_create();
        lzg_sp_destroy(gd->gene_pool);
        gd->gene_pool = gene_pool;

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

        {
            uint32_t ne = g->n_edges;
            gd->v_offsets = calloc(ne + 1, sizeof(uint32_t));
            gd->j_offsets = calloc(ne + 1, sizeof(uint32_t));

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
            for (uint32_t e = 0; e < ne; e++)
                gd->v_offsets[e + 1] += gd->v_offsets[e];
            gd->total_v_entries = gd->v_offsets[ne];

            gd->v_gene_ids    = malloc(gd->total_v_entries * sizeof(uint32_t));
            gd->v_gene_counts = malloc(gd->total_v_entries * sizeof(uint64_t));
            uint32_t *v_cursor = calloc(ne, sizeof(uint32_t));

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
                    gd->v_gene_counts[pos] = edge_v_genes->values[i];
                    v_cursor[csr_idx]++;
                }
            }
            free(v_cursor);
            gd->total_v_entries = sort_compact_gene_csr(ne, gd->v_offsets,
                                                        gd->v_gene_ids,
                                                        gd->v_gene_counts);

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
            gd->j_gene_counts = malloc(gd->total_j_entries * sizeof(uint64_t));
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
                    gd->j_gene_counts[pos] = edge_j_genes->values[i];
                    j_cursor[csr_idx]++;
                }
            }
            free(j_cursor);
            gd->total_j_entries = sort_compact_gene_csr(ne, gd->j_offsets,
                                                        gd->j_gene_ids,
                                                        gd->j_gene_counts);
        }

        g->gene_data = gd;
        lzg_hm_destroy(v_marginal_counts);
        lzg_hm_destroy(j_marginal_counts);
        lzg_hm_destroy(vj_pair_counts);
        lzg_hm_destroy(edge_v_genes);
        lzg_hm_destroy(edge_j_genes);
    }

    free(builder_to_csr);
    lzg_eb_destroy(eb);
    node_builder_destroy(build_nodes);

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
    g->edge_counts     = malloc(eb->n_edges * sizeof(uint64_t));
    g->edge_sp_id      = malloc(eb->n_edges * sizeof(uint32_t));
    g->edge_sp_len     = malloc(eb->n_edges * sizeof(uint8_t));
    g->edge_prefix_id  = malloc(eb->n_edges * sizeof(uint32_t));

    g->outgoing_counts = calloc(n_nodes, sizeof(uint64_t));
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
        g->outgoing_counts[src_idx] += eb->counts[e];
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
    (void)lzg_graph_ensure_query_edge_hashes(g);

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
            gd->v_gene_counts = malloc(gd->total_v_entries * sizeof(uint64_t));
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
                    gd->v_gene_counts[pos] = edge_v_genes->values[i];
                    v_cursor[csr_idx]++;
                }
            }
            free(v_cursor);
            gd->total_v_entries = sort_compact_gene_csr(ne, gd->v_offsets,
                                                        gd->v_gene_ids,
                                                        gd->v_gene_counts);

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
            gd->j_gene_counts = malloc(gd->total_j_entries * sizeof(uint64_t));
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
                    gd->j_gene_counts[pos] = edge_j_genes->values[i];
                    j_cursor[csr_idx]++;
                }
            }
            free(j_cursor);
            gd->total_j_entries = sort_compact_gene_csr(ne, gd->j_offsets,
                                                        gd->j_gene_ids,
                                                        gd->j_gene_counts);
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
        memset(g->outgoing_counts, 0, nn * sizeof(uint64_t));
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
            uint64_t total   = g->outgoing_counts[u];
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
