#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif

#include "graph_build_ingest.h"
#include "lzgraph/lz76.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <unistd.h>
#endif

#define LZG_WRAP_STACK_CAP 512u

static inline uint64_t pack_build_node_key(uint32_t sp_id, uint32_t pos) {
    return ((uint64_t)sp_id << 32) | (uint64_t)pos;
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

    {
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
    }

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

static LZGError grow_length_counts(uint32_t seq_len,
                                   uint64_t **len_counts,
                                   uint32_t *len_cap) {
    if (seq_len < *len_cap) return LZG_OK;

    {
        uint32_t new_cap = seq_len + 64;
        uint64_t *new_counts = realloc(*len_counts, new_cap * sizeof(uint64_t));
        if (!new_counts) return LZG_ERR_ALLOC;
        memset(new_counts + *len_cap, 0, (new_cap - *len_cap) * sizeof(uint64_t));
        *len_counts = new_counts;
        *len_cap = new_cap;
    }
    return LZG_OK;
}

static void trim_eol(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[n - 1] = '\0';
        n--;
    }
}

LZGNodeBuilder *lzg_node_builder_create(uint32_t initial_capacity) {
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

void lzg_node_builder_destroy(LZGNodeBuilder *nb) {
    if (!nb) return;
    lzg_hm_destroy(nb->key_to_id);
    free(nb->sp_ids);
    free(nb->positions);
    free(nb);
}

void lzg_build_resources_destroy(LZGBuildResources *res) {
    if (!res) return;
    lzg_eb_destroy(res->edge_builder);
    lzg_node_builder_destroy(res->build_nodes);
    if (res->gene_pool) lzg_sp_destroy(res->gene_pool);
    if (res->v_marginal_counts) lzg_hm_destroy(res->v_marginal_counts);
    if (res->j_marginal_counts) lzg_hm_destroy(res->j_marginal_counts);
    if (res->vj_pair_counts) lzg_hm_destroy(res->vj_pair_counts);
    if (res->edge_v_genes) lzg_hm_destroy(res->edge_v_genes);
    if (res->edge_j_genes) lzg_hm_destroy(res->edge_j_genes);
    lzg_hm_destroy(res->initial_counts);
    lzg_hm_destroy(res->terminal_counts);
    lzg_hm_destroy(res->outgoing_counts);
    free(res->len_counts);
}

LZGError lzg_accumulate_sequence_record(LZGGraph *g,
                                        LZGBuildResources *res,
                                        const char *seq,
                                        uint64_t count,
                                        const char *v_gene,
                                        const char *j_gene,
                                        uint32_t *max_len) {
    uint32_t seq_len = (uint32_t)strlen(seq);
    uint32_t node_ids[LZG_MAX_TOKENS];
    uint32_t n_tokens;

    LZGError err = encode_sequence_structural(g, res->build_nodes, seq, seq_len,
                                              node_ids, &n_tokens);
    if (err != LZG_OK) return err;
    if (n_tokens == 0) return LZG_OK;

    if (res->initial_counts)
        (void)lzg_hm_add_u64(res->initial_counts, node_ids[0], count, NULL);
    if (res->terminal_counts)
        (void)lzg_hm_add_u64(res->terminal_counts, node_ids[n_tokens - 1], count, NULL);

    if (res->gene_pool) {
        uint32_t v_gene_id = lzg_sp_intern(res->gene_pool, v_gene);
        uint32_t j_gene_id = lzg_sp_intern(res->gene_pool, j_gene);

        (void)lzg_hm_add_u64(res->v_marginal_counts, v_gene_id, count, NULL);
        (void)lzg_hm_add_u64(res->j_marginal_counts, j_gene_id, count, NULL);

        {
            uint64_t vj_key = ((uint64_t)v_gene_id << 32) | j_gene_id;
            (void)lzg_hm_add_u64(res->vj_pair_counts, vj_key, count, NULL);
        }

        for (uint32_t i = 0; i < n_tokens - 1; i++) {
            uint32_t edge_idx = UINT32_MAX;
            err = lzg_eb_record(res->edge_builder, node_ids[i], node_ids[i + 1], count, &edge_idx);
            if (err != LZG_OK) return err;

            if (res->outgoing_counts)
                (void)lzg_hm_add_u64(res->outgoing_counts, node_ids[i], count, NULL);

            {
                uint64_t vk = ((uint64_t)edge_idx << 32) | v_gene_id;
                uint64_t jk = ((uint64_t)edge_idx << 32) | j_gene_id;
                (void)lzg_hm_add_u64(res->edge_v_genes, vk, count, NULL);
                (void)lzg_hm_add_u64(res->edge_j_genes, jk, count, NULL);
            }
        }
    } else {
        for (uint32_t i = 0; i < n_tokens - 1; i++) {
            uint32_t edge_idx = UINT32_MAX;
            err = lzg_eb_record(res->edge_builder, node_ids[i], node_ids[i + 1], count, &edge_idx);
            if (err != LZG_OK) return err;

            if (res->outgoing_counts)
                (void)lzg_hm_add_u64(res->outgoing_counts, node_ids[i], count, NULL);
        }
    }

    if (res->outgoing_counts)
        (void)lzg_hm_get_or_insert(res->outgoing_counts, node_ids[n_tokens - 1], 0, NULL);

    err = grow_length_counts(seq_len, &res->len_counts, &res->len_cap);
    if (err != LZG_OK) return err;
    res->len_counts[seq_len] += count;
    if (seq_len > *max_len) *max_len = seq_len;

    return LZG_OK;
}

double lzg_build_monotonic_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
    return 0.0;
}

const char *lzg_build_variant_name(LZGVariant variant) {
    switch (variant) {
        case LZG_VARIANT_AAP: return "aap";
        case LZG_VARIANT_NDP: return "ndp";
        case LZG_VARIANT_NAIVE: return "naive";
        default: return "unknown";
    }
}

long long lzg_build_current_rss_kb(void) {
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

const char *lzg_build_stream_mode_name(LZGStreamInputMode mode) {
    switch (mode) {
        case LZG_STREAM_MODE_PLAIN: return "plain";
        case LZG_STREAM_MODE_SEQCOUNT: return "plain_seqcount";
        case LZG_STREAM_MODE_MIXED: return "mixed";
        default: return "pending";
    }
}

uint64_t lzg_detect_regular_file_size(const char *path) {
    struct stat st;
    if (!path || stat(path, &st) != 0) return 0;
#if defined(_WIN32)
    if ((st.st_mode & _S_IFREG) == 0 || st.st_size <= 0) return 0;
#else
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) return 0;
#endif
    return (uint64_t)st.st_size;
}

ptrdiff_t lzg_getline_portable(char **lineptr, size_t *cap, FILE *fh) {
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
            {
                char *tmp = realloc(*lineptr, new_cap);
                if (!tmp) {
                    errno = ENOMEM;
                    return -1;
                }
                *lineptr = tmp;
                *cap = new_cap;
            }
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

void lzg_build_format_duration(double seconds, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    if (seconds < 0.0 || !isfinite(seconds)) {
        snprintf(buf, cap, "unknown");
        return;
    }

    {
        unsigned long long total = (unsigned long long)(seconds + 0.5);
        unsigned long long h = total / 3600ULL;
        unsigned long long m = (total % 3600ULL) / 60ULL;
        unsigned long long s = total % 60ULL;
        snprintf(buf, cap, "%02llu:%02llu:%02llu", h, m, s);
    }
}

void lzg_update_stream_mode(LZGStreamBuildStats *stats,
                            LZGParsedLineKind kind,
                            const char *path,
                            uint64_t line_no) {
    if (!stats) return;
    if (kind == LZG_LINE_EMPTY) {
        stats->blank_lines++;
        return;
    }

    {
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
                     lzg_build_stream_mode_name(stats->mode),
                     lzg_build_stream_mode_name(current));
            stats->warned_mixed_mode = true;
        }
        stats->mode = LZG_STREAM_MODE_MIXED;
    }
}

void lzg_maybe_log_stream_progress(const char *path,
                                   uint64_t lines_seen,
                                   uint64_t sequences_seen,
                                   const LZGBuildResources *res,
                                   LZGStreamBuildStats *stats) {
    if (!stats || !res || lines_seen == 0) return;

    bool hit_line_checkpoint = (lines_seen % LZG_STREAM_PROGRESS_EVERY) == 0;
    double now = lzg_build_monotonic_seconds();
    bool hit_time_checkpoint = (stats->last_log_time > 0.0) &&
                               (now - stats->last_log_time >= LZG_STREAM_PROGRESS_MIN_SEC);
    if (!hit_line_checkpoint && !hit_time_checkpoint) return;

    double elapsed = (stats->start_time > 0.0 && now > stats->start_time) ? (now - stats->start_time) : 0.0;
    double window_elapsed = (stats->last_log_time > 0.0 && now > stats->last_log_time)
        ? (now - stats->last_log_time) : elapsed;
    double rate = elapsed > 0.0 ? (double)lines_seen / elapsed : 0.0;
    uint64_t line_delta = lines_seen - stats->last_lines_seen;
    uint64_t byte_delta = stats->bytes_seen - stats->last_bytes_seen;
    uint32_t node_count = res->build_nodes ? res->build_nodes->count : 0u;
    uint32_t node_delta = node_count - stats->last_nodes_seen;
    uint32_t edge_count = res->edge_builder ? res->edge_builder->n_edges : 0u;
    uint32_t edge_delta = edge_count - stats->last_edges_seen;
    double inst_rate = window_elapsed > 0.0 ? (double)line_delta / window_elapsed : rate;
    double avg_mbps = elapsed > 0.0 ? ((double)stats->bytes_seen / (1024.0 * 1024.0)) / elapsed : 0.0;
    double inst_mbps = window_elapsed > 0.0 ? ((double)byte_delta / (1024.0 * 1024.0)) / window_elapsed : avg_mbps;
    long long rss_kb = lzg_build_current_rss_kb();
    if (rss_kb > stats->peak_rss_kb) stats->peak_rss_kb = rss_kb;
    double pct = 0.0;
    double eta_sec = -1.0;

    if (stats->file_size_bytes > 0) {
        pct = 100.0 * (double)stats->bytes_seen / (double)stats->file_size_bytes;
        {
            double bytes_per_sec = window_elapsed > 0.0 ? (double)byte_delta / window_elapsed : 0.0;
            if (bytes_per_sec <= 0.0 && elapsed > 0.0)
                bytes_per_sec = (double)stats->bytes_seen / elapsed;
            if (bytes_per_sec > 0.0 && stats->bytes_seen < stats->file_size_bytes)
                eta_sec = (double)(stats->file_size_bytes - stats->bytes_seen) / bytes_per_sec;
        }
    }

    {
        char eta_buf[32];
        lzg_build_format_duration(eta_sec, eta_buf, sizeof(eta_buf));

        if (rss_kb >= 0) {
            if (stats->file_size_bytes > 0) {
                LZG_INFO("stream build: phase=ingest file=%s mode=%s lines=%llu sequences=%llu blank=%llu pct=%.2f bytes=%.1f/%.1fMB nodes=%u edges=%u d_nodes=%u d_edges=%u rss=%.1fMB peak_rss=%.1fMB rate=%.0f inst_rate=%.0f avg_mbps=%.1f inst_mbps=%.1f eta=%s",
                         path,
                         lzg_build_stream_mode_name(stats->mode),
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
                         lzg_build_stream_mode_name(stats->mode),
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
                     lzg_build_stream_mode_name(stats->mode),
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
                     lzg_build_stream_mode_name(stats->mode),
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
    }

    stats->last_log_time = now;
    stats->last_lines_seen = lines_seen;
    stats->last_bytes_seen = stats->bytes_seen;
    stats->last_nodes_seen = node_count;
    stats->last_edges_seen = edge_count;
}

LZGError lzg_parse_plain_sequence_line(char *line,
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

    {
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
        if (*seq == '\0')
            return LZG_FAIL(LZG_ERR_INVALID_SEQUENCE, "empty sequence in plain text input");

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
}
