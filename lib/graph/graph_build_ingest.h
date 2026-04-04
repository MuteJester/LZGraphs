#ifndef LZGRAPH_GRAPH_BUILD_INGEST_H
#define LZGRAPH_GRAPH_BUILD_INGEST_H

#include "lzgraph/graph.h"
#include <stddef.h>
#include <stdio.h>

#define LZG_STREAM_PROGRESS_EVERY 1000000ULL
#define LZG_STREAM_PROGRESS_MIN_SEC 5.0

typedef struct {
    LZGHashMap *key_to_id;
    uint32_t   *sp_ids;
    uint32_t   *positions;
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

typedef struct {
    LZGEdgeBuilder *edge_builder;
    LZGNodeBuilder *build_nodes;
    LZGStringPool  *gene_pool;
    LZGHashMap     *v_marginal_counts;
    LZGHashMap     *j_marginal_counts;
    LZGHashMap     *vj_pair_counts;
    LZGHashMap     *edge_v_genes;
    LZGHashMap     *edge_j_genes;
    LZGHashMap     *initial_counts;
    LZGHashMap     *terminal_counts;
    LZGHashMap     *outgoing_counts;
    uint64_t       *len_counts;
    uint32_t        len_cap;
} LZGBuildResources;

LZGNodeBuilder *lzg_node_builder_create(uint32_t initial_capacity);

void lzg_node_builder_destroy(LZGNodeBuilder *nb);

void lzg_build_resources_destroy(LZGBuildResources *res);

LZGError lzg_accumulate_sequence_record(LZGGraph *g,
                                        LZGBuildResources *res,
                                        const char *seq,
                                        uint64_t count,
                                        const char *v_gene,
                                        const char *j_gene,
                                        uint32_t *max_len);

double lzg_build_monotonic_seconds(void);

const char *lzg_build_variant_name(LZGVariant variant);

long long lzg_build_current_rss_kb(void);

const char *lzg_build_stream_mode_name(LZGStreamInputMode mode);

uint64_t lzg_detect_regular_file_size(const char *path);

ptrdiff_t lzg_getline_portable(char **lineptr, size_t *cap, FILE *fh);

void lzg_build_format_duration(double seconds, char *buf, size_t cap);

void lzg_update_stream_mode(LZGStreamBuildStats *stats,
                            LZGParsedLineKind kind,
                            const char *path,
                            uint64_t line_no);

void lzg_maybe_log_stream_progress(const char *path,
                                   uint64_t lines_seen,
                                   uint64_t sequences_seen,
                                   const LZGBuildResources *res,
                                   LZGStreamBuildStats *stats);

LZGError lzg_parse_plain_sequence_line(char *line,
                                       char **out_seq,
                                       uint64_t *out_count,
                                       LZGParsedLineKind *out_kind);

#endif /* LZGRAPH_GRAPH_BUILD_INGEST_H */
