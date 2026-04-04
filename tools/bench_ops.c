#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "lzgraph/analytics.h"
#include "lzgraph/common.h"
#include "lzgraph/diversity.h"
#include "lzgraph/features.h"
#include "lzgraph/graph.h"
#include "lzgraph/graph_ops.h"
#include "lzgraph/io.h"
#include "lzgraph/occupancy.h"
#include "lzgraph/pgen_dist.h"
#include "lzgraph/simulate.h"

#define BENCH_DEFAULT_SECONDS     180.0
#define BENCH_DEFAULT_BATCH_SIZE  128u
#define BENCH_DEFAULT_QUERY_COUNT 256u
#define BENCH_QUERY_BUILD_COUNT   256u
#define BENCH_DEFAULT_RICHNESS_D  1000.0
#define BENCH_DEFAULT_OVERLAP_A   100.0
#define BENCH_DEFAULT_OVERLAP_B   100.0

typedef struct {
    const char *graph_path;
    const char *plain_path;
    const char *variant_name;
    const char *ops_filter;
    const char *save_path;
    const char *json_out;
    double seconds_per_op;
    double smoothing;
    double richness_depth;
    double overlap_a;
    double overlap_b;
    uint32_t batch_size;
    uint32_t query_count;
    bool list_ops;
} BenchOptions;

typedef struct {
    LZGGraph *graph;
    LZGGraph *aux_graph;
    LZGSimResult *query_sims;
    const char **queries;
    const char **batch_queries;
    double *batch_logps;
    LZGSimResult *sim_batch;
    LZGGeneSimResult *gene_batch;
    double *aligned_features;
    double *mass_profile;
    double feature_stats[LZG_FEATURE_STATS_DIM];
    LZGSaturationPoint *saturation_points;
    uint32_t saturation_cap;
    uint32_t batch_size;
    uint32_t query_count;
    uint32_t aux_query_count;
    uint32_t max_pos;
    double seconds_per_op;
    double richness_depth;
    double overlap_a;
    double overlap_b;
    char save_path[4096];
    bool save_path_owned;
    LZGRng rng;
    LZGRng aux_rng;
    volatile double sink_d;
    volatile uint64_t sink_u64;
} BenchContext;

typedef LZGError (*BenchFn)(BenchContext *ctx, uint64_t iter);
typedef double (*BenchItemsFn)(const BenchContext *ctx);

typedef struct {
    const char  *name;
    const char  *unit_label;
    BenchFn      fn;
    BenchItemsFn items_fn;
    bool         needs_queries;
    bool         needs_aux;
    bool         needs_gene_data;
    bool         skip_when_gene_data;
} BenchOp;

typedef struct {
    const char *name;
    const char *unit_label;
    bool skipped;
    const char *skip_reason;
    double seconds;
    uint64_t calls;
    double calls_per_sec;
    double units_per_sec;
    double units_per_180s;
} BenchRunResult;

static double bench_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void bench_free_sim_array(LZGSimResult *results, uint32_t n) {
    if (!results) return;
    for (uint32_t i = 0; i < n; i++)
        lzg_sim_result_free(&results[i]);
}

static void bench_free_gene_sim_array(LZGGeneSimResult *results, uint32_t n) {
    if (!results) return;
    for (uint32_t i = 0; i < n; i++)
        lzg_gene_sim_result_free(&results[i]);
}

static void bench_context_destroy(BenchContext *ctx) {
    if (!ctx) return;
    bench_free_sim_array(ctx->query_sims, ctx->query_count);
    bench_free_sim_array(ctx->sim_batch, ctx->batch_size);
    bench_free_gene_sim_array(ctx->gene_batch, ctx->batch_size);
    free(ctx->query_sims);
    free(ctx->queries);
    free(ctx->batch_queries);
    free(ctx->batch_logps);
    free(ctx->sim_batch);
    free(ctx->gene_batch);
    free(ctx->aligned_features);
    free(ctx->mass_profile);
    free(ctx->saturation_points);
    if (ctx->aux_graph) lzg_graph_destroy(ctx->aux_graph);
    if (ctx->graph) lzg_graph_destroy(ctx->graph);
    if (ctx->save_path_owned && ctx->save_path[0] != '\0')
        unlink(ctx->save_path);
}

static void bench_usage(FILE *stream, const char *argv0) {
    fprintf(stream,
            "Usage:\n"
            "  %s --graph PATH.lzg [options]\n"
            "  %s --plain PATH.txt [--variant aap|ndp|naive] [options]\n\n"
            "Options:\n"
            "  --seconds N        Seconds per operation benchmark (default: %.0f)\n"
            "  --batch N          Batch size for batch ops (default: %u)\n"
            "  --queries N        Number of simulated query sequences to prepare (default: %u)\n"
            "  --richness-depth D Depth for predicted_richness() (default: %.0f)\n"
            "  --overlap-a D      First depth for predicted_overlap() (default: %.0f)\n"
            "  --overlap-b D      Second depth for predicted_overlap() (default: %.0f)\n"
            "  --save-path PATH   Temporary .lzg path for save/load benchmarks\n"
            "  --json-out PATH    Save benchmark results as JSON\n"
            "  --ops a,b,c        Comma-separated benchmark names to run (default: all)\n"
            "  --list-ops         Print available benchmark names and exit\n"
            "  --smoothing X      Smoothing for --plain build (default: 0)\n"
            "  --help             Show this message\n",
            argv0, argv0,
            BENCH_DEFAULT_SECONDS,
            BENCH_DEFAULT_BATCH_SIZE,
            BENCH_DEFAULT_QUERY_COUNT,
            BENCH_DEFAULT_RICHNESS_D,
            BENCH_DEFAULT_OVERLAP_A,
            BENCH_DEFAULT_OVERLAP_B);
}

static bool bench_parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    unsigned long v;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool bench_parse_double(const char *s, double *out) {
    char *end = NULL;
    double v;
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || !end || *end != '\0' || !isfinite(v)) return false;
    *out = v;
    return true;
}

static LZGVariant bench_parse_variant(const char *name, bool *ok) {
    *ok = true;
    if (!name || strcmp(name, "aap") == 0) return LZG_VARIANT_AAP;
    if (strcmp(name, "ndp") == 0) return LZG_VARIANT_NDP;
    if (strcmp(name, "naive") == 0) return LZG_VARIANT_NAIVE;
    *ok = false;
    return LZG_VARIANT_AAP;
}

static bool bench_matches_filter(const char *filter, const char *name) {
    const char *p;
    size_t name_len;

    if (!filter || strcmp(filter, "all") == 0) return true;
    name_len = strlen(name);
    p = filter;

    while (*p) {
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);

        if (len == name_len && strncmp(p, name, name_len) == 0)
            return true;
        if (!comma) break;
        p = comma + 1;
    }

    return false;
}

static double bench_items_one(const BenchContext *ctx) {
    (void)ctx;
    return 1.0;
}

static double bench_items_batch(const BenchContext *ctx) {
    return (double)ctx->batch_size;
}

static LZGError bench_graph_summary(BenchContext *ctx, uint64_t iter) {
    LZGGraphSummary summary;
    LZGError err = lzg_graph_summary(ctx->graph, &summary);
    if (err == LZG_OK)
        ctx->sink_u64 += (uint64_t)summary.n_nodes + (uint64_t)summary.n_edges + iter;
    return err;
}

static LZGError bench_simulate_one(BenchContext *ctx, uint64_t iter) {
    LZGError err = lzg_simulate(ctx->graph, 1u, &ctx->rng, ctx->sim_batch);
    (void)iter;
    if (err != LZG_OK) return err;
    ctx->sink_d += ctx->sim_batch[0].log_prob;
    bench_free_sim_array(ctx->sim_batch, 1u);
    return LZG_OK;
}

static LZGError bench_simulate_batch(BenchContext *ctx, uint64_t iter) {
    LZGError err = lzg_simulate(ctx->graph, ctx->batch_size, &ctx->rng, ctx->sim_batch);
    (void)iter;
    if (err != LZG_OK) return err;
    for (uint32_t i = 0; i < ctx->batch_size; i++)
        ctx->sink_d += ctx->sim_batch[i].log_prob;
    bench_free_sim_array(ctx->sim_batch, ctx->batch_size);
    return LZG_OK;
}

static const char *bench_query_at(const BenchContext *ctx, uint64_t iter) {
    return ctx->queries[(uint32_t)(iter % ctx->query_count)];
}

static void bench_prepare_batch_queries(BenchContext *ctx, uint64_t iter) {
    uint32_t base = (uint32_t)((iter * ctx->batch_size) % ctx->query_count);
    for (uint32_t i = 0; i < ctx->batch_size; i++)
        ctx->batch_queries[i] = ctx->queries[(base + i) % ctx->query_count];
}

static LZGError bench_lzpgen_one(BenchContext *ctx, uint64_t iter) {
    const char *seq = bench_query_at(ctx, iter);
    ctx->sink_d += lzg_walk_log_prob(ctx->graph, seq, (uint32_t)strlen(seq));
    return LZG_OK;
}

static LZGError bench_lzpgen_batch(BenchContext *ctx, uint64_t iter) {
    bench_prepare_batch_queries(ctx, iter);
    if (lzg_walk_log_prob_batch(ctx->graph, ctx->batch_queries,
                                ctx->batch_size, ctx->batch_logps) != LZG_OK)
        return lzg_last_error();
    for (uint32_t i = 0; i < ctx->batch_size; i++)
        ctx->sink_d += ctx->batch_logps[i];
    return LZG_OK;
}

static LZGError bench_sequence_perplexity(BenchContext *ctx, uint64_t iter) {
    const char *seq = bench_query_at(ctx, iter);
    ctx->sink_d += lzg_sequence_perplexity(ctx->graph, seq, (uint32_t)strlen(seq));
    return LZG_OK;
}

static LZGError bench_repertoire_perplexity(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    ctx->sink_d += lzg_repertoire_perplexity(ctx->graph, ctx->batch_queries, ctx->batch_size);
    return LZG_OK;
}

static LZGError bench_path_entropy_rate(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    ctx->sink_d += lzg_path_entropy_rate(ctx->graph, ctx->batch_queries, ctx->batch_size);
    return LZG_OK;
}

static LZGError bench_path_count(BenchContext *ctx, uint64_t iter) {
    double out = 0.0;
    (void)iter;
    if (lzg_graph_path_count(ctx->graph, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += out;
    return LZG_OK;
}

static LZGError bench_pgen_diagnostics(BenchContext *ctx, uint64_t iter) {
    LZGPgenDiagnostics diag;
    (void)iter;
    if (lzg_pgen_diagnostics(ctx->graph, 1e-6, &diag) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += diag.total_absorbed;
    return LZG_OK;
}

static LZGError bench_pgen_moments(BenchContext *ctx, uint64_t iter) {
    LZGPgenMoments mom;
    (void)iter;
    if (lzg_pgen_moments(ctx->graph, &mom) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += mom.mean;
    return LZG_OK;
}

static LZGError bench_pgen_analytical(BenchContext *ctx, uint64_t iter) {
    LZGPgenDist dist;
    (void)iter;
    if (lzg_pgen_analytical(ctx->graph, &dist) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += dist.global.mean;
    return LZG_OK;
}

static LZGError bench_pgen_dynamic_range(BenchContext *ctx, uint64_t iter) {
    LZGDynamicRange dr;
    (void)iter;
    if (lzg_pgen_dynamic_range(ctx->graph, &dr) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += dr.dynamic_range_orders;
    return LZG_OK;
}

static LZGError bench_effective_diversity(BenchContext *ctx, uint64_t iter) {
    LZGEffectiveDiversity div;
    (void)iter;
    if (lzg_effective_diversity(ctx->graph, &div) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += div.effective_diversity;
    return LZG_OK;
}

static LZGError bench_power_sum_2(BenchContext *ctx, uint64_t iter) {
    double m = 0.0;
    (void)iter;
    if (lzg_power_sum(ctx->graph, 2.0, &m) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += m;
    return LZG_OK;
}

static LZGError bench_hill_numbers(BenchContext *ctx, uint64_t iter) {
    const double orders[] = {0.0, 1.0, 2.0, 5.0};
    double out[4];
    (void)iter;
    if (lzg_hill_numbers(ctx->graph, orders, 4u, out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += out[1];
    return LZG_OK;
}

static LZGError bench_hill_curve(BenchContext *ctx, uint64_t iter) {
    LZGHillCurve hc = {0};
    (void)iter;
    if (lzg_hill_curve(ctx->graph, NULL, 0u, &hc) != LZG_OK)
        return lzg_last_error();
    if (hc.n > 0) ctx->sink_d += hc.hill_numbers[0];
    lzg_hill_curve_free(&hc);
    return LZG_OK;
}

static LZGError bench_predicted_richness(BenchContext *ctx, uint64_t iter) {
    double out = 0.0;
    (void)iter;
    if (lzg_predicted_richness(ctx->graph, ctx->richness_depth, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += out;
    return LZG_OK;
}

static LZGError bench_predicted_overlap(BenchContext *ctx, uint64_t iter) {
    double out = 0.0;
    (void)iter;
    if (lzg_predicted_overlap(ctx->graph, ctx->overlap_a, ctx->overlap_b, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += out;
    return LZG_OK;
}

static LZGError bench_feature_mass_profile(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    if (lzg_feature_mass_profile(ctx->graph, ctx->mass_profile, ctx->max_pos) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += ctx->mass_profile[ctx->max_pos > 0 ? ctx->max_pos - 1u : 0u];
    return LZG_OK;
}

static LZGError bench_feature_stats(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    if (lzg_feature_stats(ctx->graph, ctx->feature_stats) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += ctx->feature_stats[0];
    return LZG_OK;
}

static LZGError bench_jensen_shannon(BenchContext *ctx, uint64_t iter) {
    double out = 0.0;
    (void)iter;
    if (lzg_jensen_shannon_divergence(ctx->graph, ctx->aux_graph, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += out;
    return LZG_OK;
}

static LZGError bench_feature_aligned(BenchContext *ctx, uint64_t iter) {
    uint32_t dim = 0;
    (void)iter;
    if (lzg_feature_aligned(ctx->graph, ctx->aux_graph, ctx->aligned_features, &dim) != LZG_OK)
        return lzg_last_error();
    if (dim > 0) ctx->sink_d += ctx->aligned_features[0];
    return LZG_OK;
}

static LZGError bench_graph_union(BenchContext *ctx, uint64_t iter) {
    LZGGraph *out = NULL;
    (void)iter;
    if (lzg_graph_union(ctx->graph, ctx->aux_graph, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_u64 += out ? out->n_edges : 0u;
    lzg_graph_destroy(out);
    return LZG_OK;
}

static LZGError bench_graph_intersection(BenchContext *ctx, uint64_t iter) {
    LZGGraph *out = NULL;
    (void)iter;
    if (lzg_graph_intersection(ctx->graph, ctx->aux_graph, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_u64 += out ? out->n_edges : 0u;
    lzg_graph_destroy(out);
    return LZG_OK;
}

static LZGError bench_graph_difference(BenchContext *ctx, uint64_t iter) {
    LZGGraph *out = NULL;
    (void)iter;
    if (lzg_graph_difference(ctx->graph, ctx->aux_graph, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_u64 += out ? out->n_edges : 0u;
    lzg_graph_destroy(out);
    return LZG_OK;
}

static LZGError bench_graph_weighted_merge(BenchContext *ctx, uint64_t iter) {
    LZGGraph *out = NULL;
    (void)iter;
    if (lzg_graph_weighted_merge(ctx->graph, ctx->aux_graph, 1.0, 1.0, &out) != LZG_OK)
        return lzg_last_error();
    ctx->sink_u64 += out ? out->n_edges : 0u;
    lzg_graph_destroy(out);
    return LZG_OK;
}

static LZGError bench_save(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    return lzg_graph_save(ctx->graph, ctx->save_path);
}

static LZGError bench_load(BenchContext *ctx, uint64_t iter) {
    LZGGraph *loaded = NULL;
    LZGError err;
    (void)iter;
    err = lzg_graph_load(ctx->save_path, &loaded);
    if (err != LZG_OK) return err;
    ctx->sink_u64 += loaded ? loaded->n_nodes : 0u;
    lzg_graph_destroy(loaded);
    return LZG_OK;
}

static LZGError bench_gene_simulate_one(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    if (lzg_gene_simulate(ctx->graph, 1u, &ctx->aux_rng, ctx->gene_batch) != LZG_OK)
        return lzg_last_error();
    ctx->sink_d += ctx->gene_batch[0].base.log_prob;
    bench_free_gene_sim_array(ctx->gene_batch, 1u);
    return LZG_OK;
}

static LZGError bench_gene_simulate_batch(BenchContext *ctx, uint64_t iter) {
    (void)iter;
    if (lzg_gene_simulate(ctx->graph, ctx->batch_size, &ctx->aux_rng, ctx->gene_batch) != LZG_OK)
        return lzg_last_error();
    for (uint32_t i = 0; i < ctx->batch_size; i++)
        ctx->sink_d += ctx->gene_batch[i].base.log_prob;
    bench_free_gene_sim_array(ctx->gene_batch, ctx->batch_size);
    return LZG_OK;
}

static const BenchOp BENCH_OPS[] = {
    {"graph_summary",       "calls", bench_graph_summary,       bench_items_one,   false, false, false, false},
    {"simulate_1",          "seq",   bench_simulate_one,        bench_items_one,   false, false, false, false},
    {"simulate_batch",      "seq",   bench_simulate_batch,      bench_items_batch, false, false, false, false},
    {"lzpgen_1",            "seq",   bench_lzpgen_one,          bench_items_one,   true,  false, false, false},
    {"lzpgen_batch",        "seq",   bench_lzpgen_batch,        bench_items_batch, true,  false, false, false},
    {"sequence_perplexity", "seq",   bench_sequence_perplexity, bench_items_one,   true,  false, false, false},
    {"repertoire_perplexity","seq",  bench_repertoire_perplexity, bench_items_batch,true,  false, false, false},
    {"path_entropy_rate",   "seq",   bench_path_entropy_rate,   bench_items_batch, true,  false, false, false},
    {"path_count",          "calls", bench_path_count,          bench_items_one,   false, false, false, false},
    {"pgen_diagnostics",    "calls", bench_pgen_diagnostics,    bench_items_one,   false, false, false, false},
    {"pgen_moments",        "calls", bench_pgen_moments,        bench_items_one,   false, false, false, false},
    {"pgen_analytical",     "calls", bench_pgen_analytical,     bench_items_one,   false, false, false, false},
    {"pgen_dynamic_range",  "calls", bench_pgen_dynamic_range,  bench_items_one,   false, false, false, false},
    {"effective_diversity", "calls", bench_effective_diversity, bench_items_one,   false, false, false, false},
    {"power_sum_2",         "calls", bench_power_sum_2,         bench_items_one,   false, false, false, false},
    {"hill_numbers",        "calls", bench_hill_numbers,        bench_items_one,   false, false, false, false},
    {"hill_curve",          "calls", bench_hill_curve,          bench_items_one,   false, false, false, false},
    {"predicted_richness",  "calls", bench_predicted_richness,  bench_items_one,   false, false, false, false},
    {"predicted_overlap",   "calls", bench_predicted_overlap,   bench_items_one,   false, false, false, false},
    {"feature_mass_profile","calls", bench_feature_mass_profile,bench_items_one,   false, false, false, false},
    {"feature_stats",       "calls", bench_feature_stats,       bench_items_one,   false, false, false, false},
    {"jsd",                 "calls", bench_jensen_shannon,      bench_items_one,   false, true,  false, false},
    {"feature_aligned",     "calls", bench_feature_aligned,     bench_items_one,   false, true,  false, false},
    {"graph_union",         "calls", bench_graph_union,         bench_items_one,   false, true,  false, true},
    {"graph_intersection",  "calls", bench_graph_intersection,  bench_items_one,   false, true,  false, true},
    {"graph_difference",    "calls", bench_graph_difference,    bench_items_one,   false, true,  false, true},
    {"graph_weighted_merge","calls", bench_graph_weighted_merge,bench_items_one,   false, true,  false, true},
    {"save",                "calls", bench_save,                bench_items_one,   false, false, false, false},
    {"load",                "calls", bench_load,                bench_items_one,   false, false, false, false},
    {"gene_simulate_1",     "seq",   bench_gene_simulate_one,   bench_items_one,   false, false, true,  false},
    {"gene_simulate_batch", "seq",   bench_gene_simulate_batch, bench_items_batch, false, false, true,  false},
};

static const size_t BENCH_OP_COUNT = sizeof(BENCH_OPS) / sizeof(BENCH_OPS[0]);

static bool bench_op_applicable(const BenchOp *op, const BenchContext *ctx) {
    if (op->needs_aux && !ctx->aux_graph) return false;
    if (op->needs_gene_data && !ctx->graph->gene_data) return false;
    if (op->skip_when_gene_data && ctx->graph->gene_data) return false;
    return true;
}

static LZGError bench_prepare_runtime_buffers(BenchContext *ctx) {
    if (ctx->batch_queries && ctx->batch_logps && ctx->sim_batch && ctx->gene_batch)
        return LZG_OK;

    if (!ctx->batch_queries)
        ctx->batch_queries = calloc(ctx->batch_size, sizeof(char *));
    if (!ctx->batch_logps)
        ctx->batch_logps = calloc(ctx->batch_size, sizeof(double));
    if (!ctx->sim_batch)
        ctx->sim_batch = calloc(ctx->batch_size, sizeof(LZGSimResult));
    if (!ctx->gene_batch)
        ctx->gene_batch = calloc(ctx->batch_size, sizeof(LZGGeneSimResult));
    if (!ctx->batch_queries || !ctx->batch_logps || !ctx->sim_batch || !ctx->gene_batch)
        return LZG_ERR_ALLOC;

    lzg_rng_seed(&ctx->rng, 123456789ULL);
    lzg_rng_seed(&ctx->aux_rng, 987654321ULL);
    return LZG_OK;
}

static LZGError bench_prepare_query_workload(BenchContext *ctx) {
    LZGError err;

    if (ctx->query_sims && ctx->queries)
        return LZG_OK;

    err = bench_prepare_runtime_buffers(ctx);
    if (err != LZG_OK) return err;

    if (!ctx->query_sims)
        ctx->query_sims = calloc(ctx->query_count, sizeof(LZGSimResult));
    if (!ctx->queries)
        ctx->queries = calloc(ctx->query_count, sizeof(char *));
    if (!ctx->query_sims || !ctx->queries)
        return LZG_ERR_ALLOC;

    {
        err = lzg_simulate(ctx->graph, ctx->query_count, &ctx->rng, ctx->query_sims);
        if (err != LZG_OK) return err;
    }

    for (uint32_t i = 0; i < ctx->query_count; i++)
        ctx->queries[i] = ctx->query_sims[i].sequence;

    for (uint32_t i = 0; i < ctx->batch_size; i++)
        ctx->batch_queries[i] = ctx->queries[i % ctx->query_count];

    return LZG_OK;
}

static LZGError bench_prepare_aux_graph(BenchContext *ctx) {
    uint32_t aux_count;
    LZGGraph *aux;
    if (ctx->graph->gene_data) return LZG_OK;

    aux_count = ctx->query_count < BENCH_QUERY_BUILD_COUNT
        ? ctx->query_count
        : BENCH_QUERY_BUILD_COUNT;
    if (aux_count == 0) return LZG_OK;

    aux = lzg_graph_create(ctx->graph->variant);
    if (!aux) return LZG_ERR_ALLOC;

    if (lzg_graph_build(aux, ctx->queries, aux_count, NULL, NULL, NULL, 0.0, 0) != LZG_OK) {
        lzg_graph_destroy(aux);
        return lzg_last_error();
    }

    ctx->aux_query_count = aux_count;
    ctx->aux_graph = aux;
    return LZG_OK;
}

static LZGError bench_prepare_feature_buffers(BenchContext *ctx,
                                              bool need_mass_profile,
                                              bool need_aligned_features) {
    if (need_mass_profile && !ctx->mass_profile) {
        ctx->max_pos = ctx->graph->max_length + 1u;
        if (ctx->max_pos == 0u) ctx->max_pos = 32u;

        ctx->mass_profile = calloc(ctx->max_pos, sizeof(double));
        if (!ctx->mass_profile) return LZG_ERR_ALLOC;
    }

    if (need_aligned_features && !ctx->aligned_features && ctx->graph->n_nodes > 0) {
        ctx->aligned_features = calloc(ctx->graph->n_nodes, sizeof(double));
        if (!ctx->aligned_features) return LZG_ERR_ALLOC;
    }

    return LZG_OK;
}

static LZGError bench_prepare_save_path(BenchContext *ctx, const BenchOptions *opt) {
    if (opt->save_path) {
        snprintf(ctx->save_path, sizeof(ctx->save_path), "%s", opt->save_path);
        ctx->save_path_owned = false;
    } else {
        snprintf(ctx->save_path, sizeof(ctx->save_path),
                 "/tmp/lzgraph_bench_%ld.lzg", (long)getpid());
        ctx->save_path_owned = true;
    }

    if (ctx->save_path_owned)
        unlink(ctx->save_path);
    return lzg_graph_save(ctx->graph, ctx->save_path);
}

static LZGError bench_load_or_build_graph(BenchContext *ctx, const BenchOptions *opt) {
    if (opt->graph_path) {
        return lzg_graph_load(opt->graph_path, &ctx->graph);
    }

    if (opt->plain_path) {
        bool ok = false;
        LZGVariant variant = bench_parse_variant(opt->variant_name, &ok);
        if (!ok) return LZG_FAIL(LZG_ERR_INVALID_VARIANT, "unknown variant '%s'", opt->variant_name);

        ctx->graph = lzg_graph_create(variant);
        if (!ctx->graph) return LZG_ERR_ALLOC;
        return lzg_graph_build_plain_file(ctx->graph, opt->plain_path, opt->smoothing);
    }

    return LZG_FAIL(LZG_ERR_INVALID_ARG, "provide either --graph or --plain");
}

static void bench_print_graph_info(const BenchContext *ctx,
                                   const BenchOptions *opt,
                                   const LZGGraphSummary *summary,
                                   const LZGPgenDiagnostics *diag) {
    printf("LZGraphs C Benchmark Harness\n");
    printf("============================\n\n");
    printf("source: %s\n", opt->graph_path ? opt->graph_path : opt->plain_path);
    printf("mode: %s\n", opt->graph_path ? "load" : "build");
    printf("seconds_per_op: %.3f\n", opt->seconds_per_op);
    printf("batch_size: %u\n", ctx->batch_size);
    printf("query_count: %u\n", ctx->query_count);

    if (summary) {
        printf("graph: nodes=%u edges=%u initials=%u terminals=%u dag=%d\n",
               summary->n_nodes, summary->n_edges, summary->n_initial,
               summary->n_terminal, (int)summary->is_dag);
    } else {
        printf("graph: nodes=%u edges=%u\n", ctx->graph->n_nodes, ctx->graph->n_edges);
    }

    if (diag) {
        printf("raw_absorbed=%.12f raw_leaked=%.12f is_proper=%d\n",
               diag->total_absorbed, diag->total_leaked, (int)diag->is_proper);
    }

    if (ctx->save_path[0] != '\0')
        printf("save_path: %s\n", ctx->save_path);
    printf("\n");
}

static void bench_print_row_header(void) {
    printf("%-22s %10s %14s %14s %14s %14s\n",
           "operation", "seconds", "calls", "calls/sec", "units/sec", "units/180s");
    printf("%-22s %10s %14s %14s %14s %14s\n",
           "----------------------", "----------", "--------------",
           "--------------", "--------------", "--------------");
}

static LZGError bench_run_one(const BenchOp *op, BenchContext *ctx, double seconds,
                              BenchRunResult *result) {
    double start, elapsed, units_per_iter, units_per_sec, units_per_180;
    uint64_t calls = 0;
    LZGError err;

    err = op->fn(ctx, 0u);
    if (err != LZG_OK) return err;

    start = bench_now_seconds();
    while (bench_now_seconds() - start < seconds) {
        for (uint32_t chunk = 0; chunk < 256u; chunk++) {
            err = op->fn(ctx, calls + 1u);
            if (err != LZG_OK) return err;
            calls++;
            if (bench_now_seconds() - start >= seconds)
                break;
        }
    }

    elapsed = bench_now_seconds() - start;
    units_per_iter = op->items_fn(ctx);
    units_per_sec = elapsed > 0.0 ? ((double)calls * units_per_iter) / elapsed : 0.0;
    units_per_180 = units_per_sec * 180.0;

    if (result) {
        result->name = op->name;
        result->unit_label = op->unit_label;
        result->skipped = false;
        result->skip_reason = NULL;
        result->seconds = elapsed;
        result->calls = calls;
        result->calls_per_sec = elapsed > 0.0 ? (double)calls / elapsed : 0.0;
        result->units_per_sec = units_per_sec;
        result->units_per_180s = units_per_180;
    }

    printf("%-22s %10.3f %14llu %14.2f %14.2f %14.2f\n",
           op->name,
           elapsed,
           (unsigned long long)calls,
           elapsed > 0.0 ? (double)calls / elapsed : 0.0,
           units_per_sec,
           units_per_180);
    return LZG_OK;
}

static void bench_result_mark_skipped(BenchRunResult *result,
                                      const BenchOp *op,
                                      const char *reason) {
    if (!result) return;
    result->name = op->name;
    result->unit_label = op->unit_label;
    result->skipped = true;
    result->skip_reason = reason;
    result->seconds = 0.0;
    result->calls = 0u;
    result->calls_per_sec = 0.0;
    result->units_per_sec = 0.0;
    result->units_per_180s = 0.0;
}

static void bench_json_write_string(FILE *f, const char *s) {
    fputc('"', f);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
                case '\\': fputs("\\\\", f); break;
                case '"':  fputs("\\\"", f); break;
                case '\n': fputs("\\n", f); break;
                case '\r': fputs("\\r", f); break;
                case '\t': fputs("\\t", f); break;
                default:
                    if (*p < 0x20) {
                        fprintf(f, "\\u%04x", (unsigned)*p);
                    } else {
                        fputc(*p, f);
                    }
            }
        }
    }
    fputc('"', f);
}

static LZGError bench_write_json(const char *path,
                                 const BenchOptions *opt,
                                 const BenchContext *ctx,
                                 const LZGGraphSummary *summary,
                                 const LZGPgenDiagnostics *diag,
                                 const BenchRunResult *results,
                                 size_t n_results) {
    FILE *f;
    time_t now = time(NULL);
    struct tm tm_now;
    char ts[64];

    if (!path) return LZG_OK;

    f = fopen(path, "w");
    if (!f) return LZG_FAIL(LZG_ERR_IO_OPEN, "failed to open json output '%s'", path);

    localtime_r(&now, &tm_now);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", &tm_now);

    fprintf(f, "{\n");
    fprintf(f, "  \"timestamp\": ");
    bench_json_write_string(f, ts);
    fprintf(f, ",\n");
    fprintf(f, "  \"source\": ");
    bench_json_write_string(f, opt->graph_path ? opt->graph_path : opt->plain_path);
    fprintf(f, ",\n");
    fprintf(f, "  \"mode\": ");
    bench_json_write_string(f, opt->graph_path ? "load" : "build");
    fprintf(f, ",\n");
    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"seconds_per_op\": %.12f,\n", opt->seconds_per_op);
    fprintf(f, "    \"batch_size\": %u,\n", ctx->batch_size);
    fprintf(f, "    \"query_count\": %u,\n", ctx->query_count);
    fprintf(f, "    \"richness_depth\": %.12f,\n", ctx->richness_depth);
    fprintf(f, "    \"overlap_a\": %.12f,\n", ctx->overlap_a);
    fprintf(f, "    \"overlap_b\": %.12f,\n", ctx->overlap_b);
    fprintf(f, "    \"ops_filter\": ");
    bench_json_write_string(f, opt->ops_filter ? opt->ops_filter : "all");
    fprintf(f, "\n  },\n");
    fprintf(f, "  \"graph\": {\n");
    fprintf(f, "    \"n_nodes\": %u,\n", summary->n_nodes);
    fprintf(f, "    \"n_edges\": %u,\n", summary->n_edges);
    fprintf(f, "    \"n_initial\": %u,\n", summary->n_initial);
    fprintf(f, "    \"n_terminal\": %u,\n", summary->n_terminal);
    fprintf(f, "    \"max_out_degree\": %u,\n", summary->max_out_degree);
    fprintf(f, "    \"max_in_degree\": %u,\n", summary->max_in_degree);
    fprintf(f, "    \"n_isolates\": %u,\n", summary->n_isolates);
    fprintf(f, "    \"is_dag\": %s\n", summary->is_dag ? "true" : "false");
    fprintf(f, "  },\n");
    fprintf(f, "  \"raw_diagnostics\": ");
    if (diag) {
        fprintf(f, "{\n");
        fprintf(f, "    \"total_absorbed\": %.12f,\n", diag->total_absorbed);
        fprintf(f, "    \"total_leaked\": %.12f,\n", diag->total_leaked);
        fprintf(f, "    \"initial_prob_sum\": %.12f,\n", diag->initial_prob_sum);
        fprintf(f, "    \"is_proper\": %s\n", diag->is_proper ? "true" : "false");
        fprintf(f, "  },\n");
    } else {
        fprintf(f, "null,\n");
    }
    fprintf(f, "  \"results\": [\n");

    for (size_t i = 0; i < n_results; i++) {
        const BenchRunResult *r = &results[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"name\": ");
        bench_json_write_string(f, r->name);
        fprintf(f, ",\n");
        fprintf(f, "      \"unit_label\": ");
        bench_json_write_string(f, r->unit_label);
        fprintf(f, ",\n");
        fprintf(f, "      \"skipped\": %s,\n", r->skipped ? "true" : "false");
        fprintf(f, "      \"skip_reason\": ");
        if (r->skip_reason) bench_json_write_string(f, r->skip_reason);
        else fputs("null", f);
        fprintf(f, ",\n");
        fprintf(f, "      \"seconds\": %.12f,\n", r->seconds);
        fprintf(f, "      \"calls\": %llu,\n", (unsigned long long)r->calls);
        fprintf(f, "      \"calls_per_sec\": %.12f,\n", r->calls_per_sec);
        fprintf(f, "      \"units_per_sec\": %.12f,\n", r->units_per_sec);
        fprintf(f, "      \"units_per_180s\": %.12f\n", r->units_per_180s);
        fprintf(f, "    }%s\n", (i + 1u < n_results) ? "," : "");
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"sink\": {\n");
    fprintf(f, "    \"sink_d\": %.12f,\n", ctx->sink_d);
    fprintf(f, "    \"sink_u64\": %llu\n", (unsigned long long)ctx->sink_u64);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    return LZG_OK;
}

static void bench_list_ops(void) {
    for (size_t i = 0; i < BENCH_OP_COUNT; i++)
        printf("%s\n", BENCH_OPS[i].name);
}

static bool bench_parse_args(int argc, char **argv, BenchOptions *opt) {
    *opt = (BenchOptions){
        .variant_name = "aap",
        .seconds_per_op = BENCH_DEFAULT_SECONDS,
        .smoothing = 0.0,
        .richness_depth = BENCH_DEFAULT_RICHNESS_D,
        .overlap_a = BENCH_DEFAULT_OVERLAP_A,
        .overlap_b = BENCH_DEFAULT_OVERLAP_B,
        .batch_size = BENCH_DEFAULT_BATCH_SIZE,
        .query_count = BENCH_DEFAULT_QUERY_COUNT,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--graph") == 0 && i + 1 < argc) {
            opt->graph_path = argv[++i];
        } else if (strcmp(arg, "--plain") == 0 && i + 1 < argc) {
            opt->plain_path = argv[++i];
        } else if (strcmp(arg, "--variant") == 0 && i + 1 < argc) {
            opt->variant_name = argv[++i];
        } else if (strcmp(arg, "--ops") == 0 && i + 1 < argc) {
            opt->ops_filter = argv[++i];
        } else if (strcmp(arg, "--save-path") == 0 && i + 1 < argc) {
            opt->save_path = argv[++i];
        } else if (strcmp(arg, "--json-out") == 0 && i + 1 < argc) {
            opt->json_out = argv[++i];
        } else if (strcmp(arg, "--seconds") == 0 && i + 1 < argc) {
            if (!bench_parse_double(argv[++i], &opt->seconds_per_op) || opt->seconds_per_op <= 0.0)
                return false;
        } else if (strcmp(arg, "--smoothing") == 0 && i + 1 < argc) {
            if (!bench_parse_double(argv[++i], &opt->smoothing) || opt->smoothing < 0.0)
                return false;
        } else if (strcmp(arg, "--richness-depth") == 0 && i + 1 < argc) {
            if (!bench_parse_double(argv[++i], &opt->richness_depth) || opt->richness_depth <= 0.0)
                return false;
        } else if (strcmp(arg, "--overlap-a") == 0 && i + 1 < argc) {
            if (!bench_parse_double(argv[++i], &opt->overlap_a) || opt->overlap_a <= 0.0)
                return false;
        } else if (strcmp(arg, "--overlap-b") == 0 && i + 1 < argc) {
            if (!bench_parse_double(argv[++i], &opt->overlap_b) || opt->overlap_b <= 0.0)
                return false;
        } else if (strcmp(arg, "--batch") == 0 && i + 1 < argc) {
            if (!bench_parse_u32(argv[++i], &opt->batch_size) || opt->batch_size == 0u)
                return false;
        } else if (strcmp(arg, "--queries") == 0 && i + 1 < argc) {
            if (!bench_parse_u32(argv[++i], &opt->query_count) || opt->query_count == 0u)
                return false;
        } else if (strcmp(arg, "--list-ops") == 0) {
            opt->list_ops = true;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            bench_usage(stdout, argv[0]);
            exit(0);
        } else {
            return false;
        }
    }

    if (opt->list_ops) return true;
    if (!!opt->graph_path == !!opt->plain_path) return false;
    if (opt->query_count < opt->batch_size) opt->query_count = opt->batch_size;
    return true;
}

int main(int argc, char **argv) {
    BenchOptions opt;
    BenchContext ctx = {0};
    BenchRunResult results[BENCH_OP_COUNT];
    size_t n_results = 0;
    LZGError err;
    LZGGraphSummary summary = {0};
    LZGPgenDiagnostics diag = {0};
    bool need_runtime_buffers = false;
    bool need_query_workload = false;
    bool need_aux_graph = false;
    bool need_feature_mass_profile = false;
    bool need_aligned_features = false;
    bool need_save_path = false;
    bool need_diag = false;
    bool diag_ready = false;

    if (!bench_parse_args(argc, argv, &opt)) {
        bench_usage(stderr, argv[0]);
        return 2;
    }

    if (opt.list_ops) {
        bench_list_ops();
        return 0;
    }

    ctx.batch_size = opt.batch_size;
    ctx.query_count = opt.query_count;
    ctx.seconds_per_op = opt.seconds_per_op;
    ctx.richness_depth = opt.richness_depth;
    ctx.overlap_a = opt.overlap_a;
    ctx.overlap_b = opt.overlap_b;

    lzg_log_set(LZG_LOG_NONE, NULL, NULL);

    err = bench_load_or_build_graph(&ctx, &opt);
    if (err != LZG_OK) {
        fprintf(stderr, "failed to prepare graph: %s\n", lzg_error_message());
        bench_context_destroy(&ctx);
        return 1;
    }

    if (lzg_graph_summary(ctx.graph, &summary) != LZG_OK) {
        fprintf(stderr, "failed to summarize graph: %s\n", lzg_error_message());
        bench_context_destroy(&ctx);
        return 1;
    }

    for (size_t i = 0; i < BENCH_OP_COUNT; i++) {
        const BenchOp *op = &BENCH_OPS[i];

        if (!bench_matches_filter(opt.ops_filter, op->name))
            continue;

        if (op->needs_queries || op->needs_aux)
            need_query_workload = true;
        if (op->needs_aux && !ctx.graph->gene_data)
            need_aux_graph = true;
        if (strcmp(op->name, "simulate_1") == 0 || strcmp(op->name, "simulate_batch") == 0 ||
            strcmp(op->name, "gene_simulate_1") == 0 || strcmp(op->name, "gene_simulate_batch") == 0)
            need_runtime_buffers = true;
        if (strcmp(op->name, "feature_mass_profile") == 0)
            need_feature_mass_profile = true;
        if (strcmp(op->name, "feature_aligned") == 0)
            need_aligned_features = true;
        if (strcmp(op->name, "save") == 0 || strcmp(op->name, "load") == 0)
            need_save_path = true;
        if (strcmp(op->name, "pgen_diagnostics") == 0)
            need_diag = true;
    }

    if (need_query_workload)
        need_runtime_buffers = true;

    if (need_runtime_buffers) {
        err = bench_prepare_runtime_buffers(&ctx);
        if (err != LZG_OK) {
            fprintf(stderr, "failed to prepare runtime buffers: %s\n", lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
    }

    if (need_query_workload) {
        err = bench_prepare_query_workload(&ctx);
        if (err != LZG_OK) {
            fprintf(stderr, "failed to prepare query workload: %s\n", lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
    }

    if (need_aux_graph) {
        err = bench_prepare_aux_graph(&ctx);
        if (err != LZG_OK) {
            fprintf(stderr, "warning: auxiliary graph disabled: %s\n", lzg_error_message());
            ctx.aux_graph = NULL;
        }
    }

    if (need_feature_mass_profile || need_aligned_features) {
        err = bench_prepare_feature_buffers(&ctx, need_feature_mass_profile, need_aligned_features);
        if (err != LZG_OK) {
            fprintf(stderr, "failed to prepare feature buffers: %s\n", lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
    }

    if (need_save_path) {
        err = bench_prepare_save_path(&ctx, &opt);
        if (err != LZG_OK) {
            fprintf(stderr, "failed to prepare save path: %s\n", lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
    }

    if (need_diag) {
        if (lzg_pgen_diagnostics(ctx.graph, 1e-12, &diag) != LZG_OK) {
            fprintf(stderr, "failed to inspect graph diagnostics: %s\n", lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
        diag_ready = true;
    }

    bench_print_graph_info(&ctx, &opt, &summary, diag_ready ? &diag : NULL);
    bench_print_row_header();

    for (size_t i = 0; i < BENCH_OP_COUNT; i++) {
        const BenchOp *op = &BENCH_OPS[i];

        if (!bench_matches_filter(opt.ops_filter, op->name))
            continue;
        if (!bench_op_applicable(op, &ctx)) {
            bench_result_mark_skipped(&results[n_results++], op, "not_applicable");
            printf("%-22s %10s %14s %14s %14s %14s\n",
                   op->name, "SKIP", "-", "-", "-", "-");
            continue;
        }

        err = bench_run_one(op, &ctx, opt.seconds_per_op, &results[n_results]);
        if (err != LZG_OK) {
            fprintf(stderr, "benchmark '%s' failed: %s\n", op->name, lzg_error_message());
            bench_context_destroy(&ctx);
            return 1;
        }
        n_results++;
    }

    printf("\nbench_sink_d=%.12f bench_sink_u64=%llu\n",
           ctx.sink_d, (unsigned long long)ctx.sink_u64);

    err = bench_write_json(opt.json_out, &opt, &ctx, &summary,
                           diag_ready ? &diag : NULL, results, n_results);
    if (err != LZG_OK) {
        fprintf(stderr, "failed to write json output: %s\n", lzg_error_message());
        bench_context_destroy(&ctx);
        return 1;
    }

    bench_context_destroy(&ctx);
    return 0;
}
