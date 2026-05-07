/**
 * @file fbg_build.c
 * @brief Grammar lifecycle, count aggregation, and finalize.
 *
 * Reference: fbg_prototype.py `__init__` + `_finalize`.
 */
#include "fbg_internal.h"
#include "../graph/graph_build_ingest.h"   /* stream helpers (reused)   */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Lifecycle ───────────────────────────────────────────────── */

LZGFbg *lzg_fbg_new(void) {
    LZGFbg *g = (LZGFbg *)calloc(1, sizeof(LZGFbg));
    if (!g) return NULL;
    lzg_fbg_alphabet_init(g->char_to_idx, g->idx_to_char, &g->alphabet_size);
    return g;
}

void lzg_fbg_destroy(LZGFbg *g) {
    if (!g) return;
    free(g->nts);
    free(g->rules);
    free(g->length_counts);
    if (g->rule_marginal) lzg_hm_destroy(g->rule_marginal);
    free(g);
}

/* ── Count aggregation (build time only) ─────────────────────── */

/**
 * Intermediate per-NT bucket: a dynamic array of (rule-fields, count) pairs
 * keyed by a within-NT local key. Packed into a hashmap during build, then
 * flattened into the final LZGFbg.rules[] slab at finalize.
 */
typedef struct {
    uint8_t  a, z;           /* source NT boundary chars (idx space)        */
    uint8_t  is_start;

    /* Per-rule storage. Grows as needed. */
    LZGFbgRule *rules;
    uint32_t    count;
    uint32_t    cap;

    /* Local dedup: packed local key → index into rules[]. */
    LZGHashMap *lookup;
} BuildNT;

typedef struct {
    /* nt_table[(a, z)] → BuildNT*  (BuildNT lives in nt_pool).
     * start_nt is stored as nt_pool[0] with is_start=1. */
    BuildNT     *nt_pool;
    uint32_t     n_nts;
    uint32_t     cap_nts;

    /* Flat (a, z) index → BuildNT* lookup. */
    int32_t      nt_index[LZG_FBG_ALPHABET_SIZE][LZG_FBG_ALPHABET_SIZE];
    int32_t      start_index;   /* -1 until S sees its first rule */

    /* Length histogram (weighted). */
    uint64_t    *length_counts;
    uint32_t     length_cap;
    uint32_t     max_length;
} BuildState;

static void build_state_init(BuildState *bs) {
    bs->nt_pool = NULL;
    bs->n_nts = 0;
    bs->cap_nts = 0;
    for (uint32_t i = 0; i < LZG_FBG_ALPHABET_SIZE; i++)
        for (uint32_t j = 0; j < LZG_FBG_ALPHABET_SIZE; j++)
            bs->nt_index[i][j] = -1;
    bs->start_index = -1;
    bs->length_counts = (uint64_t *)calloc(128, sizeof(uint64_t));
    bs->length_cap = 128;
    bs->max_length = 0;
}

static void build_state_free(BuildState *bs) {
    if (!bs) return;
    for (uint32_t i = 0; i < bs->n_nts; i++) {
        free(bs->nt_pool[i].rules);
        if (bs->nt_pool[i].lookup) lzg_hm_destroy(bs->nt_pool[i].lookup);
    }
    free(bs->nt_pool);
    free(bs->length_counts);
}

static LZGError build_state_ensure_nt(BuildState *bs, bool is_start,
                                      uint8_t a, uint8_t z,
                                      BuildNT **out_nt) {
    int32_t idx;
    if (is_start) {
        idx = bs->start_index;
        if (idx < 0) {
            idx = (int32_t)bs->n_nts;
            bs->start_index = idx;
        }
    } else {
        idx = bs->nt_index[a][z];
        if (idx < 0) {
            idx = (int32_t)bs->n_nts;
            bs->nt_index[a][z] = idx;
        }
    }
    if ((uint32_t)idx >= bs->n_nts) {
        /* Need to allocate a new BuildNT. */
        if (bs->n_nts >= bs->cap_nts) {
            uint32_t new_cap = bs->cap_nts ? bs->cap_nts * 2 : 32;
            BuildNT *tmp = (BuildNT *)realloc(bs->nt_pool,
                                              (size_t)new_cap * sizeof(BuildNT));
            if (!tmp) return LZG_FAIL(LZG_ERR_ALLOC, "fbg build: nt_pool realloc");
            bs->nt_pool = tmp;
            bs->cap_nts = new_cap;
        }
        BuildNT *nt = &bs->nt_pool[bs->n_nts];
        memset(nt, 0, sizeof(*nt));
        nt->a = a;
        nt->z = z;
        nt->is_start = is_start ? 1 : 0;
        nt->cap = 8;
        nt->rules = (LZGFbgRule *)calloc(nt->cap, sizeof(LZGFbgRule));
        if (!nt->rules) return LZG_FAIL(LZG_ERR_ALLOC, "fbg build: rules alloc");
        nt->lookup = lzg_hm_create(16);
        if (!nt->lookup) return LZG_FAIL(LZG_ERR_ALLOC, "fbg build: lookup alloc");
        bs->n_nts++;
    }
    *out_nt = &bs->nt_pool[idx];
    return LZG_OK;
}

static LZGError build_state_record_step(BuildState *bs,
                                         const LZGFbgStep *step,
                                         double weight) {
    uint8_t src_a, src_z;
    if (step->is_start) {
        src_a = LZG_FBG_IDX_AT;
        src_z = LZG_FBG_IDX_DOLL;
    } else {
        src_a = step->a_char;
        src_z = step->z_char;
    }

    BuildNT *nt;
    LZGError err = build_state_ensure_nt(bs, step->is_start != 0,
                                          src_a, src_z, &nt);
    if (err != LZG_OK) return err;

    uint64_t key = lzg_fbg_rule_local_key(step->kind,
                                           step->a_run_len, step->z_run_len,
                                           step->dst_a, step->dst_z,
                                           step->a_char, step->z_char);

    bool inserted = false;
    uint64_t *slot = lzg_hm_get_or_insert(nt->lookup, key,
                                           (uint64_t)nt->count, &inserted);
    uint32_t rule_idx = (uint32_t)*slot;

    if (inserted) {
        if (nt->count >= nt->cap) {
            uint32_t new_cap = nt->cap * 2;
            LZGFbgRule *tmp = (LZGFbgRule *)realloc(
                nt->rules, (size_t)new_cap * sizeof(LZGFbgRule));
            if (!tmp) return LZG_FAIL(LZG_ERR_ALLOC, "fbg build: rules realloc");
            nt->rules = tmp;
            nt->cap = new_cap;
        }
        LZGFbgRule *r = &nt->rules[nt->count];
        memset(r, 0, sizeof(*r));
        r->kind       = step->kind;
        /* For INTERNAL rules, the rule's a/z chars mirror the source NT's
         * (because the peel consumes the boundary chars). For leaves, the
         * leaf content a_char / z_char come from the step. */
        r->a_char     = step->a_char;
        r->z_char     = step->z_char;
        r->a_run_len  = step->a_run_len;
        r->z_run_len  = step->z_run_len;
        r->dst_a      = step->dst_a;
        r->dst_z      = step->dst_z;
        r->count      = 0;
        nt->count++;
    }

    /* Accumulate weighted count. We store in .count as a uint64 if weight is
     * integral, else carry fractional in a parallel path. To keep things
     * simple and match the prototype (which uses float counts), we reinterpret
     * LZGFbgRule.count as bits of a double. That's ugly; use a side-buffer
     * instead — see finalize. For build, keep fractional count in a temporary
     * double slot; here we stash into the rule's weight field which is unused
     * pre-finalize. */
    nt->rules[rule_idx].weight += weight;
    return LZG_OK;
}

/* ── Public build ────────────────────────────────────────────── */

/* Forward decl for the internal finalizer below. */
static LZGError finalize_from_build_state(LZGFbg *g, BuildState *bs,
                                           LZGFbgAbundanceMode mode,
                                           double smoothing,
                                           LZGFbgBackoff backoff);

LZGError lzg_fbg_build(LZGFbg *g,
                       const char **sequences, uint32_t n,
                       const uint64_t *abundances,
                       LZGFbgAbundanceMode mode,
                       double smoothing,
                       LZGFbgBackoff backoff) {
    if (!g || !sequences) return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_build: NULL");
    if (n == 0) return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "fbg_build: no sequences");
    if (smoothing < 0.0)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE, "fbg_build: smoothing < 0");

    /* Step 1: observe all chars → build alphabet. */
    for (uint32_t s = 0; s < n; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        for (const char *p = seq; *p; p++) {
            LZGError err = lzg_fbg_alphabet_observe(
                g->char_to_idx, g->idx_to_char, &g->alphabet_size, *p);
            if (err != LZG_OK) return err;
        }
    }

    /* Step 2: decompose each sequence, accumulate rule counts. */
    BuildState bs;
    build_state_init(&bs);
    if (!bs.length_counts) {
        build_state_free(&bs);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_build: length_counts alloc");
    }

    LZGFbgStep steps[LZG_FBG_MAX_STEPS];

    for (uint32_t s = 0; s < n; s++) {
        const char *seq = sequences[s];
        if (!seq) continue;
        uint32_t seq_len = (uint32_t)strlen(seq);
        if (seq_len == 0) continue;

        uint64_t abund = abundances ? abundances[s] : 1;
        double w = lzg_fbg_abund_weight(mode, abund);
        if (w <= 0.0) continue;

        uint32_t n_steps = 0;
        LZGError err = lzg_fbg_decompose(g, seq, seq_len, steps, &n_steps);
        if (err != LZG_OK) {
            build_state_free(&bs);
            return err;
        }

        for (uint32_t t = 0; t < n_steps; t++) {
            err = build_state_record_step(&bs, &steps[t], w);
            if (err != LZG_OK) {
                build_state_free(&bs);
                return err;
            }
        }

        /* Length histogram (weighted). */
        if (seq_len >= bs.length_cap) {
            uint32_t new_cap = seq_len + 64;
            uint64_t *tmp = (uint64_t *)realloc(bs.length_counts,
                                                (size_t)new_cap * sizeof(uint64_t));
            if (!tmp) {
                build_state_free(&bs);
                return LZG_FAIL(LZG_ERR_ALLOC, "fbg_build: length_counts realloc");
            }
            memset(tmp + bs.length_cap, 0,
                   (size_t)(new_cap - bs.length_cap) * sizeof(uint64_t));
            bs.length_counts = tmp;
            bs.length_cap = new_cap;
        }
        /* Length histogram counts in "raw abundance" units for interpretability. */
        bs.length_counts[seq_len] += abundances ? abundances[s] : 1u;
        if (seq_len > bs.max_length) bs.max_length = seq_len;
    }

    /* Step 3: finalize — compute MLE weights, NTs, spectral radius, backoff. */
    LZGError err = finalize_from_build_state(g, &bs, mode, smoothing, backoff);
    build_state_free(&bs);
    return err;
}

/* ── Streaming build from plain-text file ────────────────────── */

/* Progress log cadence — mirror the existing graph streamer's constants. */
#define FBG_STREAM_LOG_EVERY   1000000ULL
#define FBG_STREAM_LOG_MIN_SEC 5.0

static uint32_t build_state_total_rules(const BuildState *bs) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < bs->n_nts; i++) total += bs->nt_pool[i].count;
    return total;
}

/* Grow bs->length_counts to at least `seq_len + 1` entries (zero-filled). */
static LZGError bs_ensure_length_cap(BuildState *bs, uint32_t seq_len) {
    if (seq_len < bs->length_cap) return LZG_OK;
    uint32_t new_cap = seq_len + 64;
    uint64_t *tmp = (uint64_t *)realloc(bs->length_counts,
                                         (size_t)new_cap * sizeof(uint64_t));
    if (!tmp) return LZG_FAIL(LZG_ERR_ALLOC, "fbg_build_file: length_counts");
    memset(tmp + bs->length_cap, 0,
           (size_t)(new_cap - bs->length_cap) * sizeof(uint64_t));
    bs->length_counts = tmp;
    bs->length_cap = new_cap;
    return LZG_OK;
}

LZGError lzg_fbg_build_file(LZGFbg *g, const char *path,
                             LZGFbgAbundanceMode mode,
                             double smoothing,
                             LZGFbgBackoff backoff) {
    if (!g || !path || !path[0])
        return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_build_file: NULL");
    if (smoothing < 0.0)
        return LZG_FAIL(LZG_ERR_PARAM_OUT_OF_RANGE,
                        "fbg_build_file: smoothing < 0");

    FILE *fh = fopen(path, "r");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN,
                              "fbg_build_file: cannot open '%s': %s",
                              path, strerror(errno));

    BuildState bs;
    build_state_init(&bs);
    if (!bs.length_counts) {
        build_state_free(&bs);
        fclose(fh);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg_build_file: init");
    }

    uint64_t file_size = lzg_detect_regular_file_size(path);
    double start_time = lzg_build_monotonic_seconds();
    double last_log = start_time;
    long long peak_rss = lzg_build_current_rss_kb();
    LZG_INFO("fbg stream build: start file=%s size=%.1fMB",
             path, (double)file_size / (1024.0 * 1024.0));

    char *line = NULL;
    size_t line_cap = 0;
    ptrdiff_t nread;
    uint64_t lines_seen = 0;
    uint64_t sequences_seen = 0;
    uint64_t skipped_decompose = 0;
    uint64_t bytes_seen = 0;
    uint32_t max_len = 0;
    errno = 0;

    LZGFbgStep steps[LZG_FBG_MAX_STEPS];
    LZGError err = LZG_OK;

    while ((nread = lzg_getline_portable(&line, &line_cap, fh)) != -1) {
        lines_seen++;
        bytes_seen += (uint64_t)nread;

        char *seq = NULL;
        uint64_t count = 0;
        LZGParsedLineKind kind = LZG_LINE_EMPTY;
        err = lzg_parse_plain_sequence_line(line, &seq, &count, &kind);
        if (err != LZG_OK) goto fail;
        if (!seq || count == 0) continue;
        sequences_seen++;

        double w = lzg_fbg_abund_weight(mode, count);
        if (w <= 0.0) continue;

        uint32_t seq_len = (uint32_t)strlen(seq);
        if (seq_len == 0) continue;

        /* Observe alphabet incrementally. */
        bool alphabet_full = false;
        for (uint32_t i = 0; i < seq_len; i++) {
            LZGError aerr = lzg_fbg_alphabet_observe(
                g->char_to_idx, g->idx_to_char, &g->alphabet_size, seq[i]);
            if (aerr != LZG_OK) { alphabet_full = true; break; }
        }
        if (alphabet_full) { skipped_decompose++; continue; }

        /* Decompose. Bad sequences (length > 256, weird chars slipping in,
         * overflow) are skipped rather than aborting the whole build. */
        uint32_t n_steps = 0;
        LZGError derr = lzg_fbg_decompose(g, seq, seq_len, steps, &n_steps);
        if (derr != LZG_OK) { skipped_decompose++; continue; }

        for (uint32_t t = 0; t < n_steps; t++) {
            err = build_state_record_step(&bs, &steps[t], w);
            if (err != LZG_OK) goto fail;
        }

        err = bs_ensure_length_cap(&bs, seq_len);
        if (err != LZG_OK) goto fail;
        bs.length_counts[seq_len] += count;
        if (seq_len > max_len) max_len = seq_len;
        if (seq_len > bs.max_length) bs.max_length = seq_len;

        /* Periodic progress log. */
        if (lines_seen % FBG_STREAM_LOG_EVERY == 0) {
            double now = lzg_build_monotonic_seconds();
            if (now - last_log >= FBG_STREAM_LOG_MIN_SEC) {
                long long rss = lzg_build_current_rss_kb();
                if (rss > peak_rss) peak_rss = rss;
                double pct = file_size > 0
                    ? 100.0 * (double)bytes_seen / (double)file_size : 0.0;
                LZG_INFO("fbg stream: lines=%llu seqs=%llu nts=%u rules=%u "
                         "bytes=%.1fMB/%.1fMB (%.1f%%) rss=%lldMB elapsed=%.1fs "
                         "skipped=%llu",
                         (unsigned long long)lines_seen,
                         (unsigned long long)sequences_seen,
                         bs.n_nts,
                         build_state_total_rules(&bs),
                         (double)bytes_seen / (1024.0 * 1024.0),
                         (double)file_size / (1024.0 * 1024.0),
                         pct,
                         rss / 1024,
                         now - start_time,
                         (unsigned long long)skipped_decompose);
                last_log = now;
            }
        }
    }

    if (ferror(fh) || (!feof(fh) && errno != 0)) {
        int saved = errno;
        err = (saved == ENOMEM)
            ? LZG_FAIL(LZG_ERR_ALLOC, "fbg stream OOM reading '%s'", path)
            : LZG_FAIL(LZG_ERR_IO_READ, "fbg stream failed reading '%s'", path);
        goto fail;
    }

    free(line); line = NULL;
    fclose(fh); fh = NULL;

    double ingest_elapsed = lzg_build_monotonic_seconds() - start_time;
    LZG_INFO("fbg stream ingest done: lines=%llu seqs=%llu nts=%u rules=%u "
             "skipped=%llu elapsed=%.1fs",
             (unsigned long long)lines_seen,
             (unsigned long long)sequences_seen,
             bs.n_nts, build_state_total_rules(&bs),
             (unsigned long long)skipped_decompose, ingest_elapsed);

    err = finalize_from_build_state(g, &bs, mode, smoothing, backoff);
    build_state_free(&bs);
    (void)peak_rss;
    (void)max_len;
    return err;

fail:
    free(line);
    if (fh) fclose(fh);
    build_state_free(&bs);
    return err;
}

/* ── Finalize ─────────────────────────────────────────────────── */

static double spectral_radius_dense(const double *T, uint32_t n);

double lzg_fbg_spectral_radius_dense(const double *T, uint32_t n) {
    return spectral_radius_dense(T, n);
}

/* Power-iteration spectral radius for a small non-negative matrix T[n*n].
 * Returns max |λ| ≈ ρ(T). For non-negative T this is Perron-Frobenius. */
static double spectral_radius_dense(const double *T, uint32_t n) {
    if (n == 0) return 0.0;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y) { free(x); free(y); return 0.0; }

    for (uint32_t i = 0; i < n; i++) x[i] = 1.0 / (double)n;

    double lambda = 0.0;
    for (uint32_t iter = 0; iter < 200; iter++) {
        for (uint32_t i = 0; i < n; i++) {
            double s = 0.0;
            const double *row = T + (size_t)i * n;
            for (uint32_t j = 0; j < n; j++) s += row[j] * x[j];
            y[i] = s;
        }
        double norm2 = 0.0;
        for (uint32_t i = 0; i < n; i++) norm2 += y[i] * y[i];
        double norm = sqrt(norm2);
        if (norm < 1e-300) { lambda = 0.0; break; }
        double new_lambda = norm;
        for (uint32_t i = 0; i < n; i++) x[i] = y[i] / norm;
        if (fabs(new_lambda - lambda) < 1e-12 * (fabs(new_lambda) + 1e-300)) {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;
    }

    free(x); free(y);
    return lambda;
}

static LZGError finalize_from_build_state(LZGFbg *g, BuildState *bs,
                                          LZGFbgAbundanceMode mode,
                                          double smoothing,
                                          LZGFbgBackoff backoff) {
    /* Order NTs: S last (so that DP over non-S NTs completes before S). */
    uint32_t order_cap = bs->n_nts;
    uint32_t *order = (uint32_t *)malloc((size_t)order_cap * sizeof(uint32_t));
    if (!order) return LZG_FAIL(LZG_ERR_ALLOC, "fbg finalize: order alloc");
    uint32_t pos = 0;
    for (uint32_t i = 0; i < bs->n_nts; i++)
        if (!bs->nt_pool[i].is_start) order[pos++] = i;
    /* Then S, if present */
    if (bs->start_index >= 0) order[pos++] = (uint32_t)bs->start_index;
    /* ... but actually the plan says S should be at index 0 for a stable API. */
    /* Reorder: S first, then non-S. Reverse + rotate. */
    /* Simpler: fill a new array with S at [0], non-S after. */
    uint32_t *final_order = (uint32_t *)malloc((size_t)order_cap * sizeof(uint32_t));
    if (!final_order) { free(order); return LZG_FAIL(LZG_ERR_ALLOC, "fbg finalize: order2"); }
    uint32_t fpos = 0;
    if (bs->start_index >= 0) final_order[fpos++] = (uint32_t)bs->start_index;
    for (uint32_t i = 0; i < bs->n_nts; i++)
        if (!bs->nt_pool[i].is_start) final_order[fpos++] = i;
    free(order);

    uint32_t n_nts = fpos;
    if (n_nts == 0) { free(final_order); return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "fbg: no nts"); }

    /* Count total rules. */
    uint32_t n_rules_total = 0;
    for (uint32_t i = 0; i < n_nts; i++)
        n_rules_total += bs->nt_pool[final_order[i]].count;
    if (n_rules_total == 0) { free(final_order); return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "fbg: no rules"); }

    /* Allocate final arrays. */
    g->nts = (LZGFbgNT *)calloc(n_nts, sizeof(LZGFbgNT));
    g->rules = (LZGFbgRule *)calloc(n_rules_total, sizeof(LZGFbgRule));
    if (!g->nts || !g->rules) {
        free(g->nts); free(g->rules);
        g->nts = NULL; g->rules = NULL;
        free(final_order);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg finalize: alloc");
    }
    g->n_nts = n_nts;
    g->n_rules = n_rules_total;
    g->start_nt = 0;

    /* Copy rules and per-NT stats. Build counts are in .weight (double); move
     * to .count as uint64 (rounding) for display, and preserve the fractional
     * count in a parallel accumulator via .weight before we overwrite it with
     * MLE probability. */
    uint32_t n_internal = 0;
    uint32_t rule_cursor = 0;
    for (uint32_t i = 0; i < n_nts; i++) {
        BuildNT *src = &bs->nt_pool[final_order[i]];
        LZGFbgNT *dst = &g->nts[i];
        dst->a = src->a;
        dst->z = src->z;
        dst->is_start = src->is_start;
        dst->rule_offset = rule_cursor;
        dst->n_rules = src->count;

        double total = 0.0;
        for (uint32_t r = 0; r < src->count; r++) total += src->rules[r].weight;
        dst->total_count = total;
        dst->unseen_mass = 0.0;

        /* MLE with Laplace α. */
        double denom = total + smoothing * (double)src->count;
        for (uint32_t r = 0; r < src->count; r++) {
            LZGFbgRule *rule = &g->rules[rule_cursor];
            *rule = src->rules[r];
            double c = src->rules[r].weight;  /* fractional count */
            rule->count = (uint64_t)(c + 0.5);  /* rounded display count */
            if (denom > 0.0) rule->weight = (c + smoothing) / denom;
            else              rule->weight = 0.0;
            /* Preserve exact fractional count where count rounding would lose it:
             * we keep rule->count for display but use rule->weight × total to
             * recover fractional count when needed. */
            if (rule->kind == LZG_FBG_RULE_INTERNAL) n_internal++;
            rule_cursor++;
        }
    }
    g->n_internal = n_internal;

    /* Length counts. */
    g->length_counts = (uint64_t *)calloc((size_t)(bs->max_length + 1),
                                           sizeof(uint64_t));
    if (!g->length_counts) {
        free(final_order);
        return LZG_FAIL(LZG_ERR_ALLOC, "fbg finalize: length_counts");
    }
    memcpy(g->length_counts, bs->length_counts,
           (size_t)(bs->max_length + 1) * sizeof(uint64_t));
    g->max_length = bs->max_length;

    g->abundance_mode = mode;
    g->smoothing = smoothing;
    g->backoff = backoff;

    free(final_order);

    /* Finalize derived state (T, spectral radius, backoff tables). */
    return lzg_fbg_finalize(g);
}

/* The shared finalize entry point — also called by posterior / subtract after
 * they populate g->rules and g->nts directly. */
LZGError lzg_fbg_finalize(LZGFbg *g) {
    if (!g) return LZG_FAIL(LZG_ERR_NULL_ARG, "fbg_finalize: NULL");
    if (g->n_nts == 0) return LZG_FAIL(LZG_ERR_EMPTY_INPUT, "fbg_finalize: n_nts=0");

    /* Build (a, z) → nt-index lookup. Also persisted on g->nt_by_az. */
    for (uint32_t k = 0; k < LZG_FBG_ALPHABET_SIZE * LZG_FBG_ALPHABET_SIZE; k++)
        g->nt_by_az[k] = -1;
    int32_t s_idx = -1;
    for (uint32_t i = 0; i < g->n_nts; i++) {
        if (g->nts[i].is_start) {
            s_idx = (int32_t)i;
        } else {
            uint32_t key = (uint32_t)g->nts[i].a * LZG_FBG_ALPHABET_SIZE
                         + (uint32_t)g->nts[i].z;
            g->nt_by_az[key] = (int32_t)i;
        }
    }
    (void)s_idx;

    /* Local alias for the transition-matrix construction below. */
    #define IDX_OF(a, z) g->nt_by_az[(uint32_t)(a) * LZG_FBG_ALPHABET_SIZE + (uint32_t)(z)]

    /* Transition matrix T[n_nts × n_nts]: T[i][j] = Σ p(internal r: Mi → ... Mj). */
    uint32_t n = g->n_nts;
    double *T = (double *)calloc((size_t)n * n, sizeof(double));
    if (!T) return LZG_FAIL(LZG_ERR_ALLOC, "fbg_finalize: T alloc");
    for (uint32_t i = 0; i < n; i++) {
        const LZGFbgNT *nt = &g->nts[i];
        for (uint32_t r = 0; r < nt->n_rules; r++) {
            const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
            if (rule->kind != LZG_FBG_RULE_INTERNAL) continue;
            int32_t j = IDX_OF(rule->dst_a, rule->dst_z);
            if (j < 0) continue;  /* destination NT not present */
            T[(size_t)i * n + (uint32_t)j] += rule->weight;
        }
    }

    #undef IDX_OF

    g->spectral_radius = spectral_radius_dense(T, n);
    g->is_consistent = (g->spectral_radius < 1.0);
    free(T);

    /* Backoff: Good-Turing δ_nt + marginal. */
    if (g->rule_marginal) { lzg_hm_destroy(g->rule_marginal); g->rule_marginal = NULL; }
    if (g->backoff == LZG_FBG_BACKOFF_GT) {
        /* δ_nt = (# rules with count ≤ 1) / total_count, capped at 0.5. */
        for (uint32_t i = 0; i < g->n_nts; i++) {
            LZGFbgNT *nt = &g->nts[i];
            double total = nt->total_count;
            if (total <= 0.0) { nt->unseen_mass = 0.5; continue; }
            uint32_t n_sing = 0;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                /* Recover fractional count = weight × (total + smoothing·n) − smoothing.
                 * For smoothing=0 this collapses to weight × total, matching the
                 * prototype's count ≤ 1 criterion. */
                double denom = total + g->smoothing * (double)nt->n_rules;
                double frac_count = rule->weight * denom - g->smoothing;
                if (frac_count <= 1.0) n_sing++;
            }
            double delta = (double)n_sing / total;
            if (delta > 0.5) delta = 0.5;
            nt->unseen_mass = delta;
        }

        /* Marginal: sum counts by marginal key, then divide by total counts. */
        double global_total = 0.0;
        for (uint32_t i = 0; i < g->n_nts; i++) global_total += g->nts[i].total_count;
        if (global_total <= 0.0) global_total = 1.0;

        g->rule_marginal = lzg_hm_create(g->n_rules * 2);
        if (!g->rule_marginal)
            return LZG_FAIL(LZG_ERR_ALLOC, "fbg_finalize: marginal alloc");
        /* Accumulate raw counts per marginal key. */
        for (uint32_t i = 0; i < g->n_nts; i++) {
            LZGFbgNT *nt = &g->nts[i];
            double denom = nt->total_count + g->smoothing * (double)nt->n_rules;
            for (uint32_t r = 0; r < nt->n_rules; r++) {
                const LZGFbgRule *rule = &g->rules[nt->rule_offset + r];
                double c = rule->weight * denom - g->smoothing;
                if (c < 0.0) c = 0.0;
                uint64_t key = lzg_fbg_rule_marginal_key(
                    rule->kind, rule->a_char, rule->z_char,
                    rule->a_run_len, rule->z_run_len,
                    rule->dst_a, rule->dst_z);
                /* We reinterpret the uint64 value as double bits to accumulate. */
                uint64_t *slot = lzg_hm_get(g->rule_marginal, key);
                double acc = 0.0;
                if (slot) { memcpy(&acc, slot, sizeof(acc)); }
                acc += c;
                uint64_t bits;
                memcpy(&bits, &acc, sizeof(bits));
                lzg_hm_put(g->rule_marginal, key, bits);
            }
        }
        /* Normalize by global_total. */
        for (uint32_t i = 0; i < g->rule_marginal->capacity; i++) {
            uint64_t k = g->rule_marginal->keys[i];
            if (k == LZG_HM_EMPTY || k == LZG_HM_DELETED) continue;
            double c;
            memcpy(&c, &g->rule_marginal->values[i], sizeof(c));
            c /= global_total;
            uint64_t bits;
            memcpy(&bits, &c, sizeof(bits));
            g->rule_marginal->values[i] = bits;
        }
    }

    LZG_INFO("fbg finalize: n_nts=%u n_rules=%u ρ(T)=%.6f consistent=%d",
             g->n_nts, g->n_rules, g->spectral_radius, (int)g->is_consistent);
    return LZG_OK;
}
