/**
 * @file io.c
 * @brief Binary save/load for FlashBackGrammar.
 *
 * Self-contained format (not the LZG2 graph format — different mathematical
 * object, different layout). Little-endian assumed; on big-endian hosts the
 * reader/writer would need byte-swapping (not a concern for TCR repertoire
 * workflows today).
 *
 * Layout (all uint fields LE):
 *
 *   [0]  magic                    uint32   0x46424731  ("FBG1")
 *   [4]  format_version           uint16   1
 *   [6]  reserved                 uint16   0
 *   [8]  abundance_mode           uint8
 *   [9]  backoff                  uint8
 *   [10] reserved                 uint16   0
 *   [12] smoothing                float64
 *   [20] alphabet_size            uint8
 *   [21] reserved                 uint8[3]
 *   [24] idx_to_char[22]          uint8[22]   (alphabet_size valid entries)
 *   [46] reserved                 uint8[2]
 *   [48] n_nts                    uint32
 *   [52] start_nt                 uint32
 *   [56] n_rules                  uint32
 *   [60] n_internal               uint32
 *   [64] max_length               uint32
 *   [68] reserved                 uint32
 *   [72] nts[n_nts] each 32 bytes
 *   [..] rules[n_rules] each 40 bytes (LZGFlashbackGrammarRule is 24 bytes but we re-pack)
 *   [..] length_counts[max_length + 1] uint64
 *   [..] body_crc32c              uint32
 *
 * Derived state (nt_by_az, T, spectral_radius, rule_marginal) is recomputed
 * on load via lzg_flashback_grammar_finalize().
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "lzgraph/crc32c.h"

#include "flashback_grammar_internal.h"

#define FBG_MAGIC         0x46424731u    /* "FBG1" */
#define FBG_FORMAT_VER    1

/* On-disk packed NT. 32 bytes. */
typedef struct {
    uint8_t  a;
    uint8_t  z;
    uint8_t  is_start;
    uint8_t  _pad0;
    uint32_t rule_offset;
    uint32_t n_rules;
    uint32_t _pad1;
    double   total_count;
    double   unseen_mass;
} DiskNT;   /* size = 32 bytes */

/* On-disk packed rule. 32 bytes. */
typedef struct {
    uint8_t  kind;
    uint8_t  a_char;
    uint8_t  z_char;
    uint8_t  a_run_len;
    uint8_t  z_run_len;
    uint8_t  dst_a;
    uint8_t  dst_z;
    uint8_t  _pad;
    uint64_t count;
    double   weight;
} DiskRule;  /* size = 24 bytes */

/* ── save ────────────────────────────────────────────────────── */

static LZGError write_all(FILE *fh, const void *data, size_t n,
                           uint32_t *running_crc) {
    if (n == 0) return LZG_OK;
    if (fwrite(data, 1, n, fh) != n)
        return LZG_FAIL(LZG_ERR_IO_WRITE, "flashback_grammar_save: write failed (%s)",
                        strerror(errno));
    *running_crc = lzg_crc32c_update(*running_crc, data, n);
    return LZG_OK;
}

LZGError lzg_flashback_grammar_save(const LZGFlashbackGrammar *g, const char *path) {
    if (!g || !path || !path[0])
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_save: NULL argument");

    FILE *fh = fopen(path, "wb");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN, "flashback_grammar_save: cannot open %s: %s",
                              path, strerror(errno));

    uint32_t crc = 0;  /* CRC accumulates over entire body */

    /* Header. */
    uint32_t magic    = FBG_MAGIC;
    uint16_t version  = FBG_FORMAT_VER;
    uint16_t res16    = 0;
    uint8_t  mode     = (uint8_t)g->abundance_mode;
    uint8_t  backoff  = (uint8_t)g->backoff;
    double   smooth   = g->smoothing;

    LZGError err = LZG_OK;
    #define W(ptr, n) do { err = write_all(fh, (ptr), (n), &crc); if (err != LZG_OK) goto fail; } while (0)

    W(&magic,   4);
    W(&version, 2);
    W(&res16,   2);
    W(&mode,    1);
    W(&backoff, 1);
    W(&res16,   2);
    W(&smooth,  8);

    /* Alphabet. 1 + 3 pad + 22 chars + 2 pad = 28 bytes */
    uint8_t pad3[3] = {0, 0, 0};
    uint8_t pad2[2] = {0, 0};
    W(&g->alphabet_size, 1);
    W(pad3, 3);
    W(g->idx_to_char, LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE);
    W(pad2, 2);

    /* Counts. */
    uint32_t n_nts_u32 = g->n_nts;
    uint32_t start_u32 = g->start_nt;
    uint32_t n_rules_u32 = g->n_rules;
    uint32_t n_internal_u32 = g->n_internal;
    uint32_t max_length_u32 = g->max_length;
    uint32_t res32 = 0;
    W(&n_nts_u32,      4);
    W(&start_u32,      4);
    W(&n_rules_u32,    4);
    W(&n_internal_u32, 4);
    W(&max_length_u32, 4);
    W(&res32,          4);

    /* NTs. */
    for (uint32_t i = 0; i < g->n_nts; i++) {
        const LZGFlashbackGrammarNT *nt = &g->nts[i];
        DiskNT d = {0};
        d.a           = nt->a;
        d.z           = nt->z;
        d.is_start    = nt->is_start;
        d.rule_offset = nt->rule_offset;
        d.n_rules     = nt->n_rules;
        d.total_count = nt->total_count;
        d.unseen_mass = nt->unseen_mass;
        W(&d, sizeof(DiskNT));
    }

    /* Rules. */
    for (uint32_t i = 0; i < g->n_rules; i++) {
        const LZGFlashbackGrammarRule *r = &g->rules[i];
        DiskRule d = {0};
        d.kind      = r->kind;
        d.a_char    = r->a_char;
        d.z_char    = r->z_char;
        d.a_run_len = r->a_run_len;
        d.z_run_len = r->z_run_len;
        d.dst_a     = r->dst_a;
        d.dst_z     = r->dst_z;
        d.count     = r->count;
        d.weight    = r->weight;
        W(&d, sizeof(DiskRule));
    }

    /* Length counts. */
    if (g->max_length > 0 && g->length_counts) {
        size_t n = (size_t)g->max_length + 1;
        W(g->length_counts, n * sizeof(uint64_t));
    }

    /* Trailing CRC. */
    if (fwrite(&crc, 1, 4, fh) != 4) {
        err = LZG_FAIL(LZG_ERR_IO_WRITE, "flashback_grammar_save: trailing crc write failed");
        goto fail;
    }

    #undef W
    fclose(fh);
    LZG_INFO("fbg saved: %s  n_nts=%u  n_rules=%u  bytes written",
             path, g->n_nts, g->n_rules);
    return LZG_OK;

fail:
    fclose(fh);
    return err;
}

/* ── load ────────────────────────────────────────────────────── */

static LZGError read_all(FILE *fh, void *buf, size_t n, uint32_t *running_crc) {
    if (n == 0) return LZG_OK;
    if (fread(buf, 1, n, fh) != n)
        return LZG_FAIL(LZG_ERR_IO_READ, "flashback_grammar_load: read failed (%s)",
                        feof(fh) ? "EOF" : strerror(errno));
    *running_crc = lzg_crc32c_update(*running_crc, buf, n);
    return LZG_OK;
}

LZGError lzg_flashback_grammar_load(const char *path, LZGFlashbackGrammar **out) {
    if (!path || !out)
        return LZG_FAIL(LZG_ERR_NULL_ARG, "flashback_grammar_load: NULL argument");
    *out = NULL;

    FILE *fh = fopen(path, "rb");
    if (!fh) return LZG_FAIL(LZG_ERR_IO_OPEN, "flashback_grammar_load: cannot open %s: %s",
                              path, strerror(errno));

    uint32_t crc = 0;
    LZGError err = LZG_OK;
    #define R(ptr, n) do { err = read_all(fh, (ptr), (n), &crc); if (err != LZG_OK) goto fail; } while (0)

    uint32_t magic;
    uint16_t version, res16;
    uint8_t  mode, backoff;
    double   smoothing;

    R(&magic,     4);
    R(&version,   2);
    R(&res16,     2);
    R(&mode,      1);
    R(&backoff,   1);
    R(&res16,     2);
    R(&smoothing, 8);

    if (magic != FBG_MAGIC) {
        err = LZG_FAIL(LZG_ERR_IO_CORRUPT,
                       "flashback_grammar_load: bad magic 0x%08X (expected 0x%08X)",
                       magic, FBG_MAGIC);
        goto fail;
    }
    if (version != FBG_FORMAT_VER) {
        err = LZG_FAIL(LZG_ERR_IO_VERSION,
                       "flashback_grammar_load: format version %u unsupported (want %u)",
                       version, FBG_FORMAT_VER);
        goto fail;
    }

    /* Alphabet. */
    uint8_t alphabet_size;
    uint8_t pad3[3];
    uint8_t idx_to_char[LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE];
    uint8_t pad2[2];
    R(&alphabet_size, 1);
    R(pad3, 3);
    R(idx_to_char, LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE);
    R(pad2, 2);

    if (alphabet_size > LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE) {
        err = LZG_FAIL(LZG_ERR_IO_CORRUPT,
                       "flashback_grammar_load: alphabet_size=%u exceeds cap %u",
                       alphabet_size, LZG_FLASHBACK_GRAMMAR_ALPHABET_SIZE);
        goto fail;
    }

    /* Counts. */
    uint32_t n_nts, start_nt, n_rules, n_internal, max_length, res32;
    R(&n_nts,      4);
    R(&start_nt,   4);
    R(&n_rules,    4);
    R(&n_internal, 4);
    R(&max_length, 4);
    R(&res32,      4);

    /* Allocate grammar. */
    LZGFlashbackGrammar *g = lzg_flashback_grammar_new();
    if (!g) { err = LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_load: new"); goto fail; }

    /* Rebuild alphabet. */
    memset(g->char_to_idx, LZG_FLASHBACK_GRAMMAR_CHAR_UNKNOWN, 256);
    for (uint8_t i = 0; i < alphabet_size; i++) {
        uint8_t ch = idx_to_char[i];
        g->idx_to_char[i] = ch;
        g->char_to_idx[ch] = i;
    }
    g->alphabet_size = alphabet_size;

    g->n_nts = n_nts;
    g->start_nt = start_nt;
    g->n_rules = n_rules;
    g->n_internal = n_internal;
    g->max_length = max_length;
    g->abundance_mode = (LZGFlashbackGrammarAbundanceMode)mode;
    g->smoothing = smoothing;
    g->backoff = (LZGFlashbackGrammarBackoff)backoff;

    g->nts = (LZGFlashbackGrammarNT *)calloc(n_nts, sizeof(LZGFlashbackGrammarNT));
    g->rules = (LZGFlashbackGrammarRule *)calloc(n_rules, sizeof(LZGFlashbackGrammarRule));
    if (!g->nts || !g->rules) {
        lzg_flashback_grammar_destroy(g);
        err = LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_load: nts/rules");
        goto fail;
    }

    /* NTs. */
    for (uint32_t i = 0; i < n_nts; i++) {
        DiskNT d;
        err = read_all(fh, &d, sizeof(d), &crc);
        if (err != LZG_OK) { lzg_flashback_grammar_destroy(g); goto fail; }
        g->nts[i].a           = d.a;
        g->nts[i].z           = d.z;
        g->nts[i].is_start    = d.is_start;
        g->nts[i].rule_offset = d.rule_offset;
        g->nts[i].n_rules     = d.n_rules;
        g->nts[i].total_count = d.total_count;
        g->nts[i].unseen_mass = d.unseen_mass;
    }

    /* Rules. */
    for (uint32_t i = 0; i < n_rules; i++) {
        DiskRule d;
        err = read_all(fh, &d, sizeof(d), &crc);
        if (err != LZG_OK) { lzg_flashback_grammar_destroy(g); goto fail; }
        g->rules[i].kind      = d.kind;
        g->rules[i].a_char    = d.a_char;
        g->rules[i].z_char    = d.z_char;
        g->rules[i].a_run_len = d.a_run_len;
        g->rules[i].z_run_len = d.z_run_len;
        g->rules[i].dst_a     = d.dst_a;
        g->rules[i].dst_z     = d.dst_z;
        g->rules[i].count     = d.count;
        g->rules[i].weight    = d.weight;
    }

    /* Length counts. */
    if (max_length > 0) {
        size_t n = (size_t)max_length + 1;
        g->length_counts = (uint64_t *)calloc(n, sizeof(uint64_t));
        if (!g->length_counts) {
            lzg_flashback_grammar_destroy(g);
            err = LZG_FAIL(LZG_ERR_ALLOC, "flashback_grammar_load: length_counts");
            goto fail;
        }
        err = read_all(fh, g->length_counts, n * sizeof(uint64_t), &crc);
        if (err != LZG_OK) { lzg_flashback_grammar_destroy(g); goto fail; }
    }

    /* CRC check. */
    uint32_t expected_crc;
    if (fread(&expected_crc, 1, 4, fh) != 4) {
        lzg_flashback_grammar_destroy(g);
        err = LZG_FAIL(LZG_ERR_IO_READ, "flashback_grammar_load: CRC read failed");
        goto fail;
    }
    if (expected_crc != crc) {
        lzg_flashback_grammar_destroy(g);
        err = LZG_FAIL(LZG_ERR_IO_CORRUPT,
                       "flashback_grammar_load: CRC mismatch (file=0x%08X, computed=0x%08X)",
                       expected_crc, crc);
        goto fail;
    }

    fclose(fh);
    #undef R

    /* Rebuild derived state (nt_by_az, T, spectral_radius, backoff tables). */
    LZGError ferr = lzg_flashback_grammar_finalize(g);
    if (ferr != LZG_OK) { lzg_flashback_grammar_destroy(g); return ferr; }

    *out = g;
    LZG_INFO("fbg loaded: %s  n_nts=%u  n_rules=%u",
             path, g->n_nts, g->n_rules);
    return LZG_OK;

fail:
    fclose(fh);
    return err;
}
