/**
 * @file io.c
 * @brief LZG2/3 section-based binary serialization with CRC-32C.
 */
#include "lzgraph/io.h"
#include "lzgraph/crc32c.h"
#include "lzgraph/gene_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Helpers ───────────────────────────────────────────────── */

static uint64_t align8(uint64_t v) { return (v + 7) & ~(uint64_t)7; }

/* Section buffer for building before writing */
typedef struct {
    uint32_t tag;
    uint8_t *data;
    uint64_t size;
    uint32_t crc;
} SectionBuf;

static SectionBuf *sbuf_create(uint32_t tag, uint64_t size) {
    SectionBuf *s = calloc(1, sizeof(SectionBuf));
    s->tag = tag;
    s->size = size;
    s->data = calloc(1, size > 0 ? size : 1);
    return s;
}

static void sbuf_destroy(SectionBuf *s) {
    if (s) { free(s->data); free(s); }
}

/* ═══════════════════════════════════════════════════════════════ */
/* WRITER                                                          */
/* ═══════════════════════════════════════════════════════════════ */

/* Build a section buffer for the string pool */
static SectionBuf *build_strp(const LZGGraph *g) {
    /* Calculate size: 4 (count) + Σ (2 + len_i) */
    uint64_t sz = 4;
    for (uint32_t i = 0; i < g->pool->count; i++)
        sz += 2 + lzg_sp_len(g->pool, i);

    SectionBuf *s = sbuf_create(LZG_IO_TAG_STRP, sz);
    uint8_t *p = s->data;

    uint32_t count = g->pool->count;
    memcpy(p, &count, 4); p += 4;

    for (uint32_t i = 0; i < count; i++) {
        uint16_t len = (uint16_t)lzg_sp_len(g->pool, i);
        memcpy(p, &len, 2); p += 2;
        memcpy(p, lzg_sp_get(g->pool, i), len); p += len;
    }

    s->crc = lzg_crc32c(s->data, s->size);
    return s;
}

/* Macro: build a section from a raw array */
#define BUILD_RAW_SECTION(tag_val, ptr, bytes) do { \
    SectionBuf *_s = sbuf_create(tag_val, bytes); \
    memcpy(_s->data, ptr, bytes); \
    _s->crc = lzg_crc32c(_s->data, _s->size); \
    sections[n_sec++] = _s; \
} while(0)

LZGError lzg_graph_save(const LZGGraph *g, const char *path) {
    if (!g) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph pointer is NULL");
    if (!path) return LZG_FAIL(LZG_ERR_NULL_ARG, "file path is NULL");

    uint32_t nn = g->n_nodes, ne = g->n_edges;
    SectionBuf *sections[20];
    uint32_t n_sec = 0;

    /* ── Build all sections ── */

    /* STRP: String pool */
    sections[n_sec++] = build_strp(g);

    /* CSRA: CSR adjacency */
    {
        uint64_t sz = (nn + 1) * 4 + ne * 4;
        SectionBuf *s = sbuf_create(LZG_IO_TAG_CSRA, sz);
        memcpy(s->data, g->row_offsets, (nn + 1) * 4);
        memcpy(s->data + (nn + 1) * 4, g->col_indices, ne * 4);
        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* EWGT: Edge weights + counts */
    {
        uint64_t sz = ne * 8 + ne * 8;
        SectionBuf *s = sbuf_create(LZG_IO_TAG_EWGT, sz);
        memcpy(s->data, g->edge_weights, ne * 8);
        memcpy(s->data + ne * 8, g->edge_counts, ne * 8);
        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* ELZC: Edge LZ constraints */
    {
        uint32_t sp_len_padded = align8(ne);
        uint64_t sz = ne * 4 + sp_len_padded + ne * 4;
        SectionBuf *s = sbuf_create(LZG_IO_TAG_ELZC, sz);
        uint8_t *p = s->data;
        memcpy(p, g->edge_sp_id, ne * 4); p += ne * 4;
        memcpy(p, g->edge_sp_len, ne); p += sp_len_padded;
        memcpy(p, g->edge_prefix_id, ne * 4);
        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* NODE: Per-node data */
    {
        uint32_t sp_len_padded = align8(nn);
        uint64_t sz = nn * 8 + nn * 4 + sp_len_padded + nn * 4 + nn; /* outgoing, sp_id, sp_len, pos, is_sink */
        SectionBuf *s = sbuf_create(LZG_IO_TAG_NODE, sz);
        uint8_t *p = s->data;
        memcpy(p, g->outgoing_counts, nn * 8); p += nn * 8;
        memcpy(p, g->node_sp_id, nn * 4); p += nn * 4;
        memcpy(p, g->node_sp_len, nn); p += sp_len_padded;
        memcpy(p, g->node_pos, nn * 4); p += nn * 4;
        memcpy(p, g->node_is_sink, nn);
        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* INIT and TERM sections removed — sentinel model uses root_node + node_is_sink */

    /* LEND: Length distribution */
    {
        uint64_t sz = 4 + (g->max_length + 1) * 8;
        SectionBuf *s = sbuf_create(LZG_IO_TAG_LEND, sz);
        memcpy(s->data, &g->max_length, 4);
        memcpy(s->data + 4, g->length_counts, (g->max_length + 1) * 8);
        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* META: Metadata (key-value pairs) */
    {
        uint64_t sz = 4 + (2 + 15 + 1 + 8); /* n_entries + smoothing_alpha */
        SectionBuf *s = sbuf_create(LZG_IO_TAG_META, sz);
        uint8_t *p = s->data;

        uint32_t n_entries = 1;
        memcpy(p, &n_entries, 4); p += 4;

        /* smoothing_alpha */
        uint16_t kl = 15;
        memcpy(p, &kl, 2); p += 2;
        memcpy(p, "smoothing_alpha", 15); p += 15;
        *p++ = 0x03; /* f64 */
        memcpy(p, &g->smoothing_alpha, 8);

        s->crc = lzg_crc32c(s->data, s->size);
        sections[n_sec++] = s;
    }

    /* GENE: V/J gene data (optional) */
    if (g->gene_data) {
        LZGGeneData *gd = (LZGGeneData *)g->gene_data;
        /* Calculate size */
        uint64_t gene_sz = 6 * 4;  /* header: nv, nj, nvj, tv, tj, pool_count */
        /* Gene string pool */
        for (uint32_t i = 0; i < gd->gene_pool->count; i++)
            gene_sz += 2 + lzg_sp_len(gd->gene_pool, i);
        /* Marginals */
        gene_sz += gd->n_v_genes * (4 + 8);  /* ids + probs */
        gene_sz += gd->n_j_genes * (4 + 8);
        /* VJ joint */
        gene_sz += gd->n_vj_pairs * (4 + 4 + 8);
        /* Per-edge CSR */
        gene_sz += (ne + 1) * 4 + gd->total_v_entries * (4 + 8);  /* v_offsets + ids + counts */
        gene_sz += (ne + 1) * 4 + gd->total_j_entries * (4 + 8);  /* j_offsets + ids + counts */

        SectionBuf *gs = sbuf_create(LZG_IO_TAG_GENE, gene_sz);
        uint8_t *gp = gs->data;

        /* Header */
        memcpy(gp, &gd->n_v_genes, 4); gp += 4;
        memcpy(gp, &gd->n_j_genes, 4); gp += 4;
        memcpy(gp, &gd->n_vj_pairs, 4); gp += 4;
        memcpy(gp, &gd->total_v_entries, 4); gp += 4;
        memcpy(gp, &gd->total_j_entries, 4); gp += 4;
        uint32_t gpool_count = gd->gene_pool->count;
        memcpy(gp, &gpool_count, 4); gp += 4;

        /* Gene string pool */
        for (uint32_t i = 0; i < gpool_count; i++) {
            uint16_t slen = (uint16_t)lzg_sp_len(gd->gene_pool, i);
            memcpy(gp, &slen, 2); gp += 2;
            memcpy(gp, lzg_sp_get(gd->gene_pool, i), slen); gp += slen;
        }

        /* V marginals */
        memcpy(gp, gd->v_marginal_ids, gd->n_v_genes * 4); gp += gd->n_v_genes * 4;
        memcpy(gp, gd->v_marginal_probs, gd->n_v_genes * 8); gp += gd->n_v_genes * 8;
        /* J marginals */
        memcpy(gp, gd->j_marginal_ids, gd->n_j_genes * 4); gp += gd->n_j_genes * 4;
        memcpy(gp, gd->j_marginal_probs, gd->n_j_genes * 8); gp += gd->n_j_genes * 8;
        /* VJ joint */
        memcpy(gp, gd->vj_v_ids, gd->n_vj_pairs * 4); gp += gd->n_vj_pairs * 4;
        memcpy(gp, gd->vj_j_ids, gd->n_vj_pairs * 4); gp += gd->n_vj_pairs * 4;
        memcpy(gp, gd->vj_probs, gd->n_vj_pairs * 8); gp += gd->n_vj_pairs * 8;
        /* Per-edge V CSR */
        memcpy(gp, gd->v_offsets, (ne + 1) * 4); gp += (ne + 1) * 4;
        memcpy(gp, gd->v_gene_ids, gd->total_v_entries * 4); gp += gd->total_v_entries * 4;
        memcpy(gp, gd->v_gene_counts, gd->total_v_entries * 8); gp += gd->total_v_entries * 8;
        /* Per-edge J CSR */
        memcpy(gp, gd->j_offsets, (ne + 1) * 4); gp += (ne + 1) * 4;
        memcpy(gp, gd->j_gene_ids, gd->total_j_entries * 4); gp += gd->total_j_entries * 4;
        memcpy(gp, gd->j_gene_counts, gd->total_j_entries * 8); gp += gd->total_j_entries * 8;

        gs->crc = lzg_crc32c(gs->data, gs->size);
        sections[n_sec++] = gs;
    }

    /* TOPO: Topological order (optional, saves recomputation on load) */
    if (g->topo_valid && g->topo_order) {
        BUILD_RAW_SECTION(LZG_IO_TAG_TOPO, g->topo_order, nn * 4);
    }

    /* ── Compute offsets ── */

    uint64_t base = 64 + (uint64_t)n_sec * 32;
    base = align8(base);

    uint64_t *offsets = malloc(n_sec * sizeof(uint64_t));
    for (uint32_t i = 0; i < n_sec; i++) {
        offsets[i] = base;
        base = align8(base + sections[i]->size);
    }

    /* ── Write file ── */

    FILE *f = fopen(path, "wb");
    if (!f) { free(offsets); return LZG_ERR_IO; }

    /* Header */
    LZGIOHeader hdr = {0};
    hdr.magic = LZG_IO_MAGIC;
    hdr.format_version = LZG_IO_FORMAT_VERSION;
    hdr.min_reader_version = LZG_IO_FORMAT_VERSION;
    hdr.endian_tag = LZG_IO_ENDIAN_LE;
    hdr.variant = (uint8_t)g->variant;
    hdr.compression = 0;
    hdr.flags = 0;
    hdr.n_sections = n_sec;
    hdr.n_nodes = nn;
    hdr.n_edges = ne;
    hdr.n_sequences = 0; /* not tracked currently */
    hdr.creation_timestamp = (uint64_t)time(NULL);
    hdr.lib_version_major = 3;
    hdr.lib_version_minor = 0;
    hdr.lib_version_patch = 1;
    hdr.header_crc32c = lzg_crc32c(&hdr, 40);

    fwrite(&hdr, 64, 1, f);

    /* Section table */
    for (uint32_t i = 0; i < n_sec; i++) {
        LZGIOSectionEntry entry = {0};
        entry.tag = sections[i]->tag;
        entry.flags = 0;
        entry.offset = offsets[i];
        entry.size = sections[i]->size;
        entry.uncompressed_size = 0;
        entry.crc32c = sections[i]->crc;
        fwrite(&entry, 32, 1, f);
    }

    /* Pad to first section */
    uint64_t cur = 64 + (uint64_t)n_sec * 32;
    while (cur < offsets[0]) { fputc(0, f); cur++; }

    /* Section data with alignment padding */
    for (uint32_t i = 0; i < n_sec; i++) {
        fwrite(sections[i]->data, sections[i]->size, 1, f);
        uint64_t next = (i + 1 < n_sec) ? offsets[i + 1] : align8(offsets[i] + sections[i]->size);
        uint64_t end = offsets[i] + sections[i]->size;
        while (end < next) { fputc(0, f); end++; }
    }

    /* Compute file CRC over everything written so far */
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *file_buf = malloc(file_size);
    fread(file_buf, file_size, 1, f);
    uint32_t file_crc = lzg_crc32c(file_buf, file_size);
    free(file_buf);

    /* Trailer */
    fseek(f, 0, SEEK_END);
    LZGIOTrailer trailer = { file_crc, LZG_IO_TRAILER_MAGIC };
    fwrite(&trailer, 8, 1, f);

    fclose(f);

    /* Cleanup */
    for (uint32_t i = 0; i < n_sec; i++) sbuf_destroy(sections[i]);
    free(offsets);

    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* READER                                                          */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_load(const char *path, LZGGraph **out) {
    if (!path) return LZG_FAIL(LZG_ERR_NULL_ARG, "file path is NULL");
    if (!out) return LZG_FAIL(LZG_ERR_NULL_ARG, "output pointer is NULL");

    FILE *f = fopen(path, "rb");
    if (!f) return LZG_FAIL(LZG_ERR_IO_OPEN, "cannot open file '%s'", path);

    /* ── Header ── */
    LZGIOHeader hdr;
    if (fread(&hdr, 64, 1, f) != 1) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_READ, "failed to read LZG header from '%s'", path);
    }
    if (hdr.magic != LZG_IO_MAGIC) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_CORRUPT, "invalid magic number in '%s' (not an LZG file)", path);
    }
    if (hdr.min_reader_version > LZG_IO_FORMAT_VERSION) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_VERSION, "file '%s' requires format version %u (we support %u)", path, hdr.min_reader_version, LZG_IO_FORMAT_VERSION);
    }
    if (hdr.format_version < 2 || hdr.format_version > LZG_IO_FORMAT_VERSION) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_VERSION, "unsupported format version %u in '%s'", hdr.format_version, path);
    }
    if (hdr.endian_tag != LZG_IO_ENDIAN_LE) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_CORRUPT, "unsupported endianness in '%s'", path);
    }
    uint32_t hdr_crc = lzg_crc32c(&hdr, 40);
    if (hdr_crc != hdr.header_crc32c) {
        fclose(f); return LZG_FAIL(LZG_ERR_IO_CORRUPT, "header CRC mismatch in '%s'", path);
    }

    /* ── Section table ── */
    uint32_t n_sec = hdr.n_sections;
    LZGIOSectionEntry *sec_table = malloc(n_sec * sizeof(LZGIOSectionEntry));
    if (fread(sec_table, 32, n_sec, f) != n_sec) {
        free(sec_table); fclose(f); return LZG_ERR_IO;
    }

    /* Helper: linear scan for a section by tag (n_sec ≤ 20) */
    #define FIND_SECTION(tag_val, out_ptr) do { \
        out_ptr = NULL; \
        for (uint32_t _i = 0; _i < n_sec; _i++) \
            if (sec_table[_i].tag == (tag_val)) { out_ptr = &sec_table[_i]; break; } \
    } while(0)

    /* Helper: read and verify a section */
    #define READ_SECTION(entry, buf) do { \
        buf = malloc((entry)->size); \
        fseek(f, (long)(entry)->offset, SEEK_SET); \
        if (fread(buf, (entry)->size, 1, f) != 1) goto fail; \
        uint32_t _crc = lzg_crc32c(buf, (entry)->size); \
        if (_crc != (entry)->crc32c) goto fail; \
    } while(0)

    /* ── Allocate graph ── */
    LZGGraph *g = calloc(1, sizeof(LZGGraph));
    g->variant = (LZGVariant)hdr.variant;
    g->n_nodes = hdr.n_nodes;
    g->n_edges = hdr.n_edges;
    uint32_t nn = g->n_nodes, ne = g->n_edges;
    bool counts64 = hdr.format_version >= 3;

    /* ── STRP: String pool ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_STRP, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);

        g->pool = lzg_sp_create(1024);
        uint8_t *p = buf;
        uint32_t count; memcpy(&count, p, 4); p += 4;
        for (uint32_t i = 0; i < count; i++) {
            uint16_t len; memcpy(&len, p, 2); p += 2;
            lzg_sp_intern_n(g->pool, (const char *)p, len); p += len;
        }
        free(buf);
    }

    /* ── CSRA ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_CSRA, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);
        g->row_offsets = malloc((nn + 1) * 4);
        g->col_indices = malloc(ne * 4);
        memcpy(g->row_offsets, buf, (nn + 1) * 4);
        memcpy(g->col_indices, buf + (nn + 1) * 4, ne * 4);
        free(buf);
    }

    /* ── EWGT ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_EWGT, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);
        g->edge_weights = malloc(ne * 8);
        g->edge_counts  = malloc(ne * 8);
        memcpy(g->edge_weights, buf, ne * 8);
        if (counts64) {
            memcpy(g->edge_counts, buf + ne * 8, ne * 8);
        } else {
            uint32_t *src = (uint32_t *)(buf + ne * 8);
            for (uint32_t i = 0; i < ne; i++) g->edge_counts[i] = src[i];
        }
        free(buf);
    }

    /* ── ELZC ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_ELZC, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);
        uint32_t sp_len_padded = align8(ne);
        g->edge_sp_id    = malloc(ne * 4);
        g->edge_sp_len   = malloc(ne);
        g->edge_prefix_id = malloc(ne * 4);
        memcpy(g->edge_sp_id, buf, ne * 4);
        memcpy(g->edge_sp_len, buf + ne * 4, ne);
        memcpy(g->edge_prefix_id, buf + ne * 4 + sp_len_padded, ne * 4);
        free(buf);
    }

    /* ── NODE ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_NODE, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);
        uint32_t sp_len_padded = align8(nn);
        g->outgoing_counts = malloc(nn * 8);
        g->node_sp_id      = malloc(nn * 4);
        g->node_sp_len     = malloc(nn);
        g->node_pos        = malloc(nn * 4);
        g->node_is_sink    = malloc(nn);
        uint8_t *p = buf;
        if (counts64) {
            memcpy(g->outgoing_counts, p, nn * 8); p += nn * 8;
        } else {
            uint32_t *src = (uint32_t *)p;
            for (uint32_t i = 0; i < nn; i++) g->outgoing_counts[i] = src[i];
            p += nn * 4;
        }
        memcpy(g->node_sp_id, p, nn * 4); p += nn * 4;
        memcpy(g->node_sp_len, p, nn); p += sp_len_padded;
        memcpy(g->node_pos, p, nn * 4); p += nn * 4;
        memcpy(g->node_is_sink, p, nn);
        free(buf);
    }

    /* INIT and TERM sections removed — sentinel model uses root_node + node_is_sink */

    /* ── LEND ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_LEND, e);
        if (!e) goto fail;
        uint8_t *buf; READ_SECTION(e, buf);
        memcpy(&g->max_length, buf, 4);
        g->length_counts = malloc((g->max_length + 1) * 8);
        if (counts64) {
            memcpy(g->length_counts, buf + 4, (g->max_length + 1) * 8);
        } else {
            uint32_t *src = (uint32_t *)(buf + 4);
            for (uint32_t i = 0; i <= g->max_length; i++) g->length_counts[i] = src[i];
        }
        free(buf);
    }

    /* ── META ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_META, e);
        if (e) {
            uint8_t *buf; READ_SECTION(e, buf);
            uint8_t *p = buf;
            uint32_t n_entries; memcpy(&n_entries, p, 4); p += 4;
            for (uint32_t i = 0; i < n_entries; i++) {
                uint16_t kl; memcpy(&kl, p, 2); p += 2;
                char key[256];
                memcpy(key, p, kl); key[kl] = '\0'; p += kl;
                uint8_t vtype = *p++;
                if (strcmp(key, "smoothing_alpha") == 0 && vtype == 0x03) {
                    memcpy(&g->smoothing_alpha, p, 8); p += 8;
                } else if (strcmp(key, "min_initial_count") == 0 && vtype == 0x01) {
                    memcpy(&g->smoothing_alpha /* deprecated */, p, 4); p += 4;
                } else {
                    /* Skip unknown key-value */
                    switch (vtype) {
                        case 0x01: p += 4; break;
                        case 0x02: p += 8; break;
                        case 0x03: p += 8; break;
                        case 0x04: { uint16_t sl; memcpy(&sl, p, 2); p += 2 + sl; break; }
                        case 0x05: p += 1; break;
                        default: break;
                    }
                }
            }
            free(buf);
        }
    }

    /* ── GENE (optional) ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_GENE, e);
        if (e) {
            uint8_t *buf; READ_SECTION(e, buf);
            uint8_t *rp = buf;

            LZGGeneData *gd = lzg_gene_data_create();
            memcpy(&gd->n_v_genes, rp, 4); rp += 4;
            memcpy(&gd->n_j_genes, rp, 4); rp += 4;
            memcpy(&gd->n_vj_pairs, rp, 4); rp += 4;
            memcpy(&gd->total_v_entries, rp, 4); rp += 4;
            memcpy(&gd->total_j_entries, rp, 4); rp += 4;
            uint32_t gpool_count;
            memcpy(&gpool_count, rp, 4); rp += 4;

            /* Gene string pool */
            gd->gene_pool = lzg_sp_create(gpool_count + 16);
            for (uint32_t gi = 0; gi < gpool_count; gi++) {
                uint16_t slen; memcpy(&slen, rp, 2); rp += 2;
                lzg_sp_intern_n(gd->gene_pool, (const char *)rp, slen);
                rp += slen;
            }

            /* V marginals */
            gd->v_marginal_ids = malloc(gd->n_v_genes * 4);
            gd->v_marginal_probs = malloc(gd->n_v_genes * 8);
            memcpy(gd->v_marginal_ids, rp, gd->n_v_genes * 4); rp += gd->n_v_genes * 4;
            memcpy(gd->v_marginal_probs, rp, gd->n_v_genes * 8); rp += gd->n_v_genes * 8;

            /* J marginals */
            gd->j_marginal_ids = malloc(gd->n_j_genes * 4);
            gd->j_marginal_probs = malloc(gd->n_j_genes * 8);
            memcpy(gd->j_marginal_ids, rp, gd->n_j_genes * 4); rp += gd->n_j_genes * 4;
            memcpy(gd->j_marginal_probs, rp, gd->n_j_genes * 8); rp += gd->n_j_genes * 8;

            /* VJ joint */
            gd->vj_v_ids = malloc(gd->n_vj_pairs * 4);
            gd->vj_j_ids = malloc(gd->n_vj_pairs * 4);
            gd->vj_probs = malloc(gd->n_vj_pairs * 8);
            memcpy(gd->vj_v_ids, rp, gd->n_vj_pairs * 4); rp += gd->n_vj_pairs * 4;
            memcpy(gd->vj_j_ids, rp, gd->n_vj_pairs * 4); rp += gd->n_vj_pairs * 4;
            memcpy(gd->vj_probs, rp, gd->n_vj_pairs * 8); rp += gd->n_vj_pairs * 8;

            /* Per-edge V CSR */
            gd->v_offsets = malloc((ne + 1) * 4);
            gd->v_gene_ids = malloc(gd->total_v_entries * 4);
            gd->v_gene_counts = malloc(gd->total_v_entries * 8);
            memcpy(gd->v_offsets, rp, (ne + 1) * 4); rp += (ne + 1) * 4;
            memcpy(gd->v_gene_ids, rp, gd->total_v_entries * 4); rp += gd->total_v_entries * 4;
            if (counts64) {
                memcpy(gd->v_gene_counts, rp, gd->total_v_entries * 8); rp += gd->total_v_entries * 8;
            } else {
                uint32_t *src = (uint32_t *)rp;
                for (uint32_t i = 0; i < gd->total_v_entries; i++) gd->v_gene_counts[i] = src[i];
                rp += gd->total_v_entries * 4;
            }

            /* Per-edge J CSR */
            gd->j_offsets = malloc((ne + 1) * 4);
            gd->j_gene_ids = malloc(gd->total_j_entries * 4);
            gd->j_gene_counts = malloc(gd->total_j_entries * 8);
            memcpy(gd->j_offsets, rp, (ne + 1) * 4); rp += (ne + 1) * 4;
            memcpy(gd->j_gene_ids, rp, gd->total_j_entries * 4); rp += gd->total_j_entries * 4;
            if (counts64) {
                memcpy(gd->j_gene_counts, rp, gd->total_j_entries * 8); rp += gd->total_j_entries * 8;
            } else {
                uint32_t *src = (uint32_t *)rp;
                for (uint32_t i = 0; i < gd->total_j_entries; i++) gd->j_gene_counts[i] = src[i];
                rp += gd->total_j_entries * 4;
            }

            g->gene_data = gd;
            free(buf);
        }
    }

    /* ── TOPO (optional) ── */
    {
        LZGIOSectionEntry *e; FIND_SECTION(LZG_IO_TAG_TOPO, e);
        if (e) {
            uint8_t *buf; READ_SECTION(e, buf);
            g->topo_order = malloc(nn * 4);
            memcpy(g->topo_order, buf, nn * 4);
            g->topo_valid = true;
            free(buf);
        }
    }

    fclose(f);
    free(sec_table);
    /* Recompute topo if not loaded */
    if (!g->topo_valid) {
        LZGError err = lzg_graph_topo_sort(g);
        if (err != LZG_OK) { lzg_graph_destroy(g); return err; }
    }

    /* Reconstruct root_node from node_is_sink (already loaded) */
    g->root_node = UINT32_MAX;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        if (g->node_sp_len[i] == 1 && sp[0] == LZG_START_SENTINEL) {
            g->root_node = i;
            break;
        }
    }

    (void)lzg_graph_ensure_query_edge_hashes(g);

    *out = g;
    return LZG_OK;

fail:
    fclose(f);
    free(sec_table);
    if (g) lzg_graph_destroy(g);
    return LZG_ERR_IO;

    #undef FIND_SECTION
    #undef READ_SECTION
}
