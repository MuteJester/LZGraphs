#include "io_reader.h"
#include "io_internal.h"
#include "lzgraph/crc32c.h"
#include "lzgraph/gene_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_counts_u64_or_u32(uint64_t *dst,
                                   const uint8_t *src,
                                   uint32_t n,
                                   bool counts64) {
    if (counts64) {
        memcpy(dst, src, (uint64_t)n * 8);
    } else {
        const uint32_t *counts32 = (const uint32_t *)src;
        for (uint32_t i = 0; i < n; i++)
            dst[i] = counts32[i];
    }
}

static LZGError read_header(FILE *file,
                            const char *path,
                            LZGIOHeader *header) {
    if (fread(header, 64, 1, file) != 1) {
        return LZG_FAIL(LZG_ERR_IO_READ,
                        "failed to read LZG header from '%s'",
                        path);
    }
    if (header->magic != LZG_IO_MAGIC) {
        return LZG_FAIL(LZG_ERR_IO_CORRUPT,
                        "invalid magic number in '%s' (not an LZG file)",
                        path);
    }
    if (header->min_reader_version > LZG_IO_FORMAT_VERSION) {
        return LZG_FAIL(LZG_ERR_IO_VERSION,
                        "file '%s' requires format version %u (we support %u)",
                        path,
                        header->min_reader_version,
                        LZG_IO_FORMAT_VERSION);
    }
    if (header->format_version < 2 ||
        header->format_version > LZG_IO_FORMAT_VERSION) {
        return LZG_FAIL(LZG_ERR_IO_VERSION,
                        "unsupported format version %u in '%s'",
                        header->format_version,
                        path);
    }
    if (header->endian_tag != LZG_IO_ENDIAN_LE) {
        return LZG_FAIL(LZG_ERR_IO_CORRUPT,
                        "unsupported endianness in '%s'",
                        path);
    }

    {
        uint32_t header_crc = lzg_crc32c(header, 40);
        if (header_crc != header->header_crc32c) {
            return LZG_FAIL(LZG_ERR_IO_CORRUPT,
                            "header CRC mismatch in '%s'",
                            path);
        }
    }

    return LZG_OK;
}

static LZGError read_section_table(FILE *file,
                                   uint32_t n_sections,
                                   LZGIOSectionEntry **out_table) {
    *out_table = malloc((uint64_t)n_sections * sizeof(LZGIOSectionEntry));
    if (!*out_table) return LZG_ERR_ALLOC;

    if (fread(*out_table, 32, n_sections, file) != n_sections) {
        free(*out_table);
        *out_table = NULL;
        return LZG_ERR_IO;
    }

    return LZG_OK;
}

static LZGError load_strp_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_STRP);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    g->pool = lzg_sp_create(1024);
    if (!g->pool) {
        free(buf);
        return LZG_ERR_ALLOC;
    }

    {
        uint8_t *cursor = buf;
        uint32_t count = 0;
        memcpy(&count, cursor, 4);
        cursor += 4;
        for (uint32_t i = 0; i < count; i++) {
            uint16_t len = 0;
            memcpy(&len, cursor, 2);
            cursor += 2;
            lzg_sp_intern_n(g->pool, (const char *)cursor, len);
            cursor += len;
        }
    }

    free(buf);
    return LZG_OK;
}

static LZGError load_csra_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_CSRA);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    g->row_offsets = malloc(((uint64_t)g->n_nodes + 1) * 4);
    g->col_indices = malloc((uint64_t)g->n_edges * 4);
    if (!g->row_offsets || !g->col_indices) {
        free(buf);
        return LZG_ERR_ALLOC;
    }

    memcpy(g->row_offsets, buf, ((uint64_t)g->n_nodes + 1) * 4);
    memcpy(g->col_indices, buf + ((uint64_t)g->n_nodes + 1) * 4,
           (uint64_t)g->n_edges * 4);
    free(buf);
    return LZG_OK;
}

static LZGError load_ewgt_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  bool counts64,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_EWGT);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    g->edge_weights = malloc((uint64_t)g->n_edges * 8);
    g->edge_counts = malloc((uint64_t)g->n_edges * 8);
    if (!g->edge_weights || !g->edge_counts) {
        free(buf);
        return LZG_ERR_ALLOC;
    }

    memcpy(g->edge_weights, buf, (uint64_t)g->n_edges * 8);
    copy_counts_u64_or_u32(g->edge_counts,
                           buf + (uint64_t)g->n_edges * 8,
                           g->n_edges,
                           counts64);

    free(buf);
    return LZG_OK;
}

static LZGError load_elzc_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_ELZC);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    {
        uint32_t sp_len_padded = (uint32_t)lzg_io_align8(g->n_edges);
        g->edge_sp_id = malloc((uint64_t)g->n_edges * 4);
        g->edge_sp_len = malloc(g->n_edges);
        g->edge_prefix_id = malloc((uint64_t)g->n_edges * 4);
        if (!g->edge_sp_id || !g->edge_sp_len || !g->edge_prefix_id) {
            free(buf);
            return LZG_ERR_ALLOC;
        }

        memcpy(g->edge_sp_id, buf, (uint64_t)g->n_edges * 4);
        memcpy(g->edge_sp_len, buf + (uint64_t)g->n_edges * 4, g->n_edges);
        memcpy(g->edge_prefix_id,
               buf + (uint64_t)g->n_edges * 4 + sp_len_padded,
               (uint64_t)g->n_edges * 4);
    }

    free(buf);
    return LZG_OK;
}

static LZGError load_node_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  bool counts64,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_NODE);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    {
        uint32_t sp_len_padded = (uint32_t)lzg_io_align8(g->n_nodes);
        uint8_t *cursor = buf;

        g->outgoing_counts = malloc((uint64_t)g->n_nodes * 8);
        g->node_sp_id = malloc((uint64_t)g->n_nodes * 4);
        g->node_sp_len = malloc(g->n_nodes);
        g->node_pos = malloc((uint64_t)g->n_nodes * 4);
        g->node_is_sink = malloc(g->n_nodes);
        if (!g->outgoing_counts || !g->node_sp_id || !g->node_sp_len ||
            !g->node_pos || !g->node_is_sink) {
            free(buf);
            return LZG_ERR_ALLOC;
        }

        copy_counts_u64_or_u32(g->outgoing_counts, cursor, g->n_nodes, counts64);
        cursor += counts64 ? (uint64_t)g->n_nodes * 8 : (uint64_t)g->n_nodes * 4;
        memcpy(g->node_sp_id, cursor, (uint64_t)g->n_nodes * 4);
        cursor += (uint64_t)g->n_nodes * 4;
        memcpy(g->node_sp_len, cursor, g->n_nodes);
        cursor += sp_len_padded;
        memcpy(g->node_pos, cursor, (uint64_t)g->n_nodes * 4);
        cursor += (uint64_t)g->n_nodes * 4;
        memcpy(g->node_is_sink, cursor, g->n_nodes);
    }

    free(buf);
    return LZG_OK;
}

static LZGError load_lend_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  bool counts64,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_LEND);
    if (!entry) return LZG_ERR_IO;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    memcpy(&g->max_length, buf, 4);
    g->length_counts = malloc(((uint64_t)g->max_length + 1) * 8);
    if (!g->length_counts) {
        free(buf);
        return LZG_ERR_ALLOC;
    }

    copy_counts_u64_or_u32(g->length_counts,
                           buf + 4,
                           g->max_length + 1,
                           counts64);

    free(buf);
    return LZG_OK;
}

static void skip_meta_value(uint8_t **cursor, uint8_t value_type) {
    switch (value_type) {
        case 0x01:
            *cursor += 4;
            break;
        case 0x02:
        case 0x03:
            *cursor += 8;
            break;
        case 0x04: {
            uint16_t slen = 0;
            memcpy(&slen, *cursor, 2);
            *cursor += 2 + slen;
            break;
        }
        case 0x05:
            *cursor += 1;
            break;
        default:
            break;
    }
}

static LZGError load_meta_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_META);
    if (!entry) return LZG_OK;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    {
        uint8_t *cursor = buf;
        uint32_t n_entries = 0;
        memcpy(&n_entries, cursor, 4);
        cursor += 4;

        for (uint32_t i = 0; i < n_entries; i++) {
            uint16_t key_len = 0;
            char key[256];
            uint8_t value_type = 0;

            memcpy(&key_len, cursor, 2);
            cursor += 2;
            memcpy(key, cursor, key_len);
            key[key_len] = '\0';
            cursor += key_len;
            value_type = *cursor++;

            if (strcmp(key, "smoothing_alpha") == 0 && value_type == 0x03) {
                memcpy(&g->smoothing_alpha, cursor, 8);
                cursor += 8;
            } else if (strcmp(key, "min_initial_count") == 0 && value_type == 0x01) {
                memcpy(&g->smoothing_alpha, cursor, 4);
                cursor += 4;
            } else {
                skip_meta_value(&cursor, value_type);
            }
        }
    }

    free(buf);
    return LZG_OK;
}

static LZGError load_gene_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  bool counts64,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_GENE);
    if (!entry) return LZG_OK;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    {
        uint8_t *cursor = buf;
        uint32_t pool_count = 0;
        LZGGeneData *gd = lzg_gene_data_create();
        if (!gd) {
            free(buf);
            return LZG_ERR_ALLOC;
        }

        memcpy(&gd->n_v_genes, cursor, 4); cursor += 4;
        memcpy(&gd->n_j_genes, cursor, 4); cursor += 4;
        memcpy(&gd->n_vj_pairs, cursor, 4); cursor += 4;
        memcpy(&gd->total_v_entries, cursor, 4); cursor += 4;
        memcpy(&gd->total_j_entries, cursor, 4); cursor += 4;
        memcpy(&pool_count, cursor, 4); cursor += 4;

        gd->gene_pool = lzg_sp_create(pool_count + 16);
        if (!gd->gene_pool) {
            lzg_gene_data_destroy(gd);
            free(buf);
            return LZG_ERR_ALLOC;
        }

        for (uint32_t i = 0; i < pool_count; i++) {
            uint16_t slen = 0;
            memcpy(&slen, cursor, 2);
            cursor += 2;
            lzg_sp_intern_n(gd->gene_pool, (const char *)cursor, slen);
            cursor += slen;
        }

        gd->v_marginal_ids = malloc((uint64_t)gd->n_v_genes * 4);
        gd->v_marginal_probs = malloc((uint64_t)gd->n_v_genes * 8);
        gd->j_marginal_ids = malloc((uint64_t)gd->n_j_genes * 4);
        gd->j_marginal_probs = malloc((uint64_t)gd->n_j_genes * 8);
        gd->vj_v_ids = malloc((uint64_t)gd->n_vj_pairs * 4);
        gd->vj_j_ids = malloc((uint64_t)gd->n_vj_pairs * 4);
        gd->vj_probs = malloc((uint64_t)gd->n_vj_pairs * 8);
        gd->v_offsets = malloc(((uint64_t)g->n_edges + 1) * 4);
        gd->v_gene_ids = malloc((uint64_t)gd->total_v_entries * 4);
        gd->v_gene_counts = malloc((uint64_t)gd->total_v_entries * 8);
        gd->j_offsets = malloc(((uint64_t)g->n_edges + 1) * 4);
        gd->j_gene_ids = malloc((uint64_t)gd->total_j_entries * 4);
        gd->j_gene_counts = malloc((uint64_t)gd->total_j_entries * 8);

        memcpy(gd->v_marginal_ids, cursor, (uint64_t)gd->n_v_genes * 4);
        cursor += (uint64_t)gd->n_v_genes * 4;
        memcpy(gd->v_marginal_probs, cursor, (uint64_t)gd->n_v_genes * 8);
        cursor += (uint64_t)gd->n_v_genes * 8;
        memcpy(gd->j_marginal_ids, cursor, (uint64_t)gd->n_j_genes * 4);
        cursor += (uint64_t)gd->n_j_genes * 4;
        memcpy(gd->j_marginal_probs, cursor, (uint64_t)gd->n_j_genes * 8);
        cursor += (uint64_t)gd->n_j_genes * 8;
        memcpy(gd->vj_v_ids, cursor, (uint64_t)gd->n_vj_pairs * 4);
        cursor += (uint64_t)gd->n_vj_pairs * 4;
        memcpy(gd->vj_j_ids, cursor, (uint64_t)gd->n_vj_pairs * 4);
        cursor += (uint64_t)gd->n_vj_pairs * 4;
        memcpy(gd->vj_probs, cursor, (uint64_t)gd->n_vj_pairs * 8);
        cursor += (uint64_t)gd->n_vj_pairs * 8;

        memcpy(gd->v_offsets, cursor, ((uint64_t)g->n_edges + 1) * 4);
        cursor += ((uint64_t)g->n_edges + 1) * 4;
        memcpy(gd->v_gene_ids, cursor, (uint64_t)gd->total_v_entries * 4);
        cursor += (uint64_t)gd->total_v_entries * 4;
        copy_counts_u64_or_u32(gd->v_gene_counts, cursor, gd->total_v_entries, counts64);
        cursor += counts64 ? (uint64_t)gd->total_v_entries * 8
                           : (uint64_t)gd->total_v_entries * 4;

        memcpy(gd->j_offsets, cursor, ((uint64_t)g->n_edges + 1) * 4);
        cursor += ((uint64_t)g->n_edges + 1) * 4;
        memcpy(gd->j_gene_ids, cursor, (uint64_t)gd->total_j_entries * 4);
        cursor += (uint64_t)gd->total_j_entries * 4;
        copy_counts_u64_or_u32(gd->j_gene_counts, cursor, gd->total_j_entries, counts64);

        g->gene_data = gd;
    }

    free(buf);
    return LZG_OK;
}

static LZGError load_topo_section(FILE *file,
                                  const LZGIOSectionEntry *sec_table,
                                  uint32_t n_sections,
                                  LZGGraph *g) {
    const LZGIOSectionEntry *entry =
        lzg_io_find_section(sec_table, n_sections, LZG_IO_TAG_TOPO);
    if (!entry) return LZG_OK;

    uint8_t *buf = NULL;
    LZGError err = lzg_io_read_section(file, entry, &buf);
    if (err != LZG_OK) return err;

    g->topo_order = malloc((uint64_t)g->n_nodes * 4);
    if (!g->topo_order) {
        free(buf);
        return LZG_ERR_ALLOC;
    }

    memcpy(g->topo_order, buf, (uint64_t)g->n_nodes * 4);
    g->topo_valid = true;
    free(buf);
    return LZG_OK;
}

static void reconstruct_root_node(LZGGraph *g) {
    g->root_node = UINT32_MAX;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        const char *sp = lzg_sp_get(g->pool, g->node_sp_id[i]);
        if (g->node_sp_len[i] == 1 && sp[0] == LZG_START_SENTINEL) {
            g->root_node = i;
            break;
        }
    }
}

LZGError lzg_graph_load_impl(const char *path, LZGGraph **out) {
    if (!path) return LZG_FAIL(LZG_ERR_NULL_ARG, "file path is NULL");
    if (!out) return LZG_FAIL(LZG_ERR_NULL_ARG, "output pointer is NULL");

    LZGError err = LZG_OK;
    FILE *file = fopen(path, "rb");
    LZGIOHeader header;
    LZGIOSectionEntry *sec_table = NULL;
    LZGGraph *g = NULL;
    bool counts64 = false;

    if (!file)
        return LZG_FAIL(LZG_ERR_IO_OPEN, "cannot open file '%s'", path);

    err = read_header(file, path, &header);
    if (err != LZG_OK) goto fail;

    err = read_section_table(file, header.n_sections, &sec_table);
    if (err != LZG_OK) goto fail;

    g = calloc(1, sizeof(LZGGraph));
    if (!g) {
        err = LZG_ERR_ALLOC;
        goto fail;
    }
    g->variant = (LZGVariant)header.variant;
    g->n_nodes = header.n_nodes;
    g->n_edges = header.n_edges;
    counts64 = header.format_version >= 3;

    err = load_strp_section(file, sec_table, header.n_sections, g);
    if (err != LZG_OK) goto fail;
    err = load_csra_section(file, sec_table, header.n_sections, g);
    if (err != LZG_OK) goto fail;
    err = load_ewgt_section(file, sec_table, header.n_sections, counts64, g);
    if (err != LZG_OK) goto fail;
    err = load_elzc_section(file, sec_table, header.n_sections, g);
    if (err != LZG_OK) goto fail;
    err = load_node_section(file, sec_table, header.n_sections, counts64, g);
    if (err != LZG_OK) goto fail;
    err = load_lend_section(file, sec_table, header.n_sections, counts64, g);
    if (err != LZG_OK) goto fail;
    err = load_meta_section(file, sec_table, header.n_sections, g);
    if (err != LZG_OK) goto fail;
    err = load_gene_section(file, sec_table, header.n_sections, counts64, g);
    if (err != LZG_OK) goto fail;
    err = load_topo_section(file, sec_table, header.n_sections, g);
    if (err != LZG_OK) goto fail;

    fclose(file);
    free(sec_table);

    if (!g->topo_valid) {
        err = lzg_graph_topo_sort(g);
        if (err != LZG_OK) {
            lzg_graph_destroy(g);
            return err;
        }
    }

    reconstruct_root_node(g);
    (void)lzg_graph_ensure_query_edge_hashes(g);

    *out = g;
    return LZG_OK;

fail:
    if (file) fclose(file);
    free(sec_table);
    if (g) lzg_graph_destroy(g);
    return err == LZG_OK ? LZG_ERR_IO : err;
}
