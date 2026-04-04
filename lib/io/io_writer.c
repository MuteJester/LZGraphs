#include "io_writer.h"
#include "io_internal.h"
#include "lzgraph/crc32c.h"
#include "lzgraph/gene_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    LZGIOSectionBuf *items[LZG_IO_MAX_SECTIONS];
    uint32_t count;
} LZGIOSectionList;

static void destroy_section_list(LZGIOSectionList *list) {
    if (!list) return;
    lzg_io_destroy_section_array(list->items, list->count);
}

static LZGError append_section(LZGIOSectionList *list, LZGIOSectionBuf *section) {
    if (!list || !section) return LZG_ERR_ALLOC;
    list->items[list->count++] = section;
    return LZG_OK;
}

static LZGIOSectionBuf *build_strp_section(const LZGGraph *g) {
    uint64_t size = 4;
    for (uint32_t i = 0; i < g->pool->count; i++)
        size += 2 + lzg_sp_len(g->pool, i);

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_STRP, size);
    if (!section) return NULL;

    {
        uint8_t *cursor = section->data;
        uint32_t count = g->pool->count;
        memcpy(cursor, &count, 4);
        cursor += 4;

        for (uint32_t i = 0; i < count; i++) {
            uint16_t len = (uint16_t)lzg_sp_len(g->pool, i);
            memcpy(cursor, &len, 2);
            cursor += 2;
            memcpy(cursor, lzg_sp_get(g->pool, i), len);
            cursor += len;
        }
    }

    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_csra_section(const LZGGraph *g) {
    uint32_t nn = g->n_nodes;
    uint32_t ne = g->n_edges;
    uint64_t size = (uint64_t)(nn + 1) * 4 + (uint64_t)ne * 4;

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_CSRA, size);
    if (!section) return NULL;

    memcpy(section->data, g->row_offsets, (nn + 1) * 4);
    memcpy(section->data + (nn + 1) * 4, g->col_indices, ne * 4);
    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_ewgt_section(const LZGGraph *g) {
    uint32_t ne = g->n_edges;
    uint64_t size = (uint64_t)ne * 8 + (uint64_t)ne * 8;

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_EWGT, size);
    if (!section) return NULL;

    memcpy(section->data, g->edge_weights, ne * 8);
    memcpy(section->data + ne * 8, g->edge_counts, ne * 8);
    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_elzc_section(const LZGGraph *g) {
    uint32_t ne = g->n_edges;
    uint32_t sp_len_padded = (uint32_t)lzg_io_align8(ne);
    uint64_t size = (uint64_t)ne * 4 + sp_len_padded + (uint64_t)ne * 4;

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_ELZC, size);
    if (!section) return NULL;

    {
        uint8_t *cursor = section->data;
        memcpy(cursor, g->edge_sp_id, ne * 4);
        cursor += ne * 4;
        memcpy(cursor, g->edge_sp_len, ne);
        cursor += sp_len_padded;
        memcpy(cursor, g->edge_prefix_id, ne * 4);
    }

    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_node_section(const LZGGraph *g) {
    uint32_t nn = g->n_nodes;
    uint32_t sp_len_padded = (uint32_t)lzg_io_align8(nn);
    uint64_t size = (uint64_t)nn * 8 + (uint64_t)nn * 4 + sp_len_padded
                    + (uint64_t)nn * 4 + nn;

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_NODE, size);
    if (!section) return NULL;

    {
        uint8_t *cursor = section->data;
        memcpy(cursor, g->outgoing_counts, nn * 8);
        cursor += nn * 8;
        memcpy(cursor, g->node_sp_id, nn * 4);
        cursor += nn * 4;
        memcpy(cursor, g->node_sp_len, nn);
        cursor += sp_len_padded;
        memcpy(cursor, g->node_pos, nn * 4);
        cursor += nn * 4;
        memcpy(cursor, g->node_is_sink, nn);
    }

    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_lend_section(const LZGGraph *g) {
    uint64_t size = 4 + ((uint64_t)g->max_length + 1) * 8;
    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_LEND, size);
    if (!section) return NULL;

    memcpy(section->data, &g->max_length, 4);
    memcpy(section->data + 4, g->length_counts, ((uint64_t)g->max_length + 1) * 8);
    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_meta_section(const LZGGraph *g) {
    uint64_t size = 4 + (2 + 15 + 1 + 8);
    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_META, size);
    if (!section) return NULL;

    {
        uint8_t *cursor = section->data;
        uint32_t n_entries = 1;
        uint16_t key_len = 15;

        memcpy(cursor, &n_entries, 4);
        cursor += 4;
        memcpy(cursor, &key_len, 2);
        cursor += 2;
        memcpy(cursor, "smoothing_alpha", 15);
        cursor += 15;
        *cursor++ = 0x03;
        memcpy(cursor, &g->smoothing_alpha, 8);
    }

    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGIOSectionBuf *build_gene_section(const LZGGraph *g) {
    if (!g->gene_data) return NULL;

    LZGGeneData *gd = (LZGGeneData *)g->gene_data;
    uint32_t ne = g->n_edges;
    uint64_t size = 6 * 4;
    for (uint32_t i = 0; i < gd->gene_pool->count; i++)
        size += 2 + lzg_sp_len(gd->gene_pool, i);
    size += (uint64_t)gd->n_v_genes * (4 + 8);
    size += (uint64_t)gd->n_j_genes * (4 + 8);
    size += (uint64_t)gd->n_vj_pairs * (4 + 4 + 8);
    size += (uint64_t)(ne + 1) * 4 + (uint64_t)gd->total_v_entries * (4 + 8);
    size += (uint64_t)(ne + 1) * 4 + (uint64_t)gd->total_j_entries * (4 + 8);

    LZGIOSectionBuf *section = lzg_io_section_buf_create(LZG_IO_TAG_GENE, size);
    if (!section) return NULL;

    {
        uint8_t *cursor = section->data;
        uint32_t pool_count = gd->gene_pool->count;

        memcpy(cursor, &gd->n_v_genes, 4); cursor += 4;
        memcpy(cursor, &gd->n_j_genes, 4); cursor += 4;
        memcpy(cursor, &gd->n_vj_pairs, 4); cursor += 4;
        memcpy(cursor, &gd->total_v_entries, 4); cursor += 4;
        memcpy(cursor, &gd->total_j_entries, 4); cursor += 4;
        memcpy(cursor, &pool_count, 4); cursor += 4;

        for (uint32_t i = 0; i < pool_count; i++) {
            uint16_t slen = (uint16_t)lzg_sp_len(gd->gene_pool, i);
            memcpy(cursor, &slen, 2);
            cursor += 2;
            memcpy(cursor, lzg_sp_get(gd->gene_pool, i), slen);
            cursor += slen;
        }

        memcpy(cursor, gd->v_marginal_ids, gd->n_v_genes * 4); cursor += gd->n_v_genes * 4;
        memcpy(cursor, gd->v_marginal_probs, gd->n_v_genes * 8); cursor += gd->n_v_genes * 8;
        memcpy(cursor, gd->j_marginal_ids, gd->n_j_genes * 4); cursor += gd->n_j_genes * 4;
        memcpy(cursor, gd->j_marginal_probs, gd->n_j_genes * 8); cursor += gd->n_j_genes * 8;
        memcpy(cursor, gd->vj_v_ids, gd->n_vj_pairs * 4); cursor += gd->n_vj_pairs * 4;
        memcpy(cursor, gd->vj_j_ids, gd->n_vj_pairs * 4); cursor += gd->n_vj_pairs * 4;
        memcpy(cursor, gd->vj_probs, gd->n_vj_pairs * 8); cursor += gd->n_vj_pairs * 8;
        memcpy(cursor, gd->v_offsets, (ne + 1) * 4); cursor += (ne + 1) * 4;
        memcpy(cursor, gd->v_gene_ids, gd->total_v_entries * 4); cursor += gd->total_v_entries * 4;
        memcpy(cursor, gd->v_gene_counts, gd->total_v_entries * 8); cursor += gd->total_v_entries * 8;
        memcpy(cursor, gd->j_offsets, (ne + 1) * 4); cursor += (ne + 1) * 4;
        memcpy(cursor, gd->j_gene_ids, gd->total_j_entries * 4); cursor += gd->total_j_entries * 4;
        memcpy(cursor, gd->j_gene_counts, gd->total_j_entries * 8);
    }

    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

static LZGError build_sections(const LZGGraph *g, LZGIOSectionList *sections) {
    LZGError err = append_section(sections, build_strp_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_csra_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_ewgt_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_elzc_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_node_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_lend_section(g));
    if (err != LZG_OK) return err;
    err = append_section(sections, build_meta_section(g));
    if (err != LZG_OK) return err;

    if (g->gene_data) {
        err = append_section(sections, build_gene_section(g));
        if (err != LZG_OK) return err;
    }

    if (g->topo_valid && g->topo_order) {
        err = append_section(sections,
                             lzg_io_build_raw_section(LZG_IO_TAG_TOPO,
                                                      g->topo_order,
                                                      (uint64_t)g->n_nodes * 4));
        if (err != LZG_OK) return err;
    }

    return LZG_OK;
}

static uint64_t *compute_section_offsets(const LZGIOSectionList *sections) {
    uint64_t base = 64 + (uint64_t)sections->count * 32;
    base = lzg_io_align8(base);

    uint64_t *offsets = malloc(sections->count * sizeof(uint64_t));
    if (!offsets) return NULL;

    for (uint32_t i = 0; i < sections->count; i++) {
        offsets[i] = base;
        base = lzg_io_align8(base + sections->items[i]->size);
    }

    return offsets;
}

static void fill_header(const LZGGraph *g,
                        uint32_t n_sections,
                        LZGIOHeader *header) {
    memset(header, 0, sizeof(*header));
    header->magic = LZG_IO_MAGIC;
    header->format_version = LZG_IO_FORMAT_VERSION;
    header->min_reader_version = LZG_IO_FORMAT_VERSION;
    header->endian_tag = LZG_IO_ENDIAN_LE;
    header->variant = (uint8_t)g->variant;
    header->compression = 0;
    header->flags = 0;
    header->n_sections = n_sections;
    header->n_nodes = g->n_nodes;
    header->n_edges = g->n_edges;
    header->n_sequences = 0;
    header->creation_timestamp = (uint64_t)time(NULL);
    header->lib_version_major = 3;
    header->lib_version_minor = 0;
    header->lib_version_patch = 1;
    header->header_crc32c = lzg_crc32c(header, 40);
}

LZGError lzg_graph_save_impl(const LZGGraph *g, const char *path) {
    if (!g) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph pointer is NULL");
    if (!path) return LZG_FAIL(LZG_ERR_NULL_ARG, "file path is NULL");

    LZGError err = LZG_OK;
    LZGIOSectionList sections = {0};
    uint64_t *offsets = NULL;
    FILE *file = NULL;

    err = build_sections(g, &sections);
    if (err != LZG_OK) goto cleanup;

    offsets = compute_section_offsets(&sections);
    if (!offsets) {
        err = LZG_ERR_ALLOC;
        goto cleanup;
    }

    file = fopen(path, "wb");
    if (!file) {
        err = LZG_ERR_IO;
        goto cleanup;
    }

    {
        LZGIOHeader header;
        fill_header(g, sections.count, &header);
        fwrite(&header, 64, 1, file);
    }

    for (uint32_t i = 0; i < sections.count; i++) {
        LZGIOSectionEntry entry = {0};
        entry.tag = sections.items[i]->tag;
        entry.flags = 0;
        entry.offset = offsets[i];
        entry.size = sections.items[i]->size;
        entry.uncompressed_size = 0;
        entry.crc32c = sections.items[i]->crc;
        fwrite(&entry, 32, 1, file);
    }

    {
        uint64_t cursor = 64 + (uint64_t)sections.count * 32;
        while (cursor < offsets[0]) {
            fputc(0, file);
            cursor++;
        }
    }

    for (uint32_t i = 0; i < sections.count; i++) {
        fwrite(sections.items[i]->data, sections.items[i]->size, 1, file);

        {
            uint64_t next = (i + 1 < sections.count)
                ? offsets[i + 1]
                : lzg_io_align8(offsets[i] + sections.items[i]->size);
            uint64_t end = offsets[i] + sections.items[i]->size;
            while (end < next) {
                fputc(0, file);
                end++;
            }
        }
    }

    {
        long file_size = ftell(file);
        uint8_t *file_buf = NULL;
        uint32_t file_crc = 0;

        fseek(file, 0, SEEK_SET);
        file_buf = malloc(file_size);
        if (!file_buf) {
            err = LZG_ERR_ALLOC;
            goto cleanup;
        }
        {
            size_t read_count = fread(file_buf, file_size, 1, file);
            (void)read_count;
        }
        file_crc = lzg_crc32c(file_buf, file_size);
        free(file_buf);

        fseek(file, 0, SEEK_END);
        {
            LZGIOTrailer trailer = { file_crc, LZG_IO_TRAILER_MAGIC };
            fwrite(&trailer, 8, 1, file);
        }
    }

cleanup:
    if (file) fclose(file);
    free(offsets);
    destroy_section_list(&sections);
    return err;
}
