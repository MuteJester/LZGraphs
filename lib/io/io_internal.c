#include "io_internal.h"
#include "lzgraph/crc32c.h"
#include <stdlib.h>
#include <string.h>

uint64_t lzg_io_align8(uint64_t v) {
    return (v + 7) & ~(uint64_t)7;
}

LZGIOSectionBuf *lzg_io_section_buf_create(uint32_t tag, uint64_t size) {
    LZGIOSectionBuf *section = calloc(1, sizeof(LZGIOSectionBuf));
    if (!section) return NULL;

    section->tag = tag;
    section->size = size;
    section->data = calloc(1, size > 0 ? size : 1);
    if (!section->data) {
        free(section);
        return NULL;
    }

    return section;
}

void lzg_io_section_buf_destroy(LZGIOSectionBuf *section) {
    if (!section) return;
    free(section->data);
    free(section);
}

LZGIOSectionBuf *lzg_io_build_raw_section(uint32_t tag,
                                          const void *data,
                                          uint64_t bytes) {
    LZGIOSectionBuf *section = lzg_io_section_buf_create(tag, bytes);
    if (!section) return NULL;

    memcpy(section->data, data, bytes);
    section->crc = lzg_crc32c(section->data, section->size);
    return section;
}

void lzg_io_destroy_section_array(LZGIOSectionBuf **sections,
                                  uint32_t n_sections) {
    if (!sections) return;
    for (uint32_t i = 0; i < n_sections; i++)
        lzg_io_section_buf_destroy(sections[i]);
}

const LZGIOSectionEntry *lzg_io_find_section(const LZGIOSectionEntry *sec_table,
                                             uint32_t n_sections,
                                             uint32_t tag) {
    if (!sec_table) return NULL;

    for (uint32_t i = 0; i < n_sections; i++) {
        if (sec_table[i].tag == tag)
            return &sec_table[i];
    }

    return NULL;
}

LZGError lzg_io_read_section(FILE *file,
                             const LZGIOSectionEntry *entry,
                             uint8_t **out_buf) {
    if (!file || !entry || !out_buf) return LZG_ERR_INVALID_ARG;

    *out_buf = malloc(entry->size);
    if (!*out_buf) return LZG_ERR_ALLOC;

    fseek(file, (long)entry->offset, SEEK_SET);
    if (fread(*out_buf, entry->size, 1, file) != 1) {
        free(*out_buf);
        *out_buf = NULL;
        return LZG_ERR_IO;
    }

    {
        uint32_t crc = lzg_crc32c(*out_buf, entry->size);
        if (crc != entry->crc32c) {
            free(*out_buf);
            *out_buf = NULL;
            return LZG_ERR_IO;
        }
    }

    return LZG_OK;
}
