#ifndef LZGRAPH_IO_INTERNAL_H
#define LZGRAPH_IO_INTERNAL_H

#include "lzgraph/io.h"
#include <stdio.h>

#define LZG_IO_MAX_SECTIONS 20u

typedef struct {
    uint32_t tag;
    uint8_t *data;
    uint64_t size;
    uint32_t crc;
} LZGIOSectionBuf;

uint64_t lzg_io_align8(uint64_t v);

LZGIOSectionBuf *lzg_io_section_buf_create(uint32_t tag, uint64_t size);

void lzg_io_section_buf_destroy(LZGIOSectionBuf *section);

LZGIOSectionBuf *lzg_io_build_raw_section(uint32_t tag,
                                          const void *data,
                                          uint64_t bytes);

void lzg_io_destroy_section_array(LZGIOSectionBuf **sections,
                                  uint32_t n_sections);

const LZGIOSectionEntry *lzg_io_find_section(const LZGIOSectionEntry *sec_table,
                                             uint32_t n_sections,
                                             uint32_t tag);

LZGError lzg_io_read_section(FILE *file,
                             const LZGIOSectionEntry *entry,
                             uint8_t **out_buf);

#endif /* LZGRAPH_IO_INTERNAL_H */
