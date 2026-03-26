/**
 * @file io.h
 * @brief LZG2 binary file format: section-based save/load with CRC-32C.
 *
 * File extension: .lzg
 * See FILE_FORMAT.md for the complete specification.
 */
#ifndef LZGRAPH_IO_H
#define LZGRAPH_IO_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"

/* ── Format constants ──────────────────────────────────────── */

#define LZG_IO_MAGIC           0x4C5A4732u  /* "LZG2" */
#define LZG_IO_FORMAT_VERSION         2
#define LZG_IO_ENDIAN_LE       0x01
#define LZG_IO_TRAILER_MAGIC   0x454E444Cu  /* "ENDL" */

/* Section tags (4-byte ASCII as u32 LE) */
#define LZG_IO_TAG_STRP  0x50525453u  /* String Pool          */
#define LZG_IO_TAG_CSRA  0x41525343u  /* CSR Adjacency        */
#define LZG_IO_TAG_EWGT  0x54475745u  /* Edge Weights         */
#define LZG_IO_TAG_ELZC  0x435A4C45u  /* Edge LZ Constraints  */
#define LZG_IO_TAG_NODE  0x45444F4Eu  /* Node Data            */
#define LZG_IO_TAG_INIT  0x54494E49u  /* Initial States       */
#define LZG_IO_TAG_TERM  0x4D524554u  /* Terminal States      */
#define LZG_IO_TAG_LEND  0x444E454Cu  /* Length Distribution   */
#define LZG_IO_TAG_META  0x4154454Du  /* Metadata             */
#define LZG_IO_TAG_RFRM  0x4D524652u  /* Reading Frames (NDP) */
#define LZG_IO_TAG_DICT  0x54434944u  /* Dictionary (Naive)   */
#define LZG_IO_TAG_GENE  0x454E4547u  /* Gene Data            */
#define LZG_IO_TAG_TOPO  0x4F504F54u  /* Topological Order    */

/* ── Packed struct support (GCC/Clang vs MSVC) ────────────── */

#ifdef _MSC_VER
  #define LZG_PACKED_BEGIN __pragma(pack(push, 1))
  #define LZG_PACKED_END   __pragma(pack(pop))
  #define LZG_PACKED_ATTR
#else
  #define LZG_PACKED_BEGIN
  #define LZG_PACKED_END
  #define LZG_PACKED_ATTR  __attribute__((packed))
#endif

/* ── File header (64 bytes) ────────────────────────────────── */

LZG_PACKED_BEGIN
typedef struct LZG_PACKED_ATTR {
    uint32_t magic;               /* 0x4C5A4732 */
    uint16_t format_version;      /* 2 */
    uint16_t min_reader_version;  /* 2 */
    uint8_t  endian_tag;          /* 0x01 = LE */
    uint8_t  variant;             /* 0=AAP, 1=NDP, 2=Naive */
    uint8_t  compression;         /* 0=none, 1=zstd, 2=gzip */
    uint8_t  flags;               /* bit 0: has_gene_data */
    uint32_t n_sections;
    uint32_t n_nodes;
    uint32_t n_edges;
    uint64_t n_sequences;
    uint64_t creation_timestamp;
    uint32_t header_crc32c;
    uint16_t lib_version_major;
    uint16_t lib_version_minor;
    uint16_t lib_version_patch;
    uint8_t  _reserved[14];
} LZGIOHeader;
LZG_PACKED_END

/* ── Section table entry (32 bytes) ────────────────────────── */

LZG_PACKED_BEGIN
typedef struct LZG_PACKED_ATTR {
    uint32_t tag;
    uint32_t flags;             /* bit 0: compressed */
    uint64_t offset;            /* absolute file offset */
    uint64_t size;              /* bytes on disk */
    uint32_t uncompressed_size; /* 0 if not compressed */
    uint32_t crc32c;            /* CRC of section data */
} LZGIOSectionEntry;
LZG_PACKED_END

/* ── File trailer (8 bytes) ────────────────────────────────── */

LZG_PACKED_BEGIN
typedef struct LZG_PACKED_ATTR {
    uint32_t file_crc32c;
    uint32_t trailer_magic;     /* 0x454E444C "ENDL" */
} LZGIOTrailer;
LZG_PACKED_END

/* ── API ───────────────────────────────────────────────────── */

/**
 * Save a graph in LZG2 format.
 */
LZGError lzg_graph_save(const LZGGraph *g, const char *path);

/**
 * Load a graph from LZG2 format.
 * Recomputes topo sort and live index on load.
 */
LZGError lzg_graph_load(const char *path, LZGGraph **out);

#endif /* LZGRAPH_IO_H */
