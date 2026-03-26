/**
 * @file crc32c.h
 * @brief CRC-32C (Castagnoli) checksum.
 *
 * Software table-driven implementation. On x86 with SSE4.2, a hardware-
 * accelerated version can be substituted at link time.
 */
#ifndef LZGRAPH_CRC32C_H
#define LZGRAPH_CRC32C_H

#include "lzgraph/common.h"

/** Compute CRC-32C over a byte buffer. */
uint32_t lzg_crc32c(const void *data, size_t len);

/** Update a running CRC-32C with additional data. */
uint32_t lzg_crc32c_update(uint32_t crc, const void *data, size_t len);

#endif /* LZGRAPH_CRC32C_H */
