/**
 * @file crc32c.c
 * @brief CRC-32C (Castagnoli) — polynomial 0x1EDC6F41.
 * Software table-driven, ~1 GB/s on modern CPUs.
 */
#include "lzgraph/crc32c.h"

/* Lookup table generated for polynomial 0x82F63B78 (bit-reversed 0x1EDC6F41) */
static uint32_t crc32c_table[256];
static int table_ready = 0;

static void build_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c >> 1) ^ ((c & 1) ? 0x82F63B78u : 0);
        crc32c_table[i] = c;
    }
    table_ready = 1;
}

uint32_t lzg_crc32c_update(uint32_t crc, const void *data, size_t len) {
    if (!table_ready) build_table();
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++)
        crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

uint32_t lzg_crc32c(const void *data, size_t len) {
    return lzg_crc32c_update(0, data, len);
}
