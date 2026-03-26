/**
 * @file rng.h
 * @brief xoshiro256++ fast PRNG.
 */
#ifndef LZGRAPH_RNG_H
#define LZGRAPH_RNG_H

#include "lzgraph/common.h"

typedef struct {
    uint64_t s[4];
} LZGRng;

/** Seed from a single 64-bit value (splitmix64 expansion). */
void lzg_rng_seed(LZGRng *rng, uint64_t seed);

/** Next random uint64. */
LZG_INLINE uint64_t lzg_rng_next(LZGRng *rng) {
    const uint64_t s0 = rng->s[0], s3 = rng->s[3];
    const uint64_t result = ((s0 + s3) << 23 | (s0 + s3) >> 41) + s0;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= s0;
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = (rng->s[3] << 45) | (rng->s[3] >> 19);
    return result;
}

/** Uniform double in [0, 1). */
LZG_INLINE double lzg_rng_double(LZGRng *rng) {
    return (lzg_rng_next(rng) >> 11) * 0x1.0p-53;
}

/** Uniform uint32 in [0, n). */
LZG_INLINE uint32_t lzg_rng_bounded(LZGRng *rng, uint32_t n) {
    return (uint32_t)((lzg_rng_next(rng) >> 33) * (uint64_t)n >> 31);
}

#endif /* LZGRAPH_RNG_H */
