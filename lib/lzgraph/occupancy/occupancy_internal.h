#ifndef LZGRAPH_OCCUPANCY_INTERNAL_H
#define LZGRAPH_OCCUPANCY_INTERNAL_H

#include "lzgraph/hash_map.h"
#include "lzgraph/occupancy.h"
#include "lzgraph/simulate.h"

enum {
    LZG_OCCUPANCY_SPLIT_N_SIM = 5000u,
    LZG_OCCUPANCY_TAYLOR_K_MAX = 50u,
    LZG_OCCUPANCY_WYNN_MIN_TERMS = 5u,
};

typedef struct {
    LZGSimResult *samples;
    LZGHashMap *seen;
    double *unique_probs;
    uint32_t n_samples;
    uint32_t n_unique;
} LZGOccupancySplit;

void lzg_occupancy_split_destroy(LZGOccupancySplit *split);

LZGError lzg_occupancy_discover_probabilities(const LZGGraph *g, double d,
                                              LZGOccupancySplit *out);
double lzg_occupancy_large_contribution(const LZGOccupancySplit *split,
                                        double d);

LZGError lzg_occupancy_fill_residual_power_sums(const LZGGraph *g,
                                                const LZGOccupancySplit *split,
                                                const double *power_sum_cache,
                                                uint32_t n_terms,
                                                double *out_residual);
LZGError lzg_occupancy_accelerated_residual(double d,
                                            const double *residual_power_sums,
                                            uint32_t n_terms,
                                            double *out);
LZGError lzg_occupancy_build_power_sum_cache(const LZGGraph *g,
                                             uint32_t n_terms,
                                             double *out_cache);

LZGError lzg_occupancy_richness_impl(const LZGGraph *g, double d,
                                     const double *power_sum_cache,
                                     uint32_t n_terms,
                                     double *out);

#endif /* LZGRAPH_OCCUPANCY_INTERNAL_H */
