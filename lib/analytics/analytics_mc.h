#ifndef LZGRAPH_ANALYTICS_MC_H
#define LZGRAPH_ANALYTICS_MC_H

#include "lzgraph/analytics.h"

typedef struct {
    double   *log_probs;
    uint32_t  n;
} LZGAnalyticsMCResult;

LZGError lzg_analytics_mc_run(const LZGGraph *g, uint32_t n_samples,
                              uint64_t seed, LZGAnalyticsMCResult *out);

void lzg_analytics_mc_free(LZGAnalyticsMCResult *mc);

bool lzg_analytics_mc_is_valid_log_prob(double log_prob);

uint32_t lzg_analytics_mc_valid_count(const LZGAnalyticsMCResult *mc);

double lzg_analytics_mc_absorbed_mass(const LZGAnalyticsMCResult *mc);

double lzg_analytics_mc_support_estimate(const LZGAnalyticsMCResult *mc);

void lzg_analytics_mc_entropy_stats(const LZGAnalyticsMCResult *mc,
                                    double *sum_lp, uint32_t *valid);

double lzg_analytics_mc_power_mean(const LZGAnalyticsMCResult *mc, double alpha);

double lzg_analytics_mc_hill_estimate(const LZGAnalyticsMCResult *mc, double alpha);

#endif /* LZGRAPH_ANALYTICS_MC_H */
