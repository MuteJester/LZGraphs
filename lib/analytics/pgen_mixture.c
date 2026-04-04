#include "pgen_mixture.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    double *counts;
    double *sum_logp;
    double *sum_logp2;
    uint32_t max_length;
} LZGLengthStats;

static void lzg_length_stats_destroy(LZGLengthStats *stats) {
    if (!stats) return;
    free(stats->counts);
    free(stats->sum_logp);
    free(stats->sum_logp2);
}

static void lzg_free_simulations(LZGSimResult *sims, uint32_t n_sim) {
    if (!sims) return;
    for (uint32_t i = 0; i < n_sim; i++)
        lzg_sim_result_free(&sims[i]);
    free(sims);
}

static LZGError lzg_collect_length_stats(const LZGSimResult *sims,
                                         uint32_t n_sim,
                                         LZGLengthStats *stats) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < n_sim; i++) {
        if (sims[i].n_tokens > stats->max_length)
            stats->max_length = sims[i].n_tokens;
    }

    stats->counts = calloc(stats->max_length + 1, sizeof(double));
    stats->sum_logp = calloc(stats->max_length + 1, sizeof(double));
    stats->sum_logp2 = calloc(stats->max_length + 1, sizeof(double));
    if (!stats->counts || !stats->sum_logp || !stats->sum_logp2) {
        lzg_length_stats_destroy(stats);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t i = 0; i < n_sim; i++) {
        uint32_t length = sims[i].n_tokens;
        double logp = sims[i].log_prob;
        stats->counts[length] += 1.0;
        stats->sum_logp[length] += logp;
        stats->sum_logp2[length] += logp * logp;
    }

    return LZG_OK;
}

static void lzg_fill_mixture_components(const LZGLengthStats *stats,
                                        uint32_t n_sim,
                                        LZGPgenDist *out) {
    uint32_t nc = 0;
    for (uint32_t length = 1;
         length <= stats->max_length && nc < LZG_PGEN_MAX_COMPONENTS;
         length++) {
        if (stats->counts[length] < 5.0) continue;

        double weight = stats->counts[length] / n_sim;
        double mean = stats->sum_logp[length] / stats->counts[length];
        double variance = stats->sum_logp2[length] / stats->counts[length] - mean * mean;
        double sigma = variance > 0.0 ? sqrt(variance) : 0.01;

        out->weights[nc] = weight;
        out->means[nc] = mean;
        out->stds[nc] = sigma;
        out->walk_lengths[nc] = (int32_t)length;
        nc++;
    }
    out->n_components = nc;
}

LZGError lzg_pgen_build_analytical_mixture(const LZGGraph *g,
                                           const LZGPgenMoments *global,
                                           LZGPgenDist *out) {
    if (!g || !global || !out) return LZG_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));
    out->global = *global;

    LZGRng rng;
    lzg_rng_seed(&rng, 12345);

    {
        uint32_t n_sim = 10000;
        LZGSimResult *sims = malloc(n_sim * sizeof(LZGSimResult));
        if (!sims) return LZG_ERR_ALLOC;

        LZGError err = lzg_simulate(g, n_sim, &rng, sims);
        if (err != LZG_OK) {
            free(sims);
            return err;
        }

        LZGLengthStats stats;
        err = lzg_collect_length_stats(sims, n_sim, &stats);
        if (err != LZG_OK) {
            lzg_free_simulations(sims, n_sim);
            return err;
        }

        lzg_fill_mixture_components(&stats, n_sim, out);

        lzg_length_stats_destroy(&stats);
        lzg_free_simulations(sims, n_sim);
    }

    return LZG_OK;
}
