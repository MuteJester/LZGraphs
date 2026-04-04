#include "exact_model.h"
#include "walk_engine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LZG_ACCEPTED_MODEL_MC_SAMPLES 4096u
#define LZG_ACCEPTED_MODEL_MC_SEED    0x4c5a475241504855ULL

typedef struct LZGExactModel_ {
    double root_absorption;
} LZGExactModel;

void lzg_exact_model_invalidate(LZGGraph *g) {
    if (!g || !g->exact_model_cache) return;
    free(g->exact_model_cache);
    g->exact_model_cache = NULL;
}

static LZGError estimate_root_absorption(const LZGGraph *g, double *out_absorption) {
    LZGWalkEngineConfig cfg = {
        .edge_filter = NULL,
        .edge_filter_ctx = NULL,
        .stop_on_dead_end = true,
    };
    LZGRng rng;
    uint32_t absorbed = 0;

    if (!g || !out_absorption) return LZG_ERR_INVALID_ARG;
    if (lzg_graph_ensure_query_edge_hashes((LZGGraph *)g) != LZG_OK)
        return lzg_last_error();

    lzg_rng_seed(&rng, LZG_ACCEPTED_MODEL_MC_SEED);

    for (uint32_t i = 0; i < LZG_ACCEPTED_MODEL_MC_SAMPLES; i++) {
        LZGWalkEngineResult walk;
        bool ok = lzg_walk_engine_run(g, &rng, &cfg, &walk);

        if (!ok) return LZG_ERR_INTERNAL;

        if (walk.outcome == LZG_WALK_ENGINE_OUTCOME_ABSORBED) {
            absorbed++;
        } else if (walk.outcome != LZG_WALK_ENGINE_OUTCOME_LEAKED) {
            return LZG_ERR_INTERNAL;
        }
    }

    *out_absorption = (double)absorbed / (double)LZG_ACCEPTED_MODEL_MC_SAMPLES;
    return LZG_OK;
}

LZGError lzg_exact_model_ensure(LZGGraph *g) {
    if (!g) return LZG_ERR_INVALID_ARG;
    if (g->exact_model_cache) return LZG_OK;
    if (!g->topo_valid || !g->pool || g->root_node >= g->n_nodes)
        return LZG_ERR_NOT_BUILT;

    LZGExactModel *model = calloc(1, sizeof(*model));
    if (!model) return LZG_ERR_ALLOC;

    {
        LZGError err = estimate_root_absorption(g, &model->root_absorption);
        if (err != LZG_OK) {
            free(model);
            return err;
        }
    }

    g->exact_model_cache = model;
    return LZG_OK;
}

double lzg_exact_model_root_absorption(const LZGGraph *g) {
    if (!g || !g->exact_model_cache) return 0.0;
    return g->exact_model_cache->root_absorption;
}

uint32_t lzg_exact_model_mc_samples(const LZGGraph *g) {
    (void)g;
    return LZG_ACCEPTED_MODEL_MC_SAMPLES;
}

LZGError lzg_exact_model_sample(const LZGGraph *g, LZGRng *rng,
                                LZGExactSample *out) {
    LZGWalkEngineConfig cfg = {
        .edge_filter = NULL,
        .edge_filter_ctx = NULL,
        .stop_on_dead_end = true,
    };
    double root_absorption;

    if (!g || !rng || !out) return LZG_ERR_INVALID_ARG;
    if (lzg_graph_ensure_query_edge_hashes((LZGGraph *)g) != LZG_OK)
        return lzg_last_error();

    {
        LZGError err = lzg_exact_model_ensure((LZGGraph *)g);
        if (err != LZG_OK) return err;
    }

    root_absorption = lzg_exact_model_root_absorption(g);
    if (root_absorption <= LZG_EPS)
        return LZG_ERR_NO_LIVE_PATHS;

    for (;;) {
        LZGWalkEngineResult walk;
        bool ok = lzg_walk_engine_run(g, rng, &cfg, &walk);

        if (!ok) return LZG_ERR_INTERNAL;
        if (walk.outcome == LZG_WALK_ENGINE_OUTCOME_LEAKED)
            continue;
        if (walk.outcome != LZG_WALK_ENGINE_OUTCOME_ABSORBED)
            return LZG_ERR_INTERNAL;

        memset(out, 0, sizeof(*out));
        memcpy(out->sequence, walk.sequence, (size_t)walk.seq_len + 1u);
        out->seq_len = walk.seq_len;
        out->n_tokens = walk.n_tokens;
        out->log_prob = walk.log_prob - log(root_absorption);
        return LZG_OK;
    }
}
