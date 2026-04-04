/**
 * @file genomic_simulate.c
 * @brief Gene-constrained sequence simulation with backtracking.
 *
 * Four-way edge filter: LZ76-valid (via walk dict) + V gene + J gene
 * + not blacklisted. Stack-based walk with backtracking on dead ends.
 *
 * Log-probabilities: each walk's log_prob is the product of transition
 * probabilities along the final path, where each transition is normalized
 * over the gene-filtered, LZ-valid, non-blacklisted edge set at that step.
 * This is the gene-conditional walk probability, not the unconditional one.
 * With backtracking, dead-end edges are excluded from normalization at
 * their parent, so the reported probability implicitly conditions on
 * reaching absorption through the gene-constrained subgraph.
 */
#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif
#include "lzgraph/simulate.h"
#include "lzgraph/walk_dict.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/lz76.h"
#include "walk_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Execute a single gene-constrained walk with backtracking.
 */
typedef struct {
    const LZGGeneData *gene_data;
    uint32_t v_id;
    uint32_t j_id;
} GeneWalkFilterCtx;

static bool gene_edge_filter(const LZGGraph *g, uint32_t edge, void *ctx) {
    (void)g;
    const GeneWalkFilterCtx *filter = (const GeneWalkFilterCtx *)ctx;

    if (filter->v_id != LZG_SP_NOT_FOUND &&
        !lzg_edge_has_v(filter->gene_data, edge, filter->v_id)) {
        return false;
    }
    if (filter->j_id != LZG_SP_NOT_FOUND &&
        !lzg_edge_has_j(filter->gene_data, edge, filter->j_id)) {
        return false;
    }
    return true;
}

static bool genomic_walk(
    const LZGGraph *g, const LZGGeneData *gd,
    LZGRng *rng, uint32_t v_id, uint32_t j_id,
    LZGGeneSimResult *result)
{
    GeneWalkFilterCtx filter_ctx = {
        .gene_data = gd,
        .v_id = v_id,
        .j_id = j_id,
    };
    LZGWalkEngineConfig cfg = {
        .edge_filter = gene_edge_filter,
        .edge_filter_ctx = &filter_ctx,
    };
    LZGWalkEngineResult walk;

    if (!lzg_walk_engine_run(g, rng, &cfg, &walk) ||
        walk.outcome != LZG_WALK_ENGINE_OUTCOME_ABSORBED) {
        return false;
    }

    result->base.sequence = strdup(walk.sequence);
    result->base.log_prob = walk.log_prob;
    result->base.seq_len = walk.seq_len;
    result->base.n_tokens = walk.n_tokens;
    result->v_gene_id = v_id;
    result->j_gene_id = j_id;
    return true;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Public API                                                      */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_gene_simulate(const LZGGraph *g, uint32_t n,
                             LZGRng *rng, LZGGeneSimResult *out) {
    if (!g || !rng || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph, rng, and output must not be NULL");
    if (!g->gene_data) return LZG_FAIL(LZG_ERR_NO_GENE_DATA, "graph has no V/J gene annotations");
    if (!g->topo_valid) return LZG_FAIL(LZG_ERR_NOT_BUILT, "graph not finalized");
    if (lzg_graph_ensure_query_edge_hashes((LZGGraph *)g) != LZG_OK)
        return LZG_FAIL(LZG_ERR_ALLOC, "failed to initialize query edge hash cache");

    const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;
    if (gd->n_vj_pairs == 0) return LZG_FAIL(LZG_ERR_NO_GENE_DATA, "no VJ pairs in gene data");

    double *vj_cum = malloc(gd->n_vj_pairs * sizeof(double));
    if (!vj_cum) return LZG_ERR_ALLOC;
    vj_cum[0] = gd->vj_probs[0];
    for (uint32_t i = 1; i < gd->n_vj_pairs; i++)
        vj_cum[i] = vj_cum[i - 1] + gd->vj_probs[i];
    vj_cum[gd->n_vj_pairs - 1] = 1.0;

    uint32_t generated = 0;
    uint32_t max_attempts = n * 10;
    uint32_t attempts = 0;

    while (generated < n && attempts < max_attempts) {
        double r = lzg_rng_double(rng);
        uint32_t vj = 0;
        while (vj < gd->n_vj_pairs - 1 && vj_cum[vj] < r) vj++;

        uint32_t v_id = gd->vj_v_ids[vj];
        uint32_t j_id = gd->vj_j_ids[vj];

        if (genomic_walk(g, gd, rng, v_id, j_id, &out[generated]))
            generated++;
        attempts++;
    }

    free(vj_cum);

    for (uint32_t i = generated; i < n; i++) {
        out[i].base.sequence = strdup("");
        out[i].base.log_prob = LZG_LOG_EPS;
        out[i].base.seq_len  = 0;
        out[i].base.n_tokens = 0;
        out[i].v_gene_id = LZG_SP_NOT_FOUND;
        out[i].j_gene_id = LZG_SP_NOT_FOUND;
    }

    return LZG_OK;
}

LZGError lzg_gene_simulate_vj(const LZGGraph *g, uint32_t n,
                                LZGRng *rng,
                                uint32_t v_gene_id, uint32_t j_gene_id,
                                LZGGeneSimResult *out) {
    if (!g || !rng || !out) return LZG_FAIL(LZG_ERR_NULL_ARG, "graph, rng, and output must not be NULL");
    if (!g->gene_data) return LZG_FAIL(LZG_ERR_NO_GENE_DATA, "graph has no V/J gene annotations");
    if (!g->topo_valid) return LZG_FAIL(LZG_ERR_NOT_BUILT, "graph not finalized");
    if (lzg_graph_ensure_query_edge_hashes((LZGGraph *)g) != LZG_OK)
        return LZG_FAIL(LZG_ERR_ALLOC, "failed to initialize query edge hash cache");

    const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;

    uint32_t generated = 0;
    uint32_t max_attempts = n * 10;
    uint32_t attempts = 0;

    while (generated < n && attempts < max_attempts) {
        if (genomic_walk(g, gd, rng, v_gene_id, j_gene_id, &out[generated]))
            generated++;
        attempts++;
    }

    for (uint32_t i = generated; i < n; i++) {
        out[i].base.sequence = strdup("");
        out[i].base.log_prob = LZG_LOG_EPS;
        out[i].base.seq_len  = 0;
        out[i].base.n_tokens = 0;
        out[i].v_gene_id = v_gene_id;
        out[i].j_gene_id = j_gene_id;
    }

    return LZG_OK;
}
