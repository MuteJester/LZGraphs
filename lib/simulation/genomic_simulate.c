/**
 * @file genomic_simulate.c
 * @brief Gene-constrained sequence simulation with backtracking.
 *
 * Four-way edge filter: LZ76-valid (via walk dict) + V gene + J gene
 * + not blacklisted. Stack-based walk with backtracking on dead ends.
 */
#ifndef _MSC_VER
#define _POSIX_C_SOURCE 200809L
#endif
#include "lzgraph/simulate.h"
#include "lzgraph/walk_dict.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/lz76.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Walk stack frame ──────────────────────────────────────────── */

#define MAX_STACK_DEPTH 128
#define MAX_BLACKLIST    64

typedef struct {
    uint32_t node;
    uint32_t edge_used;
    uint8_t  sp_len;
    uint32_t blacklist[MAX_BLACKLIST];
    uint32_t n_blacklisted;
} GeneWalkFrame;

static bool is_blacklisted(const GeneWalkFrame *f, uint32_t edge) {
    for (uint32_t i = 0; i < f->n_blacklisted; i++)
        if (f->blacklist[i] == edge) return true;
    return false;
}

/**
 * Collect edges passing: LZ validity + gene filter + not blacklisted.
 */
static uint32_t collect_valid_edges(
    const LZGGraph *g, const LZGGeneData *gd, const LZGWalkDict *wd,
    uint32_t node, uint32_t v_id, uint32_t j_id,
    const GeneWalkFrame *frame,
    uint32_t *out_edges, double *out_wts, uint32_t max_out)
{
    uint32_t e_start = g->row_offsets[node];
    uint32_t e_end   = g->row_offsets[node + 1];
    uint32_t n_valid = 0;

    for (uint32_t e = e_start; e < e_end && n_valid < max_out; e++) {
        if (frame && is_blacklisted(frame, e)) continue;
        if (!lzg_wd_edge_valid(g, e, wd)) continue;
        if (v_id != LZG_SP_NOT_FOUND && !lzg_edge_has_v(gd, e, v_id)) continue;
        if (j_id != LZG_SP_NOT_FOUND && !lzg_edge_has_j(gd, e, j_id)) continue;

        out_edges[n_valid] = e;
        out_wts[n_valid]   = g->edge_weights[e];
        n_valid++;
    }
    return n_valid;
}

/**
 * Execute a single gene-constrained walk with backtracking.
 */
static bool genomic_walk(
    const LZGGraph *g, const LZGGeneData *gd,
    LZGRng *rng, uint32_t v_id, uint32_t j_id,
    LZGGeneSimResult *result)
{
    char seq_buf[1024];
    GeneWalkFrame stack[MAX_STACK_DEPTH];
    uint32_t depth = 0;

    LZGWalkDict wd = lzg_wd_create();

    /* Start at root (@) node */
    if (g->root_node >= g->n_nodes) { lzg_wd_destroy(&wd); return false; }
    uint32_t current = g->root_node;

    uint32_t seq_pos = 0; /* @ doesn't contribute to output */

    /* Record @ token */
    lzg_wd_record_node(&wd, g, current);

    stack[0].node = current;
    stack[0].edge_used = UINT32_MAX;
    stack[0].sp_len = 0; /* @ doesn't contribute to output */
    stack[0].n_blacklisted = 0;
    depth = 1;

    uint32_t n_tokens = 1;

    while (depth > 0 && depth < MAX_STACK_DEPTH) {
        GeneWalkFrame *top = &stack[depth - 1];
        current = top->node;

        /* Check if this is a $-sink node */
        bool is_sink = g->node_is_sink && g->node_is_sink[current];
        if (is_sink) break; /* reached terminal — done */

        uint32_t valid_edges[512];
        double   valid_wts[512];
        uint32_t n_valid = collect_valid_edges(
            g, gd, &wd, current, v_id, j_id, top,
            valid_edges, valid_wts, 512);

        if (n_valid == 0) {
            /* Dead end — backtrack */
            if (depth <= 1) { lzg_wd_destroy(&wd); return false; }

            seq_pos -= top->sp_len;
            n_tokens--;
            uint32_t dead_edge = top->edge_used;
            depth--;

            /* Rebuild dictionary from stack */
            lzg_wd_destroy(&wd);
            wd = lzg_wd_create();
            lzg_wd_record_node(&wd, g, stack[0].node);
            for (uint32_t d = 1; d < depth; d++)
                lzg_wd_record_edge(&wd, g, stack[d].edge_used);

            GeneWalkFrame *parent = &stack[depth - 1];
            if (parent->n_blacklisted < MAX_BLACKLIST)
                parent->blacklist[parent->n_blacklisted++] = dead_edge;

            continue;
        }

        /* Sample from valid edges */
        double Z = 0.0;
        for (uint32_t k = 0; k < n_valid; k++) Z += valid_wts[k];

        double r = lzg_rng_double(rng) * Z;
        double cum = 0.0;
        uint32_t chosen = n_valid - 1;
        for (uint32_t k = 0; k < n_valid; k++) {
            cum += valid_wts[k];
            if (r < cum) { chosen = k; break; }
        }

        uint32_t next_node = g->col_indices[valid_edges[chosen]];
        const char *nsp = lzg_sp_get(g->pool, g->node_sp_id[next_node]);
        uint8_t nsp_len = g->node_sp_len[next_node];

        /* Strip sentinels from output */
        const char *copy_src = nsp;
        uint8_t copy_len = nsp_len;
        if (nsp_len > 0 && nsp[nsp_len - 1] == LZG_END_SENTINEL) copy_len--;
        if (nsp_len > 0 && nsp[0] == LZG_START_SENTINEL) { copy_src++; copy_len--; }

        if (seq_pos + copy_len < sizeof(seq_buf)) {
            memcpy(seq_buf + seq_pos, copy_src, copy_len);
            seq_pos += copy_len;
        }
        n_tokens++;

        lzg_wd_record_edge(&wd, g, valid_edges[chosen]);

        if (depth < MAX_STACK_DEPTH) {
            stack[depth].node = next_node;
            stack[depth].edge_used = valid_edges[chosen];
            stack[depth].sp_len = copy_len;
            stack[depth].n_blacklisted = 0;
            depth++;
        }
    }

    seq_buf[seq_pos] = '\0';
    result->base.sequence = strdup(seq_buf);
    result->base.log_prob = lzg_walk_log_prob(g, seq_buf, seq_pos);
    result->base.seq_len  = seq_pos;
    result->base.n_tokens = n_tokens;
    result->v_gene_id = v_id;
    result->j_gene_id = j_id;

    lzg_wd_destroy(&wd);
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
