/**
 * @file gene_data.c
 * @brief Gene data construction and query.
 */
#include "lzgraph/gene_data.h"
#include "lzgraph/hash_map.h"
#include <stdlib.h>
#include <string.h>

LZGGeneData *lzg_gene_data_create(void) {
    LZGGeneData *gd = calloc(1, sizeof(LZGGeneData));
    if (!gd) return NULL;
    gd->gene_pool = lzg_sp_create(256);
    return gd;
}

void lzg_gene_data_destroy(LZGGeneData *gd) {
    if (!gd) return;
    lzg_sp_destroy(gd->gene_pool);
    free(gd->v_offsets);     free(gd->v_gene_ids);    free(gd->v_gene_counts);
    free(gd->j_offsets);     free(gd->j_gene_ids);    free(gd->j_gene_counts);
    free(gd->v_marginal_ids); free(gd->v_marginal_probs);
    free(gd->j_marginal_ids); free(gd->j_marginal_probs);
    free(gd->vj_v_ids);     free(gd->vj_j_ids);      free(gd->vj_probs);
    free(gd);
}

/* Binary search in sorted uint32 array */
static int32_t bsearch_u32(const uint32_t *arr, uint32_t lo, uint32_t hi,
                            uint32_t target) {
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] == target) return (int32_t)mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

double lzg_edge_v_prob(const LZGGeneData *gd, uint32_t edge_idx,
                        uint32_t v_gene_id) {
    uint32_t lo = gd->v_offsets[edge_idx];
    uint32_t hi = gd->v_offsets[edge_idx + 1];
    if (lo == hi) return 0.0;

    int32_t idx = bsearch_u32(gd->v_gene_ids, lo, hi, v_gene_id);
    if (idx < 0) return 0.0;

    /* Sum counts for this edge's V genes */
    uint64_t total = 0;
    for (uint32_t i = lo; i < hi; i++) total += gd->v_gene_counts[i];
    return total > 0 ? (double)gd->v_gene_counts[idx] / total : 0.0;
}

double lzg_edge_j_prob(const LZGGeneData *gd, uint32_t edge_idx,
                        uint32_t j_gene_id) {
    uint32_t lo = gd->j_offsets[edge_idx];
    uint32_t hi = gd->j_offsets[edge_idx + 1];
    if (lo == hi) return 0.0;

    int32_t idx = bsearch_u32(gd->j_gene_ids, lo, hi, j_gene_id);
    if (idx < 0) return 0.0;

    uint64_t total = 0;
    for (uint32_t i = lo; i < hi; i++) total += gd->j_gene_counts[i];
    return total > 0 ? (double)gd->j_gene_counts[idx] / total : 0.0;
}

bool lzg_edge_has_v(const LZGGeneData *gd, uint32_t edge_idx,
                     uint32_t v_gene_id) {
    uint32_t lo = gd->v_offsets[edge_idx];
    uint32_t hi = gd->v_offsets[edge_idx + 1];
    return bsearch_u32(gd->v_gene_ids, lo, hi, v_gene_id) >= 0;
}

bool lzg_edge_has_j(const LZGGeneData *gd, uint32_t edge_idx,
                     uint32_t j_gene_id) {
    uint32_t lo = gd->j_offsets[edge_idx];
    uint32_t hi = gd->j_offsets[edge_idx + 1];
    return bsearch_u32(gd->j_gene_ids, lo, hi, j_gene_id) >= 0;
}
