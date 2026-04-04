#include "diversity_internal.h"
#include "lzgraph/hash_map.h"
#include "lzgraph/string_pool.h"
#include <math.h>
#include <string.h>

static uint64_t lzg_diversity_node_identity_key(const LZGGraph *g,
                                                uint32_t node_idx) {
    uint64_t key = lzg_hash_bytes(lzg_sp_get(g->pool, g->node_sp_id[node_idx]),
                                  lzg_sp_len(g->pool, g->node_sp_id[node_idx]));

    key ^= (uint64_t)g->node_pos[node_idx] * 2654435761ULL;
    return key;
}

static void lzg_diversity_store_frequency(LZGHashMap *freq_map,
                                          LZGHashMap *all_labels,
                                          uint64_t key,
                                          double value) {
    uint64_t bits = 0;

    memcpy(&bits, &value, sizeof(bits));
    lzg_hm_put(freq_map, key, bits);
    lzg_hm_put(all_labels, key, 1);
}

LZGError lzg_jensen_shannon_divergence_impl(const LZGGraph *a,
                                            const LZGGraph *b,
                                            double *out) {
    uint32_t n_nodes_a;
    uint32_t n_nodes_b;
    double total_a = 0.0;
    double total_b = 0.0;
    LZGHashMap *freq_a;
    LZGHashMap *freq_b;
    LZGHashMap *all_labels;

    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;

    n_nodes_a = a->n_nodes;
    n_nodes_b = b->n_nodes;

    for (uint32_t i = 0; i < n_nodes_a; i++) total_a += a->outgoing_counts[i];
    for (uint32_t i = 0; i < n_nodes_b; i++) total_b += b->outgoing_counts[i];
    if (total_a < 1.0 || total_b < 1.0) {
        *out = 0.0;
        return LZG_OK;
    }

    freq_a = lzg_hm_create(n_nodes_a * 2u);
    freq_b = lzg_hm_create(n_nodes_b * 2u);
    all_labels = lzg_hm_create((n_nodes_a + n_nodes_b) * 2u);
    if (!freq_a || !freq_b || !all_labels) {
        lzg_hm_destroy(freq_a);
        lzg_hm_destroy(freq_b);
        lzg_hm_destroy(all_labels);
        return LZG_ERR_ALLOC;
    }

    for (uint32_t i = 0; i < n_nodes_a; i++) {
        double freq = a->outgoing_counts[i] / total_a;
        lzg_diversity_store_frequency(freq_a, all_labels,
                                      lzg_diversity_node_identity_key(a, i),
                                      freq);
    }

    for (uint32_t i = 0; i < n_nodes_b; i++) {
        double freq = b->outgoing_counts[i] / total_b;
        lzg_diversity_store_frequency(freq_b, all_labels,
                                      lzg_diversity_node_identity_key(b, i),
                                      freq);
    }

    {
        double jsd = 0.0;
        const double eps = 1e-300;

        for (uint32_t i = 0; i < all_labels->capacity; i++) {
            uint64_t key = all_labels->keys[i];
            double pa = 0.0;
            double pb = 0.0;
            double mix;
            uint64_t *va;
            uint64_t *vb;

            if (key == LZG_HM_EMPTY || key == LZG_HM_DELETED) continue;

            va = lzg_hm_get(freq_a, key);
            vb = lzg_hm_get(freq_b, key);
            if (va) memcpy(&pa, va, sizeof(pa));
            if (vb) memcpy(&pb, vb, sizeof(pb));

            mix = 0.5 * (pa + pb);
            if (mix < eps) continue;

            if (pa > eps) jsd += 0.5 * pa * log(pa / mix);
            if (pb > eps) jsd += 0.5 * pb * log(pb / mix);
        }

        *out = fmax(jsd, 0.0);
    }

    lzg_hm_destroy(freq_a);
    lzg_hm_destroy(freq_b);
    lzg_hm_destroy(all_labels);
    return LZG_OK;
}
