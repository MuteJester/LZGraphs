/**
 * @file graph_ops.c
 * @brief Graph operations: hill_curve, graph_summary, set operations.
 */
#include "lzgraph/graph_ops.h"
#include "lzgraph/analytics.h"
#include "lzgraph/edge_builder.h"
#include "lzgraph/hash_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════ */
/* Hill curve                                                      */
/* ═══════════════════════════════════════════════════════════════ */

static const double DEFAULT_ORDERS[] = {
    0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10
};
#define DEFAULT_N_ORDERS 12

LZGError lzg_hill_curve(const LZGGraph *g, const double *orders,
                         uint32_t n, LZGHillCurve *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (!orders || n == 0) { orders = DEFAULT_ORDERS; n = DEFAULT_N_ORDERS; }
    out->n = n;
    out->orders = malloc(n * sizeof(double));
    out->hill_numbers = malloc(n * sizeof(double));
    memcpy(out->orders, orders, n * sizeof(double));
    LZGError err = lzg_hill_numbers(g, orders, n, out->hill_numbers);
    if (err != LZG_OK) { free(out->orders); free(out->hill_numbers); }
    return err;
}

void lzg_hill_curve_free(LZGHillCurve *hc) {
    if (!hc) return;
    free(hc->orders); free(hc->hill_numbers);
    hc->orders = NULL; hc->hill_numbers = NULL;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Graph summary                                                   */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_summary(const LZGGraph *g, LZGGraphSummary *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    out->n_nodes = g->n_nodes; out->n_edges = g->n_edges;
    out->n_initial = 1; /* single root @ */
    /* Count sink nodes */
    out->n_terminal = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++)
        if (g->node_is_sink && g->node_is_sink[i]) out->n_terminal++;
    out->is_dag = g->topo_valid;
    out->max_out_degree = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        uint32_t d = g->row_offsets[i + 1] - g->row_offsets[i];
        if (d > out->max_out_degree) out->max_out_degree = d;
    }
    uint32_t *in_deg = calloc(g->n_nodes, sizeof(uint32_t));
    for (uint32_t e = 0; e < g->n_edges; e++) in_deg[g->col_indices[e]]++;
    out->max_in_degree = 0; out->n_isolates = 0;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        if (in_deg[i] > out->max_in_degree) out->max_in_degree = in_deg[i];
        if (in_deg[i] == 0 && (g->row_offsets[i+1] - g->row_offsets[i]) == 0)
            out->n_isolates++;
    }
    free(in_deg);
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shared helper: build node label string from graph node         */
/* ═══════════════════════════════════════════════════════════════ */

static uint32_t intern_node_label(const LZGGraph *src, uint32_t node_idx,
                                   LZGStringPool *dst_pool) {
    const char *sp = lzg_sp_get(src->pool, src->node_sp_id[node_idx]);
    uint32_t pos = src->node_pos[node_idx];
    char buf[256];
    int len;
    if (src->variant == LZG_VARIANT_NAIVE)
        len = snprintf(buf, sizeof(buf), "%s", sp);
    else
        len = snprintf(buf, sizeof(buf), "%s_%u", sp, pos);
    return lzg_sp_intern_n(dst_pool, buf, (uint32_t)len);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shared: collect edges from one graph into unified hash maps     */
/* ═══════════════════════════════════════════════════════════════ */

/*
 * For each edge in src, compute unified node label IDs in dst_pool
 * and store edge_key → count in the edge_counts map.
 * Also collect initial/terminal counts and outgoing counts.
 */
static void collect_graph_edges(const LZGGraph *src,
                                 LZGStringPool *dst_pool,
                                 LZGHashMap *edge_counts,
                                 LZGHashMap *node_set,
                                 LZGHashMap *initial_cts,
                                 LZGHashMap *terminal_cts,
                                 LZGHashMap *outgoing_cts) {
    for (uint32_t u = 0; u < src->n_nodes; u++) {
        uint32_t nid = intern_node_label(src, u, dst_pool);
        lzg_hm_put(node_set, nid, 1);

        uint32_t e_start = src->row_offsets[u];
        uint32_t e_end   = src->row_offsets[u + 1];
        for (uint32_t e = e_start; e < e_end; e++) {
            uint32_t v = src->col_indices[e];
            uint32_t vid = intern_node_label(src, v, dst_pool);
            lzg_hm_put(node_set, vid, 1);

            uint64_t ekey = ((uint64_t)nid << 32) | vid;
            uint64_t *existing = lzg_hm_get(edge_counts, ekey);
            uint64_t count = src->edge_counts[e];
            if (existing) *existing += count;
            else lzg_hm_put(edge_counts, ekey, count);

            uint64_t *oc = lzg_hm_get(outgoing_cts, nid);
            if (oc) *oc += count; else lzg_hm_put(outgoing_cts, nid, count);
        }
    }

    /* Initial/terminal counts removed — sentinel model derives them from @ and $ nodes */
}

/* ═══════════════════════════════════════════════════════════════ */
/* Shared: build graph from edge count map                         */
/* ═══════════════════════════════════════════════════════════════ */

static LZGError build_from_edge_map(
    LZGVariant variant, double smoothing, uint32_t min_init,
    LZGStringPool *pool,
    LZGHashMap *edge_counts,
    LZGHashMap *node_set,
    LZGHashMap *initial_cts,
    LZGHashMap *terminal_cts,
    LZGHashMap *outgoing_cts,
    uint32_t *len_counts, uint32_t max_len,
    LZGGraph **out)
{
    LZGGraph *g = calloc(1, sizeof(LZGGraph));
    g->variant = variant;
    g->smoothing_alpha = smoothing;
    (void)min_init; /* deprecated */
    g->pool = pool;

    /* Build EdgeBuilder from the edge_counts map */
    LZGEdgeBuilder *eb = lzg_eb_create(edge_counts->count * 2);

    for (uint32_t i = 0; i < edge_counts->capacity; i++) {
        if (edge_counts->keys[i] == LZG_HM_EMPTY ||
            edge_counts->keys[i] == LZG_HM_DELETED) continue;

        uint32_t src = (uint32_t)(edge_counts->keys[i] >> 32);
        uint32_t dst = (uint32_t)(edge_counts->keys[i] & 0xFFFFFFFF);
        uint32_t count = (uint32_t)edge_counts->values[i];

        if (count == 0) continue; /* dropped edge */
        lzg_eb_record(eb, src, dst, count);
    }

    /* Rebuild outgoing_cts from the filtered edges to be consistent */
    lzg_hm_clear(outgoing_cts);
    for (uint32_t i = 0; i < eb->n_edges; i++) {
        uint64_t *oc = lzg_hm_get(outgoing_cts, eb->src_ids[i]);
        if (oc) *oc += eb->counts[i];
        else lzg_hm_put(outgoing_cts, eb->src_ids[i], eb->counts[i]);
    }

    LZGError err = lzg_graph_finalize_from_edges(
        g, eb, node_set, initial_cts, terminal_cts,
        outgoing_cts, len_counts, max_len);

    if (err != LZG_OK) { lzg_graph_destroy(g); *out = NULL; return err; }
    *out = g;
    return LZG_OK;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Merge length distributions                                      */
/* ═══════════════════════════════════════════════════════════════ */

static uint32_t *merge_lengths(const LZGGraph *a, const LZGGraph *b,
                                uint32_t *out_max) {
    uint32_t ml = (a->max_length > b->max_length) ? a->max_length : b->max_length;
    uint32_t *lc = calloc(ml + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i <= a->max_length; i++) lc[i] += a->length_counts[i];
    for (uint32_t i = 0; i <= b->max_length; i++) lc[i] += b->length_counts[i];
    *out_max = ml;
    return lc;
}

/* ═══════════════════════════════════════════════════════════════ */
/* Union: A + B                                                    */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_union(const LZGGraph *a, const LZGGraph *b,
                          LZGGraph **out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;
    if (a->variant != b->variant) return LZG_FAIL(LZG_ERR_VARIANT_MISMATCH, "cannot combine graphs with different variants (a=%d, b=%d)", a->variant, b->variant);

    LZGStringPool *pool = lzg_sp_create(4096);
    LZGHashMap *ec = lzg_hm_create((a->n_edges + b->n_edges) * 2);
    LZGHashMap *ns = lzg_hm_create(a->n_nodes + b->n_nodes);
    LZGHashMap *ic = lzg_hm_create(256);
    LZGHashMap *tc = lzg_hm_create(256);
    LZGHashMap *oc = lzg_hm_create(4096);

    /* Collect from both — counts are summed for shared edges */
    collect_graph_edges(a, pool, ec, ns, ic, tc, oc);
    collect_graph_edges(b, pool, ec, ns, ic, tc, oc);

    uint32_t ml;
    uint32_t *lc = merge_lengths(a, b, &ml);

    return build_from_edge_map(a->variant, a->smoothing_alpha,
                                0, pool,
                                ec, ns, ic, tc, oc, lc, ml, out);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Intersection: min(A, B) — only shared edges                     */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_intersection(const LZGGraph *a, const LZGGraph *b,
                                 LZGGraph **out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;
    if (a->variant != b->variant) return LZG_FAIL(LZG_ERR_VARIANT_MISMATCH, "cannot combine graphs with different variants (a=%d, b=%d)", a->variant, b->variant);

    LZGStringPool *pool = lzg_sp_create(4096);

    /* Collect edges from A */
    LZGHashMap *ec_a = lzg_hm_create(a->n_edges * 2);
    LZGHashMap *ns_a = lzg_hm_create(a->n_nodes);
    LZGHashMap *dummy_ic = lzg_hm_create(16);
    LZGHashMap *dummy_tc = lzg_hm_create(16);
    LZGHashMap *dummy_oc = lzg_hm_create(16);
    collect_graph_edges(a, pool, ec_a, ns_a, dummy_ic, dummy_tc, dummy_oc);
    lzg_hm_destroy(dummy_ic); lzg_hm_destroy(dummy_tc); lzg_hm_destroy(dummy_oc);

    /* Collect edges from B */
    LZGHashMap *ec_b = lzg_hm_create(b->n_edges * 2);
    LZGHashMap *ns_b = lzg_hm_create(b->n_nodes);
    dummy_ic = lzg_hm_create(16); dummy_tc = lzg_hm_create(16); dummy_oc = lzg_hm_create(16);
    collect_graph_edges(b, pool, ec_b, ns_b, dummy_ic, dummy_tc, dummy_oc);
    lzg_hm_destroy(dummy_ic); lzg_hm_destroy(dummy_tc); lzg_hm_destroy(dummy_oc);

    /* Build intersection: only edges in BOTH, count = min */
    LZGHashMap *ec = lzg_hm_create(a->n_edges * 2);
    LZGHashMap *ns = lzg_hm_create(a->n_nodes + b->n_nodes);
    LZGHashMap *ic = lzg_hm_create(256);
    LZGHashMap *tc = lzg_hm_create(256);
    LZGHashMap *oc = lzg_hm_create(4096);

    for (uint32_t i = 0; i < ec_a->capacity; i++) {
        if (ec_a->keys[i] == LZG_HM_EMPTY || ec_a->keys[i] == LZG_HM_DELETED)
            continue;
        uint64_t *b_val = lzg_hm_get(ec_b, ec_a->keys[i]);
        if (!b_val) continue; /* not in B — skip */

        uint32_t count_a = (uint32_t)ec_a->values[i];
        uint32_t count_b = (uint32_t)*b_val;
        uint32_t count = (count_a < count_b) ? count_a : count_b;

        uint32_t src = (uint32_t)(ec_a->keys[i] >> 32);
        uint32_t dst = (uint32_t)(ec_a->keys[i] & 0xFFFFFFFF);
        lzg_hm_put(ec, ec_a->keys[i], count);
        lzg_hm_put(ns, src, 1);
        lzg_hm_put(ns, dst, 1);

        uint64_t *o = lzg_hm_get(oc, src);
        if (o) *o += count; else lzg_hm_put(oc, src, count);
    }

    lzg_hm_destroy(ec_a); lzg_hm_destroy(ec_b);
    lzg_hm_destroy(ns_a); lzg_hm_destroy(ns_b);

    /* Use A's initial/terminal for shared nodes */
    collect_graph_edges(a, pool, lzg_hm_create(1)/*dummy*/, lzg_hm_create(1),
                        ic, tc, lzg_hm_create(1));
    /* That's messy — let me just re-collect initials/terminals properly */
    /* Initial/terminal derived from @ and $ nodes in the sentinel model */
    lzg_hm_clear(ic); lzg_hm_clear(tc);

    uint32_t ml;
    uint32_t *lc = merge_lengths(a, b, &ml); /* approximate */

    return build_from_edge_map(a->variant, a->smoothing_alpha,
                                0, pool,
                                ec, ns, ic, tc, oc, lc, ml, out);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Difference: max(A - B, 0) — remove B's contribution from A     */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_difference(const LZGGraph *a, const LZGGraph *b,
                               LZGGraph **out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;
    if (a->variant != b->variant) return LZG_FAIL(LZG_ERR_VARIANT_MISMATCH, "cannot combine graphs with different variants (a=%d, b=%d)", a->variant, b->variant);

    LZGStringPool *pool = lzg_sp_create(4096);

    /* Collect edges from both */
    LZGHashMap *ec_a = lzg_hm_create(a->n_edges * 2);
    LZGHashMap *ns_a = lzg_hm_create(a->n_nodes);
    LZGHashMap *ic = lzg_hm_create(256);
    LZGHashMap *tc = lzg_hm_create(256);
    LZGHashMap *dummy = lzg_hm_create(16);
    collect_graph_edges(a, pool, ec_a, ns_a, ic, tc, dummy);
    lzg_hm_destroy(dummy);

    LZGHashMap *ec_b = lzg_hm_create(b->n_edges * 2);
    LZGHashMap *ns_b = lzg_hm_create(b->n_nodes);
    dummy = lzg_hm_create(16);
    LZGHashMap *d2 = lzg_hm_create(16); LZGHashMap *d3 = lzg_hm_create(16);
    collect_graph_edges(b, pool, ec_b, ns_b, dummy, d2, d3);
    lzg_hm_destroy(dummy); lzg_hm_destroy(d2); lzg_hm_destroy(d3);
    lzg_hm_destroy(ns_b);

    /* Subtract B's counts from A's */
    LZGHashMap *ec = lzg_hm_create(a->n_edges * 2);
    LZGHashMap *ns = lzg_hm_create(a->n_nodes);
    LZGHashMap *oc = lzg_hm_create(4096);

    for (uint32_t i = 0; i < ec_a->capacity; i++) {
        if (ec_a->keys[i] == LZG_HM_EMPTY || ec_a->keys[i] == LZG_HM_DELETED)
            continue;

        uint32_t count_a = (uint32_t)ec_a->values[i];
        uint64_t *b_val = lzg_hm_get(ec_b, ec_a->keys[i]);
        uint32_t count_b = b_val ? (uint32_t)*b_val : 0;

        int32_t diff = (int32_t)count_a - (int32_t)count_b;
        if (diff <= 0) continue; /* edge fully subtracted */

        uint32_t src = (uint32_t)(ec_a->keys[i] >> 32);
        uint32_t dst = (uint32_t)(ec_a->keys[i] & 0xFFFFFFFF);
        lzg_hm_put(ec, ec_a->keys[i], (uint32_t)diff);
        lzg_hm_put(ns, src, 1);
        lzg_hm_put(ns, dst, 1);

        uint64_t *o = lzg_hm_get(oc, src);
        if (o) *o += (uint32_t)diff; else lzg_hm_put(oc, src, (uint32_t)diff);
    }

    lzg_hm_destroy(ec_a); lzg_hm_destroy(ec_b); lzg_hm_destroy(ns_a);

    /* Keep only initials/terminals that survived */
    LZGHashMap *ic_f = lzg_hm_create(256);
    LZGHashMap *tc_f = lzg_hm_create(256);
    for (uint32_t i = 0; i < ic->capacity; i++) {
        if (ic->keys[i] != LZG_HM_EMPTY && ic->keys[i] != LZG_HM_DELETED) {
            if (lzg_hm_get(ns, ic->keys[i]))
                lzg_hm_put(ic_f, ic->keys[i], ic->values[i]);
        }
    }
    for (uint32_t i = 0; i < tc->capacity; i++) {
        if (tc->keys[i] != LZG_HM_EMPTY && tc->keys[i] != LZG_HM_DELETED) {
            if (lzg_hm_get(ns, tc->keys[i]))
                lzg_hm_put(tc_f, tc->keys[i], tc->values[i]);
        }
    }
    lzg_hm_destroy(ic); lzg_hm_destroy(tc);

    uint32_t ml = a->max_length;
    uint32_t *lc = calloc(ml + 1, sizeof(uint32_t));
    memcpy(lc, a->length_counts, (ml + 1) * sizeof(uint32_t));

    return build_from_edge_map(a->variant, a->smoothing_alpha,
                                0, pool,
                                ec, ns, ic_f, tc_f, oc, lc, ml, out);
}

/* ═══════════════════════════════════════════════════════════════ */
/* Weighted merge: alpha*A + beta*B                                */
/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_graph_weighted_merge(const LZGGraph *a, const LZGGraph *b,
                                   double alpha, double beta,
                                   LZGGraph **out) {
    if (!a || !b || !out) return LZG_ERR_INVALID_ARG;
    if (a->variant != b->variant) return LZG_FAIL(LZG_ERR_VARIANT_MISMATCH, "cannot combine graphs with different variants (a=%d, b=%d)", a->variant, b->variant);

    LZGStringPool *pool = lzg_sp_create(4096);

    /* Collect edges from both separately */
    LZGHashMap *ec_a = lzg_hm_create(a->n_edges * 2);
    LZGHashMap *ns = lzg_hm_create(a->n_nodes + b->n_nodes);
    LZGHashMap *ic = lzg_hm_create(256);
    LZGHashMap *tc = lzg_hm_create(256);
    LZGHashMap *dummy = lzg_hm_create(16);
    collect_graph_edges(a, pool, ec_a, ns, ic, tc, dummy);
    lzg_hm_destroy(dummy);

    LZGHashMap *ec_b = lzg_hm_create(b->n_edges * 2);
    dummy = lzg_hm_create(16);
    LZGHashMap *ic_b = lzg_hm_create(256);
    LZGHashMap *tc_b = lzg_hm_create(256);
    collect_graph_edges(b, pool, ec_b, ns, ic_b, tc_b, dummy);
    lzg_hm_destroy(dummy);

    /* Combine: alpha*A + beta*B */
    LZGHashMap *ec = lzg_hm_create((a->n_edges + b->n_edges) * 2);
    LZGHashMap *oc = lzg_hm_create(4096);

    /* Add alpha*A */
    for (uint32_t i = 0; i < ec_a->capacity; i++) {
        if (ec_a->keys[i] == LZG_HM_EMPTY || ec_a->keys[i] == LZG_HM_DELETED)
            continue;
        uint32_t c = (uint32_t)round(alpha * ec_a->values[i]);
        if (c == 0) continue;
        lzg_hm_put(ec, ec_a->keys[i], c);
    }

    /* Add beta*B */
    for (uint32_t i = 0; i < ec_b->capacity; i++) {
        if (ec_b->keys[i] == LZG_HM_EMPTY || ec_b->keys[i] == LZG_HM_DELETED)
            continue;
        uint32_t c = (uint32_t)round(beta * ec_b->values[i]);
        if (c == 0) continue;
        uint64_t *existing = lzg_hm_get(ec, ec_b->keys[i]);
        if (existing) *existing += c;
        else lzg_hm_put(ec, ec_b->keys[i], c);
    }

    /* Rebuild outgoing from merged edges */
    for (uint32_t i = 0; i < ec->capacity; i++) {
        if (ec->keys[i] == LZG_HM_EMPTY || ec->keys[i] == LZG_HM_DELETED)
            continue;
        uint32_t src = (uint32_t)(ec->keys[i] >> 32);
        uint64_t *o = lzg_hm_get(oc, src);
        if (o) *o += ec->values[i]; else lzg_hm_put(oc, src, ec->values[i]);
    }

    lzg_hm_destroy(ec_a); lzg_hm_destroy(ec_b);

    /* Merge initial/terminal: alpha*A + beta*B */
    LZGHashMap *ic_m = lzg_hm_create(256);
    for (uint32_t i = 0; i < ic->capacity; i++) {
        if (ic->keys[i] != LZG_HM_EMPTY && ic->keys[i] != LZG_HM_DELETED) {
            uint64_t c = (uint64_t)round(alpha * ic->values[i]);
            if (c > 0) lzg_hm_put(ic_m, ic->keys[i], c);
        }
    }
    for (uint32_t i = 0; i < ic_b->capacity; i++) {
        if (ic_b->keys[i] != LZG_HM_EMPTY && ic_b->keys[i] != LZG_HM_DELETED) {
            uint64_t c = (uint64_t)round(beta * ic_b->values[i]);
            uint64_t *ex = lzg_hm_get(ic_m, ic_b->keys[i]);
            if (ex) *ex += c; else if (c > 0) lzg_hm_put(ic_m, ic_b->keys[i], c);
        }
    }
    lzg_hm_destroy(ic); lzg_hm_destroy(ic_b);

    LZGHashMap *tc_m = lzg_hm_create(256);
    for (uint32_t i = 0; i < tc->capacity; i++) {
        if (tc->keys[i] != LZG_HM_EMPTY && tc->keys[i] != LZG_HM_DELETED) {
            uint64_t c = (uint64_t)round(alpha * tc->values[i]);
            if (c > 0) lzg_hm_put(tc_m, tc->keys[i], c);
        }
    }
    for (uint32_t i = 0; i < tc_b->capacity; i++) {
        if (tc_b->keys[i] != LZG_HM_EMPTY && tc_b->keys[i] != LZG_HM_DELETED) {
            uint64_t c = (uint64_t)round(beta * tc_b->values[i]);
            uint64_t *ex = lzg_hm_get(tc_m, tc_b->keys[i]);
            if (ex) *ex += c; else if (c > 0) lzg_hm_put(tc_m, tc_b->keys[i], c);
        }
    }
    lzg_hm_destroy(tc); lzg_hm_destroy(tc_b);

    uint32_t ml;
    uint32_t *lc = merge_lengths(a, b, &ml);

    return build_from_edge_map(a->variant, a->smoothing_alpha,
                                0, pool,
                                ec, ns, ic_m, tc_m, oc, lc, ml, out);
}
