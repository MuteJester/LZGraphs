/**
 * @file test_analytics.c
 * @brief Tests for Phase 5: analytics (Hill numbers, diversity, diagnostics).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/analytics.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/simulate.h"
#include "lzgraph/rng.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static LZGGraph *build_graph(void) {
    const char *seqs[] = {
        "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
        "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
        "CASRGGTVYEQYF", "CSVSTSETGDTEQYF", "CASSPPDGILGYTF",
        "CASSLDSRAGANYF", "CASSYTGQENVLHF", "CASSQRRDRSPQYF",
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    lzg_graph_build(g, seqs, 12, NULL, NULL, NULL, 0.0, 0);
    return g;
}

static LZGGraph *build_coinflip_graph(uint32_t a_count, uint32_t b_count) {
    uint32_t n = a_count + b_count;
    const char **seqs = malloc(n * sizeof(*seqs));
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_AAP);
    if (!seqs || !g) {
        free(seqs);
        if (g) lzg_graph_destroy(g);
        return NULL;
    }

    for (uint32_t i = 0; i < a_count; i++) seqs[i] = "A";
    for (uint32_t i = 0; i < b_count; i++) seqs[a_count + i] = "C";

    lzg_graph_build(g, seqs, n, NULL, NULL, NULL, 0.0, 0);
    free(seqs);
    return g;
}

static LZGGraph *build_flashback_graph(void) {
    const char *seqs[] = {
        "CASS", "CASST", "CAT", "CATS", "CASSLG", "CASSLG"
    };
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE);
    if (!g) return NULL;
    if (lzg_flashback_graph_build(g, seqs, 6, NULL, 0.0) != LZG_OK) {
        lzg_graph_destroy(g);
        return NULL;
    }
    return g;
}

static double direct_mc_hill(const LZGSimResult *sim, uint32_t n, double alpha) {
    if (fabs(alpha - 1.0) < 1e-12) {
        double sum_lp = 0.0;
        uint32_t valid = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (sim[i].log_prob <= LZG_LOG_EPS + 1.0) continue;
            sum_lp += sim[i].log_prob;
            valid++;
        }
        return valid > 0 ? exp(-sum_lp / valid) : 0.0;
    }

    double sum = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (sim[i].log_prob <= LZG_LOG_EPS + 1.0) continue;
        sum += exp((alpha - 1.0) * sim[i].log_prob);
        valid++;
    }

    if (valid == 0) return 0.0;
    double moment = sum / valid;
    return moment > 0.0 ? pow(moment, 1.0 / (1.0 - alpha)) : 0.0;
}

/* ═══════════════════════════════════════════════════════════════ */

static void test_simulation_potential_size(void) {
    LZGGraph *g = build_graph();
    double count;
    LZGError err = lzg_graph_path_count(g, &count);
    ASSERT_MSG(err == LZG_OK, "ok");
    printf("\n    LZ-valid paths: %.0f", count);
    ASSERT_MSG(count >= 12, "at least as many paths as training seqs");
    ASSERT_MSG(count < 1e10, "finite and reasonable");
    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_diagnostics(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    LZGError err = lzg_pgen_diagnostics(g, 1e-2, &diag);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    absorbed=%.6f leaked=%.6f init_sum=%.6f",
           diag.total_absorbed, diag.total_leaked, diag.initial_prob_sum);

    ASSERT_MSG(fabs(diag.initial_prob_sum - 1.0) < 1e-10, "init sums to 1");
    ASSERT_MSG(fabs(diag.total_absorbed - (47.0 / 48.0)) < 0.02,
               "MC absorbed mass tracks known leaky graph");
    ASSERT_MSG(fabs(diag.total_leaked - (1.0 / 48.0)) < 0.02,
               "MC leaked mass tracks known leaky graph");

    lzg_graph_destroy(g);
    PASS();
}

static void test_leaky_graph_has_exact_raw_diagnostics_but_proper_public_m1(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    double m1;
    LZGError err = lzg_pgen_diagnostics(g, 1e-2, &diag);
    ASSERT_MSG(err == LZG_OK, "diag ok");
    err = lzg_power_sum(g, 1.0, &m1);
    ASSERT_MSG(err == LZG_OK, "power sum ok");

    printf("\n    absorbed=%.6f leaked=%.6f M(1)=%.6f",
           diag.total_absorbed, diag.total_leaked, m1);

    ASSERT_MSG(fabs(diag.total_absorbed - (47.0 / 48.0)) < 0.02,
               "raw diagnostics stay close to known leakage");
    ASSERT_MSG(fabs(m1 - 1.0) < 1e-12,
               "public power_sum(1) is proper under the accepted model");

    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_diagnostics_atol_contract(void) {
    LZGGraph *g = build_graph();
    LZGPgenDiagnostics diag;
    LZGError err = lzg_pgen_diagnostics(g, 0.0, &diag);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    absorbed=%.6f atol=0 => is_proper=%d",
           diag.total_absorbed, (int)diag.is_proper);
    ASSERT_MSG(diag.is_proper == false, "strict atol contract honored");

    lzg_graph_destroy(g);
    PASS();
}

static void test_effective_diversity(void) {
    LZGGraph *g = build_graph();
    LZGEffectiveDiversity div;
    LZGError err = lzg_effective_diversity(g, &div);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    H=%.4f nats, N_eff=%.2f, uniformity=%.4e",
           div.entropy_nats, div.effective_diversity, div.uniformity);

    ASSERT_MSG(div.entropy_nats > 0, "positive entropy");
    ASSERT_MSG(div.effective_diversity > 1, "N_eff > 1");
    ASSERT_MSG(div.uniformity > 0 && div.uniformity <= 1.0, "uniformity in (0,1]");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d0(void) {
    /* D(0) = path count */
    LZGGraph *g = build_graph();
    double d0;
    lzg_hill_number(g, 0.0, &d0);

    double paths;
    lzg_graph_path_count(g, &paths);

    printf("\n    D(0)=%.0f, paths=%.0f", d0, paths);
    ASSERT_MSG(d0 > 0 && paths > 0, "positive support estimates");
    ASSERT_MSG(fabs(d0 - paths) / fmax(paths, 1.0) < 0.25,
               "D(0) and path count agree within MC error");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d1(void) {
    /* D(1) = exp(H) = effective diversity */
    LZGGraph *g = build_graph();
    double d1;
    lzg_hill_number(g, 1.0, &d1);

    LZGEffectiveDiversity div;
    lzg_effective_diversity(g, &div);

    printf("\n    D(1)=%.4f, exp(H)=%.4f", d1, div.effective_diversity);
    ASSERT_MSG(fabs(d1 - div.effective_diversity) / fmax(d1, 1) < 0.15, "D(1) == exp(H)");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d2(void) {
    /* D(2) = 1 / M(2) on the accepted-sequence model. */
    LZGGraph *g = build_graph();
    double m2;
    lzg_power_sum(g, 2.0, &m2);
    double d2;
    lzg_hill_number(g, 2.0, &d2);

    double expected_d2 = pow(m2, -1.0);

    printf("\n    M(2)=%.6e, D(2)=%.2f, expected=%.2f",
           m2, d2, expected_d2);

    ASSERT_MSG(m2 > 0, "M(2) > 0");
    ASSERT_MSG(fabs(d2 - expected_d2) / fmax(d2, 1) < 0.15, "D(2) == 1 / M(2)");

    /* D(2) ≤ D(1) ≤ D(0) (Hill number monotonicity) */
    double d0, d1;
    lzg_hill_number(g, 0.0, &d0);
    lzg_hill_number(g, 1.0, &d1);
    printf(" [D0=%.0f > D1=%.1f > D2=%.1f]", d0, d1, d2);
    ASSERT_MSG(d0 >= d1 - 1e-6, "D0 ≥ D1");
    ASSERT_MSG(d1 >= d2 - 1e-6, "D1 ≥ D2");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_number_d0_mc_matches_direct_formula(void) {
    const uint32_t n_samples = 4096;
    LZGGraph *g = build_coinflip_graph(3, 1);
    ASSERT_MSG(g != NULL, "coinflip graph");

    double d0;
    LZGError err = lzg_hill_number_mc(g, 0.0, n_samples, &d0);
    ASSERT_MSG(err == LZG_OK, "ok");

    LZGSimResult *sim = calloc(n_samples, sizeof(*sim));
    ASSERT_MSG(sim != NULL, "simulation alloc");

    LZGRng rng;
    lzg_rng_seed(&rng, 33333ULL);
    err = lzg_simulate(g, n_samples, &rng, sim);
    ASSERT_MSG(err == LZG_OK, "simulate ok");

    double expected = direct_mc_hill(sim, n_samples, 0.0);
    printf("\n    D(0)_mc=%.10f direct=%.10f", d0, expected);
    ASSERT_MSG(fabs(d0 - expected) < 1e-10, "D(0) matches direct MC formula");

    for (uint32_t i = 0; i < n_samples; i++) lzg_sim_result_free(&sim[i]);
    free(sim);
    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_numbers_batch(void) {
    LZGGraph *g = build_graph();
    double orders[] = {0, 1, 2, 3};
    double hills[4];
    LZGError err = lzg_hill_numbers(g, orders, 4, hills);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    D(0)=%.0f D(1)=%.2f D(2)=%.2f D(3)=%.2f",
           hills[0], hills[1], hills[2], hills[3]);

    /* Monotonicity: D(0) ≥ D(1) ≥ D(2) ≥ D(3) */
    for (int i = 0; i < 3; i++)
        ASSERT_MSG(hills[i] >= hills[i+1] - 1e-6, "monotone");

    lzg_graph_destroy(g);
    PASS();
}

static void test_hill_numbers_mc_match_direct_formula(void) {
    const uint32_t n_samples = 4096;
    LZGGraph *g = build_coinflip_graph(3, 1);
    ASSERT_MSG(g != NULL, "coinflip graph");

    double orders[] = {0.0, 1.0, 2.0};
    double hills[3];
    LZGError err = lzg_hill_numbers_mc(g, orders, 3, n_samples, hills);
    ASSERT_MSG(err == LZG_OK, "ok");

    LZGSimResult *sim = calloc(n_samples, sizeof(*sim));
    ASSERT_MSG(sim != NULL, "simulation alloc");

    LZGRng rng;
    lzg_rng_seed(&rng, 44444ULL);
    err = lzg_simulate(g, n_samples, &rng, sim);
    ASSERT_MSG(err == LZG_OK, "simulate ok");

    for (uint32_t i = 0; i < 3; i++) {
        double expected = direct_mc_hill(sim, n_samples, orders[i]);
        printf("\n    D(%.0f)_mc=%.10f direct=%.10f",
               orders[i], hills[i], expected);
        ASSERT_MSG(fabs(hills[i] - expected) < 1e-10,
                   "batch hill matches direct MC formula");
    }

    for (uint32_t i = 0; i < n_samples; i++) lzg_sim_result_free(&sim[i]);
    free(sim);
    lzg_graph_destroy(g);
    PASS();
}

static void test_pgen_dynamic_range(void) {
    LZGGraph *g = build_graph();
    LZGDynamicRange dr;
    LZGError err = lzg_pgen_dynamic_range(g, &dr);
    ASSERT_MSG(err == LZG_OK, "ok");

    printf("\n    max_logP=%.2f min_logP=%.2f range=%.1f orders",
           dr.max_log_prob, dr.min_log_prob, dr.dynamic_range_orders);

    ASSERT_MSG(dr.max_log_prob > dr.min_log_prob, "max > min");
    ASSERT_MSG(dr.max_log_prob < 0, "max < 0");
    ASSERT_MSG(dr.dynamic_range_nats > 0, "positive range");

    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_length_derivatives_partition(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    double global[5];
    LZGError err = lzg_flashback_pseq_derivatives(g, 1.0, 4, global);
    ASSERT_MSG(err == LZG_OK, "global derivatives");

    double *by_length = NULL;
    uint8_t *present = NULL;
    uint32_t max_length = 0;
    err = lzg_flashback_pseq_length_derivatives(
        g, 1.0, 4, &by_length, &present, &max_length);
    ASSERT_MSG(err == LZG_OK, "length derivatives");
    ASSERT_MSG(by_length != NULL && present != NULL, "length outputs");

    double sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    uint32_t lengths_present = 0;
    for (uint32_t length = 0; length <= max_length; length++) {
        if (!present[length]) continue;
        lengths_present++;
        for (uint32_t k = 0; k <= 4; k++)
            sums[k] += by_length[(size_t)length * 5 + k];
    }
    ASSERT_MSG(lengths_present >= 3, "multiple generated lengths");
    for (uint32_t k = 0; k <= 4; k++) {
        double scale = fmax(1.0, fabs(global[k]));
        ASSERT_MSG(fabs(sums[k] - global[k]) / scale < 1e-13,
                   "length derivatives partition global derivative");
    }

    free(by_length);
    free(present);
    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_attribution_conservation(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    LZGPseqAttribution attribution = {0};
    LZGError err = lzg_flashback_pseq_attribution(g, 1.0, &attribution);
    ASSERT_MSG(err == LZG_OK, "attribution");
    ASSERT_MSG(attribution.n_nodes == g->n_nodes, "node count");
    ASSERT_MSG(attribution.n_edges == g->n_edges, "edge count");
    ASSERT_MSG(fabs(attribution.log_mass) < 1e-14, "generated mass is one");
    ASSERT_MSG(fabs(attribution.node_probability[g->root_node] - 1.0) < 1e-14,
               "root marginal is one");

    double *incoming = calloc(g->n_nodes, sizeof(double));
    double *outgoing = calloc(g->n_nodes, sizeof(double));
    ASSERT_MSG(incoming != NULL && outgoing != NULL, "flow arrays");
    double entropy = 0.0;
    for (uint32_t u = 0; u < g->n_nodes; u++) {
        ASSERT_MSG(attribution.node_probability[u] >= 0.0 &&
                   attribution.node_probability[u] <= 1.0,
                   "bounded node marginal");
        for (uint32_t e = g->row_offsets[u]; e < g->row_offsets[u + 1]; e++) {
            const double flow = attribution.edge_probability[e];
            ASSERT_MSG(flow >= 0.0 && flow <= 1.0, "bounded edge marginal");
            outgoing[u] += flow;
            incoming[g->col_indices[e]] += flow;
            entropy -= flow * log(g->edge_weights[e]);
        }
    }
    double sink_mass = 0.0;
    for (uint32_t u = 0; u < g->n_nodes; u++) {
        const bool sink = g->row_offsets[u] == g->row_offsets[u + 1];
        if (sink) sink_mass += attribution.node_probability[u];
        if (u != g->root_node)
            ASSERT_MSG(fabs(incoming[u] - attribution.node_probability[u]) < 2e-14,
                       "incoming flow equals node marginal");
        if (!sink)
            ASSERT_MSG(fabs(outgoing[u] - attribution.node_probability[u]) < 2e-14,
                       "outgoing flow equals node marginal");
    }
    ASSERT_MSG(fabs(sink_mass - 1.0) < 2e-14, "sink marginals sum to one");

    double log_mass, raw[2], central[2];
    err = lzg_flashback_pseq_tilted_moments(
        g, 1.0, 1, &log_mass, raw, central);
    ASSERT_MSG(err == LZG_OK, "tilted moment");
    ASSERT_MSG(fabs(entropy + raw[1]) < 2e-14,
               "edge surprisal contributions sum to entropy");

    free(incoming);
    free(outgoing);
    lzg_flashback_pseq_attribution_destroy(&attribution);
    ASSERT_MSG(attribution.node_probability == NULL &&
               attribution.edge_probability == NULL,
               "destroy clears result");
    lzg_flashback_pseq_attribution_destroy(&attribution);
    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_edge_threshold_diversity(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    const double thresholds[] = {-INFINITY, 0.1, INFINITY};
    double log_d0[3], log_d1[3], log_d2[3], mass[3];
    uint64_t kept[3];
    LZGError err = lzg_flashback_edge_threshold_diversity(
        g, thresholds, 3, log_d0, log_d1, log_d2, mass, kept);
    ASSERT_MSG(err == LZG_OK, "threshold diversity");

    double path_count, d1, d2;
    err = lzg_flashback_path_count(g, &path_count);
    ASSERT_MSG(err == LZG_OK, "path count");
    err = lzg_flashback_hill_number(g, 1.0, &d1);
    ASSERT_MSG(err == LZG_OK, "D1");
    err = lzg_flashback_hill_number(g, 2.0, &d2);
    ASSERT_MSG(err == LZG_OK, "D2");
    ASSERT_MSG(fabs(log_d0[0] - log(path_count)) < 2e-14,
               "unpruned D0");
    ASSERT_MSG(fabs(log_d1[0] - log(d1)) < 2e-14, "unpruned D1");
    ASSERT_MSG(fabs(log_d2[0] - log(d2)) < 2e-14, "unpruned D2");
    ASSERT_MSG(fabs(mass[0] - 1.0) < 2e-14, "unpruned mass");
    ASSERT_MSG(kept[0] == g->n_edges, "all edges retained");

    uint64_t expected_kept = 0;
    for (uint32_t e = 0; e < g->n_edges; e++)
        if (g->edge_weights[e] > thresholds[1]) expected_kept++;
    ASSERT_MSG(kept[1] == expected_kept, "strict threshold edge count");
    ASSERT_MSG(kept[2] == 0, "infinite threshold removes every edge");
    ASSERT_MSG(isinf(log_d0[2]) && log_d0[2] < 0.0,
               "empty D0 is negative infinity");
    ASSERT_MSG(isinf(log_d1[2]) && log_d1[2] < 0.0,
               "empty D1 is negative infinity");
    ASSERT_MSG(isinf(log_d2[2]) && log_d2[2] < 0.0,
               "empty D2 is negative infinity");
    ASSERT_MSG(mass[2] == 0.0, "empty mass is zero");

    const double unsorted[] = {0.2, 0.1};
    err = lzg_flashback_edge_threshold_diversity(
        g, unsorted, 2, log_d0, log_d1, log_d2, mass, kept);
    ASSERT_MSG(err == LZG_ERR_INVALID_ARG, "C API requires sorted thresholds");

    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_histogram_conservation(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    const uint32_t bins = 257;
    double *weights = NULL;
    double spacing = 0.0, true_max = 0.0;
    uint32_t max_edges = 0;
    LZGError err = lzg_flashback_pseq_histogram(
        g, bins, 1.0, -1, &weights, &spacing, &true_max, &max_edges);
    ASSERT_MSG(err == LZG_OK && weights != NULL, "global histogram");
    ASSERT_MSG(spacing > 0.0 && true_max > 0.0 && max_edges > 0,
               "histogram metadata");

    double mass = 0.0, mean = 0.0;
    for (uint32_t i = 0; i < bins; i++) {
        ASSERT_MSG(weights[i] >= 0.0, "non-negative histogram mass");
        mass += weights[i];
        mean += (double)i * spacing * weights[i];
    }

    double derivatives[2];
    err = lzg_flashback_pseq_derivatives(g, 1.0, 1, derivatives);
    ASSERT_MSG(err == LZG_OK, "derivatives");
    ASSERT_MSG(fabs(mass - derivatives[0]) < 1e-13,
               "histogram conserves generated probability mass");
    ASSERT_MSG(fabs(mean + derivatives[1]) < 1e-12,
               "linear transport preserves mean surprisal");

    double length_mass = 0.0;
    for (int64_t length = 0; length <= 16; length++) {
        double *length_weights = NULL;
        err = lzg_flashback_pseq_histogram(
            g, bins, 1.0, length, &length_weights, &spacing,
            &true_max, &max_edges);
        ASSERT_MSG(err == LZG_OK && length_weights != NULL,
                   "length histogram");
        for (uint32_t i = 0; i < bins; i++) length_mass += length_weights[i];
        free(length_weights);
    }
    ASSERT_MSG(fabs(length_mass - mass) < 1e-13,
               "length histograms partition global histogram mass");

    free(weights);
    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_histogram_pair(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");
    const uint32_t bins = 257;
    double *paired_counting = NULL, *paired_generated = NULL;
    double *single_counting = NULL, *single_generated = NULL;
    double pair_spacing, pair_max, single_spacing, single_max;
    uint32_t pair_edges, single_edges;
    LZGError err = lzg_flashback_pseq_histogram_pair(
        g, bins, &paired_counting, &paired_generated,
        &pair_spacing, &pair_max, &pair_edges);
    ASSERT_MSG(err == LZG_OK && paired_counting && paired_generated,
               "paired histogram");
    err = lzg_flashback_pseq_histogram(
        g, bins, 0.0, -1, &single_counting,
        &single_spacing, &single_max, &single_edges);
    ASSERT_MSG(err == LZG_OK && single_counting, "single counting histogram");
    ASSERT_MSG(pair_spacing == single_spacing && pair_max == single_max &&
               pair_edges == single_edges, "counting metadata agrees");
    err = lzg_flashback_pseq_histogram(
        g, bins, 1.0, -1, &single_generated,
        &single_spacing, &single_max, &single_edges);
    ASSERT_MSG(err == LZG_OK && single_generated, "single generated histogram");
    ASSERT_MSG(pair_spacing == single_spacing && pair_max == single_max &&
               pair_edges == single_edges, "generated metadata agrees");
    for (uint32_t i = 0; i < bins; i++) {
        ASSERT_MSG(fabs(paired_counting[i] - single_counting[i]) <
                       1e-12 * fmax(1.0, fabs(single_counting[i])),
                   "paired counting bins match single kernel");
        ASSERT_MSG(fabs(paired_generated[i] - single_generated[i]) <
                       1e-12 * fmax(1.0, fabs(single_generated[i])),
                   "paired generated bins match single kernel");
    }
    free(paired_counting); free(paired_generated);
    free(single_counting); free(single_generated);
    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_tilted_moments(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    double log_mass, raw[5], central[5], derivatives[5];
    LZGError err = lzg_flashback_pseq_tilted_moments(
        g, 1.0, 4, &log_mass, raw, central);
    ASSERT_MSG(err == LZG_OK, "tilted moments");
    err = lzg_flashback_pseq_derivatives(g, 1.0, 4, derivatives);
    ASSERT_MSG(err == LZG_OK, "derivatives");
    ASSERT_MSG(fabs(log_mass - log(derivatives[0])) < 1e-14,
               "log mass matches unnormalized derivative");
    for (uint32_t r = 0; r <= 4; r++) {
        double expected = derivatives[r] / derivatives[0];
        ASSERT_MSG(fabs(raw[r] - expected) < 2e-13 * fmax(1.0, fabs(expected)),
                   "normalized raw moment matches derivative ratio");
    }
    ASSERT_MSG(fabs(central[0] - 1.0) < 1e-15 && central[1] == 0.0,
               "central moment conventions");
    ASSERT_MSG(fabs(central[2] - (raw[2] - raw[1] * raw[1])) < 2e-13,
               "central variance matches raw identity");

    double extreme_log_mass, extreme_raw[3], extreme_central[3];
    err = lzg_flashback_pseq_tilted_moments(
        g, -1000.0, 2, &extreme_log_mass, extreme_raw, extreme_central);
    ASSERT_MSG(err == LZG_OK && isfinite(extreme_log_mass),
               "extreme negative tilt remains finite");
    ASSERT_MSG(isfinite(extreme_raw[1]) && extreme_central[2] >= 0.0,
               "extreme tilted moments remain valid");

    lzg_graph_destroy(g);
    PASS();
}

static void test_flashback_pseq_saddlepoint_roots(void) {
    LZGGraph *g = build_flashback_graph();
    ASSERT_MSG(g != NULL, "flashback graph");

    double log_mass, raw[3], central[3];
    LZGError err = lzg_flashback_pseq_tilted_moments(
        g, 1.0, 2, &log_mass, raw, central);
    ASSERT_MSG(err == LZG_OK && central[2] > 0.0, "base tilted moments");
    const double mean = -raw[1];
    const double delta = 0.25 * sqrt(central[2]);
    double x[3] = {mean - delta, mean, mean + delta};
    double pdf[3], cdf[3], saddle[3];
    uint32_t iterations[3];
    err = lzg_flashback_pseq_saddlepoint_batch(
        g, x, 3, pdf, cdf, saddle, iterations);
    ASSERT_MSG(err == LZG_OK, "saddlepoint batch");
    ASSERT_MSG(cdf[0] <= cdf[1] && cdf[1] <= cdf[2], "monotone CDF");
    for (uint32_t i = 0; i < 3; i++) {
        ASSERT_MSG(pdf[i] >= 0.0 && cdf[i] >= 0.0 && cdf[i] <= 1.0,
                   "valid saddlepoint outputs");
        ASSERT_MSG(iterations[i] <= 20, "safeguarded Newton converges quickly");
        double tilted_log_mass, tilted_raw[3], tilted_central[3];
        err = lzg_flashback_pseq_tilted_moments(
            g, 1.0 - saddle[i], 2, &tilted_log_mass,
            tilted_raw, tilted_central);
        ASSERT_MSG(err == LZG_OK, "root tilted moments");
        ASSERT_MSG(fabs(-tilted_raw[1] - x[i]) < 2e-12,
                   "saddlepoint satisfies K'(t)=x");
    }

    lzg_graph_destroy(g);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("C-LZGraph Unit Tests — Phase 5: Analytics\n");
    printf("==========================================\n\n");

    printf("[analytics]\n");
    RUN_TEST(test_simulation_potential_size);
    RUN_TEST(test_pgen_diagnostics);
    RUN_TEST(test_leaky_graph_has_exact_raw_diagnostics_but_proper_public_m1);
    RUN_TEST(test_pgen_diagnostics_atol_contract);
    RUN_TEST(test_effective_diversity);
    RUN_TEST(test_hill_number_d0);
    RUN_TEST(test_hill_number_d1);
    RUN_TEST(test_hill_number_d2);
    RUN_TEST(test_hill_number_d0_mc_matches_direct_formula);
    RUN_TEST(test_hill_numbers_batch);
    RUN_TEST(test_hill_numbers_mc_match_direct_formula);
    RUN_TEST(test_pgen_dynamic_range);
    RUN_TEST(test_flashback_pseq_length_derivatives_partition);
    RUN_TEST(test_flashback_pseq_attribution_conservation);
    RUN_TEST(test_flashback_edge_threshold_diversity);
    RUN_TEST(test_flashback_pseq_histogram_conservation);
    RUN_TEST(test_flashback_pseq_histogram_pair);
    RUN_TEST(test_flashback_pseq_tilted_moments);
    RUN_TEST(test_flashback_pseq_saddlepoint_roots);

    printf("\n==========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
