/**
 * @file sharing.c
 * @brief Sharing spectrum prediction via Gauss-Hermite quadrature.
 */
#include "lzgraph/sharing.h"
#include "lzgraph/pgen_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Precomputed Gauss-Hermite nodes/weights (probabilist, n=20) ── */
/* These are for ∫ f(x) exp(-x²/2) dx ≈ Σ w_i f(x_i)               */
/* Generated offline. 20 points suffice for smooth integrands.       */

#define GH_N 20

static const double gh_nodes[GH_N] = {
    -5.38748089001, -4.60368244955, -3.94476404012, -3.34785456738,
    -2.78880605843, -2.25497400209, -1.73853771212, -1.23407621540,
    -0.73747372854, -0.24534070830,  0.24534070830,  0.73747372854,
     1.23407621540,  1.73853771212,  2.25497400209,  2.78880605843,
     3.34785456738,  3.94476404012,  4.60368244955,  5.38748089001,
};

static const double gh_weights[GH_N] = {
    2.22939364554e-13, 4.39934099226e-10, 1.08606937077e-07, 7.80255647853e-06,
    2.28338636017e-04, 3.24377334224e-03, 2.48105208875e-02, 1.09017206020e-01,
    2.86675505363e-01, 4.62243669601e-01, 4.62243669601e-01, 2.86675505363e-01,
    1.09017206020e-01, 2.48105208875e-02, 3.24377334224e-03, 2.28338636017e-04,
    7.80255647853e-06, 1.08606937077e-07, 4.39934099226e-10, 2.22939364554e-13,
};

static const double INV_SQRT_2PI = 0.3989422804014327;

/* ── Standard normal CDF ─────────────────────────────────── */

static double std_norm_cdf(double x) {
    return 0.5 * erfc(-x / 1.4142135623730951);
}

/* ── Poisson-Binomial PMF via DP convolution ─────────────── */
/* Returns Pr[X = k] for k = 1..max_k where X ~ PoissonBinomial(q) */

static void poisson_binomial_pmf(const double *q, uint32_t n,
                                  uint32_t max_k, double *pmf_out) {
    /* DP: pmf[j] = Pr[exactly j successes out of first i trials] */
    uint32_t sz = (max_k < n ? max_k : n) + 1;
    double *pmf = calloc(sz, sizeof(double));
    double *tmp = calloc(sz, sizeof(double));
    pmf[0] = 1.0;

    for (uint32_t i = 0; i < n; i++) {
        double qi = q[i];
        double ri = 1.0 - qi;
        tmp[0] = pmf[0] * ri;
        for (uint32_t j = 1; j < sz; j++)
            tmp[j] = pmf[j] * ri + pmf[j - 1] * qi;
        double *swap = pmf; pmf = tmp; tmp = swap;
    }

    /* Copy k=1..max_k to output */
    for (uint32_t k = 0; k < max_k; k++)
        pmf_out[k] = (k + 1 < sz) ? fmax(pmf[k + 1], 0.0) : 0.0;

    free(pmf);
    free(tmp);
}

/* ── Normal approximation PMF with continuity correction ──── */

static void normal_approx_pmf(const double *q, uint32_t n,
                                uint32_t max_k, double *pmf_out) {
    double mu = 0.0, var = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        mu += q[i];
        var += q[i] * (1.0 - q[i]);
    }

    if (var < 1e-15) {
        /* Degenerate: use Poisson approximation */
        if (mu < 1e-15) {
            memset(pmf_out, 0, max_k * sizeof(double));
            return;
        }
        double log_mu = log(mu);
        for (uint32_t k = 0; k < max_k; k++) {
            double kp1 = (double)(k + 1);
            pmf_out[k] = exp(-mu + kp1 * log_mu - lgamma(kp1 + 1.0));
        }
        return;
    }

    double sigma = sqrt(var);
    for (uint32_t k = 0; k < max_k; k++) {
        double kp1 = (double)(k + 1);
        double upper = (kp1 + 0.5 - mu) / sigma;
        double lower = (kp1 - 0.5 - mu) / sigma;
        pmf_out[k] = fmax(std_norm_cdf(upper) - std_norm_cdf(lower), 0.0);
    }
}

/* ── Sharing PMF for a single sequence probability ────────── */

static void sharing_pmf(double p_seq, const double *draw_counts,
                         uint32_t n_donors, uint32_t max_k,
                         double *pmf_out) {
    /* Compute q_i = 1 - exp(-d_i * p) for each donor */
    double *q = malloc(n_donors * sizeof(double));
    for (uint32_t i = 0; i < n_donors; i++)
        q[i] = 1.0 - exp(-draw_counts[i] * p_seq);

    if (n_donors <= 500) {
        poisson_binomial_pmf(q, n_donors, max_k, pmf_out);
    } else {
        normal_approx_pmf(q, n_donors, max_k, pmf_out);
    }

    free(q);
}

/* ═══════════════════════════════════════════════════════════════ */

LZGError lzg_predict_sharing(const LZGGraph *g,
                                       const double *draw_counts,
                                       uint32_t n_donors,
                                       uint32_t max_k,
                                       LZGSharingSpectrum *out) {
    if (!g || !draw_counts || !out || n_donors == 0)
        return LZG_ERR_INVALID_ARG;

    /* Get analytical PGEN distribution */
    LZGPgenDist dist;
    LZGError err = lzg_pgen_analytical(g, &dist);
    if (err != LZG_OK) return err;

    if (max_k == 0) max_k = (n_donors < 500) ? n_donors : 500;

    double *spectrum = calloc(max_k, sizeof(double));
    double *pmf_buf  = malloc(max_k * sizeof(double));

    /* Quadrature over each Gaussian mixture component */
    for (uint32_t c = 0; c < dist.n_components; c++) {
        double w_L = dist.weights[c];
        double mu_L = dist.means[c];
        double sigma_L = dist.stds[c];

        if (w_L < 1e-15) continue;

        if (sigma_L < 1e-15) {
            /* Point mass: evaluate directly */
            double p = exp(mu_L);
            double count_weight = w_L / fmax(p, LZG_EPS);
            sharing_pmf(p, draw_counts, n_donors, max_k, pmf_buf);
            for (uint32_t k = 0; k < max_k; k++)
                spectrum[k] += count_weight * pmf_buf[k];
            continue;
        }

        /* Count-weight via completing the square:
         * ∫ g_L(x)/e^x · h(x) dx
         * = w_L · exp(-μ + σ²/2) · (1/√2π) · Σ_j w_j · h(μ - σ² + σ·u_j) */
        double log_pf = log(fmax(w_L, 1e-300)) - mu_L + sigma_L * sigma_L / 2.0;
        double prefactor = exp(log_pf) * INV_SQRT_2PI;

        for (uint32_t j = 0; j < GH_N; j++) {
            double log_p = mu_L - sigma_L * sigma_L + sigma_L * gh_nodes[j];
            double p = exp(log_p);

            sharing_pmf(p, draw_counts, n_donors, max_k, pmf_buf);

            double w = prefactor * gh_weights[j];
            for (uint32_t k = 0; k < max_k; k++)
                spectrum[k] += w * pmf_buf[k];
        }
    }

    /* Clip negatives */
    double total = 0.0;
    for (uint32_t k = 0; k < max_k; k++) {
        if (spectrum[k] < 0.0) spectrum[k] = 0.0;
        total += spectrum[k];
    }

    out->spectrum = spectrum;
    out->max_k = max_k;
    out->n_donors = n_donors;
    out->expected_total = total;

    double d_total = 0.0;
    for (uint32_t i = 0; i < n_donors; i++) d_total += draw_counts[i];
    out->total_draws = d_total;

    free(pmf_buf);
    return LZG_OK;
}

void lzg_sharing_spectrum_free(LZGSharingSpectrum *ss) {
    if (ss && ss->spectrum) { free(ss->spectrum); ss->spectrum = NULL; }
}
