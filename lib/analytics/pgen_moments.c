#include "pgen_moments.h"
#include "lzgraph/forward.h"
#include <math.h>
#include <string.h>

static void mom_seed(double *acc, double p, void *ctx) {
    (void)ctx;
    double lp = log(fmax(p, 1e-300));
    acc[0] += p;
    acc[1] += p * lp;
    acc[2] += p * lp * lp;
    acc[3] += p * lp * lp * lp;
    acc[4] += p * lp * lp * lp * lp;
}

static void mom_edge(double *dst, const double *src, double w, double Z, void *ctx) {
    (void)ctx;
    double ratio = w / Z;
    double lr = log(fmax(ratio, 1e-300));
    dst[0] = src[0] * ratio;
    dst[1] = src[0] * ratio * lr + src[1] * ratio;
    dst[2] = src[0] * ratio * lr * lr + 2.0 * src[1] * ratio * lr + src[2] * ratio;
    dst[3] = src[0] * ratio * lr * lr * lr + 3.0 * src[1] * ratio * lr * lr
             + 3.0 * src[2] * ratio * lr + src[3] * ratio;
    dst[4] = src[0] * ratio * lr * lr * lr * lr + 4.0 * src[1] * ratio * lr * lr * lr
             + 6.0 * src[2] * ratio * lr * lr + 4.0 * src[3] * ratio * lr + src[4] * ratio;
}

static void mom_absorb(double *total, const double *acc, double sp, void *ctx) {
    (void)ctx;
    double lsp = log(fmax(sp, 1e-300));
    double m = acc[0] * sp;
    total[0] += m;
    total[1] += (acc[1] * sp + acc[0] * sp * lsp);
    total[2] += (acc[2] * sp + 2.0 * acc[1] * sp * lsp + acc[0] * sp * lsp * lsp);
    total[3] += (acc[3] * sp + 3.0 * acc[2] * sp * lsp + 3.0 * acc[1] * sp * lsp * lsp
                 + acc[0] * sp * lsp * lsp * lsp);
    total[4] += (acc[4] * sp + 4.0 * acc[3] * sp * lsp + 6.0 * acc[2] * sp * lsp * lsp
                 + 4.0 * acc[1] * sp * lsp * lsp * lsp
                 + acc[0] * sp * lsp * lsp * lsp * lsp);
}

static void mom_cont(double *co, const double *acc, double sp, void *ctx) {
    (void)ctx;
    double csp = 1.0 - sp;
    double lcsp = log(fmax(csp, 1e-300));
    co[0] = acc[0] * csp;
    co[1] = acc[1] * csp + acc[0] * csp * lcsp;
    co[2] = acc[2] * csp + 2.0 * acc[1] * csp * lcsp + acc[0] * csp * lcsp * lcsp;
    co[3] = acc[3] * csp + 3.0 * acc[2] * csp * lcsp + 3.0 * acc[1] * csp * lcsp * lcsp
            + acc[0] * csp * lcsp * lcsp * lcsp;
    co[4] = acc[4] * csp + 4.0 * acc[3] * csp * lcsp + 6.0 * acc[2] * csp * lcsp * lcsp
            + 4.0 * acc[1] * csp * lcsp * lcsp * lcsp
            + acc[0] * csp * lcsp * lcsp * lcsp * lcsp;
}

LZGError lzg_pgen_compute_moments(const LZGGraph *g, LZGPgenMoments *out) {
    if (!g || !out) return LZG_ERR_INVALID_ARG;
    if (!g->topo_valid) return LZG_ERR_NOT_BUILT;

    LZGFwdOps ops = {
        .seed = mom_seed,
        .edge = mom_edge,
        .absorb = mom_absorb,
        .cont = mom_cont,
        .acc_dim = 5,
        .ctx = NULL,
    };

    double total[5] = {0};
    LZGError err = lzg_forward_propagate(g, &ops, total);
    if (err != LZG_OK) return err;

    double mass = total[0];
    if (mass < 1e-300) {
        memset(out, 0, sizeof(*out));
        return LZG_OK;
    }

    out->total_mass = mass;
    out->mean = total[1] / mass;
    {
        double var = total[2] / mass - out->mean * out->mean;
        out->variance = fmax(var, 0.0);
    }
    out->std = sqrt(out->variance);

    if (out->std > 1e-15) {
        double m3 = total[3] / mass - 3.0 * out->mean * total[2] / mass
                     + 2.0 * out->mean * out->mean * out->mean;
        out->skewness = m3 / (out->std * out->std * out->std);

        double m4 = total[4] / mass - 4.0 * out->mean * total[3] / mass
                     + 6.0 * out->mean * out->mean * total[2] / mass
                     - 3.0 * out->mean * out->mean * out->mean * out->mean;
        out->kurtosis = m4 / (out->variance * out->variance) - 3.0;
    } else {
        out->skewness = 0.0;
        out->kurtosis = 0.0;
    }

    return LZG_OK;
}
