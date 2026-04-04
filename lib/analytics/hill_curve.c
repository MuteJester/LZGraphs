#include "lzgraph/analytics.h"
#include <stdlib.h>
#include <string.h>

static const double DEFAULT_ORDERS[] = {
    0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0
};

#define DEFAULT_N_ORDERS 12u

LZGError lzg_hill_curve(const LZGGraph *g, const double *orders,
                        uint32_t n, LZGHillCurve *out) {
    LZGError err;

    if (!g || !out) return LZG_ERR_INVALID_ARG;

    if (!orders || n == 0) {
        orders = DEFAULT_ORDERS;
        n = DEFAULT_N_ORDERS;
    }

    out->n = 0;
    out->orders = malloc(n * sizeof(double));
    out->hill_numbers = malloc(n * sizeof(double));
    if (!out->orders || !out->hill_numbers) {
        free(out->orders);
        free(out->hill_numbers);
        out->orders = NULL;
        out->hill_numbers = NULL;
        return LZG_ERR_ALLOC;
    }

    memcpy(out->orders, orders, n * sizeof(double));
    err = lzg_hill_numbers(g, orders, n, out->hill_numbers);
    if (err != LZG_OK) {
        free(out->orders);
        free(out->hill_numbers);
        out->orders = NULL;
        out->hill_numbers = NULL;
        return err;
    }

    out->n = n;
    return LZG_OK;
}

void lzg_hill_curve_free(LZGHillCurve *hc) {
    if (!hc) return;

    free(hc->orders);
    free(hc->hill_numbers);
    hc->orders = NULL;
    hc->hill_numbers = NULL;
    hc->n = 0;
}
