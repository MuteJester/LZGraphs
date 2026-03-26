/**
 * @file wynn.c
 * @brief Wynn epsilon algorithm (Shanks transformation).
 *
 * The epsilon table is built from partial sums:
 *   eps[-1][j] = 0       (conceptual)
 *   eps[0][j]  = S_j     (partial sums)
 *   eps[k][j]  = eps[k-2][j+1] + 1 / (eps[k-1][j+1] - eps[k-1][j])
 *
 * The even-indexed columns eps[2m][0] are the Shanks transforms.
 * The best estimate is the last even-indexed entry computed.
 */
#include "lzgraph/wynn.h"
#include <stdlib.h>
#include <math.h>

double lzg_wynn_epsilon(const double *partial_sums, uint32_t n) {
    if (n == 0) return 0.0;
    if (n == 1) return partial_sums[0];
    if (n == 2) return partial_sums[1];

    /* Allocate epsilon table: two columns (current and previous) */
    /* We only need two columns at a time for the recurrence. */
    double *prev = calloc(n, sizeof(double)); /* eps[k-2] */
    double *curr = malloc(n * sizeof(double)); /* eps[k-1] */
    double *next = malloc(n * sizeof(double)); /* eps[k]   */

    if (!prev || !curr || !next) {
        free(prev); free(curr); free(next);
        return partial_sums[n - 1];
    }

    /* Initialize: eps[0][j] = S_j */
    for (uint32_t j = 0; j < n; j++)
        curr[j] = partial_sums[j];
    /* prev[j] = 0 (eps[-1]) — already zeroed by calloc */

    double best = partial_sums[n - 1];
    uint32_t width = n; /* number of valid entries in current column */

    /* Build epsilon table column by column */
    for (uint32_t k = 1; width >= 2; k++) {
        uint32_t new_width = width - 1;

        for (uint32_t j = 0; j < new_width; j++) {
            double diff = curr[j + 1] - curr[j];
            if (fabs(diff) < 1e-300) {
                /* Denominator too small — use current best */
                next[j] = curr[j + 1];
            } else {
                next[j] = prev[j + 1] + 1.0 / diff;
            }
        }

        /* Even-indexed columns (k even) contain the Shanks transforms */
        if (k % 2 == 0 && new_width > 0) {
            best = next[0]; /* eps[k][0] is the best current estimate */
        }

        /* Rotate: prev ← curr, curr ← next */
        double *tmp = prev;
        prev = curr;
        curr = next;
        next = tmp;
        width = new_width;
    }

    free(prev);
    free(curr);
    free(next);
    return best;
}
