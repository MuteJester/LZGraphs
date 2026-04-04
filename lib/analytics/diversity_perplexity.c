#include "diversity_internal.h"
#include "lzgraph/lz76.h"
#include "lzgraph/simulate.h"
#include "lzgraph/string_pool.h"
#include <math.h>
#include <string.h>

static bool lzg_diversity_sequence_stats(const LZGGraph *g,
                                         const char *seq,
                                         uint32_t seq_len,
                                         double *out_log_prob,
                                         uint32_t *out_token_count) {
    LZGTokens tokens;
    double log_prob;

    if (!g || !seq || seq_len == 0) return false;

    log_prob = lzg_walk_log_prob(g, seq, seq_len);
    if (log_prob <= LZG_LOG_EPS + 1.0) return false;

    lzg_lz76_decompose(seq, seq_len, (LZGStringPool *)g->pool, &tokens);
    if (tokens.count == 0) return false;

    *out_log_prob = log_prob;
    *out_token_count = tokens.count;
    return true;
}

double lzg_sequence_perplexity_impl(const LZGGraph *g,
                                    const char *seq, uint32_t seq_len) {
    double log_prob;
    uint32_t token_count;

    if (!lzg_diversity_sequence_stats(g, seq, seq_len, &log_prob, &token_count))
        return INFINITY;

    return exp(-log_prob / (double)token_count);
}

double lzg_repertoire_perplexity_impl(const LZGGraph *g,
                                      const char **sequences, uint32_t n) {
    double sum_cross_entropy = 0.0;
    uint32_t valid = 0;

    if (!g || !sequences || n == 0) return INFINITY;

    for (uint32_t i = 0; i < n; i++) {
        double log_prob;
        uint32_t token_count;
        uint32_t seq_len = (uint32_t)strlen(sequences[i]);

        if (!lzg_diversity_sequence_stats(g, sequences[i], seq_len,
                                          &log_prob, &token_count)) {
            return INFINITY;
        }

        sum_cross_entropy += (-log_prob) / (double)token_count;
        valid++;
    }

    if (valid == 0) return INFINITY;
    return exp(sum_cross_entropy / (double)valid);
}

double lzg_path_entropy_rate_impl(const LZGGraph *g,
                                  const char **sequences, uint32_t n) {
    double sum_bits_per_token = 0.0;
    uint32_t valid = 0;

    if (!g || !sequences || n == 0) return 0.0;

    for (uint32_t i = 0; i < n; i++) {
        double log_prob;
        uint32_t token_count;
        uint32_t seq_len = (uint32_t)strlen(sequences[i]);

        if (!lzg_diversity_sequence_stats(g, sequences[i], seq_len,
                                          &log_prob, &token_count)) {
            continue;
        }

        sum_bits_per_token += (-log_prob) / ((double)token_count * M_LN2);
        valid++;
    }

    return valid > 0 ? sum_bits_per_token / (double)valid : 0.0;
}
