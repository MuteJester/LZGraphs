/**
 * @file diversity.c
 * @brief Public diversity API over internal diversity submodules.
 */
#include "lzgraph/diversity.h"
#include "diversity_internal.h"

double lzg_sequence_perplexity(const LZGGraph *g,
                               const char *seq, uint32_t seq_len) {
    return lzg_sequence_perplexity_impl(g, seq, seq_len);
}

double lzg_repertoire_perplexity(const LZGGraph *g,
                                 const char **sequences, uint32_t n) {
    return lzg_repertoire_perplexity_impl(g, sequences, n);
}

double lzg_path_entropy_rate(const LZGGraph *g,
                             const char **sequences, uint32_t n) {
    return lzg_path_entropy_rate_impl(g, sequences, n);
}

LZGError lzg_k_diversity(const char **sequences, uint32_t n,
                         LZGVariant variant,
                         uint32_t sample_size, uint32_t draws,
                         LZGRng *rng, LZGKDiversity *out) {
    return lzg_k_diversity_impl(sequences, n, variant, sample_size, draws,
                                rng, out);
}

LZGError lzg_saturation_curve(const char **sequences, uint32_t n,
                              LZGVariant variant,
                              uint32_t log_every,
                              LZGSaturationPoint *out,
                              uint32_t *out_count) {
    return lzg_saturation_curve_impl(sequences, n, variant, log_every, out,
                                     out_count);
}

LZGError lzg_jensen_shannon_divergence(const LZGGraph *a,
                                       const LZGGraph *b,
                                       double *out) {
    return lzg_jensen_shannon_divergence_impl(a, b, out);
}
