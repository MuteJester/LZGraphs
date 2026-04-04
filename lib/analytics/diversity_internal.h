#ifndef LZGRAPH_DIVERSITY_INTERNAL_H
#define LZGRAPH_DIVERSITY_INTERNAL_H

#include "lzgraph/diversity.h"

#ifndef M_LN2
#define M_LN2 0.6931471805599453
#endif

double lzg_sequence_perplexity_impl(const LZGGraph *g,
                                    const char *seq, uint32_t seq_len);
double lzg_repertoire_perplexity_impl(const LZGGraph *g,
                                      const char **sequences, uint32_t n);
double lzg_path_entropy_rate_impl(const LZGGraph *g,
                                  const char **sequences, uint32_t n);

LZGError lzg_k_diversity_impl(const char **sequences, uint32_t n,
                              LZGVariant variant,
                              uint32_t sample_size, uint32_t draws,
                              LZGRng *rng, LZGKDiversity *out);

LZGError lzg_saturation_curve_impl(const char **sequences, uint32_t n,
                                   LZGVariant variant,
                                   uint32_t log_every,
                                   LZGSaturationPoint *out,
                                   uint32_t *out_count);

LZGError lzg_jensen_shannon_divergence_impl(const LZGGraph *a,
                                            const LZGGraph *b,
                                            double *out);

#endif /* LZGRAPH_DIVERSITY_INTERNAL_H */
