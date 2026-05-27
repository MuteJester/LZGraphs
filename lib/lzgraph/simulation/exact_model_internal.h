#ifndef LZGRAPH_EXACT_MODEL_H
#define LZGRAPH_EXACT_MODEL_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"

typedef struct {
    char     sequence[1024];
    uint32_t seq_len;
    uint32_t n_tokens;
    double   log_prob;
} LZGExactSample;

LZGError lzg_exact_model_ensure(LZGGraph *g);
void lzg_exact_model_invalidate(LZGGraph *g);
double lzg_exact_model_root_absorption(const LZGGraph *g);
uint32_t lzg_exact_model_mc_samples(const LZGGraph *g);
LZGError lzg_exact_model_sample(const LZGGraph *g, LZGRng *rng,
                                LZGExactSample *out);

#endif /* LZGRAPH_EXACT_MODEL_H */
