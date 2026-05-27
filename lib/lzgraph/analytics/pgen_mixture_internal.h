#ifndef LZGRAPH_PGEN_MIXTURE_H
#define LZGRAPH_PGEN_MIXTURE_H

#include "lzgraph/pgen_dist.h"

LZGError lzg_pgen_build_analytical_mixture(const LZGGraph *g,
                                           const LZGPgenMoments *global,
                                           LZGPgenDist *out);

#endif /* LZGRAPH_PGEN_MIXTURE_H */
