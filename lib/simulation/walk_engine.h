#ifndef LZGRAPH_WALK_ENGINE_H
#define LZGRAPH_WALK_ENGINE_H

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/rng.h"
#include "lzgraph/walk_dict.h"

#define LZG_WALK_ENGINE_MAX_DEPTH 128u
#define LZG_WALK_ENGINE_MAX_BLACKLIST 64u
#define LZG_WALK_ENGINE_MAX_VALID_EDGES 512u
#define LZG_WALK_ENGINE_SEQ_BUF_CAP 1024u

typedef bool (*LZGWalkEdgeFilterFn)(const LZGGraph *g, uint32_t edge, void *ctx);

typedef enum {
    LZG_WALK_ENGINE_OUTCOME_FAILED = 0,
    LZG_WALK_ENGINE_OUTCOME_ABSORBED,
    LZG_WALK_ENGINE_OUTCOME_LEAKED,
} LZGWalkEngineOutcome;

typedef struct {
    LZGWalkEdgeFilterFn edge_filter;
    void               *edge_filter_ctx;
    bool                stop_on_dead_end;
} LZGWalkEngineConfig;

typedef struct {
    uint32_t node;
    uint32_t edge_taken;
    uint8_t  sp_len;
    double   log_edge_prob;
    uint32_t blacklist[LZG_WALK_ENGINE_MAX_BLACKLIST];
    uint32_t n_blacklisted;
} LZGWalkEngineFrame;

typedef struct {
    char     sequence[LZG_WALK_ENGINE_SEQ_BUF_CAP];
    uint32_t seq_len;
    uint32_t n_tokens;
    uint32_t depth;
    double   log_prob;
    LZGWalkEngineOutcome outcome;
} LZGWalkEngineResult;

bool lzg_walk_engine_run(const LZGGraph *g, LZGRng *rng,
                         const LZGWalkEngineConfig *cfg,
                         LZGWalkEngineResult *out);

#endif /* LZGRAPH_WALK_ENGINE_H */
