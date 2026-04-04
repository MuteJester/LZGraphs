#include "walk_engine.h"
#include <math.h>
#include <string.h>

static bool is_blacklisted(const LZGWalkEngineFrame *frame, uint32_t edge) {
    for (uint32_t i = 0; i < frame->n_blacklisted; i++) {
        if (frame->blacklist[i] == edge) return true;
    }
    return false;
}

static bool edge_passes_filter(const LZGWalkEngineConfig *cfg,
                               const LZGGraph *g, uint32_t edge) {
    return !cfg || !cfg->edge_filter || cfg->edge_filter(g, edge, cfg->edge_filter_ctx);
}

static bool is_sink_node(const LZGGraph *g, uint32_t node) {
    return g->node_is_sink ? g->node_is_sink[node] : false;
}

static uint32_t collect_valid_edges(const LZGGraph *g, const LZGWalkDict *wd,
                                    uint32_t node, const LZGWalkEngineFrame *frame,
                                    const LZGWalkEngineConfig *cfg,
                                    uint32_t *out_edges, double *out_wts) {
    uint32_t e_start = g->row_offsets[node];
    uint32_t e_end = g->row_offsets[node + 1];
    uint32_t n_valid = 0;

    for (uint32_t e = e_start; e < e_end && n_valid < LZG_WALK_ENGINE_MAX_VALID_EDGES; e++) {
        if (frame && is_blacklisted(frame, e)) continue;
        if (!lzg_wd_edge_valid(g, e, wd)) continue;
        if (!edge_passes_filter(cfg, g, e)) continue;

        out_edges[n_valid] = e;
        out_wts[n_valid] = g->edge_weights[e];
        n_valid++;
    }

    return n_valid;
}

static uint32_t sample_edge_index(LZGRng *rng, const double *weights, uint32_t n_weights) {
    double Z = 0.0;
    for (uint32_t i = 0; i < n_weights; i++) Z += weights[i];

    double r = lzg_rng_double(rng) * Z;
    double cum = 0.0;
    uint32_t chosen = n_weights - 1;
    for (uint32_t i = 0; i < n_weights; i++) {
        cum += weights[i];
        if (r < cum) {
            chosen = i;
            break;
        }
    }

    return chosen;
}

static double valid_weight_sum(const double *weights, uint32_t n_weights) {
    double total = 0.0;
    for (uint32_t i = 0; i < n_weights; i++)
        total += weights[i];
    return total;
}

static uint8_t append_node_token(const LZGGraph *g, uint32_t node,
                                 char *seq_buf, uint32_t *seq_pos) {
    const char *sp = lzg_sp_get(g->pool, g->node_sp_id[node]);
    uint8_t sp_len = g->node_sp_len[node];
    const char *copy_src = sp;
    uint8_t copy_len = sp_len;

    if (sp_len > 0 && sp[sp_len - 1] == LZG_END_SENTINEL) {
        copy_len = sp_len - 1;
    }
    if (sp_len > 0 && sp[0] == LZG_START_SENTINEL) {
        copy_src = sp + 1;
        copy_len = sp_len - 1;
    }

    if (*seq_pos + copy_len < LZG_WALK_ENGINE_SEQ_BUF_CAP) {
        memcpy(seq_buf + *seq_pos, copy_src, copy_len);
        *seq_pos += copy_len;
    }

    return copy_len;
}

static void rebuild_walk_dict(LZGWalkDict *wd, const LZGGraph *g,
                              const LZGWalkEngineFrame *stack, uint32_t depth) {
    lzg_wd_reset(wd);
    lzg_wd_record_node(wd, g, stack[0].node);
    for (uint32_t d = 1; d < depth; d++) {
        lzg_wd_record_edge(wd, g, stack[d].edge_taken);
    }
}

bool lzg_walk_engine_run(const LZGGraph *g, LZGRng *rng,
                         const LZGWalkEngineConfig *cfg,
                         LZGWalkEngineResult *out) {
    if (!g || !rng || !out) return false;

    out->sequence[0] = '\0';
    out->seq_len = 0;
    out->n_tokens = 0;
    out->depth = 0;
    out->log_prob = 0.0;
    out->outcome = LZG_WALK_ENGINE_OUTCOME_FAILED;

    if (g->root_node >= g->n_nodes) return false;

    LZGWalkDict wd = lzg_wd_create();
    LZGWalkEngineFrame stack[LZG_WALK_ENGINE_MAX_DEPTH];
    uint32_t valid_edges[LZG_WALK_ENGINE_MAX_VALID_EDGES];
    double valid_wts[LZG_WALK_ENGINE_MAX_VALID_EDGES];
    uint32_t depth = 1;
    uint32_t seq_pos = 0;
    double log_prob = 0.0;
    bool success = true;

    lzg_wd_record_node(&wd, g, g->root_node);
    stack[0].node = g->root_node;
    stack[0].edge_taken = UINT32_MAX;
    stack[0].sp_len = 0;
    stack[0].log_edge_prob = 0.0;
    stack[0].n_blacklisted = 0;

    while (depth > 0 && depth < LZG_WALK_ENGINE_MAX_DEPTH) {
        LZGWalkEngineFrame *top = &stack[depth - 1];
        uint32_t current = top->node;

        if (is_sink_node(g, current)) {
            out->outcome = LZG_WALK_ENGINE_OUTCOME_ABSORBED;
            break;
        }

        uint32_t n_valid = collect_valid_edges(g, &wd, current, top, cfg,
                                               valid_edges, valid_wts);
        if (n_valid == 0) {
            if (cfg && cfg->stop_on_dead_end) {
                out->outcome = LZG_WALK_ENGINE_OUTCOME_LEAKED;
                break;
            }

            if (depth <= 1) {
                success = false;
                break;
            }

            seq_pos -= top->sp_len;
            if (top->edge_taken != UINT32_MAX)
                log_prob -= top->log_edge_prob;
            uint32_t dead_edge = top->edge_taken;
            depth--;

            rebuild_walk_dict(&wd, g, stack, depth);

            LZGWalkEngineFrame *parent = &stack[depth - 1];
            if (parent->n_blacklisted < LZG_WALK_ENGINE_MAX_BLACKLIST) {
                parent->blacklist[parent->n_blacklisted++] = dead_edge;
            }
            continue;
        }

        double total_w = valid_weight_sum(valid_wts, n_valid);
        uint32_t chosen = sample_edge_index(rng, valid_wts, n_valid);
        uint32_t edge = valid_edges[chosen];
        uint32_t next_node = g->col_indices[edge];
        uint8_t copy_len = append_node_token(g, next_node, out->sequence, &seq_pos);
        double edge_log_prob = log(valid_wts[chosen] / total_w);

        lzg_wd_record_edge(&wd, g, edge);
        log_prob += edge_log_prob;

        if (depth < LZG_WALK_ENGINE_MAX_DEPTH) {
            stack[depth].node = next_node;
            stack[depth].edge_taken = edge;
            stack[depth].sp_len = copy_len;
            stack[depth].log_edge_prob = edge_log_prob;
            stack[depth].n_blacklisted = 0;
            depth++;
        }
    }

    out->sequence[seq_pos] = '\0';
    out->seq_len = seq_pos;
    out->n_tokens = depth > 0 ? depth - 1u : 0u;
    out->depth = depth;
    out->log_prob = log_prob;

    lzg_wd_destroy(&wd);
    return success;
}
