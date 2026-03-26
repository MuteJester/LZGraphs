# LZGraphs C Reimplementation: Algorithm Architecture and API Design

## Table of Contents

1. [Unified Forward Propagation Framework](#1-unified-forward-propagation-framework)
2. [Module Organization](#2-module-organization)
3. [The Public C API](#3-the-public-c-api)
4. [Graph Variants (AAP vs NDP)](#4-graph-variants-aap-vs-ndp)
5. [Python Binding Strategy](#5-python-binding-strategy)
6. [Appendix: Full API Reference](#appendix-full-api-reference)

---

## 1. Unified Forward Propagation Framework

### 1.1 The Problem

The Python codebase contains at least seven distinct forward propagation implementations that all traverse the DAG in topological order, propagating different state through edges:

| Method | State propagated | Terminal action |
|--------|-----------------|-----------------|
| `simulation_potential_size()` | `int64` path count | Sum into total |
| `pgen_diagnostics()` | `float64` mass (m0) | Absorb at stop, leak at dead-ends |
| `effective_diversity()` | `(m0, m1)` mass + weighted log-prob | Absorb `stop * (m1 + ls*m0)` |
| `lzpgen_moments()` | `(m0..m4)` five moment accumulators | Binomial expansion into T0..T4 |
| `lzpgen_analytical_distribution()` | `(m0..m4, depth)` moments + depth | Per-length + global terminals |
| `pgen_dynamic_range()` | `(min_cost, max_cost, backpointers)` | Track best/worst complete paths |
| `_raw_power_sum(alpha)` | `float64` mass with `w^alpha` edges | Sum at terminals |

All share the same structure:
1. Seed initial states
2. For each node in topological order: handle terminal absorption, then propagate to successors

### 1.2 Generic Forward Propagation Engine

The core engine is parameterized by a `LZFwdOps` struct of function pointers (callbacks). The engine owns the traversal logic; the callbacks own the domain-specific math.

```c
/*
 * Generic state: opaque blob of `state_size` bytes per node.
 * The engine allocates a flat array of N * state_size bytes.
 */

typedef struct {
    size_t state_size;  /* bytes per node state */

    /* Zero-initialize a state slot */
    void (*state_zero)(void *state);

    /* Seed: write initial state for node `idx` given probability `p`.
     * Called once per initial state. May be called multiple times
     * for the same idx (accumulate). */
    void (*seed)(void *state, double p, void *user);

    /* Test whether a node has any mass to propagate.
     * Returns 0 if the node can be skipped entirely. */
    int (*has_mass)(const void *state);

    /* Terminal absorption: given the current node state and stop_prob,
     * accumulate into the terminal accumulator `term_acc`.
     * Returns the continue_factor (1.0 - stop_prob, or 1.0 if not terminal). */
    double (*absorb_terminal)(const void *state, double stop_prob,
                              void *term_acc, void *user);

    /* Apply continue factor: modify state in-place to account for
     * log(1 - stop_prob) being added to the accumulated log-probability.
     * Only called when continue_factor < 1.0. */
    void (*apply_continue)(void *state, double continue_factor, void *user);

    /* Propagate from source to destination through an edge of weight `w`.
     * `continue_factor` is the factor already applied to source state.
     * Accumulates into `dst` (which may already have mass from other parents). */
    void (*propagate_edge)(const void *src, void *dst,
                           double w, double continue_factor, void *user);

    /* Finalize: called once after the full sweep. Reads term_acc,
     * writes output to `result`. Returns 0 on success. */
    int (*finalize)(const void *term_acc, void *result, void *user);

} LZFwdOps;

/* The engine itself */
int lzg_forward_propagate(
    const LZGraph        *g,
    const LZFwdOps       *ops,
    void                 *term_acc,      /* terminal accumulator, caller-allocated */
    void                 *result,        /* output, caller-allocated */
    void                 *user           /* arbitrary user data for callbacks */
);
```

### 1.3 Concrete Instantiations

Each analytical method becomes a thin wrapper that sets up `LZFwdOps` + allocates the right-sized state:

**Path counting (simulation_potential_size)**
- `state_size = sizeof(int64_t)`
- `seed`: `*(int64_t*)s = 1` for initial states
- `propagate_edge`: `*(int64_t*)dst += *(int64_t*)src` (weight ignored)
- `absorb_terminal`: accumulate into `int64_t` total

**Moment propagation (lzpgen_moments)**
- `state_size = 5 * sizeof(double)` (m0..m4)
- `seed`: binomial expansion of `p * log(p)^k`
- `propagate_edge`: binomial expansion with `log(w)`
- `absorb_terminal`: binomial expansion with `log(stop_prob)` into T0..T4

**Power sum (Hill numbers)**
- `state_size = sizeof(double)`
- `seed`: `s += p^alpha`
- `propagate_edge`: `dst += src * w^alpha * continue_factor^alpha`
- `absorb_terminal`: `total += state * stop_prob^alpha`

**Dynamic range (min/max paths)**
- `state_size = 2 * sizeof(double) + 2 * sizeof(int32_t)` (min_cost, max_cost, min_pred, max_pred)
- `propagate_edge`: relax min/max with `-log(w)`

### 1.4 LZ76-Constrained Forward Propagation

The standard forward propagation ignores LZ76 dictionary constraints (it counts all topological paths, not just LZ-valid ones). This is correct for `simulation_potential_size` (upper bound), `pgen_diagnostics`, `effective_diversity`, and `lzpgen_moments` because the graph's edge weights already encode the LZ-constrained transition probabilities.

The LZ constraint matters for **simulation** and **lzpgen_distribution** (Monte Carlo), which are random-walk-based, not DP-based. These remain separate from the forward propagation engine and use the walk cache / CSR representation.

### 1.5 Topological Order

The engine requires a pre-computed topological order. Store it as a flat `int32_t[]` array in the graph struct, computed once and cached:

```c
/* Kahn's algorithm, O(V + E), no recursion */
int lzg_compute_topo_order(LZGraph *g);
```

---

## 2. Module Organization

### 2.1 Module Dependency Graph

```
                    lzg_core.h          (graph struct, edge data, error codes)
                       |
          +------------+------------+
          |            |            |
    lzg_construct   lzg_topo     lzg_lz76
    (build graph)  (topo sort)  (LZ decomposition)
          |            |
          +-----+------+
                |
           lzg_forward        (generic forward propagation engine)
                |
     +----------+----------+----------+
     |          |          |          |
  lzg_prob   lzg_analytics  lzg_occupancy  lzg_posterior
  (walk_prob) (moments,     (richness,     (Bayesian
   simulate)   Hill, etc.)   overlap)       update)
                                    |
                              lzg_pgen_dist
                              (Gaussian mixture,
                               saddlepoint)
```

### 2.2 Module Specifications

#### `lzg_core` -- Graph Structure and Memory

**Inputs**: Raw edge/node data from construction.
**Outputs**: The opaque `LZGraph` handle.
**Dependencies**: None (leaf module).

```c
/* --- Error handling --- */
typedef enum {
    LZG_OK = 0,
    LZG_ERR_ALLOC,          /* malloc failed */
    LZG_ERR_CYCLE,          /* graph has cycles, topo sort impossible */
    LZG_ERR_EMPTY,          /* no sequences / empty graph */
    LZG_ERR_NO_INITIAL,     /* no initial states */
    LZG_ERR_NO_TERMINAL,    /* no terminal mass absorbed */
    LZG_ERR_MISSING_NODE,   /* node not found */
    LZG_ERR_MISSING_EDGE,   /* edge not found */
    LZG_ERR_NO_GENE_DATA,   /* gene operation on non-genetic graph */
    LZG_ERR_INVALID_ARG,    /* bad parameter */
} LZGError;

const char *lzg_error_string(LZGError err);

/* --- Edge data --- */
typedef struct {
    double   weight;          /* P(B|A), normalized */
    uint32_t count;           /* raw traversal count */
    /* Gene data: stored as parallel arrays for cache efficiency */
    uint16_t n_v_genes;
    uint16_t n_j_genes;
    uint32_t *v_gene_ids;     /* indices into graph's gene string table */
    uint32_t *v_gene_counts;
    uint32_t *j_gene_ids;
    uint32_t *j_gene_counts;
    uint32_t v_sum;
    uint32_t j_sum;
} LZGEdge;

/* --- Graph variant --- */
typedef enum {
    LZG_VARIANT_AAP,   /* Amino Acid Positional */
    LZG_VARIANT_NDP,   /* Nucleotide Double Positional */
} LZGVariant;

/* --- The graph handle (opaque to callers) --- */
typedef struct LZGraph LZGraph;

/* Lifecycle */
LZGraph    *lzg_create(LZGVariant variant);
void        lzg_destroy(LZGraph *g);

/* Accessors (read-only view into internals) */
uint32_t    lzg_num_nodes(const LZGraph *g);
uint32_t    lzg_num_edges(const LZGraph *g);
int         lzg_has_gene_data(const LZGraph *g);
LZGVariant  lzg_variant(const LZGraph *g);
const char *lzg_node_label(const LZGraph *g, uint32_t node_id);
const char *lzg_node_subpattern(const LZGraph *g, uint32_t node_id);
```

**Internal layout of `struct LZGraph`** (in `lzg_core_internal.h`, not exposed):

```c
struct LZGraph {
    LZGVariant variant;
    int        has_genes;

    /* --- Node storage --- */
    uint32_t   n_nodes;
    uint32_t   n_edges;
    char     **node_labels;       /* "ABC_5" style, indexed by node_id */
    char     **node_subpatterns;  /* "ABC" extracted, indexed by node_id */

    /* --- CSR adjacency --- */
    uint32_t  *csr_offsets;       /* [n_nodes + 1] */
    uint32_t  *csr_neighbors;     /* [n_edges] */
    LZGEdge   *csr_edges;         /* [n_edges], parallel to csr_neighbors */

    /* --- Cumulative weight arrays (for simulation) --- */
    double    *csr_cumweights;    /* [n_edges] */

    /* --- Node metadata --- */
    double    *initial_probs;     /* [n_initial_states] */
    uint32_t  *initial_ids;       /* [n_initial_states] */
    double    *initial_cumprobs;  /* [n_initial_states] */
    uint32_t   n_initial_states;

    double    *stop_probs;        /* [n_nodes], NAN if not terminal */
    uint32_t  *terminal_counts;   /* [n_nodes], 0 if not terminal */
    uint32_t  *outgoing_counts;   /* [n_nodes] */

    double    *node_probs;        /* [n_nodes], P(node) */

    /* --- Topological order (computed lazily) --- */
    uint32_t  *topo_order;        /* [n_nodes] or NULL */
    int        topo_valid;

    /* --- Gene string table (interned) --- */
    uint32_t   n_gene_strings;
    char     **gene_strings;      /* gene name lookup by ID */
    /* hash map: gene_name -> gene_id (for construction) */

    /* --- Walk cache (for simulation, built lazily) --- */
    /* LZ76 constraint precomputation per successor */
    uint8_t   *succ_is_single;    /* [n_edges] bool */
    uint32_t  *succ_prefix_hash;  /* [n_edges] for fast set lookup */

    /* --- Configuration --- */
    double     smoothing_alpha;
    uint32_t   min_initial_state_count;
    int        impute_missing_edges;
};
```

The CSR (Compressed Sparse Row) representation is chosen because:
- Cache-friendly sequential access during topological traversal
- O(1) successor lookup by node ID
- Compatible with the existing C simulation extension
- Parallel to cumulative weight arrays for O(log k) bisection sampling

#### `lzg_lz76` -- LZ76 Decomposition

**Inputs**: Raw sequence string (amino acid or nucleotide).
**Outputs**: Array of subpattern strings + positional metadata.
**Dependencies**: None.

```c
typedef struct {
    uint32_t    n_tokens;
    const char **tokens;       /* subpattern strings */
    uint32_t   *positions;     /* cumulative end positions */
    uint32_t   *frames;        /* reading frames (NDP only, NULL for AAP) */
} LZDecomposition;

LZGError lzg_lz76_decompose(const char *sequence, uint32_t len,
                             LZDecomposition *out);
void     lzg_lz76_free(LZDecomposition *d);
```

#### `lzg_construct` -- Graph Construction

**Inputs**: Arrays of sequences, optional V/J genes, optional abundances.
**Outputs**: Populated `LZGraph *`.
**Dependencies**: `lzg_core`, `lzg_lz76`.

```c
typedef struct {
    const char **sequences;    /* array of sequence strings */
    uint32_t     n_sequences;
    const uint32_t *abundances; /* NULL for uniform count=1 */
    const char **v_genes;       /* NULL if no gene data */
    const char **j_genes;       /* NULL if no gene data */
} LZGBuildInput;

LZGError lzg_build(LZGraph *g, const LZGBuildInput *input);

/* Incremental: encode a single sequence into node labels */
LZGError lzg_encode_sequence(const LZGraph *g, const char *seq,
                             uint32_t *out_node_ids, uint32_t *out_n);
```

#### `lzg_topo` -- Topological Sort and DAG Utilities

**Dependencies**: `lzg_core`.

```c
LZGError lzg_compute_topo_order(LZGraph *g);
int      lzg_is_dag(const LZGraph *g);
```

#### `lzg_forward` -- Generic Forward Propagation Engine

**Dependencies**: `lzg_core`, `lzg_topo`.

```c
/* (LZFwdOps defined in section 1.2) */
LZGError lzg_forward_propagate(
    const LZGraph   *g,
    const LZFwdOps  *ops,
    void            *term_acc,
    void            *result,
    void            *user
);
```

#### `lzg_prob` -- Walk Probability and Simulation

**Dependencies**: `lzg_core`, `lzg_lz76`.

```c
/* Walk probability */
double   lzg_walk_log_probability(const LZGraph *g,
                                  const uint32_t *walk, uint32_t walk_len);
double   lzg_sequence_log_probability(const LZGraph *g, const char *seq);

/* Batch probability */
LZGError lzg_batch_log_probability(const LZGraph *g,
                                   const char **sequences, uint32_t n,
                                   double *out_log_probs);

/* Simulation */
LZGError lzg_simulate(const LZGraph *g, uint32_t n, uint64_t seed,
                       char **out_sequences);
LZGError lzg_simulate_with_walks(const LZGraph *g, uint32_t n, uint64_t seed,
                                  char **out_sequences,
                                  uint32_t **out_walks, uint32_t *out_walk_lens);
void     lzg_free_strings(char **strings, uint32_t n);
void     lzg_free_walks(uint32_t **walks, uint32_t n);
```

#### `lzg_analytics` -- Analytical Distribution Methods

**Dependencies**: `lzg_core`, `lzg_forward`.

```c
/* Exact moments */
typedef struct {
    double mean, variance, std, skewness, kurtosis, total_mass;
} LZGMoments;

LZGError lzg_lzpgen_moments(const LZGraph *g, LZGMoments *out);

/* Effective diversity */
typedef struct {
    double entropy_nats, entropy_bits, effective_diversity;
    int64_t simulation_potential_size;
    double uniformity;
} LZGDiversity;

LZGError lzg_effective_diversity(const LZGraph *g, LZGDiversity *out);

/* Hill numbers */
LZGError lzg_hill_numbers(const LZGraph *g,
                          const double *orders, uint32_t n_orders,
                          double *out_hills);

/* Dynamic range */
typedef struct {
    double max_log_prob, min_log_prob;
    double dynamic_range_nats, dynamic_range_oom;
    char  *most_probable_seq;
    char  *least_probable_seq;
    uint32_t *most_probable_walk;
    uint32_t  most_probable_walk_len;
    uint32_t *least_probable_walk;
    uint32_t  least_probable_walk_len;
} LZGDynamicRange;

LZGError lzg_pgen_dynamic_range(const LZGraph *g, LZGDynamicRange *out);
void     lzg_dynamic_range_free(LZGDynamicRange *dr);

/* PGEN diagnostics */
typedef struct {
    double initial_prob_sum, total_absorbed, total_leaked;
    double max_edge_weight_deviation;
    uint32_t num_dead_end_non_terminals;
    int is_proper;
} LZGPgenDiag;

LZGError lzg_pgen_diagnostics(const LZGraph *g, double atol, LZGPgenDiag *out);

/* Monte Carlo PGEN distribution */
LZGError lzg_lzpgen_distribution(const LZGraph *g, uint32_t n, uint64_t seed,
                                  double *out_log_probs);
```

#### `lzg_pgen_dist` -- Gaussian Mixture Distribution Object

**Dependencies**: `lzg_forward` (for construction), standalone for evaluation.

```c
typedef struct {
    uint32_t n_components;
    double  *weights;         /* [n_components] */
    double  *means;           /* [n_components] */
    double  *stds;            /* [n_components] */
    int32_t *walk_lengths;    /* [n_components] */
    /* Cumulants */
    double kappa_1, kappa_2, kappa_3, kappa_4;
    double total_mass;
} LZGPgenDist;

LZGError lzg_analytical_distribution(const LZGraph *g, LZGPgenDist *out);
void     lzg_pgen_dist_free(LZGPgenDist *d);

/* Evaluation (pure math, no graph needed) */
LZGError lzg_pgen_dist_pdf(const LZGPgenDist *d,
                           const double *x, uint32_t n, double *out);
LZGError lzg_pgen_dist_cdf(const LZGPgenDist *d,
                           const double *x, uint32_t n, double *out);
LZGError lzg_pgen_dist_ppf(const LZGPgenDist *d,
                           const double *q, uint32_t n, double *out);
LZGError lzg_pgen_dist_rvs(const LZGPgenDist *d,
                           uint32_t n, uint64_t seed, double *out);
```

#### `lzg_occupancy` -- Richness, Overlap, and Sharing Spectrum

**Dependencies**: `lzg_analytics`, `lzg_pgen_dist`.

```c
/* Predicted richness */
LZGError lzg_predicted_richness(const LZGraph *g, double d,
                                const char *model, /* "poisson" or "gamma" */
                                double alpha,       /* gamma shape; ignored for poisson */
                                double *out);

/* Richness curve (vectorized) */
LZGError lzg_richness_curve(const LZGraph *g,
                            const double *d_values, uint32_t n,
                            const char *model, double alpha,
                            double *out);

/* Predicted overlap */
LZGError lzg_predicted_overlap(const LZGraph *g,
                               double d_i, double d_j,
                               const char *model, double alpha,
                               double *out);

/* Sharing spectrum */
typedef struct {
    double  *spectrum;          /* [max_k] */
    uint32_t max_k;
    double   expected_total_unique;
    uint32_t n_donors;
    double   total_draws;
} LZGSharingSpectrum;

LZGError lzg_predict_sharing_spectrum(
    const LZGraph *g,
    const double *draw_counts, uint32_t n_donors,
    uint32_t max_k, uint32_t n_quadrature,
    LZGSharingSpectrum *out);
void     lzg_sharing_spectrum_free(LZGSharingSpectrum *s);
```

#### `lzg_posterior` -- Bayesian Posterior Update

**Dependencies**: `lzg_core`, `lzg_construct`, `lzg_prob`.

```c
/* Full posterior graph */
LZGError lzg_get_posterior(const LZGraph *prior,
                           const char **sequences, uint32_t n_seq,
                           const uint32_t *abundances, /* NULL for uniform */
                           double kappa,
                           LZGraph **out_posterior);

/* Fast posterior simulation (no graph clone) */
LZGError lzg_simulate_posterior(
    const LZGraph *prior,
    const char **sequences, uint32_t n_seq,
    const uint32_t *abundances,
    double kappa, uint32_t n_sim, uint64_t seed,
    char **out_sequences);
```

#### `lzg_serial` -- Serialization

**Dependencies**: `lzg_core`.

```c
/* Binary format (custom, fast) */
LZGError lzg_save_binary(const LZGraph *g, const char *path);
LZGError lzg_load_binary(const char *path, LZGraph **out);

/* JSON (interop) */
LZGError lzg_save_json(const LZGraph *g, const char *path);
LZGError lzg_load_json(const char *path, LZGraph **out);
```

#### `lzg_genes` -- Gene Logic (V/J Selection, Prediction)

**Dependencies**: `lzg_core`.

```c
/* Random V/J selection */
LZGError lzg_select_random_vj(const LZGraph *g, const char *mode,
                               uint64_t *rng_state,
                               uint32_t *out_v_id, uint32_t *out_j_id);

/* Gene prediction from a walk */
typedef struct {
    uint32_t *v_gene_ids;   /* sorted by descending score */
    double   *v_scores;
    uint32_t  n_v;
    uint32_t *j_gene_ids;
    double   *j_scores;
    uint32_t  n_j;
} LZGGenePrediction;

LZGError lzg_predict_vj(const LZGraph *g,
                         const uint32_t *walk, uint32_t walk_len,
                         const char *mode, /* "max_sum", "max_product", "full" */
                         uint32_t top_n,
                         LZGGenePrediction *out);
void     lzg_gene_prediction_free(LZGGenePrediction *p);
```

---

## 3. The Public C API

### 3.1 Handle Design: Opaque with Accessors

The `LZGraph` is an **opaque handle** (`typedef struct LZGraph LZGraph`). Callers never see the internal layout. Rationale:

- **ABI stability**: Internal struct layout can change without breaking compiled bindings.
- **Safety**: Python bindings cannot accidentally corrupt internal state.
- **Encapsulation**: Forces all mutation through validated API functions.
- **Thread safety**: Enables future internal locking without API changes.

Read-only accessors (`lzg_num_nodes`, `lzg_node_label`, etc.) provide efficient zero-copy access to internal data.

### 3.2 Error Handling: Return Codes + Thread-Local Detail

Every function returns `LZGError`. For functions that also return a value (like `lzg_walk_log_probability`), the result is written to a caller-provided pointer, and the return value is the error code.

```c
LZGError lzg_walk_log_probability(const LZGraph *g,
                                  const uint32_t *walk, uint32_t walk_len,
                                  double *out_log_prob);
```

For detailed error messages:

```c
/* Thread-local error detail buffer */
const char *lzg_last_error_detail(void);
```

This avoids the errno anti-pattern (global mutable state) while keeping the API simple. The Python binding checks the return code and raises a typed Python exception with the detail string.

### 3.3 Memory Ownership Convention

**Principle**: The caller who allocated the memory is responsible for freeing it.

- **Caller-allocated outputs**: For fixed-size outputs (`LZGMoments`, `double *out_log_probs`), the caller allocates and passes a pointer. The library writes into it. No free needed.

- **Library-allocated outputs**: For variable-size outputs (`char **out_sequences`, `LZGDynamicRange`), the library allocates. Each such type has a corresponding free function (`lzg_free_strings`, `lzg_dynamic_range_free`).

- **Graph ownership**: `lzg_create` allocates, `lzg_destroy` frees. `lzg_get_posterior` allocates a new graph.

### 3.4 Thread Safety

The API is designed for the following concurrency model:

- **Read-only operations are thread-safe**: Multiple threads can call `lzg_walk_log_probability`, `lzg_hill_numbers`, etc. on the same `LZGraph *` concurrently. All these functions take `const LZGraph *`.

- **Mutation requires exclusive access**: `lzg_build`, `lzg_compute_topo_order`, and `lzg_get_posterior` mutate the graph and require exclusive (single-writer) access.

- **Simulation is thread-safe with separate seeds**: Each `lzg_simulate` call takes its own seed and uses a thread-local xoshiro256++ state. No shared mutable state.

The lazy topo_order cache is the one subtle point. The internal implementation uses a simple atomic flag + mutex for the one-time computation:

```c
/* Internal: thread-safe lazy topo order */
if (!atomic_load(&g->topo_valid)) {
    pthread_mutex_lock(&g->topo_mutex);
    if (!g->topo_valid) {
        lzg_compute_topo_order_internal(g);
        atomic_store(&g->topo_valid, 1);
    }
    pthread_mutex_unlock(&g->topo_mutex);
}
```

---

## 4. Graph Variants (AAP vs NDP)

### 4.1 What Differs

The two graph variants differ in exactly two static methods:

| Operation | AAPLZGraph | NDPLZGraph |
|-----------|-----------|-----------|
| `encode_sequence(seq)` | LZ76 decomposition -> `"{token}_{cumulative_pos}"` | LZ76 decomposition -> `"{token}{frame%3}_{cumulative_pos}"` |
| `extract_subpattern(label)` | `label[:label.rfind('_')]` | `label.split('_',1)[0][:-1]` |

Everything else (construction, traversal, analytics, simulation) is identical.

### 4.2 Design: Variant Enum + Function Pointers in the Graph Struct

The variant is stored as an enum in the graph struct. Two function pointers handle the variant-specific logic:

```c
/* In lzg_core_internal.h */
struct LZGraph {
    LZGVariant variant;

    /* Variant-specific function pointers, set once at creation */
    LZGError (*encode_fn)(const char *seq, uint32_t len,
                          char **out_labels, uint32_t *out_n);
    void     (*extract_fn)(const char *label, char *out_subpattern,
                           uint32_t *out_len);
    /* ... rest of struct ... */
};
```

At `lzg_create(variant)`:

```c
LZGraph *lzg_create(LZGVariant variant) {
    LZGraph *g = calloc(1, sizeof(LZGraph));
    g->variant = variant;
    switch (variant) {
        case LZG_VARIANT_AAP:
            g->encode_fn  = lzg_encode_aap;
            g->extract_fn = lzg_extract_aap;
            break;
        case LZG_VARIANT_NDP:
            g->encode_fn  = lzg_encode_ndp;
            g->extract_fn = lzg_extract_ndp;
            break;
    }
    return g;
}
```

**Why function pointers over alternatives:**

- *vs. preprocessor (`#ifdef`)* -- Would require compiling two separate libraries. Unacceptable for a single shared library that serves both graph types.

- *vs. switch statements everywhere* -- Scatters variant logic across every call site. The function pointer approach isolates the variant-specific code to exactly two functions, called indirectly through a uniform interface.

- *vs. C++ virtual dispatch* -- This is plain C. Function pointers are the idiomatic equivalent and carry zero overhead beyond the single indirect call.

### 4.3 The Encoding Functions

```c
/* AAP: "CASSLGIR" -> ["C_1", "A_2", "SS_4", "L_5", "G_6", "I_7", "RR_9"] */
static LZGError lzg_encode_aap(const char *seq, uint32_t len,
                                char **out_labels, uint32_t *out_n) {
    LZDecomposition d;
    lzg_lz76_decompose(seq, len, &d);
    for (uint32_t i = 0; i < d.n_tokens; i++) {
        /* label = "{token}_{position}" */
        snprintf(out_labels[i], ...);
    }
    *out_n = d.n_tokens;
    lzg_lz76_free(&d);
    return LZG_OK;
}

/* NDP: "ATGCG" -> ["ATG0_3", "CG0_5"] */
static LZGError lzg_encode_ndp(const char *seq, uint32_t len,
                                char **out_labels, uint32_t *out_n) {
    LZDecomposition d;
    lzg_lz76_decompose(seq, len, &d);
    /* d.frames is populated for NDP */
    for (uint32_t i = 0; i < d.n_tokens; i++) {
        /* label = "{token}{frame}_{position}" */
        snprintf(out_labels[i], ...);
    }
    *out_n = d.n_tokens;
    lzg_lz76_free(&d);
    return LZG_OK;
}
```

---

## 5. Python Binding Strategy

### 5.1 Recommendation: CPython C Extension

**Selected approach: CPython C extension** (direct `PyObject *` types, like the existing `_fast_walk.c`).

Rationale:

| Criterion | CPython ext | ctypes | pybind11 | Cython |
|-----------|------------|--------|----------|--------|
| **Build dependency** | None (uses Python.h) | None | pybind11 C++ headers | Cython transpiler |
| **Compilation** | Standard `setup.py` | No compilation | Needs C++ compiler | Needs Cython + C |
| **NumPy integration** | Excellent (buffer protocol) | Manual | Good (auto conversion) | Excellent |
| **Callback overhead** | Zero (direct C calls) | High (per-call FFI) | Low | Low |
| **Reference counting** | Manual (error-prone but controllable) | Automatic | Automatic | Automatic |
| **Existing precedent** | Yes (`_fast_walk.c` already uses this) | No | No | No |
| **Package size** | Minimal | Minimal | +200KB headers | +Cython dep |
| **PyPy compat** | Limited | Good | No | Limited |

Key factors driving the choice:

1. **Consistency**: The project already has `_fast_walk.c` as a CPython C extension. Using the same approach means one build system, one compilation model, one set of patterns.

2. **Zero overhead for hot paths**: Simulation and batch probability computation are the performance-critical paths. CPython extensions call directly into C with no FFI marshaling overhead.

3. **No additional dependencies**: pybind11 requires C++; Cython requires the Cython transpiler. CPython extensions only need `Python.h` which is always available.

4. **Fine-grained memory control**: The LZGraph handle wraps a C-allocated struct. CPython's `tp_dealloc` slot gives precise control over when `lzg_destroy` is called.

### 5.2 Python Type Structure

```c
/* --- The Python wrapper type --- */
typedef struct {
    PyObject_HEAD
    LZGraph *graph;          /* owned; freed in tp_dealloc */
} PyLZGraph;

static PyTypeObject PyLZGraph_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "lzgraphs._core.LZGraph",
    .tp_basicsize = sizeof(PyLZGraph),
    .tp_dealloc   = (destructor)PyLZGraph_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_methods   = PyLZGraph_methods,
    .tp_getset    = PyLZGraph_getsetters,
    .tp_new       = PyLZGraph_new,
    .tp_init      = PyLZGraph_init,
};
```

### 5.3 Python API Surface

The Python binding module (`_core.pyd` / `_core.so`) exposes a single `LZGraph` type. The existing pure-Python `AAPLZGraph` and `NDPLZGraph` classes become thin wrappers:

```python
class AAPLZGraph:
    """Amino Acid Positional LZGraph."""

    def __init__(self, data, *, abundances=None, v_genes=None, j_genes=None,
                 verbose=True, smoothing_alpha=0.0, min_initial_state_count=5):
        # Normalize input (pure Python, handles DataFrame/Series/list)
        normalized = self._normalize_input(data, ...)

        # Delegate to C
        self._graph = _core.LZGraph(
            variant="aap",
            sequences=normalized['sequences'],
            abundances=normalized['abundances'],
            v_genes=normalized['v_genes'],
            j_genes=normalized['j_genes'],
            smoothing_alpha=smoothing_alpha,
            min_initial_state_count=min_initial_state_count,
        )

    def simulate(self, n, seed=None, return_walks=False):
        return self._graph.simulate(n, seed, return_walks)

    def walk_log_probability(self, sequence):
        return self._graph.walk_log_probability(sequence)

    def lzpgen_moments(self):
        return self._graph.lzpgen_moments()

    # ... etc
```

### 5.4 NumPy Zero-Copy Returns

For bulk operations (simulate, batch_log_probability, lzpgen_distribution), the C extension returns NumPy arrays using the buffer protocol. The C code allocates the array, wraps it as a NumPy array with a custom destructor capsule, and returns it. No data copying.

```c
static PyObject *
PyLZGraph_lzpgen_distribution(PyLZGraph *self, PyObject *args) {
    uint32_t n;
    uint64_t seed;
    if (!PyArg_ParseTuple(args, "IK", &n, &seed))
        return NULL;

    npy_intp dims[1] = { (npy_intp)n };
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!arr) return NULL;

    double *data = (double *)PyArray_DATA((PyArrayObject *)arr);
    LZGError err = lzg_lzpgen_distribution(self->graph, n, seed, data);
    if (err != LZG_OK) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_RuntimeError, lzg_error_string(err));
        return NULL;
    }
    return arr;
}
```

### 5.5 Error Mapping

```c
static int raise_lzg_error(LZGError err) {
    switch (err) {
        case LZG_OK: return 0;
        case LZG_ERR_ALLOC:
            PyErr_NoMemory(); break;
        case LZG_ERR_CYCLE:
            PyErr_SetString(PyExc_RuntimeError,
                "Graph contains cycles; this operation requires a DAG.");
            break;
        case LZG_ERR_EMPTY:
            /* Map to LZGraphs' EmptyDataError */
            PyErr_SetString(EmptyDataError, lzg_last_error_detail());
            break;
        case LZG_ERR_NO_GENE_DATA:
            PyErr_SetString(NoGeneDataError, lzg_last_error_detail());
            break;
        /* ... */
    }
    return -1;
}
```

### 5.6 Migration Path

The C library can be adopted incrementally:

1. **Phase 1**: Replace `_fast_walk.c` (simulation only) with calls to `lzg_simulate`. The Python classes keep their NetworkX graph for everything else.

2. **Phase 2**: Move construction to C. The Python class holds both a `_core.LZGraph` handle and a NetworkX graph (for compatibility with methods not yet ported).

3. **Phase 3**: Move analytical methods (moments, Hill numbers, dynamic range, occupancy) to C via the forward propagation engine.

4. **Phase 4**: Move probability computation, posterior, and remaining methods. Drop NetworkX dependency entirely.

At each phase, the Python API remains identical. The only visible change is speed.

---

## Appendix: Full API Reference

### Summary of All Public C Functions

```
Lifecycle:
  lzg_create(variant) -> LZGraph*
  lzg_destroy(LZGraph*)

Construction:
  lzg_build(g, input) -> err
  lzg_encode_sequence(g, seq, out_ids, out_n) -> err

Probability:
  lzg_walk_log_probability(g, walk, len, out) -> err
  lzg_sequence_log_probability(g, seq, out) -> err
  lzg_batch_log_probability(g, seqs, n, out) -> err

Simulation:
  lzg_simulate(g, n, seed, out_seqs) -> err
  lzg_simulate_with_walks(g, n, seed, out_seqs, out_walks, out_lens) -> err

Analytics:
  lzg_lzpgen_moments(g, out) -> err
  lzg_effective_diversity(g, out) -> err
  lzg_hill_numbers(g, orders, n_orders, out) -> err
  lzg_pgen_dynamic_range(g, out) -> err
  lzg_pgen_diagnostics(g, atol, out) -> err
  lzg_lzpgen_distribution(g, n, seed, out) -> err

Analytical Distribution:
  lzg_analytical_distribution(g, out) -> err
  lzg_pgen_dist_pdf(d, x, n, out) -> err
  lzg_pgen_dist_cdf(d, x, n, out) -> err
  lzg_pgen_dist_ppf(d, q, n, out) -> err
  lzg_pgen_dist_rvs(d, n, seed, out) -> err

Occupancy:
  lzg_predicted_richness(g, d, model, alpha, out) -> err
  lzg_richness_curve(g, d_vals, n, model, alpha, out) -> err
  lzg_predicted_overlap(g, di, dj, model, alpha, out) -> err
  lzg_predict_sharing_spectrum(g, draws, n_donors, max_k, nq, out) -> err

Posterior:
  lzg_get_posterior(prior, seqs, n, abundances, kappa, out) -> err
  lzg_simulate_posterior(prior, seqs, n, abundances, kappa, n_sim, seed, out) -> err

Genes:
  lzg_select_random_vj(g, mode, rng, out_v, out_j) -> err
  lzg_predict_vj(g, walk, len, mode, top_n, out) -> err

Topology:
  lzg_compute_topo_order(g) -> err
  lzg_is_dag(g) -> int
  lzg_simulation_potential_size(g, out) -> err

Serialization:
  lzg_save_binary(g, path) -> err
  lzg_load_binary(path, out) -> err
  lzg_save_json(g, path) -> err
  lzg_load_json(path, out) -> err

Accessors:
  lzg_num_nodes(g) -> uint32_t
  lzg_num_edges(g) -> uint32_t
  lzg_has_gene_data(g) -> int
  lzg_variant(g) -> LZGVariant
  lzg_node_label(g, id) -> const char*
  lzg_node_subpattern(g, id) -> const char*
  lzg_error_string(err) -> const char*
  lzg_last_error_detail() -> const char*

Memory:
  lzg_free_strings(strs, n)
  lzg_free_walks(walks, n)
  lzg_dynamic_range_free(dr)
  lzg_sharing_spectrum_free(s)
  lzg_gene_prediction_free(p)
  lzg_pgen_dist_free(d)
  lzg_lz76_free(d)
```

### Metrics Module (Standalone Functions, No Graph Handle)

The entropy/diversity metrics from `metrics/entropy.py` and `metrics/diversity.py` operate on graph-level summary statistics (node probabilities, edge weight distributions) rather than the graph structure itself. These are best kept as pure Python functions that read attributes from the Python wrapper, or as C functions taking arrays:

```c
/* Information-theoretic metrics on probability arrays */
double lzg_shannon_entropy(const double *probs, uint32_t n, double base);
double lzg_jensen_shannon_divergence(const double *p, const double *q,
                                      uint32_t n);
double lzg_kl_divergence(const double *p, const double *q, uint32_t n);

/* K-diversity requires sampling + encoding, delegates to lzg_build internally */
/* Best kept as a Python-level function that calls lzg_encode_sequence in a loop */
```
