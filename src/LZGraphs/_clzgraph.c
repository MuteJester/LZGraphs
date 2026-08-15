/**
 * @file _clzgraph.c
 * @brief CPython C extension wrapping the C-LZGraph library.
 *
 * All public C functions are exposed as module-level Python functions.
 * The LZGraph Python class (in _graph.py) calls these via PyCapsule.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/lz76.h"
#include "lzgraph/simulate.h"
#include "lzgraph/analytics.h"
#include "lzgraph/pgen_dist.h"
#include "lzgraph/occupancy.h"
#include "lzgraph/sharing.h"
#include "lzgraph/publicness.h"
#include "lzgraph/diversity.h"
#include "lzgraph/graph_ops.h"
#include "lzgraph/features.h"
#include "lzgraph/io.h"
#include "lzgraph/posterior.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/rng.h"
#include "lzgraph/flashback.h"
#include "lzgraph/flashback_graph.h"
#include "lzgraph/flashback_grammar.h"
#include "lzgraph/naive_graph.h"
#include "lzgraph/flat_flashback.h"

/* ── Custom exception pointers (loaded at module init) ────── */

static PyObject *LZGExc_NoGeneDataError = NULL;
static PyObject *LZGExc_ConvergenceError = NULL;
static PyObject *LZGExc_CorruptFileError = NULL;

/* ── Helpers ──────────────────────────────────────────────── */

static const char *CAPSULE_NAME = "LZGGraph";

static void capsule_destructor(PyObject *capsule) {
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (g) lzg_graph_destroy(g);
}

static int pyssize_to_u32(Py_ssize_t n, const char *what, uint32_t *out) {
    if (n < 0 || (unsigned long long)n > (unsigned long long)UINT32_MAX) {
        PyErr_Format(PyExc_OverflowError,
                     "%s length exceeds uint32 limit", what);
        return 0;
    }
    *out = (uint32_t)n;
    return 1;
}

static uint64_t *pylist_to_u64_array(PyObject *list,
                                     Py_ssize_t expected_n,
                                     const char *what) {
    if (!PyList_Check(list)) {
        PyErr_Format(PyExc_TypeError, "%s must be a list", what);
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(list);
    if (expected_n >= 0 && n != expected_n) {
        PyErr_Format(PyExc_ValueError, "%s length must match sequences", what);
        return NULL;
    }

    uint32_t n_u32 = 0;
    if (!pyssize_to_u32(n, what, &n_u32)) return NULL;

    uint64_t *arr = (uint64_t *)malloc((size_t)n_u32 * sizeof(uint64_t));
    if (!arr) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned long long value =
            PyLong_AsUnsignedLongLong(PyList_GET_ITEM(list, i));
        if (PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
                PyErr_Clear();
                PyErr_Format(PyExc_OverflowError,
                             "%s[%zd] exceeds uint64 limit", what, i);
            }
            free(arr);
            return NULL;
        }
        if (value > (unsigned long long)UINT64_MAX) {
            PyErr_Format(PyExc_OverflowError,
                         "%s[%zd] exceeds uint64 limit", what, i);
            free(arr);
            return NULL;
        }
        arr[i] = (uint64_t)value;
    }

    return arr;
}

/* Convert LZGError to Python exception using thread-local error message.
 * Returns NULL (convenience for: return set_lzg_error(err);). */
static PyObject *set_lzg_error(LZGError err) {
    const char *msg = lzg_error_message();
    int has_msg = msg && msg[0];

    switch (err) {
        case LZG_ERR_ALLOC:
            return PyErr_NoMemory();

        /* Input validation → ValueError */
        case LZG_ERR_NULL_ARG:
        case LZG_ERR_EMPTY_INPUT:
        case LZG_ERR_INVALID_SEQUENCE:
        case LZG_ERR_INVALID_VARIANT:
        case LZG_ERR_LENGTH_MISMATCH:
        case LZG_ERR_PARAM_OUT_OF_RANGE:
            PyErr_SetString(PyExc_ValueError,
                has_msg ? msg : "invalid argument");
            return NULL;

        /* Graph state → RuntimeError */
        case LZG_ERR_NOT_BUILT:
        case LZG_ERR_NO_LIVE_PATHS:
        case LZG_ERR_HAS_CYCLES:
            PyErr_SetString(PyExc_RuntimeError,
                has_msg ? msg : "graph state error");
            return NULL;

        /* Gene data → NoGeneDataError */
        case LZG_ERR_NO_GENE_DATA:
        case LZG_ERR_GENE_NOT_FOUND:
            PyErr_SetString(LZGExc_NoGeneDataError,
                has_msg ? msg : "gene data error");
            return NULL;

        /* Graph operations → ValueError */
        case LZG_ERR_VARIANT_MISMATCH:
        case LZG_ERR_MISSING_EDGE:
            PyErr_SetString(PyExc_ValueError,
                has_msg ? msg : "graph operation error");
            return NULL;

        /* IO: file-not-found/read/write → OSError; corrupt/version → CorruptFileError */
        case LZG_ERR_IO_OPEN:
        case LZG_ERR_IO_READ:
        case LZG_ERR_IO_WRITE:
            PyErr_SetString(PyExc_OSError,
                has_msg ? msg : "I/O error");
            return NULL;
        case LZG_ERR_IO_CORRUPT:
        case LZG_ERR_IO_VERSION:
            PyErr_SetString(LZGExc_CorruptFileError,
                has_msg ? msg : "corrupt or unsupported LZG file");
            return NULL;

        /* Numerical → ConvergenceError */
        case LZG_ERR_CONVERGENCE:
            PyErr_SetString(LZGExc_ConvergenceError,
                has_msg ? msg : "numerical method did not converge");
            return NULL;

        /* Overflow → OverflowError */
        case LZG_ERR_OVERFLOW:
            PyErr_SetString(PyExc_OverflowError,
                has_msg ? msg : "internal buffer capacity exceeded");
            return NULL;

        /* Internal bug → RuntimeError with clear "report this" message */
        case LZG_ERR_INTERNAL:
            PyErr_Format(PyExc_RuntimeError,
                "internal error (please report this bug): %s",
                has_msg ? msg : "unknown invariant violation");
            return NULL;

        default:
            if (has_msg)
                PyErr_Format(PyExc_RuntimeError, "%s", msg);
            else
                PyErr_Format(PyExc_RuntimeError, "LZGraph error code %d", (int)err);
            return NULL;
    }
}

/* Parse variant string to enum */
static int parse_variant(const char *s, LZGVariant *out) {
    if (strcmp(s, "aap") == 0)        { *out = LZG_VARIANT_AAP; return 1; }
    else if (strcmp(s, "ndp") == 0)   { *out = LZG_VARIANT_NDP; return 1; }
    else if (strcmp(s, "naive") == 0) { *out = LZG_VARIANT_NAIVE; return 1; }
    PyErr_Format(PyExc_ValueError,
                 "variant must be 'aap', 'ndp', or 'naive', got '%s'", s);
    return 0;
}

/* Extract a list of C strings from a Python list. Caller frees the array. */
static const char **pylist_to_cstrings(PyObject *list, Py_ssize_t *out_n) {
    Py_ssize_t n = PyList_GET_SIZE(list);
    const char **arr = (const char **)malloc(n * sizeof(char *));
    if (!arr) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        arr[i] = PyUnicode_AsUTF8(item);
        if (!arr[i]) { free(arr); return NULL; }
    }
    *out_n = n;
    return arr;
}

/* ── graph_build(sequences, variant, abundances, v_genes, j_genes,
                  smoothing, min_init) → capsule ─────────── */

static PyObject *py_graph_build(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *seq_list;
    const char *variant_str = "aap";
    PyObject *abund_obj = Py_None;
    PyObject *vgenes_obj = Py_None;
    PyObject *jgenes_obj = Py_None;
    double smoothing = 0.0;

    static char *kwlist[] = {
        "sequences", "variant", "abundances", "v_genes", "j_genes",
        "smoothing", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|sOOOd", kwlist,
            &PyList_Type, &seq_list, &variant_str,
            &abund_obj, &vgenes_obj, &jgenes_obj,
            &smoothing))
        return NULL;

    LZGVariant variant;
    if (!parse_variant(variant_str, &variant)) return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_seqs_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_seqs_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    /* Abundances */
    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    /* V/J genes */
    const char **v_genes = NULL, **j_genes = NULL;
    Py_ssize_t nv = 0, nj = 0;
    if (vgenes_obj != Py_None) {
        if (!PyList_Check(vgenes_obj)) {
            free(seqs); free(abundances);
            PyErr_SetString(PyExc_TypeError, "v_genes must be a list");
            return NULL;
        }
        v_genes = pylist_to_cstrings(vgenes_obj, &nv);
        if (!v_genes) { free(seqs); free(abundances); return NULL; }
    }
    if (jgenes_obj != Py_None) {
        if (!PyList_Check(jgenes_obj)) {
            free(seqs); free(abundances); free((void *)v_genes);
            PyErr_SetString(PyExc_TypeError, "j_genes must be a list");
            return NULL;
        }
        j_genes = pylist_to_cstrings(jgenes_obj, &nj);
        if (!j_genes) { free(seqs); free(abundances); free((void *)v_genes); return NULL; }
    }

    /* Build */
    LZGGraph *g = lzg_graph_create(variant);
    if (!g) {
        free(seqs); free(abundances); free((void *)v_genes); free((void *)j_genes);
        return PyErr_NoMemory();
    }

    LZGError err = lzg_graph_build(g, seqs, n_seqs_u32, abundances,
                                    v_genes, j_genes, smoothing, 0);
    free(seqs); free(abundances); free((void *)v_genes); free((void *)j_genes);

    if (err != LZG_OK) {
        lzg_graph_destroy(g);
        return set_lzg_error(err);
    }

    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

/* ── graph_build_file(path, variant, smoothing) → capsule ───── */

static PyObject *py_graph_build_file(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    const char *path = NULL;
    const char *variant_str = "aap";
    double smoothing = 0.0;

    static char *kwlist[] = {"path", "variant", "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|sd", kwlist,
            &path, &variant_str, &smoothing))
        return NULL;

    LZGVariant variant;
    if (!parse_variant(variant_str, &variant)) return NULL;

    LZGGraph *g = lzg_graph_create(variant);
    if (!g) return PyErr_NoMemory();

    LZGError err = lzg_graph_build_plain_file(g, path, smoothing);
    if (err != LZG_OK) {
        lzg_graph_destroy(g);
        return set_lzg_error(err);
    }

    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

/* ── graph_info(capsule) → dict of basic properties ──────── */

static PyObject *py_graph_info(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    const char *vstr = "aap";
    if (g->variant == LZG_VARIANT_NDP) vstr = "ndp";
    else if (g->variant == LZG_VARIANT_NAIVE) vstr = "naive";
    else if (g->variant == LZG_VARIANT_NAIVE_POS) vstr = "naive_positional";
    else if (g->variant == LZG_VARIANT_FLAT_FB) vstr = "flattened_flashback";

    return Py_BuildValue("{s:I, s:I, s:s, s:O, s:O}",
        "n_nodes", g->n_nodes,
        "n_edges", g->n_edges,
        "variant", vstr,
        "has_gene_data", g->gene_data ? Py_True : Py_False,
        "is_dag", g->topo_order ? Py_True : Py_False);
}

/* ── simulate(capsule, n, seed) → (sequences, log_probs, n_tokens) ── */

static PyObject *py_simulate(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;

    static char *kwlist[] = {"graph", "n", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|L", kwlist, &cap, &n, &seed))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0xDEADBEEF));

    LZGSimResult *results = (LZGSimResult *)calloc(n, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();

    LZGError err = lzg_simulate(g, n, &rng, results);
    if (err != LZG_OK) {
        free(results);
        return set_lzg_error(err);
    }

    PyObject *seq_list = PyList_New(n);
    PyObject *lp_list = PyList_New(n);
    PyObject *nt_list = PyList_New(n);
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(seq_list, i, PyUnicode_FromString(results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lp_list, i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nt_list, i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);

    return Py_BuildValue("(OOO)", seq_list, lp_list, nt_list);
}

/* ── gene_simulate(capsule, n, seed, v_gene_id, j_gene_id) ── */

static PyObject *py_gene_simulate(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;
    unsigned int v_id = UINT32_MAX, j_id = UINT32_MAX;

    static char *kwlist[] = {"graph", "n", "seed", "v_gene_id", "j_gene_id", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|LII", kwlist,
                                      &cap, &n, &seed, &v_id, &j_id))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0xBEEFCAFE));

    LZGGeneSimResult *results = (LZGGeneSimResult *)calloc(n, sizeof(LZGGeneSimResult));
    if (!results) return PyErr_NoMemory();

    LZGError err;
    if (v_id == UINT32_MAX && j_id == UINT32_MAX)
        err = lzg_gene_simulate(g, n, &rng, results);
    else
        err = lzg_gene_simulate_vj(g, n, &rng, v_id, j_id, results);

    if (err != LZG_OK) {
        free(results);
        return set_lzg_error(err);
    }

    PyObject *seq_list = PyList_New(n);
    PyObject *lp_list = PyList_New(n);
    PyObject *nt_list = PyList_New(n);
    PyObject *vg_list = PyList_New(n);
    PyObject *jg_list = PyList_New(n);

    const LZGGeneData *gd = (const LZGGeneData *)g->gene_data;

    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(seq_list, i, PyUnicode_FromString(
            results[i].base.sequence ? results[i].base.sequence : ""));
        PyList_SET_ITEM(lp_list, i, PyFloat_FromDouble(results[i].base.log_prob));
        PyList_SET_ITEM(nt_list, i, PyLong_FromUnsignedLong(results[i].base.n_tokens));

        const char *vname = (gd && results[i].v_gene_id != LZG_SP_NOT_FOUND)
            ? lzg_sp_get(gd->gene_pool, results[i].v_gene_id) : "";
        const char *jname = (gd && results[i].j_gene_id != LZG_SP_NOT_FOUND)
            ? lzg_sp_get(gd->gene_pool, results[i].j_gene_id) : "";
        PyList_SET_ITEM(vg_list, i, PyUnicode_FromString(vname));
        PyList_SET_ITEM(jg_list, i, PyUnicode_FromString(jname));

        lzg_gene_sim_result_free(&results[i]);
    }
    free(results);

    return Py_BuildValue("(OOOOO)", seq_list, lp_list, nt_list, vg_list, jg_list);
}

/* ── lzpgen(capsule, sequence_or_list) → float or list[float] ── */

static PyObject *py_lzpgen(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_arg;

    if (!PyArg_ParseTuple(args, "OO", &cap, &seq_arg))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    if (PyUnicode_Check(seq_arg)) {
        /* Single string */
        const char *seq = PyUnicode_AsUTF8(seq_arg);
        if (!seq) return NULL;
        double lp = lzg_walk_log_prob(g, seq, (uint32_t)strlen(seq));
        return PyFloat_FromDouble(lp);
    }

    if (PyList_Check(seq_arg)) {
        Py_ssize_t n = PyList_GET_SIZE(seq_arg);
        PyObject *result = PyList_New(n);
        for (Py_ssize_t i = 0; i < n; i++) {
            const char *seq = PyUnicode_AsUTF8(PyList_GET_ITEM(seq_arg, i));
            if (!seq) { Py_DECREF(result); return NULL; }
            double lp = lzg_walk_log_prob(g, seq, (uint32_t)strlen(seq));
            PyList_SET_ITEM(result, i, PyFloat_FromDouble(lp));
        }
        return result;
    }

    PyErr_SetString(PyExc_TypeError, "sequence must be str or list[str]");
    return NULL;
}

/* ── path_count(capsule[, n]) → float ─────────────────────── */

static PyObject *py_path_count(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int n_samples = 0;
    if (!PyArg_ParseTuple(args, "O|I", &cap, &n_samples)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double count;
    LZGError err = lzg_graph_path_count_mc(g, (uint32_t)n_samples, &count);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(count);
}

/* ── effective_diversity(capsule) → float ─────────────────── */

static PyObject *py_effective_diversity(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGEffectiveDiversity div;
    LZGError err = lzg_effective_diversity(g, &div);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(div.effective_diversity);
}

/* ── diversity_profile(capsule) → dict ────────────────────── */

static PyObject *py_diversity_profile(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGEffectiveDiversity div;
    LZGError err = lzg_effective_diversity(g, &div);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d, s:d, s:d, s:d}",
        "entropy_nats", div.entropy_nats,
        "entropy_bits", div.entropy_bits,
        "effective_diversity", div.effective_diversity,
        "uniformity", div.uniformity);
}

/* ── hill_number(capsule, alpha) → float ──────────────────── */

static PyObject *py_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double alpha;
    unsigned int n_samples = 0;
    if (!PyArg_ParseTuple(args, "Od|I", &cap, &alpha, &n_samples)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double d;
    LZGError err = lzg_hill_number_mc(g, alpha, (uint32_t)n_samples, &d);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(d);
}

/* ── hill_numbers(capsule, orders_list) → list[float] ─────── */

static PyObject *py_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_list;
    unsigned int n_samples = 0;
    if (!PyArg_ParseTuple(args, "OO!|I", &cap, &PyList_Type,
                          &orders_list, &n_samples)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(orders_list);
    double *orders = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!orders || !out) { free(orders); free(out); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n; i++)
        orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_list, i));
    if (PyErr_Occurred()) { free(orders); free(out); return NULL; }

    LZGError err = lzg_hill_numbers_mc(g, orders, (uint32_t)n,
                                       (uint32_t)n_samples, out);
    free(orders);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }

    PyObject *result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

/* ── hill_curve(capsule, orders_list_or_none) → dict ──────── */

static PyObject *py_hill_curve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_obj;
    if (!PyArg_ParseTuple(args, "OO", &cap, &orders_obj)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    double *orders = NULL;
    uint32_t n = 0;
    if (orders_obj != Py_None && PyList_Check(orders_obj)) {
        n = (uint32_t)PyList_GET_SIZE(orders_obj);
        orders = (double *)malloc(n * sizeof(double));
        if (!orders) return PyErr_NoMemory();
        for (uint32_t i = 0; i < n; i++)
            orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_obj, i));
        if (PyErr_Occurred()) { free(orders); return NULL; }
    }

    LZGHillCurve hc;
    LZGError err = lzg_hill_curve(g, orders, n, &hc);
    free(orders);
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *o_list = PyList_New(hc.n);
    PyObject *v_list = PyList_New(hc.n);
    for (uint32_t i = 0; i < hc.n; i++) {
        PyList_SET_ITEM(o_list, i, PyFloat_FromDouble(hc.orders[i]));
        PyList_SET_ITEM(v_list, i, PyFloat_FromDouble(hc.hill_numbers[i]));
    }
    lzg_hill_curve_free(&hc);
    return Py_BuildValue("{s:O, s:O}", "orders", o_list, "values", v_list);
}

/* ── power_sum(capsule, alpha) → float ────────────────────── */

static PyObject *py_power_sum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double m;
    LZGError err = lzg_power_sum(g, alpha, &m);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(m);
}

/* ── pgen_diagnostics(capsule, atol) → dict ───────────────── */

static PyObject *py_pgen_diagnostics(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double atol = 1e-6;
    if (!PyArg_ParseTuple(args, "O|d", &cap, &atol)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenDiagnostics diag;
    LZGError err = lzg_pgen_diagnostics(g, atol, &diag);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d, s:d, s:d, s:O, s:I}",
        "total_absorbed", diag.total_absorbed,
        "total_leaked", diag.total_leaked,
        "initial_prob_sum", diag.initial_prob_sum,
        "is_proper", diag.is_proper ? Py_True : Py_False,
        "mc_samples", diag.mc_samples);
}

/* ── pgen_dynamic_range(capsule) → float ──────────────────── */

static PyObject *py_pgen_dynamic_range(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_pgen_dynamic_range(g, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(dr.dynamic_range_orders);
}

/* ── pgen_dynamic_range_detail(capsule) → dict ────────────── */

static PyObject *py_pgen_dynamic_range_detail(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_pgen_dynamic_range(g, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d, s:d, s:d, s:d}",
        "max_log_prob", dr.max_log_prob,
        "min_log_prob", dr.min_log_prob,
        "dynamic_range_nats", dr.dynamic_range_nats,
        "dynamic_range_orders", dr.dynamic_range_orders);
}

/* ── pgen_moments(capsule) → dict ─────────────────────────── */

static PyObject *py_pgen_moments(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenMoments m;
    LZGError err = lzg_pgen_moments(g, &m);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d, s:d, s:d, s:d, s:d, s:d}",
        "mean", m.mean, "variance", m.variance, "std", m.std,
        "skewness", m.skewness, "kurtosis", m.kurtosis,
        "total_mass", m.total_mass);
}

/* ── pgen_analytical(capsule) → dict (components) ─────────── */

static PyObject *py_pgen_analytical(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenDist dist;
    LZGError err = lzg_pgen_analytical(g, &dist);
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *w = PyList_New(dist.n_components);
    PyObject *mu = PyList_New(dist.n_components);
    PyObject *sd = PyList_New(dist.n_components);
    for (uint32_t i = 0; i < dist.n_components; i++) {
        PyList_SET_ITEM(w, i, PyFloat_FromDouble(dist.weights[i]));
        PyList_SET_ITEM(mu, i, PyFloat_FromDouble(dist.means[i]));
        PyList_SET_ITEM(sd, i, PyFloat_FromDouble(dist.stds[i]));
    }
    return Py_BuildValue("{s:O, s:O, s:O, s:d}",
        "weights", w, "means", mu, "stds", sd,
        "global_mean", dist.global.mean);
}

/* ── predicted_richness(capsule, d) → float ───────────────── */

static PyObject *py_predicted_richness(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double d;
    if (!PyArg_ParseTuple(args, "Od", &cap, &d)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double out;
    LZGError err = lzg_predicted_richness(g, d, &out);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(out);
}

/* ── predicted_overlap(capsule, d_i, d_j) → float ─────────── */

static PyObject *py_predicted_overlap(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double di, dj;
    if (!PyArg_ParseTuple(args, "Odd", &cap, &di, &dj)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double out;
    LZGError err = lzg_predicted_overlap(g, di, dj, &out);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(out);
}

/* ── richness_curve(capsule, d_list) → list[float] ────────── */

static PyObject *py_richness_curve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *d_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &d_list)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(d_list);
    double *ds = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!ds || !out) { free(ds); free(out); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n; i++)
        ds[i] = PyFloat_AsDouble(PyList_GET_ITEM(d_list, i));
    if (PyErr_Occurred()) { free(ds); free(out); return NULL; }

    LZGError err = lzg_richness_curve(g, ds, (uint32_t)n, out);
    free(ds);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }

    PyObject *result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

/* ── pseq_discovery_curve(probabilities, counts, draws) → dict ─ */

static PyObject *py_pseq_discovery_curve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *probability_obj, *multiplicity_obj, *draw_obj;
    if (!PyArg_ParseTuple(args, "OOO", &probability_obj,
                          &multiplicity_obj, &draw_obj))
        return NULL;
    PyObject *probability_values = PySequence_Fast(
        probability_obj, "probabilities must be a sequence");
    PyObject *multiplicity_values = PySequence_Fast(
        multiplicity_obj, "multiplicities must be a sequence");
    PyObject *draw_values = PySequence_Fast(
        draw_obj, "draw_counts must be a sequence");
    if (!probability_values || !multiplicity_values || !draw_values) {
        Py_XDECREF(probability_values); Py_XDECREF(multiplicity_values);
        Py_XDECREF(draw_values);
        return NULL;
    }
    const Py_ssize_t n_atoms_py = PySequence_Fast_GET_SIZE(probability_values);
    const Py_ssize_t n_multiplicities_py =
        PySequence_Fast_GET_SIZE(multiplicity_values);
    const Py_ssize_t n_draws_py = PySequence_Fast_GET_SIZE(draw_values);
    uint32_t n_atoms, n_draws;
    if (n_atoms_py != n_multiplicities_py) {
        PyErr_SetString(PyExc_ValueError,
                        "probabilities and multiplicities must have equal length");
        goto discovery_input_error;
    }
    if (!pyssize_to_u32(n_atoms_py, "probabilities", &n_atoms) ||
        !pyssize_to_u32(n_draws_py, "draw_counts", &n_draws))
        goto discovery_input_error;

    double *probabilities = n_atoms ? (double *)malloc(
        (size_t)n_atoms * sizeof(double)) : NULL;
    double *multiplicities = n_atoms ? (double *)malloc(
        (size_t)n_atoms * sizeof(double)) : NULL;
    double *draws = n_draws ? (double *)malloc(
        (size_t)n_draws * sizeof(double)) : NULL;
    double *richness = n_draws ? (double *)malloc(
        (size_t)n_draws * sizeof(double)) : NULL;
    double *novelty = n_draws ? (double *)malloc(
        (size_t)n_draws * sizeof(double)) : NULL;
    if ((n_atoms && (!probabilities || !multiplicities)) ||
        (n_draws && (!draws || !richness || !novelty))) {
        free(probabilities); free(multiplicities); free(draws);
        free(richness); free(novelty);
        Py_DECREF(probability_values); Py_DECREF(multiplicity_values);
        Py_DECREF(draw_values);
        return PyErr_NoMemory();
    }
    PyObject **probability_items = PySequence_Fast_ITEMS(probability_values);
    PyObject **multiplicity_items = PySequence_Fast_ITEMS(multiplicity_values);
    PyObject **draw_items = PySequence_Fast_ITEMS(draw_values);
    for (uint32_t i = 0; i < n_atoms; i++) {
        probabilities[i] = PyFloat_AsDouble(probability_items[i]);
        multiplicities[i] = PyFloat_AsDouble(multiplicity_items[i]);
        if (PyErr_Occurred()) goto discovery_conversion_error;
    }
    for (uint32_t i = 0; i < n_draws; i++) {
        draws[i] = PyFloat_AsDouble(draw_items[i]);
        if (PyErr_Occurred()) goto discovery_conversion_error;
    }
    Py_DECREF(probability_values); Py_DECREF(multiplicity_values);
    Py_DECREF(draw_values);

    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_pseq_discovery_curve(
        probabilities, multiplicities, n_atoms, draws, n_draws,
        richness, novelty);
    Py_END_ALLOW_THREADS
    free(probabilities); free(multiplicities); free(draws);
    if (err != LZG_OK) {
        free(richness); free(novelty);
        return set_lzg_error(err);
    }

    PyObject *richness_list = PyList_New(n_draws_py);
    PyObject *novelty_list = PyList_New(n_draws_py);
    if (!richness_list || !novelty_list) {
        Py_XDECREF(richness_list); Py_XDECREF(novelty_list);
        free(richness); free(novelty);
        return NULL;
    }
    for (uint32_t i = 0; i < n_draws; i++) {
        PyObject *richness_value = PyFloat_FromDouble(richness[i]);
        PyObject *novelty_value = PyFloat_FromDouble(novelty[i]);
        if (!richness_value || !novelty_value) {
            Py_XDECREF(richness_value); Py_XDECREF(novelty_value);
            Py_DECREF(richness_list); Py_DECREF(novelty_list);
            free(richness); free(novelty);
            return NULL;
        }
        PyList_SET_ITEM(richness_list, i, richness_value);
        PyList_SET_ITEM(novelty_list, i, novelty_value);
    }
    free(richness); free(novelty);
    return Py_BuildValue(
        "{s:N,s:N}",
        "expected_richness", richness_list,
        "novelty_probability", novelty_list);

discovery_conversion_error:
    free(probabilities); free(multiplicities); free(draws);
    free(richness); free(novelty);
    Py_DECREF(probability_values); Py_DECREF(multiplicity_values);
    Py_DECREF(draw_values);
    return NULL;

discovery_input_error:
    Py_DECREF(probability_values); Py_DECREF(multiplicity_values);
    Py_DECREF(draw_values);
    return NULL;
}

/* ── predict_sharing(capsule, draws_list, max_k) → dict ───── */

static PyObject *py_predict_sharing(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *draws_list;
    unsigned int max_k = 0;
    if (!PyArg_ParseTuple(args, "OO!|I", &cap, &PyList_Type, &draws_list, &max_k))
        return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t nd = PyList_GET_SIZE(draws_list);
    double *draws = (double *)malloc(nd * sizeof(double));
    if (!draws) return PyErr_NoMemory();
    for (Py_ssize_t i = 0; i < nd; i++)
        draws[i] = PyFloat_AsDouble(PyList_GET_ITEM(draws_list, i));
    if (PyErr_Occurred()) { free(draws); return NULL; }

    if (max_k == 0) max_k = (uint32_t)nd;

    LZGSharingSpectrum ss;
    LZGError err = lzg_predict_sharing(g, draws, (uint32_t)nd, max_k, &ss);
    free(draws);
    if (err != LZG_OK) return set_lzg_error(err);

    /* spectrum[k] for k=0..max_k-1 corresponds to "expected sequences shared
     * by exactly 1, 2, ..., max_k donors". The C side allocates and writes
     * exactly max_k entries; reading max_k+1 was an OOB-read bug that
     * surfaced as a denormal-tail nondeterminism. */
    PyObject *spec = PyList_New(ss.max_k);
    for (uint32_t i = 0; i < ss.max_k; i++)
        PyList_SET_ITEM(spec, i, PyFloat_FromDouble(ss.spectrum[i]));
    lzg_sharing_spectrum_free(&ss);

    return Py_BuildValue("{s:O, s:d, s:I}",
        "spectrum", spec, "expected_total", ss.expected_total,
        "n_donors", ss.n_donors);
}

/* ── Buffer helpers for the publicness batch API ───────────── */

/* The publicness entry points move one PMF of up to 65,536 doubles per atom,
 * which is well past the point where a Python list round-trip makes sense.
 * They take the buffer protocol instead. Every array is float64 so the format
 * check stays portable across platforms; a complex128 array is handed over as
 * its float64 view, and integral quantities (multiplicities, bin edges) as
 * exact float64 integers. */

static int get_f64_buffer(PyObject *obj, Py_buffer *view, int writable,
                          const char *what) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;

    if (PyObject_GetBuffer(obj, view, flags) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous%s float64 array",
                     what, writable ? ", writable" : "");
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(double) ||
        !view->format || strcmp(view->format, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must have dtype float64", what);
        return 0;
    }
    return 1;
}

static void release_f64_buffers(Py_buffer *views, int n) {
    for (int i = 0; i < n; i++) PyBuffer_Release(&views[i]);
}

/* Acquire several float64 buffers at once, releasing what was already taken
 * if any of them fails. */
static int get_f64_buffers(PyObject **objs, Py_buffer *views,
                           const int *writable, const char **names, int n) {
    for (int i = 0; i < n; i++) {
        if (!get_f64_buffer(objs[i], &views[i], writable[i], names[i])) {
            release_f64_buffers(views, i);
            return 0;
        }
    }
    return 1;
}

static Py_ssize_t f64_buffer_count(const Py_buffer *view) {
    return view->len / (Py_ssize_t)sizeof(double);
}

/* ── publicness_moments(p, depths, mult, mean, var) → None ── */

static PyObject *py_publicness_moments(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[5] = {0, 0, 0, 1, 1};
    static const char *names[5] = {"probabilities", "depths", "multiplicities",
                                   "mean output", "variance output"};
    PyObject *objs[5];
    Py_buffer views[5];
    Py_ssize_t n_atoms, n_groups;
    LZGError err;

    if (!PyArg_ParseTuple(args, "OOOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 5)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    if (f64_buffer_count(&views[2]) != n_groups ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[4]) != n_atoms) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and both "
            "outputs must match the probability count");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = lzg_publicness_moments((const double *)views[0].buf,
                                 (uint32_t)n_atoms,
                                 (const double *)views[1].buf,
                                 (const double *)views[2].buf,
                                 (uint32_t)n_groups,
                                 (double *)views[3].buf,
                                 (double *)views[4].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 5);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

/* ── publicness_pgf(p, depths, mult, out) → None ──────────── */

static PyObject *py_publicness_pgf(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[4] = {0, 0, 0, 1};
    static const char *names[4] = {"probabilities", "depths", "multiplicities",
                                   "generating-function output"};
    PyObject *objs[4];
    Py_buffer views[4];
    Py_ssize_t n_atoms, n_groups, out_count, k_fft;
    LZGError err;

    if (!PyArg_ParseTuple(args, "OOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 4)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    out_count = f64_buffer_count(&views[3]);
    /* The output is the float64 view of an (n_atoms, k_fft) complex array. */
    k_fft = (n_atoms > 0 && out_count % (2 * n_atoms) == 0)
                ? out_count / (2 * n_atoms) : -1;

    if (f64_buffer_count(&views[2]) != n_groups || k_fft < 2) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and the "
            "output must be the float64 view of an (n_atoms, k_fft) "
            "complex128 array");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX || k_fft > UINT32_MAX) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    /* Release the GIL: this is n_atoms * n_groups * k_fft/2 complex powers,
     * which reaches tens of seconds on a foundation-scale cohort. */
    Py_BEGIN_ALLOW_THREADS
    err = lzg_publicness_pgf((const double *)views[0].buf, (uint32_t)n_atoms,
                             (const double *)views[1].buf,
                             (const double *)views[2].buf,
                             (uint32_t)n_groups, (uint32_t)k_fft,
                             (double *)views[3].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 4);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

/* ── publicness_accumulate(pmf, mean, var, weight, edges,
 *                          counts, retained, sigma, floor, tol) → None ── */

static PyObject *py_publicness_accumulate(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[7] = {0, 0, 0, 0, 0, 1, 1};
    static const char *names[7] = {"pmf", "mean", "variance", "weight",
                                   "bin edges", "count output",
                                   "retained-mass output"};
    PyObject *objs[7];
    Py_buffer views[7];
    Py_ssize_t n_atoms, k_fft, n_bins, pmf_count;
    double tail_sigma, tail_floor, mass_tol;
    LZGError err;

    if (!PyArg_ParseTuple(args, "OOOOOOOddd", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4], &objs[5], &objs[6],
                          &tail_sigma, &tail_floor, &mass_tol)) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 7)) return NULL;

    n_atoms = f64_buffer_count(&views[1]);
    pmf_count = f64_buffer_count(&views[0]);
    k_fft = (n_atoms > 0 && pmf_count % n_atoms == 0) ? pmf_count / n_atoms : -1;
    n_bins = f64_buffer_count(&views[4]) - 1;

    if (k_fft < 1 || n_bins < 1 ||
        f64_buffer_count(&views[2]) != n_atoms ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[6]) != n_atoms ||
        f64_buffer_count(&views[5]) != n_bins) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_ValueError,
            "pmf must be (n_atoms, k_fft); mean, variance, weight and the "
            "retained-mass output must each have n_atoms entries; and the "
            "count output must have one entry fewer than the bin edges");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || k_fft > UINT32_MAX || n_bins > UINT32_MAX) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = lzg_publicness_accumulate((const double *)views[0].buf,
                                    (uint32_t)n_atoms, (uint32_t)k_fft,
                                    (const double *)views[1].buf,
                                    (const double *)views[2].buf,
                                    (const double *)views[3].buf,
                                    (const double *)views[4].buf,
                                    (uint32_t)n_bins,
                                    tail_sigma, tail_floor, mass_tol,
                                    (double *)views[5].buf,
                                    (double *)views[6].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 7);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

/* ── sequence_perplexity(capsule, seq) → float ────────────── */

static PyObject *py_sequence_perplexity(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    return PyFloat_FromDouble(lzg_sequence_perplexity(g, seq, (uint32_t)strlen(seq)));
}

/* ── repertoire_perplexity(capsule, seq_list) → float ─────── */

static PyObject *py_repertoire_perplexity(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &seq_list)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    Py_ssize_t n;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;
    double pp = lzg_repertoire_perplexity(g, seqs, (uint32_t)n);
    free(seqs);
    return PyFloat_FromDouble(pp);
}

/* ── path_entropy_rate(capsule, seq_list) → float ─────────── */

static PyObject *py_path_entropy_rate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &seq_list)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    Py_ssize_t n;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;
    double rate = lzg_path_entropy_rate(g, seqs, (uint32_t)n);
    free(seqs);
    return PyFloat_FromDouble(rate);
}

/* ── jensen_shannon_divergence(cap_a, cap_b) → float ──────── */

static PyObject *py_jsd(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap_a, *cap_b;
    if (!PyArg_ParseTuple(args, "OO", &cap_a, &cap_b)) return NULL;
    LZGGraph *a = (LZGGraph *)PyCapsule_GetPointer(cap_a, CAPSULE_NAME);
    LZGGraph *b = (LZGGraph *)PyCapsule_GetPointer(cap_b, CAPSULE_NAME);
    if (!a || !b) return NULL;
    double out;
    LZGError err = lzg_jensen_shannon_divergence(a, b, &out);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(out);
}

/* ── graph_summary(capsule) → dict ────────────────────────── */

static PyObject *py_summary(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGGraphSummary s;
    LZGError err = lzg_graph_summary(g, &s);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:I, s:I, s:I, s:I, s:I, s:I, s:I, s:O}",
        "n_nodes", s.n_nodes, "n_edges", s.n_edges,
        "n_initial", s.n_initial, "n_terminal", s.n_terminal,
        "max_out_degree", s.max_out_degree, "max_in_degree", s.max_in_degree,
        "n_isolates", s.n_isolates,
        "is_dag", s.is_dag ? Py_True : Py_False);
}

/* ── graph_union/intersection/difference(cap_a, cap_b) → cap */

static PyObject *py_graph_setop(PyObject *self, PyObject *args,
    LZGError (*op)(const LZGGraph*, const LZGGraph*, LZGGraph**))
{
    (void)self;
    PyObject *ca, *cb;
    if (!PyArg_ParseTuple(args, "OO", &ca, &cb)) return NULL;
    LZGGraph *a = (LZGGraph *)PyCapsule_GetPointer(ca, CAPSULE_NAME);
    LZGGraph *b = (LZGGraph *)PyCapsule_GetPointer(cb, CAPSULE_NAME);
    if (!a || !b) return NULL;
    LZGGraph *out = NULL;
    LZGError err = op(a, b, &out);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(out, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_graph_union(PyObject *s, PyObject *a) { return py_graph_setop(s, a, lzg_graph_union); }
static PyObject *py_graph_intersection(PyObject *s, PyObject *a) { return py_graph_setop(s, a, lzg_graph_intersection); }
static PyObject *py_graph_difference(PyObject *s, PyObject *a) { return py_graph_setop(s, a, lzg_graph_difference); }

/* ── weighted_merge(cap_a, cap_b, alpha, beta) → cap ──────── */

static PyObject *py_weighted_merge(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *ca, *cb; double alpha, beta;
    if (!PyArg_ParseTuple(args, "OOdd", &ca, &cb, &alpha, &beta)) return NULL;
    LZGGraph *a = (LZGGraph *)PyCapsule_GetPointer(ca, CAPSULE_NAME);
    LZGGraph *b = (LZGGraph *)PyCapsule_GetPointer(cb, CAPSULE_NAME);
    if (!a || !b) return NULL;
    LZGGraph *out = NULL;
    LZGError err = lzg_graph_weighted_merge(a, b, alpha, beta, &out);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(out, CAPSULE_NAME, capsule_destructor);
}

/* ── posterior(cap, sequences, abundances, kappa) → cap ───── */

static PyObject *py_posterior(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap, *seq_list;
    PyObject *abund_obj = Py_None;
    double kappa = 1.0;

    static char *kwlist[] = {"graph", "sequences", "abundances", "kappa", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|Od", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj, &kappa))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n;
    uint32_t n_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *post = NULL;
    LZGError err = lzg_graph_posterior(g, seqs, n_u32, abundances, kappa, &post);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(post, CAPSULE_NAME, capsule_destructor);
}

/* ── feature_stats(capsule) → list[float] ─────────────────── */

static PyObject *py_feature_stats(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    double stats[LZG_FEATURE_STATS_DIM];
    LZGError err = lzg_feature_stats(g, stats);
    if (err != LZG_OK) return set_lzg_error(err);
    PyObject *result = PyList_New(LZG_FEATURE_STATS_DIM);
    for (int i = 0; i < LZG_FEATURE_STATS_DIM; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(stats[i]));
    return result;
}

/* ── feature_mass_profile(capsule, max_pos) → list[float] ── */

static PyObject *py_feature_mass_profile(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int max_pos = 30;
    if (!PyArg_ParseTuple(args, "O|I", &cap, &max_pos)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double *out = (double *)calloc(max_pos + 1, sizeof(double));
    if (!out) return PyErr_NoMemory();
    LZGError err = lzg_feature_mass_profile(g, out, max_pos);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }
    PyObject *result = PyList_New(max_pos + 1);
    for (unsigned int i = 0; i <= max_pos; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

/* ── feature_aligned(ref_cap, query_cap) → list[float] ───── */

static PyObject *py_feature_aligned(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *ref_cap, *query_cap;
    if (!PyArg_ParseTuple(args, "OO", &ref_cap, &query_cap)) return NULL;
    LZGGraph *ref = (LZGGraph *)PyCapsule_GetPointer(ref_cap, CAPSULE_NAME);
    LZGGraph *query = (LZGGraph *)PyCapsule_GetPointer(query_cap, CAPSULE_NAME);
    if (!ref || !query) return NULL;

    double *out = (double *)calloc(ref->n_nodes, sizeof(double));
    if (!out) return PyErr_NoMemory();
    uint32_t dim;
    LZGError err = lzg_feature_aligned(ref, query, out, &dim);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }

    PyObject *result = PyList_New(dim);
    for (uint32_t i = 0; i < dim; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

/* ── save / load ──────────────────────────────────────────── */

static PyObject *py_save(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *path;
    if (!PyArg_ParseTuple(args, "Os", &cap, &path)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    LZGError err = lzg_graph_save(g, path);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

static PyObject *py_load(PyObject *self, PyObject *arg) {
    (void)self;
    const char *path = PyUnicode_AsUTF8(arg);
    if (!path) return NULL;
    LZGGraph *g = NULL;
    LZGError err = lzg_graph_load(path, &g);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

/* ── gene data access ─────────────────────────────────────── */

static PyObject *py_gene_info(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g || !g->gene_data) Py_RETURN_NONE;
    LZGGeneData *gd = (LZGGeneData *)g->gene_data;

    /* V marginals */
    PyObject *v_dict = PyDict_New();
    for (uint32_t i = 0; i < gd->n_v_genes; i++) {
        const char *name = lzg_sp_get(gd->gene_pool, gd->v_marginal_ids[i]);
        PyDict_SetItemString(v_dict, name, PyFloat_FromDouble(gd->v_marginal_probs[i]));
    }
    /* J marginals */
    PyObject *j_dict = PyDict_New();
    for (uint32_t i = 0; i < gd->n_j_genes; i++) {
        const char *name = lzg_sp_get(gd->gene_pool, gd->j_marginal_ids[i]);
        PyDict_SetItemString(j_dict, name, PyFloat_FromDouble(gd->j_marginal_probs[i]));
    }
    /* VJ distribution */
    PyObject *vj_list = PyList_New(gd->n_vj_pairs);
    for (uint32_t i = 0; i < gd->n_vj_pairs; i++) {
        const char *v = lzg_sp_get(gd->gene_pool, gd->vj_v_ids[i]);
        const char *j = lzg_sp_get(gd->gene_pool, gd->vj_j_ids[i]);
        PyList_SET_ITEM(vj_list, i,
            Py_BuildValue("{s:s, s:s, s:d}", "v", v, "j", j, "prob", gd->vj_probs[i]));
    }

    return Py_BuildValue("{s:O, s:O, s:O}", "v_marginals", v_dict,
                         "j_marginals", j_dict, "vj_distribution", vj_list);
}

/* Find gene ID by name (for gene_simulate_vj) */
static PyObject *py_find_gene_id(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *name;
    if (!PyArg_ParseTuple(args, "Os", &cap, &name)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g || !g->gene_data) {
        PyErr_SetString(PyExc_RuntimeError, "no gene data");
        return NULL;
    }
    LZGGeneData *gd = (LZGGeneData *)g->gene_data;
    uint32_t id = lzg_sp_find(gd->gene_pool, name);
    return PyLong_FromUnsignedLong(id);
}

/* ── lz76_decompose(string) → list[str] ──────────────────── */

static PyObject *py_lz76_decompose(PyObject *self, PyObject *arg) {
    (void)self;
    const char *seq = PyUnicode_AsUTF8(arg);
    if (!seq) return NULL;
    uint32_t len = (uint32_t)strlen(seq);

    LZGStringPool *pool = lzg_sp_create(64);
    if (!pool) return PyErr_NoMemory();

    LZGTokens tokens;
    LZGError err = lzg_lz76_decompose(seq, len, pool, &tokens);
    if (err != LZG_OK) {
        lzg_sp_destroy(pool);
        return set_lzg_error(err);
    }

    PyObject *result = PyList_New(tokens.count);
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        PyList_SET_ITEM(result, i, PyUnicode_FromString(sp));
    }
    lzg_sp_destroy(pool);
    return result;
}

/* ── flashback_decompose(string) → list[str] ───────────────── */

static PyObject *py_flashback_decompose(PyObject *self, PyObject *arg) {
    (void)self;
    const char *seq = PyUnicode_AsUTF8(arg);
    if (!seq) return NULL;
    uint32_t len = (uint32_t)strlen(seq);

    LZGStringPool *pool = lzg_sp_create(64);
    if (!pool) return PyErr_NoMemory();

    LZGFlashbackTokens tokens;
    LZGError err = lzg_flashback_decompose(seq, len, pool, &tokens);
    if (err != LZG_OK) {
        lzg_sp_destroy(pool);
        return set_lzg_error(err);
    }

    PyObject *result = PyList_New(tokens.count);
    for (uint32_t i = 0; i < tokens.count; i++) {
        const char *sp = lzg_sp_get(pool, tokens.sp_ids[i]);
        PyList_SET_ITEM(result, i, PyUnicode_FromString(sp));
    }
    lzg_sp_destroy(pool);
    return result;
}

/* ── flashback_reverse(list[str]) → str ────────────────────── */

static PyObject *py_flashback_reverse(PyObject *self, PyObject *arg) {
    (void)self;
    if (!PyList_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list of strings");
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(arg);
    LZGStringPool *pool = lzg_sp_create(64);
    if (!pool) return PyErr_NoMemory();

    LZGFlashbackTokens tokens;
    tokens.count = (uint32_t)n;

    for (Py_ssize_t i = 0; i < n; i++) {
        const char *s = PyUnicode_AsUTF8(PyList_GET_ITEM(arg, i));
        if (!s) { lzg_sp_destroy(pool); return NULL; }
        tokens.sp_ids[i] = lzg_sp_intern(pool, s);
    }

    char buf[2048];
    uint32_t out_len = 0;
    LZGError err = lzg_flashback_reverse(pool, &tokens, buf, sizeof(buf), &out_len);
    lzg_sp_destroy(pool);

    if (err != LZG_OK) return set_lzg_error(err);
    return PyUnicode_FromStringAndSize(buf, out_len);
}

/* ══════════════════════════════════════════════════════════════ */
/* FlashBackGraph Python wrappers                                  */
/* ══════════════════════════════════════════════════════════════ */

static PyObject *py_fb_graph_build(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *seq_list, *abund_obj = Py_None;
    double smoothing = 0.0;
    static char *kwlist[] = {"sequences", "abundances", "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|Od", kwlist,
            &PyList_Type, &seq_list, &abund_obj, &smoothing))
        return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_seqs_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_seqs_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE);
    if (!g) { free(seqs); free(abundances); return PyErr_NoMemory(); }

    LZGError err = lzg_flashback_graph_build(g, seqs, n_seqs_u32, abundances, smoothing);
    free(seqs); free(abundances);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_fb_graph_build_file(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    const char *path = NULL;
    double smoothing = 0.0;
    static char *kwlist[] = {"path", "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|d", kwlist, &path, &smoothing))
        return NULL;

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE);
    if (!g) return PyErr_NoMemory();
    LZGError err = lzg_flashback_graph_build_file(g, path, smoothing);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

/* ── FlashBack streaming builder ──────────────────────────────
 *
 * The C-level stream lifecycle separates "release resources"
 * (finalize/abort) from "free struct" (destroy). The capsule holds
 * the stream struct for the entirety of the Python object's life;
 * the destructor calls destroy(), which is idempotent w.r.t. having
 * already been finalized or aborted. This avoids the
 * PyCapsule_SetPointer-to-NULL trap entirely (which CPython rejects).
 */

static const char *FB_STREAM_CAPSULE_NAME = "LZGFlashbackStream";

static void fb_stream_capsule_destructor(PyObject *capsule) {
    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        capsule, FB_STREAM_CAPSULE_NAME);
    if (s) lzg_flashback_stream_destroy(s);
}

static PyObject *py_fb_stream_open(PyObject *self,
                                    PyObject *args, PyObject *kw) {
    (void)self;
    double smoothing = 0.0;
    static char *kwlist[] = {"smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|d", kwlist, &smoothing))
        return NULL;
    LZGFlashbackStream *s = lzg_flashback_stream_open(smoothing);
    if (!s) return PyErr_NoMemory();
    return PyCapsule_New(s, FB_STREAM_CAPSULE_NAME,
                          fb_stream_capsule_destructor);
}

static PyObject *py_fb_stream_add(PyObject *self,
                                   PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    PyObject *seq_list;
    PyObject *abund_obj = Py_None;
    static char *kwlist[] = {"stream", "sequences", "abundances", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|O", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj))
        return NULL;

    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        cap, FB_STREAM_CAPSULE_NAME);
    if (!s) return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_seqs_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_seqs_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGError err = lzg_flashback_stream_add(s, seqs, n_seqs_u32, abundances);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

static PyObject *py_fb_stream_peek(PyObject *self, PyObject *cap) {
    (void)self;
    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        cap, FB_STREAM_CAPSULE_NAME);
    if (!s) return NULL;
    uint32_t nn = 0, ne = 0;
    lzg_flashback_stream_peek(s, &nn, &ne);
    return Py_BuildValue("{s:I,s:I}", "n_nodes", nn, "n_edges", ne);
}

static PyObject *py_fb_stream_finalize(PyObject *self, PyObject *cap) {
    (void)self;
    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        cap, FB_STREAM_CAPSULE_NAME);
    if (!s) return NULL;

    LZGGraph *g = NULL;
    LZGError err = lzg_flashback_stream_finalize(s, &g);
    /* Stream struct stays alive; the capsule destructor will free it.
     * The graph (on success) is now ours to wrap in a separate capsule. */
    if (err != LZG_OK) {
        return set_lzg_error(err);
    }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_fb_stream_abort(PyObject *self, PyObject *cap) {
    (void)self;
    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        cap, FB_STREAM_CAPSULE_NAME);
    if (s) lzg_flashback_stream_abort(s);
    Py_RETURN_NONE;
}

static PyObject *py_fb_stream_snapshot(PyObject *self, PyObject *cap) {
    (void)self;
    LZGFlashbackStream *s = (LZGFlashbackStream *)PyCapsule_GetPointer(
        cap, FB_STREAM_CAPSULE_NAME);
    if (!s) return NULL;

    LZGGraph *g = NULL;
    LZGError err = lzg_flashback_stream_snapshot(s, &g);
    if (err != LZG_OK) {
        return set_lzg_error(err);
    }
    /* Stream remains alive; we just hand back a new graph that borrows
     * the stream's string pool via refcount. */
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_fb_simulate(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;
    static char *kwlist[] = {"graph", "n", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|L", kwlist, &cap, &n, &seed))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0xFB0CAFE));

    LZGSimResult *results = calloc(n, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();

    LZGError err = lzg_flashback_simulate(g, n, &rng, results);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *sl = PyList_New(n), *lp = PyList_New(n), *nt = PyList_New(n);
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(sl, i, PyUnicode_FromString(results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lp, i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nt, i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(OOO)", sl, lp, nt);
}

static PyObject *py_fb_pgen(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_arg;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seq_arg)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    if (PyUnicode_Check(seq_arg)) {
        const char *seq = PyUnicode_AsUTF8(seq_arg);
        if (!seq) return NULL;
        return PyFloat_FromDouble(lzg_flashback_pgen(g, seq, (uint32_t)strlen(seq)));
    }
    if (PyList_Check(seq_arg)) {
        Py_ssize_t n = PyList_GET_SIZE(seq_arg);
        PyObject *result = PyList_New(n);
        for (Py_ssize_t i = 0; i < n; i++) {
            const char *seq = PyUnicode_AsUTF8(PyList_GET_ITEM(seq_arg, i));
            if (!seq) { Py_DECREF(result); return NULL; }
            double lp = lzg_flashback_pgen(g, seq, (uint32_t)strlen(seq));
            PyList_SET_ITEM(result, i, PyFloat_FromDouble(lp));
        }
        return result;
    }
    PyErr_SetString(PyExc_TypeError, "sequence must be str or list[str]");
    return NULL;
}


static PyObject *py_fb_top_k_walks(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int k = 100;
    int most_probable = 1;
    static char *kwlist[] = {"graph", "k", "most_probable", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Ip", kwlist, &cap, &k, &most_probable))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGSimResult *results = calloc(k, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();

    uint32_t actual_k = 0;
    LZGError err = lzg_flashback_top_k_walks(g, k, (bool)most_probable, results, &actual_k);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *sl = PyList_New(actual_k);
    PyObject *lp = PyList_New(actual_k);
    for (uint32_t i = 0; i < actual_k; i++) {
        PyList_SET_ITEM(sl, i, PyUnicode_FromString(results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lp, i, PyFloat_FromDouble(results[i].log_prob));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(OO)", sl, lp);
}

static PyObject *py_fb_path_count(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    double count;
    LZGError err = lzg_flashback_path_count(g, &count);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(count);
}

static PyObject *py_fb_path_count_exact(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    uint32_t *limbs = NULL;
    uint32_t n = 0;
    LZGError err = lzg_flashback_path_count_exact(g, &limbs, &n);
    if (err != LZG_OK) return set_lzg_error(err);
    if (n == 0) {
        free(limbs);
        return PyLong_FromLong(0);
    }
    /* Render big-endian hex, then let CPython parse it. Base 16 is exempt
       from the sys.set_int_max_str_digits() limit that applies to base 10. */
    size_t len = (size_t)n * 8 + 1;
    char *buf = (char *)malloc(len);
    if (!buf) {
        free(limbs);
        return PyErr_NoMemory();
    }
    char *p = buf;
    p += snprintf(p, len, "%" PRIx32, limbs[n - 1]);
    for (uint32_t i = n - 1; i > 0; i--)
        p += snprintf(p, len - (size_t)(p - buf), "%08" PRIx32, limbs[i - 1]);
    PyObject *res = PyLong_FromString(buf, NULL, 16);
    free(buf);
    free(limbs);
    return res;
}

static PyObject *py_fb_path_count_by_length(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    double *counts = NULL;
    uint32_t max_length = 0;
    LZGError err = lzg_flashback_path_count_by_length(
        g, &counts, &max_length);
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *result = PyDict_New();
    if (!result) { free(counts); return NULL; }
    for (uint32_t length = 0; length <= max_length; length++) {
        if (counts[length] == 0.0) continue;
        PyObject *key = PyLong_FromUnsignedLong(length);
        PyObject *value = PyFloat_FromDouble(counts[length]);
        if (!key || !value || PyDict_SetItem(result, key, value) < 0) {
            Py_XDECREF(key); Py_XDECREF(value);
            Py_DECREF(result); free(counts);
            return NULL;
        }
        Py_DECREF(key);
        Py_DECREF(value);
    }
    free(counts);
    return result;
}

static PyObject *readonly_memoryview(void *data, size_t n_bytes) {
    if (n_bytes > (size_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "graph array is too large");
        return NULL;
    }
    if (n_bytes == 0)
        return PyMemoryView_FromMemory("", 0, PyBUF_READ);
    return PyMemoryView_FromMemory((char *)data, (Py_ssize_t)n_bytes, PyBUF_READ);
}

typedef struct {
    PyObject_HEAD
    PyObject *owner;
    void *data;
    Py_ssize_t length;
} OwnedReadonlyBuffer;

static int owned_readonly_buffer_getbuffer(
    PyObject *exporter, Py_buffer *view, int flags) {
    OwnedReadonlyBuffer *buffer = (OwnedReadonlyBuffer *)exporter;
    return PyBuffer_FillInfo(
        view, exporter, buffer->length ? buffer->data : (void *)"",
        buffer->length, 1, flags);
}

static void owned_readonly_buffer_dealloc(PyObject *exporter) {
    OwnedReadonlyBuffer *buffer = (OwnedReadonlyBuffer *)exporter;
    Py_XDECREF(buffer->owner);
    Py_TYPE(exporter)->tp_free(exporter);
}

static PyBufferProcs owned_readonly_buffer_procs = {
    .bf_getbuffer = owned_readonly_buffer_getbuffer,
};

static PyTypeObject OwnedReadonlyBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "LZGraphs._clzgraph._OwnedReadonlyBuffer",
    .tp_basicsize = sizeof(OwnedReadonlyBuffer),
    .tp_dealloc = owned_readonly_buffer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_buffer = &owned_readonly_buffer_procs,
};

/* Return a read-only view which keeps owner alive for as long as the view.
 * PyMemoryView_FromMemory has no base object, so it is only suitable for
 * arrays whose lifetime is already tied to a graph capsule. */
static PyObject *owned_readonly_memoryview(
    PyObject *owner, void *data, size_t n_bytes) {
    if (n_bytes > (size_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "native array is too large");
        return NULL;
    }
    OwnedReadonlyBuffer *buffer = (OwnedReadonlyBuffer *)
        OwnedReadonlyBufferType.tp_alloc(&OwnedReadonlyBufferType, 0);
    if (!buffer) return NULL;
    Py_INCREF(owner);
    buffer->owner = owner;
    buffer->data = data;
    buffer->length = (Py_ssize_t)n_bytes;
    PyObject *view = PyMemoryView_FromObject((PyObject *)buffer);
    Py_DECREF(buffer);
    return view;
}

static PyObject *py_fb_pseq_structure(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    PyObject *symbol_lengths = PyBytes_FromStringAndSize(NULL, g->n_nodes);
    PyObject *sinks = PyBytes_FromStringAndSize(NULL, g->n_nodes);
    if (!symbol_lengths || !sinks) {
        Py_XDECREF(symbol_lengths); Py_XDECREF(sinks);
        return NULL;
    }
    uint8_t *sink_data = (uint8_t *)PyBytes_AS_STRING(sinks);
    for (uint32_t u = 0; u < g->n_nodes; u++)
        sink_data[u] = g->row_offsets[u] == g->row_offsets[u + 1];
    double min_surprisal, max_surprisal;
    uint32_t max_edges;
    LZGError err = lzg_flashback_pseq_init(
        g, (uint8_t *)PyBytes_AS_STRING(symbol_lengths),
        &min_surprisal, &max_surprisal, &max_edges);
    if (err != LZG_OK) {
        Py_DECREF(symbol_lengths); Py_DECREF(sinks);
        return set_lzg_error(err);
    }

    PyObject *row = readonly_memoryview(
        g->row_offsets, ((size_t)g->n_nodes + 1) * sizeof(uint32_t));
    PyObject *col = readonly_memoryview(
        g->col_indices, (size_t)g->n_edges * sizeof(uint32_t));
    PyObject *weights = readonly_memoryview(
        g->edge_weights, (size_t)g->n_edges * sizeof(double));
    PyObject *topological_order = readonly_memoryview(
        g->topo_order, (size_t)g->n_nodes * sizeof(uint32_t));
    if (!row || !col || !weights || !topological_order) {
        Py_XDECREF(row); Py_XDECREF(col); Py_XDECREF(weights);
        Py_XDECREF(topological_order);
        Py_DECREF(symbol_lengths); Py_DECREF(sinks);
        return NULL;
    }

    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:N,s:N,s:I,s:d,s:d,s:I}",
        "row_offsets", row,
        "col_indices", col,
        "weights", weights,
        "topological_order", topological_order,
        "sink_mask", sinks,
        "symbol_lengths", symbol_lengths,
        "root", g->root_node,
        "min_surprisal", min_surprisal,
        "max_surprisal", max_surprisal,
        "max_edges", max_edges);
}

static PyObject *py_fb_pseq_length_derivatives(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double q;
    unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    double *derivatives = NULL;
    uint8_t *present = NULL;
    uint32_t max_length = 0;
    LZGError err = lzg_flashback_pseq_length_derivatives(
        g, q, order, &derivatives, &present, &max_length);
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *result = PyDict_New();
    if (!result) {
        free(derivatives); free(present);
        return NULL;
    }
    const size_t dim = (size_t)order + 1;
    for (uint32_t length = 0; length <= max_length; length++) {
        if (!present[length]) continue;
        PyObject *key = PyLong_FromUnsignedLong(length);
        PyObject *jet = PyList_New((Py_ssize_t)dim);
        if (!key || !jet) {
            Py_XDECREF(key); Py_XDECREF(jet); Py_DECREF(result);
            free(derivatives); free(present);
            return NULL;
        }
        for (size_t r = 0; r < dim; r++) {
            PyObject *value = PyFloat_FromDouble(
                derivatives[(size_t)length * dim + r]);
            if (!value) {
                Py_DECREF(key); Py_DECREF(jet); Py_DECREF(result);
                free(derivatives); free(present);
                return NULL;
            }
            PyList_SET_ITEM(jet, (Py_ssize_t)r, value);
        }
        if (PyDict_SetItem(result, key, jet) < 0) {
            Py_DECREF(key); Py_DECREF(jet); Py_DECREF(result);
            free(derivatives); free(present);
            return NULL;
        }
        Py_DECREF(key);
        Py_DECREF(jet);
    }
    free(derivatives); free(present);
    return result;
}

static PyObject *py_fb_pseq_derivatives(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double q;
    unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    double derivatives[9];
    LZGError err = lzg_flashback_pseq_derivatives(
        g, q, order, derivatives);
    if (err != LZG_OK) return set_lzg_error(err);
    PyObject *result = PyList_New((Py_ssize_t)order + 1);
    if (!result) return NULL;
    for (uint32_t r = 0; r <= order; r++) {
        PyObject *value = PyFloat_FromDouble(derivatives[r]);
        if (!value) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)r, value);
    }
    return result;
}

static PyObject *py_fb_pseq_tilted_moments(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double q;
    unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    if (order > 8) {
        PyErr_SetString(PyExc_ValueError, "order must be between 0 and 8");
        return NULL;
    }

    double log_mass, raw[9], central[9];
    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_pseq_tilted_moments(
        g, q, order, &log_mass, raw, central);
    Py_END_ALLOW_THREADS
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *raw_list = PyList_New((Py_ssize_t)order + 1);
    PyObject *central_list = PyList_New((Py_ssize_t)order + 1);
    if (!raw_list || !central_list) {
        Py_XDECREF(raw_list); Py_XDECREF(central_list);
        return NULL;
    }
    for (uint32_t r = 0; r <= order; r++) {
        PyObject *raw_value = PyFloat_FromDouble(raw[r]);
        PyObject *central_value = PyFloat_FromDouble(central[r]);
        if (!raw_value || !central_value) {
            Py_XDECREF(raw_value); Py_XDECREF(central_value);
            Py_DECREF(raw_list); Py_DECREF(central_list);
            return NULL;
        }
        PyList_SET_ITEM(raw_list, (Py_ssize_t)r, raw_value);
        PyList_SET_ITEM(central_list, (Py_ssize_t)r, central_value);
    }
    return Py_BuildValue(
        "{s:d,s:N,s:N}",
        "log_mass", log_mass,
        "raw_moments", raw_list,
        "central_moments", central_list);
}

static PyObject *py_fb_pseq_saddlepoint_batch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *values_obj;
    if (!PyArg_ParseTuple(args, "OO", &cap, &values_obj)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    PyObject *values = PySequence_Fast(values_obj, "x must be a sequence");
    if (!values) return NULL;
    Py_ssize_t n_py = PySequence_Fast_GET_SIZE(values);
    if (n_py < 0 || (uint64_t)n_py > UINT32_MAX) {
        Py_DECREF(values);
        PyErr_SetString(PyExc_OverflowError, "too many saddlepoint values");
        return NULL;
    }
    const uint32_t n = (uint32_t)n_py;
    double *x = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *pdf = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *cdf = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *saddle = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    uint32_t *iterations = n ? (uint32_t *)malloc(
        (size_t)n * sizeof(uint32_t)) : NULL;
    if (n && (!x || !pdf || !cdf || !saddle || !iterations)) {
        free(x); free(pdf); free(cdf); free(saddle); free(iterations);
        Py_DECREF(values);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(values);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            free(x); free(pdf); free(cdf); free(saddle); free(iterations);
            Py_DECREF(values);
            return NULL;
        }
    }
    Py_DECREF(values);

    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_pseq_saddlepoint_batch(
        g, x, n, pdf, cdf, saddle, iterations);
    Py_END_ALLOW_THREADS
    free(x);
    if (err != LZG_OK) {
        free(pdf); free(cdf); free(saddle); free(iterations);
        return set_lzg_error(err);
    }

    PyObject *pdf_list = PyList_New(n_py);
    PyObject *cdf_list = PyList_New(n_py);
    PyObject *saddle_list = PyList_New(n_py);
    PyObject *iteration_list = PyList_New(n_py);
    if (!pdf_list || !cdf_list || !saddle_list || !iteration_list) {
        Py_XDECREF(pdf_list); Py_XDECREF(cdf_list);
        Py_XDECREF(saddle_list); Py_XDECREF(iteration_list);
        free(pdf); free(cdf); free(saddle); free(iterations);
        return NULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        PyObject *pdf_value = PyFloat_FromDouble(pdf[i]);
        PyObject *cdf_value = PyFloat_FromDouble(cdf[i]);
        PyObject *saddle_value = PyFloat_FromDouble(saddle[i]);
        PyObject *iteration_value = PyLong_FromUnsignedLong(iterations[i]);
        if (!pdf_value || !cdf_value || !saddle_value || !iteration_value) {
            Py_XDECREF(pdf_value); Py_XDECREF(cdf_value);
            Py_XDECREF(saddle_value); Py_XDECREF(iteration_value);
            Py_DECREF(pdf_list); Py_DECREF(cdf_list);
            Py_DECREF(saddle_list); Py_DECREF(iteration_list);
            free(pdf); free(cdf); free(saddle); free(iterations);
            return NULL;
        }
        PyList_SET_ITEM(pdf_list, i, pdf_value);
        PyList_SET_ITEM(cdf_list, i, cdf_value);
        PyList_SET_ITEM(saddle_list, i, saddle_value);
        PyList_SET_ITEM(iteration_list, i, iteration_value);
    }
    free(pdf); free(cdf); free(saddle); free(iterations);
    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N}",
        "pdf", pdf_list,
        "cdf", cdf_list,
        "saddle", saddle_list,
        "iterations", iteration_list);
}

static PyObject *py_fb_edge_threshold_diversity(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *thresholds_obj;
    if (!PyArg_ParseTuple(args, "OO", &cap, &thresholds_obj)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    PyObject *values = PySequence_Fast(
        thresholds_obj, "thresholds must be a sequence");
    if (!values) return NULL;
    const Py_ssize_t n_py = PySequence_Fast_GET_SIZE(values);
    if (n_py < 0 || (uint64_t)n_py > UINT32_MAX) {
        Py_DECREF(values);
        PyErr_SetString(PyExc_OverflowError, "too many edge thresholds");
        return NULL;
    }
    const uint32_t n = (uint32_t)n_py;
    double *thresholds = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *log_d0 = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *log_d1 = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *log_d2 = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *mass = n ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    uint64_t *kept = n ? (uint64_t *)malloc((size_t)n * sizeof(uint64_t)) : NULL;
    if (n && (!thresholds || !log_d0 || !log_d1 || !log_d2 || !mass || !kept)) {
        free(thresholds); free(log_d0); free(log_d1);
        free(log_d2); free(mass); free(kept);
        Py_DECREF(values);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(values);
    for (uint32_t i = 0; i < n; i++) {
        thresholds[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            free(thresholds); free(log_d0); free(log_d1);
            free(log_d2); free(mass); free(kept);
            Py_DECREF(values);
            return NULL;
        }
    }
    Py_DECREF(values);

    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_edge_threshold_diversity(
        g, thresholds, n, log_d0, log_d1, log_d2, mass, kept);
    Py_END_ALLOW_THREADS
    free(thresholds);
    if (err != LZG_OK) {
        free(log_d0); free(log_d1); free(log_d2); free(mass); free(kept);
        return set_lzg_error(err);
    }

    PyObject *d0_list = PyList_New(n_py);
    PyObject *d1_list = PyList_New(n_py);
    PyObject *d2_list = PyList_New(n_py);
    PyObject *mass_list = PyList_New(n_py);
    PyObject *kept_list = PyList_New(n_py);
    if (!d0_list || !d1_list || !d2_list || !mass_list || !kept_list) {
        Py_XDECREF(d0_list); Py_XDECREF(d1_list); Py_XDECREF(d2_list);
        Py_XDECREF(mass_list); Py_XDECREF(kept_list);
        free(log_d0); free(log_d1); free(log_d2); free(mass); free(kept);
        return NULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        PyObject *d0_value = PyFloat_FromDouble(log_d0[i]);
        PyObject *d1_value = PyFloat_FromDouble(log_d1[i]);
        PyObject *d2_value = PyFloat_FromDouble(log_d2[i]);
        PyObject *mass_value = PyFloat_FromDouble(mass[i]);
        PyObject *kept_value = PyLong_FromUnsignedLongLong(kept[i]);
        if (!d0_value || !d1_value || !d2_value || !mass_value || !kept_value) {
            Py_XDECREF(d0_value); Py_XDECREF(d1_value); Py_XDECREF(d2_value);
            Py_XDECREF(mass_value); Py_XDECREF(kept_value);
            Py_DECREF(d0_list); Py_DECREF(d1_list); Py_DECREF(d2_list);
            Py_DECREF(mass_list); Py_DECREF(kept_list);
            free(log_d0); free(log_d1); free(log_d2); free(mass); free(kept);
            return NULL;
        }
        PyList_SET_ITEM(d0_list, i, d0_value);
        PyList_SET_ITEM(d1_list, i, d1_value);
        PyList_SET_ITEM(d2_list, i, d2_value);
        PyList_SET_ITEM(mass_list, i, mass_value);
        PyList_SET_ITEM(kept_list, i, kept_value);
    }
    free(log_d0); free(log_d1); free(log_d2); free(mass); free(kept);
    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:N}",
        "log_d0", d0_list,
        "log_d1", d1_list,
        "log_d2", d2_list,
        "surviving_mass", mass_list,
        "kept_edges", kept_list);
}

#define PSEQ_ATTRIBUTION_CAPSULE_NAME "LZGraphs.PseqAttribution"

static void pseq_attribution_capsule_destructor(PyObject *capsule) {
    LZGPseqAttribution *result = (LZGPseqAttribution *)PyCapsule_GetPointer(
        capsule, PSEQ_ATTRIBUTION_CAPSULE_NAME);
    if (!result) {
        PyErr_Clear();
        return;
    }
    lzg_flashback_pseq_attribution_destroy(result);
    free(result);
}

static PyObject *py_fb_pseq_attribution(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGPseqAttribution *result = (LZGPseqAttribution *)calloc(
        1, sizeof(LZGPseqAttribution));
    if (!result) return PyErr_NoMemory();
    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_pseq_attribution(g, q, result);
    Py_END_ALLOW_THREADS
    if (err != LZG_OK) {
        free(result);
        return set_lzg_error(err);
    }

    PyObject *owner = PyCapsule_New(
        result, PSEQ_ATTRIBUTION_CAPSULE_NAME,
        pseq_attribution_capsule_destructor);
    if (!owner) {
        lzg_flashback_pseq_attribution_destroy(result);
        free(result);
        return NULL;
    }
    PyObject *node_probability = owned_readonly_memoryview(
        owner,
        result->node_probability,
        (size_t)result->n_nodes * sizeof(double));
    PyObject *edge_probability = owned_readonly_memoryview(
        owner,
        result->edge_probability,
        (size_t)result->n_edges * sizeof(double));
    if (!node_probability || !edge_probability) {
        Py_XDECREF(node_probability); Py_XDECREF(edge_probability);
        Py_DECREF(owner);
        return NULL;
    }
    return Py_BuildValue(
        "{s:d,s:d,s:N,s:N,s:N}",
        "q", result->q,
        "log_mass", result->log_mass,
        "node_probability", node_probability,
        "edge_probability", edge_probability,
        "owner", owner);
}

static PyObject *py_fb_pseq_histogram(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int bins;
    double q;
    long long length;
    if (!PyArg_ParseTuple(args, "OIdL", &cap, &bins, &q, &length)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    double *weights = NULL;
    double spacing, true_max_surprisal;
    uint32_t max_edges;
    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_pseq_histogram(
        g, bins, q, (int64_t)length, &weights, &spacing,
        &true_max_surprisal, &max_edges);
    Py_END_ALLOW_THREADS
    if (err != LZG_OK) return set_lzg_error(err);
    PyObject *values = PyList_New(bins);
    if (!values) {
        free(weights);
        return NULL;
    }
    for (uint32_t i = 0; i < bins; i++) {
        PyObject *value = PyFloat_FromDouble(weights[i]);
        if (!value) {
            Py_DECREF(values); free(weights);
            return NULL;
        }
        PyList_SET_ITEM(values, i, value);
    }
    free(weights);
    return Py_BuildValue(
        "{s:N,s:d,s:d,s:I}",
        "weights", values,
        "spacing", spacing,
        "true_max_surprisal", true_max_surprisal,
        "max_edges", max_edges);
}

static PyObject *py_fb_pseq_histogram_pair(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int bins;
    if (!PyArg_ParseTuple(args, "OI", &cap, &bins)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    double *counting = NULL, *generated = NULL;
    double spacing, true_max_surprisal;
    uint32_t max_edges;
    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_pseq_histogram_pair(
        g, bins, &counting, &generated, &spacing,
        &true_max_surprisal, &max_edges);
    Py_END_ALLOW_THREADS
    if (err != LZG_OK) return set_lzg_error(err);

    PyObject *counting_values = PyList_New(bins);
    PyObject *generated_values = PyList_New(bins);
    if (!counting_values || !generated_values) {
        Py_XDECREF(counting_values); Py_XDECREF(generated_values);
        free(counting); free(generated);
        return NULL;
    }
    for (uint32_t i = 0; i < bins; i++) {
        PyObject *counting_value = PyFloat_FromDouble(counting[i]);
        PyObject *generated_value = PyFloat_FromDouble(generated[i]);
        if (!counting_value || !generated_value) {
            Py_XDECREF(counting_value); Py_XDECREF(generated_value);
            Py_DECREF(counting_values); Py_DECREF(generated_values);
            free(counting); free(generated);
            return NULL;
        }
        PyList_SET_ITEM(counting_values, i, counting_value);
        PyList_SET_ITEM(generated_values, i, generated_value);
    }
    free(counting); free(generated);
    return Py_BuildValue(
        "{s:N,s:N,s:d,s:d,s:I}",
        "counting_weights", counting_values,
        "generated_weights", generated_values,
        "spacing", spacing,
        "true_max_surprisal", true_max_surprisal,
        "max_edges", max_edges);
}

static PyObject *py_fb_effective_diversity(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGEffectiveDiversity ed;
    LZGError err = lzg_flashback_effective_diversity(g, &ed);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "entropy_nats", ed.entropy_nats, "entropy_bits", ed.entropy_bits,
        "effective_diversity", ed.effective_diversity, "uniformity", ed.uniformity);
}

static PyObject *py_fb_power_sum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double m;
    LZGError err = lzg_flashback_power_sum(g, alpha, &m);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(m);
}

static PyObject *py_fb_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double d;
    LZGError err = lzg_flashback_hill_number(g, alpha, &d);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(d);
}

static PyObject *py_fb_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &orders_list)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    Py_ssize_t n = PyList_GET_SIZE(orders_list);
    double *orders = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    if (!orders || !out) { free(orders); free(out); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++)
        orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_list, i));
    LZGError err = lzg_flashback_hill_numbers(g, orders, (uint32_t)n, out);
    free(orders);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }
    PyObject *result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

static PyObject *py_fb_dynamic_range(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_flashback_dynamic_range(g, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "max_log_prob", dr.max_log_prob, "min_log_prob", dr.min_log_prob,
        "dynamic_range_nats", dr.dynamic_range_nats,
        "dynamic_range_orders", dr.dynamic_range_orders);
}

static PyObject *py_fb_pgen_diagnostics(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double atol;
    if (!PyArg_ParseTuple(args, "Od", &cap, &atol)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenDiagnostics diag;
    LZGError err = lzg_flashback_pgen_diagnostics(g, atol, &diag);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:O,s:I}",
        "total_absorbed", diag.total_absorbed, "total_leaked", diag.total_leaked,
        "initial_prob_sum", diag.initial_prob_sum,
        "is_proper", diag.is_proper ? Py_True : Py_False,
        "mc_samples", diag.mc_samples);
}

/* ── fb_fix_special_nodes(capsule) → None ──────────────────── */

/* ── fb_posterior(cap, sequences, abundances, kappa) → cap ───── */

static PyObject *py_fb_posterior(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap, *seq_list;
    PyObject *abund_obj = Py_None;
    double kappa = 1.0;

    static char *kwlist[] = {"graph", "sequences", "abundances", "kappa", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|Od", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj, &kappa))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(seq_list);
    uint32_t n_u32 = 0;
    if (!pyssize_to_u32(n, "sequences", &n_u32)) return NULL;

    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *post = NULL;
    LZGError err = lzg_flashback_graph_posterior(
        g, seqs, n_u32, abundances, kappa, &post);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(post, CAPSULE_NAME, capsule_destructor);
}

/* ── fb_subtract(cap, sequences, abundances) → cap ───── */

static PyObject *py_fb_subtract(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap, *seq_list;
    PyObject *abund_obj = Py_None;

    static char *kwlist[] = {"graph", "sequences", "abundances", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|O", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(seq_list);
    uint32_t n_u32 = 0;
    if (!pyssize_to_u32(n, "sequences", &n_u32)) return NULL;

    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *sub = NULL;
    LZGError err = lzg_flashback_graph_subtract(
        g, seqs, n_u32, abundances, &sub);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(sub, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_fb_fix_special_nodes(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    lzg_flashback_fix_special_nodes(g);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════ */
/* NaiveGraph — positional "{AA}_{pos}" encoding                   */
/* ══════════════════════════════════════════════════════════════ */

static PyObject *py_naive_decompose(PyObject *self, PyObject *arg) {
    (void)self;
    const char *seq = PyUnicode_AsUTF8(arg);
    if (!seq) return NULL;
    uint32_t len = (uint32_t)strlen(seq);

    LZGStringPool *pool = lzg_sp_create(len + 8u);
    if (!pool) return PyErr_NoMemory();

    uint32_t ids[LZG_NAIVE_MAX_WALK];
    uint32_t count = 0;
    LZGError err = lzg_naive_encode(seq, len, pool, ids, &count);
    if (err != LZG_OK) { lzg_sp_destroy(pool); return set_lzg_error(err); }

    PyObject *list = PyList_New(count);
    if (!list) { lzg_sp_destroy(pool); return NULL; }
    for (uint32_t i = 0; i < count; i++) {
        PyObject *s = PyUnicode_FromString(lzg_sp_get(pool, ids[i]));
        if (!s) { Py_DECREF(list); lzg_sp_destroy(pool); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    lzg_sp_destroy(pool);
    return list;
}

static PyObject *py_naive_graph_build(PyObject *self, PyObject *args,
                                      PyObject *kw) {
    (void)self;
    PyObject *seq_list, *abund_obj = Py_None;
    unsigned int max_length = 0;
    double smoothing = 0.0;
    static char *kwlist[] = {"sequences", "abundances", "max_length",
                             "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|OId", kwlist,
            &PyList_Type, &seq_list, &abund_obj, &max_length, &smoothing))
        return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_seqs_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_seqs_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE_POS);
    if (!g) { free(seqs); free(abundances); return PyErr_NoMemory(); }

    LZGError err = lzg_naive_graph_build(g, seqs, n_seqs_u32, abundances,
                                         (uint32_t)max_length, smoothing);
    free(seqs); free(abundances);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_naive_graph_build_file(PyObject *self, PyObject *args,
                                           PyObject *kw) {
    (void)self;
    const char *path = NULL;
    unsigned int max_length = 0;
    double smoothing = 0.0;
    static char *kwlist[] = {"path", "max_length", "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|Id", kwlist, &path,
                                     &max_length, &smoothing))
        return NULL;

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_NAIVE_POS);
    if (!g) return PyErr_NoMemory();
    LZGError err = lzg_naive_graph_build_file(g, path, (uint32_t)max_length,
                                              smoothing);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_naive_pgen(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_arg;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seq_arg)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    if (PyUnicode_Check(seq_arg)) {
        const char *seq = PyUnicode_AsUTF8(seq_arg);
        if (!seq) return NULL;
        return PyFloat_FromDouble(
            lzg_naive_pgen(g, seq, (uint32_t)strlen(seq)));
    }
    if (PyList_Check(seq_arg)) {
        Py_ssize_t n = PyList_GET_SIZE(seq_arg);
        PyObject *result = PyList_New(n);
        if (!result) return NULL;
        for (Py_ssize_t i = 0; i < n; i++) {
            const char *seq = PyUnicode_AsUTF8(PyList_GET_ITEM(seq_arg, i));
            if (!seq) { Py_DECREF(result); return NULL; }
            double lp = lzg_naive_pgen(g, seq, (uint32_t)strlen(seq));
            PyList_SET_ITEM(result, i, PyFloat_FromDouble(lp));
        }
        return result;
    }
    PyErr_SetString(PyExc_TypeError, "sequence must be str or list[str]");
    return NULL;
}

static PyObject *py_naive_simulate(PyObject *self, PyObject *args,
                                   PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;
    static char *kwlist[] = {"graph", "n", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|L", kwlist, &cap, &n, &seed))
        return NULL;

    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0x0A11ECAFEULL));

    LZGSimResult *results = calloc(n, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();

    LZGError err = lzg_naive_simulate(g, n, &rng, results);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *sl = PyList_New(n), *lp = PyList_New(n), *nt = PyList_New(n);
    if (!sl || !lp || !nt) {
        Py_XDECREF(sl); Py_XDECREF(lp); Py_XDECREF(nt);
        for (unsigned int i = 0; i < n; i++) lzg_sim_result_free(&results[i]);
        free(results);
        return NULL;
    }
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(sl, i, PyUnicode_FromString(
            results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lp, i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nt, i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(NNN)", sl, lp, nt);
}

static PyObject *py_naive_path_count(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    double count;
    LZGError err = lzg_naive_path_count(g, &count);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(count);
}

static PyObject *py_naive_path_count_exact(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    uint32_t *limbs = NULL;
    uint32_t n = 0;
    LZGError err = lzg_naive_path_count_exact(g, &limbs, &n);
    if (err != LZG_OK) return set_lzg_error(err);
    if (n == 0) { free(limbs); return PyLong_FromLong(0); }
    /* Render big-endian hex, then let CPython parse it. Base 16 is exempt
       from the sys.set_int_max_str_digits() limit that applies to base 10. */
    size_t len = (size_t)n * 8 + 1;
    char *buf = (char *)malloc(len);
    if (!buf) { free(limbs); return PyErr_NoMemory(); }
    char *p = buf;
    p += snprintf(p, len, "%" PRIx32, limbs[n - 1]);
    for (uint32_t i = n - 1; i > 0; i--)
        p += snprintf(p, len - (size_t)(p - buf), "%08" PRIx32, limbs[i - 1]);
    PyObject *res = PyLong_FromString(buf, NULL, 16);
    free(buf);
    free(limbs);
    return res;
}

static PyObject *py_naive_effective_diversity(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGEffectiveDiversity ed;
    LZGError err = lzg_naive_effective_diversity(g, &ed);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "entropy_nats", ed.entropy_nats, "entropy_bits", ed.entropy_bits,
        "effective_diversity", ed.effective_diversity,
        "uniformity", ed.uniformity);
}

static PyObject *py_naive_power_sum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double m;
    LZGError err = lzg_naive_power_sum(g, alpha, &m);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(m);
}

static PyObject *py_naive_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double d;
    LZGError err = lzg_naive_hill_number(g, alpha, &d);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(d);
}

static PyObject *py_naive_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &orders_list))
        return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    Py_ssize_t n = PyList_GET_SIZE(orders_list);
    double *orders = malloc((size_t)n * sizeof(double));
    double *out = malloc((size_t)n * sizeof(double));
    if (!orders || !out) { free(orders); free(out); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++)
        orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_list, i));
    LZGError err = lzg_naive_hill_numbers(g, orders, (uint32_t)n, out);
    free(orders);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }
    PyObject *result = PyList_New(n);
    if (!result) { free(out); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

static PyObject *py_naive_dynamic_range(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_naive_dynamic_range(g, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "max_log_prob", dr.max_log_prob, "min_log_prob", dr.min_log_prob,
        "dynamic_range_nats", dr.dynamic_range_nats,
        "dynamic_range_orders", dr.dynamic_range_orders);
}

static PyObject *py_naive_pgen_diagnostics(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double atol;
    if (!PyArg_ParseTuple(args, "Od", &cap, &atol)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenDiagnostics diag;
    LZGError err = lzg_naive_pgen_diagnostics(g, atol, &diag);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:O,s:I}",
        "total_absorbed", diag.total_absorbed,
        "total_leaked", diag.total_leaked,
        "initial_prob_sum", diag.initial_prob_sum,
        "is_proper", diag.is_proper ? Py_True : Py_False,
        "mc_samples", diag.mc_samples);
}


/* ══════════════════════════════════════════════════════════════ */
/* FlattenedFlashBack — bilateral scan without run compression     */
/* ══════════════════════════════════════════════════════════════ */

static PyObject *py_flat_decompose(PyObject *self, PyObject *arg) {
    (void)self;
    const char *seq = PyUnicode_AsUTF8(arg);
    if (!seq) return NULL;
    uint32_t len = (uint32_t)strlen(seq);

    LZGStringPool *pool = lzg_sp_create(len + 8u);
    if (!pool) return PyErr_NoMemory();

    uint32_t ids[LZG_FLAT_MAX_WALK];
    uint32_t count = 0;
    LZGError err = lzg_flat_encode(seq, len, pool, ids, &count);
    if (err != LZG_OK) { lzg_sp_destroy(pool); return set_lzg_error(err); }

    PyObject *list = PyList_New(count);
    if (!list) { lzg_sp_destroy(pool); return NULL; }
    for (uint32_t i = 0; i < count; i++) {
        PyObject *s = PyUnicode_FromString(lzg_sp_get(pool, ids[i]));
        if (!s) { Py_DECREF(list); lzg_sp_destroy(pool); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    lzg_sp_destroy(pool);
    return list;
}

static PyObject *py_flat_graph_build(PyObject *self, PyObject *args,
                                     PyObject *kw) {
    (void)self;
    PyObject *seq_list, *abund_obj = Py_None;
    unsigned int max_length = 0;
    double smoothing = 0.0;
    static char *kwlist[] = {"sequences", "abundances", "max_length",
                             "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|OId", kwlist,
            &PyList_Type, &seq_list, &abund_obj, &max_length, &smoothing))
        return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGGraph *g = lzg_graph_create(LZG_VARIANT_FLAT_FB);
    if (!g) { free(seqs); free(abundances); return PyErr_NoMemory(); }
    LZGError err = lzg_flat_graph_build(g, seqs, n_u32, abundances,
                                        (uint32_t)max_length, smoothing);
    free(seqs); free(abundances);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_flat_graph_build_file(PyObject *self, PyObject *args,
                                          PyObject *kw) {
    (void)self;
    const char *path = NULL;
    unsigned int max_length = 0;
    double smoothing = 0.0;
    static char *kwlist[] = {"path", "max_length", "smoothing", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|Id", kwlist, &path,
                                     &max_length, &smoothing))
        return NULL;
    LZGGraph *g = lzg_graph_create(LZG_VARIANT_FLAT_FB);
    if (!g) return PyErr_NoMemory();
    LZGError err = lzg_flat_graph_build_file(g, path, (uint32_t)max_length,
                                             smoothing);
    if (err != LZG_OK) { lzg_graph_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_flat_pseq(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seq_arg;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seq_arg)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    if (PyUnicode_Check(seq_arg)) {
        const char *seq = PyUnicode_AsUTF8(seq_arg);
        if (!seq) return NULL;
        return PyFloat_FromDouble(lzg_flat_pseq(g, seq, (uint32_t)strlen(seq)));
    }
    if (PyList_Check(seq_arg)) {
        Py_ssize_t n = PyList_GET_SIZE(seq_arg);
        PyObject *result = PyList_New(n);
        if (!result) return NULL;
        for (Py_ssize_t i = 0; i < n; i++) {
            const char *seq = PyUnicode_AsUTF8(PyList_GET_ITEM(seq_arg, i));
            if (!seq) { Py_DECREF(result); return NULL; }
            PyList_SET_ITEM(result, i, PyFloat_FromDouble(
                lzg_flat_pseq(g, seq, (uint32_t)strlen(seq))));
        }
        return result;
    }
    PyErr_SetString(PyExc_TypeError, "sequence must be str or list[str]");
    return NULL;
}

static PyObject *py_flat_simulate(PyObject *self, PyObject *args,
                                  PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;
    static char *kwlist[] = {"graph", "n", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|L", kwlist, &cap, &n, &seed))
        return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0x0F1A7CAFEULL));

    LZGSimResult *results = calloc(n, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();
    LZGError err = lzg_flat_simulate(g, n, &rng, results);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *sl = PyList_New(n), *lp = PyList_New(n), *nt = PyList_New(n);
    if (!sl || !lp || !nt) {
        Py_XDECREF(sl); Py_XDECREF(lp); Py_XDECREF(nt);
        for (unsigned int i = 0; i < n; i++) lzg_sim_result_free(&results[i]);
        free(results);
        return NULL;
    }
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(sl, i, PyUnicode_FromString(
            results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lp, i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nt, i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(NNN)", sl, lp, nt);
}

static PyObject *py_flat_path_count_exact(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    uint32_t *limbs = NULL, n = 0;
    LZGError err = lzg_flat_path_count_exact(g, &limbs, &n);
    if (err != LZG_OK) return set_lzg_error(err);
    if (n == 0) { free(limbs); return PyLong_FromLong(0); }
    size_t len = (size_t)n * 8 + 1;
    char *buf = (char *)malloc(len);
    if (!buf) { free(limbs); return PyErr_NoMemory(); }
    char *p = buf;
    p += snprintf(p, len, "%" PRIx32, limbs[n - 1]);
    for (uint32_t i = n - 1; i > 0; i--)
        p += snprintf(p, len - (size_t)(p - buf), "%08" PRIx32, limbs[i - 1]);
    PyObject *res = PyLong_FromString(buf, NULL, 16);
    free(buf); free(limbs);
    return res;
}

static PyObject *py_flat_effective_diversity(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGEffectiveDiversity ed;
    LZGError err = lzg_flat_effective_diversity(g, &ed);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "entropy_nats", ed.entropy_nats, "entropy_bits", ed.entropy_bits,
        "effective_diversity", ed.effective_diversity,
        "uniformity", ed.uniformity);
}

static PyObject *py_flat_power_sum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double m;
    LZGError err = lzg_flat_power_sum(g, alpha, &m);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(m);
}

static PyObject *py_flat_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double d;
    LZGError err = lzg_flat_hill_number(g, alpha, &d);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(d);
}

static PyObject *py_flat_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &orders_list))
        return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    Py_ssize_t n = PyList_GET_SIZE(orders_list);
    double *orders = malloc((size_t)n * sizeof(double));
    double *out = malloc((size_t)n * sizeof(double));
    if (!orders || !out) { free(orders); free(out); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++)
        orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_list, i));
    LZGError err = lzg_flat_hill_numbers(g, orders, (uint32_t)n, out);
    free(orders);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }
    PyObject *result = PyList_New(n);
    if (!result) { free(out); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    free(out);
    return result;
}

static PyObject *py_flat_dynamic_range(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_flat_dynamic_range(g, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
        "max_log_prob", dr.max_log_prob, "min_log_prob", dr.min_log_prob,
        "dynamic_range_nats", dr.dynamic_range_nats,
        "dynamic_range_orders", dr.dynamic_range_orders);
}

static PyObject *py_flat_pseq_diagnostics(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double atol;
    if (!PyArg_ParseTuple(args, "Od", &cap, &atol)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    LZGPgenDiagnostics diag;
    LZGError err = lzg_flat_pseq_diagnostics(g, atol, &diag);
    if (err != LZG_OK) return set_lzg_error(err);
    return Py_BuildValue("{s:d,s:d,s:d,s:O,s:I}",
        "total_absorbed", diag.total_absorbed,
        "total_leaked", diag.total_leaked,
        "initial_prob_sum", diag.initial_prob_sum,
        "is_proper", diag.is_proper ? Py_True : Py_False,
        "mc_samples", diag.mc_samples);
}

static PyObject *py_flat_fix_special_nodes(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    lzg_flat_fix_special_nodes(g);
    Py_RETURN_NONE;
}

static PyObject *py_naive_fix_special_nodes(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    lzg_naive_fix_special_nodes(g);
    Py_RETURN_NONE;
}

/* ── k_diversity(seqs, k, variant, draws, seed) → dict ────── */

static PyObject *py_k_diversity(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *seq_list;
    unsigned int k, draws = 100;
    const char *variant_str = "aap";
    long long seed = -1;

    static char *kwlist[] = {"sequences", "k", "variant", "draws", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!I|sIL", kwlist,
            &PyList_Type, &seq_list, &k, &variant_str, &draws, &seed))
        return NULL;

    LZGVariant variant;
    if (!parse_variant(variant_str, &variant)) return NULL;

    Py_ssize_t n;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    LZGRng rng;
    lzg_rng_seed(&rng, seed >= 0 ? (uint64_t)seed : 12345);

    LZGKDiversity kd;
    LZGError err = lzg_k_diversity(seqs, (uint32_t)n, variant, k, draws, &rng, &kd);
    free(seqs);
    if (err != LZG_OK) return set_lzg_error(err);

    return Py_BuildValue("{s:d, s:d, s:d, s:d}",
        "mean", kd.mean, "std", kd.std,
        "ci_low", kd.ci_low, "ci_high", kd.ci_high);
}

/* ── saturation_curve(seqs, variant, log_every) → list ────── */

static PyObject *py_saturation_curve(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *seq_list;
    const char *variant_str = "aap";
    unsigned int log_every = 100;

    static char *kwlist[] = {"sequences", "variant", "log_every", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|sI", kwlist,
            &PyList_Type, &seq_list, &variant_str, &log_every))
        return NULL;

    LZGVariant variant;
    if (!parse_variant(variant_str, &variant)) return NULL;

    Py_ssize_t n;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    uint32_t max_points = (uint32_t)(n / log_every) + 2;
    LZGSaturationPoint *pts = (LZGSaturationPoint *)malloc(max_points * sizeof(LZGSaturationPoint));
    if (!pts) { free(seqs); return PyErr_NoMemory(); }

    uint32_t out_count;
    LZGError err = lzg_saturation_curve(seqs, (uint32_t)n, variant, log_every, pts, &out_count);
    free(seqs);
    if (err != LZG_OK) { free(pts); return set_lzg_error(err); }

    PyObject *result = PyList_New(out_count);
    for (uint32_t i = 0; i < out_count; i++) {
        PyList_SET_ITEM(result, i, Py_BuildValue("{s:I, s:I, s:I}",
            "n_sequences", pts[i].n_sequences,
            "n_nodes", pts[i].n_nodes,
            "n_edges", pts[i].n_edges));
    }
    free(pts);
    return result;
}

/* ── Logging ───────────────────────────────────────────────── */

/* Default stderr logger */
static void stderr_log_cb(LZGLogLevel level, const char *msg, void *data) {
    (void)data;
    static const char *prefixes[] = {"", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"};
    const char *pfx = (level >= 1 && level <= 5) ? prefixes[level] : "?";
    fprintf(stderr, "[LZGraph/%s] %s\n", pfx, msg);
}

/* Python callable logger */
static PyObject *py_log_callback = NULL;

static void python_log_cb(LZGLogLevel level, const char *msg, void *data) {
    (void)data;
    if (!py_log_callback) return;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *result = PyObject_CallFunction(py_log_callback, "is", (int)level, msg);
    Py_XDECREF(result);
    if (PyErr_Occurred()) PyErr_Clear();  /* don't propagate from callback */
    PyGILState_Release(gstate);
}

static int parse_log_level(const char *s, LZGLogLevel *out) {
    if (strcmp(s, "none") == 0)       { *out = LZG_LOG_NONE; return 1; }
    else if (strcmp(s, "error") == 0) { *out = LZG_LOG_ERROR; return 1; }
    else if (strcmp(s, "warn") == 0)  { *out = LZG_LOG_WARN; return 1; }
    else if (strcmp(s, "info") == 0)  { *out = LZG_LOG_INFO; return 1; }
    else if (strcmp(s, "debug") == 0) { *out = LZG_LOG_DEBUG; return 1; }
    else if (strcmp(s, "trace") == 0) { *out = LZG_LOG_TRACE; return 1; }
    PyErr_Format(PyExc_ValueError,
        "level must be 'none','error','warn','info','debug','trace', got '%s'", s);
    return 0;
}

/* set_log_level(level_str) — enable stderr logging at given level */
static PyObject *py_set_log_level(PyObject *self, PyObject *arg) {
    (void)self;
    const char *level_str = PyUnicode_AsUTF8(arg);
    if (!level_str) return NULL;
    LZGLogLevel level;
    if (!parse_log_level(level_str, &level)) return NULL;
    Py_XDECREF(py_log_callback);
    py_log_callback = NULL;
    lzg_log_set(level, level == LZG_LOG_NONE ? NULL : stderr_log_cb, NULL);
    Py_RETURN_NONE;
}

/* set_log_callback(callable, level_str) — custom Python callback */
static PyObject *py_set_log_callback(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cb;
    const char *level_str = "info";
    if (!PyArg_ParseTuple(args, "O|s", &cb, &level_str)) return NULL;

    if (cb == Py_None) {
        /* Disable */
        Py_XDECREF(py_log_callback);
        py_log_callback = NULL;
        lzg_log_set(LZG_LOG_NONE, NULL, NULL);
        Py_RETURN_NONE;
    }

    if (!PyCallable_Check(cb)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable or None");
        return NULL;
    }

    LZGLogLevel level;
    if (!parse_log_level(level_str, &level)) return NULL;

    Py_XDECREF(py_log_callback);
    py_log_callback = cb;
    Py_INCREF(py_log_callback);
    lzg_log_set(level, python_log_cb, NULL);
    Py_RETURN_NONE;
}

/* ── Graph introspection ─────────────────────────────────── */

/**
 * Reconstruct a node label string from its components.
 * AAP:   "{subpattern}_{position}"   e.g. "SL_5"
 * NDP:   original label stored in pool (subpattern includes frame digit)
 *        We reconstruct as "{subpattern}{frame}_{position}" but since we
 *        stripped the frame in parse, we just use sp + "_" + pos.
 * Naive: "{subpattern}"              e.g. "SL"
 * Sentinels: "@" and "$"-suffixed nodes use sp directly with position.
 */
static PyObject *reconstruct_node_label(const LZGGraph *g, uint32_t node_id) {
    const char *sp = lzg_sp_get(g->pool, g->node_sp_id[node_id]);
    uint32_t pos = g->node_pos[node_id];
    if (g->variant == LZG_VARIANT_NAIVE || pos == UINT32_MAX) {
        return PyUnicode_FromString(sp);
    }
    /* AAP / NDP: "sp_pos" */
    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%u", sp, pos);
    return PyUnicode_FromString(buf);
}

/* graph_nodes(capsule) → list of node label strings */
static PyObject *py_graph_nodes(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    PyObject *list = PyList_New(g->n_nodes);
    if (!list) return NULL;
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        PyObject *label = reconstruct_node_label(g, i);
        if (!label) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, label);
    }
    return list;
}

/* graph_edges(capsule) → list of (src_label, dst_label, weight, count) tuples */
static PyObject *py_graph_edges(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    PyObject *list = PyList_New(g->n_edges);
    if (!list) return NULL;

    uint32_t idx = 0;
    for (uint32_t src = 0; src < g->n_nodes; src++) {
        uint32_t start = g->row_offsets[src];
        uint32_t end   = g->row_offsets[src + 1];
        for (uint32_t e = start; e < end; e++) {
            uint32_t dst = g->col_indices[e];
            PyObject *src_label = reconstruct_node_label(g, src);
            PyObject *dst_label = reconstruct_node_label(g, dst);
            if (!src_label || !dst_label) {
                Py_XDECREF(src_label); Py_XDECREF(dst_label);
                Py_DECREF(list); return NULL;
            }
            PyObject *tup = Py_BuildValue("(NNdK)",
                src_label, dst_label, g->edge_weights[e],
                (unsigned long long)g->edge_counts[e]);
            if (!tup) { Py_DECREF(list); return NULL; }
            PyList_SET_ITEM(list, idx++, tup);
        }
    }
    return list;
}

/* graph_length_distribution(capsule) → dict {length: count} */
static PyObject *py_graph_length_distribution(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    PyObject *dict = PyDict_New();
    if (!dict) return NULL;
    for (uint32_t i = 0; i <= g->max_length; i++) {
        if (g->length_counts[i] > 0) {
            PyObject *key = PyLong_FromUnsignedLong(i);
            PyObject *val = PyLong_FromUnsignedLongLong(
                (unsigned long long)g->length_counts[i]);
            PyDict_SetItem(dict, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
        }
    }
    return dict;
}

/* graph_adjacency_csr(capsule) → dict with row_offsets, col_indices, weights as lists */
static PyObject *py_graph_adjacency_csr(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    /* row_offsets: list of n_nodes+1 ints */
    PyObject *ro = PyList_New(g->n_nodes + 1);
    for (uint32_t i = 0; i <= g->n_nodes; i++)
        PyList_SET_ITEM(ro, i, PyLong_FromUnsignedLong(g->row_offsets[i]));

    /* col_indices: list of n_edges ints */
    PyObject *ci = PyList_New(g->n_edges);
    for (uint32_t i = 0; i < g->n_edges; i++)
        PyList_SET_ITEM(ci, i, PyLong_FromUnsignedLong(g->col_indices[i]));

    /* weights: list of n_edges floats */
    PyObject *wt = PyList_New(g->n_edges);
    for (uint32_t i = 0; i < g->n_edges; i++)
        PyList_SET_ITEM(wt, i, PyFloat_FromDouble(g->edge_weights[i]));

    /* counts: list of n_edges ints */
    PyObject *ct = PyList_New(g->n_edges);
    for (uint32_t i = 0; i < g->n_edges; i++)
        PyList_SET_ITEM(ct, i, PyLong_FromUnsignedLongLong(
            (unsigned long long)g->edge_counts[i]));

    return Py_BuildValue("{s:N, s:N, s:N, s:N, s:I, s:I}",
        "row_offsets", ro, "col_indices", ci,
        "weights", wt, "counts", ct,
        "n_nodes", g->n_nodes, "n_edges", g->n_edges);
}

/* graph_degrees(capsule) → dict with out_degrees, in_degrees as lists */
static PyObject *py_graph_degrees(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;

    PyObject *out_list = PyList_New(g->n_nodes);
    for (uint32_t i = 0; i < g->n_nodes; i++)
        PyList_SET_ITEM(out_list, i,
            PyLong_FromUnsignedLong(g->row_offsets[i + 1] - g->row_offsets[i]));

    /* In-degrees: count destinations */
    uint32_t *in_deg = calloc(g->n_nodes, sizeof(uint32_t));
    if (!in_deg) { Py_DECREF(out_list); return PyErr_NoMemory(); }
    for (uint32_t e = 0; e < g->n_edges; e++)
        in_deg[g->col_indices[e]]++;

    PyObject *in_list = PyList_New(g->n_nodes);
    for (uint32_t i = 0; i < g->n_nodes; i++)
        PyList_SET_ITEM(in_list, i, PyLong_FromUnsignedLong(in_deg[i]));
    free(in_deg);

    return Py_BuildValue("{s:N, s:N}", "out_degrees", out_list, "in_degrees", in_list);
}

/* ── Module method table ──────────────────────────────────── */

/* Forward declarations for FBG wrappers (definitions are below). */
static PyObject *py_fbg_build(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_build_file(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_info(PyObject *self, PyObject *arg);
static PyObject *py_fbg_nts(PyObject *self, PyObject *arg);
static PyObject *py_fbg_rules_at(PyObject *self, PyObject *args);
static PyObject *py_fbg_decompose(PyObject *self, PyObject *args);
static PyObject *py_fbg_tree_to_string(PyObject *self, PyObject *args);
static PyObject *py_fbg_length_counts(PyObject *self, PyObject *arg);
static PyObject *py_fbg_pgen(PyObject *self, PyObject *args);
static PyObject *py_fbg_pgen_mle(PyObject *self, PyObject *args);
static PyObject *py_fbg_pgen_batch(PyObject *self, PyObject *args);
static PyObject *py_fbg_nt_unseen_mass(PyObject *self, PyObject *args);
static PyObject *py_fbg_rule_marginal(PyObject *self, PyObject *args);
static PyObject *py_fbg_path_count_series(PyObject *self, PyObject *args);
static PyObject *py_fbg_length_distribution(PyObject *self, PyObject *args);
static PyObject *py_fbg_entropy(PyObject *self, PyObject *arg);
static PyObject *py_fbg_effective_diversity(PyObject *self, PyObject *arg);
static PyObject *py_fbg_power_sum(PyObject *self, PyObject *args);
static PyObject *py_fbg_hill_number(PyObject *self, PyObject *args);
static PyObject *py_fbg_hill_numbers(PyObject *self, PyObject *args);
static PyObject *py_fbg_simulate(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_top_k_sequences(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_dynamic_range(PyObject *self, PyObject *args);
static PyObject *py_fbg_posterior(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_subtract(PyObject *self, PyObject *args, PyObject *kw);
static PyObject *py_fbg_save(PyObject *self, PyObject *args);
static PyObject *py_fbg_load(PyObject *self, PyObject *arg);

static PyMethodDef module_methods[] = {
    {"graph_build",             (PyCFunction)py_graph_build,           METH_VARARGS | METH_KEYWORDS, NULL},
    {"graph_build_file",        (PyCFunction)py_graph_build_file,      METH_VARARGS | METH_KEYWORDS, NULL},
    {"graph_info",              py_graph_info,                         METH_O, NULL},
    {"simulate",                (PyCFunction)py_simulate,              METH_VARARGS | METH_KEYWORDS, NULL},
    {"gene_simulate",           (PyCFunction)py_gene_simulate,         METH_VARARGS | METH_KEYWORDS, NULL},
    {"lzpgen",                  py_lzpgen,                             METH_VARARGS, NULL},
    {"path_count",              py_path_count,                         METH_VARARGS, NULL},
    {"effective_diversity",     py_effective_diversity,                 METH_O, NULL},
    {"diversity_profile",       py_diversity_profile,                  METH_O, NULL},
    {"hill_number",             py_hill_number,                        METH_VARARGS, NULL},
    {"hill_numbers",            py_hill_numbers,                       METH_VARARGS, NULL},
    {"hill_curve",              py_hill_curve,                         METH_VARARGS, NULL},
    {"power_sum",               py_power_sum,                          METH_VARARGS, NULL},
    {"pgen_diagnostics",        py_pgen_diagnostics,                   METH_VARARGS, NULL},
    {"pgen_dynamic_range",      py_pgen_dynamic_range,                 METH_O, NULL},
    {"pgen_dynamic_range_detail", py_pgen_dynamic_range_detail,        METH_O, NULL},
    {"pgen_moments",            py_pgen_moments,                       METH_O, NULL},
    {"pgen_analytical",         py_pgen_analytical,                    METH_O, NULL},
    {"predicted_richness",      py_predicted_richness,                 METH_VARARGS, NULL},
    {"predicted_overlap",       py_predicted_overlap,                  METH_VARARGS, NULL},
    {"richness_curve",          py_richness_curve,                     METH_VARARGS, NULL},
    {"pseq_discovery_curve",    py_pseq_discovery_curve,               METH_VARARGS, NULL},
    {"predict_sharing",         py_predict_sharing,                    METH_VARARGS, NULL},
    {"publicness_moments",      py_publicness_moments,                 METH_VARARGS, NULL},
    {"publicness_pgf",          py_publicness_pgf,                     METH_VARARGS, NULL},
    {"publicness_accumulate",   py_publicness_accumulate,              METH_VARARGS, NULL},
    {"sequence_perplexity",     py_sequence_perplexity,                METH_VARARGS, NULL},
    {"repertoire_perplexity",   py_repertoire_perplexity,              METH_VARARGS, NULL},
    {"path_entropy_rate",       py_path_entropy_rate,                  METH_VARARGS, NULL},
    {"jensen_shannon_divergence", py_jsd,                              METH_VARARGS, NULL},
    {"summary",                 py_summary,                            METH_O, NULL},
    {"graph_union",             py_graph_union,                        METH_VARARGS, NULL},
    {"graph_intersection",      py_graph_intersection,                 METH_VARARGS, NULL},
    {"graph_difference",        py_graph_difference,                   METH_VARARGS, NULL},
    {"weighted_merge",          py_weighted_merge,                     METH_VARARGS, NULL},
    {"posterior",               (PyCFunction)py_posterior,              METH_VARARGS | METH_KEYWORDS, NULL},
    {"feature_stats",           py_feature_stats,                      METH_O, NULL},
    {"feature_mass_profile",    py_feature_mass_profile,               METH_VARARGS, NULL},
    {"feature_aligned",         py_feature_aligned,                    METH_VARARGS, NULL},
    {"save",                    py_save,                               METH_VARARGS, NULL},
    {"load",                    py_load,                               METH_O, NULL},
    {"gene_info",               py_gene_info,                          METH_O, NULL},
    {"find_gene_id",            py_find_gene_id,                       METH_VARARGS, NULL},
    {"lz76_decompose",          py_lz76_decompose,                     METH_O, NULL},
    {"flashback_decompose",     py_flashback_decompose,                METH_O, NULL},
    {"flashback_reverse",       py_flashback_reverse,                  METH_O, NULL},
    {"fb_graph_build",          (PyCFunction)py_fb_graph_build,        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_graph_build_file",     (PyCFunction)py_fb_graph_build_file,   METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_stream_open",          (PyCFunction)py_fb_stream_open,        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_stream_add",           (PyCFunction)py_fb_stream_add,         METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_stream_peek",          py_fb_stream_peek,                     METH_O, NULL},
    {"fb_stream_finalize",      py_fb_stream_finalize,                 METH_O, NULL},
    {"fb_stream_abort",         py_fb_stream_abort,                    METH_O, NULL},
    {"fb_stream_snapshot",      py_fb_stream_snapshot,                 METH_O, NULL},
    {"fb_simulate",             (PyCFunction)py_fb_simulate,           METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_pgen",                 py_fb_pgen,                            METH_VARARGS, NULL},
    {"fb_path_count",           py_fb_path_count,                      METH_O, NULL},
    {"fb_path_count_exact",     py_fb_path_count_exact,                METH_O, NULL},
    {"fb_path_count_by_length", py_fb_path_count_by_length,            METH_O, NULL},
    {"fb_pseq_structure",       py_fb_pseq_structure,                  METH_O, NULL},
    {"fb_pseq_length_derivatives", py_fb_pseq_length_derivatives,      METH_VARARGS, NULL},
    {"fb_pseq_derivatives",     py_fb_pseq_derivatives,                METH_VARARGS, NULL},
    {"fb_pseq_tilted_moments", py_fb_pseq_tilted_moments,             METH_VARARGS, NULL},
    {"fb_pseq_saddlepoint_batch", py_fb_pseq_saddlepoint_batch,        METH_VARARGS, NULL},
    {"fb_edge_threshold_diversity", py_fb_edge_threshold_diversity,    METH_VARARGS, NULL},
    {"fb_pseq_attribution",    py_fb_pseq_attribution,                 METH_VARARGS, NULL},
    {"fb_pseq_histogram",       py_fb_pseq_histogram,                  METH_VARARGS, NULL},
    {"fb_pseq_histogram_pair",  py_fb_pseq_histogram_pair,             METH_VARARGS, NULL},
    {"fb_effective_diversity",  py_fb_effective_diversity,              METH_O, NULL},
    {"fb_power_sum",            py_fb_power_sum,                       METH_VARARGS, NULL},
    {"fb_hill_number",          py_fb_hill_number,                     METH_VARARGS, NULL},
    {"fb_hill_numbers",         py_fb_hill_numbers,                    METH_VARARGS, NULL},
    {"fb_dynamic_range",        py_fb_dynamic_range,                   METH_O, NULL},
    {"fb_pgen_diagnostics",     py_fb_pgen_diagnostics,                METH_VARARGS, NULL},
    {"fb_top_k_walks",          (PyCFunction)py_fb_top_k_walks,        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_fix_special_nodes",    py_fb_fix_special_nodes,               METH_O, NULL},
    {"fb_posterior",            (PyCFunction)py_fb_posterior,          METH_VARARGS | METH_KEYWORDS, NULL},
    {"fb_subtract",             (PyCFunction)py_fb_subtract,           METH_VARARGS | METH_KEYWORDS, NULL},

    {"naive_decompose",            py_naive_decompose,                    METH_O, NULL},
    {"naive_graph_build",          (PyCFunction)py_naive_graph_build,     METH_VARARGS | METH_KEYWORDS, NULL},
    {"naive_graph_build_file",     (PyCFunction)py_naive_graph_build_file, METH_VARARGS | METH_KEYWORDS, NULL},
    {"naive_simulate",             (PyCFunction)py_naive_simulate,        METH_VARARGS | METH_KEYWORDS, NULL},
    {"naive_pgen",                 py_naive_pgen,                         METH_VARARGS, NULL},
    {"naive_path_count",           py_naive_path_count,                   METH_O, NULL},
    {"naive_path_count_exact",     py_naive_path_count_exact,             METH_O, NULL},
    {"naive_effective_diversity",  py_naive_effective_diversity,          METH_O, NULL},
    {"naive_power_sum",            py_naive_power_sum,                    METH_VARARGS, NULL},
    {"naive_hill_number",          py_naive_hill_number,                  METH_VARARGS, NULL},
    {"naive_hill_numbers",         py_naive_hill_numbers,                 METH_VARARGS, NULL},
    {"naive_dynamic_range",        py_naive_dynamic_range,                METH_O, NULL},
    {"naive_pgen_diagnostics",     py_naive_pgen_diagnostics,             METH_VARARGS, NULL},
    {"naive_fix_special_nodes",    py_naive_fix_special_nodes,            METH_O, NULL},

    {"flat_decompose",             py_flat_decompose,                     METH_O, NULL},
    {"flat_graph_build",           (PyCFunction)py_flat_graph_build,      METH_VARARGS | METH_KEYWORDS, NULL},
    {"flat_graph_build_file",      (PyCFunction)py_flat_graph_build_file, METH_VARARGS | METH_KEYWORDS, NULL},
    {"flat_simulate",              (PyCFunction)py_flat_simulate,         METH_VARARGS | METH_KEYWORDS, NULL},
    {"flat_pseq",                  py_flat_pseq,                          METH_VARARGS, NULL},
    {"flat_path_count_exact",      py_flat_path_count_exact,              METH_O, NULL},
    {"flat_effective_diversity",   py_flat_effective_diversity,           METH_O, NULL},
    {"flat_power_sum",             py_flat_power_sum,                     METH_VARARGS, NULL},
    {"flat_hill_number",           py_flat_hill_number,                   METH_VARARGS, NULL},
    {"flat_hill_numbers",          py_flat_hill_numbers,                  METH_VARARGS, NULL},
    {"flat_dynamic_range",         py_flat_dynamic_range,                 METH_O, NULL},
    {"flat_pseq_diagnostics",      py_flat_pseq_diagnostics,              METH_VARARGS, NULL},
    {"flat_fix_special_nodes",     py_flat_fix_special_nodes,             METH_O, NULL},
    {"k_diversity",             (PyCFunction)py_k_diversity,            METH_VARARGS | METH_KEYWORDS, NULL},
    {"saturation_curve",        (PyCFunction)py_saturation_curve,      METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_log_level",           py_set_log_level,                      METH_O, NULL},
    {"set_log_callback",        py_set_log_callback,                   METH_VARARGS, NULL},
    {"graph_nodes",             py_graph_nodes,                        METH_O, NULL},
    {"graph_edges",             py_graph_edges,                        METH_O, NULL},
    {"graph_length_distribution", py_graph_length_distribution,        METH_O, NULL},
    {"graph_adjacency_csr",     py_graph_adjacency_csr,                METH_O, NULL},
    {"graph_degrees",           py_graph_degrees,                      METH_O, NULL},

    /* ── FlashBackGrammar (new PCFG variant) ─────────────── */
    {"fbg_build",               (PyCFunction)py_fbg_build,             METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_build_file",          (PyCFunction)py_fbg_build_file,        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_info",                py_fbg_info,                           METH_O, NULL},
    {"fbg_nts",                 py_fbg_nts,                            METH_O, NULL},
    {"fbg_rules_at",            py_fbg_rules_at,                       METH_VARARGS, NULL},
    {"fbg_decompose",           py_fbg_decompose,                      METH_VARARGS, NULL},
    {"fbg_tree_to_string",      py_fbg_tree_to_string,                 METH_VARARGS, NULL},
    {"fbg_length_counts",       py_fbg_length_counts,                  METH_O, NULL},
    {"fbg_pgen",                py_fbg_pgen,                           METH_VARARGS, NULL},
    {"fbg_pgen_mle",            py_fbg_pgen_mle,                       METH_VARARGS, NULL},
    {"fbg_pgen_batch",          py_fbg_pgen_batch,                     METH_VARARGS, NULL},
    {"fbg_nt_unseen_mass",      py_fbg_nt_unseen_mass,                 METH_VARARGS, NULL},
    {"fbg_rule_marginal",       py_fbg_rule_marginal,                  METH_VARARGS, NULL},
    {"fbg_path_count_series",   py_fbg_path_count_series,              METH_VARARGS, NULL},
    {"fbg_length_distribution", py_fbg_length_distribution,            METH_VARARGS, NULL},
    {"fbg_entropy",             py_fbg_entropy,                        METH_O, NULL},
    {"fbg_effective_diversity", py_fbg_effective_diversity,            METH_O, NULL},
    {"fbg_power_sum",           py_fbg_power_sum,                      METH_VARARGS, NULL},
    {"fbg_hill_number",         py_fbg_hill_number,                    METH_VARARGS, NULL},
    {"fbg_hill_numbers",        py_fbg_hill_numbers,                   METH_VARARGS, NULL},
    {"fbg_simulate",            (PyCFunction)py_fbg_simulate,          METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_top_k_sequences",     (PyCFunction)py_fbg_top_k_sequences,   METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_dynamic_range",       py_fbg_dynamic_range,                  METH_VARARGS, NULL},
    {"fbg_posterior",           (PyCFunction)py_fbg_posterior,         METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_subtract",            (PyCFunction)py_fbg_subtract,          METH_VARARGS | METH_KEYWORDS, NULL},
    {"fbg_save",                py_fbg_save,                           METH_VARARGS, NULL},
    {"fbg_load",                py_fbg_load,                           METH_O, NULL},

    {NULL, NULL, 0, NULL}
};

/* ══════════════════════════════════════════════════════════════ */
/* FlashBackGrammar bindings                                     */
/* ══════════════════════════════════════════════════════════════ */

static const char *FBG_CAPSULE_NAME = "LZGFlashbackGrammar";

static void fbg_capsule_destructor(PyObject *capsule) {
    LZGFlashbackGrammar *g = (LZGFlashbackGrammar *)PyCapsule_GetPointer(capsule, FBG_CAPSULE_NAME);
    if (g) lzg_flashback_grammar_destroy(g);
}

static LZGFlashbackGrammar *fbg_from_capsule(PyObject *capsule) {
    return (LZGFlashbackGrammar *)PyCapsule_GetPointer(capsule, FBG_CAPSULE_NAME);
}

static int parse_abundance_mode(const char *s, LZGFlashbackGrammarAbundanceMode *out) {
    if (!s || !strcmp(s, "linear")) { *out = LZG_FLASHBACK_GRAMMAR_ABUND_LINEAR; return 1; }
    if (!strcmp(s, "none"))         { *out = LZG_FLASHBACK_GRAMMAR_ABUND_NONE;   return 1; }
    if (!strcmp(s, "log"))          { *out = LZG_FLASHBACK_GRAMMAR_ABUND_LOG;    return 1; }
    PyErr_Format(PyExc_ValueError,
                 "abundance_mode must be 'none', 'linear', or 'log' (got %R)",
                 s ? PyUnicode_FromString(s) : Py_None);
    return 0;
}

static int parse_backoff(const char *s, LZGFlashbackGrammarBackoff *out) {
    if (!s || !strcmp(s, "none")) { *out = LZG_FLASHBACK_GRAMMAR_BACKOFF_NONE; return 1; }
    if (!strcmp(s, "gt"))         { *out = LZG_FLASHBACK_GRAMMAR_BACKOFF_GT;   return 1; }
    PyErr_Format(PyExc_ValueError,
                 "backoff must be 'none' or 'gt' (got %R)",
                 s ? PyUnicode_FromString(s) : Py_None);
    return 0;
}

/* ── fbg_build(sequences, abundances=None, abundance_mode='linear',
 *             smoothing=0.0, backoff='none') → capsule ─────── */

static PyObject *py_fbg_build(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *seq_list;
    PyObject *abund_obj = Py_None;
    const char *mode_str = "linear";
    const char *backoff_str = "none";
    double smoothing = 0.0;
    static char *kwlist[] = {"sequences", "abundances", "abundance_mode",
                             "smoothing", "backoff", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|Osds", kwlist,
            &PyList_Type, &seq_list, &abund_obj, &mode_str, &smoothing,
            &backoff_str))
        return NULL;

    LZGFlashbackGrammarAbundanceMode mode;
    LZGFlashbackGrammarBackoff backoff;
    if (!parse_abundance_mode(mode_str, &mode)) return NULL;
    if (!parse_backoff(backoff_str, &backoff)) return NULL;

    Py_ssize_t n_seqs;
    uint32_t n_seqs_u32 = 0;
    if (!pyssize_to_u32(PyList_GET_SIZE(seq_list), "sequences", &n_seqs_u32))
        return NULL;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGFlashbackGrammar *g = lzg_flashback_grammar_new();
    if (!g) { free(seqs); free(abundances); return PyErr_NoMemory(); }

    LZGError err = lzg_flashback_grammar_build(g, seqs, n_seqs_u32, abundances,
                                  mode, smoothing, backoff);
    free(seqs); free(abundances);
    if (err != LZG_OK) { lzg_flashback_grammar_destroy(g); return set_lzg_error(err); }
    return PyCapsule_New(g, FBG_CAPSULE_NAME, fbg_capsule_destructor);
}

/* ── fbg_build_file(path, abundance_mode, smoothing, backoff) → cap ─ */

static PyObject *py_fbg_build_file(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    const char *path = NULL;
    const char *mode_str = "linear";
    const char *backoff_str = "none";
    double smoothing = 0.0;
    static char *kwlist[] = {"path", "abundance_mode", "smoothing", "backoff", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|sds", kwlist,
            &path, &mode_str, &smoothing, &backoff_str))
        return NULL;

    LZGFlashbackGrammarAbundanceMode mode;
    LZGFlashbackGrammarBackoff backoff;
    if (!parse_abundance_mode(mode_str, &mode)) return NULL;
    if (!parse_backoff(backoff_str, &backoff)) return NULL;

    LZGFlashbackGrammar *g = lzg_flashback_grammar_new();
    if (!g) return PyErr_NoMemory();

    /* Release the GIL — this can run for many minutes on a multi-GB file. */
    LZGError err;
    Py_BEGIN_ALLOW_THREADS
    err = lzg_flashback_grammar_build_file(g, path, mode, smoothing, backoff);
    Py_END_ALLOW_THREADS
    if (err != LZG_OK) { lzg_flashback_grammar_destroy(g); return set_lzg_error(err); }

    return PyCapsule_New(g, FBG_CAPSULE_NAME, fbg_capsule_destructor);
}

/* ── fbg_info(cap) → dict ──────────────────────────────────── */

static PyObject *py_fbg_info(PyObject *self, PyObject *arg) {
    (void)self;
    LZGFlashbackGrammar *g = fbg_from_capsule(arg);
    if (!g) return NULL;

    PyObject *d = PyDict_New();
    if (!d) return NULL;
    #define SET_ULONG(k, v) do { PyObject *o = PyLong_FromUnsignedLong((unsigned long)(v)); PyDict_SetItemString(d, k, o); Py_DECREF(o); } while (0)
    #define SET_DOUBLE(k, v) do { PyObject *o = PyFloat_FromDouble((double)(v)); PyDict_SetItemString(d, k, o); Py_DECREF(o); } while (0)
    #define SET_BOOL(k, v) do { PyObject *o = PyBool_FromLong((long)(v)); PyDict_SetItemString(d, k, o); Py_DECREF(o); } while (0)

    SET_ULONG("n_nts", g->n_nts);
    SET_ULONG("n_rules", g->n_rules);
    SET_ULONG("n_internal_rules", g->n_internal);
    SET_ULONG("n_leaf_rules", g->n_rules - g->n_internal);
    SET_ULONG("alphabet_size", g->alphabet_size);
    SET_ULONG("max_length", g->max_length);
    SET_DOUBLE("spectral_radius", g->spectral_radius);
    SET_BOOL("is_consistent", g->is_consistent);
    SET_DOUBLE("smoothing", g->smoothing);
    SET_ULONG("abundance_mode", g->abundance_mode);
    SET_ULONG("backoff", g->backoff);
    SET_ULONG("start_nt", g->start_nt);

    #undef SET_ULONG
    #undef SET_DOUBLE
    #undef SET_BOOL
    return d;
}

/* ── fbg_nts(cap) → list of (a_char:str, z_char:str, is_start:bool, total:float, unseen_mass:float, n_rules:int) ── */

static PyObject *py_fbg_nts(PyObject *self, PyObject *arg) {
    (void)self;
    LZGFlashbackGrammar *g = fbg_from_capsule(arg);
    if (!g) return NULL;

    PyObject *list = PyList_New(g->n_nts);
    if (!list) return NULL;
    for (uint32_t i = 0; i < g->n_nts; i++) {
        const LZGFlashbackGrammarNT *nt = &g->nts[i];
        char a_ch = (char)g->idx_to_char[nt->a];
        char z_ch = (char)g->idx_to_char[nt->z];
        PyObject *tup = Py_BuildValue("(s#s#OdkI)",
            &a_ch, (Py_ssize_t)1,
            &z_ch, (Py_ssize_t)1,
            nt->is_start ? Py_True : Py_False,
            nt->total_count,
            (unsigned long)nt->n_rules,
            (unsigned int)nt->unseen_mass);   /* placeholder, replaced below */
        /* Rebuild with correct unseen_mass as double — Py_BuildValue doesn't
         * do two doubles + bool nicely, so fix here. */
        Py_DECREF(tup);
        tup = PyTuple_New(6);
        PyTuple_SET_ITEM(tup, 0, PyUnicode_FromStringAndSize(&a_ch, 1));
        PyTuple_SET_ITEM(tup, 1, PyUnicode_FromStringAndSize(&z_ch, 1));
        Py_INCREF(nt->is_start ? Py_True : Py_False);
        PyTuple_SET_ITEM(tup, 2, nt->is_start ? Py_True : Py_False);
        PyTuple_SET_ITEM(tup, 3, PyFloat_FromDouble(nt->total_count));
        PyTuple_SET_ITEM(tup, 4, PyLong_FromUnsignedLong(nt->n_rules));
        PyTuple_SET_ITEM(tup, 5, PyFloat_FromDouble(nt->unseen_mass));
        PyList_SET_ITEM(list, i, tup);
    }
    return list;
}

/* ── fbg_rules_at(cap, nt_index) → list of rule dicts ──────── */

static PyObject *py_fbg_rules_at(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int nt_idx;
    if (!PyArg_ParseTuple(args, "OI", &cap, &nt_idx)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    if (nt_idx >= g->n_nts) {
        PyErr_Format(PyExc_IndexError, "nt_idx %u out of range [0, %u)",
                     nt_idx, g->n_nts);
        return NULL;
    }

    const LZGFlashbackGrammarNT *nt = &g->nts[nt_idx];
    PyObject *list = PyList_New(nt->n_rules);
    if (!list) return NULL;

    static const char *KIND_NAMES[] = {
        "internal", "leaf_single", "leaf_run", "leaf_pair"
    };
    for (uint32_t r = 0; r < nt->n_rules; r++) {
        const LZGFlashbackGrammarRule *rule = &g->rules[nt->rule_offset + r];
        char a_ch = (char)g->idx_to_char[rule->a_char];
        char z_ch = (char)g->idx_to_char[rule->z_char];
        char dst_a_ch = rule->kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL
                        ? (char)g->idx_to_char[rule->dst_a] : '\0';
        char dst_z_ch = rule->kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL
                        ? (char)g->idx_to_char[rule->dst_z] : '\0';

        PyObject *d = PyDict_New();
        if (!d) { Py_DECREF(list); return NULL; }
        PyDict_SetItemString(d, "kind",
            PyUnicode_FromString(KIND_NAMES[rule->kind]));
        PyDict_SetItemString(d, "a_char", PyUnicode_FromStringAndSize(&a_ch, 1));
        PyDict_SetItemString(d, "z_char", PyUnicode_FromStringAndSize(&z_ch, 1));
        PyDict_SetItemString(d, "a_run_len",
            PyLong_FromUnsignedLong(rule->a_run_len));
        PyDict_SetItemString(d, "z_run_len",
            PyLong_FromUnsignedLong(rule->z_run_len));
        if (rule->kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) {
            PyDict_SetItemString(d, "dst_a",
                PyUnicode_FromStringAndSize(&dst_a_ch, 1));
            PyDict_SetItemString(d, "dst_z",
                PyUnicode_FromStringAndSize(&dst_z_ch, 1));
        }
        PyDict_SetItemString(d, "count", PyLong_FromUnsignedLongLong(rule->count));
        PyDict_SetItemString(d, "weight", PyFloat_FromDouble(rule->weight));
        PyList_SET_ITEM(list, r, d);
    }
    return list;
}

/* ── fbg_decompose(cap, seq) → list of step dicts ──────────── */

static PyObject *py_fbg_decompose(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *seq;
    Py_ssize_t seq_len;
    if (!PyArg_ParseTuple(args, "Os#", &cap, &seq, &seq_len)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    LZGFlashbackGrammarStep steps[LZG_FLASHBACK_GRAMMAR_MAX_STEPS];
    uint32_t n_steps = 0;
    LZGError err = lzg_flashback_grammar_decompose(g, seq, (uint32_t)seq_len, steps, &n_steps);
    if (err != LZG_OK) return set_lzg_error(err);

    static const char *KIND_NAMES[] = {
        "internal", "leaf_single", "leaf_run", "leaf_pair"
    };
    PyObject *list = PyList_New(n_steps);
    if (!list) return NULL;
    for (uint32_t i = 0; i < n_steps; i++) {
        const LZGFlashbackGrammarStep *s = &steps[i];
        char a_ch = (char)g->idx_to_char[s->a_char];
        char z_ch = (char)g->idx_to_char[s->z_char];

        PyObject *d = PyDict_New();
        PyDict_SetItemString(d, "kind",
            PyUnicode_FromString(KIND_NAMES[s->kind]));
        PyDict_SetItemString(d, "is_start",
            PyBool_FromLong(s->is_start));
        PyDict_SetItemString(d, "a_char", PyUnicode_FromStringAndSize(&a_ch, 1));
        PyDict_SetItemString(d, "z_char", PyUnicode_FromStringAndSize(&z_ch, 1));
        PyDict_SetItemString(d, "a_run_len", PyLong_FromUnsignedLong(s->a_run_len));
        PyDict_SetItemString(d, "z_run_len", PyLong_FromUnsignedLong(s->z_run_len));
        if (s->kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) {
            char da = (char)g->idx_to_char[s->dst_a];
            char dz = (char)g->idx_to_char[s->dst_z];
            PyDict_SetItemString(d, "dst_a", PyUnicode_FromStringAndSize(&da, 1));
            PyDict_SetItemString(d, "dst_z", PyUnicode_FromStringAndSize(&dz, 1));
        }
        PyList_SET_ITEM(list, i, d);
    }
    return list;
}

/* ── fbg_tree_to_string(cap, [step_dicts]) → str ─────────────── */

static PyObject *py_fbg_tree_to_string(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    PyObject *step_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &step_list))
        return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(step_list);
    if (n <= 0 || n > LZG_FLASHBACK_GRAMMAR_MAX_STEPS) {
        PyErr_SetString(PyExc_ValueError,
                        "steps list must have 1..LZG_FLASHBACK_GRAMMAR_MAX_STEPS entries");
        return NULL;
    }
    LZGFlashbackGrammarStep steps[LZG_FLASHBACK_GRAMMAR_MAX_STEPS];
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *d = PyList_GET_ITEM(step_list, i);
        if (!PyDict_Check(d)) {
            PyErr_Format(PyExc_TypeError, "step %zd must be a dict", i);
            return NULL;
        }
        memset(&steps[i], 0, sizeof(steps[i]));
        /* Simplified parser: expects the dict shape produced by py_fbg_decompose. */
        const char *kind_s = PyUnicode_AsUTF8(PyDict_GetItemString(d, "kind"));
        if      (!strcmp(kind_s, "internal"))    steps[i].kind = LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL;
        else if (!strcmp(kind_s, "leaf_single")) steps[i].kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE;
        else if (!strcmp(kind_s, "leaf_run"))    steps[i].kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN;
        else if (!strcmp(kind_s, "leaf_pair"))   steps[i].kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR;
        else {
            PyErr_Format(PyExc_ValueError, "unknown kind %s", kind_s);
            return NULL;
        }
        const char *a_s = PyUnicode_AsUTF8(PyDict_GetItemString(d, "a_char"));
        const char *z_s = PyUnicode_AsUTF8(PyDict_GetItemString(d, "z_char"));
        steps[i].a_char = g->char_to_idx[(uint8_t)a_s[0]];
        steps[i].z_char = g->char_to_idx[(uint8_t)z_s[0]];
        steps[i].a_run_len = (uint8_t)PyLong_AsUnsignedLong(
            PyDict_GetItemString(d, "a_run_len"));
        steps[i].z_run_len = (uint8_t)PyLong_AsUnsignedLong(
            PyDict_GetItemString(d, "z_run_len"));
        if (steps[i].kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) {
            const char *da = PyUnicode_AsUTF8(PyDict_GetItemString(d, "dst_a"));
            const char *dz = PyUnicode_AsUTF8(PyDict_GetItemString(d, "dst_z"));
            steps[i].dst_a = g->char_to_idx[(uint8_t)da[0]];
            steps[i].dst_z = g->char_to_idx[(uint8_t)dz[0]];
        }
        PyObject *is_start = PyDict_GetItemString(d, "is_start");
        steps[i].is_start = is_start && PyObject_IsTrue(is_start) ? 1 : 0;
    }
    char out[2048];
    uint32_t out_len = 0;
    LZGError err = lzg_flashback_grammar_tree_to_string(g, steps, (uint32_t)n,
                                           out, sizeof(out), &out_len);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyUnicode_FromStringAndSize(out, (Py_ssize_t)out_len);
}

/* ── fbg_length_counts(cap) → list[int] ─────────────────────── */

static PyObject *py_fbg_length_counts(PyObject *self, PyObject *arg) {
    (void)self;
    LZGFlashbackGrammar *g = fbg_from_capsule(arg);
    if (!g) return NULL;
    uint32_t n = g->max_length + 1;
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (uint32_t i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyLong_FromUnsignedLongLong(g->length_counts[i]));
    return list;
}

/* ── fbg_pgen(cap, seq) / fbg_pgen_mle(cap, seq) → float ──── */

static PyObject *py_fbg_pgen(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *seq;
    Py_ssize_t seq_len;
    if (!PyArg_ParseTuple(args, "Os#", &cap, &seq, &seq_len)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    double lp = lzg_flashback_grammar_pgen(g, seq, (uint32_t)seq_len);
    return PyFloat_FromDouble(lp);
}

static PyObject *py_fbg_pgen_mle(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *seq;
    Py_ssize_t seq_len;
    if (!PyArg_ParseTuple(args, "Os#", &cap, &seq, &seq_len)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    double lp = lzg_flashback_grammar_pgen_mle(g, seq, (uint32_t)seq_len);
    return PyFloat_FromDouble(lp);
}

static PyObject *py_fbg_pgen_batch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    PyObject *seq_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &seq_list))
        return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    Py_ssize_t n;
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!out) { free(seqs); return PyErr_NoMemory(); }

    LZGError err = lzg_flashback_grammar_pgen_batch(g, seqs, (uint32_t)n, out);
    free(seqs);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }

    PyObject *list = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(out[i]));
    free(out);
    return list;
}

/* ── fbg_nt_unseen_mass(cap, nt_idx) → float ──────────────── */

static PyObject *py_fbg_nt_unseen_mass(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int nt_idx;
    if (!PyArg_ParseTuple(args, "OI", &cap, &nt_idx)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    if (nt_idx >= g->n_nts) {
        PyErr_Format(PyExc_IndexError, "nt_idx out of range");
        return NULL;
    }
    return PyFloat_FromDouble(g->nts[nt_idx].unseen_mass);
}

/* ── fbg_rule_marginal(cap, rule_dict) → float ────────────── */

static PyObject *py_fbg_rule_marginal(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    PyObject *rule;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &rule))
        return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g || !g->rule_marginal) return PyFloat_FromDouble(0.0);

    /* Decode the rule dict into kind + chars + run lens. */
    const char *kind_s = PyUnicode_AsUTF8(PyDict_GetItemString(rule, "kind"));
    uint8_t kind;
    if (!strcmp(kind_s, "internal"))    kind = LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL;
    else if (!strcmp(kind_s, "leaf_single")) kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE;
    else if (!strcmp(kind_s, "leaf_run"))    kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN;
    else if (!strcmp(kind_s, "leaf_pair"))   kind = LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR;
    else { PyErr_Format(PyExc_ValueError, "unknown kind %s", kind_s); return NULL; }

    const char *a_s = PyUnicode_AsUTF8(PyDict_GetItemString(rule, "a_char"));
    const char *z_s = PyUnicode_AsUTF8(PyDict_GetItemString(rule, "z_char"));
    uint8_t a_char = g->char_to_idx[(uint8_t)a_s[0]];
    uint8_t z_char = g->char_to_idx[(uint8_t)z_s[0]];
    uint8_t a_run = (uint8_t)PyLong_AsUnsignedLong(PyDict_GetItemString(rule, "a_run_len"));
    uint8_t z_run = (uint8_t)PyLong_AsUnsignedLong(PyDict_GetItemString(rule, "z_run_len"));
    uint8_t dst_a = 0, dst_z = 0;
    if (kind == LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL) {
        const char *da = PyUnicode_AsUTF8(PyDict_GetItemString(rule, "dst_a"));
        const char *dz = PyUnicode_AsUTF8(PyDict_GetItemString(rule, "dst_z"));
        dst_a = g->char_to_idx[(uint8_t)da[0]];
        dst_z = g->char_to_idx[(uint8_t)dz[0]];
    }

    /* Rebuild the packed key (same formula as flashback_grammar_internal.h). */
    uint64_t key = (uint64_t)kind;
    switch (kind) {
    case LZG_FLASHBACK_GRAMMAR_RULE_INTERNAL:
        key |= ((uint64_t)a_run) << 8;
        key |= ((uint64_t)z_run) << 16;
        key |= ((uint64_t)dst_a) << 24;
        key |= ((uint64_t)dst_z) << 32;
        break;
    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_SINGLE:
        key |= ((uint64_t)a_char) << 8;
        break;
    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_RUN:
        key |= ((uint64_t)a_char) << 8;
        key |= ((uint64_t)a_run)  << 16;
        break;
    case LZG_FLASHBACK_GRAMMAR_RULE_LEAF_PAIR:
        key |= ((uint64_t)a_char) << 8;
        key |= ((uint64_t)a_run)  << 16;
        key |= ((uint64_t)z_char) << 24;
        key |= ((uint64_t)z_run)  << 32;
        break;
    }

    uint64_t *slot = lzg_hm_get(g->rule_marginal, key);
    if (!slot) return PyFloat_FromDouble(0.0);
    double d;
    memcpy(&d, slot, sizeof(d));
    return PyFloat_FromDouble(d);
}

/* ── fbg_path_count_series / fbg_length_distribution ──────── */

static PyObject *fbg_series_impl(PyObject *args, bool weighted) {
    PyObject *cap;
    unsigned int L_max;
    if (!PyArg_ParseTuple(args, "OI", &cap, &L_max)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    double *buf = (double *)calloc((size_t)(L_max + 1), sizeof(double));
    if (!buf) return PyErr_NoMemory();

    LZGError err = weighted
        ? lzg_flashback_grammar_length_distribution(g, L_max, buf)
        : lzg_flashback_grammar_path_count_series(g, L_max, buf);
    if (err != LZG_OK) { free(buf); return set_lzg_error(err); }

    PyObject *list = PyList_New(L_max + 1);
    for (unsigned int i = 0; i <= L_max; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    free(buf);
    return list;
}

static PyObject *py_fbg_path_count_series(PyObject *self, PyObject *args) {
    (void)self;
    return fbg_series_impl(args, /*weighted=*/false);
}

static PyObject *py_fbg_length_distribution(PyObject *self, PyObject *args) {
    (void)self;
    return fbg_series_impl(args, /*weighted=*/true);
}

/* ── fbg_entropy / fbg_effective_diversity ────────────────── */

static PyObject *py_fbg_entropy(PyObject *self, PyObject *arg) {
    (void)self;
    LZGFlashbackGrammar *g = fbg_from_capsule(arg);
    if (!g) return NULL;
    double H = 0.0;
    LZGError err = lzg_flashback_grammar_entropy(g, &H);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(H);
}

static PyObject *py_fbg_effective_diversity(PyObject *self, PyObject *arg) {
    (void)self;
    LZGFlashbackGrammar *g = fbg_from_capsule(arg);
    if (!g) return NULL;
    LZGEffectiveDiversity ed;
    LZGError err = lzg_flashback_grammar_effective_diversity(g, &ed);
    if (err != LZG_OK) return set_lzg_error(err);
    PyObject *d = PyDict_New();
    PyDict_SetItemString(d, "entropy_nats", PyFloat_FromDouble(ed.entropy_nats));
    PyDict_SetItemString(d, "entropy_bits", PyFloat_FromDouble(ed.entropy_bits));
    PyDict_SetItemString(d, "effective_diversity",
        PyFloat_FromDouble(ed.effective_diversity));
    PyDict_SetItemString(d, "uniformity", PyFloat_FromDouble(ed.uniformity));
    return d;
}

/* ── fbg_power_sum / fbg_hill_number / fbg_hill_numbers ───── */

static PyObject *py_fbg_power_sum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    double M = 0.0;
    LZGError err = lzg_flashback_grammar_power_sum(g, alpha, &M);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(M);
}

static PyObject *py_fbg_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    double D = 0.0;
    LZGError err = lzg_flashback_grammar_hill_number(g, alpha, &D);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(D);
}

static PyObject *py_fbg_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    PyObject *alpha_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &alpha_list))
        return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(alpha_list);
    double *alphas = (double *)malloc((size_t)n * sizeof(double));
    if (!alphas) return PyErr_NoMemory();
    for (Py_ssize_t i = 0; i < n; i++) {
        alphas[i] = PyFloat_AsDouble(PyList_GET_ITEM(alpha_list, i));
        if (PyErr_Occurred()) { free(alphas); return NULL; }
    }
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!out) { free(alphas); return PyErr_NoMemory(); }

    LZGError err = lzg_flashback_grammar_hill_numbers(g, alphas, (uint32_t)n, out);
    free(alphas);
    if (err != LZG_OK) { free(out); return set_lzg_error(err); }

    PyObject *list = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(out[i]));
    free(out);
    return list;
}

/* ── fbg_simulate / fbg_top_k_sequences / fbg_dynamic_range ── */

static PyObject *py_fbg_simulate(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int n;
    long long seed = -1;
    static char *kwlist[] = {"grammar", "n", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|L", kwlist,
            &cap, &n, &seed)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    LZGRng rng;
    if (seed >= 0) lzg_rng_seed(&rng, (uint64_t)seed);
    else lzg_rng_seed(&rng, (uint64_t)((size_t)cap ^ 0xFB65A7ULL));

    LZGSimResult *results = (LZGSimResult *)calloc(n, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();
    LZGError err = lzg_flashback_grammar_simulate(g, n, &rng, results);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *seqs = PyList_New(n);
    PyObject *lps  = PyList_New(n);
    PyObject *nts  = PyList_New(n);
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(seqs, i, PyUnicode_FromString(
            results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lps,  i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nts,  i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(OOO)", seqs, lps, nts);
}

static PyObject *py_fbg_top_k_sequences(PyObject *self, PyObject *args,
                                         PyObject *kw) {
    (void)self;
    PyObject *cap;
    unsigned int K;
    int most_probable = 1;
    unsigned int L_max = 30;
    static char *kwlist[] = {"grammar", "k", "most_probable", "max_length", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OI|pI", kwlist,
            &cap, &K, &most_probable, &L_max)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    LZGSimResult *results = (LZGSimResult *)calloc(K, sizeof(LZGSimResult));
    if (!results) return PyErr_NoMemory();
    uint32_t n_out = 0;
    LZGError err = lzg_flashback_grammar_top_k_sequences(g, K, (bool)most_probable,
                                            L_max, results, &n_out);
    if (err != LZG_OK) { free(results); return set_lzg_error(err); }

    PyObject *seqs = PyList_New(n_out);
    PyObject *lps  = PyList_New(n_out);
    PyObject *nts  = PyList_New(n_out);
    for (uint32_t i = 0; i < n_out; i++) {
        PyList_SET_ITEM(seqs, i, PyUnicode_FromString(
            results[i].sequence ? results[i].sequence : ""));
        PyList_SET_ITEM(lps,  i, PyFloat_FromDouble(results[i].log_prob));
        PyList_SET_ITEM(nts,  i, PyLong_FromUnsignedLong(results[i].n_tokens));
        lzg_sim_result_free(&results[i]);
    }
    free(results);
    return Py_BuildValue("(OOO)", seqs, lps, nts);
}

static PyObject *py_fbg_dynamic_range(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned int L_max;
    if (!PyArg_ParseTuple(args, "OI", &cap, &L_max)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    LZGDynamicRange dr;
    LZGError err = lzg_flashback_grammar_dynamic_range(g, L_max, &dr);
    if (err != LZG_OK) return set_lzg_error(err);
    PyObject *d = PyDict_New();
    PyDict_SetItemString(d, "max_log_prob", PyFloat_FromDouble(dr.max_log_prob));
    PyDict_SetItemString(d, "min_log_prob", PyFloat_FromDouble(dr.min_log_prob));
    PyDict_SetItemString(d, "dynamic_range_nats",
                          PyFloat_FromDouble(dr.dynamic_range_nats));
    PyDict_SetItemString(d, "dynamic_range_orders",
                          PyFloat_FromDouble(dr.dynamic_range_orders));
    return d;
}

/* ── fbg_posterior / fbg_subtract ─────────────────────────── */

static PyObject *py_fbg_posterior(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    PyObject *seq_list;
    PyObject *abund_obj = Py_None;
    double kappa = 1.0;
    static char *kwlist[] = {"grammar", "sequences", "abundances", "kappa", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|Od", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj, &kappa))
        return NULL;
    LZGFlashbackGrammar *prior = fbg_from_capsule(cap);
    if (!prior) return NULL;

    Py_ssize_t n_seqs;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGFlashbackGrammar *post = NULL;
    LZGError err = lzg_flashback_grammar_posterior(prior, seqs, (uint32_t)n_seqs,
                                      abundances, kappa, &post);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);

    return PyCapsule_New(post, FBG_CAPSULE_NAME, fbg_capsule_destructor);
}

static PyObject *py_fbg_subtract(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *cap;
    PyObject *seq_list;
    PyObject *abund_obj = Py_None;
    static char *kwlist[] = {"grammar", "sequences", "abundances", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO!|O", kwlist,
            &cap, &PyList_Type, &seq_list, &abund_obj))
        return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;

    Py_ssize_t n_seqs;
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    uint64_t *abundances = NULL;
    if (abund_obj != Py_None) {
        abundances = pylist_to_u64_array(abund_obj, n_seqs, "abundances");
        if (!abundances) { free(seqs); return NULL; }
    }

    LZGFlashbackGrammar *sub_g = NULL;
    LZGError err = lzg_flashback_grammar_subtract(g, seqs, (uint32_t)n_seqs,
                                     abundances, &sub_g);
    free(seqs); free(abundances);
    if (err != LZG_OK) return set_lzg_error(err);

    return PyCapsule_New(sub_g, FBG_CAPSULE_NAME, fbg_capsule_destructor);
}

/* ── fbg_save / fbg_load ──────────────────────────────────── */

static PyObject *py_fbg_save(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *path;
    if (!PyArg_ParseTuple(args, "Os", &cap, &path)) return NULL;
    LZGFlashbackGrammar *g = fbg_from_capsule(cap);
    if (!g) return NULL;
    LZGError err = lzg_flashback_grammar_save(g, path);
    if (err != LZG_OK) return set_lzg_error(err);
    Py_RETURN_NONE;
}

static PyObject *py_fbg_load(PyObject *self, PyObject *arg) {
    (void)self;
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "fbg_load: expected a str path");
        return NULL;
    }
    const char *path = PyUnicode_AsUTF8(arg);
    if (!path) return NULL;
    LZGFlashbackGrammar *g = NULL;
    LZGError err = lzg_flashback_grammar_load(path, &g);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyCapsule_New(g, FBG_CAPSULE_NAME, fbg_capsule_destructor);
}

/* ── Module definition ────────────────────────────────────── */

static struct PyModuleDef clzgraph_module = {
    PyModuleDef_HEAD_INIT,
    "_clzgraph",
    "C-LZGraph Python bindings",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__clzgraph(void) {
    if (PyType_Ready(&OwnedReadonlyBufferType) < 0) return NULL;
    PyObject *m = PyModule_Create(&clzgraph_module);
    if (!m) return NULL;

    /* Import custom exception classes from LZGraphs._errors */
    PyObject *errors_mod = PyImport_ImportModule("LZGraphs._errors");
    if (errors_mod) {
        LZGExc_NoGeneDataError = PyObject_GetAttrString(errors_mod, "NoGeneDataError");
        LZGExc_ConvergenceError = PyObject_GetAttrString(errors_mod, "ConvergenceError");
        LZGExc_CorruptFileError = PyObject_GetAttrString(errors_mod, "CorruptFileError");
        Py_DECREF(errors_mod);
    }
    /* Fallback: if import fails, use stdlib exceptions */
    if (!LZGExc_NoGeneDataError) LZGExc_NoGeneDataError = PyExc_RuntimeError;
    if (!LZGExc_ConvergenceError) LZGExc_ConvergenceError = PyExc_RuntimeError;
    if (!LZGExc_CorruptFileError) LZGExc_CorruptFileError = PyExc_OSError;

    return m;
}
