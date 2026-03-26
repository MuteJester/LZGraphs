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

#include "lzgraph/common.h"
#include "lzgraph/graph.h"
#include "lzgraph/lz76.h"
#include "lzgraph/simulate.h"
#include "lzgraph/analytics.h"
#include "lzgraph/pgen_dist.h"
#include "lzgraph/occupancy.h"
#include "lzgraph/sharing.h"
#include "lzgraph/diversity.h"
#include "lzgraph/graph_ops.h"
#include "lzgraph/features.h"
#include "lzgraph/io.h"
#include "lzgraph/posterior.h"
#include "lzgraph/gene_data.h"
#include "lzgraph/rng.h"

#include <string.h>
#include <math.h>

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
    const char **seqs = pylist_to_cstrings(seq_list, &n_seqs);
    if (!seqs) return NULL;

    /* Abundances */
    uint32_t *abundances = NULL;
    if (abund_obj != Py_None) {
        if (!PyList_Check(abund_obj)) {
            free(seqs);
            PyErr_SetString(PyExc_TypeError, "abundances must be a list");
            return NULL;
        }
        Py_ssize_t na = PyList_GET_SIZE(abund_obj);
        if (na != n_seqs) {
            free(seqs);
            PyErr_SetString(PyExc_ValueError, "abundances length must match sequences");
            return NULL;
        }
        abundances = (uint32_t *)malloc(na * sizeof(uint32_t));
        if (!abundances) { free(seqs); return PyErr_NoMemory(); }
        for (Py_ssize_t i = 0; i < na; i++)
            abundances[i] = (uint32_t)PyLong_AsUnsignedLong(PyList_GET_ITEM(abund_obj, i));
        if (PyErr_Occurred()) { free(seqs); free(abundances); return NULL; }
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

    LZGError err = lzg_graph_build(g, seqs, (uint32_t)n_seqs, abundances,
                                    v_genes, j_genes, smoothing, 0);
    free(seqs); free(abundances); free((void *)v_genes); free((void *)j_genes);

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

/* ── path_count(capsule) → float ─────────────────────────── */

static PyObject *py_path_count(PyObject *self, PyObject *arg) {
    (void)self;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(arg, CAPSULE_NAME);
    if (!g) return NULL;
    double count;
    LZGError err = lzg_graph_path_count(g, &count);
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
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;
    double d;
    LZGError err = lzg_hill_number(g, alpha, &d);
    if (err != LZG_OK) return set_lzg_error(err);
    return PyFloat_FromDouble(d);
}

/* ── hill_numbers(capsule, orders_list) → list[float] ─────── */

static PyObject *py_hill_numbers(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *orders_list;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyList_Type, &orders_list)) return NULL;
    LZGGraph *g = (LZGGraph *)PyCapsule_GetPointer(cap, CAPSULE_NAME);
    if (!g) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(orders_list);
    double *orders = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!orders || !out) { free(orders); free(out); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n; i++)
        orders[i] = PyFloat_AsDouble(PyList_GET_ITEM(orders_list, i));
    if (PyErr_Occurred()) { free(orders); free(out); return NULL; }

    LZGError err = lzg_hill_numbers(g, orders, (uint32_t)n, out);
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
    return Py_BuildValue("{s:d, s:d, s:d, s:O}",
        "total_absorbed", diag.total_absorbed,
        "total_leaked", diag.total_leaked,
        "initial_prob_sum", diag.initial_prob_sum,
        "is_proper", diag.is_proper ? Py_True : Py_False);
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

    PyObject *spec = PyList_New(ss.max_k + 1);
    for (uint32_t i = 0; i <= ss.max_k; i++)
        PyList_SET_ITEM(spec, i, PyFloat_FromDouble(ss.spectrum[i]));
    lzg_sharing_spectrum_free(&ss);

    return Py_BuildValue("{s:O, s:d, s:I}",
        "spectrum", spec, "expected_total", ss.expected_total,
        "n_donors", ss.n_donors);
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
    const char **seqs = pylist_to_cstrings(seq_list, &n);
    if (!seqs) return NULL;

    uint32_t *abundances = NULL;
    if (abund_obj != Py_None && PyList_Check(abund_obj)) {
        abundances = (uint32_t *)malloc(n * sizeof(uint32_t));
        if (!abundances) { free(seqs); return PyErr_NoMemory(); }
        for (Py_ssize_t i = 0; i < n; i++)
            abundances[i] = (uint32_t)PyLong_AsUnsignedLong(PyList_GET_ITEM(abund_obj, i));
    }

    LZGGraph *post = NULL;
    LZGError err = lzg_graph_posterior(g, seqs, (uint32_t)n, abundances, kappa, &post);
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
            PyObject *tup = Py_BuildValue("(NNdI)",
                src_label, dst_label, g->edge_weights[e], g->edge_counts[e]);
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
            PyObject *val = PyLong_FromUnsignedLong(g->length_counts[i]);
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
        PyList_SET_ITEM(ct, i, PyLong_FromUnsignedLong(g->edge_counts[i]));

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

static PyMethodDef module_methods[] = {
    {"graph_build",             (PyCFunction)py_graph_build,           METH_VARARGS | METH_KEYWORDS, NULL},
    {"graph_info",              py_graph_info,                         METH_O, NULL},
    {"simulate",                (PyCFunction)py_simulate,              METH_VARARGS | METH_KEYWORDS, NULL},
    {"gene_simulate",           (PyCFunction)py_gene_simulate,         METH_VARARGS | METH_KEYWORDS, NULL},
    {"lzpgen",                  py_lzpgen,                             METH_VARARGS, NULL},
    {"path_count",              py_path_count,                         METH_O, NULL},
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
    {"predict_sharing",         py_predict_sharing,                    METH_VARARGS, NULL},
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
    {"k_diversity",             (PyCFunction)py_k_diversity,            METH_VARARGS | METH_KEYWORDS, NULL},
    {"saturation_curve",        (PyCFunction)py_saturation_curve,      METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_log_level",           py_set_log_level,                      METH_O, NULL},
    {"set_log_callback",        py_set_log_callback,                   METH_VARARGS, NULL},
    {"graph_nodes",             py_graph_nodes,                        METH_O, NULL},
    {"graph_edges",             py_graph_edges,                        METH_O, NULL},
    {"graph_length_distribution", py_graph_length_distribution,        METH_O, NULL},
    {"graph_adjacency_csr",     py_graph_adjacency_csr,                METH_O, NULL},
    {"graph_degrees",           py_graph_degrees,                      METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

/* ── Module definition ────────────────────────────────────── */

static struct PyModuleDef clzgraph_module = {
    PyModuleDef_HEAD_INIT,
    "_clzgraph",
    "C-LZGraph Python bindings",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__clzgraph(void) {
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
