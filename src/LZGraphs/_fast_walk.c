/*
 * _fast_walk.c — CPython C extension for fast Markov chain random walks.
 *
 * Implements the full simulate() loop in C including string assembly,
 * for ~100-200x speedup over the original pure-Python implementation.
 * Uses xoshiro256++ for fast, high-quality RNG.
 *
 * The extension is optional: if it fails to compile (no C compiler),
 * LZGraphs falls back to the pure-Python bisect-based implementation.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* ========================================================================
 * xoshiro256++ RNG — public domain by David Blackman and Sebastiano Vigna
 * ======================================================================== */

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

typedef struct {
    uint64_t s[4];
} xoshiro256_state;

static inline uint64_t xoshiro256pp_next(xoshiro256_state *state) {
    const uint64_t result = rotl(state->s[0] + state->s[3], 23) + state->s[0];
    const uint64_t t = state->s[1] << 17;
    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];
    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);
    return result;
}

static inline double xoshiro256pp_double(xoshiro256_state *state) {
    return (double)(xoshiro256pp_next(state) >> 11) * 0x1.0p-53;
}

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void seed_xoshiro256(xoshiro256_state *state, uint64_t seed) {
    state->s[0] = splitmix64(&seed);
    state->s[1] = splitmix64(&seed);
    state->s[2] = splitmix64(&seed);
    state->s[3] = splitmix64(&seed);
}

/* ========================================================================
 * Binary search (bisect_left) on a double array
 * ======================================================================== */

static inline Py_ssize_t bisect_left_double(
    const double *arr, Py_ssize_t n, double value
) {
    Py_ssize_t lo = 0, hi = n;
    while (lo < hi) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* ========================================================================
 * simulate_walks — full simulation with string assembly in C
 *
 * Args:
 *   n_walks       : int
 *   offsets       : intp array [n_nodes+1] (buffer)
 *   neighbors     : intp array [total_edges] (buffer)
 *   cumweights    : float64 array [total_edges] (buffer)
 *   stop_probs    : float64 array [n_nodes] (buffer)
 *   initial_ids   : intp array [n_initial] (buffer)
 *   initial_cw    : float64 array [n_initial] (buffer)
 *   seed          : uint64
 *   clean_labels  : list[str] — label for each node ID
 *   return_walks  : bool — if True, return (walk, seq) tuples
 *   id_to_node    : list[str] — node names (only used if return_walks)
 *
 * Returns:
 *   list[str]  or  list[tuple[list[str], str]]
 * ======================================================================== */

static PyObject* py_simulate_walks(PyObject *self, PyObject *args) {
    int n_walks, return_walks;
    Py_buffer offsets_buf, neighbors_buf, cumweights_buf;
    Py_buffer stop_probs_buf, initial_ids_buf, initial_cw_buf;
    unsigned long long seed;
    PyObject *clean_labels;  /* Python list of str */
    PyObject *id_to_node;    /* Python list of str */
    PyObject *result_list = NULL;

    if (!PyArg_ParseTuple(args, "iy*y*y*y*y*y*KOpO",
            &n_walks,
            &offsets_buf, &neighbors_buf, &cumweights_buf,
            &stop_probs_buf, &initial_ids_buf, &initial_cw_buf,
            &seed,
            &clean_labels,
            &return_walks,
            &id_to_node))
        return NULL;

    const Py_ssize_t *offsets = (const Py_ssize_t *)offsets_buf.buf;
    const Py_ssize_t *neighbors = (const Py_ssize_t *)neighbors_buf.buf;
    const double *cumweights = (const double *)cumweights_buf.buf;
    const double *stop_probs = (const double *)stop_probs_buf.buf;
    const Py_ssize_t *initial_ids = (const Py_ssize_t *)initial_ids_buf.buf;
    const double *initial_cw = (const double *)initial_cw_buf.buf;
    const Py_ssize_t n_initial = initial_cw_buf.len / (Py_ssize_t)sizeof(double);

    if (n_initial <= 0) {
        PyErr_SetString(PyExc_ValueError,
            "Cannot simulate: graph has no initial states.");
        goto cleanup;
    }

    /* Pre-fetch label UTF-8 data for fast string assembly */
    const Py_ssize_t n_labels = PyList_GET_SIZE(clean_labels);
    const char **label_ptrs = (const char **)PyMem_Malloc(n_labels * sizeof(char *));
    Py_ssize_t *label_lens = (Py_ssize_t *)PyMem_Malloc(n_labels * sizeof(Py_ssize_t));
    if (!label_ptrs || !label_lens) {
        PyMem_Free(label_ptrs);
        PyMem_Free(label_lens);
        PyErr_NoMemory();
        goto cleanup;
    }
    for (Py_ssize_t i = 0; i < n_labels; i++) {
        PyObject *s = PyList_GET_ITEM(clean_labels, i);
        label_ptrs[i] = PyUnicode_AsUTF8AndSize(s, &label_lens[i]);
        if (!label_ptrs[i]) {
            PyMem_Free(label_ptrs);
            PyMem_Free(label_lens);
            goto cleanup;
        }
    }

    xoshiro256_state rng;
    seed_xoshiro256(&rng, (uint64_t)seed);

    result_list = PyList_New(n_walks);
    if (!result_list) {
        PyMem_Free(label_ptrs);
        PyMem_Free(label_lens);
        goto cleanup;
    }

    /* Reusable walk buffer */
    Py_ssize_t walk_cap = 64;
    Py_ssize_t *walk_buf = (Py_ssize_t *)PyMem_Malloc(walk_cap * sizeof(Py_ssize_t));
    /* Reusable string buffer */
    Py_ssize_t str_cap = 256;
    char *str_buf = (char *)PyMem_Malloc(str_cap);
    if (!walk_buf || !str_buf) {
        PyMem_Free(walk_buf);
        PyMem_Free(str_buf);
        PyMem_Free(label_ptrs);
        PyMem_Free(label_lens);
        Py_DECREF(result_list);
        PyErr_NoMemory();
        goto cleanup;
    }

    for (int i = 0; i < n_walks; i++) {
        /* Pick initial state */
        double r = xoshiro256pp_double(&rng);
        Py_ssize_t init_idx = bisect_left_double(initial_cw, n_initial, r);
        if (init_idx >= n_initial) init_idx = n_initial - 1;
        Py_ssize_t current = initial_ids[init_idx];

        Py_ssize_t walk_len = 0;
        walk_buf[walk_len++] = current;

        /* Build string incrementally */
        Py_ssize_t str_len = 0;
        Py_ssize_t llen = label_lens[current];
        if (str_len + llen > str_cap) {
            str_cap = (str_len + llen) * 2;
            str_buf = (char *)PyMem_Realloc(str_buf, str_cap);
            if (!str_buf) goto oom;
        }
        memcpy(str_buf + str_len, label_ptrs[current], llen);
        str_len += llen;

        while (1) {
            double sp = stop_probs[current];
            if (sp == sp) {
                if (xoshiro256pp_double(&rng) < sp)
                    break;
            }

            Py_ssize_t start = offsets[current];
            Py_ssize_t end = offsets[current + 1];
            if (start == end)
                break;

            r = xoshiro256pp_double(&rng);
            Py_ssize_t idx = bisect_left_double(cumweights + start, end - start, r);
            if (idx >= end - start) idx = end - start - 1;
            current = neighbors[start + idx];

            /* Grow walk buffer if needed */
            if (walk_len >= walk_cap) {
                walk_cap *= 2;
                Py_ssize_t *new_buf = (Py_ssize_t *)PyMem_Realloc(walk_buf, walk_cap * sizeof(Py_ssize_t));
                if (!new_buf) goto oom;
                walk_buf = new_buf;
            }
            walk_buf[walk_len++] = current;

            /* Append label to string buffer */
            llen = label_lens[current];
            if (str_len + llen > str_cap) {
                str_cap = (str_len + llen) * 2;
                char *new_str = (char *)PyMem_Realloc(str_buf, str_cap);
                if (!new_str) goto oom;
                str_buf = new_str;
            }
            memcpy(str_buf + str_len, label_ptrs[current], llen);
            str_len += llen;
        }

        /* Create Python string from buffer */
        PyObject *seq = PyUnicode_FromStringAndSize(str_buf, str_len);
        if (!seq) goto oom;

        if (return_walks) {
            /* Build walk list of node name strings */
            PyObject *walk = PyList_New(walk_len);
            if (!walk) { Py_DECREF(seq); goto oom; }
            for (Py_ssize_t j = 0; j < walk_len; j++) {
                PyObject *node_name = PyList_GET_ITEM(id_to_node, walk_buf[j]);
                Py_INCREF(node_name);
                PyList_SET_ITEM(walk, j, node_name);
            }
            PyObject *tup = PyTuple_Pack(2, walk, seq);
            Py_DECREF(walk);
            Py_DECREF(seq);
            if (!tup) goto oom;
            PyList_SET_ITEM(result_list, i, tup);
        } else {
            PyList_SET_ITEM(result_list, i, seq);
        }
    }

    PyMem_Free(walk_buf);
    PyMem_Free(str_buf);
    PyMem_Free(label_ptrs);
    PyMem_Free(label_lens);
    goto cleanup;

oom:
    PyMem_Free(walk_buf);
    PyMem_Free(str_buf);
    PyMem_Free(label_ptrs);
    PyMem_Free(label_lens);
    Py_XDECREF(result_list);
    result_list = NULL;
    if (!PyErr_Occurred())
        PyErr_NoMemory();

cleanup:
    PyBuffer_Release(&offsets_buf);
    PyBuffer_Release(&neighbors_buf);
    PyBuffer_Release(&cumweights_buf);
    PyBuffer_Release(&stop_probs_buf);
    PyBuffer_Release(&initial_ids_buf);
    PyBuffer_Release(&initial_cw_buf);

    return result_list;
}

/* ========================================================================
 * Module definition
 * ======================================================================== */

static PyMethodDef FastWalkMethods[] = {
    {"simulate_walks", py_simulate_walks, METH_VARARGS,
     "Run n random walks on a CSR-encoded graph with string assembly.\n\n"
     "Args:\n"
     "    n_walks (int): Number of walks.\n"
     "    offsets (array): CSR row offsets [n_nodes+1], dtype=intp.\n"
     "    neighbors (array): Flat neighbor IDs, dtype=intp.\n"
     "    cumweights (array): Flat cumulative weights, dtype=float64.\n"
     "    stop_probs (array): Per-node stop probability (NaN=none), dtype=float64.\n"
     "    initial_ids (array): Initial state IDs, dtype=intp.\n"
     "    initial_cumprobs (array): Cumulative initial probs, dtype=float64.\n"
     "    seed (int): RNG seed (xoshiro256++).\n"
     "    clean_labels (list[str]): Subpattern label for each node.\n"
     "    return_walks (bool): If True, return (walk, seq) tuples.\n"
     "    id_to_node (list[str]): Node names for walk output.\n\n"
     "Returns:\n"
     "    list[str] or list[tuple[list[str], str]]\n"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fast_walk_module = {
    PyModuleDef_HEAD_INIT,
    "_fast_walk",
    "C-accelerated random walk simulation for LZGraphs.\n"
    "Uses xoshiro256++ RNG for high-quality, fast random number generation.\n"
    "This module is optional — LZGraphs falls back to pure Python if unavailable.",
    -1,
    FastWalkMethods
};

PyMODINIT_FUNC PyInit__fast_walk(void) {
    return PyModule_Create(&fast_walk_module);
}
