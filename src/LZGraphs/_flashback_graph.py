"""FlashBackGraph — Markovian graph from FlashBack decomposition."""

import numpy as np
from . import _clzgraph as _c
from ._simulation_result import SimulationResult


class FlashBackGraph:
    """FlashBack decomposition graph for sequence repertoire analysis.

    Builds a Markovian DAG where nodes are FlashBack tokens and edges
    are transitions between consecutive tokens. The FlashBack algorithm
    recursively peels matching character runs from both ends of the
    sentinel-wrapped sequence.

    Since the graph is Markovian, all analytics (path count, diversity,
    entropy, Hill numbers) are computed exactly via forward DP — no
    Monte Carlo approximation needed.

    Args:
        sequences: List of strings.
        abundances: Per-sequence counts. None = all 1.
        smoothing: Laplace smoothing alpha for edge weights.
    """

    def __init__(self, sequences, *, abundances=None, smoothing=0.0):
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        self._cap = _c.fb_graph_build(
            seqs,
            list(abundances) if abundances is not None else None,
            smoothing,
        )
        self._info = _c.graph_info(self._cap)

    @classmethod
    def _from_capsule(cls, capsule):
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._info = _c.graph_info(capsule)
        return obj

    @classmethod
    def from_file(cls, path, *, smoothing=0.0):
        """Build from a plain text file (streaming, constant memory).

        Supported: one sequence per line, or sequence<TAB>abundance.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        return cls._from_capsule(_c.fb_graph_build_file(path, smoothing))

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self):
        return f"FlashBackGraph(nodes={self.n_nodes}, edges={self.n_edges})"

    def __len__(self):
        return self.n_nodes

    def __contains__(self, sequence):
        return self.flashback_pgen(sequence) > -690.0

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    # ── Basic properties ────────────────────────────────────

    @property
    def n_nodes(self):
        return self._info['n_nodes']

    @property
    def n_edges(self):
        return self._info['n_edges']

    @property
    def variant(self):
        return 'flashback'

    @property
    def is_dag(self):
        return self._info['is_dag']

    @property
    def path_count(self):
        """Exact number of distinct walks (via DP)."""
        if not hasattr(self, '_path_count_cache'):
            self._path_count_cache = float(_c.fb_path_count(self._cap))
        return self._path_count_cache

    # ── Structural ──────────────────────────────────────────

    @property
    def n_sequences(self):
        return sum(self.length_distribution.values())

    @property
    def length_distribution(self):
        if not hasattr(self, '_length_dist_cache'):
            self._length_dist_cache = _c.graph_length_distribution(self._cap)
        return dict(self._length_dist_cache)

    @property
    def nodes(self):
        if not hasattr(self, '_nodes_cache'):
            raw = _c.graph_nodes(self._cap)
            self._all_nodes_cache = raw
            self._nodes_cache = [n for n in raw
                                 if not n.startswith('@') and '$' not in n]
        return list(self._nodes_cache)

    @property
    def all_nodes(self):
        if not hasattr(self, '_all_nodes_cache'):
            self._all_nodes_cache = _c.graph_nodes(self._cap)
        return list(self._all_nodes_cache)

    @property
    def edges(self):
        if not hasattr(self, '_edges_cache'):
            raw = _c.graph_edges(self._cap)
            self._all_edges_cache = raw
            self._edges_cache = [
                e for e in raw
                if not e[0].startswith('@') and '$' not in e[0]
                and not e[1].startswith('@') and '$' not in e[1]
            ]
        return list(self._edges_cache)

    @property
    def all_edges(self):
        if not hasattr(self, '_all_edges_cache'):
            self._all_edges_cache = _c.graph_edges(self._cap)
        return list(self._all_edges_cache)

    @property
    def n_initial(self):
        return self._get_summary()['n_initial']

    @property
    def n_terminal(self):
        return self._get_summary()['n_terminal']

    @property
    def max_out_degree(self):
        return self._get_summary()['max_out_degree']

    @property
    def max_in_degree(self):
        return self._get_summary()['max_in_degree']

    @property
    def density(self):
        n = self.n_nodes
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    @property
    def out_degrees(self):
        return np.array(self._get_degrees()['out_degrees'], dtype=np.uint32)

    @property
    def in_degrees(self):
        return np.array(self._get_degrees()['in_degrees'], dtype=np.uint32)

    @property
    def initial_nodes(self):
        """Node labels with in-degree 0 (excluding sentinel nodes)."""
        all_n = self.all_nodes
        in_deg = self._get_degrees()['in_degrees']
        return [n for n, d in zip(all_n, in_deg)
                if d == 0 and not n.startswith('@') and '$' not in n]

    @property
    def terminal_nodes(self):
        """Node labels with out-degree 0 (excluding sentinel nodes)."""
        all_n = self.all_nodes
        out_deg = self._get_degrees()['out_degrees']
        return [n for n, d in zip(all_n, out_deg)
                if d == 0 and not n.startswith('@') and '$' not in n]

    @property
    def hub_nodes(self):
        """Top nodes by total degree (in + out), descending.

        Returns list of (node_label, in_degree, out_degree) tuples,
        excluding sentinel nodes, sorted by total degree.
        """
        all_n = self.all_nodes
        degs = self._get_degrees()
        in_deg = degs['in_degrees']
        out_deg = degs['out_degrees']
        entries = [
            (n, int(i), int(o))
            for n, i, o in zip(all_n, in_deg, out_deg)
            if not n.startswith('@') and '$' not in n
        ]
        entries.sort(key=lambda x: x[1] + x[2], reverse=True)
        return entries

    @property
    def node_degree_map(self):
        """Dict mapping node label -> (in_degree, out_degree).

        Excludes sentinel nodes.
        """
        all_n = self.all_nodes
        degs = self._get_degrees()
        in_deg = degs['in_degrees']
        out_deg = degs['out_degrees']
        return {
            n: (int(i), int(o))
            for n, i, o in zip(all_n, in_deg, out_deg)
            if not n.startswith('@') and '$' not in n
        }

    @property
    def edge_weight_map(self):
        """Dict mapping (src, dst) -> transition probability.

        Excludes edges involving sentinel nodes.
        """
        return {(e[0], e[1]): e[2] for e in self.edges}

    @property
    def isolated_nodes(self):
        """Nodes with both in-degree and out-degree 0 (excluding sentinels)."""
        all_n = self.all_nodes
        degs = self._get_degrees()
        in_deg = degs['in_degrees']
        out_deg = degs['out_degrees']
        return [n for n, i, o in zip(all_n, in_deg, out_deg)
                if i == 0 and o == 0
                and not n.startswith('@') and '$' not in n]

    def _get_summary(self):
        if not hasattr(self, '_summary_cache'):
            self._summary_cache = _c.summary(self._cap)
        return self._summary_cache

    def _get_degrees(self):
        if not hasattr(self, '_degrees_cache'):
            self._degrees_cache = _c.graph_degrees(self._cap)
        return self._degrees_cache

    # ── Adjacency ───────────────────────────────────────────

    def adjacency_csr(self):
        """CSR (Compressed Sparse Row) adjacency representation."""
        csr = self._get_csr()
        return {
            'row_offsets': csr['row_offsets'].copy(),
            'col_indices': csr['col_indices'].copy(),
            'weights': csr['weights'].copy(),
            'counts': csr['counts'].copy(),
        }

    def _get_csr(self):
        if not hasattr(self, '_csr_cache'):
            raw = _c.graph_adjacency_csr(self._cap)
            self._csr_cache = {
                'row_offsets': np.array(raw['row_offsets'], dtype=np.uint32),
                'col_indices': np.array(raw['col_indices'], dtype=np.uint32),
                'weights': np.array(raw['weights'], dtype=np.float64),
                'counts': np.array(raw['counts'], dtype=np.uint64),
            }
        return self._csr_cache

    def _get_node_index(self):
        """Cached label → integer index dict."""
        if not hasattr(self, '_node_index'):
            self._node_index = {n: i for i, n in enumerate(self.all_nodes)}
        return self._node_index

    def _get_reverse_csr(self):
        """Build a reverse (transpose) CSR for predecessor lookups."""
        if not hasattr(self, '_rcsr_cache'):
            csr = self._get_csr()
            n = len(csr['row_offsets']) - 1
            col = csr['col_indices']
            w = csr['weights']
            c = csr['counts']
            # Count in-degree per node
            in_deg = np.zeros(n, dtype=np.uint32)
            for j in col:
                in_deg[j] += 1
            # Build row_offsets for transpose
            r_offsets = np.zeros(n + 1, dtype=np.uint32)
            np.cumsum(in_deg, out=r_offsets[1:])
            # Fill columns/weights/counts
            r_col = np.empty(len(col), dtype=np.uint32)
            r_w = np.empty(len(col), dtype=np.float64)
            r_c = np.empty(len(col), dtype=np.uint64)
            pos = r_offsets[:-1].copy()
            for src in range(n):
                for e in range(csr['row_offsets'][src], csr['row_offsets'][src + 1]):
                    dst = col[e]
                    p = pos[dst]
                    r_col[p] = src
                    r_w[p] = w[e]
                    r_c[p] = c[e]
                    pos[dst] += 1
            self._rcsr_cache = {
                'row_offsets': r_offsets,
                'col_indices': r_col,
                'weights': r_w,
                'counts': r_c,
            }
        return self._rcsr_cache

    def successors(self, node_label):
        """Get successor nodes with edge weights.

        Returns list of (target_label, weight, count) tuples.
        """
        idx_map = self._get_node_index()
        idx = idx_map.get(node_label)
        if idx is None:
            raise KeyError(f"node '{node_label}' not found in graph")
        csr = self._get_csr()
        nodes = self._all_nodes_cache
        start = csr['row_offsets'][idx]
        end = csr['row_offsets'][idx + 1]
        col = csr['col_indices']
        w = csr['weights']
        c = csr['counts']
        return [(nodes[col[e]], float(w[e]), int(c[e]))
                for e in range(start, end)]

    def predecessors(self, node_label):
        """Get predecessor nodes with edge weights.

        Returns list of (source_label, weight, count) tuples.
        """
        idx_map = self._get_node_index()
        idx = idx_map.get(node_label)
        if idx is None:
            raise KeyError(f"node '{node_label}' not found in graph")
        rcsr = self._get_reverse_csr()
        nodes = self._all_nodes_cache
        start = rcsr['row_offsets'][idx]
        end = rcsr['row_offsets'][idx + 1]
        col = rcsr['col_indices']
        w = rcsr['weights']
        c = rcsr['counts']
        return [(nodes[col[e]], float(w[e]), int(c[e]))
                for e in range(start, end)]

    # ── Simulation ──────────────────────────────────────────

    def simulate(self, n, *, seed=None):
        """Generate n sequences by Markov random walk.

        Returns SimulationResult (iterable of sequences with .log_probs).
        """
        seed_val = seed if seed is not None else -1
        seqs, lps, nts = _c.fb_simulate(self._cap, n, seed=seed_val)
        return SimulationResult(seqs, lps, nts)

    def top_k_walks(self, k=100, *, most_probable=True):
        """Find the K most (or least) probable walks through the graph.

        Uses exact forward DP on the DAG's topological order — no simulation
        or approximation. Returns a SimulationResult with sequences sorted
        by probability (descending if most_probable, ascending otherwise).

        Args:
            k: Number of walks to return.
            most_probable: If True, return highest-probability walks.
                          If False, return lowest-probability walks.

        Returns:
            SimulationResult with .sequences, .log_probs attributes.
        """
        seqs, lps = _c.fb_top_k_walks(self._cap, k=k, most_probable=most_probable)
        nts = [0] * len(seqs)  # token counts not tracked in top-k
        return SimulationResult(seqs, lps, nts)

    # ── Probability ─────────────────────────────────────────

    def flashback_pgen(self, sequence, *, log=True):
        """Exact probability of sequence(s) under the FlashBack model.

        Args:
            sequence: A single string, or a list of strings.
            log: If True, return log-probability.
        """
        raw = _c.fb_pgen(self._cap, sequence)
        if isinstance(raw, float):
            return raw if log else np.exp(raw)
        arr = np.array(raw, dtype=np.float64)
        return arr if log else np.exp(arr)

    # ── Exact analytics ────────────────────────────────────

    def effective_diversity(self):
        """Exact effective diversity exp(H) via forward DP."""
        return _c.fb_effective_diversity(self._cap)['effective_diversity']

    def diversity_profile(self):
        """Full Shannon diversity breakdown (exact)."""
        return _c.fb_effective_diversity(self._cap)

    def hill_number(self, alpha):
        """Exact Hill number D(alpha) via forward DP."""
        return _c.fb_hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders):
        """Exact Hill numbers for multiple orders."""
        result = _c.fb_hill_numbers(self._cap, [float(o) for o in orders])
        return np.array(result, dtype=np.float64)

    def hill_curve(self, orders=None):
        """Hill diversity curve (exact)."""
        if orders is None:
            orders = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
        o_list = [float(o) for o in orders]
        result = _c.fb_hill_numbers(self._cap, o_list)
        return {
            'orders': np.array(o_list, dtype=np.float64),
            'values': np.array(result, dtype=np.float64),
        }

    def power_sum(self, alpha):
        """Exact power sum M(alpha) via forward DP."""
        return _c.fb_power_sum(self._cap, float(alpha))

    def pgen_diagnostics(self, atol=1e-6):
        """Exact absorbed vs leaked mass (via DP)."""
        return _c.fb_pgen_diagnostics(self._cap, atol)

    def pgen_dynamic_range(self):
        """Exact dynamic range in orders of magnitude."""
        return _c.fb_dynamic_range(self._cap)['dynamic_range_orders']

    def pgen_dynamic_range_detail(self):
        """Full dynamic range breakdown (exact)."""
        return _c.fb_dynamic_range(self._cap)

    # ── PGEN Distribution ──────────────────────────────────

    def pgen_moments(self):
        """Moments of the forward-DP log-PGEN distribution."""
        return _c.pgen_moments(self._cap)

    def pgen_distribution(self):
        """Analytical Gaussian mixture approximation of log-PGEN."""
        from ._pgen_dist import PgenDistribution
        raw = _c.pgen_analytical(self._cap)
        return PgenDistribution(raw)

    # ── Graph operations ───────────────────────────────────

    def union(self, other):
        cap = _c.graph_union(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def intersection(self, other):
        cap = _c.graph_intersection(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def difference(self, other):
        cap = _c.graph_difference(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def weighted_merge(self, other, alpha=1.0, beta=1.0):
        cap = _c.weighted_merge(self._cap, other._cap, alpha, beta)
        return FlashBackGraph._from_capsule(cap)

    def posterior(self, sequences, *, abundances=None, kappa=1.0):
        """Bayesian posterior graph given new observed sequences.

        Keeps the prior's topology and updates each edge weight via the
        Dirichlet-Multinomial rule:

            w_post(u->v) = (kappa * w_prior(u->v) + c_ind(u->v))
                           / (kappa + n_ind(u))

        where c_ind and n_ind are the individual's edge count and total
        outgoing count at u, derived from FlashBack decomposition of
        ``sequences``. kappa=0 -> pure individual, kappa->inf -> pure prior.
        Sequences inducing edges not present in the prior are ignored.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        cap = _c.fb_posterior(self._cap, seqs, abundances=abs_list, kappa=float(kappa))
        return FlashBackGraph._from_capsule(cap)

    def without(self, sequences, *, abundances=None):
        """Return a new graph with the contribution of ``sequences`` removed.

        For each sequence (with abundance ``a``, defaulting to 1) the walk
        through the graph's FlashBack decomposition is traced and ``a`` is
        subtracted from every edge's raw count on that walk. Subtraction is
        clamped at zero. Edges whose count reaches zero are physically
        pruned; isolated nodes are retained for node-index stability and are
        filtered by the Python accessors. Weights are renormalised per
        source node after pruning.

        Use case: leave-donor-out foundation construction from an existing
        graph, in seconds instead of rebuilding from source data.

        Args:
            sequences: list of strings to remove.
            abundances: per-sequence counts. None defaults to 1 each.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        cap = _c.fb_subtract(self._cap, seqs, abundances=abs_list)
        return FlashBackGraph._from_capsule(cap)

    # Alias — same operation, more explicit name when you want it to read as
    # an imperative ("remove these sequences from the graph").
    def remove_sequences(self, sequences, *, abundances=None):
        """Alias for :meth:`without`. Same semantics, more explicit name."""
        return self.without(sequences, abundances=abundances)

    # ── Features ───────────────────────────────────────────

    def feature_stats(self):
        return np.array(_c.feature_stats(self._cap), dtype=np.float64)

    def feature_mass_profile(self, max_pos=30):
        return np.array(_c.feature_mass_profile(self._cap, max_pos), dtype=np.float64)

    # ── IO ─────────────────────────────────────────────────

    def save(self, path):
        _c.save(self._cap, str(path))

    @classmethod
    def load(cls, path):
        cap = _c.load(str(path))
        _c.fb_fix_special_nodes(cap)
        return cls._from_capsule(cap)

    def summary(self):
        return _c.summary(self._cap)
