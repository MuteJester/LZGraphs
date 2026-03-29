"""LZGraph — Python wrapper around C-LZGraph."""

import numpy as np
from . import _clzgraph as _c
from ._errors import LZGraphError, NoGeneDataError
from ._simulation_result import SimulationResult


class LZGraph:
    """LZ76 compression graph for sequence repertoire analysis.

    Wraps the C-LZGraph library for high-performance graph construction,
    simulation, and analytics with full LZ76 dictionary constraint enforcement.

    Args:
        sequences: List of CDR3 amino acid (aap) or nucleotide (ndp/naive) strings.
        variant: Graph encoding variant — 'aap', 'ndp', or 'naive'.
        abundances: Per-sequence counts. None = all 1.
        v_genes: V gene annotation per sequence. None = no gene data.
        j_genes: J gene annotation per sequence. None = no gene data.
        smoothing: Laplace smoothing alpha for edge weights.
    """

    def __init__(
        self,
        sequences,
        *,
        variant='aap',
        abundances=None,
        v_genes=None,
        j_genes=None,
        smoothing=0.0,
    ):
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        self._cap = _c.graph_build(
            seqs, variant,
            list(abundances) if abundances is not None else None,
            list(v_genes) if v_genes is not None else None,
            list(j_genes) if j_genes is not None else None,
            smoothing,
        )
        self._info = _c.graph_info(self._cap)
        self._gene_cache = None

    @classmethod
    def _from_capsule(cls, capsule):
        """Internal: wrap an existing C capsule."""
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._info = _c.graph_info(capsule)
        obj._gene_cache = None
        return obj

    @classmethod
    def from_file(cls, path, *, variant='aap', smoothing=0.0,
                  strict_input=False, expect_format=None):
        """Build directly from a plain text file without Python list materialization.

        Supported file formats:
        - one sequence per line
        - ``sequence<TAB>abundance``

        This path is intended for large plain repertoire files and does not
        support headered tabular inputs or gene columns.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path:
            raise ValueError("path must be non-empty")
        if strict_input or expect_format is not None:
            from ._io import validate_input
            report = validate_input(
                path,
                variant=variant,
                strict_input=strict_input,
                expect_format=expect_format,
            )
            if not report['ok']:
                raise ValueError(report['summary'])
        return cls._from_capsule(
            _c.graph_build_file(path, variant, smoothing)
        )

    # ── Dunder methods ──────────────────────────────────────

    def __repr__(self):
        return (f"LZGraph(variant='{self.variant}', "
                f"nodes={self.n_nodes}, edges={self.n_edges})")

    def __len__(self):
        return self.n_nodes

    def __contains__(self, sequence):
        return self.lzpgen(sequence) > -690.0

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    # ── Basic properties ──────────────────────────────────────

    @property
    def n_nodes(self):
        """Number of nodes in the graph (including sentinel nodes)."""
        return self._info['n_nodes']

    @property
    def n_edges(self):
        """Number of directed edges in the graph."""
        return self._info['n_edges']

    @property
    def variant(self):
        """Graph encoding variant: 'aap', 'ndp', or 'naive'."""
        return self._info['variant']

    @property
    def has_gene_data(self):
        """Whether V/J gene annotation data is available."""
        return self._info['has_gene_data']

    @property
    def is_dag(self):
        """Whether the graph is a directed acyclic graph."""
        return self._info['is_dag']

    @property
    def path_count(self):
        """Estimated number of distinct LZ-valid walks (Chao1 lower bound)."""
        if not hasattr(self, '_path_count_cache'):
            self._path_count_cache = int(_c.path_count(self._cap))
        return self._path_count_cache

    # ── Structural properties ────────────────────────────────

    @property
    def n_sequences(self):
        """Total number of sequences used to build the graph."""
        return sum(self.length_distribution.values())

    @property
    def length_distribution(self):
        """Sequence length distribution: {length: count}.

        Example:
            >>> graph.length_distribution
            {10: 42, 11: 138, 12: 267, ...}
        """
        if not hasattr(self, '_length_dist_cache'):
            self._length_dist_cache = _c.graph_length_distribution(self._cap)
        return dict(self._length_dist_cache)

    @property
    def nodes(self):
        """List of node label strings (excluding @ and $ sentinels).

        For AAP variant: labels like 'C_1', 'A_2', 'SL_5'.
        For NDP variant: labels like 'T_1', 'G_2', 'TG_4'.
        For Naive variant: labels like 'C', 'A', 'SL'.

        Use ``all_nodes`` to include sentinel nodes.
        """
        if not hasattr(self, '_nodes_cache'):
            raw = _c.graph_nodes(self._cap)
            self._all_nodes_cache = raw
            self._nodes_cache = [n for n in raw
                                 if not n.startswith('@') and '$' not in n]
        return list(self._nodes_cache)

    @property
    def all_nodes(self):
        """List of all node label strings, including @ and $ sentinels."""
        if not hasattr(self, '_all_nodes_cache'):
            self._all_nodes_cache = _c.graph_nodes(self._cap)
        return list(self._all_nodes_cache)

    @property
    def edges(self):
        """List of (source, target, weight, count) tuples for all edges.

        - source/target: node label strings
        - weight: transition probability P(target | source)
        - count: raw transition count

        Excludes edges involving @ and $ sentinel nodes.
        Use ``all_edges`` to include sentinel edges.
        """
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
        """List of all (source, target, weight, count) tuples, including sentinels."""
        if not hasattr(self, '_all_edges_cache'):
            self._all_edges_cache = _c.graph_edges(self._cap)
        return list(self._all_edges_cache)

    @property
    def n_initial(self):
        """Number of initial states (nodes reachable from root)."""
        return self._get_summary()['n_initial']

    @property
    def n_terminal(self):
        """Number of terminal (sink) nodes."""
        return self._get_summary()['n_terminal']

    @property
    def max_out_degree(self):
        """Maximum out-degree of any node."""
        return self._get_summary()['max_out_degree']

    @property
    def max_in_degree(self):
        """Maximum in-degree of any node."""
        return self._get_summary()['max_in_degree']

    @property
    def density(self):
        """Graph density: n_edges / (n_nodes * (n_nodes - 1)).

        For a DAG with n nodes, the maximum possible edges is n*(n-1)/2,
        so density ranges from 0 (no edges) to ~0.5 (complete DAG).
        """
        n = self.n_nodes
        if n <= 1:
            return 0.0
        return self.n_edges / (n * (n - 1))

    @property
    def out_degrees(self):
        """Out-degree of each node as a numpy array (indexed by node ID)."""
        return np.array(self._get_degrees()['out_degrees'], dtype=np.uint32)

    @property
    def in_degrees(self):
        """In-degree of each node as a numpy array (indexed by node ID)."""
        return np.array(self._get_degrees()['in_degrees'], dtype=np.uint32)

    def _get_summary(self):
        if not hasattr(self, '_summary_cache'):
            self._summary_cache = _c.summary(self._cap)
        return self._summary_cache

    def _get_degrees(self):
        if not hasattr(self, '_degrees_cache'):
            self._degrees_cache = _c.graph_degrees(self._cap)
        return self._degrees_cache

    # ── Adjacency ────────────────────────────────────────────

    def adjacency_csr(self):
        """CSR (Compressed Sparse Row) adjacency representation.

        Returns a dict with numpy arrays ready for scipy.sparse.csr_matrix:
            - row_offsets: np.ndarray[uint32] of shape (n_nodes + 1,)
            - col_indices: np.ndarray[uint32] of shape (n_edges,)
            - weights: np.ndarray[float64] of shape (n_edges,) — transition probabilities
            - counts: np.ndarray[uint64] of shape (n_edges,) — raw counts

        Example:
            >>> from scipy.sparse import csr_matrix
            >>> csr = graph.adjacency_csr()
            >>> A = csr_matrix((csr['weights'], csr['col_indices'], csr['row_offsets']),
            ...                shape=(graph.n_nodes, graph.n_nodes))
        """
        raw = _c.graph_adjacency_csr(self._cap)
        return {
            'row_offsets': np.array(raw['row_offsets'], dtype=np.uint32),
            'col_indices': np.array(raw['col_indices'], dtype=np.uint32),
            'weights': np.array(raw['weights'], dtype=np.float64),
            'counts': np.array(raw['counts'], dtype=np.uint64),
        }

    def successors(self, node_label):
        """Get successor nodes and edge weights for a given node.

        Args:
            node_label: Node label string (e.g., 'C_1' for AAP).

        Returns:
            List of (target_label, weight, count) tuples.
        """
        # Find node index
        all_nodes = self.all_nodes
        try:
            idx = all_nodes.index(node_label)
        except ValueError:
            raise KeyError(f"node '{node_label}' not found in graph")

        raw = _c.graph_adjacency_csr(self._cap)
        start = raw['row_offsets'][idx]
        end = raw['row_offsets'][idx + 1]

        result = []
        for e in range(start, end):
            dst_idx = raw['col_indices'][e]
            result.append((all_nodes[dst_idx], raw['weights'][e], raw['counts'][e]))
        return result

    # ── Simulation ──────────────────────────────────────────

    def simulate(self, n, *, seed=None, v_gene=None, j_gene=None,
                 sample_genes=False):
        """Generate n sequences from the LZ-constrained generative model.

        Args:
            n: Number of sequences to generate.
            seed: RNG seed for reproducibility. None = random.
            v_gene: Constrain to this V gene (str). Requires gene data.
            j_gene: Constrain to this J gene (str). Requires gene data.
            sample_genes: If True and no v/j specified, sample VJ pairs
                from the joint distribution. Requires gene data.

        Returns:
            SimulationResult — iterable of sequences with .log_probs, .n_tokens.
        """
        seed_val = seed if seed is not None else -1

        if v_gene is not None or j_gene is not None or sample_genes:
            if not self.has_gene_data:
                raise NoGeneDataError("graph has no gene data")

            v_id = 0xFFFFFFFF  # LZG_SP_NOT_FOUND
            j_id = 0xFFFFFFFF
            if v_gene is not None:
                v_id = _c.find_gene_id(self._cap, v_gene)
                if v_id == 0xFFFFFFFF:
                    raise ValueError(f"V gene '{v_gene}' not found in graph")
            if j_gene is not None:
                j_id = _c.find_gene_id(self._cap, j_gene)
                if j_id == 0xFFFFFFFF:
                    raise ValueError(f"J gene '{j_gene}' not found in graph")

            seqs, lps, nts, vgs, jgs = _c.gene_simulate(
                self._cap, n, seed_val, v_id, j_id)
            return SimulationResult(seqs, lps, nts, vgs, jgs)
        else:
            seqs, lps, nts = _c.simulate(self._cap, n, seed=seed_val)
            return SimulationResult(seqs, lps, nts)

    # ── LZPGEN ──────────────────────────────────────────────

    def lzpgen(self, sequence, *, log=True):
        """Probability of sequence(s) under the LZ-constrained model.

        Args:
            sequence: A single string, or a list of strings.
            log: If True (default), return log-probability.
                 If False, return probability.

        Returns:
            float for single string, np.ndarray for list.
        """
        raw = _c.lzpgen(self._cap, sequence)

        if isinstance(raw, float):
            return raw if log else np.exp(raw)

        arr = np.array(raw, dtype=np.float64)
        return arr if log else np.exp(arr)

    # ── Analytics ───────────────────────────────────────────

    def effective_diversity(self):
        """Effective diversity: exp(Shannon entropy). Equivalent to D(1)."""
        return _c.effective_diversity(self._cap)

    def diversity_profile(self):
        """Full Shannon diversity breakdown.

        Returns dict with entropy_nats, entropy_bits, effective_diversity, uniformity.
        """
        return _c.diversity_profile(self._cap)

    def hill_number(self, alpha):
        """Hill diversity number D(alpha)."""
        return _c.hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders):
        """Hill numbers for multiple orders. Returns np.ndarray."""
        result = _c.hill_numbers(self._cap, [float(o) for o in orders])
        return np.array(result, dtype=np.float64)

    def hill_curve(self, orders=None):
        """Hill diversity curve.

        Returns {'orders': np.ndarray, 'values': np.ndarray}.
        """
        raw = _c.hill_curve(self._cap, [float(o) for o in orders] if orders else None)
        return {
            'orders': np.array(raw['orders'], dtype=np.float64),
            'values': np.array(raw['values'], dtype=np.float64),
        }

    def power_sum(self, alpha):
        """Raw power sum M(alpha) = sum_s pi(s)^alpha."""
        return _c.power_sum(self._cap, float(alpha))

    def pgen_diagnostics(self, atol=1e-6):
        """Check if the model is a proper probability distribution."""
        return _c.pgen_diagnostics(self._cap, atol)

    def pgen_dynamic_range(self):
        """Dynamic range of PGEN in orders of magnitude."""
        return _c.pgen_dynamic_range(self._cap)

    def pgen_dynamic_range_detail(self):
        """Full dynamic range breakdown."""
        return _c.pgen_dynamic_range_detail(self._cap)

    # ── PGEN Distribution ───────────────────────────────────

    def pgen_moments(self):
        """Moments of the log-PGEN distribution."""
        return _c.pgen_moments(self._cap)

    def pgen_distribution(self):
        """Analytical Gaussian mixture of log-PGEN."""
        from ._pgen_dist import PgenDistribution
        raw = _c.pgen_analytical(self._cap)
        return PgenDistribution(raw)

    # ── Occupancy ───────────────────────────────────────────

    def predicted_richness(self, depth):
        """Expected distinct sequences at sampling depth d."""
        return _c.predicted_richness(self._cap, float(depth))

    def predicted_overlap(self, d_i, d_j):
        """Expected shared sequences between samples at depths d_i, d_j."""
        return _c.predicted_overlap(self._cap, float(d_i), float(d_j))

    def richness_curve(self, depths):
        """Predicted richness at multiple depths. Returns np.ndarray."""
        d_list = [float(d) for d in depths]
        result = _c.richness_curve(self._cap, d_list)
        return np.array(result, dtype=np.float64)

    # ── Sharing ─────────────────────────────────────────────

    def predict_sharing(self, draw_counts, max_k=None):
        """Predicted sharing spectrum across donors."""
        draws = [float(d) for d in draw_counts]
        mk = max_k if max_k is not None else 0
        raw = _c.predict_sharing(self._cap, draws, mk)
        raw['spectrum'] = np.array(raw['spectrum'], dtype=np.float64)
        return raw

    # ── Diversity ───────────────────────────────────────────

    def sequence_perplexity(self, sequence):
        """Perplexity of a single sequence."""
        return _c.sequence_perplexity(self._cap, sequence)

    def repertoire_perplexity(self, sequences):
        """Average perplexity across a repertoire."""
        return _c.repertoire_perplexity(self._cap, list(sequences))

    def path_entropy_rate(self, sequences):
        """Entropy rate (bits/token) estimated from sequences."""
        return _c.path_entropy_rate(self._cap, list(sequences))

    # ── Graph Operations ────────────────────────────────────

    def union(self, other):
        """Union: sum edge counts from both graphs."""
        cap = _c.graph_union(self._cap, other._cap)
        return LZGraph._from_capsule(cap)

    def intersection(self, other):
        """Intersection: keep shared edges, min counts."""
        cap = _c.graph_intersection(self._cap, other._cap)
        return LZGraph._from_capsule(cap)

    def difference(self, other):
        """Difference: subtract other's edge counts."""
        cap = _c.graph_difference(self._cap, other._cap)
        return LZGraph._from_capsule(cap)

    def weighted_merge(self, other, alpha=1.0, beta=1.0):
        """Weighted merge: alpha * self + beta * other."""
        cap = _c.weighted_merge(self._cap, other._cap, alpha, beta)
        return LZGraph._from_capsule(cap)

    def posterior(self, sequences, *, abundances=None, kappa=1.0):
        """Bayesian posterior graph given new observed sequences."""
        cap = _c.posterior(
            self._cap, list(sequences),
            list(abundances) if abundances is not None else None,
            kappa,
        )
        return LZGraph._from_capsule(cap)

    def summary(self):
        """Structural summary dict."""
        return _c.summary(self._cap)

    # ── Features ────────────────────────────────────────────

    def feature_aligned(self, query):
        """Project query into this graph's node feature space.

        self is the reference. Returns np.ndarray of shape (self.n_nodes,).
        """
        raw = _c.feature_aligned(self._cap, query._cap)
        return np.array(raw, dtype=np.float64)

    def feature_mass_profile(self, max_pos=30):
        """Position-based mass distribution profile."""
        raw = _c.feature_mass_profile(self._cap, max_pos)
        return np.array(raw, dtype=np.float64)

    def feature_stats(self):
        """15-element graph statistics vector for ML pipelines."""
        raw = _c.feature_stats(self._cap)
        return np.array(raw, dtype=np.float64)

    # ── IO ──────────────────────────────────────────────────

    def save(self, path):
        """Save to LZG binary format (.lzg)."""
        _c.save(self._cap, str(path))

    @classmethod
    def load(cls, path):
        """Load from LZG binary format."""
        cap = _c.load(str(path))
        return cls._from_capsule(cap)

    # ── Gene Data ───────────────────────────────────────────

    def _ensure_gene_cache(self):
        if self._gene_cache is None:
            raw = _c.gene_info(self._cap)
            if raw is None:
                raise NoGeneDataError("graph has no gene data")
            self._gene_cache = raw

    @property
    def v_genes(self):
        """V gene names in the graph."""
        self._ensure_gene_cache()
        return list(self._gene_cache['v_marginals'].keys())

    @property
    def j_genes(self):
        """J gene names in the graph."""
        self._ensure_gene_cache()
        return list(self._gene_cache['j_marginals'].keys())

    @property
    def v_marginals(self):
        """V gene marginal probabilities."""
        self._ensure_gene_cache()
        return dict(self._gene_cache['v_marginals'])

    @property
    def j_marginals(self):
        """J gene marginal probabilities."""
        self._ensure_gene_cache()
        return dict(self._gene_cache['j_marginals'])

    @property
    def vj_distribution(self):
        """Joint VJ distribution."""
        self._ensure_gene_cache()
        return list(self._gene_cache['vj_distribution'])
