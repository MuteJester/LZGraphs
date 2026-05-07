"""FlashBackGrammar — boundary-indexed PCFG over FlashBack decomposition trees.

Unlike ``FlashBackGraph`` (a Markov chain on linearised tokens), this class
models the **tree** of FlashBack decomposition directly as a weighted
probabilistic context-free grammar with non-terminals indexed by the
boundary characters of the current middle. Rules are shared across all
recursion depths that share a boundary pair, which gives cross-depth motif
reuse while still guaranteeing every generated tree is a canonical FlashBack
decomposition.

All analytics (path count series, entropy, Hill numbers, dynamic range) are
computed exactly via small linear systems over the non-terminal space.

See ``.private/flashback_grammar_2026-04-22/implementation_plan.md`` for the
full design.
"""

from __future__ import annotations

import math

import numpy as np

from . import _clzgraph as _c
from ._simulation_result import SimulationResult


class FlashBackGrammar:
    """Weighted probabilistic context-free grammar over FlashBack trees.

    Args:
        sequences: Training CDR3 strings.
        abundances: Per-sequence observed counts; defaults to 1 each.
        abundance_mode: How abundances translate into rule-count contributions.

            - ``'none'``   — each sequence counts once regardless of abundance.
            - ``'linear'`` — rule counts scaled by abundance (default).
            - ``'log'``    — rule counts scaled by ``log1p(abundance)``.

        smoothing: Laplace α over observed rules at each non-terminal.
        backoff: Handling of rules unseen at a given non-terminal during pgen.

            - ``'none'`` — strict MLE: any unseen rule collapses pgen to ε.
            - ``'gt'``   — Good-Turing reserved-mass + marginal-frequency
              backoff. Analytics (H, Hill, sampling) always use pure MLE;
              backoff only affects ``flashback_pgen``.
    """

    def __init__(self, sequences, *, abundances=None,
                 abundance_mode: str = 'linear',
                 smoothing: float = 0.0,
                 backoff: str = 'none'):
        if not sequences:
            raise ValueError("sequences must be a non-empty iterable")
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        self._cap = _c.fbg_build(
            seqs,
            abundances=abs_list,
            abundance_mode=abundance_mode,
            smoothing=float(smoothing),
            backoff=backoff,
        )
        self._info = _c.fbg_info(self._cap)

    # ── Internal constructor for ops that return a capsule ──

    @classmethod
    def _from_capsule(cls, capsule):
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._info = _c.fbg_info(capsule)
        return obj

    @classmethod
    def from_file(cls, path, *, abundance_mode: str = 'linear',
                  smoothing: float = 0.0, backoff: str = 'none'):
        """Build from a plain text file by streaming.

        One sequence per line, or ``sequence<TAB>abundance``. The file is
        parsed in C with constant memory usage — safe for multi-GB inputs.
        Lines whose alphabet exceeds the grammar's cap, or whose
        decomposition fails, are skipped with a warning count reported at
        the end.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        cap = _c.fbg_build_file(
            path,
            abundance_mode=abundance_mode,
            smoothing=float(smoothing),
            backoff=backoff,
        )
        return cls._from_capsule(cap)

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self):
        return (f"FlashBackGrammar("
                f"n_nts={self.n_nonterminals}, "
                f"n_rules={self.n_rules}, "
                f"ρ={self.spectral_radius:.4f})")

    def __len__(self):
        return self.n_rules

    def __contains__(self, sequence):
        return self.flashback_pgen(sequence) > -690.0

    # ── Basic properties ────────────────────────────────────

    @property
    def variant(self):
        return 'flashback_grammar'

    @property
    def n_nonterminals(self):
        """Number of non-terminals (1 for S + one per observed (a, z))."""
        return self._info['n_nts']

    @property
    def n_rules(self):
        """Total number of rules across all non-terminals."""
        return self._info['n_rules']

    @property
    def n_internal_rules(self):
        return self._info['n_internal_rules']

    @property
    def n_leaf_rules(self):
        return self._info['n_leaf_rules']

    @property
    def alphabet(self):
        """Observed characters including the sentinels @ and $."""
        return list(self._info.get('_alphabet', []))  # not directly exposed

    @property
    def spectral_radius(self):
        """ρ(T) — the grammar is consistent iff < 1."""
        return self._info['spectral_radius']

    @property
    def is_consistent(self):
        return self._info['is_consistent']

    @property
    def smoothing(self):
        return self._info['smoothing']

    @property
    def backoff_mode(self):
        m = self._info['backoff']
        return 'gt' if m == 1 else 'none'

    @property
    def abundance_mode(self):
        return ['none', 'linear', 'log'][self._info['abundance_mode']]

    @property
    def max_length(self):
        """Longest training sequence length observed at build time."""
        return self._info['max_length']

    @property
    def length_counts(self):
        """Raw training length histogram: dict {length: count}."""
        arr = _c.fbg_length_counts(self._cap)
        return {L: int(c) for L, c in enumerate(arr) if c > 0}

    @property
    def n_sequences(self):
        """Σ of raw training length counts (weighted by abundance)."""
        return sum(self.length_counts.values())

    # ── Non-terminals and rules ─────────────────────────────

    @property
    def nonterminals(self):
        """List of ``(a_char, z_char, is_start)`` tuples in storage order."""
        return [(a, z, bool(start))
                for (a, z, start, _, _, _) in _c.fbg_nts(self._cap)]

    def rules_at(self, a, z):
        """Rules at non-terminal ``M(a, z)``.

        Returns list of rule dicts with keys:
            kind, a_char, z_char, a_run_len, z_run_len,
            dst_a, dst_z (internal only), count, weight.
        """
        idx = self._find_nt_index(a, z, is_start=False)
        if idx is None:
            raise KeyError(f"no non-terminal M{(a, z)!r}")
        return _c.fbg_rules_at(self._cap, idx)

    def rules_at_start(self):
        """Rules at the start non-terminal S."""
        # S is always at index 0 by our convention.
        return _c.fbg_rules_at(self._cap, self._info['start_nt'])

    @property
    def sentinel_rule_weights(self):
        """Distribution over (first_char, last_char) pairs at S.

        Returns dict mapping (a', z') → weight.
        """
        out = {}
        for r in self.rules_at_start():
            if r['kind'] == 'internal':
                out[(r['dst_a'], r['dst_z'])] = r['weight']
        return out

    def _find_nt_index(self, a, z, *, is_start=False):
        for i, (na, nz, start, _, _, _) in enumerate(_c.fbg_nts(self._cap)):
            if is_start and start:
                return i
            if not start and na == a and nz == z:
                return i
        return None

    def top_rules(self, k=20, by='weight'):
        """Top-k rules across all non-terminals, sorted by weight or count."""
        if by not in ('weight', 'count'):
            raise ValueError("by must be 'weight' or 'count'")
        nts = _c.fbg_nts(self._cap)
        entries = []
        for i, (a, z, start, _, _, _) in enumerate(nts):
            nt_key = 'S' if start else (a, z)
            for r in _c.fbg_rules_at(self._cap, i):
                entries.append((nt_key, r))
        entries.sort(key=lambda e: -e[1][by])
        return entries[:k]

    # ── Probability ─────────────────────────────────────────

    def flashback_pgen(self, sequence, *, log=True):
        """Exact probability of ``sequence`` under the grammar.

        Respects the grammar's ``backoff`` configuration.

        Args:
            sequence: A single string, or a list of strings.
            log: If True, return log-probability.
        """
        if isinstance(sequence, str):
            lp = _c.fbg_pgen(self._cap, sequence)
            return lp if log else math.exp(lp)
        arr = np.array(_c.fbg_pgen_batch(self._cap, list(sequence)),
                       dtype=np.float64)
        return arr if log else np.exp(arr)

    def pgen_mle(self, sequence, *, log=True):
        """Probability under pure MLE, ignoring the grammar's backoff setting.

        Use this when sampler and scorer must agree on the same distribution
        (e.g. Monte-Carlo entropy estimation).
        """
        if isinstance(sequence, str):
            lp = _c.fbg_pgen_mle(self._cap, sequence)
            return lp if log else math.exp(lp)
        arr = np.array([_c.fbg_pgen_mle(self._cap, s) for s in sequence],
                       dtype=np.float64)
        return arr if log else np.exp(arr)

    # ── Exact analytics ────────────────────────────────────

    def path_count(self, max_length):
        """Total count of distinct sequences of length ≤ ``max_length``."""
        return float(np.sum(self.path_count_series(max_length)))

    def path_count_series(self, max_length):
        """[x^L] of Z_S(x) with unit weights — unique-sequence count per length."""
        if max_length is None or max_length < 0:
            raise ValueError("max_length must be a non-negative int")
        return np.array(_c.fbg_path_count_series(self._cap, int(max_length)),
                        dtype=np.float64)

    def length_distribution(self, max_length):
        """Probability P(len = L) for L ∈ [0, max_length]."""
        if max_length is None or max_length < 0:
            raise ValueError("max_length must be a non-negative int")
        return np.array(_c.fbg_length_distribution(self._cap, int(max_length)),
                        dtype=np.float64)

    def effective_diversity(self):
        """Exact ``exp(H)`` — effective number of distinct sequences."""
        return _c.fbg_effective_diversity(self._cap)['effective_diversity']

    def diversity_profile(self):
        """Full Shannon diversity breakdown (entropy_nats, entropy_bits, etc)."""
        return _c.fbg_effective_diversity(self._cap)

    def entropy(self):
        """Shannon entropy of the sequence distribution (nats)."""
        return _c.fbg_entropy(self._cap)

    def hill_number(self, alpha):
        """Exact Hill number D(α)."""
        return _c.fbg_hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders):
        """Exact Hill numbers for multiple α values."""
        return np.array(_c.fbg_hill_numbers(self._cap,
                                            [float(a) for a in orders]),
                        dtype=np.float64)

    def hill_curve(self, orders=None):
        """Hill diversity curve (exact)."""
        if orders is None:
            orders = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
        o_list = [float(o) for o in orders]
        values = self.hill_numbers(o_list)
        return {
            'orders': np.array(o_list, dtype=np.float64),
            'values': values,
        }

    def power_sum(self, alpha):
        """Exact power sum M(α) = Σ P(s)^α."""
        return _c.fbg_power_sum(self._cap, float(alpha))

    def pgen_dynamic_range(self, max_length=30):
        """Exact dynamic range in orders of magnitude (log10) up to ``max_length``."""
        return _c.fbg_dynamic_range(self._cap, int(max_length))['dynamic_range_orders']

    def pgen_dynamic_range_detail(self, max_length=30):
        """Full dynamic range breakdown: max/min log-prob, nats, orders."""
        return _c.fbg_dynamic_range(self._cap, int(max_length))

    # ── Sampling ──────────────────────────────────────────

    def simulate(self, n, *, seed=None):
        """Generate ``n`` sequences by top-down grammar sampling.

        Every generated sequence is a canonical FlashBack decomposition by
        construction. Returns a ``SimulationResult``.
        """
        seed_val = seed if seed is not None else -1
        seqs, lps, nts = _c.fbg_simulate(self._cap, int(n), seed=seed_val)
        return SimulationResult(seqs, lps, nts)

    def top_k_sequences(self, k=100, *, most_probable=True, max_length=30):
        """Exact top-K sequences by log-probability (forward DP, no MC).

        Args:
            k: Number of walks to return.
            most_probable: If True, top K by highest log-prob; otherwise lowest.
            max_length: Cap on output sequence length.
        """
        seqs, lps, nts = _c.fbg_top_k_sequences(
            self._cap, int(k),
            most_probable=bool(most_probable),
            max_length=int(max_length),
        )
        return SimulationResult(seqs, lps, nts)

    # ── Grammar operations ─────────────────────────────────

    def posterior(self, sequences, *, abundances=None, kappa=1.0):
        """Dirichlet-Multinomial posterior given new observations.

            w_post(r | nt) = (κ · w_prior(r | nt) + c_ind(r | nt))
                              / (κ + n_ind(nt))

        - ``κ = 0``      → pure individual MLE (restricted to prior topology).
        - ``κ → ∞``      → posterior ≈ prior.
        - Individual rules not present in the prior are ignored.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        cap = _c.fbg_posterior(self._cap, seqs,
                               abundances=abs_list, kappa=float(kappa))
        return FlashBackGrammar._from_capsule(cap)

    def without(self, sequences, *, abundances=None):
        """Return a new grammar with ``sequences`` removed from training counts.

        For each sequence (with abundance ``a``, defaulting to 1) the canonical
        FlashBack tree is enumerated and ``a`` (weighted by the grammar's
        ``abundance_mode``) is subtracted from every rule on that tree.
        Rules whose count reaches 0 are pruned; non-terminals with no
        remaining rules are dropped. Raises ``ValueError`` if the result
        would be empty or if the start non-terminal is removed.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        cap = _c.fbg_subtract(self._cap, seqs, abundances=abs_list)
        return FlashBackGrammar._from_capsule(cap)

    def remove_sequences(self, sequences, *, abundances=None):
        """Alias for :meth:`without`."""
        return self.without(sequences, abundances=abundances)

    # ── IO ─────────────────────────────────────────────────

    def save(self, path):
        """Save to a binary file (format: FBG1 + CRC32c check)."""
        _c.fbg_save(self._cap, str(path))

    @classmethod
    def load(cls, path):
        """Load a grammar saved with :meth:`save`."""
        cap = _c.fbg_load(str(path))
        return cls._from_capsule(cap)

    # ── Summary ────────────────────────────────────────────

    def summary(self):
        """Dict of the grammar's structural + analytical highlights."""
        diversity = self.diversity_profile()
        return {
            'variant': self.variant,
            'n_nonterminals': self.n_nonterminals,
            'n_rules': self.n_rules,
            'n_internal_rules': self.n_internal_rules,
            'n_leaf_rules': self.n_leaf_rules,
            'spectral_radius': self.spectral_radius,
            'is_consistent': self.is_consistent,
            'abundance_mode': self.abundance_mode,
            'backoff_mode': self.backoff_mode,
            'smoothing': self.smoothing,
            'n_sequences': self.n_sequences,
            'max_length': self.max_length,
            'entropy_nats': diversity['entropy_nats'],
            'effective_diversity': diversity['effective_diversity'],
        }
