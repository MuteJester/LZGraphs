"""Mixin shared between LZGraph and FlashBackGraph.

Both classes wrap a C capsule that exposes the same DAG graph operations.
This module holds the methods that were byte-identical between the two
classes (with negligible docstring-only differences). FlashBackGrammar
does not use this mixin: it has rules, not nodes/edges, and so does not
share these methods.

Subclasses are expected to provide:
  * ``self._cap`` — the C capsule wrapping the graph
  * ``self.all_nodes`` — property returning the full node label list,
    side-effecting ``self._all_nodes_cache``
  * ``self.pgen(sequence)`` — scoring method (used by ``__contains__``)
  * ``self.union`` / ``self.intersection`` / ``self.difference`` — set ops
    (used by ``__or__`` / ``__and__`` / ``__sub__``)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _clzgraph as _c
from ._constants import LOG_EPS_THRESHOLD


class _GraphCommonMixin:
    """Shared method bodies for classes wrapping a CSR DAG capsule."""
    # ── Lazy-cache slots (populated by first access) ──
    _csr_cache: dict[str, np.ndarray] | None = None
    _degrees_cache: dict[str, list[int]] | None = None
    _node_index: dict[str, int] | None = None
    _rcsr_cache: dict[str, np.ndarray] | None = None
    _summary_cache: dict[str, Any] | None = None


    def __len__(self) -> int:
        """Number of training sequences seen by this graph."""
        return self.n_sequences

    def __contains__(self, sequence: str) -> bool:
        return self.pgen(sequence) > LOG_EPS_THRESHOLD

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def _get_summary(self) -> dict[str, Any]:
        if self._summary_cache is None:
            self._summary_cache = _c.summary(self._cap)
        return self._summary_cache

    def _get_degrees(self) -> dict[str, list[int]]:
        if self._degrees_cache is None:
            self._degrees_cache = _c.graph_degrees(self._cap)
        return self._degrees_cache

    def _get_csr(self) -> dict[str, np.ndarray]:
        if self._csr_cache is None:
            raw = _c.graph_adjacency_csr(self._cap)
            self._csr_cache = {
                'row_offsets': np.array(raw['row_offsets'], dtype=np.uint32),
                'col_indices': np.array(raw['col_indices'], dtype=np.uint32),
                'weights': np.array(raw['weights'], dtype=np.float64),
                'counts': np.array(raw['counts'], dtype=np.uint64),
            }
        return self._csr_cache

    def _get_node_index(self) -> dict[str, int]:
        """Cached label → integer index dict."""
        if self._node_index is None:
            self._node_index = {n: i for i, n in enumerate(self.all_nodes)}
        return self._node_index

    def _get_reverse_csr(self) -> dict[str, np.ndarray]:
        """Build a reverse (transpose) CSR for predecessor lookups."""
        if self._rcsr_cache is None:
            csr = self._get_csr()
            n = len(csr['row_offsets']) - 1
            col = csr['col_indices']
            w = csr['weights']
            c = csr['counts']
            in_deg = np.zeros(n, dtype=np.uint32)
            for j in col:
                in_deg[j] += 1
            r_offsets = np.zeros(n + 1, dtype=np.uint32)
            np.cumsum(in_deg, out=r_offsets[1:])
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

    def successors(self, node_label: str) -> list[tuple[str, float, int]]:
        """Get successor nodes with edge weights.

        Args:
            node_label: Node label string (e.g., 'C_1' for AAP).

        Returns:
            List of (target_label, weight, count) tuples.
        """
        idx_map = self._get_node_index()
        idx = idx_map.get(node_label)
        if idx is None:
            raise KeyError(f"node '{node_label}' not found in graph")
        csr = self._get_csr()
        nodes = self.all_nodes
        start = csr['row_offsets'][idx]
        end = csr['row_offsets'][idx + 1]
        col = csr['col_indices']
        w = csr['weights']
        c = csr['counts']
        return [(nodes[col[e]], float(w[e]), int(c[e]))
                for e in range(start, end)]

    def predecessors(self, node_label: str) -> list[tuple[str, float, int]]:
        """Get predecessor nodes with edge weights.

        Returns list of (source_label, weight, count) tuples.
        """
        idx_map = self._get_node_index()
        idx = idx_map.get(node_label)
        if idx is None:
            raise KeyError(f"node '{node_label}' not found in graph")
        rcsr = self._get_reverse_csr()
        nodes = self.all_nodes
        start = rcsr['row_offsets'][idx]
        end = rcsr['row_offsets'][idx + 1]
        col = rcsr['col_indices']
        w = rcsr['weights']
        c = rcsr['counts']
        return [(nodes[col[e]], float(w[e]), int(c[e]))
                for e in range(start, end)]

    @property
    def initial_nodes(self) -> list[str]:
        """Node labels with in-degree 0 (excluding sentinel nodes)."""
        all_n = self.all_nodes
        in_deg = self._get_degrees()['in_degrees']
        return [n for n, d in zip(all_n, in_deg)
                if d == 0 and not n.startswith('@') and '$' not in n]

    @property
    def terminal_nodes(self) -> list[str]:
        """Node labels with out-degree 0 (excluding sentinel nodes)."""
        all_n = self.all_nodes
        out_deg = self._get_degrees()['out_degrees']
        return [n for n, d in zip(all_n, out_deg)
                if d == 0 and not n.startswith('@') and '$' not in n]

    @property
    def hub_nodes(self) -> list[tuple[str, int, int]]:
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
    def node_degree_map(self) -> dict[str, tuple[int, int]]:
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
    def edge_weight_map(self) -> dict[tuple[str, str], float]:
        """Dict mapping (src, dst) -> transition probability.

        Excludes edges involving sentinel nodes.
        """
        return {(e[0], e[1]): e[2] for e in self.edges}

    @property
    def isolated_nodes(self) -> list[str]:
        """Nodes with both in-degree and out-degree 0 (excluding sentinels)."""
        all_n = self.all_nodes
        degs = self._get_degrees()
        in_deg = degs['in_degrees']
        out_deg = degs['out_degrees']
        return [n for n, i, o in zip(all_n, in_deg, out_deg)
                if i == 0 and o == 0
                and not n.startswith('@') and '$' not in n]
