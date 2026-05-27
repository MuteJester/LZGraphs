"""Simulation result container."""
from __future__ import annotations

from typing import Iterator

import numpy as np


class SimulationResult:
    """Result container for LZGraph.simulate().

    Iterates as sequences (list[str]) for convenience.
    Full metadata accessible via attributes.

    Supports indexing (returns string) and slicing (returns new
    SimulationResult with aligned metadata).
    """

    __slots__ = ('sequences', 'log_probs', 'n_tokens', 'v_genes', 'j_genes')

    sequences: list[str]
    log_probs: np.ndarray
    n_tokens: np.ndarray
    v_genes: list[str] | None
    j_genes: list[str] | None

    def __init__(
        self,
        sequences: list[str],
        log_probs,
        n_tokens,
        v_genes: list[str] | None = None,
        j_genes: list[str] | None = None,
    ) -> None:
        self.sequences = sequences
        self.log_probs = np.asarray(log_probs, dtype=np.float64)
        self.n_tokens = np.asarray(n_tokens, dtype=np.uint32)
        self.v_genes = v_genes
        self.j_genes = j_genes

    def __iter__(self) -> Iterator[str]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SimulationResult(
                self.sequences[idx],
                self.log_probs[idx],
                self.n_tokens[idx],
                self.v_genes[idx] if self.v_genes is not None else None,
                self.j_genes[idx] if self.j_genes is not None else None,
            )
        return self.sequences[idx]

    def __repr__(self) -> str:
        return f"SimulationResult(n={len(self)})"
