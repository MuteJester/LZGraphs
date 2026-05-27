"""Shared fixtures for Tier 1 regression tests.

Loads a small, deterministic slice of ``subset_sequences.tsv`` for use
across all snapshot tests. The slice is the first 200 productive
sequences; the exact list is the lock-in input.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_TSV = DATA_DIR / "regression_input.tsv"

# Fixed slice — change only with deliberate snapshot regen.
N_SEQUENCES = 200


def _load_subset(n: int) -> list[dict[str, str]]:
    """Return the first n productive rows from the regression input TSV."""
    rows: list[dict[str, str]] = []
    with INPUT_TSV.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("productive") != "True":
                continue
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


@pytest.fixture(scope="session")
def repertoire_rows() -> list[dict[str, str]]:
    """200 productive rows from the regression input TSV, deterministic order."""
    return _load_subset(N_SEQUENCES)


@pytest.fixture(scope="session")
def cdr3_sequences(repertoire_rows) -> list[str]:
    """Just the CDR3 amino-acid sequences (junction_aa column)."""
    return [r["junction_aa"] for r in repertoire_rows]


@pytest.fixture(scope="session")
def v_genes(repertoire_rows) -> list[str]:
    return [r["v_call"] for r in repertoire_rows]


@pytest.fixture(scope="session")
def j_genes(repertoire_rows) -> list[str]:
    return [r["j_call"] for r in repertoire_rows]


# Small fixed probe set for per-sequence calculations (pgen, FBAS, etc.).
# Mix of in-repertoire and likely-out-of-repertoire sequences.
PROBE_SEQUENCES = [
    "CASSLGIRRT",
    "CASSLGYEQYF",
    "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF",
    "CASSFGQGSYEQYF",
    "CASSQETQYF",
    "CASRRDGSFNEKLFF",
    "CAAAAAAAAAA",   # likely out-of-distribution
]


@pytest.fixture(scope="session")
def probe_sequences() -> list[str]:
    return PROBE_SEQUENCES
