"""
Pytest Configuration and Shared Fixtures
========================================

This module provides shared fixtures for all test modules. Fixtures are loaded
once per session to improve test performance.

Fixtures:
- test_data_nucleotide: DataFrame for NaiveLZGraph tests (ExampleData1.csv)
- test_data_ndp: DataFrame for NDPLZGraph tests (ExampleData2.csv)
- test_data_aap: DataFrame for AAPLZGraph tests (ExampleData3.csv)
- lz_dictionary: Pre-generated k-mer dictionary for NaiveLZGraph
- naive_lzgraph: Pre-built NaiveLZGraph instance
- ndp_lzgraph: Pre-built NDPLZGraph instance
- aap_lzgraph: Pre-built AAPLZGraph instance
"""

import os
from copy import deepcopy

import pytest
import pandas as pd
import numpy as np

# Ensure reproducible tests
np.random.seed(42)


# =============================================================================
# Path Configuration
# =============================================================================

def get_test_data_dir():
    """Get the absolute path to the test data directory."""
    return os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Data Loading Fixtures (Session Scope - loaded once)
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return get_test_data_dir()


@pytest.fixture(scope="session")
def test_data_nucleotide(test_data_dir):
    """
    Load nucleotide sequence data for NaiveLZGraph tests.

    Contains CDR3 nucleotide rearrangement sequences.
    Used by: test_naive_lzgraph.py
    """
    filepath = os.path.join(test_data_dir, 'ExampleData1.csv')
    return pd.read_csv(filepath, index_col=0)


@pytest.fixture(scope="session")
def test_data_ndp(test_data_dir):
    """
    Load data for NDPLZGraph (Nucleotide-position Dependent) tests.

    Contains CDR3 sequences with V/J gene annotations.
    Used by: test_ndp_lzgraph.py, test_metrics.py
    """
    filepath = os.path.join(test_data_dir, 'ExampleData2.csv')
    return pd.read_csv(filepath, index_col=0)


@pytest.fixture(scope="session")
def test_data_aap(test_data_dir):
    """
    Load data for AAPLZGraph (Amino Acid Positional) tests.

    Contains CDR3 amino acid sequences with V/J gene annotations.
    Used by: test_aap_lzgraph.py, test_metrics.py
    """
    filepath = os.path.join(test_data_dir, 'ExampleData3.csv')
    return pd.read_csv(filepath, index_col=0)


# =============================================================================
# Dictionary Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def lz_dictionary():
    """
    Pre-generated k-mer dictionary for NaiveLZGraph.

    Generates all possible k-mers up to length 6 for the
    nucleotide alphabet {A, T, G, C}.
    """
    from LZGraphs import generate_kmer_dictionary
    return generate_kmer_dictionary(6)


# =============================================================================
# Graph Instance Fixtures (Session Scope - built once)
# =============================================================================

@pytest.fixture(scope="session")
def naive_lzgraph(test_data_nucleotide, lz_dictionary):
    """
    Pre-built NaiveLZGraph instance for testing.

    Built from test_data_nucleotide using the standard k-mer dictionary.
    """
    from LZGraphs import NaiveLZGraph
    return NaiveLZGraph(
        test_data_nucleotide['cdr3_rearrangement'],
        lz_dictionary
    )


@pytest.fixture(scope="session")
def ndp_lzgraph(test_data_ndp):
    """
    Pre-built NDPLZGraph instance for testing.

    Built from test_data_ndp which includes gene annotations.
    """
    from LZGraphs import NDPLZGraph
    return NDPLZGraph(test_data_ndp)


@pytest.fixture(scope="session")
def aap_lzgraph(test_data_aap):
    """
    Pre-built AAPLZGraph instance for testing.

    Built from test_data_aap which includes amino acid sequences
    and gene annotations.
    """
    from LZGraphs import AAPLZGraph
    return AAPLZGraph(test_data_aap)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def sample_nucleotide_sequences(test_data_nucleotide):
    """Return a small sample of nucleotide sequences for quick tests."""
    return test_data_nucleotide['cdr3_rearrangement'].iloc[:15].tolist()


@pytest.fixture
def sample_amino_acid_sequences(test_data_aap):
    """Return a small sample of amino acid sequences for quick tests."""
    return test_data_aap['cdr3_amino_acid'].iloc[:15].tolist()


@pytest.fixture
def single_nucleotide_sequence(test_data_ndp):
    """Return a single nucleotide sequence for targeted tests."""
    return test_data_ndp['cdr3_rearrangement'].iloc[0]


@pytest.fixture
def single_amino_acid_sequence(test_data_aap):
    """Return a single amino acid sequence for targeted tests."""
    return test_data_aap['cdr3_amino_acid'].iloc[0]


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def assert_probability_valid(probability, allow_zero=False):
    """
    Assert that a probability value is valid (between 0 and 1).

    Args:
        probability: The probability value to check
        allow_zero: If True, allow probability == 0
    """
    if allow_zero:
        assert 0 <= probability <= 1, f"Probability {probability} not in [0, 1]"
    else:
        assert 0 < probability <= 1, f"Probability {probability} not in (0, 1]"


def assert_log_probability_valid(log_prob):
    """
    Assert that a log-probability value is valid (non-positive).

    Args:
        log_prob: The log-probability value to check
    """
    assert log_prob <= 0, f"Log-probability {log_prob} should be <= 0"


@pytest.fixture
def aap_lzgraph_copy(aap_lzgraph):
    """Function-scoped deep copy of aap_lzgraph for tests that mutate the graph."""
    return deepcopy(aap_lzgraph)


@pytest.fixture
def ndp_lzgraph_copy(ndp_lzgraph):
    """Function-scoped deep copy of ndp_lzgraph for tests that mutate the graph."""
    return deepcopy(ndp_lzgraph)


@pytest.fixture
def naive_lzgraph_copy(naive_lzgraph):
    """Function-scoped deep copy of naive_lzgraph for tests that mutate the graph."""
    return deepcopy(naive_lzgraph)


def assert_entropy_valid(entropy_value, max_entropy=None):
    """
    Assert that an entropy value is valid (non-negative).

    Args:
        entropy_value: The entropy value to check
        max_entropy: Optional upper bound for entropy
    """
    assert entropy_value >= 0, f"Entropy {entropy_value} should be >= 0"
    if max_entropy is not None:
        assert entropy_value <= max_entropy, \
            f"Entropy {entropy_value} exceeds maximum {max_entropy}"
