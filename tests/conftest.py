"""Shared fixtures for LZGraphs Python test suite."""

import os
import pytest

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = TESTS_DIR  # CSVs are in the tests/ folder


@pytest.fixture(scope='session')
def aap_sequences():
    """Small amino acid sequence list for quick tests."""
    return [
        'CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
        'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF',
    ]


@pytest.fixture(scope='session')
def aap_v_genes():
    return ['TRBV5-1', 'TRBV5-1', 'TRBV12-3',
            'TRBV12-3', 'TRBV5-1', 'TRBV5-1']


@pytest.fixture(scope='session')
def aap_j_genes():
    return ['TRBJ1-1', 'TRBJ2-7', 'TRBJ1-1',
            'TRBJ1-1', 'TRBJ2-7', 'TRBJ2-7']


@pytest.fixture(scope='session')
def aap_graph(aap_sequences):
    from LZGraphs import LZGraph
    return LZGraph(aap_sequences, variant='aap')


@pytest.fixture(scope='session')
def aap_gene_graph(aap_sequences, aap_v_genes, aap_j_genes):
    from LZGraphs import LZGraph
    return LZGraph(aap_sequences, variant='aap',
                   v_genes=aap_v_genes, j_genes=aap_j_genes)


@pytest.fixture
def tmp_lzg(tmp_path):
    """Return a temp path for .lzg files."""
    return str(tmp_path / 'test.lzg')
