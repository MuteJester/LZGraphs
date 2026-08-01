"""Shared fixtures for LZGraphs Python test suite."""

import os

import pytest

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = TESTS_DIR  # CSVs are in the tests/ folder

# Every variable LZGraphs._term._caps consults to pick a rendering mode or a
# colour depth. Tests that drive the CLI in a subprocess must not inherit
# these from whatever shell or runner happens to be executing them: on
# GitHub Actions, CI=true and GITHUB_ACTIONS=true are always set, which
# correctly forces auto-mode to plain and made a test asserting rich-on-a-tty
# fail there while passing on every developer machine.
_TERM_ENV_VARS = (
    'CI', 'GITHUB_ACTIONS', 'GITLAB_CI',
    'NO_COLOR', 'FORCE_COLOR', 'TERM', 'COLORTERM',
)


def clean_term_env(extra=None):
    """A copy of ``os.environ`` with terminal-detection variables neutralised.

    Drops every variable in :data:`_TERM_ENV_VARS`, then applies a fixed
    256-colour baseline so colour depth is identical everywhere, and finally
    layers ``extra`` on top. A test that wants one of these set (``NO_COLOR``,
    ``TERM=dumb``, ``CI``) passes it in ``extra`` and gets exactly that, with
    no interference from the ambient environment.
    """
    env = {k: v for k, v in os.environ.items() if k not in _TERM_ENV_VARS}
    env['TERM'] = 'xterm-256color'
    env['COLORTERM'] = 'truecolor'
    if extra:
        env.update(extra)
    return env


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
