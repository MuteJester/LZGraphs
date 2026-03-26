"""Tests for the lzg CLI."""

import json
import os
import subprocess
import sys
import pytest

LZG = [sys.executable, '-m', 'LZGraphs.cli']


def run_lzg(*args, input_text=None):
    """Run lzg CLI command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        LZG + list(args),
        capture_output=True, text=True, input=input_text, timeout=30,
    )
    return result.stdout, result.stderr, result.returncode


@pytest.fixture
def seq_file(tmp_path):
    p = tmp_path / 'seqs.txt'
    p.write_text('CASSLGIRRT\nCASSLGYEQYF\nCASSLEPSGGTDTQYF\n'
                 'CASSDTSGGTDTQYF\nCASSFGQGSYEQYF\nCASSQETQYF\n')
    return str(p)


@pytest.fixture
def lzg_file(tmp_path, seq_file):
    p = str(tmp_path / 'test.lzg')
    run_lzg('-q', 'build', seq_file, '-o', p)
    return p


class TestVersion:
    def test_version(self):
        out, _, rc = run_lzg('--version')
        assert rc == 0
        assert '3.0.0' in out


class TestBuild:
    def test_build_txt(self, seq_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('-q', 'build', seq_file, '-o', out_path)
        assert rc == 0
        assert os.path.exists(out_path)

    def test_build_stdin(self, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, _, rc = run_lzg('-q', 'build', '-', '-o', out_path,
                           input_text='CASSLGIRRT\nCASSLGYEQYF\n')
        assert rc == 0


class TestInfo:
    def test_info(self, lzg_file):
        out, _, rc = run_lzg('-q', 'info', lzg_file)
        assert rc == 0
        assert 'GR\tvariant\taap' in out
        assert 'DV\teffective_diversity' in out

    def test_info_json(self, lzg_file):
        out, _, rc = run_lzg('-q', 'info', lzg_file, '--json')
        assert rc == 0
        d = json.loads(out)
        assert 'n_nodes' in d


class TestScore:
    def test_score(self, lzg_file):
        out, _, rc = run_lzg('-q', 'score', lzg_file,
                             input_text='CASSLGIRRT\nCASSLGYEQYF\n')
        assert rc == 0
        assert 'sequence\tlzpgen' in out
        lines = out.strip().split('\n')
        assert len(lines) == 3  # header + 2 sequences

    def test_score_pipe(self, lzg_file, seq_file):
        out, _, rc = run_lzg('-q', 'score', lzg_file, seq_file)
        assert rc == 0
        lines = out.strip().split('\n')
        assert len(lines) == 7  # header + 6 sequences


class TestSimulate:
    def test_simulate(self, lzg_file):
        out, _, rc = run_lzg('-q', 'simulate', lzg_file, '-n', '5', '--seed', '42')
        assert rc == 0
        lines = [l for l in out.strip().split('\n') if l]
        assert len(lines) == 5

    def test_simulate_detailed(self, lzg_file):
        out, _, rc = run_lzg('-q', 'simulate', lzg_file, '-n', '3',
                             '--seed', '42', '--with-details')
        assert rc == 0
        assert 'sequence\tlzpgen\tn_tokens' in out


class TestDiversity:
    def test_diversity(self, lzg_file):
        out, _, rc = run_lzg('-q', 'diversity', lzg_file)
        assert rc == 0
        assert 'HL\t0\t' in out
        assert 'DV\teffective_diversity' in out


class TestCompare:
    def test_compare(self, tmp_path, seq_file):
        a = str(tmp_path / 'a.lzg')
        b = str(tmp_path / 'b.lzg')
        run_lzg('-q', 'build', seq_file, '-o', a)

        other = tmp_path / 'other.txt'
        other.write_text('CASSLGIRRT\nCASSXYZ\n')
        run_lzg('-q', 'build', str(other), '-o', b)

        out, _, rc = run_lzg('-q', 'compare', a, b)
        assert rc == 0
        assert 'CP\tjsd' in out


class TestDecompose:
    def test_decompose_stdin(self):
        out, _, rc = run_lzg('decompose', input_text='CASSLGIRRT\n')
        assert rc == 0
        assert 'C|A|S|SL|G|I|R|RT' in out

    def test_decompose_json(self):
        out, _, rc = run_lzg('decompose', '--json', input_text='CASSLGIRRT\n')
        assert rc == 0
        d = json.loads(out)
        assert d[0]['n_tokens'] == 8


class TestPredict:
    def test_richness(self, lzg_file):
        out, _, rc = run_lzg('-q', 'predict', 'richness', lzg_file,
                             '--depths', '1,10,100')
        assert rc == 0
        assert 'depth\tpredicted_richness' in out
        lines = out.strip().split('\n')
        assert len(lines) == 4  # header + 3 depths

    def test_overlap(self, lzg_file):
        out, _, rc = run_lzg('-q', 'predict', 'overlap', lzg_file,
                             '--di', '100', '--dj', '100')
        assert rc == 0
        assert 'predicted_overlap' in out
