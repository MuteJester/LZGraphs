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
def seq_count_file(tmp_path):
    p = tmp_path / 'seq_counts.txt'
    p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\t2\nCASSQETQYF\t1\n')
    return str(p)


@pytest.fixture
def lzg_file(tmp_path, seq_file):
    p = str(tmp_path / 'test.lzg')
    run_lzg('-q', 'build', seq_file, '-o', p)
    return p


@pytest.fixture
def fb_lzg_file(tmp_path, seq_file):
    p = str(tmp_path / 'fb.lzg')
    out, err, rc = run_lzg('-q', 'flashback', 'build', seq_file, '-o', p)
    assert rc == 0, err
    return p


class TestVersion:
    def test_version(self):
        from LZGraphs import __version__
        out, _, rc = run_lzg('--version')
        assert rc == 0
        assert __version__ in out


class TestBuild:
    def test_build_txt(self, seq_file, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, stderr, rc = run_lzg('-q', 'build', seq_file, '-o', out_path)
        assert rc == 0
        assert os.path.exists(out_path)

    def test_build_plain_seqcount_txt(self, seq_count_file, tmp_path):
        out_path = str(tmp_path / 'out_counts.lzg')
        _, stderr, rc = run_lzg('-q', 'build', seq_count_file, '-o', out_path)
        assert rc == 0
        assert os.path.exists(out_path)

    def test_build_plain_large_seqcount_txt(self, tmp_path):
        seq_count_file = tmp_path / 'seq_counts_large.txt'
        seq_count_file.write_text(f'CASSLGIRRT\t{2**32}\n')
        out_path = str(tmp_path / 'out_counts_large.lzg')
        _, stderr, rc = run_lzg('-q', 'build', str(seq_count_file), '-o', out_path)
        assert rc == 0
        assert os.path.exists(out_path)

    def test_build_stdin(self, tmp_path):
        out_path = str(tmp_path / 'out.lzg')
        _, _, rc = run_lzg('-q', 'build', '-', '-o', out_path,
                           input_text='CASSLGIRRT\nCASSLGYEQYF\n')
        assert rc == 0

    def test_build_log_level_info_emits_phase_logs(self, seq_count_file, tmp_path):
        out_path = str(tmp_path / 'out_logged.lzg')
        _, stderr, rc = run_lzg('--log-level', 'info', 'build', seq_count_file, '-o', out_path)
        assert rc == 0
        assert 'phase=ingest' in stderr
        assert 'phase=finalize' in stderr
        assert 'phase=save status=done' in stderr

    def test_build_quiet_overridden_by_explicit_log_level(self, seq_count_file, tmp_path):
        out_path = str(tmp_path / 'out_logged_quiet.lzg')
        _, stderr, rc = run_lzg('-q', '--log-level', 'info', 'build', seq_count_file, '-o', out_path)
        assert rc == 0
        assert 'phase=ingest' in stderr

    def test_build_strict_input_fails_on_mixed_seqcount(self, tmp_path):
        mixed = tmp_path / 'mixed_seqcount.txt'
        mixed.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\n')
        out_path = str(tmp_path / 'mixed_out.lzg')
        _, stderr, rc = run_lzg(
            '--log-level', 'info',
            'build', str(mixed), '-o', out_path,
            '--strict-input', '--expect-format', 'plain_seqcount',
        )
        assert rc != 0
        assert 'phase=validate-input status=error' in stderr


class TestValidateInput:
    def test_validate_input_plain_seqcount(self, seq_count_file):
        out, _, rc = run_lzg('validate-input', seq_count_file, '--strict-input',
                             '--expect-format', 'plain_seqcount')
        assert rc == 0
        assert 'VL\tok\tyes' in out
        assert 'VL\tmode\tplain_seqcount' in out

    def test_validate_input_json_failure(self, tmp_path):
        mixed = tmp_path / 'mixed_validate.txt'
        mixed.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\n')
        out, _, rc = run_lzg(
            'validate-input', str(mixed),
            '--strict-input', '--expect-format', 'plain_seqcount',
            '--json',
        )
        assert rc == 2
        data = json.loads(out)
        assert data['ok'] is False
        assert data['mode'] == 'mixed'


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


class TestFlashBack:
    def test_build_creates_file(self, seq_file, tmp_path):
        out_path = str(tmp_path / 'fb_out.lzg')
        _, err, rc = run_lzg('-q', 'flashback', 'build', seq_file, '-o', out_path)
        assert rc == 0, err
        assert os.path.exists(out_path)

    def test_build_seqcount(self, seq_count_file, tmp_path):
        out_path = str(tmp_path / 'fb_counts.lzg')
        _, err, rc = run_lzg('-q', 'flashback', 'build', seq_count_file, '-o', out_path)
        assert rc == 0, err
        assert os.path.exists(out_path)

    def test_build_stdin(self, tmp_path):
        out_path = str(tmp_path / 'fb_stdin.lzg')
        _, err, rc = run_lzg('-q', 'flashback', 'build', '-', '-o', out_path,
                             input_text='CASSLGIRRT\nCASSLGYEQYF\nCASSQETQYF\n')
        assert rc == 0, err
        assert os.path.exists(out_path)

    def test_info_reports_flashback_variant(self, fb_lzg_file):
        out, err, rc = run_lzg('flashback', 'info', fb_lzg_file)
        assert rc == 0, err
        assert 'flashback' in out

    def test_info_json(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'info', fb_lzg_file, '--json')
        assert rc == 0, err
        d = json.loads(out)
        assert d['variant'] == 'flashback'

    def test_score_outputs_pgen(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'score', fb_lzg_file,
                               input_text='CASSLGIRRT\nCASSQETQYF\n')
        assert rc == 0, err
        assert 'pgen' in out
        assert 'CASSLGIRRT' in out

    def test_simulate_generates_n(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'simulate', fb_lzg_file, '-n', '5')
        assert rc == 0, err
        lines = [l for l in out.strip().split('\n') if l]
        assert len(lines) == 5

    def test_diversity_reports_effective_diversity(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'diversity', fb_lzg_file)
        assert rc == 0, err
        assert 'effective_diversity' in out

    def test_diversity_json(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'diversity', fb_lzg_file, '--json')
        assert rc == 0, err
        d = json.loads(out)
        assert 'effective_diversity' in d

    def test_scale_outputs_score(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'scale', fb_lzg_file,
                               '--n-sim', '2000', '--seed', '0',
                               input_text='CASSLGIRRT\nCASSQETQYF\n')
        assert rc == 0, err
        assert 'scale' in out
        assert 'CASSLGIRRT' in out

    def test_scale_json(self, fb_lzg_file):
        out, err, rc = run_lzg('-q', 'flashback', 'scale', fb_lzg_file, '--json',
                               '--n-sim', '2000', '--seed', '0',
                               input_text='CASSLGIRRT\n')
        assert rc == 0, err
        d = json.loads(out)
        assert isinstance(d, list)
        assert 'scale' in d[0]

    def test_scale_save_and_reuse_calibration(self, fb_lzg_file, tmp_path):
        cal = str(tmp_path / 'cal.json')
        _, err, rc = run_lzg('-q', 'flashback', 'scale', fb_lzg_file,
                             '--n-sim', '2000', '--seed', '0',
                             '--save-calibration', cal, input_text='CASSLGIRRT\n')
        assert rc == 0, err
        assert os.path.exists(cal)
        out, err, rc = run_lzg('-q', 'flashback', 'scale', fb_lzg_file,
                               '--calibration', cal, input_text='CASSLGIRRT\n')
        assert rc == 0, err
        assert 'CASSLGIRRT' in out
