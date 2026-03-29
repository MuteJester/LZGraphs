"""Tests for input-format detection, validation, and plain-text readers."""

import pytest

from LZGraphs._io import detect_input_kind, read_sequences, validate_input


class TestInputDetection:
    def test_detect_plain_seqcount(self, tmp_path):
        p = tmp_path / 'seq_counts.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\t2\n')
        assert detect_input_kind(str(p)) == 'plain_seqcount'

    def test_read_sequences_plain_seqcount(self, tmp_path):
        p = tmp_path / 'seq_counts.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\t2\n')
        data = read_sequences(str(p), variant='aap')
        assert data['sequences'] == ['CASSLGIRRT', 'CASSLGYEQYF']
        assert data['abundances'] == [5, 2]
        assert data['v_genes'] is None
        assert data['j_genes'] is None

    def test_read_sequences_plain_large_seqcount(self, tmp_path):
        p = tmp_path / 'seq_counts_large.txt'
        p.write_text(f'CASSLGIRRT\t{2**32}\n')
        data = read_sequences(str(p), variant='aap')
        assert data['abundances'] == [2**32]

    def test_validate_plain_seqcount_ok(self, tmp_path):
        p = tmp_path / 'seq_counts_valid.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\t2\n')
        report = validate_input(str(p), variant='aap', strict_input=True,
                                expect_format='plain_seqcount')
        assert report['ok'] is True
        assert report['mode'] == 'plain_seqcount'
        assert report['seqcount_records'] == 2
        assert report['error_count'] == 0

    def test_validate_mixed_plain_and_seqcount_warns(self, tmp_path):
        p = tmp_path / 'seq_counts_mixed.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\n')
        report = validate_input(str(p), variant='aap')
        assert report['ok'] is True
        assert report['mode'] == 'mixed'
        assert report['warning_count'] >= 1

    def test_validate_mixed_plain_and_seqcount_strict_fails(self, tmp_path):
        p = tmp_path / 'seq_counts_mixed_strict.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\n')
        report = validate_input(str(p), variant='aap', strict_input=True,
                                expect_format='plain_seqcount')
        assert report['ok'] is False
        assert report['error_count'] >= 1

    def test_read_sequences_strict_plain_seqcount_fails_on_mixed(self, tmp_path):
        p = tmp_path / 'seq_counts_mixed_read.txt'
        p.write_text('CASSLGIRRT\t5\nCASSLGYEQYF\n')
        with pytest.raises(ValueError, match='mixed'):
            read_sequences(str(p), variant='aap', strict_input=True,
                           expect_format='plain_seqcount')
