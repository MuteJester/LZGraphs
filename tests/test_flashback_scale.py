"""Tests for the SCALE error/anomaly score on FlashBackGraph.

SCALE = self-simulated, length-calibrated -log Pgen:
  calibrate_scale() simulates from the graph and builds per-length
  median/IQR of -log Pgen (the calibration cache); scale_score() returns
  (-log Pgen(s) - median[len]) / IQR[len].
"""

import math

import numpy as np
import pytest

from LZGraphs import FlashBackGraph, ScaleCalibration

SEQS = [
    'CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF', 'CASSLAPGATNEKLFF', 'CASSQETQYF',
    'CASSLGYEQYF', 'CASSFGQGSYEQYF', 'CASSLGIRRT', 'CASSPGTGGYEQYF',
    'CASSEEAGGYEQYF', 'CASSLNTEAFF',
]


@pytest.fixture
def graph():
    return FlashBackGraph(SEQS)


class TestCalibrateScale:
    def test_returns_scale_calibration(self, graph):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        assert isinstance(cal, ScaleCalibration)

    def test_calibration_has_per_length_and_global_stats(self, graph):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        assert len(cal.median_by_length) > 0
        assert len(cal.iqr_by_length) > 0
        assert math.isfinite(cal.global_median)
        assert cal.global_iqr >= 0.0

    def test_calibration_is_deterministic_with_seed(self, graph):
        a = graph.calibrate_scale(n_sim=3000, seed=42)
        b = graph.calibrate_scale(n_sim=3000, seed=42)
        assert a.global_median == pytest.approx(b.global_median)
        assert a.median_by_length == b.median_by_length


class TestScaleScore:
    def test_single_returns_float(self, graph):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        s = graph.scale_score('CASSLEPSGGTDTQYF', cal)
        assert isinstance(s, float)

    def test_batch_returns_array(self, graph):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        arr = graph.scale_score(['CASSLEPSGGTDTQYF', 'CASSQETQYF'], cal)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)

    def test_anomalous_scores_higher_than_typical(self, graph):
        cal = graph.calibrate_scale(n_sim=5000, seed=0)
        typical = graph.scale_score('CASSLEPSGGTDTQYF', cal)
        anomalous = graph.scale_score('KKKKWWWWWWPPPP', cal)
        assert anomalous > typical


class TestCalibrationIO:
    def test_save_load_roundtrip(self, graph, tmp_path):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        path = tmp_path / 'scale_cal.json'
        cal.save(path)
        loaded = ScaleCalibration.load(path)
        assert loaded.global_median == pytest.approx(cal.global_median)
        assert loaded.global_iqr == pytest.approx(cal.global_iqr)
        assert loaded.median_by_length == cal.median_by_length
        assert loaded.iqr_by_length == cal.iqr_by_length

    def test_loaded_calibration_scores_identically(self, graph, tmp_path):
        cal = graph.calibrate_scale(n_sim=3000, seed=0)
        path = tmp_path / 'scale_cal.json'
        cal.save(path)
        loaded = ScaleCalibration.load(path)
        seq = 'CASSLEPSGGTDTQYF'
        assert graph.scale_score(seq, loaded) == pytest.approx(
            graph.scale_score(seq, cal))
