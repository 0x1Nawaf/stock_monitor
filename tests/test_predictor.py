"""Tests for model training, prediction, and sequence building."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from stock_monitor.model import StockLSTM
from stock_monitor.predictor import _build_sequences, _compute_confidence


class TestBuildSequences:
    def test_basic_output_shape(self):
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        xs, ys = _build_sequences(features, targets, seq_len=20)
        assert xs.shape == (80, 20, 10)
        assert ys.shape == (80,)

    def test_skips_nan_targets(self):
        features = np.random.randn(50, 5).astype(np.float32)
        targets = np.random.randn(50).astype(np.float32)
        targets[30] = np.nan
        targets[40] = np.nan
        xs, ys = _build_sequences(features, targets, seq_len=10)
        assert not np.isnan(ys).any()
        assert len(xs) == 38

    def test_empty_when_too_short(self):
        features = np.random.randn(5, 3).astype(np.float32)
        targets = np.random.randn(5).astype(np.float32)
        xs, ys = _build_sequences(features, targets, seq_len=10)
        assert len(xs) == 0
        assert len(ys) == 0

    def test_sequence_content_correct(self):
        features = np.arange(30).reshape(10, 3).astype(np.float32)
        targets = np.arange(10, dtype=np.float32)
        xs, ys = _build_sequences(features, targets, seq_len=5)
        np.testing.assert_array_equal(xs[0], features[0:5])
        assert ys[0] == targets[4]
        np.testing.assert_array_equal(xs[1], features[1:6])
        assert ys[1] == targets[5]


class TestComputeConfidence:
    def test_consistent_predictions(self):
        confidence = _compute_confidence(0.05, [0.04, 0.05, 0.06, 0.045])
        assert 0.7 <= confidence <= 0.95

    def test_inconsistent_predictions(self):
        confidence = _compute_confidence(0.05, [-0.03, 0.08, -0.02, 0.01])
        assert confidence < 0.7

    def test_empty_windows(self):
        confidence = _compute_confidence(0.05, [])
        assert confidence == 0.5

    def test_clamped_range(self):
        confidence = _compute_confidence(0.01, [0.01, 0.01, 0.01])
        assert 0.1 <= confidence <= 0.95


class TestStockLSTM:
    def test_forward_shape(self):
        model = StockLSTM(input_size=22, hidden_size=64, num_layers=2, dropout=0.2)
        x = torch.randn(4, 60, 22)
        out = model(x)
        assert out.shape == (4,)

    def test_single_sample(self):
        model = StockLSTM(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(1, 30, 10)
        out = model(x)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()
