from __future__ import annotations

import numpy as np
import pytest
import torch

from stock_monitor.model.lstm_clf import (
    StockLSTMClassifier,
    _build_sequences,
)


class TestBuildSequences:
    def test_basic_output_shape(self):
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.array([0, 1, 2] * 33 + [1], dtype=np.int64)
        xs, ys = _build_sequences(features, targets, seq_len=20)
        assert xs.shape == (80, 20, 10)
        assert ys.shape == (80,)

    def test_skips_invalid_targets(self):
        features = np.random.randn(50, 5).astype(np.float32)
        targets = np.array([0, 1, 2] * 16 + [1, -1], dtype=np.int64)
        targets[30] = -1
        targets[40] = -1
        xs, ys = _build_sequences(features, targets, seq_len=10)
        assert (ys >= 0).all()
        assert len(xs) == 38

    def test_empty_when_too_short(self):
        features = np.random.randn(5, 3).astype(np.float32)
        targets = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        xs, ys = _build_sequences(features, targets, seq_len=10)
        assert len(xs) == 0
        assert len(ys) == 0

    def test_sequence_content_correct(self):
        features = np.arange(30).reshape(10, 3).astype(np.float32)
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        xs, ys = _build_sequences(features, targets, seq_len=5)
        np.testing.assert_array_equal(xs[0], features[0:5])
        assert ys[0] == targets[4]
        np.testing.assert_array_equal(xs[1], features[1:6])
        assert ys[1] == targets[5]


class TestStockLSTMClassifier:
    def test_forward_shape(self):
        model = StockLSTMClassifier(input_size=22, hidden_size=64, num_layers=2, dropout=0.2)
        x = torch.randn(4, 60, 22)
        out = model(x)
        assert out.shape == (4, 3)

    def test_single_sample(self):
        model = StockLSTMClassifier(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(1, 30, 10)
        out = model(x)
        assert out.shape == (1, 3)
        assert torch.isfinite(out).all()

    def test_softmax_sums_to_one(self):
        model = StockLSTMClassifier(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(2, 30, 10)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_attention_mechanism(self):
        model = StockLSTMClassifier(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
        assert hasattr(model, "attention")
        x = torch.randn(1, 20, 10)
        out = model(x)
        assert out.shape == (1, 3)
