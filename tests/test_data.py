"""Tests for feature engineering and data preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_monitor.data import build_features, build_targets, prepare_dataset


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data with 200 trading days."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestBuildFeatures:
    def test_output_shape(self, sample_ohlcv):
        features = build_features(sample_ohlcv)
        assert len(features) > 0
        assert features.shape[1] == 22

    def test_no_nan_values(self, sample_ohlcv):
        features = build_features(sample_ohlcv)
        assert not features.isna().any().any()

    def test_no_inf_values(self, sample_ohlcv):
        features = build_features(sample_ohlcv)
        assert not np.isinf(features.values).any()

    def test_expected_columns(self, sample_ohlcv):
        features = build_features(sample_ohlcv)
        expected = {
            "return_1d", "return_5d", "return_10d", "return_20d",
            "momentum_3", "momentum_7",
            "rsi",
            "macd", "macd_signal", "macd_hist",
            "bb_pctb",
            "trend", "price_sma20", "price_sma50", "sma20_sma50",
            "volume_ratio", "volume_change",
            "atr_ratio",
            "stoch_k",
            "range_position",
            "vol_10d", "vol_20d",
        }
        assert set(features.columns) == expected

    def test_rsi_in_range(self, sample_ohlcv):
        features = build_features(sample_ohlcv)
        assert features["rsi"].min() >= 0.0
        assert features["rsi"].max() <= 1.0


class TestBuildTargets:
    def test_default_horizon(self, sample_ohlcv):
        targets = build_targets(sample_ohlcv)
        assert len(targets) == len(sample_ohlcv)
        last_5 = targets.iloc[-5:]
        assert last_5.isna().all()

    def test_custom_horizon(self, sample_ohlcv):
        targets = build_targets(sample_ohlcv, horizon=1)
        assert targets.iloc[-1:].isna().all()
        assert not targets.iloc[:-1].isna().all()

    def test_return_calculation(self, sample_ohlcv):
        targets = build_targets(sample_ohlcv, horizon=1)
        close = sample_ohlcv["Close"]
        expected_0 = (close.iloc[1] - close.iloc[0]) / close.iloc[0]
        assert abs(targets.iloc[0] - expected_0) < 1e-10


class TestPrepareDataset:
    def test_aligned_indices(self, sample_ohlcv):
        features, targets = prepare_dataset(sample_ohlcv)
        assert features.index.equals(targets.index)

    def test_features_and_targets_length(self, sample_ohlcv):
        features, targets = prepare_dataset(sample_ohlcv)
        assert len(features) == len(targets)
        assert len(features) > 0
