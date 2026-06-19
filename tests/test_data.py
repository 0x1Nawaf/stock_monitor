from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_monitor.features import build_base_features, build_all_features
from stock_monitor.targets import (
    build_classification_targets,
    build_regression_targets,
    get_class_weights,
    target_distribution,
    TargetClass,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
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


class TestBuildBaseFeatures:
    def test_output_has_many_features(self, sample_ohlcv):
        features = build_base_features(sample_ohlcv)
        assert features.shape[1] >= 38
        assert len(features) > 0

    def test_no_nan_values(self, sample_ohlcv):
        features = build_base_features(sample_ohlcv)
        assert not features.isna().any().any()

    def test_no_inf_values(self, sample_ohlcv):
        features = build_base_features(sample_ohlcv)
        assert not np.isinf(features.values).any()

    def test_expected_columns_present(self, sample_ohlcv):
        features = build_base_features(sample_ohlcv)
        expected = {
            "return_1d", "return_5d", "return_10d", "return_20d",
            "momentum_3", "momentum_7", "momentum_14", "momentum_21",
            "rsi", "rsi_5", "rsi_21",
            "macd", "macd_signal", "macd_hist",
            "bb_pctb", "bb_width",
            "price_sma20", "price_sma50", "price_sma200",
            "sma20_sma50", "sma50_sma200",
            "volume_spike", "volume_trend",
            "atr_ratio", "stoch_k", "range_position",
            "vol_5d", "vol_10d", "vol_20d", "vol_ratio",
            "gap", "trend_consistency",
            "high_low_range", "close_position",
            "day_of_week", "month_sin", "month_cos",
        }
        assert expected.issubset(set(features.columns))

    def test_rsi_in_range(self, sample_ohlcv):
        features = build_base_features(sample_ohlcv)
        assert features["rsi"].min() >= 0.0
        assert features["rsi"].max() <= 1.0


class TestBuildAllFeatures:
    def test_without_market_data(self, sample_ohlcv):
        features = build_all_features(sample_ohlcv)
        assert features.shape[1] >= 38
        assert not features.isna().any().any()

    def test_with_market_data(self, sample_ohlcv):
        market_df = sample_ohlcv.copy()
        market_df["Close"] = market_df["Close"] * 5
        features = build_all_features(sample_ohlcv, market_df=market_df)
        assert features.shape[1] > 38
        assert "relative_strength_1d" in features.columns


class TestClassificationTargets:
    def test_output_has_three_classes(self, sample_ohlcv):
        targets = build_classification_targets(sample_ohlcv, horizon=5)
        valid = targets[targets >= 0]
        classes = set(valid.unique())
        assert classes.issubset({TargetClass.DOWN, TargetClass.FLAT, TargetClass.UP})

    def test_last_n_rows_invalid(self, sample_ohlcv):
        targets = build_classification_targets(sample_ohlcv, horizon=5)
        last_5 = targets.iloc[-5:]
        assert (last_5 == -1).all()

    def test_custom_thresholds(self, sample_ohlcv):
        targets = build_classification_targets(
            sample_ohlcv, horizon=5, up_threshold=0.001, down_threshold=-0.001
        )
        valid = targets[targets >= 0]
        assert (valid == TargetClass.FLAT).sum() < len(valid)

    def test_symmetric_thresholds(self, sample_ohlcv):
        targets = build_classification_targets(sample_ohlcv, horizon=5)
        valid = targets[targets >= 0]
        up_count = (valid == TargetClass.UP).sum()
        down_count = (valid == TargetClass.DOWN).sum()
        assert up_count > 0
        assert down_count > 0


class TestRegressionTargets:
    def test_output_matches_original(self, sample_ohlcv):
        targets = build_regression_targets(sample_ohlcv, horizon=5)
        assert len(targets) == len(sample_ohlcv)
        assert targets.iloc[-5:].isna().all()

    def test_return_calculation(self, sample_ohlcv):
        targets = build_regression_targets(sample_ohlcv, horizon=1)
        close = sample_ohlcv["Close"]
        expected_0 = (close.iloc[1] - close.iloc[0]) / close.iloc[0]
        assert abs(targets.iloc[0] - expected_0) < 1e-10


class TestClassWeights:
    def test_balanced_weights(self):
        targets = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        weights = get_class_weights(targets)
        assert abs(weights[0] - 1.0) < 0.1
        assert abs(weights[1] - 1.0) < 0.1
        assert abs(weights[2] - 1.0) < 0.1

    def test_imbalanced_weights(self):
        targets = np.array([0, 0, 0, 0, 0, 1, 2])
        weights = get_class_weights(targets)
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]


class TestTargetDistribution:
    def test_sums_to_one(self):
        targets = np.array([0, 0, 1, 1, 1, 2, 2])
        dist = target_distribution(targets)
        assert abs(dist["DOWN"] + dist["FLAT"] + dist["UP"] - 1.0) < 1e-6
