"""Tests for signal classification and analysis logic."""
from __future__ import annotations

import pytest

from stock_monitor.config import Signal, SIGNAL_THRESHOLDS, DAILY_SIGNAL_THRESHOLDS
from stock_monitor.analyzer import _classify
from stock_monitor.predictor import PredictionResult


class TestClassify:
    def test_strong_buy(self):
        pred = PredictionResult(predicted_return=0.06, confidence=0.9, model_age_days=1.0)
        signal, score, reasons = _classify(pred)
        assert signal == Signal.STRONG_BUY
        assert score > 50

    def test_strong_sell(self):
        pred = PredictionResult(predicted_return=-0.06, confidence=0.9, model_age_days=1.0)
        signal, score, reasons = _classify(pred)
        assert signal == Signal.STRONG_SELL
        assert score < -50

    def test_hold_low_return(self):
        pred = PredictionResult(predicted_return=0.001, confidence=0.5, model_age_days=1.0)
        signal, score, reasons = _classify(pred)
        assert signal == Signal.HOLD

    def test_confidence_reduces_signal(self):
        pred_high = PredictionResult(predicted_return=0.03, confidence=0.9, model_age_days=1.0)
        pred_low = PredictionResult(predicted_return=0.03, confidence=0.2, model_age_days=1.0)
        sig_high, score_high, _ = _classify(pred_high)
        sig_low, score_low, _ = _classify(pred_low)
        assert score_high > score_low

    def test_reasons_include_prediction(self):
        pred = PredictionResult(predicted_return=0.02, confidence=0.7, model_age_days=1.0)
        _, _, reasons = _classify(pred)
        assert any("predicts" in r.lower() for r in reasons)

    def test_old_model_warning(self):
        pred = PredictionResult(predicted_return=0.02, confidence=0.7, model_age_days=6.0)
        _, _, reasons = _classify(pred)
        assert any("retrain" in r.lower() for r in reasons)

    def test_daily_thresholds(self):
        pred = PredictionResult(predicted_return=0.02, confidence=0.9, model_age_days=1.0)
        signal, _, _ = _classify(pred, thresholds=DAILY_SIGNAL_THRESHOLDS)
        assert signal == Signal.STRONG_BUY

    def test_score_clamped_to_range(self):
        pred = PredictionResult(predicted_return=0.5, confidence=0.95, model_age_days=0.1)
        _, score, _ = _classify(pred)
        assert -100 <= score <= 100

    def test_all_signals_reachable(self):
        """Verify every signal level can be triggered."""
        test_cases = [
            (0.06, 0.9, Signal.STRONG_BUY),
            (0.03, 0.9, Signal.BUY),
            (0.008, 0.9, Signal.LEAN_BUY),
            (0.001, 0.5, Signal.HOLD),
            (-0.008, 0.9, Signal.LEAN_SELL),
            (-0.03, 0.9, Signal.SELL),
            (-0.06, 0.9, Signal.STRONG_SELL),
        ]
        for ret, conf, expected_signal in test_cases:
            pred = PredictionResult(predicted_return=ret, confidence=conf, model_age_days=1.0)
            signal, _, _ = _classify(pred)
            assert signal == expected_signal, f"Expected {expected_signal} for ret={ret}, conf={conf}, got {signal}"
