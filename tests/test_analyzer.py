from __future__ import annotations

import numpy as np
import pytest

from stock_monitor.config import Signal, SIGNAL_THRESHOLDS, DAILY_SIGNAL_THRESHOLDS
from stock_monitor.model.ensemble import (
    combine_predictions,
    prediction_to_signal,
    EnsemblePrediction,
    _compute_confidence,
)
from stock_monitor.model.gbm import GBMPrediction
from stock_monitor.model.lstm_clf import LSTMPrediction
from stock_monitor.targets import TargetClass


class TestComputeConfidence:
    def test_random_probs_give_low(self):
        conf = _compute_confidence(np.array([0.33, 0.34, 0.33]), 1, 1.0)
        assert conf < 0.5

    def test_strong_prediction_gives_high(self):
        conf = _compute_confidence(np.array([0.05, 0.15, 0.80]), 2, 1.0)
        assert conf > 0.7

    def test_moderate_prediction(self):
        conf = _compute_confidence(np.array([0.20, 0.30, 0.50]), 2, 1.0)
        assert 0.4 < conf < 0.8

    def test_clamped_to_range(self):
        conf = _compute_confidence(np.array([0.0, 0.0, 1.0]), 2, 1.0)
        assert 0.0 <= conf <= 1.0

    def test_agreement_boosts_confidence(self):
        conf_high = _compute_confidence(np.array([0.1, 0.2, 0.7]), 2, 1.0)
        conf_low = _compute_confidence(np.array([0.1, 0.2, 0.7]), 2, 0.4)
        assert conf_high > conf_low


class TestPredictionToSignal:
    def test_strong_buy_high_edge(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.10, 0.15, 0.75]),
            confidence=0.9,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.65,
        )
        signal, score, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert signal == Signal.STRONG_BUY
        assert score > 50

    def test_buy_moderate_edge(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.20, 0.25, 0.55]),
            confidence=0.6,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.35,
        )
        signal, score, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert signal in (Signal.BUY, Signal.LEAN_BUY)
        assert score > 0

    def test_sell_high_edge(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.DOWN,
            probabilities=np.array([0.75, 0.15, 0.10]),
            confidence=0.9,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=-0.65,
        )
        signal, score, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert signal == Signal.STRONG_SELL
        assert score < -50

    def test_hold_flat_prediction(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.FLAT,
            probabilities=np.array([0.30, 0.40, 0.30]),
            confidence=0.4,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.0,
        )
        signal, score, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert signal == Signal.HOLD

    def test_low_agreement_reduces_score(self):
        pred_agree = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.15, 0.25, 0.60]),
            confidence=0.7,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.45,
        )
        pred_disagree = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.15, 0.25, 0.60]),
            confidence=0.5,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=False,
            ensemble_agreement=0.4,
            directional_strength=0.45,
        )
        _, score_agree, _ = prediction_to_signal(pred_agree, SIGNAL_THRESHOLDS)
        _, score_disagree, _ = prediction_to_signal(pred_disagree, SIGNAL_THRESHOLDS)
        assert score_agree > score_disagree

    def test_reasons_include_probabilities(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.1, 0.3, 0.6]),
            confidence=0.6,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.5,
        )
        _, _, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert any("Probabilities" in r for r in reasons)

    def test_old_model_warning(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.1, 0.3, 0.6]),
            confidence=0.6,
            model_age_days=7.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.5,
        )
        _, _, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert any("retrain" in r.lower() or "old" in r.lower() for r in reasons)

    def test_lean_buy_small_edge(self):
        pred = EnsemblePrediction(
            predicted_class=TargetClass.UP,
            probabilities=np.array([0.28, 0.27, 0.45]),
            confidence=0.5,
            model_age_days=1.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=0.17,
        )
        signal, score, reasons = prediction_to_signal(pred, SIGNAL_THRESHOLDS)
        assert signal in (Signal.LEAN_BUY, Signal.BUY)
        assert score > 0


class TestCombinePredictions:
    def test_gbm_only(self):
        gbm = GBMPrediction(
            predicted_class=2,
            probabilities=np.array([0.1, 0.2, 0.7]),
            confidence=0.7,
            model_age_days=1.0,
        )
        result = combine_predictions(gbm, None)
        assert result.predicted_class == 2
        assert result.confidence > 0.5

    def test_lstm_only(self):
        lstm = LSTMPrediction(
            predicted_class=0,
            probabilities=np.array([0.7, 0.2, 0.1]),
            confidence=0.7,
            model_age_days=1.0,
        )
        result = combine_predictions(None, lstm)
        assert result.predicted_class == 0

    def test_ensemble_combines_probs(self):
        gbm = GBMPrediction(
            predicted_class=2,
            probabilities=np.array([0.1, 0.2, 0.7]),
            confidence=0.7,
            model_age_days=1.0,
        )
        lstm = LSTMPrediction(
            predicted_class=2,
            probabilities=np.array([0.15, 0.25, 0.6]),
            confidence=0.6,
            model_age_days=1.0,
        )
        result = combine_predictions(gbm, lstm)
        assert result.predicted_class == 2
        assert result.gbm_agrees
        assert result.lstm_agrees
        assert result.ensemble_agreement == 1.0
        assert abs(result.probabilities.sum() - 1.0) < 1e-6

    def test_disagreement(self):
        gbm = GBMPrediction(
            predicted_class=2,
            probabilities=np.array([0.1, 0.2, 0.7]),
            confidence=0.7,
            model_age_days=1.0,
        )
        lstm = LSTMPrediction(
            predicted_class=0,
            probabilities=np.array([0.6, 0.2, 0.2]),
            confidence=0.6,
            model_age_days=1.0,
        )
        result = combine_predictions(gbm, lstm)
        assert result.ensemble_agreement < 1.0

    def test_both_none_returns_flat(self):
        result = combine_predictions(None, None)
        assert result.predicted_class == 1
        assert result.confidence == 0.0
