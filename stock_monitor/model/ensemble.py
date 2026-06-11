from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import Signal
from .gbm import GBMPrediction
from .lstm_clf import LSTMPrediction

log = logging.getLogger(__name__)

DEFAULT_GBM_WEIGHT = 0.6
DEFAULT_LSTM_WEIGHT = 0.4

BASELINE_PROB = 1.0 / 3.0


@dataclass
class EnsemblePrediction:
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    model_age_days: float
    gbm_agrees: bool
    lstm_agrees: bool
    ensemble_agreement: float


def _calibrate_confidence(probabilities: np.ndarray, predicted_class: int = -1) -> float:
    if predicted_class < 0:
        predicted_class = int(np.argmax(probabilities))

    prob_top = float(probabilities[predicted_class])

    if predicted_class == 2:
        prob_opposite = float(probabilities[0])
    elif predicted_class == 0:
        prob_opposite = float(probabilities[2])
    else:
        prob_opposite = float(max(probabilities[0], probabilities[2]))

    if prob_top + prob_opposite < 1e-8:
        return 0.0

    directional = prob_top / (prob_top + prob_opposite)
    edge = max(0.0, prob_top - BASELINE_PROB) / (1.0 - BASELINE_PROB)
    confidence = 0.7 * directional + 0.3 * edge
    return round(max(0.0, min(0.99, confidence)), 3)


def combine_predictions(
    gbm_pred: Optional[GBMPrediction],
    lstm_pred: Optional[LSTMPrediction],
    gbm_weight: float = DEFAULT_GBM_WEIGHT,
    lstm_weight: float = DEFAULT_LSTM_WEIGHT,
) -> EnsemblePrediction:
    if gbm_pred is None and lstm_pred is None:
        return EnsemblePrediction(
            predicted_class=1,
            probabilities=np.array([0.33, 0.34, 0.33]),
            confidence=0.0,
            model_age_days=0.0,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
        )

    if gbm_pred is None:
        conf = _calibrate_confidence(lstm_pred.probabilities, lstm_pred.predicted_class)
        return EnsemblePrediction(
            predicted_class=lstm_pred.predicted_class,
            probabilities=lstm_pred.probabilities,
            confidence=conf,
            model_age_days=lstm_pred.model_age_days,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
        )

    if lstm_pred is None:
        conf = _calibrate_confidence(gbm_pred.probabilities, gbm_pred.predicted_class)
        return EnsemblePrediction(
            predicted_class=gbm_pred.predicted_class,
            probabilities=gbm_pred.probabilities,
            confidence=conf,
            model_age_days=gbm_pred.model_age_days,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
        )

    total_weight = gbm_weight + lstm_weight
    w_gbm = gbm_weight / total_weight
    w_lstm = lstm_weight / total_weight

    combined_probs = w_gbm * gbm_pred.probabilities + w_lstm * lstm_pred.probabilities
    predicted_class = int(np.argmax(combined_probs))
    confidence = _calibrate_confidence(combined_probs, predicted_class)

    gbm_agrees = gbm_pred.predicted_class == predicted_class
    lstm_agrees = lstm_pred.predicted_class == predicted_class

    if gbm_agrees and lstm_agrees:
        agreement = 1.0
    elif gbm_agrees or lstm_agrees:
        agreement = 0.7
    else:
        agreement = 0.4

    age = max(gbm_pred.model_age_days, lstm_pred.model_age_days)

    return EnsemblePrediction(
        predicted_class=predicted_class,
        probabilities=combined_probs,
        confidence=confidence,
        model_age_days=age,
        gbm_agrees=gbm_agrees,
        lstm_agrees=lstm_agrees,
        ensemble_agreement=agreement,
    )


def prediction_to_signal(
    prediction: EnsemblePrediction,
    thresholds: dict[Signal, float],
    horizon_label: str = "5 trading days",
) -> tuple[Signal, int, list[str]]:
    from ..targets import TargetClass

    prob_up = float(prediction.probabilities[TargetClass.UP])
    prob_down = float(prediction.probabilities[TargetClass.DOWN])
    prob_flat = float(prediction.probabilities[TargetClass.FLAT])

    edge_up = prob_up - BASELINE_PROB
    edge_down = prob_down - BASELINE_PROB
    net_edge = edge_up - edge_down

    conviction = net_edge * prediction.ensemble_agreement
    score = int(max(-100, min(100, conviction * 300)))

    strong_buy_t = thresholds.get(Signal.STRONG_BUY, 0.04)
    buy_t = thresholds.get(Signal.BUY, 0.02)
    lean_buy_t = thresholds.get(Signal.LEAN_BUY, 0.005)
    lean_sell_t = thresholds.get(Signal.LEAN_SELL, -0.005)
    sell_t = thresholds.get(Signal.SELL, -0.02)
    strong_sell_t = thresholds.get(Signal.STRONG_SELL, -0.04)

    scale = strong_buy_t / 0.04 if strong_buy_t > 0 else 1.0
    prob_strong = 0.20 * scale
    prob_buy = 0.10 * scale
    prob_lean = 0.03 * scale
    prob_flat_edge = lean_buy_t * 2

    reasons = []

    if prediction.predicted_class == TargetClass.UP:
        if prob_up > prob_down + prob_strong and prediction.ensemble_agreement >= 0.9:
            signal = Signal.STRONG_BUY
        elif prob_up > prob_down + prob_buy:
            signal = Signal.BUY
        elif prob_up > prob_down + prob_lean:
            signal = Signal.LEAN_BUY
        else:
            signal = Signal.HOLD
    elif prediction.predicted_class == TargetClass.DOWN:
        if prob_down > prob_up + prob_strong and prediction.ensemble_agreement >= 0.9:
            signal = Signal.STRONG_SELL
        elif prob_down > prob_up + prob_buy:
            signal = Signal.SELL
        elif prob_down > prob_up + prob_lean:
            signal = Signal.LEAN_SELL
        else:
            signal = Signal.HOLD
    else:
        if edge_up > prob_flat_edge:
            signal = Signal.LEAN_BUY
        elif edge_down > prob_flat_edge:
            signal = Signal.LEAN_SELL
        else:
            signal = Signal.HOLD

    class_names = {TargetClass.UP: "UP", TargetClass.DOWN: "DOWN", TargetClass.FLAT: "FLAT"}
    reasons.append(
        f"Ensemble predicts {class_names[prediction.predicted_class]} over {horizon_label}"
    )
    reasons.append(
        f"Probabilities: UP={prob_up:.0%} FLAT={prob_flat:.0%} DOWN={prob_down:.0%}"
    )
    reasons.append(f"Confidence: {prediction.confidence:.0%}")

    if prediction.ensemble_agreement >= 0.9:
        reasons.append("Both models agree on direction")
    elif prediction.ensemble_agreement <= 0.5:
        reasons.append("Models disagree -- signal less reliable")

    if prediction.model_age_days > 5:
        reasons.append(f"Model trained {prediction.model_age_days:.0f} days ago -- consider retraining")

    return signal, score, reasons
