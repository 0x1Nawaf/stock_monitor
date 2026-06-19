from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import Signal, GBM_WEIGHT, LSTM_WEIGHT
from .gbm import GBMPrediction
from .lstm_clf import LSTMPrediction

log = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    model_age_days: float
    gbm_agrees: bool
    lstm_agrees: bool
    ensemble_agreement: float
    directional_strength: float = 0.0


def _compute_confidence(
    probabilities: np.ndarray,
    predicted_class: int,
    agreement: float,
) -> float:
    """Confidence based on probability dominance and model agreement.

    Uses the actual probability spread to determine conviction:
    - If prob_up = 0.7, prob_down = 0.1 -> very confident bullish
    - If both models agree -> boost confidence further
    """
    prob_top = float(probabilities[predicted_class])
    sorted_probs = np.sort(probabilities)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    base_confidence = prob_top

    margin_boost = min(0.15, margin * 0.5)

    agreement_factor = 0.85 + 0.15 * agreement

    confidence = (base_confidence + margin_boost) * agreement_factor

    return round(np.clip(confidence, 0.05, 0.98), 3)


def _directional_strength(probabilities: np.ndarray) -> float:
    """How strongly the prediction leans bullish or bearish. Range: -1 to +1."""
    prob_up = float(probabilities[2])
    prob_down = float(probabilities[0])
    return prob_up - prob_down


def combine_predictions(
    gbm_pred: Optional[GBMPrediction],
    lstm_pred: Optional[LSTMPrediction],
    gbm_weight: float = GBM_WEIGHT,
    lstm_weight: float = LSTM_WEIGHT,
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
            directional_strength=0.0,
        )

    if gbm_pred is None:
        conf = _compute_confidence(lstm_pred.probabilities, lstm_pred.predicted_class, 1.0)
        return EnsemblePrediction(
            predicted_class=lstm_pred.predicted_class,
            probabilities=lstm_pred.probabilities,
            confidence=conf,
            model_age_days=lstm_pred.model_age_days,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=_directional_strength(lstm_pred.probabilities),
        )

    if lstm_pred is None:
        conf = _compute_confidence(gbm_pred.probabilities, gbm_pred.predicted_class, 1.0)
        return EnsemblePrediction(
            predicted_class=gbm_pred.predicted_class,
            probabilities=gbm_pred.probabilities,
            confidence=conf,
            model_age_days=gbm_pred.model_age_days,
            gbm_agrees=True,
            lstm_agrees=True,
            ensemble_agreement=1.0,
            directional_strength=_directional_strength(gbm_pred.probabilities),
        )

    total_weight = gbm_weight + lstm_weight
    w_gbm = gbm_weight / total_weight
    w_lstm = lstm_weight / total_weight

    gbm_direction = int(np.argmax(gbm_pred.probabilities))
    lstm_direction = int(np.argmax(lstm_pred.probabilities))

    if gbm_direction == lstm_direction:
        boost = 1.15
        combined_probs = (
            w_gbm * gbm_pred.probabilities + w_lstm * lstm_pred.probabilities
        )
        combined_probs[gbm_direction] *= boost
        combined_probs /= combined_probs.sum()
        agreement = 1.0
    elif (gbm_direction == 2 and lstm_direction == 1) or (gbm_direction == 0 and lstm_direction == 1):
        combined_probs = (
            0.65 * gbm_pred.probabilities + 0.35 * lstm_pred.probabilities
        )
        agreement = 0.75
    elif (lstm_direction == 2 and gbm_direction == 1) or (lstm_direction == 0 and gbm_direction == 1):
        combined_probs = (
            0.45 * gbm_pred.probabilities + 0.55 * lstm_pred.probabilities
        )
        agreement = 0.75
    else:
        combined_probs = (
            w_gbm * gbm_pred.probabilities + w_lstm * lstm_pred.probabilities
        )
        agreement = 0.4

    predicted_class = int(np.argmax(combined_probs))
    gbm_agrees = gbm_direction == predicted_class
    lstm_agrees = lstm_direction == predicted_class

    confidence = _compute_confidence(combined_probs, predicted_class, agreement)
    dir_strength = _directional_strength(combined_probs)

    age = max(gbm_pred.model_age_days, lstm_pred.model_age_days)

    return EnsemblePrediction(
        predicted_class=predicted_class,
        probabilities=combined_probs,
        confidence=confidence,
        model_age_days=age,
        gbm_agrees=gbm_agrees,
        lstm_agrees=lstm_agrees,
        ensemble_agreement=agreement,
        directional_strength=dir_strength,
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

    dir_strength = prediction.directional_strength
    confidence = prediction.confidence
    agreement = prediction.ensemble_agreement

    score = int(np.clip(dir_strength * confidence * 200, -100, 100))

    signal = _determine_signal(
        prob_up, prob_down, prob_flat,
        confidence, agreement, prediction.predicted_class,
    )

    reasons = _build_reasons(
        signal, prediction, prob_up, prob_down, prob_flat,
        horizon_label, confidence,
    )

    return signal, score, reasons


def _determine_signal(
    prob_up: float,
    prob_down: float,
    prob_flat: float,
    confidence: float,
    agreement: float,
    predicted_class: int,
) -> Signal:
    from ..targets import TargetClass

    spread = prob_up - prob_down

    if predicted_class == TargetClass.UP:
        if spread > 0.30 and confidence >= 0.70 and agreement >= 0.9:
            return Signal.STRONG_BUY
        elif spread > 0.20 and confidence >= 0.60:
            return Signal.BUY
        elif spread > 0.08 and confidence >= 0.45:
            return Signal.LEAN_BUY
        else:
            return Signal.HOLD

    elif predicted_class == TargetClass.DOWN:
        if spread < -0.30 and confidence >= 0.70 and agreement >= 0.9:
            return Signal.STRONG_SELL
        elif spread < -0.20 and confidence >= 0.60:
            return Signal.SELL
        elif spread < -0.08 and confidence >= 0.45:
            return Signal.LEAN_SELL
        else:
            return Signal.HOLD

    else:
        if spread > 0.12 and confidence >= 0.50:
            return Signal.LEAN_BUY
        elif spread < -0.12 and confidence >= 0.50:
            return Signal.LEAN_SELL
        return Signal.HOLD


def _build_reasons(
    signal: Signal,
    prediction: EnsemblePrediction,
    prob_up: float,
    prob_down: float,
    prob_flat: float,
    horizon_label: str,
    confidence: float,
) -> list[str]:
    from ..targets import TargetClass

    reasons = []
    class_names = {TargetClass.UP: "UP", TargetClass.DOWN: "DOWN", TargetClass.FLAT: "FLAT"}

    reasons.append(
        f"Prediction: {class_names[prediction.predicted_class]} over {horizon_label} "
        f"({confidence:.0%} confidence)"
    )
    reasons.append(
        f"Probabilities: UP={prob_up:.0%} FLAT={prob_flat:.0%} DOWN={prob_down:.0%}"
    )

    if prediction.ensemble_agreement >= 0.9:
        reasons.append("Strong model consensus -- both GBM & LSTM agree")
    elif prediction.ensemble_agreement <= 0.5:
        reasons.append("Models disagree -- lower conviction")

    strength = abs(prediction.directional_strength)
    if strength > 0.4:
        reasons.append(f"Strong directional bias ({strength:.0%})")
    elif strength > 0.2:
        reasons.append(f"Moderate directional lean ({strength:.0%})")

    if prediction.model_age_days > 5:
        reasons.append(f"Model is {prediction.model_age_days:.0f} days old -- consider retraining")

    return reasons
