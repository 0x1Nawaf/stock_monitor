from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Signal, TIMEFRAME_5D, TimeframeConfig
from .data import fetch_stock_data, fetch_live_price
from .features import build_all_features, build_market_structure
from .targets import (
    build_binary_targets,
    build_binary_targets_down,
    build_classification_targets,
    get_thresholds,
    get_class_weights,
    TargetClass,
)
from .model.gbm import train_gbm, predict_gbm, GBMPrediction
from .model.lstm_clf import train_lstm_classifier, predict_lstm, LSTMPrediction
from .model.ensemble import combine_predictions, prediction_to_signal, EnsemblePrediction

log = logging.getLogger(__name__)


@dataclass
class StockAnalysis:
    ticker: str
    price: float
    change_pct: float
    signal: Signal
    score: int
    predicted_return_pct: float
    confidence: float
    model_age_days: float
    support: float
    resistance: float
    sma_20: float
    sma_50: float
    rsi: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timeframe: str = "5d"
    market: str = "US"
    currency: str = "$"
    error: Optional[str] = None
    reasons: list[str] = field(default_factory=list)
    prob_up: float = 0.0
    prob_down: float = 0.0
    prob_flat: float = 0.0
    ensemble_agreement: float = 0.0
    model_type: str = "ensemble"
    market_structure: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "change_pct": self.change_pct,
            "signal": self.signal.value,
            "score": self.score,
            "predicted_return_pct": self.predicted_return_pct,
            "confidence": self.confidence,
            "model_age_days": self.model_age_days,
            "support": self.support,
            "resistance": self.resistance,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "rsi": self.rsi,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timeframe": self.timeframe,
            "market": self.market,
            "currency": self.currency,
            "reasons": self.reasons,
            "error": self.error,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "prob_flat": self.prob_flat,
            "ensemble_agreement": self.ensemble_agreement,
            "model_type": self.model_type,
            "market_structure": self.market_structure,
        }

    @classmethod
    def failed(cls, ticker: str, message: str, timeframe: str = "5d", market: str = "US", currency: str = "$") -> StockAnalysis:
        return cls(
            ticker=ticker,
            price=0.0,
            change_pct=0.0,
            signal=Signal.HOLD,
            score=0,
            predicted_return_pct=0.0,
            confidence=0.0,
            model_age_days=0.0,
            support=0.0,
            resistance=0.0,
            sma_20=0.0,
            sma_50=0.0,
            rsi=0.0,
            timeframe=timeframe,
            market=market,
            currency=currency,
            error=message,
        )


def _support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    recent = df.tail(lookback)
    return round(float(recent["Low"].min()), 2), round(float(recent["High"].max()), 2)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    return float(atr_series.iloc[-1])


def _stop_loss(
    price: float,
    signal: Signal,
    support: float,
    resistance: float,
    atr: float,
    multiplier: float,
) -> float:
    if signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY):
        atr_stop = price - (atr * multiplier)
        return round(max(atr_stop, support * 0.99), 2)
    elif signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL):
        atr_stop = price + (atr * multiplier)
        return round(min(atr_stop, resistance * 1.01), 2)
    return 0.0


def _take_profit(
    price: float,
    signal: Signal,
    predicted_return_pct: float,
    resistance: float,
    support: float,
    atr: float,
    multiplier: float,
) -> float:
    if signal in (Signal.STRONG_BUY, Signal.BUY):
        target_from_prediction = price * (1 + predicted_return_pct / 100)
        target_from_resistance = resistance * 0.98
        atr_target = price + (atr * multiplier * 1.5)
        target = max(target_from_prediction, min(target_from_resistance, atr_target))
        return round(target, 2)
    elif signal in (Signal.STRONG_SELL, Signal.SELL):
        target_from_prediction = price * (1 + predicted_return_pct / 100)
        target_from_support = support * 1.02
        atr_target = price - (atr * multiplier * 1.5)
        target = min(target_from_prediction, max(target_from_support, atr_target))
        return round(target, 2)
    return 0.0


def _current_indicators(
    df: pd.DataFrame, features_df: pd.DataFrame
) -> tuple[float, float, float]:
    close = df["Close"]
    sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
    rsi_val = round(float(features_df["rsi"].iloc[-1]) * 100, 1)
    return sma20, sma50, rsi_val


def _analyze_market_structure(
    df: pd.DataFrame, swing_length: int = 5
) -> tuple[str, list[str]]:
    """Run BOS/MSS analysis and return a structure summary + reason lines.

    Returns:
        (structure_label, reasons) where structure_label is one of:
        "bullish_bos", "bearish_bos", "bullish_mss", "bearish_mss",
        "bullish", "bearish", or "neutral".
    """
    ms = build_market_structure(df, swing_length=swing_length)
    if ms.empty:
        return "neutral", []

    reasons: list[str] = []
    last = ms.iloc[-1]

    direction = last.get("structure_direction", 0.0)
    score = last.get("structure_score", 0.0)

    lookback = min(3, len(ms))
    recent = ms.tail(lookback)

    recent_bos_bull = recent["bos_bullish"].sum() > 0
    recent_bos_bear = recent["bos_bearish"].sum() > 0
    recent_mss_bull = recent["mss_bullish"].sum() > 0
    recent_mss_bear = recent["mss_bearish"].sum() > 0

    recent_hh = recent["swing_high_hh"].sum() > 0
    recent_hl = recent["swing_low_hl"].sum() > 0
    recent_lh = recent["swing_high_lh"].sum() > 0
    recent_ll = recent["swing_low_ll"].sum() > 0

    if recent_hh:
        reasons.append("Swing structure: Higher High detected")
    if recent_hl:
        reasons.append("Swing structure: Higher Low detected")
    if recent_lh:
        reasons.append("Swing structure: Lower High detected")
    if recent_ll:
        reasons.append("Swing structure: Lower Low detected")

    if recent_mss_bull:
        reasons.append("Market Structure Shift (MSS) -- bullish reversal signal")
        return "bullish_mss", reasons
    if recent_mss_bear:
        reasons.append("Market Structure Shift (MSS) -- bearish reversal signal")
        return "bearish_mss", reasons
    if recent_bos_bull:
        reasons.append("Break of Structure (BOS) -- bullish continuation")
        return "bullish_bos", reasons
    if recent_bos_bear:
        reasons.append("Break of Structure (BOS) -- bearish continuation")
        return "bearish_bos", reasons

    if direction > 0:
        reasons.append(f"Market structure: bullish (score {score:.0f})")
        return "bullish", reasons
    elif direction < 0:
        reasons.append(f"Market structure: bearish (score {score:.0f})")
        return "bearish", reasons

    return "neutral", reasons


_HORIZON_EXPECTED_MOVE = {
    1: (0.025, -0.025),
    5: (0.07, -0.07),
    10: (0.12, -0.12),
    21: (0.18, -0.18),
}


def _estimated_return_from_probs(
    probs: np.ndarray, horizon: int, confidence: float = 0.5
) -> float:
    """Estimate the predicted return based on probability distribution.

    Uses realistic expected moves (not just classification thresholds).
    The threshold is the MINIMUM to qualify as UP — actual moves are much larger.
    """
    expected_up, expected_down = _HORIZON_EXPECTED_MOVE.get(
        horizon, (0.005 * horizon, -0.005 * horizon)
    )

    directional_strength = float(probs[TargetClass.UP] - probs[TargetClass.DOWN])

    predicted = directional_strength * max(abs(expected_up), abs(expected_down)) * confidence

    return float(predicted)


def analyze(
    ticker: str,
    force_retrain: bool = False,
    timeframe: TimeframeConfig = TIMEFRAME_5D,
    market: str = "US",
    currency: str = "$",
    market_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    use_lstm: bool = True,
) -> StockAnalysis:
    tf_key_map = {1: "1d", 5: "5d", 10: "swing", 21: "monthly"}
    tf_key = tf_key_map.get(timeframe.horizon, f"{timeframe.horizon}d")

    try:
        df = fetch_stock_data(ticker)
        if df is None:
            return StockAnalysis.failed(ticker, "Insufficient historical data", tf_key, market, currency)

        is_swing = timeframe.horizon == 10
        features_df = build_all_features(
            df, market_df, vix_df,
            include_structure=is_swing,
            swing_length=5,
        )
        if len(features_df) < 200:
            return StockAnalysis.failed(ticker, "Insufficient features after engineering", tf_key, market, currency)

        horizon = timeframe.horizon
        up_thresh, down_thresh = get_thresholds(horizon)

        targets_up = build_binary_targets(df, horizon, up_thresh)
        targets_up = targets_up.reindex(features_df.index)

        targets_down = build_binary_targets_down(df, horizon, down_thresh)
        targets_down = targets_down.reindex(features_df.index)

        targets_cls = build_classification_targets(df, horizon, up_thresh, down_thresh)
        targets_cls = targets_cls.reindex(features_df.index)

        features_arr = features_df.values
        targets_up_arr = targets_up.values
        targets_down_arr = targets_down.values
        targets_cls_arr = targets_cls.values

        class_weights = get_class_weights(targets_cls_arr)

        gbm_model_up, gbm_model_down = train_gbm(
            ticker,
            features_arr,
            targets_up_arr,
            targets_down_arr,
            force=force_retrain,
            models_dir=timeframe.models_dir,
            max_age_days=timeframe.model_max_age_days,
        )
        gbm_pred = predict_gbm(
            ticker, gbm_model_up, gbm_model_down,
            features_arr, models_dir=timeframe.models_dir,
        )

        lstm_pred = None
        if use_lstm:
            try:
                lstm_model = train_lstm_classifier(
                    ticker,
                    features_arr,
                    targets_cls_arr,
                    force=force_retrain,
                    models_dir=timeframe.models_dir,
                    seq_len=timeframe.sequence_length,
                    max_age_days=timeframe.model_max_age_days,
                    class_weights=class_weights,
                )
                lstm_pred = predict_lstm(
                    ticker, lstm_model, features_arr,
                    models_dir=timeframe.models_dir,
                    seq_len=timeframe.sequence_length,
                )
            except Exception as exc:
                log.warning("LSTM failed for %s, using GBM only: %s", ticker, exc)

        ensemble = combine_predictions(gbm_pred, lstm_pred)
        signal, score, reasons = prediction_to_signal(
            ensemble,
            thresholds=timeframe.signal_thresholds,
            horizon_label=timeframe.label,
        )

        price = round(float(df["Close"].iloc[-1]), 2)
        prev_close = float(df["Close"].iloc[-2])
        change_pct = round((price - prev_close) / prev_close * 100, 2)

        live = fetch_live_price(ticker)
        if live is not None:
            price = live.price
            change_pct = live.change_pct

        if signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY):
            reasons.insert(0, f"Buy at {currency}{price}")
        elif signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL):
            reasons.insert(0, f"Sell at {currency}{price}")

        support, resistance = _support_resistance(df)
        sma20, sma50, rsi = _current_indicators(df, features_df)

        atr = _compute_atr(df)
        stop_loss = _stop_loss(
            price, signal, support, resistance,
            atr, timeframe.atr_stop_multiplier,
        )

        predicted_return = _estimated_return_from_probs(
            ensemble.probabilities, horizon, ensemble.confidence
        )

        take_profit = _take_profit(
            price, signal, predicted_return * 100,
            resistance, support, atr, timeframe.atr_stop_multiplier,
        )

        if stop_loss > 0:
            sl_pct = abs(stop_loss - price) / price * 100
            if signal in (Signal.STRONG_BUY, Signal.BUY):
                reasons.insert(1, f"Stop loss at {currency}{stop_loss:.2f} ({sl_pct:.1f}% below entry)")
            elif signal in (Signal.STRONG_SELL, Signal.SELL):
                reasons.insert(1, f"Stop loss at {currency}{stop_loss:.2f} ({sl_pct:.1f}% above entry)")

        if take_profit > 0:
            tp_pct = abs(take_profit - price) / price * 100
            insert_pos = 2 if stop_loss > 0 else 1
            if signal in (Signal.STRONG_BUY, Signal.BUY):
                reasons.insert(insert_pos, f"Take profit at {currency}{take_profit:.2f} (+{tp_pct:.1f}% from entry)")
            elif signal in (Signal.STRONG_SELL, Signal.SELL):
                reasons.insert(insert_pos, f"Take profit at {currency}{take_profit:.2f} (-{tp_pct:.1f}% from entry)")

        if price > sma20:
            reasons.append(f"Price above SMA(20) at {sma20}")
        else:
            reasons.append(f"Price below SMA(20) at {sma20}")

        if price > sma50:
            reasons.append(f"Price above SMA(50) at {sma50}")
        else:
            reasons.append(f"Price below SMA(50) at {sma50}")

        if rsi > 70:
            reasons.append(f"RSI at {rsi} -- overbought territory")
        elif rsi < 30:
            reasons.append(f"RSI at {rsi} -- oversold territory")

        structure_label = ""
        if is_swing:
            structure_label, structure_reasons = _analyze_market_structure(df)
            reasons.extend(structure_reasons)

            bullish_signal = signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY)
            bearish_signal = signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL)

            if structure_label == "bullish_mss" and bearish_signal:
                signal = Signal.HOLD
                score = int(score * 0.3)
                reasons.append("Signal dampened: MSS contradicts bearish prediction")
            elif structure_label == "bearish_mss" and bullish_signal:
                signal = Signal.HOLD
                score = int(score * 0.3)
                reasons.append("Signal dampened: MSS contradicts bullish prediction")
            elif structure_label == "bullish_bos" and bullish_signal:
                if signal == Signal.LEAN_BUY:
                    signal = Signal.BUY
                    reasons.append("Signal upgraded: BOS confirms bullish momentum")
                elif signal == Signal.BUY:
                    score = min(100, int(score * 1.15))
                    reasons.append("Conviction boosted: BOS confirms bullish trend")
            elif structure_label == "bearish_bos" and bearish_signal:
                if signal == Signal.LEAN_SELL:
                    signal = Signal.SELL
                    reasons.append("Signal upgraded: BOS confirms bearish momentum")
                elif signal == Signal.SELL:
                    score = max(-100, int(score * 1.15))
                    reasons.append("Conviction boosted: BOS confirms bearish trend")
            elif structure_label == "bullish_bos" and bearish_signal:
                reasons.append("Caution: bullish BOS conflicts with bearish signal")
            elif structure_label == "bearish_bos" and bullish_signal:
                reasons.append("Caution: bearish BOS conflicts with bullish signal")

        model_type = "ensemble" if lstm_pred is not None else "gbm"

        return StockAnalysis(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            signal=signal,
            score=score,
            predicted_return_pct=round(predicted_return * 100, 2),
            confidence=ensemble.confidence,
            model_age_days=ensemble.model_age_days,
            support=support,
            resistance=resistance,
            sma_20=sma20,
            sma_50=sma50,
            rsi=rsi,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=tf_key,
            market=market,
            currency=currency,
            reasons=reasons,
            prob_up=round(float(ensemble.probabilities[TargetClass.UP]), 3),
            prob_down=round(float(ensemble.probabilities[TargetClass.DOWN]), 3),
            prob_flat=round(float(ensemble.probabilities[TargetClass.FLAT]), 3),
            ensemble_agreement=ensemble.ensemble_agreement,
            model_type=model_type,
            market_structure=structure_label,
        )
    except Exception as exc:
        log.exception("Failed to analyze %s", ticker)
        return StockAnalysis.failed(ticker, str(exc), tf_key, market, currency)
