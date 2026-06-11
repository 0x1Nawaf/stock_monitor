from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Signal, TIMEFRAME_5D, TimeframeConfig
from .data import fetch_stock_data, fetch_live_price
from .features import build_all_features
from .targets import (
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


def _current_indicators(
    df: pd.DataFrame, features_df: pd.DataFrame
) -> tuple[float, float, float]:
    close = df["Close"]
    sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
    rsi_val = round(float(features_df["rsi"].iloc[-1]) * 100, 1)
    return sma20, sma50, rsi_val


def _estimated_return_from_probs(probs: np.ndarray, horizon: int) -> float:
    up_thresh, down_thresh = get_thresholds(horizon)
    expected = (
        probs[TargetClass.UP] * up_thresh * 1.5
        + probs[TargetClass.FLAT] * 0.0
        + probs[TargetClass.DOWN] * down_thresh * 1.5
    )
    return float(expected)


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

        features_df = build_all_features(df, market_df, vix_df)
        if len(features_df) < 200:
            return StockAnalysis.failed(ticker, "Insufficient features after engineering", tf_key, market, currency)

        horizon = timeframe.horizon
        up_thresh, down_thresh = get_thresholds(horizon)
        targets = build_classification_targets(df, horizon, up_thresh, down_thresh)
        targets = targets.reindex(features_df.index)

        features_arr = features_df.values
        targets_arr = targets.values

        class_weights = get_class_weights(targets_arr)

        gbm_model = train_gbm(
            ticker,
            features_arr,
            targets_arr,
            force=force_retrain,
            models_dir=timeframe.models_dir,
            max_age_days=timeframe.model_max_age_days,
            class_weights=class_weights,
        )
        gbm_pred = predict_gbm(ticker, gbm_model, features_arr, models_dir=timeframe.models_dir)

        lstm_pred = None
        if use_lstm:
            try:
                lstm_model = train_lstm_classifier(
                    ticker,
                    features_arr,
                    targets_arr,
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

        if timeframe.live_price:
            live = fetch_live_price(ticker)
            if live is not None:
                price = live.price
                change_pct = live.change_pct
                reasons.append(f"Live price as of {live.timestamp}")
            else:
                reasons.append("Live price unavailable, using last daily close")

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

        if stop_loss > 0:
            sl_pct = abs(stop_loss - price) / price * 100
            if signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY):
                reasons.insert(1, f"Stop loss at {currency}{stop_loss:.2f} ({sl_pct:.1f}% below entry)")
            elif signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL):
                reasons.insert(1, f"Stop loss at {currency}{stop_loss:.2f} ({sl_pct:.1f}% above entry)")

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

        predicted_return = _estimated_return_from_probs(ensemble.probabilities, horizon)

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
            timeframe=tf_key,
            market=market,
            currency=currency,
            reasons=reasons,
            prob_up=round(float(ensemble.probabilities[TargetClass.UP]), 3),
            prob_down=round(float(ensemble.probabilities[TargetClass.DOWN]), 3),
            prob_flat=round(float(ensemble.probabilities[TargetClass.FLAT]), 3),
            ensemble_agreement=ensemble.ensemble_agreement,
            model_type=model_type,
        )
    except Exception as exc:
        log.exception("Failed to analyze %s", ticker)
        return StockAnalysis.failed(ticker, str(exc), tf_key, market, currency)
