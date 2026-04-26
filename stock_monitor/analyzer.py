from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Signal, SIGNAL_THRESHOLDS, PREDICTION_HORIZON
from .data import fetch_stock_data, prepare_dataset
from .predictor import train_model, predict, PredictionResult

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
    error: Optional[str] = None
    reasons: list[str] = field(default_factory=list)

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
            "reasons": self.reasons,
            "error": self.error,
        }

    @classmethod
    def failed(cls, ticker: str, message: str) -> StockAnalysis:
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
            error=message,
        )


def _classify(prediction: PredictionResult) -> tuple[Signal, int, list[str]]:
    ret = prediction.predicted_return
    conf = prediction.confidence
    adjusted = ret * conf
    reasons: list[str] = []

    if adjusted >= SIGNAL_THRESHOLDS[Signal.STRONG_BUY]:
        signal = Signal.STRONG_BUY
    elif adjusted >= SIGNAL_THRESHOLDS[Signal.BUY]:
        signal = Signal.BUY
    elif adjusted >= SIGNAL_THRESHOLDS[Signal.LEAN_BUY]:
        signal = Signal.LEAN_BUY
    elif adjusted <= SIGNAL_THRESHOLDS[Signal.STRONG_SELL]:
        signal = Signal.STRONG_SELL
    elif adjusted <= SIGNAL_THRESHOLDS[Signal.SELL]:
        signal = Signal.SELL
    elif adjusted <= SIGNAL_THRESHOLDS[Signal.LEAN_SELL]:
        signal = Signal.LEAN_SELL
    else:
        signal = Signal.HOLD

    score = int(max(-100, min(100, adjusted * 2000)))

    reasons.append(
        f"LSTM predicts {ret * 100:+.2f}% over {PREDICTION_HORIZON} days"
    )
    reasons.append(f"Model confidence: {conf * 100:.0f}%")

    if conf >= 0.8:
        reasons.append("High prediction consistency across recent windows")
    elif conf <= 0.4:
        reasons.append("Low prediction consistency -- treat with caution")

    if prediction.model_age_days > 5:
        reasons.append(f"Model trained {prediction.model_age_days:.0f} days ago -- consider retraining")

    return signal, score, reasons


def _support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    recent = df.tail(lookback)
    return round(float(recent["Low"].min()), 2), round(float(recent["High"].max()), 2)


def _current_indicators(
    df: pd.DataFrame, features_df: pd.DataFrame
) -> tuple[float, float, float]:
    close = df["Close"]
    sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
    rsi_val = round(float(features_df["rsi"].iloc[-1]) * 100, 1)
    return sma20, sma50, rsi_val


def analyze(ticker: str, force_retrain: bool = False) -> StockAnalysis:
    try:
        df = fetch_stock_data(ticker)
        if df is None:
            return StockAnalysis.failed(ticker, "Insufficient historical data")

        features_df, targets = prepare_dataset(df)
        valid_mask = targets.notna()
        train_features = features_df[valid_mask].values
        train_targets = targets[valid_mask].values

        model, scaler = train_model(
            ticker, train_features, train_targets, force=force_retrain
        )

        prediction = predict(ticker, model, scaler, features_df.values)
        signal, score, reasons = _classify(prediction)

        price = round(float(df["Close"].iloc[-1]), 2)
        prev_close = float(df["Close"].iloc[-2])
        change_pct = round((price - prev_close) / prev_close * 100, 2)
        support, resistance = _support_resistance(df)
        sma20, sma50, rsi = _current_indicators(df, features_df)

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

        return StockAnalysis(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            signal=signal,
            score=score,
            predicted_return_pct=round(prediction.predicted_return * 100, 2),
            confidence=prediction.confidence,
            model_age_days=prediction.model_age_days,
            support=support,
            resistance=resistance,
            sma_20=sma20,
            sma_50=sma50,
            rsi=rsi,
            reasons=reasons,
        )
    except Exception as exc:
        log.exception("Failed to analyze %s", ticker)
        return StockAnalysis.failed(ticker, str(exc))
