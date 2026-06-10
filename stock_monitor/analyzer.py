from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Signal, SIGNAL_THRESHOLDS, PREDICTION_HORIZON, TIMEFRAME_5D, TimeframeConfig
from .data import fetch_stock_data, fetch_live_price, prepare_dataset
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
    stop_loss: float = 0.0
    timeframe: str = "5d"
    market: str = "US"
    currency: str = "$"
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
            "stop_loss": self.stop_loss,
            "timeframe": self.timeframe,
            "market": self.market,
            "currency": self.currency,
            "reasons": self.reasons,
            "error": self.error,
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


def _classify(
    prediction: PredictionResult,
    thresholds: dict[Signal, float] = SIGNAL_THRESHOLDS,
    horizon_label: str = "5 trading days",
) -> tuple[Signal, int, list[str]]:
    ret = prediction.predicted_return
    conf = prediction.confidence
    adjusted = ret * conf
    reasons: list[str] = []

    if adjusted >= thresholds[Signal.STRONG_BUY]:
        signal = Signal.STRONG_BUY
    elif adjusted >= thresholds[Signal.BUY]:
        signal = Signal.BUY
    elif adjusted >= thresholds[Signal.LEAN_BUY]:
        signal = Signal.LEAN_BUY
    elif adjusted <= thresholds[Signal.STRONG_SELL]:
        signal = Signal.STRONG_SELL
    elif adjusted <= thresholds[Signal.SELL]:
        signal = Signal.SELL
    elif adjusted <= thresholds[Signal.LEAN_SELL]:
        signal = Signal.LEAN_SELL
    else:
        signal = Signal.HOLD

    score = int(max(-100, min(100, adjusted * 2000)))

    if signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY):
        reasons.append(f"Buy signal over {horizon_label}")
    elif signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL):
        reasons.append(f"Sell signal over {horizon_label}")
    else:
        reasons.append(f"No clear direction over {horizon_label}")

    reasons.append(
        f"Model predicts {ret * 100:+.2f}% over {horizon_label}"
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


def analyze(
    ticker: str,
    force_retrain: bool = False,
    timeframe: TimeframeConfig = TIMEFRAME_5D,
    market: str = "US",
    currency: str = "$",
) -> StockAnalysis:
    tf_key_map = {1: "1d", 5: "5d", 10: "swing", 21: "monthly"}
    tf_key = tf_key_map.get(timeframe.horizon, f"{timeframe.horizon}d")

    try:
        df = fetch_stock_data(ticker)
        if df is None:
            return StockAnalysis.failed(ticker, "Insufficient historical data", tf_key, market, currency)

        features_df, targets = prepare_dataset(df, horizon=timeframe.horizon)
        valid_mask = targets.notna()
        train_features = features_df[valid_mask].values
        train_targets = targets[valid_mask].values

        model, scaler = train_model(
            ticker,
            train_features,
            train_targets,
            force=force_retrain,
            models_dir=timeframe.models_dir,
            seq_len=timeframe.sequence_length,
            max_age_days=timeframe.model_max_age_days,
        )

        prediction = predict(
            ticker,
            model,
            scaler,
            features_df.values,
            models_dir=timeframe.models_dir,
            seq_len=timeframe.sequence_length,
        )
        signal, score, reasons = _classify(
            prediction,
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
            stop_loss=stop_loss,
            timeframe=tf_key,
            market=market,
            currency=currency,
            reasons=reasons,
        )
    except Exception as exc:
        log.exception("Failed to analyze %s", ticker)
        return StockAnalysis.failed(ticker, str(exc), tf_key, market, currency)
