from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests as _requests

from .config import HISTORY_PERIOD, MIN_DATA_POINTS, PREDICTION_HORIZON

log = logging.getLogger(__name__)

_YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
_MAX_RETRIES = 3
_BACKOFF_BASE = 4


def _parse_yahoo_response(data: dict) -> pd.DataFrame:
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0]

    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": adj.get("adjclose", quote["close"]),
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    df.index.name = "Date"
    return df.dropna(subset=["Close"])


def fetch_stock_data(ticker: str, period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    url = f"{_YAHOO_BASE}/{ticker}"
    params = {"range": period, "interval": "1d", "includePrePost": "false"}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = _requests.get(url, headers=_HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                wait = _BACKOFF_BASE * (attempt + 1)
                log.warning("%s: rate limited (429), retrying in %ds", ticker, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            df = _parse_yahoo_response(resp.json())
            if len(df) < MIN_DATA_POINTS:
                log.warning("%s: only %d rows (need %d)", ticker, len(df), MIN_DATA_POINTS)
                return None
            return df
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (attempt + 1)
                log.warning("%s: %s, retrying in %ds", ticker, exc, wait)
                time.sleep(wait)
            else:
                log.error("%s: failed after %d attempts: %s", ticker, _MAX_RETRIES, exc)
                return None
    return None


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pctb(series: pd.Series, period: int = 20, width: float = 2.0) -> pd.Series:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + width * std
    lower = sma - width * std
    return (series - lower) / (upper - lower)


def _stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    lowest = df["Low"].rolling(window=period).min()
    highest = df["High"].rolling(window=period).max()
    return (df["Close"] - lowest) / (highest - lowest) * 100


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    f = pd.DataFrame(index=df.index)

    f["return_1d"] = close.pct_change(1)
    f["return_5d"] = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)

    f["momentum_3"] = close / close.shift(3) - 1
    f["momentum_7"] = close / close.shift(7) - 1

    f["rsi"] = _rsi(close) / 100.0

    macd_line, signal_line, histogram = _macd(close)
    f["macd"] = macd_line / close
    f["macd_signal"] = signal_line / close
    f["macd_hist"] = histogram / close

    f["bb_pctb"] = _bollinger_pctb(close)

    sma10 = close.rolling(window=10).mean()
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    f["trend"] = (sma10 - sma20) / close
    f["price_sma20"] = close / sma20 - 1
    f["price_sma50"] = close / sma50 - 1
    f["sma20_sma50"] = sma20 / sma50 - 1

    vol_sma20 = volume.rolling(window=20).mean()
    f["volume_ratio"] = volume / vol_sma20
    f["volume_change"] = volume.pct_change()

    f["atr_ratio"] = _atr(df) / close

    f["stoch_k"] = _stochastic_k(df) / 100.0

    high_20 = df["High"].rolling(window=20).max()
    low_20 = df["Low"].rolling(window=20).min()
    f["range_position"] = (close - low_20) / (high_20 - low_20)

    f["vol_10d"] = close.pct_change().rolling(window=10).std()
    f["vol_20d"] = close.pct_change().rolling(window=20).std()

    return f.dropna()


def build_targets(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.Series:
    return df["Close"].pct_change(horizon).shift(-horizon)


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    features = build_features(df)
    targets = build_targets(df).reindex(features.index)
    return features, targets
