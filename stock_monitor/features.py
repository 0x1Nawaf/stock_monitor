from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    f = pd.DataFrame(index=df.index)

    f["return_1d"] = close.pct_change(1)
    f["return_5d"] = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)

    f["momentum_3"] = close / close.shift(3) - 1
    f["momentum_7"] = close / close.shift(7) - 1
    f["momentum_14"] = close / close.shift(14) - 1
    f["momentum_21"] = close / close.shift(21) - 1

    f["rsi"] = _rsi(close) / 100.0
    f["rsi_5"] = _rsi(close, period=5) / 100.0
    f["rsi_21"] = _rsi(close, period=21) / 100.0

    macd_line, signal_line, histogram = _macd(close)
    f["macd"] = macd_line / close
    f["macd_signal"] = signal_line / close
    f["macd_hist"] = histogram / close

    f["bb_pctb"] = _bollinger_pctb(close)
    f["bb_width"] = (
        close.rolling(20).std() * 2 / close.rolling(20).mean()
    )

    sma10 = close.rolling(window=10).mean()
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()

    f["trend"] = (sma10 - sma20) / close
    f["price_sma20"] = close / sma20 - 1
    f["price_sma50"] = close / sma50 - 1
    f["price_sma200"] = close / sma200 - 1
    f["sma20_sma50"] = sma20 / sma50 - 1
    f["sma50_sma200"] = sma50 / sma200 - 1

    vol_sma20 = volume.rolling(window=20).mean()
    f["volume_ratio"] = volume / vol_sma20
    f["volume_change"] = volume.pct_change()
    f["volume_trend"] = volume.rolling(5).mean() / vol_sma20

    f["atr_ratio"] = _atr(df) / close
    f["stoch_k"] = _stochastic_k(df) / 100.0

    high_20 = df["High"].rolling(window=20).max()
    low_20 = df["Low"].rolling(window=20).min()
    f["range_position"] = (close - low_20) / (high_20 - low_20)

    f["vol_5d"] = close.pct_change().rolling(window=5).std()
    f["vol_10d"] = close.pct_change().rolling(window=10).std()
    f["vol_20d"] = close.pct_change().rolling(window=20).std()
    f["vol_ratio"] = f["vol_5d"] / (f["vol_20d"] + 1e-8)

    f["gap"] = df["Open"] / close.shift(1) - 1

    f["trend_consistency"] = (
        close.rolling(20).apply(lambda x: (x > x.mean()).sum() / len(x), raw=True)
    )

    f["mean_reversion"] = -(close / sma20 - 1)

    f["high_low_range"] = (df["High"] - df["Low"]) / close

    f["close_position"] = (close - df["Low"]) / (df["High"] - df["Low"] + 1e-8)

    f["day_of_week"] = pd.Series(
        df.index.dayofweek / 4.0, index=df.index
    )
    f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    return f.replace([np.inf, -np.inf], np.nan).dropna()


def build_market_features(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if market_df is None:
        return pd.DataFrame(index=df.index)

    close = df["Close"]
    mkt_close = market_df["Close"].reindex(df.index, method="ffill")

    f = pd.DataFrame(index=df.index)

    mkt_ret_1d = mkt_close.pct_change(1)
    mkt_ret_5d = mkt_close.pct_change(5)
    mkt_ret_20d = mkt_close.pct_change(20)

    f["market_return_1d"] = mkt_ret_1d
    f["market_return_5d"] = mkt_ret_5d
    f["market_return_20d"] = mkt_ret_20d

    stock_ret = close.pct_change(1)
    f["relative_strength_1d"] = stock_ret - mkt_ret_1d
    f["relative_strength_5d"] = close.pct_change(5) - mkt_ret_5d
    f["relative_strength_20d"] = close.pct_change(20) - mkt_ret_20d

    mkt_vol = mkt_close.pct_change().rolling(20).std()
    f["market_volatility"] = mkt_vol
    f["vol_spread"] = close.pct_change().rolling(20).std() - mkt_vol

    cov = stock_ret.rolling(60).cov(mkt_ret_1d)
    mkt_var = mkt_ret_1d.rolling(60).var()
    f["beta"] = cov / (mkt_var + 1e-8)
    f["alpha_20d"] = close.pct_change(20) - f["beta"] * mkt_ret_20d

    return f.replace([np.inf, -np.inf], np.nan)


def build_vix_features(
    df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if vix_df is None:
        return pd.DataFrame(index=df.index)

    vix_close = vix_df["Close"].reindex(df.index, method="ffill")
    f = pd.DataFrame(index=df.index)

    f["vix_level"] = vix_close / 100.0
    f["vix_change"] = vix_close.pct_change(1)
    f["vix_sma20_ratio"] = vix_close / vix_close.rolling(20).mean()
    f["vix_rank"] = vix_close.rolling(252).rank(pct=True)

    return f.replace([np.inf, -np.inf], np.nan)


def build_all_features(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    base = build_base_features(df)
    market = build_market_features(df, market_df)
    vix = build_vix_features(df, vix_df)

    combined = base.join(market, how="left").join(vix, how="left")
    combined = combined.replace([np.inf, -np.inf], np.nan)

    nan_cols = combined.columns[combined.isna().all()]
    if len(nan_cols) > 0:
        combined = combined.drop(columns=nan_cols)

    combined = combined.dropna()
    return combined


def get_feature_names(
    include_market: bool = False,
    include_vix: bool = False,
) -> list[str]:
    base_names = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "momentum_3", "momentum_7", "momentum_14", "momentum_21",
        "rsi", "rsi_5", "rsi_21",
        "macd", "macd_signal", "macd_hist",
        "bb_pctb", "bb_width",
        "trend", "price_sma20", "price_sma50", "price_sma200",
        "sma20_sma50", "sma50_sma200",
        "volume_ratio", "volume_change", "volume_trend",
        "atr_ratio", "stoch_k", "range_position",
        "vol_5d", "vol_10d", "vol_20d", "vol_ratio",
        "gap", "trend_consistency", "mean_reversion",
        "high_low_range", "close_position",
        "day_of_week", "month_sin", "month_cos",
    ]

    names = list(base_names)

    if include_market:
        names.extend([
            "market_return_1d", "market_return_5d", "market_return_20d",
            "relative_strength_1d", "relative_strength_5d", "relative_strength_20d",
            "market_volatility", "vol_spread", "beta", "alpha_20d",
        ])

    if include_vix:
        names.extend([
            "vix_level", "vix_change", "vix_sma20_ratio", "vix_rank",
        ])

    return names
