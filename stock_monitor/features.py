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


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff())
    return (df["Volume"] * direction).cumsum()


def _adl(df: pd.DataFrame) -> pd.Series:
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"] + 1e-8
    )
    return (clv * df["Volume"]).cumsum()


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev + 1e-8)


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    return (highest - df["Close"]) / (highest - lowest + 1e-8) * -100


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    ratio = pos_mf / (neg_mf + 1e-8)
    return 100 - (100 / (1 + ratio))


def _keltner_position(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    ema = df["Close"].ewm(span=period, adjust=False).mean()
    atr_val = _atr(df, period)
    upper = ema + mult * atr_val
    lower = ema - mult * atr_val
    return (df["Close"] - lower) / (upper - lower + 1e-8)


def _donchian_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    return (df["Close"] - lower) / (upper - lower + 1e-8)


def _trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """ADX-inspired trend strength: proportion of days moving in dominant direction."""
    changes = close.diff()
    pos_count = changes.rolling(period).apply(lambda x: (x > 0).sum(), raw=True)
    neg_count = changes.rolling(period).apply(lambda x: (x < 0).sum(), raw=True)
    return (pos_count - neg_count).abs() / period


def _consecutive_direction(close: pd.Series) -> pd.Series:
    """Count consecutive up or down days (positive=up, negative=down)."""
    direction = np.sign(close.diff())
    result = pd.Series(0.0, index=close.index)
    count = 0.0
    prev_dir = 0.0
    for i in range(len(direction)):
        d = direction.iloc[i]
        if d == prev_dir:
            count += d
        else:
            count = d
        prev_dir = d
        result.iloc[i] = count
    return result


def _volume_price_trend(df: pd.DataFrame) -> pd.Series:
    ret = df["Close"].pct_change()
    return (ret * df["Volume"]).cumsum()


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    f = pd.DataFrame(index=df.index)

    f["return_1d"] = close.pct_change(1)
    f["return_2d"] = close.pct_change(2)
    f["return_3d"] = close.pct_change(3)
    f["return_5d"] = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)
    f["return_60d"] = close.pct_change(60)

    f["momentum_3"] = close / close.shift(3) - 1
    f["momentum_5"] = close / close.shift(5) - 1
    f["momentum_7"] = close / close.shift(7) - 1
    f["momentum_10"] = close / close.shift(10) - 1
    f["momentum_14"] = close / close.shift(14) - 1
    f["momentum_21"] = close / close.shift(21) - 1

    f["acceleration_5"] = f["momentum_5"] - f["momentum_5"].shift(5)
    f["acceleration_10"] = f["momentum_10"] - f["momentum_10"].shift(10)

    f["rsi"] = _rsi(close) / 100.0
    f["rsi_5"] = _rsi(close, period=5) / 100.0
    f["rsi_9"] = _rsi(close, period=9) / 100.0
    f["rsi_21"] = _rsi(close, period=21) / 100.0
    f["rsi_divergence"] = f["rsi"] - f["rsi"].shift(5)

    macd_line, signal_line, histogram = _macd(close)
    f["macd"] = macd_line / close
    f["macd_signal"] = signal_line / close
    f["macd_hist"] = histogram / close
    f["macd_hist_change"] = f["macd_hist"] - f["macd_hist"].shift(1)
    f["macd_crossover"] = (
        (macd_line > signal_line).astype(float) -
        (macd_line.shift(1) > signal_line.shift(1)).astype(float)
    )

    f["bb_pctb"] = _bollinger_pctb(close)
    f["bb_pctb_10"] = _bollinger_pctb(close, period=10)
    f["bb_width"] = close.rolling(20).std() * 2 / close.rolling(20).mean()
    f["bb_squeeze"] = (f["bb_width"] < f["bb_width"].rolling(120).quantile(0.1)).astype(float)

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    sma10 = close.rolling(window=10).mean()
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()

    f["ema_5_21_spread"] = (ema5 - ema21) / close
    f["ema_9_21_spread"] = (ema9 - ema21) / close
    f["ema_12_trend"] = (close - ema12) / close

    f["trend_short"] = (ema5 - ema21) / close
    f["trend_medium"] = (sma20 - sma50) / close
    f["trend_long"] = (sma50 - sma200) / close

    f["price_sma10"] = close / sma10 - 1
    f["price_sma20"] = close / sma20 - 1
    f["price_sma50"] = close / sma50 - 1
    f["price_sma200"] = close / sma200 - 1
    f["sma20_sma50"] = sma20 / sma50 - 1
    f["sma50_sma200"] = sma50 / sma200 - 1

    f["golden_cross"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(float)
    f["death_cross"] = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).astype(float)
    f["above_200sma"] = (close > sma200).astype(float)
    f["sma_alignment"] = (
        (ema5 > ema21).astype(float) +
        (ema21 > sma50).astype(float) +
        (sma50 > sma200).astype(float)
    ) / 3.0

    vol_sma5 = volume.rolling(window=5).mean()
    vol_sma20 = volume.rolling(window=20).mean()
    vol_sma50 = volume.rolling(window=50).mean()
    f["volume_ratio_5_20"] = vol_sma5 / (vol_sma20 + 1)
    f["volume_ratio_20_50"] = vol_sma20 / (vol_sma50 + 1)
    f["volume_spike"] = volume / (vol_sma20 + 1)
    f["volume_trend"] = volume.rolling(5).mean() / (vol_sma20 + 1)
    f["volume_price_confirm"] = f["return_1d"] * f["volume_spike"]

    obv = _obv(df)
    obv_sma = obv.rolling(20).mean()
    f["obv_trend"] = (obv - obv_sma) / (obv_sma.abs() + 1e-8)

    adl = _adl(df)
    adl_sma = adl.rolling(20).mean()
    f["adl_trend"] = (adl - adl_sma) / (adl_sma.abs() + 1e-8)

    f["mfi"] = _mfi(df) / 100.0

    f["atr_ratio"] = _atr(df) / close
    f["atr_expansion"] = _atr(df, 5) / (_atr(df, 20) + 1e-8)
    f["stoch_k"] = _stochastic_k(df) / 100.0
    f["stoch_k_9"] = _stochastic_k(df, period=9) / 100.0
    f["williams_r"] = _williams_r(df) / 100.0
    f["cci"] = _cci(df) / 200.0

    f["keltner_pos"] = _keltner_position(df)
    f["donchian_pos"] = _donchian_position(df)
    f["donchian_pos_50"] = _donchian_position(df, period=50)

    high_20 = df["High"].rolling(window=20).max()
    low_20 = df["Low"].rolling(window=20).min()
    f["range_position"] = (close - low_20) / (high_20 - low_20 + 1e-8)

    high_52w = df["High"].rolling(window=252).max()
    low_52w = df["Low"].rolling(window=252).min()
    f["range_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-8)
    f["from_52w_high"] = close / high_52w - 1
    f["from_52w_low"] = close / low_52w - 1

    f["vol_5d"] = close.pct_change().rolling(window=5).std()
    f["vol_10d"] = close.pct_change().rolling(window=10).std()
    f["vol_20d"] = close.pct_change().rolling(window=20).std()
    f["vol_ratio"] = f["vol_5d"] / (f["vol_20d"] + 1e-8)
    f["vol_regime"] = f["vol_20d"].rolling(60).rank(pct=True)
    f["realized_vs_expected"] = f["return_1d"].abs() / (f["vol_20d"] + 1e-8)

    f["gap"] = df["Open"] / close.shift(1) - 1
    f["gap_fill"] = (
        ((df["Open"] > close.shift(1)) & (close < df["Open"])).astype(float) -
        ((df["Open"] < close.shift(1)) & (close > df["Open"])).astype(float)
    )

    f["trend_strength_10"] = _trend_strength(close, 10)
    f["trend_strength_20"] = _trend_strength(close, 20)
    f["trend_consistency"] = (
        close.rolling(20).apply(lambda x: (x > x.mean()).sum() / len(x), raw=True)
    )

    f["mean_reversion_20"] = -(close / sma20 - 1)
    f["mean_reversion_50"] = -(close / sma50 - 1)

    f["high_low_range"] = (df["High"] - df["Low"]) / close
    f["close_position"] = (close - df["Low"]) / (df["High"] - df["Low"] + 1e-8)

    f["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / close
    f["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / close
    f["body_size"] = (close - df["Open"]).abs() / close
    f["candle_direction"] = np.sign(close - df["Open"])

    f["consecutive_days"] = _consecutive_direction(close) / 10.0

    f["higher_highs"] = (
        (df["High"] > df["High"].shift(1)).rolling(5).sum() / 5
    )
    f["higher_lows"] = (
        (df["Low"] > df["Low"].shift(1)).rolling(5).sum() / 5
    )
    f["price_structure"] = f["higher_highs"] + f["higher_lows"] - 1.0

    f["day_of_week"] = pd.Series(df.index.dayofweek / 4.0, index=df.index)
    f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    f["quarter"] = (df.index.quarter - 1) / 3.0

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

    f["rs_momentum"] = f["relative_strength_5d"] - f["relative_strength_5d"].shift(5)

    mkt_sma50 = mkt_close.rolling(50).mean()
    mkt_sma200 = mkt_close.rolling(200).mean()
    f["market_trend"] = (mkt_sma50 - mkt_sma200) / mkt_close
    f["market_regime"] = (mkt_close > mkt_sma200).astype(float)

    mkt_vol = mkt_close.pct_change().rolling(20).std()
    f["market_volatility"] = mkt_vol
    f["vol_spread"] = close.pct_change().rolling(20).std() - mkt_vol

    cov = stock_ret.rolling(60).cov(mkt_ret_1d)
    mkt_var = mkt_ret_1d.rolling(60).var()
    f["beta"] = cov / (mkt_var + 1e-8)
    f["alpha_20d"] = close.pct_change(20) - f["beta"] * mkt_ret_20d

    f["correlation_20d"] = stock_ret.rolling(20).corr(mkt_ret_1d)
    f["correlation_60d"] = stock_ret.rolling(60).corr(mkt_ret_1d)
    f["decorrelation"] = f["correlation_20d"] - f["correlation_60d"]

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
    f["vix_change_1d"] = vix_close.pct_change(1)
    f["vix_change_5d"] = vix_close.pct_change(5)
    f["vix_sma20_ratio"] = vix_close / vix_close.rolling(20).mean()
    f["vix_rank"] = vix_close.rolling(252).rank(pct=True)
    f["vix_regime_low"] = (vix_close < 15).astype(float)
    f["vix_regime_high"] = (vix_close > 25).astype(float)
    f["vix_term_structure"] = vix_close.pct_change(5) - vix_close.pct_change(1)

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

    combined = combined.ffill(limit=3).dropna()
    return combined
