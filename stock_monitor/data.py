from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import HISTORY_PERIOD, MIN_DATA_POINTS
from .yahoo_client import fetch_chart

log = logging.getLogger(__name__)


def _parse_yahoo_response(data: dict) -> pd.DataFrame:
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0]

    raw_close = np.array(quote["close"], dtype="float64")
    adj_close = np.array(adj.get("adjclose", quote["close"]), dtype="float64")

    # Adjust Open/High/Low by the same ratio so OHLC relationships hold
    # after dividend/split adjustments (avoids Close < Low anomalies).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((raw_close > 0) & np.isfinite(raw_close), adj_close / raw_close, 1.0)
    ratio = np.nan_to_num(ratio, nan=1.0)

    df = pd.DataFrame(
        {
            "Open": np.array(quote["open"], dtype="float64") * ratio,
            "High": np.array(quote["high"], dtype="float64") * ratio,
            "Low": np.array(quote["low"], dtype="float64") * ratio,
            "Close": adj_close,
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    df.index.name = "Date"
    return df.dropna(subset=["Close"])


def fetch_stock_data(ticker: str, period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    data = fetch_chart(ticker, range_=period, interval="1d")
    if data is None:
        return None

    try:
        df = _parse_yahoo_response(data)
        pre_filter = len(df)
        df = df[df["Volume"] > 0]
        if pre_filter != len(df):
            log.info("%s: dropped %d zero-volume rows (non-trading days)", ticker, pre_filter - len(df))
        if len(df) < MIN_DATA_POINTS:
            log.warning("%s: only %d rows (need %d)", ticker, len(df), MIN_DATA_POINTS)
            return None
        return df
    except (KeyError, IndexError, ValueError) as exc:
        log.error("%s: failed to parse response: %s", ticker, exc)
        return None


@dataclass
class LivePrice:
    price: float
    change_pct: float
    timestamp: str


def _live_price_from_meta(meta: dict) -> Optional[LivePrice]:
    """Extract a live price from Yahoo chart response metadata.

    The ``regularMarketPrice`` field in ``meta`` reflects the most recent
    known price regardless of the chart range or interval requested, so
    it works both during and outside market hours.
    """
    price = meta.get("regularMarketPrice")
    if price is None:
        return None

    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    if prev_close and prev_close > 0:
        change_pct = round((price - prev_close) / prev_close * 100, 2)
    else:
        change_pct = 0.0

    market_time = meta.get("regularMarketTime")
    if market_time:
        ts_str = pd.Timestamp(market_time, unit="s").strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = ""

    return LivePrice(price=round(float(price), 2), change_pct=change_pct, timestamp=ts_str)


def fetch_live_price(ticker: str) -> Optional[LivePrice]:
    """Fetch the latest market price for *ticker*.

    Always uses a lightweight ``range=1d&interval=1d`` chart request so
    that ``chartPreviousClose`` reflects yesterday's close (not the start
    of a multi-year chart range cached by ``fetch_stock_data``).
    """
    data = fetch_chart(ticker, range_="1d", interval="1d")
    if data is None:
        return None

    try:
        chart_result = data["chart"]["result"][0]
        meta = chart_result.get("meta", {})

        live = _live_price_from_meta(meta)
        if live is not None:
            return live

        timestamps = chart_result.get("timestamp")
        quote = chart_result["indicators"]["quote"][0]
        if timestamps and quote.get("close"):
            closes = quote["close"]
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] is not None:
                    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
                    change_pct = (
                        round((closes[i] - prev_close) / prev_close * 100, 2)
                        if prev_close and prev_close > 0
                        else 0.0
                    )
                    ts_str = pd.Timestamp(timestamps[i], unit="s").strftime("%Y-%m-%d %H:%M")
                    return LivePrice(
                        price=round(float(closes[i]), 2),
                        change_pct=change_pct,
                        timestamp=ts_str,
                    )

        log.warning("%s: no price data in response", ticker)
        return None
    except (KeyError, IndexError, TypeError) as exc:
        log.error("%s live: failed to parse response: %s", ticker, exc)
        return None


