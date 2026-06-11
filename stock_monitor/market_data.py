from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import HISTORY_PERIOD
from .yahoo_client import fetch_chart
from .data import _parse_yahoo_response

log = logging.getLogger(__name__)

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication",
}


def fetch_market_data(period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    data = fetch_chart(SPY_TICKER, range_=period, interval="1d")
    if data is None:
        log.warning("Failed to fetch SPY market data")
        return None
    try:
        return _parse_yahoo_response(data)
    except (KeyError, IndexError, ValueError) as exc:
        log.error("Failed to parse SPY data: %s", exc)
        return None


def fetch_vix_data(period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    data = fetch_chart(VIX_TICKER, range_=period, interval="1d")
    if data is None:
        log.warning("Failed to fetch VIX data")
        return None
    try:
        return _parse_yahoo_response(data)
    except (KeyError, IndexError, ValueError) as exc:
        log.error("Failed to parse VIX data: %s", exc)
        return None


def fetch_sector_data(
    sector_etf: str,
    period: str = HISTORY_PERIOD,
) -> Optional[pd.DataFrame]:
    data = fetch_chart(sector_etf, range_=period, interval="1d")
    if data is None:
        return None
    try:
        return _parse_yahoo_response(data)
    except (KeyError, IndexError, ValueError):
        return None


def get_market_context(
    period: str = HISTORY_PERIOD,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    market_df = fetch_market_data(period)
    vix_df = fetch_vix_data(period)
    return market_df, vix_df
