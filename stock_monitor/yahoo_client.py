"""Shared Yahoo Finance HTTP client with session reuse and retry logic."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

_YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_MAX_RETRIES = 3
_BACKOFF_BASE = 4
_TIMEOUT = 15

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
    return _session


def get_session() -> requests.Session:
    """Return the shared HTTP session (connection-pooled)."""
    return _get_session()


def fetch_chart(
    ticker: str,
    range_: str = "5y",
    interval: str = "1d",
    timeout: int = _TIMEOUT,
) -> Optional[dict[str, Any]]:
    """Fetch Yahoo chart data with automatic retry on 429 rate limits.

    Returns the full JSON response dict, or None on failure.
    """
    session = _get_session()
    url = f"{_YAHOO_BASE}/{ticker}"
    params = {"range": range_, "interval": interval, "includePrePost": "false"}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = _BACKOFF_BASE * (attempt + 1)
                log.warning("%s: rate limited (429), retrying in %ds", ticker, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (attempt + 1)
                log.warning("%s: %s, retrying in %ds", ticker, exc, wait)
                time.sleep(wait)
            else:
                log.error("%s: failed after %d attempts: %s", ticker, _MAX_RETRIES, exc)
                return None
        except Exception as exc:
            log.error("%s: unexpected error: %s", ticker, exc)
            return None
    return None


def fetch_quote(ticker: str) -> tuple[float, float]:
    """Fetch current price and daily change % for a ticker.

    Returns (price, change_pct). Falls back to (0.0, 0.0) on failure.
    """
    data = fetch_chart(ticker, range_="1d", interval="1d", timeout=10)
    if data is None:
        return 0.0, 0.0
    try:
        result = data["chart"]["result"][0]
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice", 0.0)
        prev = meta.get("chartPreviousClose") or meta.get("previousClose") or 0.0
        if price and prev and prev > 0:
            change = round((price - prev) / prev * 100, 2)
        else:
            change = 0.0
        return round(float(price), 2), change
    except (KeyError, IndexError, TypeError) as exc:
        log.debug("Quote parse failed for %s: %s", ticker, exc)
        return 0.0, 0.0
