from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .yahoo_client import get_session, fetch_chart
from .config import MODELS_DIR

log = logging.getLogger(__name__)

_TIMEOUT = 15


@dataclass
class Gainer:
    ticker: str
    company: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    market_cap: float
    sector: str = ""
    volume_ratio: float = 0.0
    prev_close: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company": self.company,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "volume_ratio": self.volume_ratio,
            "market_cap": self.market_cap,
            "sector": self.sector,
            "prev_close": self.prev_close,
            "day_high": self.day_high,
            "day_low": self.day_low,
        }


_SCREENER_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
)


def _fetch_yahoo_gainers(count: int = 50) -> list[Gainer]:
    session = get_session()
    params = {
        "scrIds": "day_gainers",
        "count": count,
    }
    try:
        resp = session.get(_SCREENER_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Yahoo screener failed: %s", exc)
        return _fetch_yahoo_gainers_v6(count)

    quotes = (
        data
        .get("finance", {})
        .get("result", [{}])[0]
        .get("quotes", [])
    )
    if not quotes:
        log.info("Yahoo screener returned 0 quotes, trying fallback")
        return _fetch_yahoo_gainers_v6(count)

    gainers: list[Gainer] = []
    for q in quotes:
        ticker = q.get("symbol", "")
        if not ticker or "." in ticker or "-" in ticker:
            continue
        gainers.append(Gainer(
            ticker=ticker,
            company=q.get("shortName") or q.get("longName") or ticker,
            price=round(q.get("regularMarketPrice", 0.0), 2),
            change_pct=round(q.get("regularMarketChangePercent", 0.0), 2),
            volume=int(q.get("regularMarketVolume", 0)),
            avg_volume=int(q.get("averageDailyVolume3Month", 0)
                          or q.get("averageDailyVolume10Day", 0)),
            market_cap=q.get("marketCap", 0.0),
            sector=q.get("sector", ""),
            prev_close=round(q.get("regularMarketPreviousClose", 0.0), 2),
            day_high=round(q.get("regularMarketDayHigh", 0.0), 2),
            day_low=round(q.get("regularMarketDayLow", 0.0), 2),
            source="yahoo-screener",
        ))

    log.info("Yahoo screener: %d gainers", len(gainers))
    return gainers


def _fetch_yahoo_gainers_v6(count: int = 50) -> list[Gainer]:
    session = get_session()
    url = "https://query1.finance.yahoo.com/v6/finance/recommendationsbysymbol/AAPL"
    try:
        resp = session.get(
            "https://finance.yahoo.com/gainers",
            timeout=_TIMEOUT,
        )
    except Exception:
        pass

    url = f"https://query2.finance.yahoo.com/v1/finance/trending/US"
    try:
        resp = session.get(url, params={"count": count}, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        quotes = (
            data.get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
        tickers = [q.get("symbol", "") for q in quotes if q.get("symbol")]
        if tickers:
            return _enrich_tickers(tickers[:count])
    except Exception as exc:
        log.warning("Yahoo trending fallback failed: %s", exc)

    return []


def _enrich_tickers(tickers: list[str]) -> list[Gainer]:
    gainers: list[Gainer] = []
    for ticker in tickers:
        data = fetch_chart(ticker, range_="1d", interval="1d", timeout=10)
        if data is None:
            continue
        try:
            result = data["chart"]["result"][0]
            meta = result.get("meta", {})
            price = round(float(meta.get("regularMarketPrice", 0)), 2)
            prev = float(meta.get("chartPreviousClose", 0)
                         or meta.get("previousClose", 0))
            if price <= 0 or prev <= 0:
                continue
            change_pct = round((price - prev) / prev * 100, 2)
            if change_pct <= 0:
                continue
            gainers.append(Gainer(
                ticker=ticker,
                company=ticker,
                price=price,
                change_pct=change_pct,
                volume=0,
                avg_volume=0,
                market_cap=0.0,
                prev_close=round(prev, 2),
                source="yahoo-trending",
            ))
        except (KeyError, IndexError, TypeError):
            continue
        time.sleep(0.15)
    return gainers


def _fetch_finviz_gainers() -> list[Gainer]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed, Finviz source unavailable")
        return []

    session = get_session()
    url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    gainers: list[Gainer] = []

    try:
        resp = session.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = (
            soup.find("table", class_="screener_table")
            or soup.find("table", id="screener-views-table")
        )
        if table is None:
            for t in soup.find_all("table"):
                if t.find("a", class_="screener-link-primary"):
                    table = t
                    break
        if table is None:
            log.debug("No screener table found on Finviz gainers page")
            return []

        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            try:
                ticker_el = cells[1].find("a")
                if ticker_el is None:
                    continue
                ticker = ticker_el.get_text(strip=True).upper()
                if not ticker or not ticker.isalpha():
                    continue

                company = cells[2].get_text(strip=True) if len(cells) > 2 else ticker
                sector = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                mktcap_text = cells[6].get_text(strip=True) if len(cells) > 6 else "0"
                market_cap = _parse_market_cap(mktcap_text)

                price_text = cells[8].get_text(strip=True) if len(cells) > 8 else "0"
                price = _parse_float(price_text)

                change_text = cells[9].get_text(strip=True) if len(cells) > 9 else "0%"
                change_pct = _parse_float(change_text.replace("%", ""))

                vol_text = cells[10].get_text(strip=True) if len(cells) > 10 else "0"
                volume = _parse_int(vol_text)

                if change_pct <= 0 or price <= 0:
                    continue

                gainers.append(Gainer(
                    ticker=ticker,
                    company=company,
                    price=price,
                    change_pct=change_pct,
                    volume=volume,
                    avg_volume=0,
                    market_cap=market_cap,
                    sector=sector,
                    source="finviz",
                ))
            except (IndexError, ValueError):
                continue

    except Exception as exc:
        log.warning("Failed to fetch Finviz gainers: %s", exc)

    log.info("Finviz gainers: %d", len(gainers))
    return gainers


def _parse_float(text: str) -> float:
    try:
        return float(text.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _parse_int(text: str) -> int:
    try:
        return int(text.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _parse_market_cap(text: str) -> float:
    text = text.strip().upper()
    if not text or text == "-":
        return 0.0
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            return _parse_float(text[:-1]) * mult
    return _parse_float(text)


def _merge_gainers(
    yahoo: list[Gainer],
    finviz: list[Gainer],
) -> list[Gainer]:
    by_ticker: dict[str, Gainer] = {}

    for g in yahoo:
        by_ticker[g.ticker] = g

    for g in finviz:
        if g.ticker not in by_ticker:
            by_ticker[g.ticker] = g
        else:
            existing = by_ticker[g.ticker]
            if not existing.sector and g.sector:
                existing.sector = g.sector
            if not existing.company or existing.company == existing.ticker:
                existing.company = g.company

    return list(by_ticker.values())


def _compute_volume_ratios(gainers: list[Gainer]) -> None:
    for g in gainers:
        if g.avg_volume and g.avg_volume > 0:
            g.volume_ratio = round(g.volume / g.avg_volume, 2)
        elif g.volume > 0:
            g.volume_ratio = 1.0


def _rank_gainers(gainers: list[Gainer]) -> list[Gainer]:
    def _score(g: Gainer) -> float:
        change_score = min(g.change_pct, 50.0)
        vol_score = min(g.volume_ratio, 10.0) * 2 if g.volume_ratio > 0 else 0
        cap_bonus = 0.0
        if g.market_cap > 10e9:
            cap_bonus = 5.0
        elif g.market_cap > 1e9:
            cap_bonus = 3.0
        elif g.market_cap > 100e6:
            cap_bonus = 1.0
        return change_score + vol_score + cap_bonus

    gainers.sort(key=_score, reverse=True)
    return gainers


def _enrich_missing_data(gainers: list[Gainer]) -> None:
    for g in gainers:
        if g.volume > 0 and g.avg_volume > 0:
            continue
        data = fetch_chart(g.ticker, range_="1d", interval="1d", timeout=10)
        if data is None:
            continue
        try:
            result = data["chart"]["result"][0]
            meta = result.get("meta", {})
            if g.price <= 0:
                g.price = round(float(meta.get("regularMarketPrice", 0)), 2)
            if g.prev_close <= 0:
                g.prev_close = round(float(
                    meta.get("chartPreviousClose", 0)
                    or meta.get("previousClose", 0)
                ), 2)
            quotes = result.get("indicators", {}).get("quote", [{}])[0]
            volumes = quotes.get("volume", [])
            if volumes:
                latest_vol = [v for v in volumes if v is not None]
                if latest_vol:
                    g.volume = int(latest_vol[-1])
        except (KeyError, IndexError, TypeError):
            pass
        time.sleep(0.1)


def scan_gainers(
    min_change_pct: float = 2.0,
    min_price: float = 1.0,
    max_results: int = 25,
) -> list[Gainer]:
    log.info("Scanning for top gainers (min %.1f%%, min price $%.2f)...",
             min_change_pct, min_price)

    log.info("Fetching gainers from Yahoo Finance...")
    yahoo = _fetch_yahoo_gainers(count=50)

    log.info("Fetching gainers from Finviz...")
    finviz = _fetch_finviz_gainers()

    merged = _merge_gainers(yahoo, finviz)
    log.info("Merged: %d unique gainers from all sources", len(merged))

    filtered = [
        g for g in merged
        if g.change_pct >= min_change_pct and g.price >= min_price
    ]
    log.info("After filtering: %d gainers", len(filtered))

    _enrich_missing_data(filtered)
    _compute_volume_ratios(filtered)
    ranked = _rank_gainers(filtered)

    result = ranked[:max_results]
    log.info("Returning top %d gainers", len(result))
    return result


def _format_market_cap(cap: float) -> str:
    if cap <= 0:
        return "    N/A"
    if cap >= 1e12:
        return f"{cap / 1e12:>6.1f}T"
    if cap >= 1e9:
        return f"{cap / 1e9:>6.1f}B"
    if cap >= 1e6:
        return f"{cap / 1e6:>6.0f}M"
    return f"    N/A"


def _format_volume(vol: int) -> str:
    if vol <= 0:
        return "    N/A"
    if vol >= 1e6:
        return f"{vol / 1e6:>6.1f}M"
    if vol >= 1e3:
        return f"{vol / 1e3:>6.0f}K"
    return f"{vol:>7d}"


def format_gainers_text(gainers: list[Gainer]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "STOCK MONITOR -- Top Gainers",
        now,
        "",
    ]

    if not gainers:
        lines.append("No significant gainers found at this time.")
    else:
        lines.append(f"TOP {len(gainers)} GAINERS TODAY")
        lines.append("")
        lines.append(
            f"  {'#':<3} {'Ticker':<7} {'Price':>9} {'Change':>8} "
            f"{'Volume':>8} {'Vol.Ratio':>9} {'Mkt Cap':>8} {'Sector'}"
        )
        lines.append("  " + "-" * 80)

        for i, g in enumerate(gainers, 1):
            vol_ratio_str = f"{g.volume_ratio:>8.1f}x" if g.volume_ratio > 0 else "     N/A"
            lines.append(
                f"  {i:<3} {g.ticker:<7} ${g.price:>8.2f} {g.change_pct:>+7.2f}% "
                f"{_format_volume(g.volume)} {vol_ratio_str} "
                f"{_format_market_cap(g.market_cap)} {g.sector}"
            )
            if g.company and g.company != g.ticker:
                lines.append(f"      {g.company}")

        lines.append("")

        high_vol = [g for g in gainers if g.volume_ratio >= 2.0]
        if high_vol:
            lines.append(
                f"  Volume alerts: {len(high_vol)} gainer(s) trading at 2x+ avg volume"
            )

        large_cap = [g for g in gainers if g.market_cap >= 10e9]
        if large_cap:
            tickers = ", ".join(g.ticker for g in large_cap[:5])
            lines.append(f"  Large cap movers: {tickers}")

        lines.append("")

    lines.append("-" * 72)
    lines.append("Engine: Top Gainers Detector | Sources: Yahoo Finance, Finviz")
    lines.append("Not financial advice. Do your own research.")
    return "\n".join(lines)


def format_gainers_json(gainers: list[Gainer]) -> str:
    return json.dumps([g.to_dict() for g in gainers], indent=2)
