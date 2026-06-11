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
class GainerAnalysis:
    signal: str = "HOLD"
    score: int = 0
    predicted_return_pct: float = 0.0
    confidence: float = 0.0
    stop_loss: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    rsi: float = 0.0
    prob_up: float = 0.0
    prob_down: float = 0.0
    prob_flat: float = 0.0
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "score": self.score,
            "predicted_return_pct": self.predicted_return_pct,
            "confidence": self.confidence,
            "stop_loss": self.stop_loss,
            "support": self.support,
            "resistance": self.resistance,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "rsi": self.rsi,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "prob_flat": self.prob_flat,
            "reasons": self.reasons,
            "risks": self.risks,
            "error": self.error,
        }


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
    analysis: Optional[GainerAnalysis] = None

    def to_dict(self) -> dict:
        d = {
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
        if self.analysis:
            d["analysis"] = self.analysis.to_dict()
        return d


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


def _assess_risks(g: Gainer, a: GainerAnalysis) -> list[str]:
    risks = []

    if g.change_pct > 20:
        risks.append(f"Extreme rally (+{g.change_pct:.0f}%) -- high reversal risk")
    elif g.change_pct > 10:
        risks.append(f"Sharp rally (+{g.change_pct:.0f}%) -- pullback likely")

    if a.rsi > 80:
        risks.append(f"RSI at {a.rsi:.0f} -- severely overbought")
    elif a.rsi > 70:
        risks.append(f"RSI at {a.rsi:.0f} -- overbought territory")

    if g.volume_ratio < 1.0 and g.volume_ratio > 0:
        risks.append("Below-average volume -- weak conviction behind move")

    if g.market_cap > 0 and g.market_cap < 100e6:
        risks.append("Micro-cap stock -- high volatility and liquidity risk")
    elif g.market_cap > 0 and g.market_cap < 500e6:
        risks.append("Small-cap -- higher volatility than large caps")

    if a.prob_down > 0.35:
        risks.append(f"Model sees {a.prob_down:.0%} chance of decline")

    if a.confidence < 0.4 and a.confidence > 0:
        risks.append(f"Low model confidence ({a.confidence:.0%})")

    if g.price > 0 and a.sma_50 > 0 and g.price > a.sma_50 * 1.15:
        pct_above = (g.price / a.sma_50 - 1) * 100
        risks.append(f"Extended {pct_above:.0f}% above SMA(50) -- stretched")

    if not risks:
        risks.append("No major risk flags detected")

    return risks


def analyze_gainers(
    gainers: list[Gainer],
    use_lstm: bool = True,
    max_workers: int = 4,
) -> None:
    from .analyzer import analyze
    from .market_data import get_market_context
    from .config import TIMEFRAME_5D

    market_df = None
    vix_df = None
    try:
        market_df, vix_df = get_market_context()
    except Exception as exc:
        log.warning("Failed to fetch market context: %s", exc)

    def _analyze_one(g: Gainer) -> None:
        try:
            result = analyze(
                g.ticker,
                force_retrain=False,
                timeframe=TIMEFRAME_5D,
                market="US",
                currency="$",
                market_df=market_df,
                vix_df=vix_df,
                use_lstm=use_lstm,
            )

            ga = GainerAnalysis(
                signal=result.signal.value,
                score=result.score,
                predicted_return_pct=result.predicted_return_pct,
                confidence=result.confidence,
                stop_loss=result.stop_loss,
                support=result.support,
                resistance=result.resistance,
                sma_20=result.sma_20,
                sma_50=result.sma_50,
                rsi=result.rsi,
                prob_up=result.prob_up,
                prob_down=result.prob_down,
                prob_flat=result.prob_flat,
                reasons=result.reasons,
                error=result.error,
            )
            ga.risks = _assess_risks(g, ga)
            g.analysis = ga
        except Exception as exc:
            log.warning("Analysis failed for %s: %s", g.ticker, exc)
            g.analysis = GainerAnalysis(error=str(exc))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.info("Running ML analysis on %d gainers with %d workers...", len(gainers), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_one, g): g for g in gainers}
        done = 0
        for future in as_completed(futures):
            done += 1
            g = futures[future]
            try:
                future.result()
                log.info("Analyzed %s (%d/%d)", g.ticker, done, len(gainers))
            except Exception as exc:
                log.error("Failed %s: %s", g.ticker, exc)


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


def _signal_icon(signal: str) -> str:
    icons = {
        "STRONG BUY": "[++]", "BUY": "[+ ]", "LEAN BUY": "[ +]",
        "HOLD": "[ = ]",
        "LEAN SELL": "[ -]", "SELL": "[- ]", "STRONG SELL": "[--]",
    }
    return icons.get(signal, "[ ? ]")


def format_gainers_text(gainers: list[Gainer]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "STOCK MONITOR -- Top Gainers with Analysis",
        now,
        "",
    ]

    if not gainers:
        lines.append("No significant gainers found at this time.")
    else:
        for i, g in enumerate(gainers, 1):
            a = g.analysis
            name = g.company if g.company and g.company != g.ticker else ""

            lines.append("=" * 72)
            header = f"  #{i}  {g.ticker}"
            if name:
                header += f"  --  {name}"
            if g.sector:
                header += f"  [{g.sector}]"
            lines.append(header)
            lines.append("-" * 72)

            vol_ratio_str = f"{g.volume_ratio:.1f}x avg" if g.volume_ratio > 0 else "N/A"
            lines.append(
                f"  Price: ${g.price:.2f}  ({g.change_pct:+.2f}% today)"
                f"    Volume: {_format_volume(g.volume).strip()} ({vol_ratio_str})"
                f"    Mkt Cap:{_format_market_cap(g.market_cap)}"
            )

            if a and not a.error:
                lines.append("")
                lines.append(
                    f"  Signal: {_signal_icon(a.signal)} {a.signal}"
                    f"    Score: {a.score:+d}/100"
                    f"    Confidence: {a.confidence:.0%}"
                )
                lines.append(
                    f"  Predicted return (5d): {a.predicted_return_pct:+.2f}%"
                    f"    Probabilities: UP={a.prob_up:.0%} FLAT={a.prob_flat:.0%} DOWN={a.prob_down:.0%}"
                )

                lines.append("")
                if a.signal in ("STRONG BUY", "BUY", "LEAN BUY"):
                    lines.append(f"  >> BUY at ${g.price:.2f}")
                    if a.stop_loss > 0:
                        sl_pct = abs(a.stop_loss - g.price) / g.price * 100
                        lines.append(f"  >> Stop loss: ${a.stop_loss:.2f} ({sl_pct:.1f}% below entry)")
                    if a.resistance > 0:
                        target_pct = (a.resistance - g.price) / g.price * 100
                        lines.append(f"  >> Target (resistance): ${a.resistance:.2f} ({target_pct:+.1f}%)")
                elif a.signal in ("STRONG SELL", "SELL", "LEAN SELL"):
                    lines.append(f"  >> AVOID / SELL at ${g.price:.2f}")
                    if a.stop_loss > 0:
                        sl_pct = abs(a.stop_loss - g.price) / g.price * 100
                        lines.append(f"  >> Stop loss: ${a.stop_loss:.2f} ({sl_pct:.1f}% above entry)")
                    if a.support > 0:
                        target_pct = (a.support - g.price) / g.price * 100
                        lines.append(f"  >> Target (support): ${a.support:.2f} ({target_pct:+.1f}%)")
                else:
                    lines.append(f"  >> HOLD / WAIT -- no clear edge")

                lines.append("")
                lines.append(
                    f"  Support: ${a.support:.2f}    Resistance: ${a.resistance:.2f}"
                )
                lines.append(
                    f"  SMA(20): ${a.sma_20:.2f}    SMA(50): ${a.sma_50:.2f}    RSI: {a.rsi:.0f}"
                )

                if a.risks:
                    lines.append("")
                    lines.append("  Risks:")
                    for risk in a.risks:
                        lines.append(f"    ! {risk}")

            elif a and a.error:
                lines.append(f"  Analysis: unavailable ({a.error})")
            else:
                lines.append("  Analysis: not run")

            lines.append("")

    lines.append("=" * 72)
    lines.append("Engine: Top Gainers Detector + ML Ensemble")
    lines.append("Sources: Yahoo Finance, Finviz | Model: GBM + LSTM")
    lines.append("Not financial advice. Do your own research.")
    return "\n".join(lines)


def format_gainers_json(gainers: list[Gainer]) -> str:
    return json.dumps([g.to_dict() for g in gainers], indent=2)
