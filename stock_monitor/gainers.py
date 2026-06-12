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
class IntradayResult:
    signal: str = "HOLD"
    stop_loss: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    rsi: float = 0.0
    vwap: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    atr: float = 0.0
    trend: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": "intraday (1h)",
            "signal": self.signal,
            "stop_loss": self.stop_loss,
            "support": self.support,
            "resistance": self.resistance,
            "rsi": self.rsi,
            "vwap": self.vwap,
            "ema_9": self.ema_9,
            "ema_21": self.ema_21,
            "atr": self.atr,
            "trend": self.trend,
            "reasons": self.reasons,
        }


@dataclass
class DailyResult:
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

    def to_dict(self) -> dict:
        return {
            "label": "daily (1d)",
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
        }


@dataclass
class GainerAnalysis:
    intraday: Optional[IntradayResult] = None
    daily: Optional[DailyResult] = None
    risks: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"risks": self.risks, "error": self.error}
        if self.intraday:
            d["intraday"] = self.intraday.to_dict()
        if self.daily:
            d["daily"] = self.daily.to_dict()
        return d


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


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _compute_ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return round(ema, 2)


def _intraday_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(highs) < 2:
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0.0
    return sum(trs[-period:]) / period


def _analyze_intraday(ticker: str, current_price: float) -> Optional[IntradayResult]:
    data = fetch_chart(ticker, range_="5d", interval="1h", timeout=12)
    if data is None:
        return None
    try:
        result = data["chart"]["result"][0]
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        closes_raw = quotes.get("close", [])
        highs_raw = quotes.get("high", [])
        lows_raw = quotes.get("low", [])
        volumes_raw = quotes.get("volume", [])

        closes = [float(c) for c in closes_raw if c is not None]
        highs = [float(h) for h in highs_raw if h is not None]
        lows = [float(lo) for lo in lows_raw if lo is not None]
        volumes = [int(v) for v in volumes_raw if v is not None]

        if len(closes) < 10:
            return None

        rsi = _compute_rsi(closes)
        ema_9 = _compute_ema(closes, 9)
        ema_21 = _compute_ema(closes, 21)
        atr = _intraday_atr(highs, lows, closes)

        recent_lows = lows[-20:] if len(lows) >= 20 else lows
        recent_highs = highs[-20:] if len(highs) >= 20 else highs
        support = round(min(recent_lows), 2)
        resistance = round(max(recent_highs), 2)

        vwap = 0.0
        if volumes and len(closes) == len(volumes):
            total_vol = sum(volumes[-30:])
            if total_vol > 0:
                typical_prices = [
                    (highs[i] + lows[i] + closes[i]) / 3
                    for i in range(max(0, len(closes) - 30), len(closes))
                ]
                vp_sum = sum(
                    tp * v
                    for tp, v in zip(typical_prices, volumes[-30:])
                )
                vwap = round(vp_sum / total_vol, 2)

        trend = ""
        reasons: list[str] = []
        if ema_9 > ema_21:
            trend = "BULLISH"
            reasons.append(f"EMA(9) ${ema_9:.2f} above EMA(21) ${ema_21:.2f}")
        elif ema_9 < ema_21:
            trend = "BEARISH"
            reasons.append(f"EMA(9) ${ema_9:.2f} below EMA(21) ${ema_21:.2f}")
        else:
            trend = "NEUTRAL"

        if vwap > 0 and current_price > vwap:
            reasons.append(f"Price above VWAP ${vwap:.2f} -- intraday bullish")
        elif vwap > 0:
            reasons.append(f"Price below VWAP ${vwap:.2f} -- intraday bearish")

        if rsi > 80:
            reasons.append(f"1h RSI {rsi:.0f} -- severely overbought, pullback risk")
        elif rsi > 70:
            reasons.append(f"1h RSI {rsi:.0f} -- overbought territory")
        elif rsi < 30:
            reasons.append(f"1h RSI {rsi:.0f} -- oversold, possible bounce")

        signal = "HOLD"
        if trend == "BULLISH" and rsi < 75 and (vwap == 0 or current_price >= vwap):
            if rsi < 50:
                signal = "STRONG BUY"
            else:
                signal = "BUY"
        elif trend == "BULLISH" and rsi >= 75:
            signal = "HOLD"
            reasons.append("Bullish trend but overbought -- wait for pullback")
        elif trend == "BEARISH" and rsi > 25:
            if rsi > 50:
                signal = "SELL"
            else:
                signal = "LEAN SELL"
        elif trend == "BEARISH" and rsi <= 25:
            signal = "HOLD"
            reasons.append("Bearish but oversold -- bounce possible")
        elif trend == "BULLISH":
            signal = "LEAN BUY"

        stop_loss = 0.0
        if signal in ("STRONG BUY", "BUY", "LEAN BUY") and atr > 0:
            stop_loss = round(max(current_price - atr * 1.0, support * 0.995), 2)
        elif signal in ("SELL", "LEAN SELL") and atr > 0:
            stop_loss = round(min(current_price + atr * 1.0, resistance * 1.005), 2)

        return IntradayResult(
            signal=signal,
            stop_loss=stop_loss,
            support=support,
            resistance=resistance,
            rsi=rsi,
            vwap=vwap,
            ema_9=ema_9,
            ema_21=ema_21,
            atr=round(atr, 2),
            trend=trend,
            reasons=reasons,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        log.warning("Intraday analysis parse failed for %s: %s", ticker, exc)
        return None


def _analyze_daily(
    ticker: str,
    market_df,
    vix_df,
    use_lstm: bool,
) -> Optional[DailyResult]:
    from .analyzer import analyze
    from .config import TIMEFRAME_1D

    try:
        result = analyze(
            ticker,
            force_retrain=False,
            timeframe=TIMEFRAME_1D,
            market="US",
            currency="$",
            market_df=market_df,
            vix_df=vix_df,
            use_lstm=use_lstm,
        )
        if result.error:
            return None

        return DailyResult(
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
        )
    except Exception as exc:
        log.warning("Daily ML analysis failed for %s: %s", ticker, exc)
        return None


def _assess_risks(g: Gainer, ga: GainerAnalysis) -> list[str]:
    risks = []

    if g.change_pct > 20:
        risks.append(f"Extreme rally (+{g.change_pct:.0f}%) -- high reversal risk")
    elif g.change_pct > 10:
        risks.append(f"Sharp rally (+{g.change_pct:.0f}%) -- pullback likely")

    intra = ga.intraday
    daily = ga.daily

    intra_rsi = intra.rsi if intra else 0
    daily_rsi = daily.rsi if daily else 0
    peak_rsi = max(intra_rsi, daily_rsi)

    if peak_rsi > 80:
        risks.append(f"RSI at {peak_rsi:.0f} -- severely overbought")
    elif peak_rsi > 70:
        risks.append(f"RSI at {peak_rsi:.0f} -- overbought territory")

    if g.volume_ratio < 1.0 and g.volume_ratio > 0:
        risks.append("Below-average volume -- weak conviction behind move")

    if g.market_cap > 0 and g.market_cap < 100e6:
        risks.append("Micro-cap stock -- high volatility and liquidity risk")
    elif g.market_cap > 0 and g.market_cap < 500e6:
        risks.append("Small-cap -- higher volatility than large caps")

    if daily:
        if daily.prob_down > 0.35:
            risks.append(f"Model sees {daily.prob_down:.0%} chance of decline (daily)")
        if daily.confidence < 0.4 and daily.confidence > 0:
            risks.append(f"Low ML confidence ({daily.confidence:.0%})")
        if g.price > 0 and daily.sma_50 > 0 and g.price > daily.sma_50 * 1.15:
            pct_above = (g.price / daily.sma_50 - 1) * 100
            risks.append(f"Extended {pct_above:.0f}% above SMA(50) -- stretched")

    if intra and daily:
        intra_bull = intra.signal in ("STRONG BUY", "BUY", "LEAN BUY")
        daily_bull = daily.signal in ("STRONG BUY", "BUY", "LEAN BUY")
        intra_bear = intra.signal in ("SELL", "LEAN SELL", "STRONG SELL")
        daily_bear = daily.signal in ("SELL", "LEAN SELL", "STRONG SELL")
        if intra_bull and daily_bear:
            risks.append("Timeframe conflict: intraday bullish but daily bearish")
        elif intra_bear and daily_bull:
            risks.append("Timeframe conflict: intraday bearish but daily bullish")

    if not risks:
        risks.append("No major risk flags detected")

    return risks


def analyze_gainers(
    gainers: list[Gainer],
    use_lstm: bool = True,
    max_workers: int = 4,
) -> None:
    from .market_data import get_market_context

    market_df = None
    vix_df = None
    try:
        market_df, vix_df = get_market_context()
    except Exception as exc:
        log.warning("Failed to fetch market context: %s", exc)

    def _analyze_one(g: Gainer) -> None:
        try:
            intra = _analyze_intraday(g.ticker, g.price)
            daily = _analyze_daily(g.ticker, market_df, vix_df, use_lstm)

            ga = GainerAnalysis(intraday=intra, daily=daily)
            ga.risks = _assess_risks(g, ga)
            g.analysis = ga
        except Exception as exc:
            log.warning("Analysis failed for %s: %s", g.ticker, exc)
            g.analysis = GainerAnalysis(error=str(exc))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.info("Running dual-timeframe analysis on %d gainers with %d workers...", len(gainers), max_workers)
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


def _fmt_intraday_block(lines: list[str], price: float, intra: IntradayResult) -> None:
    lines.append(f"  INTRADAY (1h bars)    Trend: {intra.trend}    Signal: {_signal_icon(intra.signal)} {intra.signal}")
    parts = []
    if intra.vwap > 0:
        parts.append(f"VWAP ${intra.vwap:.2f}")
    parts.append(f"EMA(9) ${intra.ema_9:.2f}")
    parts.append(f"EMA(21) ${intra.ema_21:.2f}")
    parts.append(f"RSI {intra.rsi:.0f}")
    if intra.atr > 0:
        parts.append(f"ATR ${intra.atr:.2f}")
    lines.append(f"    {' | '.join(parts)}")
    lines.append(f"    Support: ${intra.support:.2f}    Resistance: ${intra.resistance:.2f}")

    if intra.signal in ("STRONG BUY", "BUY", "LEAN BUY"):
        lines.append(f"    >> BUY now at ${price:.2f}")
        if intra.stop_loss > 0:
            sl_pct = abs(intra.stop_loss - price) / price * 100
            lines.append(f"    >> Stop loss: ${intra.stop_loss:.2f} ({sl_pct:.1f}% below)")
        if intra.resistance > 0 and intra.resistance > price:
            tp_pct = (intra.resistance - price) / price * 100
            lines.append(f"    >> Intraday target: ${intra.resistance:.2f} ({tp_pct:+.1f}%)")
    elif intra.signal in ("SELL", "LEAN SELL", "STRONG SELL"):
        lines.append(f"    >> AVOID -- intraday trend bearish")
        if intra.stop_loss > 0:
            sl_pct = abs(intra.stop_loss - price) / price * 100
            lines.append(f"    >> Stop loss (if short): ${intra.stop_loss:.2f} ({sl_pct:.1f}% above)")
    else:
        lines.append(f"    >> WAIT -- no clear intraday edge")

    for r in intra.reasons:
        lines.append(f"    - {r}")


def _fmt_daily_block(lines: list[str], price: float, daily: DailyResult) -> None:
    lines.append(
        f"  DAILY (1d ML prediction)    Signal: {_signal_icon(daily.signal)} {daily.signal}"
        f"    Score: {daily.score:+d}/100    Confidence: {daily.confidence:.0%}"
    )
    lines.append(
        f"    Predicted return: {daily.predicted_return_pct:+.2f}%"
        f"    Probabilities: UP={daily.prob_up:.0%} FLAT={daily.prob_flat:.0%} DOWN={daily.prob_down:.0%}"
    )
    lines.append(
        f"    SMA(20): ${daily.sma_20:.2f}    SMA(50): ${daily.sma_50:.2f}    RSI: {daily.rsi:.0f}"
    )
    lines.append(f"    Support: ${daily.support:.2f}    Resistance: ${daily.resistance:.2f}")

    if daily.signal in ("STRONG BUY", "BUY", "LEAN BUY"):
        lines.append(f"    >> BUY for daily hold at ${price:.2f}")
        if daily.stop_loss > 0:
            sl_pct = abs(daily.stop_loss - price) / price * 100
            lines.append(f"    >> Stop loss: ${daily.stop_loss:.2f} ({sl_pct:.1f}% below)")
        if daily.resistance > 0 and daily.resistance > price:
            tp_pct = (daily.resistance - price) / price * 100
            lines.append(f"    >> Daily target: ${daily.resistance:.2f} ({tp_pct:+.1f}%)")
    elif daily.signal in ("SELL", "LEAN SELL", "STRONG SELL"):
        lines.append(f"    >> AVOID -- daily model bearish")
        if daily.stop_loss > 0:
            sl_pct = abs(daily.stop_loss - price) / price * 100
            lines.append(f"    >> Stop loss (if short): ${daily.stop_loss:.2f} ({sl_pct:.1f}% above)")
    else:
        lines.append(f"    >> HOLD -- daily model sees no clear edge")


def format_gainers_text(gainers: list[Gainer]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "STOCK MONITOR -- Top Gainers (Intraday + Daily)",
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
                if a.intraday:
                    lines.append("")
                    _fmt_intraday_block(lines, g.price, a.intraday)

                if a.daily:
                    lines.append("")
                    _fmt_daily_block(lines, g.price, a.daily)

                if not a.intraday and not a.daily:
                    lines.append("  Analysis: both timeframes unavailable")

                if a.risks:
                    lines.append("")
                    lines.append("  RISKS:")
                    for risk in a.risks:
                        lines.append(f"    ! {risk}")

            elif a and a.error:
                lines.append(f"  Analysis: unavailable ({a.error})")
            else:
                lines.append("  Analysis: not run")

            lines.append("")

    lines.append("=" * 72)
    lines.append("Engine: Top Gainers Detector + ML Ensemble")
    lines.append("Timeframes: Intraday (1h technicals) + Daily (GBM + LSTM)")
    lines.append("Sources: Yahoo Finance, Finviz")
    lines.append("Not financial advice. Do your own research.")
    return "\n".join(lines)


def format_gainers_json(gainers: list[Gainer]) -> str:
    return json.dumps([g.to_dict() for g in gainers], indent=2)
