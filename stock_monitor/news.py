from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests as _requests

from .config import MODELS_DIR

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
_TIMEOUT = 15
_TICKER_CACHE_PATH = MODELS_DIR / "ticker_list.json"
_TICKER_CACHE_MAX_AGE = 7 * 86400

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    tickers: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class NewsMover:
    ticker: str
    predicted_gain_pct: float
    news_score: float
    headline_count: int
    current_price: float = 0.0
    change_pct: float = 0.0
    top_headlines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "predicted_gain_pct": self.predicted_gain_pct,
            "news_score": self.news_score,
            "headline_count": self.headline_count,
            "current_price": self.current_price,
            "change_pct": self.change_pct,
            "top_headlines": self.top_headlines,
        }


# ---------------------------------------------------------------------------
# Fallback company name -> ticker mapping (used when NASDAQ download fails)
# ---------------------------------------------------------------------------

_FALLBACK_TICKERS: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "broadcom": "AVGO", "costco": "COST", "pepsi": "PEP", "pepsico": "PEP",
    "adobe": "ADBE", "salesforce": "CRM", "walmart": "WMT", "disney": "DIS",
    "coca-cola": "KO", "coca cola": "KO", "johnson & johnson": "JNJ",
    "jpmorgan": "JPM", "jp morgan": "JPM", "visa": "V", "mastercard": "MA",
    "procter & gamble": "PG", "procter and gamble": "PG",
    "unitedhealth": "UNH", "home depot": "HD", "chevron": "CVX",
    "exxon": "XOM", "exxonmobil": "XOM", "abbvie": "ABBV",
    "pfizer": "PFE", "merck": "MRK", "lilly": "LLY", "eli lilly": "LLY",
    "boeing": "BA", "caterpillar": "CAT", "goldman sachs": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "citigroup": "C",
    "wells fargo": "WFC", "uber": "UBER", "airbnb": "ABNB",
    "palantir": "PLTR", "snowflake": "SNOW", "crowdstrike": "CRWD",
    "datadog": "DDOG", "shopify": "SHOP", "spotify": "SPOT",
    "paypal": "PYPL", "square": "SQ", "block": "SQ",
    "coinbase": "COIN", "robinhood": "HOOD", "sofi": "SOFI",
    "rivian": "RIVN", "lucid": "LCID", "nio": "NIO",
    "moderna": "MRNA", "biontech": "BNTX", "regeneron": "REGN",
    "gilead": "GILD", "amgen": "AMGN", "biogen": "BIIB",
    "qualcomm": "QCOM", "micron": "MU", "applied materials": "AMAT",
    "lam research": "LRCX", "asml": "ASML", "arm": "ARM",
    "oracle": "ORCL", "ibm": "IBM", "cisco": "CSCO",
    "dell": "DELL", "hp": "HPQ", "servicenow": "NOW",
    "workday": "WDAY", "zoom": "ZM", "docusign": "DOCU",
    "snap": "SNAP", "pinterest": "PINS", "roblox": "RBLX",
    "unity": "U", "ea": "EA", "electronic arts": "EA",
    "activision": "ATVI", "take-two": "TTWO",
    "target": "TGT", "kroger": "KR", "starbucks": "SBUX",
    "mcdonald": "MCD", "mcdonalds": "MCD", "mcdonald's": "MCD",
    "nike": "NKE", "lululemon": "LULU",
    "general motors": "GM", "ford": "F",
    "lockheed martin": "LMT", "raytheon": "RTX",
    "3m": "MMM", "honeywell": "HON", "general electric": "GE",
    "t-mobile": "TMUS", "at&t": "T", "verizon": "VZ",
    "comcast": "CMCSA", "charter": "CHTR",
    "carvana": "CVNA", "soundhound": "SOUN", "super micro": "SMCI",
    "supermicro": "SMCI", "c3.ai": "AI", "bigbear.ai": "BBAI",
    "symbotic": "SYM", "joby": "JOBY", "lilium": "LILM",
    "rocket lab": "RKLB", "virgin galactic": "SPCE",
    "draftkings": "DKNG", "celsius": "CELH", "duolingo": "DUOL",
}

_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")

_NAME_STRIP_RE = re.compile(
    r",?\s*\b(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company|Holdings?"
    r"|Group|Plc\.?|N\.?V\.?|S\.?A\.?|SE|AG|Class\s+[A-Z]|Common\s+Stock"
    r"|Ordinary\s+Shares?|Depositary\s+Shares?|American\s+Depositary"
    r"|Warrant[s]?|Right[s]?|Unit[s]?)\b\.?",
    re.IGNORECASE,
)

# Short ticker symbols that are also common English words -- require $-prefix
_AMBIGUOUS_TICKERS: set[str] = {
    "A", "AN", "ALL", "AM", "ARE", "AT", "BE", "BIG", "CAN",
    "CAR", "DAY", "DD", "DO", "E", "F", "FOR", "GO", "HAS",
    "HE", "IT", "K", "LOW", "MAN", "MAY", "MO", "NOW", "ON",
    "ONE", "OUT", "PAY", "RUN", "SO", "SUN", "T", "THE", "TOO",
    "TWO", "U", "V", "W", "X", "Y", "Z",
}


# ---------------------------------------------------------------------------
# Dynamic ticker list from NASDAQ
# ---------------------------------------------------------------------------

def _load_ticker_list() -> dict[str, str]:
    """Download or load cached full US stock listing from NASDAQ API.

    Returns a raw mapping of {ticker: company_name}.
    Cached to disk for 7 days.
    """
    if _TICKER_CACHE_PATH.exists():
        age = time.time() - _TICKER_CACHE_PATH.stat().st_mtime
        if age < _TICKER_CACHE_MAX_AGE:
            try:
                data = json.loads(_TICKER_CACHE_PATH.read_text())
                if data:
                    log.info("Loaded %d tickers from cache", len(data))
                    return data
            except (json.JSONDecodeError, OSError):
                pass

    log.info("Downloading full US stock listing from NASDAQ...")
    raw: dict[str, str] = {}

    for exchange in ("nasdaq", "nyse", "amex"):
        url = (
            f"https://api.nasdaq.com/api/screener/stocks"
            f"?tableonly=true&limit=10000&exchange={exchange}"
        )
        try:
            resp = _requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", {}).get("table", {}).get("rows", [])
            for row in rows:
                sym = (row.get("symbol") or "").strip().upper()
                name = (row.get("name") or "").strip()
                if sym and name and "/" not in sym and "^" not in sym:
                    raw[sym] = name
            log.info("  %s: %d tickers", exchange.upper(), len(rows))
        except Exception as exc:
            log.warning("Failed to fetch %s listing: %s", exchange, exc)
        time.sleep(0.3)

    if raw:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _TICKER_CACHE_PATH.write_text(json.dumps(raw))
        log.info("Cached %d tickers to %s", len(raw), _TICKER_CACHE_PATH)
    else:
        log.warning("NASDAQ download returned 0 tickers, using fallback only")

    return raw


def _build_ticker_lookup(raw_list: dict[str, str]) -> dict[str, str]:
    """Build a name->ticker lookup dict from the NASDAQ download.

    Produces entries like:
        "carvana" -> "CVNA"
        "advanced micro devices" -> "AMD"
        "amd" -> "AMD"
    """
    lookup: dict[str, str] = {}

    for ticker, full_name in raw_list.items():
        cleaned = _NAME_STRIP_RE.sub("", full_name).strip().rstrip(",").strip()
        lower = cleaned.lower()
        if lower:
            lookup[lower] = ticker

        parts = lower.split()
        if len(parts) > 1:
            first_word = parts[0]
            if len(first_word) >= 4:
                lookup[first_word] = ticker

        ticker_lower = ticker.lower()
        if ticker_lower not in lookup:
            lookup[ticker_lower] = ticker

    for name, ticker in _FALLBACK_TICKERS.items():
        if name not in lookup:
            lookup[name] = ticker

    log.info("Built ticker lookup with %d entries", len(lookup))
    return lookup


def _compile_patterns(
    lookup: dict[str, str],
) -> list[tuple[re.Pattern, str]]:
    """Compile regex patterns for all lookup entries.

    Multi-word names get word-boundary matching.
    Single short names that could be ambiguous are skipped (handled by $TICKER regex).
    """
    patterns: list[tuple[re.Pattern, str]] = []
    seen_tickers: dict[str, str] = {}

    sorted_names = sorted(lookup.keys(), key=len, reverse=True)

    for name in sorted_names:
        ticker = lookup[name]
        if ticker in seen_tickers:
            existing = seen_tickers[ticker]
            if len(name) <= len(existing):
                continue

        words = name.split()
        if len(words) == 1 and len(name) <= 3:
            continue
        if name.upper() in _AMBIGUOUS_TICKERS:
            continue

        try:
            pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            patterns.append((pat, ticker))
            seen_tickers[ticker] = name
        except re.error:
            continue

    return patterns


# Module-level state, populated lazily in scan_news()
_ticker_lookup: dict[str, str] = {}
_ticker_patterns: list[tuple[re.Pattern, str]] = []
_all_valid_tickers: set[str] = set()


def _ensure_ticker_data() -> None:
    """Load ticker list and compile patterns (once per process)."""
    global _ticker_lookup, _ticker_patterns, _all_valid_tickers
    if _ticker_patterns:
        return
    raw = _load_ticker_list()
    _ticker_lookup = _build_ticker_lookup(raw)
    _ticker_patterns = _compile_patterns(_ticker_lookup)
    _all_valid_tickers = set(raw.keys()) | set(_FALLBACK_TICKERS.values())


# ---------------------------------------------------------------------------
# Ticker extraction (dynamic)
# ---------------------------------------------------------------------------

def _extract_tickers(text: str) -> list[str]:
    tickers: set[str] = set()

    for m in _TICKER_RE.finditer(text):
        sym = m.group(1)
        if sym in _all_valid_tickers:
            tickers.add(sym)

    for pattern, ticker in _ticker_patterns:
        if pattern.search(text):
            tickers.add(ticker)

    return sorted(tickers)


# ---------------------------------------------------------------------------
# Keyword scoring engine
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: dict[str, float] = {
    "fda approval": 10, "fda approved": 10, "fda clears": 10,
    "beat earnings": 9, "earnings beat": 9, "tops estimates": 9,
    "beats estimates": 9, "beats expectations": 9, "blowout earnings": 9,
    "record revenue": 8, "record profit": 8, "record quarter": 8,
    "all-time high": 8, "all time high": 8,
    "breakthrough": 8, "major contract": 8, "wins contract": 8,
    "acquisition": 7, "acquires": 7, "merger": 7, "takeover": 7, "buyout": 7,
    "upgrade": 6, "upgraded": 6, "price target raised": 6,
    "outperform": 6, "overweight": 6,
    "strong guidance": 6, "raises outlook": 6, "raises guidance": 6,
    "beats revenue": 6, "revenue beat": 6,
    "partnership": 5, "new deal": 5, "expands": 5, "expansion": 5,
    "surge": 5, "soars": 5, "jumps": 5, "rallies": 5, "rockets": 5,
    "skyrockets": 5, "spikes": 5,
    "buyback": 4, "share repurchase": 4, "dividend increase": 4,
    "dividend hike": 4, "special dividend": 5,
    "stock split": 4, "positive results": 4, "strong demand": 4,
    "new product": 4, "launch": 3, "innovation": 3,
    "bullish": 4, "buy rating": 5, "strong buy": 5,
    "top pick": 5, "best idea": 5,
    "ai": 3, "artificial intelligence": 3,
}

NEGATIVE_KEYWORDS: dict[str, float] = {
    "downgrade": -6, "downgraded": -6, "sell rating": -6,
    "miss": -5, "misses estimates": -7, "earnings miss": -7,
    "misses expectations": -7, "disappointing": -5,
    "layoffs": -5, "job cuts": -5, "restructuring": -4,
    "recall": -5, "safety concern": -5,
    "lawsuit": -4, "sued": -4, "investigation": -5, "sec investigation": -7,
    "fraud": -8, "scandal": -7,
    "bankruptcy": -9, "default": -7, "debt crisis": -7,
    "warning": -4, "profit warning": -6, "guidance cut": -6,
    "lowers outlook": -6, "lowers guidance": -6, "weak guidance": -5,
    "decline": -3, "drops": -3, "falls": -3, "plunges": -5, "crashes": -6,
    "bear": -3, "bearish": -4, "underperform": -5, "underweight": -5,
    "price target cut": -5, "price target lowered": -5,
}

SCORE_TO_GAIN: list[tuple[float, float]] = [
    (20, 12.0),
    (16, 10.0),
    (13, 8.0),
    (10, 6.0),
    (8, 5.0),
]

MIN_SCORE_THRESHOLD = 8.0


def _score_text(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for keyword, weight in POSITIVE_KEYWORDS.items():
        if keyword in lower:
            score += weight
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if keyword in lower:
            score += weight
    return score


def _predicted_gain(aggregate_score: float) -> float:
    for threshold, gain in SCORE_TO_GAIN:
        if aggregate_score >= threshold:
            return gain
    return 0.0


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.headline.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Fetchers -- RSS
# ---------------------------------------------------------------------------

def _fetch_rss(url: str, source_name: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        resp = _requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for item_el in root.iter("item"):
            title_el = item_el.find("title")
            link_el = item_el.find("link")
            desc_el = item_el.find("description")
            if title_el is None or title_el.text is None:
                continue
            headline = title_el.text.strip()
            url_str = (link_el.text or "").strip() if link_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            full_text = f"{headline} {desc}"
            tickers = _extract_tickers(full_text)
            score = _score_text(full_text)
            items.append(NewsItem(
                headline=headline,
                source=source_name,
                url=url_str,
                tickers=tickers,
                score=score,
            ))
    except Exception as exc:
        log.warning("Failed to fetch %s RSS (%s): %s", source_name, url, exc)
    return items


def fetch_yahoo_news() -> list[NewsItem]:
    return _fetch_rss(
        "https://finance.yahoo.com/news/rssindex",
        "yahoo",
    )


def fetch_google_news() -> list[NewsItem]:
    queries = [
        "stock+market",
        "earnings+beat+stock",
        "FDA+approval+stock",
        "analyst+upgrade+stock",
        "stock+surge+today",
    ]
    items: list[NewsItem] = []
    for q in queries:
        url = (
            f"https://news.google.com/rss/search?"
            f"q={q}&hl=en-US&gl=US&ceid=US:en"
        )
        items.extend(_fetch_rss(url, "google"))
        time.sleep(0.5)
    return items


# ---------------------------------------------------------------------------
# Fetchers -- Finviz
# ---------------------------------------------------------------------------

def _get_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed, Finviz sources unavailable")
        return None


def fetch_finviz_news() -> list[NewsItem]:
    """Scrape the Finviz general news page."""
    BeautifulSoup = _get_bs4()
    if BeautifulSoup is None:
        return []

    items: list[NewsItem] = []
    url = "https://finviz.com/news.ashx"
    try:
        resp = _requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for table in soup.find_all("table", class_="table-fixed"):
            for row in table.find_all("tr"):
                link = row.find("a", class_="tab-link-news")
                if link is None:
                    link = row.find("a")
                if link is None:
                    continue
                headline = link.get_text(strip=True)
                href = link.get("href", "")
                if not headline:
                    continue
                tickers = _extract_tickers(headline)
                score = _score_text(headline)
                items.append(NewsItem(
                    headline=headline,
                    source="finviz",
                    url=href,
                    tickers=tickers,
                    score=score,
                ))
    except Exception as exc:
        log.warning("Failed to fetch Finviz news: %s", exc)
    return items


def fetch_finviz_screener() -> list[NewsItem]:
    """Scrape Finviz signal pages: top gainers, new highs, most active.

    These pages have tickers explicitly in the table rows, so we get
    ticker-tagged news without needing name-based extraction.
    """
    BeautifulSoup = _get_bs4()
    if BeautifulSoup is None:
        return []

    signals = [
        ("ta_topgainers", "finviz-gainers"),
        ("ta_newhigh", "finviz-newhigh"),
        ("ta_mostactive", "finviz-active"),
    ]
    items: list[NewsItem] = []

    for signal_id, source_label in signals:
        url = f"https://finviz.com/screener.ashx?v=340&s={signal_id}"
        try:
            resp = _requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            table = soup.find("table", class_="screener_table") or soup.find("table", id="screener-views-table")
            if table is None:
                for t in soup.find_all("table"):
                    if t.find("a", class_="screener-link-primary"):
                        table = t
                        break
            if table is None:
                log.debug("No screener table found for %s", signal_id)
                continue

            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                ticker_link = cells[1].find("a") if len(cells) > 1 else None
                if ticker_link is None:
                    ticker_link = row.find("a", class_="screener-link-primary")
                if ticker_link is None:
                    continue

                ticker = ticker_link.get_text(strip=True).upper()
                if not ticker or not ticker.isalpha():
                    continue

                company_cell = cells[2] if len(cells) > 2 else None
                company_name = company_cell.get_text(strip=True) if company_cell else ticker

                headline = f"{company_name} ({ticker}) on {source_label.replace('finviz-', '')} signal"
                score = _score_text(headline)
                if score <= 0:
                    score = 3.0

                items.append(NewsItem(
                    headline=headline,
                    source=source_label,
                    url=url,
                    tickers=[ticker],
                    score=score,
                ))
        except Exception as exc:
            log.warning("Failed to fetch Finviz screener %s: %s", signal_id, exc)
        time.sleep(0.5)

    log.info("Fetched %d items from Finviz screener pages", len(items))
    return items


def fetch_finviz_ticker_news(ticker: str) -> list[NewsItem]:
    """Fetch news headlines for a specific ticker from its Finviz quote page."""
    BeautifulSoup = _get_bs4()
    if BeautifulSoup is None:
        return []

    items: list[NewsItem] = []
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    try:
        resp = _requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        news_table = soup.find("table", id="news-table")
        if news_table is None:
            for t in soup.find_all("table"):
                rows = t.find_all("tr")
                if rows and rows[0].find("a") and len(rows) > 3:
                    first_link = rows[0].find("a")
                    if first_link and first_link.get("href", "").startswith("http"):
                        news_table = t
                        break
        if news_table is None:
            return items

        for row in news_table.find_all("tr"):
            link = row.find("a")
            if link is None:
                continue
            headline = link.get_text(strip=True)
            href = link.get("href", "")
            if not headline:
                continue
            score = _score_text(headline)
            items.append(NewsItem(
                headline=headline,
                source=f"finviz-{ticker.lower()}",
                url=href,
                tickers=[ticker.upper()],
                score=score,
            ))
    except Exception as exc:
        log.warning("Failed to fetch Finviz ticker news for %s: %s", ticker, exc)
    return items


# ---------------------------------------------------------------------------
# Live price quote (lightweight, for news movers)
# ---------------------------------------------------------------------------

_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"


def _fetch_quote(ticker: str) -> tuple[float, float]:
    """Fetch current price and daily change % for a ticker.

    Uses ``interval=1d`` with ``meta.regularMarketPrice`` which is more
    reliable than 1-minute bars (Yahoo may block intraday access without
    session cookies).

    Returns (price, change_pct). Falls back to (0.0, 0.0) on failure.
    """
    url = f"{_YAHOO_CHART}/{ticker}"
    params = {"range": "1d", "interval": "1d", "includePrePost": "false"}
    try:
        resp = _requests.get(url, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice", 0.0)
        prev = meta.get("chartPreviousClose") or meta.get("previousClose") or 0.0
        if price and prev and prev > 0:
            change = round((price - prev) / prev * 100, 2)
        else:
            change = 0.0
        return round(float(price), 2), change
    except Exception as exc:
        log.debug("Quote fetch failed for %s: %s", ticker, exc)
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Aggregation and main entry point
# ---------------------------------------------------------------------------

def _aggregate_movers(all_items: list[NewsItem]) -> dict[str, list[NewsItem]]:
    """Group positive-scoring items by ticker."""
    ticker_items: dict[str, list[NewsItem]] = {}
    for item in all_items:
        if item.score <= 0:
            continue
        for ticker in item.tickers:
            ticker_items.setdefault(ticker, []).append(item)
    return ticker_items


def _build_movers(ticker_items: dict[str, list[NewsItem]]) -> list[NewsMover]:
    movers: list[NewsMover] = []
    for ticker, items in ticker_items.items():
        agg_score = sum(it.score for it in items)
        gain = _predicted_gain(agg_score)
        if gain < 5.0:
            continue

        sorted_items = sorted(items, key=lambda it: it.score, reverse=True)
        top = [it.headline for it in sorted_items[:3]]

        movers.append(NewsMover(
            ticker=ticker,
            predicted_gain_pct=gain,
            news_score=round(agg_score, 1),
            headline_count=len(items),
            top_headlines=top,
        ))

    movers.sort(key=lambda m: m.news_score, reverse=True)

    log.info("Fetching live prices for %d movers...", len(movers))
    for m in movers:
        price, change = _fetch_quote(m.ticker)
        m.current_price = price
        m.change_pct = change
        time.sleep(0.15)

    return movers


def scan_news(watchlist: list[str] | None = None) -> list[NewsMover]:
    """Fetch news from all sources, score, aggregate, return movers >= 5%.

    Args:
        watchlist: Optional list of tickers to always include in the scan.
                   Their per-ticker news is fetched regardless of pass 1 results.

    Two-pass approach:
      Pass 1: Fetch from RSS + Finviz general + Finviz screener, identify candidates.
      Pass 2: For each candidate ticker (+ watchlist), fetch its dedicated Finviz
              page for additional headlines, then re-aggregate.
    """
    _ensure_ticker_data()

    watchlist_set = {t.upper() for t in (watchlist or [])}

    # --- Pass 1: broad fetch ---
    log.info("Fetching news from Yahoo Finance...")
    yahoo = fetch_yahoo_news()
    log.info("Fetched %d headlines from Yahoo", len(yahoo))

    log.info("Fetching news from Google News...")
    google = fetch_google_news()
    log.info("Fetched %d headlines from Google News", len(google))

    log.info("Fetching news from Finviz general...")
    finviz_gen = fetch_finviz_news()
    log.info("Fetched %d headlines from Finviz general", len(finviz_gen))

    log.info("Fetching news from Finviz screener pages...")
    finviz_screen = fetch_finviz_screener()
    log.info("Fetched %d items from Finviz screener", len(finviz_screen))

    pass1_items = _deduplicate(yahoo + google + finviz_gen + finviz_screen)
    log.info("Pass 1 unique headlines: %d", len(pass1_items))

    pass1_ticker_items = _aggregate_movers(pass1_items)

    candidate_tickers = {
        t for t, items in pass1_ticker_items.items()
        if sum(it.score for it in items) > 0
    }
    candidate_tickers |= watchlist_set
    if watchlist_set:
        log.info("Watchlist tickers added to candidates: %s", ", ".join(sorted(watchlist_set)))
    log.info("Pass 1 candidate tickers: %d", len(candidate_tickers))

    # --- Pass 2: per-ticker deep fetch ---
    extra_items: list[NewsItem] = []
    fetched = 0
    for ticker in sorted(candidate_tickers):
        if fetched >= 25 and ticker not in watchlist_set:
            continue
        log.debug("Fetching Finviz per-ticker news for %s", ticker)
        ticker_news = fetch_finviz_ticker_news(ticker)
        if ticker_news:
            extra_items.extend(ticker_news)
            fetched += 1
        time.sleep(0.3)

    if extra_items:
        log.info("Pass 2: fetched %d extra headlines for %d tickers", len(extra_items), fetched)

    all_items = _deduplicate(pass1_items + extra_items)
    log.info("Total unique headlines after pass 2: %d", len(all_items))

    ticker_items = _aggregate_movers(all_items)
    movers = _build_movers(ticker_items)

    log.info("Found %d news movers with predicted gain >= 5%%", len(movers))
    return movers


# ---------------------------------------------------------------------------
# Formatting helpers (used by monitor.py)
# ---------------------------------------------------------------------------

def format_news_text(movers: list[NewsMover]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "STOCK MONITOR -- News Scanner",
        now,
        "",
    ]

    if not movers:
        lines.append("No stocks with predicted gain >= 5% found in current news.")
    else:
        lines.append("NEWS MOVERS (predicted gain >= 5%)")
        lines.append("")
        for m in movers:
            price_str = f"${m.current_price:>9.2f}" if m.current_price else "     N/A "
            change_str = f"{m.change_pct:>+7.2f}%" if m.current_price else "    N/A "
            lines.append(
                f"  {m.ticker:<6}  {price_str}  {change_str}  "
                f"Predicted: {m.predicted_gain_pct:>+6.1f}%  "
                f"Score: {m.news_score:>5.1f}  "
                f"{m.headline_count} article{'s' if m.headline_count != 1 else ''}"
            )
            for hl in m.top_headlines:
                lines.append(f'    - "{hl}"')
            lines.append("")

    lines.append("-" * 72)
    lines.append("Engine: News Scanner | Sources: Yahoo Finance, Google News, Finviz")
    lines.append("Not financial advice. Do your own research.")
    return "\n".join(lines)


def format_news_json(movers: list[NewsMover]) -> str:
    return json.dumps([m.to_dict() for m in movers], indent=2)
