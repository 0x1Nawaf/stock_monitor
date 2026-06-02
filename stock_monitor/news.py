from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests as _requests

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
_TIMEOUT = 15

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
    top_headlines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "predicted_gain_pct": self.predicted_gain_pct,
            "news_score": self.news_score,
            "headline_count": self.headline_count,
            "top_headlines": self.top_headlines,
        }


# ---------------------------------------------------------------------------
# Company name -> ticker mapping (~100 major companies)
# ---------------------------------------------------------------------------

COMPANY_TICKERS: dict[str, str] = {
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
}

_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
_COMPANY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE), ticker)
    for name, ticker in COMPANY_TICKERS.items()
]

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


def _extract_tickers(text: str) -> list[str]:
    tickers: set[str] = set()
    for m in _TICKER_RE.finditer(text):
        tickers.add(m.group(1))
    for pattern, ticker in _COMPANY_PATTERNS:
        if pattern.search(text):
            tickers.add(ticker)
    return sorted(tickers)


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
# Fetchers
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


def fetch_finviz_news() -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed, skipping Finviz source")
        return items

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


# ---------------------------------------------------------------------------
# Aggregation and main entry point
# ---------------------------------------------------------------------------

def scan_news() -> list[NewsMover]:
    """Fetch news from all sources, score, aggregate, return movers >= 5%."""
    log.info("Fetching news from Yahoo Finance...")
    yahoo = fetch_yahoo_news()
    log.info("Fetched %d headlines from Yahoo", len(yahoo))

    log.info("Fetching news from Google News...")
    google = fetch_google_news()
    log.info("Fetched %d headlines from Google News", len(google))

    log.info("Fetching news from Finviz...")
    finviz = fetch_finviz_news()
    log.info("Fetched %d headlines from Finviz", len(finviz))

    all_items = _deduplicate(yahoo + google + finviz)
    log.info("Total unique headlines: %d", len(all_items))

    ticker_items: dict[str, list[NewsItem]] = {}
    for item in all_items:
        if item.score <= 0:
            continue
        for ticker in item.tickers:
            ticker_items.setdefault(ticker, []).append(item)

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
            lines.append(
                f"  {m.ticker:<6}  Predicted: {m.predicted_gain_pct:>+6.1f}%  "
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
