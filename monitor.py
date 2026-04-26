#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from stock_monitor.config import WATCHLIST_PATH
from stock_monitor.analyzer import analyze
from stock_monitor.report import (
    format_text,
    format_json,
    save_report,
    load_previous_report,
    detect_changes,
)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "NFLX", "AMD", "INTC",
    "AVGO", "COST", "PEP", "ADBE", "CRM",
]


def load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [
        line.strip().upper()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Stock Monitor -- LSTM prediction engine"
    )
    parser.add_argument(
        "tickers", nargs="*", type=str.upper,
        help="Tickers to analyze (default: watchlist.txt)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force model retraining for all tickers",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    tickers = args.tickers or load_watchlist(WATCHLIST_PATH) or DEFAULT_TICKERS

    print(
        f"Analyzing {len(tickers)} tickers (LSTM engine)...",
        file=sys.stderr,
    )

    previous = load_previous_report()
    results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}", file=sys.stderr)
        results.append(analyze(ticker, force_retrain=args.retrain))
        if i < len(tickers):
            time.sleep(2)

    changes = detect_changes(results, previous)
    save_report(results)

    if args.json:
        print(format_json(results))
    else:
        print(format_text(results, changes))


if __name__ == "__main__":
    main()
