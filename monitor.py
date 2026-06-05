#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stock_monitor.config import WATCHLIST_PATH, WATCHLIST_SA_PATH, TIMEFRAME_5D, TIMEFRAME_1D, TimeframeConfig
from stock_monitor.analyzer import analyze
from stock_monitor.report import (
    format_text,
    format_json,
    save_report,
    load_previous_report,
    detect_changes,
)
from telegram_sender.sender import sendMessage, sendNewsMessage

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "NFLX", "AMD", "INTC",
    "AVGO", "COST", "PEP", "ADBE", "CRM",
]

DEFAULT_SA_TICKERS = [
    "2222.SR",  # Saudi Aramco
    "1010.SR",  # Al Rajhi Bank
    "1150.SR",  # Saudi National Bank (SNB)
    "2010.SR",  # SABIC
    "4001.SR",  # STC (Saudi Telecom)
    "1180.SR",  # Al Tawuniya
    "1211.SR",  # Ma'aden
    "2380.SR",  # Petro Rabigh
    "2020.SR",  # SABIC Agri-Nutrients
    "4190.SR",  # Jarir Marketing
    "2350.SR",  # Saudi Kayan
    "2050.SR",  # Savola Group
    "2060.SR",  # Tasnee
    "4347.SR",  # Elm Company
    "1120.SR",  # Al Rajhi REIT
]

log = logging.getLogger("monitor")

_shutdown = False


def _handle_signal(signum: int, frame) -> None:
    global _shutdown
    _shutdown = True
    log.info("Shutdown requested (signal %d)", signum)


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
        description="AI Stock Monitor -- prediction engine"
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
        "--daily", action="store_true",
        help="1-day predictions with live intraday prices",
    )
    parser.add_argument(
        "--sa", action="store_true",
        help="Analyze Saudi market (Tadawul) stocks instead of US",
    )
    parser.add_argument(
        "--news", action="store_true",
        help="Scan news for stocks likely to gain 5%+",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run continuously: retrain every hour, report every 3 hours",
    )
    parser.add_argument(
        "--train-interval", type=int, default=3600,
        help="Seconds between training cycles in daemon mode (default: 3600)",
    )
    parser.add_argument(
        "--report-interval", type=int, default=10800,
        help="Seconds between full reports in daemon mode (default: 10800)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def run_cycle(
    tickers: list[str],
    force_retrain: bool,
    output_json: bool,
    print_report: bool,
    timeframe: TimeframeConfig = TIMEFRAME_5D,
    market: str = "US",
    currency: str = "$",
) -> None:
    previous = load_previous_report()
    results = []

    for i, ticker in enumerate(tickers, 1):
        log.info("[%d/%d] %s", i, len(tickers), ticker)
        res = analyze(ticker, force_retrain=force_retrain, timeframe=timeframe, market=market, currency=currency)

        sendMessage(res)

        results.append(res)

        if i < len(tickers):
            time.sleep(2)

    changes = detect_changes(results, previous)
    save_report(results)

    if print_report:
        if output_json:
            print(format_json(results))
        else:
            print(format_text(results, changes))
        sys.stdout.flush()


def run_daemon(
    tickers: list[str],
    output_json: bool,
    train_interval: int,
    report_interval: int,
    timeframe: TimeframeConfig = TIMEFRAME_5D,
    market: str = "US",
    currency: str = "$",
) -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "Daemon started: %d tickers (%s market), train every %ds, report every %ds, timeframe=%s",
        len(tickers), market, train_interval, report_interval, timeframe.label,
    )

    last_report_time = 0.0
    cycle = 0

    while not _shutdown:
        cycle += 1
        now = time.time()
        should_report = (now - last_report_time) >= report_interval

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        log.info("--- Cycle %d at %s (report=%s) ---", cycle, ts, should_report)

        run_cycle(
            tickers=tickers,
            force_retrain=True,
            output_json=output_json,
            print_report=should_report,
            timeframe=timeframe,
            market=market,
            currency=currency,
        )

        if should_report:
            last_report_time = time.time()

        if _shutdown:
            break

        log.info("Sleeping %ds until next cycle...", train_interval)
        deadline = time.time() + train_interval
        while time.time() < deadline and not _shutdown:
            time.sleep(min(5, deadline - time.time()))

    log.info("Daemon stopped after %d cycles", cycle)


def run_news(output_json: bool, tickers: list[str] | None = None) -> None:
    from stock_monitor.news import scan_news, format_news_text, format_news_json

    movers = scan_news(watchlist=tickers)

    sendNewsMessage(movers)

    if output_json:
        print(format_news_json(movers))
    else:
        print(format_news_text(movers))
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.sa:
        market, currency = "SA", "SAR"
        watchlist_path = WATCHLIST_SA_PATH
        default_tickers = DEFAULT_SA_TICKERS
    else:
        market, currency = "US", "$"
        watchlist_path = WATCHLIST_PATH
        default_tickers = DEFAULT_TICKERS

    if args.news:
        tickers = args.tickers or load_watchlist(watchlist_path) or None
        run_news(output_json=args.json, tickers=tickers)
        return

    tickers = args.tickers or load_watchlist(watchlist_path) or default_tickers
    timeframe = TIMEFRAME_1D if args.daily else TIMEFRAME_5D

    if args.daemon:
        run_daemon(
            tickers=tickers,
            output_json=args.json,
            train_interval=args.train_interval,
            report_interval=args.report_interval,
            timeframe=timeframe,
            market=market,
            currency=currency,
        )
    else:
        log.info("Analyzing %d tickers (%s, %s market)...", len(tickers), timeframe.label, market)
        run_cycle(
            tickers=tickers,
            force_retrain=args.retrain,
            output_json=args.json,
            print_report=True,
            timeframe=timeframe,
            market=market,
            currency=currency,
        )


if __name__ == "__main__":
    main()
