"""Tests for report formatting and change detection."""
from __future__ import annotations

import json

import pytest

from stock_monitor.config import Signal
from stock_monitor.analyzer import StockAnalysis
from stock_monitor.report import format_text, format_json, detect_changes, _signal_category


def _make_analysis(ticker="AAPL", signal=Signal.BUY, score=50, price=150.0) -> StockAnalysis:
    return StockAnalysis(
        ticker=ticker,
        price=price,
        change_pct=1.5,
        signal=signal,
        score=score,
        predicted_return_pct=2.5,
        confidence=0.8,
        model_age_days=1.0,
        support=145.0,
        resistance=155.0,
        sma_20=148.0,
        sma_50=146.0,
        rsi=55.0,
    )


class TestSignalCategory:
    def test_buy_signals(self):
        assert _signal_category(Signal.STRONG_BUY) == "BUY"
        assert _signal_category(Signal.BUY) == "BUY"
        assert _signal_category(Signal.LEAN_BUY) == "BUY"

    def test_sell_signals(self):
        assert _signal_category(Signal.STRONG_SELL) == "SELL"
        assert _signal_category(Signal.SELL) == "SELL"
        assert _signal_category(Signal.LEAN_SELL) == "SELL"

    def test_hold(self):
        assert _signal_category(Signal.HOLD) == "HOLD"


class TestDetectChanges:
    def test_detects_buy_to_sell(self):
        current = [_make_analysis("AAPL", Signal.SELL, -30)]
        previous = {"results": {"AAPL": {"signal": "BUY", "score": 50}}}
        changes = detect_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["ticker"] == "AAPL"
        assert changes[0]["from"] == "BUY"
        assert changes[0]["to"] == "SELL"

    def test_no_change_same_category(self):
        current = [_make_analysis("AAPL", Signal.STRONG_BUY, 80)]
        previous = {"results": {"AAPL": {"signal": "BUY", "score": 50}}}
        changes = detect_changes(current, previous)
        assert len(changes) == 0

    def test_skips_errors(self):
        failed = StockAnalysis.failed("AAPL", "fetch error")
        previous = {"results": {"AAPL": {"signal": "BUY", "score": 50}}}
        changes = detect_changes([failed], previous)
        assert len(changes) == 0


class TestFormatText:
    def test_contains_ticker(self):
        results = [_make_analysis("TSLA")]
        text = format_text(results)
        assert "TSLA" in text

    def test_contains_signal_group(self):
        results = [_make_analysis("AAPL", Signal.BUY)]
        text = format_text(results)
        assert "BUY" in text

    def test_contains_header(self):
        results = [_make_analysis()]
        text = format_text(results)
        assert "STOCK MONITOR" in text

    def test_errors_section(self):
        results = [StockAnalysis.failed("BAD", "some error")]
        text = format_text(results)
        assert "ERRORS" in text
        assert "BAD" in text


class TestFormatJson:
    def test_valid_json(self):
        results = [_make_analysis("AAPL"), _make_analysis("TSLA")]
        output = format_json(results)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_sorted_by_score(self):
        r1 = _make_analysis("AAPL", score=30)
        r2 = _make_analysis("TSLA", score=80)
        output = format_json([r1, r2])
        parsed = json.loads(output)
        assert parsed[0]["ticker"] == "TSLA"
        assert parsed[1]["ticker"] == "AAPL"
