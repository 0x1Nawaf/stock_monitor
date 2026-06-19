from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from .config import REPORTS_DIR, MAX_REPORT_FILES, Signal
from .analyzer import StockAnalysis


def _signal_category(signal: Signal) -> str:
    if signal in (Signal.STRONG_BUY, Signal.BUY, Signal.LEAN_BUY):
        return "BUY"
    if signal in (Signal.STRONG_SELL, Signal.SELL, Signal.LEAN_SELL):
        return "SELL"
    return "HOLD"


def detect_changes(
    current: list[StockAnalysis],
    previous: dict[str, Any],
) -> list[dict[str, Any]]:
    prev_results = previous.get("results", {})
    changes: list[dict[str, Any]] = []

    for result in current:
        if result.error:
            continue
        prev = prev_results.get(result.ticker)
        if prev is None:
            continue
        prev_signal = prev.get("signal", "HOLD")
        prev_cat = (
            "BUY" if "BUY" in prev_signal
            else ("SELL" if "SELL" in prev_signal else "HOLD")
        )
        curr_cat = _signal_category(result.signal)
        if prev_cat != curr_cat:
            changes.append({
                "ticker": result.ticker,
                "from": prev_signal,
                "to": result.signal.value,
                "score_delta": result.score - prev.get("score", 0),
                "price": result.price,
                "currency": result.currency,
            })
    return changes


def _atomic_write(path, content: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode())
        os.close(fd)
        closed = True
        os.replace(tmp_path, path)
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_report(results: list[StockAnalysis]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    data = {
        "timestamp": now.isoformat(),
        "results": {
            r.ticker: r.to_dict() for r in results if not r.error
        },
    }
    content = json.dumps(data, indent=2, default=str)
    ts_file = REPORTS_DIR / f"report-{now.strftime('%Y%m%d-%H%M')}.json"
    latest_file = REPORTS_DIR / "latest.json"

    ts_file.write_text(content)
    _atomic_write(latest_file, content)

    reports = sorted(REPORTS_DIR.glob("report-*.json"))
    for stale in reports[:-MAX_REPORT_FILES]:
        stale.unlink(missing_ok=True)


def load_previous_report() -> dict[str, Any]:
    latest = REPORTS_DIR / "latest.json"
    if not latest.exists():
        return {}
    try:
        return json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def format_text(
    results: list[StockAnalysis],
    changes: list[dict[str, Any]] | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    markets = {r.market for r in results if not r.error}
    market_label = " | ".join(sorted(markets)) if markets else "US"
    lines: list[str] = [
        f"STOCK MONITOR -- AI Engine [{market_label}]",
        now,
        "",
    ]

    valid = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    valid.sort(key=lambda r: r.score, reverse=True)

    model_types = {r.model_type for r in valid}
    engine_label = "Ensemble (GBM + LSTM)" if "ensemble" in model_types else "GBM Classifier"

    if changes:
        lines.append("SIGNAL CHANGES")
        for c in changes:
            arrow = "^" if c["score_delta"] > 0 else "v"
            cur = c.get("currency", "$")
            lines.append(
                f"  {arrow} {c['ticker']}  {cur}{c['price']}  "
                f"{c['from']} -> {c['to']}  (score {c['score_delta']:+d})"
            )
        lines.append("")

    actionable = [r for r in valid if r.signal.value in ("STRONG BUY", "BUY")]
    other = [r for r in valid if r.signal.value not in ("STRONG BUY", "BUY")]

    if actionable:
        for label in ("STRONG BUY", "BUY"):
            group = [r for r in actionable if r.signal.value == label]
            if not group:
                continue
            lines.append(label)
            for r in group:
                lines.append(
                    f"  {r.ticker:<10}  {r.currency}{r.price:>10.2f}  {r.change_pct:>+7.2f}%  "
                    f"Score: {r.score:>+4d}  "
                    f"P(UP)={r.prob_up:.0%} P(DN)={r.prob_down:.0%}  "
                    f"Conf: {r.confidence:.0%}"
                )
            lines.append("")
    else:
        lines.append("No BUY signals at this time.")
        lines.append("")

    if other:
        lines.append(f"({len(other)} other ticker(s) not showing actionable signals)")
        lines.append("")

    lines.append("-" * 72)
    lines.append("DETAILS")
    lines.append("-" * 72)

    for r in actionable:
        lines.append("")
        lines.append(
            f"{r.signal.value}  {r.ticker}  {r.currency}{r.price}  ({r.change_pct:+.2f}%)  "
            f"Score: {r.score:+d}"
        )
        lines.append(
            f"  SMA(20): {r.currency}{r.sma_20}  SMA(50): {r.currency}{r.sma_50}  RSI(14): {r.rsi}"
        )
        lines.append(
            f"  Support: {r.currency}{r.support}  Resistance: {r.currency}{r.resistance}"
        )
        horizon_label = "1 day" if r.timeframe == "1d" else f"{r.timeframe.rstrip('d')} days"
        lines.append(
            f"  Predicted {r.predicted_return_pct:+.2f}% over {horizon_label}  "
            f"(confidence {r.confidence:.0%})"
        )
        lines.append(
            f"  Probabilities: UP={r.prob_up:.1%}  FLAT={r.prob_flat:.1%}  DOWN={r.prob_down:.1%}"
        )
        if r.ensemble_agreement > 0:
            lines.append(f"  Ensemble agreement: {r.ensemble_agreement:.0%}")
        for reason in r.reasons:
            lines.append(f"    {reason}")

    if errors:
        lines.append("")
        lines.append("ERRORS")
        for r in errors:
            lines.append(f"  {r.ticker}: {r.error}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"Engine: {engine_label} | Classification-based prediction")
    lines.append("Not financial advice. Do your own research.")

    return "\n".join(lines)


def format_json(results: list[StockAnalysis]) -> str:
    valid = [r for r in results if not r.error]
    valid.sort(key=lambda r: r.score, reverse=True)
    return json.dumps([r.to_dict() for r in valid], indent=2)
