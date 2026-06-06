"""Telegram Bot API notification sender for stock monitor alerts."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stock_monitor.analyzer import StockAnalysis
    from stock_monitor.news import NewsMover

log = logging.getLogger(__name__)

_SIGNAL_ICONS = {
    "STRONG BUY": "🟢🟢",
    "BUY": "🟢",
    "LEAN BUY": "🟡",
    "HOLD": "⚪",
    "LEAN SELL": "🟠",
    "SELL": "🔴",
    "STRONG SELL": "🔴🔴",
}


def _get_credentials() -> tuple[str, str] | None:
    try:
        import config
        token = config.env_config.get("telegram_bot_token", "")
        chat_id = config.env_config.get("telegram_chat_id", "")
        if token and chat_id:
            return token, chat_id
    except Exception:
        pass
    return None


def _send(token: str, chat_id: str, text: str) -> bool:
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if not resp.ok:
            log.warning("Telegram send failed: %s", resp.text)
        return resp.ok
    except Exception as exc:
        log.warning("Telegram send error: %s", exc)
        return False


def sendMessage(res: StockAnalysis) -> bool:
    """Send a per-ticker analysis alert."""
    creds = _get_credentials()
    if creds is None:
        return False
    token, chat_id = creds

    if res.error:
        return False

    signal_icon = _SIGNAL_ICONS.get(res.signal.value, "⚪")
    market_flag = "🇸🇦" if res.market == "SA" else "🇺🇸"
    tf_label = "1 day" if res.timeframe == "1d" else "5 days"
    direction = "+" if res.predicted_return_pct >= 0 else ""

    text = (
        f"{signal_icon} <b>{res.ticker}</b> — {res.signal.value}\n"
        f"\n"
        f"Price:      {res.currency}{res.price:.2f} ({res.change_pct:+.2f}%)\n"
        f"Predicted:  {direction}{res.predicted_return_pct:.2f}% over {tf_label}\n"
        f"Confidence: {res.confidence * 100:.0f}%\n"
        f"Score:      {res.score:+d}/100\n"
        f"\n"
        f"SMA(20): {res.currency}{res.sma_20}  |  SMA(50): {res.currency}{res.sma_50}\n"
        f"Support: {res.currency}{res.support}  |  Resistance: {res.currency}{res.resistance}\n"
        f"RSI(14): {res.rsi}\n"
        f"\n"
        f"<i>{market_flag} {res.market} Market  •  Model age: {res.model_age_days:.0f}d</i>"
    )

    return _send(token, chat_id, text)


def sendNewsMessage(movers: list[NewsMover]) -> bool:
    """Send a news scanner summary alert."""
    creds = _get_credentials()
    if creds is None:
        return False
    token, chat_id = creds

    if not movers:
        text = "📰 <b>News Scanner</b>\n\nNo stocks with predicted gain >= 5% found in current news."
        return _send(token, chat_id, text)

    lines = ["📰 <b>News Scanner — Top Movers</b>\n"]

    for m in movers[:10]:
        price_str = f"${m.current_price:.2f}" if m.current_price else "N/A"
        change_str = f" ({m.change_pct:+.2f}%)" if m.current_price else ""

        lines.append(
            f"<b>{m.ticker}</b>  {price_str}{change_str}\n"
            f"  ▸ Predicted gain: +{m.predicted_gain_pct:.0f}%  •  "
            f"Score: {m.news_score:.0f}  •  "
            f"{m.headline_count} article{'s' if m.headline_count != 1 else ''}"
        )
        if m.top_headlines:
            headline = m.top_headlines[0]
            if len(headline) > 80:
                headline = headline[:77] + "..."
            lines.append(f"  <i>{headline}</i>")
        lines.append("")

    lines.append("<i>Not financial advice. Do your own research.</i>")
    text = "\n".join(lines)

    return _send(token, chat_id, text)
