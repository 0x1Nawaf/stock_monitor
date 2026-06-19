from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stock_monitor.analyzer import StockAnalysis
    from stock_monitor.gainers import Gainer, GainerAnalysis
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

_MSG_IDS_FILE = Path(__file__).resolve().parent.parent / ".telegram_msg_ids"


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


def _save_message_id(msg_id: int) -> None:
    ids = _load_message_ids()
    ids.append(msg_id)
    _MSG_IDS_FILE.write_text(json.dumps(ids))


def _load_message_ids() -> list[int]:
    if not _MSG_IDS_FILE.exists():
        return []
    try:
        return json.loads(_MSG_IDS_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def _clear_message_ids() -> None:
    if _MSG_IDS_FILE.exists():
        _MSG_IDS_FILE.unlink()


def clear_previous() -> None:
    creds = _get_credentials()
    if creds is None:
        return
    token, chat_id = creds

    import requests

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": "🔄"},
        timeout=10,
    )
    if not resp.ok:
        return

    latest_id = resp.json().get("result", {}).get("message_id", 0)
    if not latest_id:
        return

    msg_ids = list(range(latest_id, max(latest_id - 200, 0), -1))

    for batch_start in range(0, len(msg_ids), 100):
        batch = msg_ids[batch_start:batch_start + 100]
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/deleteMessages",
                json={"chat_id": chat_id, "message_ids": batch},
                timeout=10,
            )
        except Exception:
            pass

    _clear_message_ids()
    log.info("Cleared Telegram chat (swept %d message IDs)", len(msg_ids))


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
            return False
        msg_id = resp.json().get("result", {}).get("message_id")
        if msg_id:
            _save_message_id(msg_id)
        return True
    except Exception as exc:
        log.warning("Telegram send error: %s", exc)
        return False


def sendMessage(res: StockAnalysis) -> bool:
    creds = _get_credentials()
    if creds is None:
        return False
    token, chat_id = creds

    if res.error:
        return False

    if res.signal.value not in ("STRONG BUY", "BUY"):
        return False

    signal_icon = _SIGNAL_ICONS.get(res.signal.value, "⚪")
    market_flag = "🇸🇦" if res.market == "SA" else "🇺🇸"
    tf_labels = {"1d": "1 day", "5d": "5 days", "swing": "swing (10 days)", "monthly": "1 month (21 days)"}
    tf_label = tf_labels.get(res.timeframe, res.timeframe)
    direction = "+" if res.predicted_return_pct >= 0 else ""

    action = ""
    if res.signal.value in ("STRONG BUY", "BUY", "LEAN BUY"):
        action = f"🟢 <b>Buy at {res.currency}{res.price:.2f}</b>\n"
        if res.stop_loss > 0:
            action += f"🛑 Stop loss: {res.currency}{res.stop_loss:.2f}\n"
    elif res.signal.value in ("STRONG SELL", "SELL", "LEAN SELL"):
        action = f"🔴 <b>Sell at {res.currency}{res.price:.2f}</b>\n"
        if res.stop_loss > 0:
            action += f"🛑 Stop loss: {res.currency}{res.stop_loss:.2f}\n"

    text = (
        f"{signal_icon} <b>{res.ticker}</b> — {res.signal.value}\n"
        f"\n"
        f"{action}"
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


def sendGainersMessage(gainers: list[Gainer]) -> bool:
    creds = _get_credentials()
    if creds is None:
        return False
    token, chat_id = creds

    if not gainers:
        text = "📈 <b>Top Gainers</b>\n\nNo significant gainers found at this time."
        return _send(token, chat_id, text)

    lines = ["📈 <b>Top Gainers Today (Intraday + Daily)</b>\n"]

    for i, g in enumerate(gainers[:15], 1):
        vol_str = ""
        if g.volume_ratio >= 2.0:
            vol_str = f"  🔥 {g.volume_ratio:.1f}x vol"

        cap_str = ""
        if g.market_cap >= 1e12:
            cap_str = f"  ({g.market_cap / 1e12:.1f}T)"
        elif g.market_cap >= 1e9:
            cap_str = f"  ({g.market_cap / 1e9:.1f}B)"

        lines.append(
            f"<b>{i}. {g.ticker}</b>  ${g.price:.2f}  "
            f"<b>{g.change_pct:+.2f}%</b>{vol_str}{cap_str}"
        )
        if g.company and g.company != g.ticker:
            name = g.company if len(g.company) <= 40 else g.company[:37] + "..."
            lines.append(f"   <i>{name}</i>")

        a = g.analysis
        if a and not a.error:
            if a.intraday:
                intra = a.intraday
                sig_icon = _SIGNAL_ICONS.get(intra.signal, "⚪")
                lines.append(f"   ⏱ <b>1H:</b> {sig_icon} {intra.signal} | {intra.trend}")
                if intra.signal in ("STRONG BUY", "BUY", "LEAN BUY"):
                    sl = f"  SL: ${intra.stop_loss:.2f}" if intra.stop_loss > 0 else ""
                    tp = f"  TP: ${intra.resistance:.2f}" if intra.resistance > 0 and intra.resistance > g.price else ""
                    lines.append(f"   💰 Entry: ${g.price:.2f}{sl}{tp}")
                elif intra.signal in ("SELL", "LEAN SELL", "STRONG SELL"):
                    lines.append(f"   ⚠️ Intraday bearish")

            if a.daily:
                daily = a.daily
                sig_icon = _SIGNAL_ICONS.get(daily.signal, "⚪")
                lines.append(f"   📅 <b>1D:</b> {sig_icon} {daily.signal} ({daily.confidence:.0%} conf)")
                if daily.signal in ("STRONG BUY", "BUY", "LEAN BUY"):
                    sl = f"  SL: ${daily.stop_loss:.2f}" if daily.stop_loss > 0 else ""
                    tp = f"  TP: ${daily.resistance:.2f}" if daily.resistance > 0 and daily.resistance > g.price else ""
                    lines.append(f"   💰 Daily: ${g.price:.2f}{sl}{tp}")
                elif daily.signal in ("SELL", "LEAN SELL", "STRONG SELL"):
                    lines.append(f"   ⚠️ Daily bearish — {daily.prob_down:.0%} downside")

            if a.risks:
                lines.append(f"   ⚡ {a.risks[0]}")
        lines.append("")

    high_vol = [g for g in gainers if g.volume_ratio >= 2.0]
    if high_vol:
        lines.append(
            f"🔥 {len(high_vol)} stock(s) trading at 2x+ average volume"
        )

    lines.append("\n<i>Not financial advice. Do your own research.</i>")
    text = "\n".join(lines)

    return _send(token, chat_id, text)


def sendNewsMessage(movers: list[NewsMover]) -> bool:
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
