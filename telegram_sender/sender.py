import requests
import config


def sendMessage(res):
    bot_token = config.env_config["telegram_bot_token"]
    chat_id = config.env_config["telegram_chat_id"]

    tf_label = "1-day" if getattr(res, "timeframe", "5d") == "1d" else "5-day"
    message = (
        "📈 **Stock Update**\n\n"
        f"✨ **Ticker:** {res.ticker}\n"
        f"💰 **Current Price:** ${res.price}\n"
        f"🔔 **Signal:** {res.signal.value}\n"
        f"🚀 **Change:** {res.change_pct}%\n"
        f"💪 **Confidence:** {res.confidence * 100:.0f}%\n"
        f"📊 **Prediction:** {tf_label}\n"
        "📣 *Stay informed and invest wisely!*"
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }

    response = requests.post(url, data=payload)

    return response.ok


def sendNewsMessage(movers):
    bot_token = config.env_config["telegram_bot_token"]
    chat_id = config.env_config["telegram_chat_id"]

    if not movers:
        message = "📰 **News Scanner**\n\nNo stocks with predicted gain >= 5% found."
    else:
        lines = ["📰 **News Scanner -- Top Movers**\n"]
        for m in movers[:10]:
            lines.append(
                f"🔥 **{m.ticker}**  Predicted: +{m.predicted_gain_pct:.0f}%  "
                f"({m.headline_count} article{'s' if m.headline_count != 1 else ''})"
            )
            if m.top_headlines:
                lines.append(f'  ➤ "{m.top_headlines[0]}"')
        lines.append("\n📣 *Not financial advice.*")
        message = "\n".join(lines)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }

    response = requests.post(url, data=payload)

    return response.ok