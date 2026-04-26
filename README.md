# Stock Monitor -- LSTM AI Engine

AI-powered stock analysis using LSTM neural networks. Predicts short-term
price movements and generates actionable buy/sell/hold signals for your
watchlist.

## How It Works

1. Fetches 2 years of daily OHLCV data via Yahoo Finance
2. Computes 18 technical features (returns, RSI, MACD, Bollinger Bands,
   moving averages, volume, ATR, stochastic, volatility)
3. Trains a per-ticker LSTM model on 60-day sequences
4. Predicts 5-day forward returns with confidence scoring
5. Generates signals: STRONG BUY / BUY / LEAN BUY / HOLD / LEAN SELL / SELL / STRONG SELL

## Usage

```bash
python3 monitor.py

python3 monitor.py AAPL TSLA NVDA

python3 monitor.py --json

python3 monitor.py --retrain

python3 monitor.py -v
```

## Files

- `monitor.py` -- CLI entry point
- `stock_monitor/` -- core package (config, data, model, predictor, analyzer, report)
- `watchlist.txt` -- tickers to track (one per line, # for comments)
- `models/` -- cached LSTM models (auto-generated)
- `reports/` -- saved analysis reports (auto-generated)

## Setup

```bash
pip install -r requirements.txt
```

## Model Details

- Architecture: 2-layer LSTM (128 hidden units) with fully connected head
- Input: 60-day sequences of 18 normalized technical features
- Output: predicted 5-day forward return
- Training: 80/20 split, Adam optimizer, ReduceLROnPlateau scheduler,
  early stopping (patience=10), gradient clipping
- Confidence: derived from prediction consistency across overlapping windows
- Models cached to disk for 7 days before automatic retraining
- Use `--retrain` to force fresh training

## Disclaimer

Not financial advice. Do your own research.
