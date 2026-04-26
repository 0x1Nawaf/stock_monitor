# Stock Monitor

LSTM-powered stock analysis. Predicts 5-day price movement for your watchlist.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 monitor.py                # full watchlist
python3 monitor.py AAPL TSLA      # specific tickers
python3 monitor.py --json         # JSON output
python3 monitor.py --retrain      # force retrain models
python3 monitor.py -v             # verbose
```

## Watchlist

Edit `watchlist.txt` -- one ticker per line, `#` for comments.

## Notes

- First run trains models (~10s per ticker). Subsequent runs load from cache.
- Models auto-retrain after 7 days. Use `--retrain` to force it.
- Not financial advice.
