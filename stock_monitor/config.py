from __future__ import annotations

from enum import Enum
from pathlib import Path


class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    LEAN_BUY = "LEAN BUY"
    HOLD = "HOLD"
    LEAN_SELL = "LEAN SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
WATCHLIST_PATH = BASE_DIR / "watchlist.txt"

HISTORY_PERIOD = "2y"
MIN_DATA_POINTS = 200

SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
PREDICTION_HORIZON = 5
MODEL_MAX_AGE_DAYS = 7

SIGNAL_THRESHOLDS: dict[Signal, float] = {
    Signal.STRONG_BUY: 0.04,
    Signal.BUY: 0.02,
    Signal.LEAN_BUY: 0.005,
    Signal.HOLD: 0.0,
    Signal.LEAN_SELL: -0.005,
    Signal.SELL: -0.02,
    Signal.STRONG_SELL: -0.04,
}

MAX_REPORT_FILES = 50
