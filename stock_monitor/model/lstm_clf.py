from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import (
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    MAX_EPOCHS,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    MODELS_DIR,
    MODEL_MAX_AGE_DAYS,
)

log = logging.getLogger(__name__)


class StockLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.head(context)


@dataclass
class LSTMPrediction:
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    model_age_days: float


def _model_path(ticker: str, models_dir: Path) -> Path:
    return models_dir / f"{ticker.upper()}_lstm_clf.pt"


def _is_fresh(ticker: str, models_dir: Path, max_age_days: int) -> bool:
    path = _model_path(ticker, models_dir)
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_days * 86400


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(features) - seq_len
    if n <= 0:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)

    raw_windows = sliding_window_view(features, window_shape=seq_len, axis=0)
    windows = np.swapaxes(raw_windows[:n], 1, 2)
    target_indices = np.arange(seq_len - 1, seq_len - 1 + n)
    valid = target_indices < len(targets)
    valid[valid] &= targets[target_indices[valid]] >= 0

    xs = windows[valid].astype(np.float32)
    ys = targets[target_indices[valid]].astype(np.int64)
    return xs, ys


def train_lstm_classifier(
    ticker: str,
    features: np.ndarray,
    targets: np.ndarray,
    force: bool = False,
    models_dir: Path = MODELS_DIR,
    seq_len: int = SEQUENCE_LENGTH,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
    class_weights: Optional[dict[int, float]] = None,
) -> StockLSTMClassifier:
    input_size = features.shape[1]

    if not force and _is_fresh(ticker, models_dir, max_age_days):
        model = _load_model(ticker, input_size, models_dir)
        if model is not None:
            log.info("Loaded cached LSTM classifier for %s", ticker)
            return model

    log.info("Training LSTM classifier for %s (%d samples)", ticker, len(features))

    from sklearn.preprocessing import StandardScaler

    valid_mask = targets >= 0
    X = features[valid_mask].astype(np.float64)
    y = targets[valid_mask].astype(np.int64)

    if not np.isfinite(X).all():
        finite_vals = X[np.isfinite(X)]
        if len(finite_vals) == 0:
            raise ValueError(f"No finite feature values for {ticker}")
        vmin = np.nanpercentile(finite_vals, 1)
        vmax = np.nanpercentile(finite_vals, 99)
        X = np.clip(np.nan_to_num(X, nan=0.0, posinf=vmax, neginf=vmin), vmin, vmax)

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = _build_sequences(X_train_scaled, y_train_raw, seq_len)
    X_val_seq, y_val_seq = _build_sequences(X_val_scaled, y_val_raw, seq_len)

    if len(X_train_seq) < BATCH_SIZE or len(X_val_seq) < BATCH_SIZE:
        raise ValueError(f"Insufficient sequences for {ticker}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train_seq),
        torch.from_numpy(y_train_seq),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_seq),
        torch.from_numpy(y_val_seq),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = _get_device()
    model = StockLSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    if class_weights is not None:
        weight_tensor = torch.tensor(
            [class_weights.get(i, 1.0) for i in range(3)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {}

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(bx)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info("LSTM early stop at epoch %d (val_loss=%.4f)", epoch + 1, best_val_loss)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()

    models_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_size": input_size,
    }
    torch.save(save_data, _model_path(ticker, models_dir))

    model.scaler_mean_ = scaler.mean_
    model.scaler_scale_ = scaler.scale_

    return model


def _load_model(
    ticker: str,
    input_size: int,
    models_dir: Path,
) -> Optional[StockLSTMClassifier]:
    path = _model_path(ticker, models_dir)
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = StockLSTMClassifier(
            input_size=data.get("input_size", input_size),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        model.load_state_dict(data["model_state"])
        model.eval()
        model.scaler_mean_ = data["scaler_mean"]
        model.scaler_scale_ = data["scaler_scale"]
        return model
    except Exception as exc:
        log.warning("Failed to load LSTM classifier for %s: %s", ticker, exc)
        return None


def predict_lstm(
    ticker: str,
    model: StockLSTMClassifier,
    features: np.ndarray,
    models_dir: Path = MODELS_DIR,
    seq_len: int = SEQUENCE_LENGTH,
) -> LSTMPrediction:
    from sklearn.preprocessing import StandardScaler

    if not hasattr(model, "scaler_mean_") or model.scaler_mean_ is None:
        path = _model_path(ticker, models_dir)
        if not path.exists():
            return LSTMPrediction(
                predicted_class=1, probabilities=np.array([0.33, 0.34, 0.33]),
                confidence=0.34, model_age_days=0.0,
            )
        data = torch.load(path, map_location="cpu", weights_only=False)
        model.scaler_mean_ = data["scaler_mean"]
        model.scaler_scale_ = data["scaler_scale"]

    scaler = StandardScaler()
    scaler.mean_ = model.scaler_mean_
    scaler.scale_ = model.scaler_scale_
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    device = _get_device()
    model.to(device).eval()

    lookback = min(5, len(features) - seq_len)
    windows = []
    for offset in range(lookback + 1):
        if offset == 0:
            w = features[-seq_len:]
        else:
            w = features[-(seq_len + offset):-offset]
        if len(w) >= seq_len:
            windows.append(w[-seq_len:])

    stacked = np.stack(windows)
    if not np.isfinite(stacked).all():
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    scaled = np.stack([scaler.transform(w) for w in stacked]).astype(np.float32)
    x = torch.from_numpy(scaled).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    avg_probs = probs.mean(axis=0)
    predicted_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_class])

    path = _model_path(ticker, models_dir)
    age = (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 0.0

    return LSTMPrediction(
        predicted_class=predicted_class,
        probabilities=avg_probs,
        confidence=confidence,
        model_age_days=round(age, 1),
    )
