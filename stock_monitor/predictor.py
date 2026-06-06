from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .config import (
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
from .model import StockLSTM

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_return: float
    confidence: float
    model_age_days: float


def _model_path(ticker: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"{ticker.upper()}.pt"


def _scaler_path(ticker: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"{ticker.upper()}_scaler.npz"


def _is_fresh(
    ticker: str,
    models_dir: Path = MODELS_DIR,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
) -> bool:
    path = _model_path(ticker, models_dir)
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_days * 86400


def _save_scaler(ticker: str, scaler: StandardScaler, models_dir: Path = MODELS_DIR) -> None:
    np.savez(
        _scaler_path(ticker, models_dir),
        mean=scaler.mean_,
        scale=scaler.scale_,
        n_features=np.array([scaler.n_features_in_]),
    )


def _load_scaler(ticker: str, models_dir: Path = MODELS_DIR) -> Optional[StandardScaler]:
    path = _scaler_path(ticker, models_dir)
    if not path.exists():
        return None
    data = np.load(path)
    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(data["n_features"][0])
    return scaler


def _build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(features) - seq_len
    if n <= 0:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)

    from numpy.lib.stride_tricks import sliding_window_view
    raw_windows = sliding_window_view(features, window_shape=seq_len, axis=0)
    windows = np.swapaxes(raw_windows[:n], 1, 2)
    target_indices = np.arange(seq_len - 1, seq_len - 1 + n)
    valid = target_indices < len(targets)
    valid[valid] &= ~np.isnan(targets[target_indices[valid]])

    xs = windows[valid].astype(np.float32)
    ys = targets[target_indices[valid]].astype(np.float32)
    return xs, ys


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(
    ticker: str,
    input_size: int,
    models_dir: Path = MODELS_DIR,
) -> tuple[Optional[StockLSTM], Optional[StandardScaler]]:
    path = _model_path(ticker, models_dir)
    if not path.exists():
        return None, None
    scaler = _load_scaler(ticker, models_dir)
    if scaler is None:
        return None, None
    device = _get_device()
    model = StockLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model, scaler


def _train_loop(
    model: StockLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_size: int,
    val_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(bx)
        val_loss /= val_size

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info("Early stop at epoch %d (val_loss=%.6f)", epoch + 1, best_val_loss)
                break

    return best_state


def train_model(
    ticker: str,
    features: np.ndarray,
    targets: np.ndarray,
    force: bool = False,
    models_dir: Path = MODELS_DIR,
    seq_len: int = SEQUENCE_LENGTH,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
) -> tuple[StockLSTM, StandardScaler]:
    input_size = features.shape[1]

    if not force and _is_fresh(ticker, models_dir, max_age_days):
        model, scaler = _load_model(ticker, input_size, models_dir)
        if model is not None and scaler is not None:
            log.info("Loaded cached model for %s", ticker)
            return model, scaler

    log.info("Training model for %s (%d samples, %d features)", ticker, len(features), input_size)

    X = np.asarray(features, dtype=np.float64)

    if not np.isfinite(X).all():
        log.warning("%s: non-finite feature values detected, clamping to percentile range", ticker)
        finite_vals = X[np.isfinite(X)]
        if len(finite_vals) == 0:
            raise ValueError(f"No finite feature values for {ticker}")
        vmin = np.nanpercentile(finite_vals, 1)
        vmax = np.nanpercentile(finite_vals, 99)
        pos_inf = np.isposinf(X)
        neg_inf = np.isneginf(X)
        nan_mask = np.isnan(X)
        X[pos_inf] = vmax
        X[neg_inf] = vmin
        X[nan_mask] = 0.0
        X = np.clip(X, vmin, vmax)

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_val_raw = targets[:split_idx], targets[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = _build_sequences(X_train_scaled, y_train_raw, seq_len)
    X_val_seq, y_val_seq = _build_sequences(X_val_scaled, y_val_raw, seq_len)

    total_sequences = len(X_train_seq) + len(X_val_seq)
    if total_sequences < BATCH_SIZE * 2:
        raise ValueError(f"Insufficient sequences for {ticker}: got {total_sequences}")

    train_ds = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    val_ds = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = _get_device()
    model = StockLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    best_state = _train_loop(
        model, train_loader, val_loader, len(train_ds), len(val_ds), device
    )

    if best_state:
        model.load_state_dict(best_state)
    model.cpu().eval()

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _model_path(ticker, models_dir))
    _save_scaler(ticker, scaler, models_dir)

    return model, scaler


def _compute_confidence(primary: float, window_preds: list[float]) -> float:
    if not window_preds:
        return 0.5
    all_preds = [primary] + window_preds
    std = float(np.std(all_preds))
    mean_abs = abs(float(np.mean(all_preds))) + 1e-8
    consistency = max(0.0, 1.0 - std / mean_abs)
    same_dir = sum(1 for p in all_preds if np.sign(p) == np.sign(primary))
    agreement = same_dir / len(all_preds)
    return round(float(np.clip(0.4 * consistency + 0.6 * agreement, 0.1, 0.95)), 3)


def predict(
    ticker: str,
    model: StockLSTM,
    scaler: StandardScaler,
    features: np.ndarray,
    models_dir: Path = MODELS_DIR,
    seq_len: int = SEQUENCE_LENGTH,
) -> PredictionResult:
    device = _get_device()
    model.to(device).eval()

    lookback = min(10, len(features) - seq_len)
    windows: list[np.ndarray] = [features[-seq_len:]]
    for offset in range(1, lookback + 1):
        w = features[-(seq_len + offset) : -offset]
        if len(w) >= seq_len:
            windows.append(w[-seq_len:])

    stacked = np.stack(windows)
    if not np.isfinite(stacked).all():
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_batch = np.stack([scaler.transform(w) for w in stacked]).astype(np.float32)
    x = torch.from_numpy(scaled_batch).to(device)

    with torch.no_grad():
        preds = model(x).cpu().numpy().tolist()

    primary = preds[0]
    confidence = _compute_confidence(primary, preds[1:])

    path = _model_path(ticker, models_dir)
    age = (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 0.0

    return PredictionResult(
        predicted_return=primary,
        confidence=confidence,
        model_age_days=round(age, 1),
    )
