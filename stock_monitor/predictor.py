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


def _model_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker.upper()}.pt"


def _scaler_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker.upper()}_scaler.npz"


def _is_fresh(ticker: str) -> bool:
    path = _model_path(ticker)
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < MODEL_MAX_AGE_DAYS * 86400


def _save_scaler(ticker: str, scaler: StandardScaler) -> None:
    np.savez(
        _scaler_path(ticker),
        mean=scaler.mean_,
        scale=scaler.scale_,
        n_features=np.array([scaler.n_features_in_]),
    )


def _load_scaler(ticker: str) -> Optional[StandardScaler]:
    path = _scaler_path(ticker)
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
    xs, ys = [], []
    for i in range(len(features) - seq_len):
        idx = i + seq_len - 1
        if idx >= len(targets) or np.isnan(targets[idx]):
            continue
        xs.append(features[i : i + seq_len])
        ys.append(targets[idx])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _load_model(
    ticker: str,
    input_size: int,
) -> tuple[Optional[StockLSTM], Optional[StandardScaler]]:
    path = _model_path(ticker)
    if not path.exists():
        return None, None
    scaler = _load_scaler(ticker)
    if scaler is None:
        return None, None
    model = StockLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
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
) -> tuple[StockLSTM, StandardScaler]:
    input_size = features.shape[1]

    if not force and _is_fresh(ticker):
        model, scaler = _load_model(ticker, input_size)
        if model is not None and scaler is not None:
            log.info("Loaded cached model for %s", ticker)
            return model, scaler

    log.info("Training model for %s (%d samples, %d features)", ticker, len(features), input_size)

    X = np.asarray(features)

    if not np.isfinite(X).all():
        log.warning("Features Are Infinte Fixing it...")
        
        finite_vals = X[np.isfinite(X)]
        vmin = np.nanpercentile(finite_vals, 1)
        vmax = np.nanpercentile(finite_vals, 99)
        inf_mask = ~np.isfinite(X)
        X[inf_mask] = np.sign(X[inf_mask]) * vmax
        X = np.clip(X, vmin, vmax)
     
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        log.warning("Fixed !")

    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)


    X, y = _build_sequences(scaled, targets, SEQUENCE_LENGTH)
    if len(X) < BATCH_SIZE * 2:
        raise ValueError(f"Insufficient sequences for {ticker}: got {len(X)}")

    split = int(len(X) * 0.8)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _model_path(ticker))
    _save_scaler(ticker, scaler)

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
) -> PredictionResult:
    model.eval()

    lookback = min(10, len(features) - SEQUENCE_LENGTH)
    windows: list[np.ndarray] = [features[-SEQUENCE_LENGTH:]]
    for offset in range(1, lookback + 1):
        w = features[-(SEQUENCE_LENGTH + offset) : -offset]
        if len(w) >= SEQUENCE_LENGTH:
            windows.append(w[-SEQUENCE_LENGTH:])

    scaled_batch = np.stack([scaler.transform(w) for w in windows]).astype(np.float32)
    x = torch.from_numpy(scaled_batch)

    with torch.no_grad():
        preds = model(x).cpu().numpy().tolist()

    primary = preds[0]
    confidence = _compute_confidence(primary, preds[1:])

    path = _model_path(ticker)
    age = (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 0.0

    return PredictionResult(
        predicted_return=primary,
        confidence=confidence,
        model_age_days=round(age, 1),
    )
