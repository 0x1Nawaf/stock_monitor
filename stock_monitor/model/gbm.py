from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import (
    MODELS_DIR,
    MODEL_MAX_AGE_DAYS,
    GBM_PARAMS,
    GBM_NUM_BOOST_ROUND,
    GBM_EARLY_STOPPING,
    N_CV_SPLITS,
)

log = logging.getLogger(__name__)


@dataclass
class GBMPrediction:
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    model_age_days: float
    prob_up: float = 0.0
    prob_down: float = 0.0


def _model_path(ticker: str, models_dir: Path, suffix: str = "") -> Path:
    return models_dir / f"{ticker.upper()}_gbm{suffix}.txt"


def _is_fresh(ticker: str, models_dir: Path, max_age_days: int) -> bool:
    path_up = _model_path(ticker, models_dir, "_up")
    path_down = _model_path(ticker, models_dir, "_down")
    if not path_up.exists() or not path_down.exists():
        return False
    age = time.time() - min(path_up.stat().st_mtime, path_down.stat().st_mtime)
    return age < max_age_days * 86400


def _time_series_split(n_samples: int, n_splits: int, min_train: int = 200):
    """Expanding window time-series splits for proper temporal validation."""
    splits = []
    fold_size = max(50, (n_samples - min_train) // n_splits)

    for i in range(n_splits):
        val_end = n_samples - i * fold_size
        val_start = val_end - fold_size
        if val_start < min_train:
            break
        splits.append((slice(0, val_start), slice(val_start, val_end)))

    return splits[::-1]


def train_gbm(
    ticker: str,
    features: np.ndarray,
    targets_up: np.ndarray,
    targets_down: np.ndarray,
    force: bool = False,
    models_dir: Path = MODELS_DIR,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
    class_weights: Optional[dict[int, float]] = None,
) -> tuple[object, object]:
    import lightgbm as lgb

    if not force and _is_fresh(ticker, models_dir, max_age_days):
        model_up = _load_gbm(ticker, models_dir, "_up")
        model_down = _load_gbm(ticker, models_dir, "_down")
        if model_up is not None and model_down is not None:
            log.info("Loaded cached GBM models for %s", ticker)
            return model_up, model_down

    log.info(
        "Training dual GBM for %s (%d samples, %d features)",
        ticker, len(features), features.shape[1],
    )

    model_up = _train_single_gbm(ticker, features, targets_up, models_dir, "_up")
    model_down = _train_single_gbm(ticker, features, targets_down, models_dir, "_down")

    return model_up, model_down


def _train_single_gbm(
    ticker: str,
    features: np.ndarray,
    targets: np.ndarray,
    models_dir: Path,
    suffix: str,
) -> object:
    import lightgbm as lgb

    valid_mask = targets >= 0
    X = features[valid_mask]
    y = targets[valid_mask].astype(int)

    splits = _time_series_split(len(X), N_CV_SPLITS)
    if not splits:
        split_idx = int(len(X) * 0.8)
        splits = [(slice(0, split_idx), slice(split_idx, len(X)))]

    best_model = None
    best_auc = 0.0

    for train_sl, val_sl in splits:
        X_train, X_val = X[train_sl], X[val_sl]
        y_train, y_val = y[train_sl], y[val_sl]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=GBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            GBM_PARAMS,
            train_data,
            num_boost_round=GBM_NUM_BOOST_ROUND,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        val_preds = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, val_preds)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_model = model

    if best_model is None:
        X_train = X[:int(len(X) * 0.85)]
        y_train = y[:int(len(X) * 0.85)]
        X_val = X[int(len(X) * 0.85):]
        y_val = y[int(len(X) * 0.85):]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=GBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        best_model = lgb.train(
            GBM_PARAMS, train_data, num_boost_round=GBM_NUM_BOOST_ROUND,
            valid_sets=[val_data], callbacks=callbacks,
        )

    final_model = _retrain_on_full(X, y, best_model.best_iteration)

    models_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(_model_path(ticker, models_dir, suffix)))
    log.info(
        "GBM%s trained for %s: %d rounds, val_AUC=%.3f",
        suffix, ticker, best_model.best_iteration, best_auc,
    )

    return final_model


def _retrain_on_full(X: np.ndarray, y: np.ndarray, num_rounds: int) -> object:
    """Retrain on full data using the best number of rounds found during CV."""
    import lightgbm as lgb

    params = dict(GBM_PARAMS)
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        params, train_data,
        num_boost_round=max(num_rounds, 100),
    )
    return model


def _load_gbm(ticker: str, models_dir: Path, suffix: str = "") -> Optional[object]:
    import lightgbm as lgb

    path = _model_path(ticker, models_dir, suffix)
    if not path.exists():
        return None
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as exc:
        log.warning("Failed to load GBM%s for %s: %s", suffix, ticker, exc)
        return None


def predict_gbm(
    ticker: str,
    model_up: object,
    model_down: object,
    features: np.ndarray,
    models_dir: Path = MODELS_DIR,
) -> GBMPrediction:
    recent = features[-1:] if features.ndim == 2 else features.reshape(1, -1)

    prob_up_raw = float(model_up.predict(recent)[0])
    prob_down_raw = float(model_down.predict(recent)[0])

    prob_up = np.clip(prob_up_raw, 0.01, 0.99)
    prob_down = np.clip(prob_down_raw, 0.01, 0.99)

    prob_flat = max(0.01, 1.0 - prob_up - prob_down)
    total = prob_up + prob_down + prob_flat
    prob_up /= total
    prob_down /= total
    prob_flat /= total

    probs = np.array([prob_down, prob_flat, prob_up])
    predicted_class = int(np.argmax(probs))

    directional_spread = abs(prob_up - prob_down)
    max_prob = max(prob_up, prob_down, prob_flat)
    confidence = 0.5 * directional_spread + 0.5 * max_prob

    path = _model_path(ticker, models_dir, "_up")
    age = (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 0.0

    return GBMPrediction(
        predicted_class=predicted_class,
        probabilities=probs,
        confidence=round(confidence, 3),
        model_age_days=round(age, 1),
        prob_up=round(prob_up, 4),
        prob_down=round(prob_down, 4),
    )


def predict_gbm_batch(
    model_up: object,
    model_down: object,
    features: np.ndarray,
) -> np.ndarray:
    """Batch prediction returning [n_samples, 3] probability array."""
    prob_up = model_up.predict(features)
    prob_down = model_down.predict(features)
    prob_flat = np.clip(1.0 - prob_up - prob_down, 0.01, None)
    total = prob_up + prob_down + prob_flat
    return np.column_stack([
        prob_down / total,
        prob_flat / total,
        prob_up / total,
    ])
