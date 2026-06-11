from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import MODELS_DIR, MODEL_MAX_AGE_DAYS

log = logging.getLogger(__name__)


@dataclass
class GBMPrediction:
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    model_age_days: float


def _model_path(ticker: str, models_dir: Path) -> Path:
    return models_dir / f"{ticker.upper()}_gbm.txt"


def _is_fresh(ticker: str, models_dir: Path, max_age_days: int) -> bool:
    path = _model_path(ticker, models_dir)
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_days * 86400


def train_gbm(
    ticker: str,
    features: np.ndarray,
    targets: np.ndarray,
    force: bool = False,
    models_dir: Path = MODELS_DIR,
    max_age_days: int = MODEL_MAX_AGE_DAYS,
    class_weights: Optional[dict[int, float]] = None,
) -> object:
    import lightgbm as lgb

    if not force and _is_fresh(ticker, models_dir, max_age_days):
        model = _load_gbm(ticker, models_dir)
        if model is not None:
            log.info("Loaded cached GBM for %s", ticker)
            return model

    log.info("Training GBM for %s (%d samples, %d features)", ticker, len(features), features.shape[1])

    valid_mask = targets >= 0
    X = features[valid_mask]
    y = targets[valid_mask].astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if class_weights is not None:
        sample_weights = np.array([class_weights.get(int(c), 1.0) for c in y_train])
    else:
        sample_weights = None

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(_model_path(ticker, models_dir)))
    log.info("GBM trained for %s: %d rounds", ticker, model.best_iteration)

    return model


def _load_gbm(ticker: str, models_dir: Path) -> Optional[object]:
    import lightgbm as lgb

    path = _model_path(ticker, models_dir)
    if not path.exists():
        return None
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as exc:
        log.warning("Failed to load GBM for %s: %s", ticker, exc)
        return None


def predict_gbm(
    ticker: str,
    model: object,
    features: np.ndarray,
    models_dir: Path = MODELS_DIR,
) -> GBMPrediction:
    import lightgbm as lgb

    recent = features[-1:] if features.ndim == 2 else features.reshape(1, -1)

    probs = model.predict(recent)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    path = _model_path(ticker, models_dir)
    age = (time.time() - path.stat().st_mtime) / 86400 if path.exists() else 0.0

    return GBMPrediction(
        predicted_class=predicted_class,
        probabilities=probs,
        confidence=confidence,
        model_age_days=round(age, 1),
    )


def predict_gbm_batch(
    model: object,
    features: np.ndarray,
) -> np.ndarray:
    return model.predict(features)


def train_gbm_cross_sectional(
    features: np.ndarray,
    targets: np.ndarray,
    models_dir: Path = MODELS_DIR,
    class_weights: Optional[dict[int, float]] = None,
) -> object:
    import lightgbm as lgb

    valid_mask = targets >= 0
    X = features[valid_mask]
    y = targets[valid_mask].astype(int)

    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if class_weights is not None:
        sample_weights = np.array([class_weights.get(int(c), 1.0) for c in y_train])
    else:
        sample_weights = None

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 0.2,
        "lambda_l2": 0.2,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    path = models_dir / "cross_sectional_gbm.txt"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    log.info("Cross-sectional GBM trained: %d rounds on %d samples", model.best_iteration, len(X_train))

    return model
