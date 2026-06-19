from __future__ import annotations

import logging
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

from .config import PREDICTION_HORIZON

log = logging.getLogger(__name__)


class TargetClass(IntEnum):
    DOWN = 0
    FLAT = 1
    UP = 2


DEFAULT_THRESHOLDS = {
    1: (0.005, -0.005),
    5: (0.015, -0.015),
    10: (0.025, -0.025),
    21: (0.04, -0.04),
}


def get_thresholds(horizon: int) -> tuple[float, float]:
    if horizon in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS[horizon]
    up = 0.003 * horizon
    return (up, -up)


def build_classification_targets(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
    up_threshold: Optional[float] = None,
    down_threshold: Optional[float] = None,
) -> pd.Series:
    if up_threshold is None or down_threshold is None:
        default_up, default_down = get_thresholds(horizon)
        up_threshold = up_threshold or default_up
        down_threshold = down_threshold or default_down

    future_return = df["Close"].pct_change(horizon).shift(-horizon)

    targets = pd.Series(TargetClass.FLAT, index=df.index, dtype=int)
    targets[future_return >= up_threshold] = TargetClass.UP
    targets[future_return <= down_threshold] = TargetClass.DOWN

    targets[future_return.isna()] = -1

    return targets


def build_binary_targets(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
    threshold: Optional[float] = None,
) -> pd.Series:
    """Binary target: 1 if price goes UP by threshold, 0 otherwise.

    This avoids the 3-class problem where GBM spreads probability too thin.
    """
    if threshold is None:
        threshold, _ = get_thresholds(horizon)

    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    targets = pd.Series(0, index=df.index, dtype=int)
    targets[future_return >= threshold] = 1
    targets[future_return.isna()] = -1
    return targets


def build_binary_targets_down(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
    threshold: Optional[float] = None,
) -> pd.Series:
    """Binary target: 1 if price goes DOWN by threshold, 0 otherwise."""
    if threshold is None:
        _, threshold = get_thresholds(horizon)

    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    targets = pd.Series(0, index=df.index, dtype=int)
    targets[future_return <= threshold] = 1
    targets[future_return.isna()] = -1
    return targets


def build_regression_targets(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
) -> pd.Series:
    return df["Close"].pct_change(horizon).shift(-horizon)


def build_targets(
    df: pd.DataFrame,
    horizon: int = PREDICTION_HORIZON,
    classification: bool = True,
    up_threshold: Optional[float] = None,
    down_threshold: Optional[float] = None,
) -> pd.Series:
    if classification:
        return build_classification_targets(
            df, horizon, up_threshold, down_threshold
        )
    return build_regression_targets(df, horizon)


def get_class_weights(targets: np.ndarray) -> dict[int, float]:
    valid = targets[targets >= 0]
    if len(valid) == 0:
        return {0: 1.0, 1: 1.0, 2: 1.0}

    counts = np.bincount(valid.astype(int), minlength=3)
    total = counts.sum()
    weights = {}
    for cls in range(3):
        if counts[cls] > 0:
            weights[cls] = total / (3.0 * counts[cls])
        else:
            weights[cls] = 1.0
    return weights


def get_binary_class_weights(targets: np.ndarray) -> tuple[float, float]:
    valid = targets[targets >= 0]
    if len(valid) == 0:
        return 1.0, 1.0
    pos = (valid == 1).sum()
    neg = (valid == 0).sum()
    if pos == 0 or neg == 0:
        return 1.0, 1.0
    return neg / pos, 1.0


def target_distribution(targets: np.ndarray) -> dict[str, float]:
    valid = targets[targets >= 0]
    if len(valid) == 0:
        return {"DOWN": 0, "FLAT": 0, "UP": 0, "total": 0}
    counts = np.bincount(valid.astype(int), minlength=3)
    total = len(valid)
    return {
        "DOWN": counts[0] / total,
        "FLAT": counts[1] / total,
        "UP": counts[2] / total,
        "total": total,
    }
