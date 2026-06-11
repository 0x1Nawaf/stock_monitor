from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import TimeframeConfig, TIMEFRAME_5D, GBM_PARAMS
from .data import fetch_stock_data
from .features import build_all_features
from .targets import (
    build_classification_targets,
    get_thresholds,
    get_class_weights,
    TargetClass,
)

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    ticker: str
    horizon: int
    total_predictions: int
    directional_accuracy: float
    class_accuracy: float
    precision_up: float
    precision_down: float
    recall_up: float
    recall_down: float
    profit_factor: float
    avg_return_on_signals: float
    predictions: list[dict] = field(default_factory=list, repr=False)


def _non_overlapping_indices(n: int, horizon: int) -> np.ndarray:
    return np.arange(0, n, horizon)


def _train_gbm_fold(
    train_X: np.ndarray, train_y: np.ndarray
) -> object:
    import lightgbm as lgb

    class_weights = get_class_weights(train_y)
    valid_train = train_y >= 0
    X_tr = train_X[valid_train]
    y_tr = train_y[valid_train].astype(int)
    sample_weights = np.array([class_weights.get(int(c), 1.0) for c in y_tr])

    split = int(len(X_tr) * 0.85)
    train_data = lgb.Dataset(X_tr[:split], label=y_tr[:split], weight=sample_weights[:split])
    val_data = lgb.Dataset(X_tr[split:], label=y_tr[split:], reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    return lgb.train(
        GBM_PARAMS, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )


def _train_lstm_fold(
    train_X: np.ndarray,
    train_y: np.ndarray,
    seq_len: int,
) -> Optional[object]:
    from .model.lstm_clf import train_lstm_classifier

    try:
        model = train_lstm_classifier(
            "__backtest__", train_X, train_y,
            force=True, seq_len=seq_len,
        )
        return model
    except Exception as exc:
        log.warning("Backtest LSTM fold failed: %s", exc)
        return None


def _predict_fold(
    model_type: str,
    gbm_model: object,
    lstm_model: Optional[object],
    test_X: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    from .model.ensemble import combine_predictions, DEFAULT_GBM_WEIGHT, DEFAULT_LSTM_WEIGHT
    from .model.gbm import GBMPrediction
    from .model.lstm_clf import predict_lstm, LSTMPrediction

    gbm_probs = gbm_model.predict(test_X)

    if model_type != "ensemble" or lstm_model is None:
        return np.argmax(gbm_probs, axis=1)

    preds = []
    for i in range(len(test_X)):
        gbm_pred = GBMPrediction(
            predicted_class=int(np.argmax(gbm_probs[i])),
            probabilities=gbm_probs[i],
            confidence=float(gbm_probs[i].max()),
            model_age_days=0.0,
        )

        lstm_pred = None
        try:
            lstm_pred_result = predict_lstm(
                "__backtest__", lstm_model, test_X[:i + 1],
                seq_len=seq_len,
            )
            lstm_pred = lstm_pred_result
        except Exception:
            pass

        ensemble = combine_predictions(gbm_pred, lstm_pred)
        preds.append(ensemble.predicted_class)

    return np.array(preds)


def walk_forward_backtest(
    ticker: str,
    timeframe: TimeframeConfig = TIMEFRAME_5D,
    train_years: int = 3,
    test_months: int = 3,
    market_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    model_type: str = "ensemble",
) -> Optional[BacktestResult]:
    df = fetch_stock_data(ticker)
    if df is None:
        log.warning("Backtest %s: no data", ticker)
        return None

    features_df = build_all_features(df, market_df, vix_df)
    horizon = timeframe.horizon
    up_thresh, down_thresh = get_thresholds(horizon)
    targets = build_classification_targets(df, horizon, up_thresh, down_thresh)
    targets = targets.reindex(features_df.index)

    features_arr = features_df.values
    targets_arr = targets.values

    train_size = train_years * 252
    test_size = test_months * 21

    if len(features_arr) < train_size + test_size:
        log.warning("Backtest %s: insufficient data (%d rows)", ticker, len(features_arr))
        return None

    all_preds = []
    all_actuals = []
    all_returns = []

    future_returns = df["Close"].pct_change(horizon).shift(-horizon).reindex(features_df.index)

    start = train_size
    while start + test_size <= len(features_arr):
        train_X = features_arr[:start]
        train_y = targets_arr[:start]

        test_end = min(start + test_size, len(features_arr))
        test_indices = _non_overlapping_indices(test_end - start, horizon)
        test_indices = test_indices + start
        test_indices = test_indices[test_indices < test_end]

        if len(test_indices) == 0:
            start += test_size
            continue

        test_X = features_arr[test_indices]
        test_y = targets_arr[test_indices]
        test_rets = future_returns.iloc[test_indices].values

        valid_test = test_y >= 0
        if valid_test.sum() == 0:
            start += test_size
            continue

        test_X = test_X[valid_test]
        test_y = test_y[valid_test]
        test_rets = test_rets[valid_test]

        gbm_model = _train_gbm_fold(train_X, train_y)

        lstm_model = None
        if model_type == "ensemble":
            lstm_model = _train_lstm_fold(
                train_X, train_y, seq_len=timeframe.sequence_length,
            )

        preds = _predict_fold(
            model_type, gbm_model, lstm_model,
            test_X, seq_len=timeframe.sequence_length,
        )

        all_preds.extend(preds.tolist())
        all_actuals.extend(test_y.astype(int).tolist())
        all_returns.extend(test_rets.tolist())

        start += test_size

    if len(all_preds) == 0:
        return None

    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)
    returns_arr = np.array(all_returns)

    class_acc = float(np.mean(preds_arr == actuals_arr))

    pred_up = preds_arr == TargetClass.UP
    pred_down = preds_arr == TargetClass.DOWN
    actual_up = actuals_arr == TargetClass.UP
    actual_down = actuals_arr == TargetClass.DOWN

    pred_direction = np.where(pred_up, 1, np.where(pred_down, -1, 0))
    actual_direction = np.where(actual_up, 1, np.where(actual_down, -1, 0))
    directional_mask = pred_direction != 0
    if directional_mask.sum() > 0:
        dir_acc = float(np.mean(pred_direction[directional_mask] == actual_direction[directional_mask]))
    else:
        dir_acc = 0.0

    prec_up = float(actual_up[pred_up].mean()) if pred_up.sum() > 0 else 0.0
    prec_down = float(actual_down[pred_down].mean()) if pred_down.sum() > 0 else 0.0
    rec_up = float(pred_up[actual_up].mean()) if actual_up.sum() > 0 else 0.0
    rec_down = float(pred_down[actual_down].mean()) if actual_down.sum() > 0 else 0.0

    signal_returns = np.where(pred_up, returns_arr, np.where(pred_down, -returns_arr, 0.0))
    active_signals = pred_direction != 0
    avg_signal_return = float(signal_returns[active_signals].mean()) if active_signals.sum() > 0 else 0.0

    gains = signal_returns[signal_returns > 0].sum()
    losses = abs(signal_returns[signal_returns < 0].sum())
    profit_factor = gains / (losses + 1e-8)

    return BacktestResult(
        ticker=ticker,
        horizon=horizon,
        total_predictions=len(preds_arr),
        directional_accuracy=round(dir_acc, 4),
        class_accuracy=round(class_acc, 4),
        precision_up=round(prec_up, 4),
        precision_down=round(prec_down, 4),
        recall_up=round(rec_up, 4),
        recall_down=round(rec_down, 4),
        profit_factor=round(profit_factor, 4),
        avg_return_on_signals=round(avg_signal_return, 4),
    )


def format_backtest_report(results: list[BacktestResult]) -> str:
    lines = [
        "BACKTEST REPORT -- Walk-Forward Evaluation",
        "=" * 72,
        "",
        f"{'Ticker':<8} {'Dir.Acc':>8} {'Cls.Acc':>8} {'Prec.UP':>8} "
        f"{'Prec.DN':>8} {'PF':>6} {'Avg.Ret':>8} {'N':>5}",
        "-" * 72,
    ]

    for r in results:
        lines.append(
            f"{r.ticker:<8} {r.directional_accuracy:>7.1%} {r.class_accuracy:>7.1%} "
            f"{r.precision_up:>7.1%} {r.precision_down:>7.1%} "
            f"{r.profit_factor:>6.2f} {r.avg_return_on_signals:>+7.2%} {r.total_predictions:>5}"
        )

    lines.append("-" * 72)

    if results:
        avg_dir = np.mean([r.directional_accuracy for r in results])
        avg_cls = np.mean([r.class_accuracy for r in results])
        avg_pf = np.mean([r.profit_factor for r in results])
        avg_ret = np.mean([r.avg_return_on_signals for r in results])
        lines.append(
            f"{'AVERAGE':<8} {avg_dir:>7.1%} {avg_cls:>7.1%} "
            f"{'':>8} {'':>8} "
            f"{avg_pf:>6.2f} {avg_ret:>+7.2%}"
        )

    lines.append("")
    lines.append("Walk-forward: train 3y, test 3mo rolling, non-overlapping predictions (ensemble)")
    return "\n".join(lines)
