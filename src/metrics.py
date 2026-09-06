"""Probabilistic and point metrics.

Primary metric is the weighted quantile loss (a discrete CRPS estimator over
the 9 quantiles TimesFM-3 emits). Plain RMSE is deliberately absent as a
headline: discharge is heavily right-skewed and RMSE would be dominated by a
handful of flood weeks.
"""
from __future__ import annotations

import numpy as np

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def pinball(y: np.ndarray, pred_q: np.ndarray,
            quantiles: np.ndarray = QUANTILES) -> np.ndarray:
    """Per-observation mean pinball loss across quantile levels.

    y       : (n,)
    pred_q  : (n, n_quantiles)
    """
    y = np.asarray(y, float)[:, None]
    pred_q = np.asarray(pred_q, float)
    q = np.asarray(quantiles, float)[None, :]
    diff = y - pred_q
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return 2.0 * loss.mean(axis=1)


def crps_from_quantiles(y, pred_q, quantiles=QUANTILES) -> float:
    """Discrete CRPS estimator = mean pinball loss over the quantile grid."""
    return float(np.nanmean(pinball(y, pred_q, quantiles)))


def wql(y, pred_q, quantiles=QUANTILES) -> float:
    """Weighted quantile loss: total pinball normalised by total |y|.

    Scale-free, so it can be compared across reservoirs and seasons.
    """
    y = np.asarray(y, float)
    num = np.nansum(pinball(y, pred_q, quantiles))
    den = np.nansum(np.abs(y))
    return float(num / den) if den > 0 else np.nan


def mae(y, yhat) -> float:
    return float(np.nanmean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def mase(y, yhat, y_train_seasonal_naive_mae: float) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE."""
    if not np.isfinite(y_train_seasonal_naive_mae) or y_train_seasonal_naive_mae <= 0:
        return np.nan
    return mae(y, yhat) / y_train_seasonal_naive_mae


def seasonal_naive_scale(train: np.ndarray, season: int = 52) -> float:
    """In-sample MAE of the seasonal-naive forecast, the MASE denominator."""
    t = np.asarray(train, float)
    if len(t) <= season:
        return np.nan
    return float(np.nanmean(np.abs(t[season:] - t[:-season])))


def skill_score(metric_model: float, metric_reference: float) -> float:
    """1 - model/reference. Positive means the model beats the reference.

    This is the headline number: reference is climatology.
    """
    if not np.isfinite(metric_reference) or metric_reference <= 0:
        return np.nan
    return float(1.0 - metric_model / metric_reference)
