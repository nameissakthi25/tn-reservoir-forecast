"""Baselines, built and scored BEFORE TimesFM so the comparison is honest.

  * SmoothedClimatology -- week-of-year quantiles, Fourier-smoothed. This is
    the bar. Raw week-of-year empirical quantiles from ~30 years are noisy and
    would make an artificially weak reference.
  * SeasonalNaive       -- same week last year.
  * Persistence         -- last observed value, carried flat.

All three are probabilistic: the two naive methods get quantiles from their own
historical error distribution at each horizon, so they are scored on the same
CRPS/WQL footing as TimesFM rather than being handicapped as point forecasts.

Smoothing and quantile estimation happen in log1p space and are mapped back by
exponentiation. Quantiles commute with monotone transforms exactly, so this is
lossless -- unlike back-transforming a mean, which is not done anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import QUANTILES

SEASON = 52


def _woy(idx: pd.DatetimeIndex) -> np.ndarray:
    w = idx.isocalendar().week.values.astype(int)
    return np.clip(w, 1, 52)


def _fourier_design(woy: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    ang = 2 * np.pi * (woy - 1) / 52.0
    cols = [np.ones_like(ang, dtype=float)]
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(k * ang), np.cos(k * ang)]
    return np.column_stack(cols)


class SmoothedClimatology:
    """Fourier-smoothed week-of-year quantile climatology, fitted in log space.

    Fitted ONLY on training years; the backtest refits per origin so the
    reference is always out-of-sample.
    """

    def __init__(self, n_harmonics: int = 3, quantiles=QUANTILES):
        self.n_harmonics = n_harmonics
        self.quantiles = np.asarray(quantiles, float)
        self.coef_: np.ndarray | None = None
        self.n_years_: float = np.nan

    def fit(self, dates: pd.DatetimeIndex, y: np.ndarray) -> "SmoothedClimatology":
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        dates, y = dates[ok], y[ok]
        z = np.log1p(np.clip(y, 0, None))
        woy = _woy(dates)
        self.n_years_ = len(y) / 52.0

        # empirical quantile per week-of-year, then smooth each level over woy
        grid = np.arange(1, 53)
        raw = np.full((52, len(self.quantiles)), np.nan)
        for i, w in enumerate(grid):
            vals = z[woy == w]
            if len(vals):
                raw[i] = np.quantile(vals, self.quantiles)
        # fill any empty week from neighbours before smoothing
        raw = pd.DataFrame(raw).interpolate(limit_direction="both").values

        X = _fourier_design(grid, self.n_harmonics)
        self.coef_ = np.linalg.lstsq(X, raw, rcond=None)[0]
        return self

    def predict_quantiles(self, dates: pd.DatetimeIndex) -> np.ndarray:
        X = _fourier_design(_woy(dates), self.n_harmonics)
        z = X @ self.coef_
        z = np.sort(z, axis=1)               # enforce monotone quantiles
        return np.expm1(z).clip(min=0.0)     # exact under a monotone transform

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        med = int(np.argmin(np.abs(self.quantiles - 0.5)))
        return self.predict_quantiles(dates)[:, med]


class _ErrorQuantileMixin:
    """Turns a point forecast into a probabilistic one via historical errors.

    Error quantiles are estimated SEPARATELY FOR EACH LEAD TIME, in log space,
    on the training data. This matters: a persistence forecast's error grows
    with lead time, so reusing 1-step error quantiles at 13 steps produces
    absurdly overconfident intervals (measured P10-P90 coverage of 0.18 against
    a nominal 0.80) and unfairly inflates any competitor's CRPS advantage.
    `err_q_` is therefore (max_horizon, n_quantiles), not a single vector.
    """

    def _set_errors(self, rows: list[np.ndarray]) -> None:
        self.err_q_ = np.vstack(rows)

    def _errors_for_lag(self, y: np.ndarray, lag: int) -> np.ndarray:
        if len(y) <= lag:
            return np.zeros(len(self.quantiles))
        z = np.log1p(np.clip(y, 0, None))
        err = (z[lag:] - z[:-lag])
        err = err[np.isfinite(err)]
        return (np.quantile(err, self.quantiles) if len(err) > 10
                else np.zeros(len(self.quantiles)))

    def _apply(self, point: np.ndarray) -> np.ndarray:
        p = np.log1p(np.clip(np.asarray(point, float), 0, None))[:, None]
        eq = self.err_q_[: len(p)]
        if len(eq) < len(p):                     # pad with the longest lead fit
            eq = np.vstack([eq, np.repeat(eq[-1:], len(p) - len(eq), axis=0)])
        return np.expm1(np.sort(p + eq, axis=1)).clip(min=0.0)


class SeasonalNaive(_ErrorQuantileMixin):
    """Same week last year."""

    def __init__(self, quantiles=QUANTILES, season: int = SEASON):
        self.quantiles = np.asarray(quantiles, float)
        self.season = season
        self.err_q_ = np.zeros((1, len(self.quantiles)))

    def fit(self, y_train: np.ndarray, max_horizon: int = 13) -> "SeasonalNaive":
        # forecast for lead t uses the observation at lag (season - t)
        y = np.asarray(y_train, float)
        self._set_errors([self._errors_for_lag(y, max(self.season - t, 1))
                          for t in range(1, max_horizon + 1)])
        return self

    def predict_point(self, history: np.ndarray, horizon: int) -> np.ndarray:
        h = np.asarray(history, float)
        # step t ahead maps to the same week one season back
        out = [h[-self.season + t] if len(h) >= self.season - t else np.nan
               for t in range(horizon)]
        return np.asarray(out, float)

    def predict_quantiles(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return self._apply(self.predict_point(history, horizon))


class Persistence(_ErrorQuantileMixin):
    """Last observed value, held flat."""

    def __init__(self, quantiles=QUANTILES):
        self.quantiles = np.asarray(quantiles, float)
        self.err_q_ = np.zeros((1, len(self.quantiles)))

    def fit(self, y_train: np.ndarray, max_horizon: int = 13) -> "Persistence":
        # lead t is a lag-t forecast, so error spread grows with t
        y = np.asarray(y_train, float)
        self._set_errors([self._errors_for_lag(y, t)
                          for t in range(1, max_horizon + 1)])
        return self

    def predict_point(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(np.asarray(history, float)[-1], horizon)

    def predict_quantiles(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return self._apply(self.predict_point(history, horizon))
