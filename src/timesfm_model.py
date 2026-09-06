"""TimesFM-3 wrapper.

Verified against the installed package on 2026-09-04 (see tests/test_timesfm_api.py):

  * class is `TimesFM3Forecaster` (NOT TimesFM3Evaluator, which is the
    benchmark harness), from the `timesfm3` package -- `timesfm` is v2.5
  * quantiles are 9: [0.1..0.9], median at index 4, no mean column.
    The repo's own timesfm-forecasting/references/api_reference.md documents
    the 2.5 API (10 cols incl. mean) and is WRONG for v3.
  * contexts            : (n_targets, context_len)
  * past_only_covariates: (n_channels, context_len)  -- channel count is
    independent of target count
  * past_future_covariates: (n_channels, context_len + horizon)
  * output .forecast    : (n_targets, horizon)
           .quantiles   : (n_targets, horizon, 9)

DANGER, guarded below: model.decode() OVERRIDES the horizon argument whenever
past-future covariates are supplied -- it recomputes horizon as
`pf.shape[-1] - context_len`. A covariate window that is one step short
silently returns a shorter forecast with no warning, which would quietly
corrupt an entire backtest. assert_pf_width() makes that a hard failure.

Also: predict_batch() silently linear-interpolates NaNs of ANY gap length in
both targets and covariates. Gap policy is therefore enforced upstream in
build_panel.py and must never be delegated to the model.
"""
from __future__ import annotations

import numpy as np

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN_IDX = 4


def assert_pf_width(pf: np.ndarray | None, context_len: int, horizon: int) -> None:
    """Guard the silent-truncation trap."""
    if pf is None:
        return
    want = context_len + horizon
    if pf.shape[-1] != want:
        raise ValueError(
            f"past_future_covariates must be exactly context+horizon wide: "
            f"got {pf.shape[-1]}, expected {want} (context={context_len}, "
            f"horizon={horizon}). TimesFM would silently forecast "
            f"{pf.shape[-1] - context_len} steps instead of {horizon}."
        )


class TimesFM3:
    def __init__(self, device: str | None = None, batch_size: int = 16):
        from timesfm3 import TimesFM3Forecaster  # imported lazily: heavy

        self.fc = TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch", device=device, per_core_batch_size=batch_size
        )
        assert list(self.fc.config.quantiles) == list(QUANTILES), \
            f"unexpected quantile grid: {self.fc.config.quantiles}"

    def forecast(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        past_only: list[np.ndarray | None] | None = None,
        past_future: list[np.ndarray | None] | None = None,
        transform: str | None = None,
    ) -> np.ndarray:
        """Returns quantiles of shape (n_series, horizon, 9).

        transform: None or "log1p". Quantiles are back-transformed by the exact
        inverse -- valid because quantiles commute with monotone maps. A mean is
        never back-transformed anywhere in this codebase.
        """
        ctxs = [np.atleast_2d(np.asarray(c, float)) for c in contexts]
        if past_future is not None:
            for c, pf in zip(ctxs, past_future):
                assert_pf_width(pf, c.shape[-1], horizon)

        if transform == "log1p":
            ctxs = [np.log1p(np.clip(c, 0, None)) for c in ctxs]
        elif transform is not None:
            raise ValueError(f"unknown transform {transform!r}")

        outs = list(self.fc.predict_batch(
            contexts=ctxs,
            horizon=horizon,
            past_only_covariates=past_only,
            past_future_covariates=past_future,
            return_quantiles=True,
        ))
        q = np.stack([o.quantiles[0] for o in outs], axis=0)  # (n, horizon, 9)

        if transform == "log1p":
            q = np.expm1(q)
        return np.sort(q.clip(min=0.0), axis=-1)
