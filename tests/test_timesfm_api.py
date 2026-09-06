"""Pin the TimesFM-3 API behaviours this project depends on.

These are regression guards, not model-quality tests. Each one corresponds to a
documented trap in README.md; if the upstream package changes any of them, the
backtest would silently become wrong rather than fail.
"""
import numpy as np
import pytest

CTX, HOR, NT = 260, 13, 3


def synth(n, length, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return np.asarray(
        [50 + 30 * np.sin(2 * np.pi * (t + 7 * i) / 52.0) + rng.normal(0, 2, length)
         for i in range(n)], dtype=np.float32)


@pytest.fixture(scope="module")
def fc():
    from timesfm3 import TimesFM3Forecaster
    return TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch",
                                              device="cpu")


def test_nine_quantiles_median_at_index_4(fc):
    """v3 emits 9 quantiles with no mean column; the 2.5 docs claim 10."""
    assert list(fc.config.quantiles) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert fc.config.median_quantile_index == 4


def test_multi_target_and_covariate_shapes(fc):
    tgt, po = synth(NT, CTX), synth(4, CTX, seed=1)
    pf = synth(2, CTX + HOR, seed=2)
    out = list(fc.predict_batch(contexts=[tgt], horizon=HOR,
                                past_only_covariates=[po],
                                past_future_covariates=[pf],
                                return_quantiles=True))[0]
    assert out.forecast.shape == (NT, HOR)
    assert out.quantiles.shape == (NT, HOR, 9)
    # covariate channel count is independent of target count
    assert po.shape[0] != NT and pf.shape[0] != NT


def test_median_column_equals_point_forecast(fc):
    out = list(fc.predict_batch(contexts=[synth(NT, CTX)], horizon=HOR,
                                return_quantiles=True))[0]
    assert np.allclose(out.quantiles[..., 4], out.forecast)
    assert np.all(np.diff(out.quantiles, axis=-1) >= -1e-6)


def test_short_pf_window_silently_truncates_horizon(fc):
    """The trap: decode() recomputes horizon from the covariate width."""
    bad = synth(2, CTX + 5, seed=3)
    out = list(fc.predict_batch(contexts=[synth(NT, CTX)], horizon=HOR,
                                past_future_covariates=[bad],
                                return_quantiles=True))[0]
    assert out.forecast.shape == (NT, 5)      # asked for 13, silently got 5


def test_our_guard_rejects_that(fc):
    import sys
    sys.path.insert(0, "src")
    from timesfm_model import assert_pf_width
    with pytest.raises(ValueError, match="silently forecast"):
        assert_pf_width(synth(2, CTX + 5), CTX, HOR)
    assert_pf_width(synth(2, CTX + HOR), CTX, HOR)   # correct width passes


def test_nans_are_silently_interpolated(fc):
    """Any gap length is filled with no warning -> policy must live upstream."""
    gap = synth(NT, CTX)
    gap[0, -30:-20] = np.nan
    out = list(fc.predict_batch(contexts=[gap], horizon=HOR))[0]
    assert np.isfinite(out.forecast).all()
