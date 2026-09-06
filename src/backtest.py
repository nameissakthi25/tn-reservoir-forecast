"""Rolling-origin backtest with whole-year holdout.

Whole years are held out because adjacent weeks leak: a 13-week horizon from a
late-December origin would otherwise be scored against weeks that a
neighbouring origin trained on.

Everything a model sees at an origin is strictly prior to that origin --
including the climatology fit, which is refitted per origin on an expanding
window so the reference is genuinely out-of-sample.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import baselines as bl
import metrics as M

PANEL = Path("data/processed/panel_biligundulu_weekly.parquet")
OUT = Path("data/processed")
HORIZONS = [1, 4, 8, 13]
MAX_H = 13
CONTEXT = 520          # 10 years of weekly context
TARGET = "discharge_cumecs"

PAST_ONLY = ["rain_mm", "temp_c", "oni", "dmi"]
PAST_FUTURE = ["woy_sin", "woy_cos", "schedule_tmc", "regime_idx"]


def load_panel(start_wy: int) -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p = p[p.water_year >= start_wy - (CONTEXT // 52) - 1].reset_index(drop=True)
    p["regime_idx"] = p.regime.map(
        {"pre1991": 0.0, "interim91": 1.0, "award07": 2.0, "cwma18": 3.0}
    ).astype(float)
    return p


def origins(p: pd.DataFrame, test_wys: list[int], every: int) -> list[int]:
    """Row positions usable as forecast origins."""
    out = []
    for i in range(CONTEXT, len(p) - MAX_H):
        if p.water_year.iloc[i] in test_wys and (i % every == 0):
            out.append(i)
    return out


def _clean(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a, dtype=float)
    return s.interpolate(limit_direction="both").fillna(0.0).values


def run(start_wy: int, every: int, use_timesfm: bool, device: str,
        transform: str | None, limit: int | None) -> pd.DataFrame:
    p = load_panel(start_wy)
    test_wys = sorted(w for w in p.water_year.unique() if w >= start_wy)
    idxs = origins(p, test_wys, every)
    if limit:
        idxs = idxs[:limit]
    print(f"panel {len(p)} weeks | test water-years {test_wys[0]}-{test_wys[-1]} "
          f"({len(test_wys)}) | origins {len(idxs)}")

    y = p[TARGET].values.astype(float)
    dates = pd.DatetimeIndex(p.week)

    # ---- assemble per-origin arrays -----------------------------------------
    ctxs, pos, pfs, truths, tdates, keep = [], [], [], [], [], []
    for i in idxs:
        fut = slice(i, i + MAX_H)
        truth = y[fut]
        if not np.isfinite(truth).all():
            continue                      # never score against imputed truth
        ctx = y[i - CONTEXT:i]
        if not np.isfinite(ctx).all():
            ctx = _clean(ctx)
        ctxs.append(ctx)
        pos.append(np.vstack([_clean(p[c].values[i - CONTEXT:i]) for c in PAST_ONLY]))
        pfs.append(np.vstack([_clean(p[c].values[i - CONTEXT:i + MAX_H])
                              for c in PAST_FUTURE]))
        truths.append(truth)
        tdates.append(dates[fut])
        keep.append(i)
    print(f"scorable origins: {len(keep)}")

    rows = []

    # ---- baselines -----------------------------------------------------------
    for k, i in enumerate(keep):
        hist_dates, hist_y = dates[:i], y[:i]
        ok = np.isfinite(hist_y)
        clim = bl.SmoothedClimatology().fit(hist_dates[ok], hist_y[ok])
        sn = bl.SeasonalNaive().fit(hist_y[ok], MAX_H)
        pers = bl.Persistence().fit(hist_y[ok], MAX_H)
        scale = M.seasonal_naive_scale(hist_y[ok])

        preds = {
            "climatology": clim.predict_quantiles(tdates[k]),
            "seasonal_naive": sn.predict_quantiles(hist_y[ok], MAX_H),
            "persistence": pers.predict_quantiles(hist_y[ok], MAX_H),
        }
        for name, q in preds.items():
            rows.append(dict(origin=i, model=name, q=q, truth=truths[k],
                             dates=tdates[k], scale=scale))

    # ---- TimesFM -------------------------------------------------------------
    if use_timesfm and keep:
        from timesfm_model import TimesFM3
        m = TimesFM3(device=device)
        for label, po, pf in (("timesfm_univariate", None, None),
                              ("timesfm_covariates", pos, pfs)):
            print(f"  running {label} on {len(ctxs)} origins ...")
            q = m.forecast(ctxs, MAX_H, past_only=po, past_future=pf,
                           transform=transform)
            for k, i in enumerate(keep):
                rows.append(dict(origin=i, model=label, q=q[k], truth=truths[k],
                                 dates=tdates[k],
                                 scale=M.seasonal_naive_scale(y[:i])))

    # ---- score ---------------------------------------------------------------
    recs = []
    for r in rows:
        for h in HORIZONS:
            j = h - 1
            yt, qh = r["truth"][j:j + 1], r["q"][j:j + 1]
            rec = dict(
                origin=r["origin"], model=r["model"], horizon=h,
                date=r["dates"][j], truth=float(yt[0]),
                median=float(qh[0, M.QUANTILES.searchsorted(0.5)]),
                pinball=float(M.pinball(yt, qh)[0]),
                abs_err=abs(float(yt[0]) - float(qh[0, 4])),
                scale=r["scale"],
            )
            # persist the full predictive quantiles for calibration analysis
            rec.update({f"q{int(round(lv*100))}": float(qh[0, ii])
                        for ii, lv in enumerate(M.QUANTILES)})
            recs.append(rec)
    return pd.DataFrame(recs)


def summarise(df: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Skill vs climatology per model x horizon, plus per-season breakdown."""
    ph = dict(zip(pd.DatetimeIndex(p.week), p.monsoon_phase))
    df = df.assign(season=df.date.map(ph))

    def agg(g):
        return pd.Series({
            "crps": g.pinball.mean(),
            "wql": g.pinball.sum() / g.truth.abs().sum(),
            "mae": g.abs_err.mean(),
            "mase": g.abs_err.mean() / g.scale.mean(),
            "n": len(g),
        })

    out = df.groupby(["model", "horizon"]).apply(agg, include_groups=False).reset_index()
    ref = out[out.model == "climatology"].set_index("horizon")
    out["skill_vs_clim"] = [
        M.skill_score(r.crps, ref.crps.get(r.horizon, np.nan)) for r in out.itertuples()
    ]
    seas = (df.groupby(["model", "horizon", "season"])
              .apply(agg, include_groups=False).reset_index())
    refs = seas[seas.model == "climatology"].set_index(["horizon", "season"]).crps
    seas["skill_vs_clim"] = [
        M.skill_score(r.crps, refs.get((r.horizon, r.season), np.nan))
        for r in seas.itertuples()
    ]
    return out, seas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-wy", type=int, default=1991)
    ap.add_argument("--every", type=int, default=4, help="origin spacing in weeks")
    ap.add_argument("--no-timesfm", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transform", default=None, choices=[None, "log1p"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    df = run(a.start_wy, a.every, not a.no_timesfm, a.device, a.transform, a.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "backtest_raw.parquet", index=False)

    p = load_panel(a.start_wy)
    overall, seas = summarise(df, p)
    overall.to_csv(OUT / "skill_overall.csv", index=False)
    seas.to_csv(OUT / "skill_by_season.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== skill vs climatology (CRPS; positive = beats climatology) ===")
    print(overall.round(4).to_string(index=False))
    print("\n=== by season ===")
    print(seas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
