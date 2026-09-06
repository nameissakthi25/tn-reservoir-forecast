"""Backtest for target 2: TN reservoir storage.

Reuses the Biligundulu harness wholesale -- baselines.py, metrics.py and the
scoring logic are target-agnostic. Only ingest and the multi-target handling
differ.

Two things about this target are structurally worse than the discharge one and
must be reported, not buried:

  1. The panel is 10 years (2015-2025), so the spec's ">=10 held-out years"
     is UNREACHABLE. With a 4-year context there are ~6 test years. Stated in
     the results rather than quietly relaxed.
  2. Storage = inflow - release - evaporation, and CWC publishes no release
     field. Skill here is contaminated by irrigation policy in a way the
     Biligundulu result is not.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import baselines as bl
import metrics as M
from ingest import climate_indices as ci

PANEL = Path("data/processed/panel_tn_storage_weekly.parquet")
OUT = Path("data/processed")
CFG = yaml.safe_load(Path("config/reservoirs.yaml").read_text())
HORIZONS = [1, 4, 8, 13]
MAX_H = 13
CONTEXT = 208          # 4 years; the panel is only ~10 years long


def usable_reservoirs() -> list[str]:
    return [r["name"] for r in CFG["reservoirs"] if r.get("usable", True)]


def catchment_points() -> list[tuple[str, float, float]]:
    """(name, lat, lon) for each usable reservoir's catchment rainfall point."""
    out = []
    for r in CFG["reservoirs"]:
        if not r.get("usable", True):
            continue
        lat, lon = r.get("catchment", [r["lat"], r["lon"]])
        out.append((r["name"], float(lat), float(lon)))
    return out


def load_panel(with_rain: bool = True) -> tuple[pd.DataFrame, list[str]]:
    p = pd.read_parquet(PANEL)
    names = [c for c in usable_reservoirs() if c in p.columns]
    p = p[["week"] + names].copy()
    # short gaps only; anything longer stays NaN and its origins are dropped
    p[names] = p[names].interpolate(limit=2, limit_area="inside")

    idx = pd.DatetimeIndex(p.week)
    p["oni"] = ci.monthly_to_weekly(ci.load_oni(), "oni", idx).values
    p["dmi"] = ci.monthly_to_weekly(ci.load_dmi(), "dmi", idx).values

    # per-reservoir catchment rainfall. Without this the covariate ablation on
    # this target is not a fair test: ENSO and seasonality alone leave the
    # single most causally direct driver out of the model.
    if with_rain:
        from ingest import rainfall as rf
        pts = [pt for pt in catchment_points() if pt[0] in names]
        daily = rf.points_daily(pts, start="2014-01-01",
                                end=str(idx.max().date())).set_index("day")
        for nm, _, _ in pts:
            wk = daily[f"rain_{nm}"].resample("W-THU").sum()
            p[f"rain_{nm}"] = wk.reindex(idx).values
    woy = idx.isocalendar().week.values.astype(float)
    p["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    p["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    month = idx.month.values
    p["is_swm"] = ((month >= 6) & (month <= 9)).astype(float)
    p["is_nem"] = ((month >= 10) & (month <= 12)).astype(float)
    return p, names


def _clean(a: np.ndarray) -> np.ndarray:
    return pd.Series(a, dtype=float).interpolate(limit_direction="both").fillna(0.0).values


PAST_FUTURE = ["woy_sin", "woy_cos", "is_swm", "is_nem"]


PAST_ONLY = ["oni", "dmi"]   # default; run() recomputes with rainfall


def past_only_cols(p: pd.DataFrame, names: list[str]) -> list[str]:
    rain = [f"rain_{n}" for n in names if f"rain_{n}" in p.columns]
    return rain + ["oni", "dmi"]


def run(every: int, use_timesfm: bool, device: str, limit: int | None,
        with_rain: bool = True) -> pd.DataFrame:
    p, names = load_panel(with_rain=with_rain)
    PAST_ONLY = past_only_cols(p, names)
    print(f"past-only covariates ({len(PAST_ONLY)}): {PAST_ONLY}")
    Y = p[names].values.astype(float)            # (weeks, n_reservoirs)
    dates = pd.DatetimeIndex(p.week)

    idxs = [i for i in range(CONTEXT, len(p) - MAX_H) if i % every == 0]
    if limit:
        idxs = idxs[:limit]
    print(f"panel {len(p)} weeks, {len(names)} reservoirs: {names}")
    print(f"test span {dates[CONTEXT].date()} -> {dates[len(p)-MAX_H-1].date()} "
          f"| origins {len(idxs)}")

    ctxs, pos, pfs, keep = [], [], [], []
    for i in idxs:
        fut = Y[i:i + MAX_H]
        if not np.isfinite(fut).all():
            continue
        ctx = Y[i - CONTEXT:i].T                 # (n_reservoirs, context)
        if not np.isfinite(ctx).all():
            ctx = np.vstack([_clean(r) for r in ctx])
        ctxs.append(ctx)
        pos.append(np.vstack([_clean(p[c].values[i - CONTEXT:i]) for c in PAST_ONLY]))
        pfs.append(np.vstack([_clean(p[c].values[i - CONTEXT:i + MAX_H])
                              for c in PAST_FUTURE]))
        keep.append(i)
    print(f"scorable origins: {len(keep)}")

    recs = []

    # ---- per-reservoir baselines --------------------------------------------
    for k, i in enumerate(keep):
        for j, nm in enumerate(names):
            hist_y, hist_d = Y[:i, j], dates[:i]
            ok = np.isfinite(hist_y)
            clim = bl.SmoothedClimatology().fit(hist_d[ok], hist_y[ok])
            sn = bl.SeasonalNaive().fit(hist_y[ok], MAX_H)
            pers = bl.Persistence().fit(hist_y[ok], MAX_H)
            scale = M.seasonal_naive_scale(hist_y[ok])
            truth, tdates = Y[i:i + MAX_H, j], dates[i:i + MAX_H]
            for name, q in (
                ("climatology", clim.predict_quantiles(tdates)),
                ("seasonal_naive", sn.predict_quantiles(hist_y[ok], MAX_H)),
                ("persistence", pers.predict_quantiles(hist_y[ok], MAX_H)),
            ):
                recs.append((i, nm, name, q, truth, tdates, scale))

    # ---- TimesFM, all reservoirs as multiple targets in one call ------------
    if use_timesfm and keep:
        from timesfm_model import TimesFM3
        m = TimesFM3(device=device)
        for label, po, pf in (("timesfm_univariate", None, None),
                              ("timesfm_covariates", pos, pfs)):
            print(f"  running {label} on {len(ctxs)} origins "
                  f"x {len(names)} targets ...")
            outs = list(m.fc.predict_batch(
                contexts=ctxs, horizon=MAX_H,
                past_only_covariates=po, past_future_covariates=pf,
                return_quantiles=True))
            for k, i in enumerate(keep):
                q_all = outs[k].quantiles          # (n_reservoirs, MAX_H, 9)
                for j, nm in enumerate(names):
                    recs.append((i, nm, label, np.sort(q_all[j].clip(min=0), -1),
                                 Y[i:i + MAX_H, j], dates[i:i + MAX_H],
                                 M.seasonal_naive_scale(Y[:i, j])))

    rows = []
    for i, nm, model, q, truth, tdates, scale in recs:
        for h in HORIZONS:
            j = h - 1
            yt, qh = truth[j:j + 1], q[j:j + 1]
            rec = dict(origin=i, reservoir=nm, model=model, horizon=h,
                       date=tdates[j], truth=float(yt[0]),
                       pinball=float(M.pinball(yt, qh)[0]),
                       abs_err=abs(float(yt[0]) - float(qh[0, 4])),
                       scale=scale)
            rec.update({f"q{int(round(lv*100))}": float(qh[0, ii])
                        for ii, lv in enumerate(M.QUANTILES)})
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def agg(g):
        return pd.Series({"crps": g.pinball.mean(),
                          "wql": g.pinball.sum() / g.truth.abs().sum(),
                          "mae": g.abs_err.mean(), "n": len(g)})

    overall = df.groupby(["model", "horizon"]).apply(agg, include_groups=False).reset_index()
    ref = overall[overall.model == "climatology"].set_index("horizon").crps
    overall["skill_vs_clim"] = [M.skill_score(r.crps, ref.get(r.horizon, np.nan))
                                for r in overall.itertuples()]

    per = (df.groupby(["reservoir", "model", "horizon"])
             .apply(agg, include_groups=False).reset_index())
    refp = per[per.model == "climatology"].set_index(["reservoir", "horizon"]).crps
    per["skill_vs_clim"] = [
        M.skill_score(r.crps, refp.get((r.reservoir, r.horizon), np.nan))
        for r in per.itertuples()]
    return overall, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--no-timesfm", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-rain", action="store_true",
                    help="ablation: drop per-reservoir rainfall covariates")
    a = ap.parse_args()

    df = run(a.every, not a.no_timesfm, a.device, a.limit,
             with_rain=not a.no_rain)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "backtest_storage_raw.parquet", index=False)
    overall, per = summarise(df)
    overall.to_csv(OUT / "skill_storage_overall.csv", index=False)
    per.to_csv(OUT / "skill_storage_by_reservoir.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== STORAGE: skill vs climatology (CRPS) ===")
    print(overall.round(4).to_string(index=False))
    print("\n=== per reservoir (TimesFM covariates only) ===")
    sub = per[per.model == "timesfm_covariates"]
    print(sub.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
