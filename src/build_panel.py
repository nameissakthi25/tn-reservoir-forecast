"""Build the tidy WEEKLY panel for the Biligundulu inflow target.

Daily -> weekly is deliberate: the spec forbids feeding daily rainfall to
TimesFM (zero-inflated), and daily discharge is even more skewed.

Missingness rule: a week is valid only if >=4 of its 7 days are observed.
Runs of more than 2 consecutive invalid weeks are left as NaN and reported --
never silently interpolated. This has to be enforced here because TimesFM's
own predict_batch() silently linear-interpolates NaNs of ANY gap length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from ingest import climate_indices as ci
from ingest import discharge as dis
from ingest import rainfall as rf

CFG = yaml.safe_load(Path("config/sources.yaml").read_text())
OUT = Path("data/processed")
MIN_DAYS_PER_WEEK = 4
MAX_CONSEC_INTERP_WEEKS = 2


def _water_year(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.where(idx.month >= 6, idx.year, idx.year - 1)


def _regime(wy: np.ndarray) -> np.ndarray:
    out = np.full(len(wy), "pre1991", dtype=object)
    out[wy >= 1991] = "interim91"
    out[wy >= 2007] = "award07"
    out[wy >= 2018] = "cwma18"
    return out


def _monsoon_phase(month: np.ndarray) -> np.ndarray:
    """SWM Jun-Sep, NEM Oct-Dec, inter-monsoon Jan-May."""
    out = np.full(len(month), "inter", dtype=object)
    out[(month >= 6) & (month <= 9)] = "SWM"
    out[(month >= 10) & (month <= 12)] = "NEM"
    return out


def _flag_long_gaps(valid: pd.Series) -> pd.Series:
    """True where a week sits in a run of >MAX_CONSEC_INTERP_WEEKS invalid weeks."""
    bad = ~valid
    grp = (bad != bad.shift()).cumsum()
    runlen = bad.groupby(grp).transform("sum")
    return bad & (runlen > MAX_CONSEC_INTERP_WEEKS)


def build(station: str | None = None) -> pd.DataFrame:
    station = station or CFG["target"]["name"]

    # ---- target: daily discharge -> weekly mean ------------------------------
    daily = dis.station_daily(station).set_index("day")
    wk = daily.discharge_cumecs.resample("W-SUN")
    panel = pd.DataFrame({
        "discharge_cumecs": wk.mean(),
        "n_days": wk.count(),
        "discharge_max_daily": wk.max(),
    })
    panel["valid"] = panel.n_days >= MIN_DAYS_PER_WEEK
    panel["unusable"] = _flag_long_gaps(panel.valid)
    panel.loc[~panel.valid, "discharge_cumecs"] = np.nan

    # short gaps (<=2 weeks) may be filled; long runs stay NaN by design
    filled = panel.discharge_cumecs.interpolate(limit=MAX_CONSEC_INTERP_WEEKS)
    panel["was_interpolated"] = panel.discharge_cumecs.isna() & filled.notna()
    panel["discharge_cumecs"] = filled.where(~panel.unusable)

    idx = panel.index

    # ---- past covariates -----------------------------------------------------
    rain = rf.catchment_daily().set_index("day")
    panel["rain_mm"] = rain.rain_mm.resample("W-SUN").sum().reindex(idx)
    panel["temp_c"] = rain.temp_c.resample("W-SUN").mean().reindex(idx)
    panel["oni"] = ci.monthly_to_weekly(ci.load_oni(), "oni", idx).values
    panel["dmi"] = ci.monthly_to_weekly(ci.load_dmi(), "dmi", idx).values

    # ---- past-future covariates (all deterministic / leakage-free) ----------
    woy = idx.isocalendar().week.values.astype(float)
    panel["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    panel["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    wy = _water_year(idx)
    panel["water_year"] = wy
    panel["regime"] = _regime(wy)
    panel["monsoon_phase"] = _monsoon_phase(idx.month.values)

    # tribunal monthly quantum, broadcast to weeks. Legally binding and
    # published in advance => usable as a future covariate with zero leakage.
    sched = CFG["schedule_tmc"]
    panel["schedule_tmc"] = [float(sched[m]) for m in idx.month]
    # only meaningful once the final award is in force
    panel.loc[panel.regime.isin(["pre1991", "interim91"]), "schedule_tmc"] = 0.0

    panel = panel.reset_index().rename(columns={"index": "week"})
    panel = panel.rename(columns={panel.columns[0]: "week"})
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build()
    path = OUT / "panel_biligundulu_weekly.parquet"
    panel.to_parquet(path, index=False)

    p = panel
    print(f"wrote {path}  rows={len(p)}")
    print(f"span: {p.week.min().date()} -> {p.week.max().date()}")
    print(f"valid weeks       : {int(p.valid.sum())} ({100*p.valid.mean():.1f}%)")
    print(f"interpolated (<=2){'':2}: {int(p.was_interpolated.sum())}")
    print(f"unusable (long gap): {int(p.unusable.sum())}")
    print(f"target NaN         : {int(p.discharge_cumecs.isna().sum())}")

    print("\nvalid weeks per water year (post-1990):")
    sub = p[p.water_year >= 1990]
    for wy_, g in sub.groupby("water_year"):
        n = int(g.valid.sum())
        print(f"  {wy_}: {n:2d}/52 {'#' * n}")


if __name__ == "__main__":
    main()
