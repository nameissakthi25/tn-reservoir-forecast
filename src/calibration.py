"""Probabilistic calibration check for both targets.

CRPS rewards sharpness and calibration together, so a model can score well while
being systematically overconfident. This asks the separate question: when the
model says P90, is the truth actually below it 90% of the time?

Three diagnostics per (model, horizon):

  * quantile coverage -- P(y <= q_level) against its nominal level
  * central 80% interval coverage (P10-P90), the band drawn on the fan chart
  * a PIT-style uniformity check on the rank of y within the 9 quantiles

The PIT here is coarse by construction: 9 quantiles give 10 bins, so the rank
statistic is discrete and a Kolmogorov-Smirnov test against a continuous
uniform would be invalid. Chi-square on bin counts is used instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metrics import QUANTILES

OUT = Path("data/processed")
QCOLS = [f"q{int(round(l * 100))}" for l in QUANTILES]


def coverage_table(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group, keys), n=len(g))
        for lv, c in zip(QUANTILES, QCOLS):
            rec[f"cov{int(round(lv*100))}"] = float((g.truth <= g[c]).mean())
        rec["cov_central80"] = float(
            ((g.truth >= g.q10) & (g.truth <= g.q90)).mean())
        # mean absolute deviation of coverage from nominal, across the grid
        dev = [abs(rec[f"cov{int(round(lv*100))}"] - lv) for lv in QUANTILES]
        rec["mean_abs_dev"] = float(np.mean(dev))
        rows.append(rec)
    return pd.DataFrame(rows)


def pit_uniformity(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Chi-square on the 10 bins induced by the 9 predictive quantiles."""
    rows = []
    for keys, g in df.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        q = g[QCOLS].values
        y = g.truth.values[:, None]
        rank = (y > q).sum(axis=1)              # 0..9 -> 10 bins
        counts = np.bincount(rank, minlength=10)
        exp = np.full(10, counts.sum() / 10.0)
        chi2 = float(((counts - exp) ** 2 / exp).sum())
        p = float(stats.chi2.sf(chi2, df=9))
        rows.append(dict(zip(group, keys), n=len(g), chi2=chi2, p_value=p,
                         bin_lo=int(counts[0]), bin_hi=int(counts[-1]),
                         pct_below_p10=100 * counts[0] / counts.sum(),
                         pct_above_p90=100 * counts[-1] / counts.sum()))
    return pd.DataFrame(rows)


def report(path: Path, label: str, models: list[str]) -> None:
    df = pd.read_parquet(path)
    if QCOLS[0] not in df.columns:
        raise SystemExit(f"{path} lacks quantile columns -- rerun the backtest")
    df = df[df.model.isin(models)]

    cov = coverage_table(df, ["model", "horizon"])
    pit = pit_uniformity(df, ["model", "horizon"])
    m = cov.merge(pit[["model", "horizon", "p_value", "pct_below_p10",
                       "pct_above_p90"]], on=["model", "horizon"])

    print(f"\n{'='*96}\n{label}\n{'='*96}")
    print("nominal ->      10    20    30    40    50    60    70    80    90 "
          "| P10-P90 | mad | PIT p")
    for r in m.sort_values(["model", "horizon"]).itertuples():
        cells = " ".join(f"{getattr(r, f'cov{int(round(l*100))}'):5.2f}"
                         for l in QUANTILES)
        flag = ""
        if r.cov_central80 < 0.70:
            flag = "  <-- OVERCONFIDENT"
        elif r.cov_central80 > 0.90:
            flag = "  <-- too wide"
        print(f"{r.model:20} h={r.horizon:<3} {cells} |  {r.cov_central80:5.2f}  "
              f"| {r.mean_abs_dev:.3f} | {r.p_value:.1e}{flag}")

    print("\ntail behaviour (nominal 10% below P10, 10% above P90):")
    for r in m.sort_values(["model", "horizon"]).itertuples():
        print(f"  {r.model:20} h={r.horizon:<3} "
              f"below P10 {r.pct_below_p10:5.1f}%   above P90 {r.pct_above_p90:5.1f}%")

    m.to_csv(OUT / f"calibration_{label.split()[0].lower()}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["timesfm_covariates", "timesfm_univariate",
                             "climatology", "persistence"])
    a = ap.parse_args()
    report(OUT / "backtest_raw.parquet", "DISCHARGE (Biligundulu)", a.models)
    report(OUT / "backtest_storage_raw.parquet", "STORAGE (6 TN reservoirs)",
           a.models)


if __name__ == "__main__":
    main()
