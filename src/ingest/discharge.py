"""Ingest CWC daily river discharge for Tamil Nadu from the National Water Data Portal.

Raw CSVs are immutable and cached under data/raw/nwdp/. This module only reads
them, applies QC, and returns a tidy daily series.

QC policy, derived from the audit in notebooks/01_data_audit.ipynb:

  * Duplicate timestamps are collapsed by median. They are genuine repeats
    (median pair ratio 1.01), NOT the same day recorded in two units.

  * The extreme tail is NOT clipped by percentile. Values like 6,692 m3/s sit
    inside coherent flood hydrographs (2,460 -> 3,271 -> 6,692 -> 6,342 -> 1,174)
    and are real events -- exactly the ones that matter. A percentile clip would
    delete them.

  * Instead a narrow single-day spike filter removes points >8x BOTH neighbours
    and >10,000 m3/s. On Biligundulu this flags exactly one row: 2018-08-19,
    recorded 74,713 between neighbours of 5,251 and 4,713. Dividing by 35.31
    (the cusecs hypothesis) yields 2,116, which would put the flood *peak*
    below its own shoulders -- so it is an entry error, not a unit error, and
    is interpolated rather than rescaled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/nwdp")
FILES = ["disch_tn_1950_2000.csv", "disch_tn_2001_2025.csv"]
VALUE = "Manual Daily River Water Discharge (m3/sec)"
SPIKE_RATIO = 8.0
SPIKE_FLOOR = 10_000.0


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    col = next(c for c in df.columns if "Discharge" in c)
    df = df.rename(columns={col: VALUE})
    df["dt"] = pd.to_datetime(
        df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df[VALUE] = pd.to_numeric(df[VALUE], errors="coerce")
    keep = ["Station", "dt", VALUE, "Latitude", "Longitude", "River", "Basin"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["dt"])


def load_raw() -> pd.DataFrame:
    """All TN discharge stations, daily, deduplicated. No spike filter."""
    frames = [_read_one(RAW / f) for f in FILES]
    return pd.concat(frames, ignore_index=True)


def station_daily(station: str = "BILIGUNDULU") -> pd.DataFrame:
    """Tidy daily series for one station with QC applied.

    Returns columns: day, discharge_cumecs, is_interpolated, is_spike.
    """
    df = load_raw()
    s = df.loc[df.Station == station, ["dt", VALUE]].copy()
    if s.empty:
        raise ValueError(f"station {station!r} not found")
    s["day"] = s["dt"].dt.normalize()

    # collapse duplicate timestamps (genuine repeats, not dual-unit entries)
    s = s.groupby("day", as_index=False)[VALUE].median()
    s = s.rename(columns={VALUE: "discharge_cumecs"}).sort_values("day")

    # reindex onto a complete daily calendar so gaps are explicit
    full = pd.date_range(s.day.min(), s.day.max(), freq="D")
    s = s.set_index("day").reindex(full).rename_axis("day").reset_index()

    v = s.discharge_cumecs.values.astype(float)
    prev, nxt = np.r_[np.nan, v[:-1]], np.r_[v[1:], np.nan]
    spike = (v > SPIKE_RATIO * prev) & (v > SPIKE_RATIO * nxt) & (v > SPIKE_FLOOR)
    s["is_spike"] = spike
    s.loc[spike, "discharge_cumecs"] = np.nan

    was_missing = s.discharge_cumecs.isna()
    # short gaps only; the panel builder enforces the >2-unit rule at weekly scale
    s["discharge_cumecs"] = s.discharge_cumecs.interpolate(limit=2)
    s["is_interpolated"] = was_missing & s.discharge_cumecs.notna()
    return s


def gap_report(s: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive missing days remaining after QC."""
    miss = s.discharge_cumecs.isna().values
    out, start = [], None
    for i, m in enumerate(miss):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((s.day.iloc[start], s.day.iloc[i - 1], i - start))
            start = None
    if start is not None:
        out.append((s.day.iloc[start], s.day.iloc[-1], len(miss) - start))
    return pd.DataFrame(out, columns=["gap_start", "gap_end", "n_days"])
