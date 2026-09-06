"""Ingest ENSO (ONI) and IOD (DMI) monthly indices. Cached to data/raw/."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

RAW = Path("data/raw/indices")
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

# ONI seasons are 3-month rolling means labelled by their centre month.
_SEASON_CENTRE = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _cached(url: str, name: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists():
        return path.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8", "ignore")
    path.write_text(text)
    return text


def load_oni() -> pd.DataFrame:
    """Monthly ONI. Columns: date (month start), oni."""
    txt = _cached(ONI_URL, "oni.ascii.txt")
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    df.columns = [c.strip().upper() for c in df.columns]
    df["month"] = df.SEAS.map(_SEASON_CENTRE)
    df = df.dropna(subset=["month"])
    # NDJ is centred on December of YR; DJF on January of YR+1
    yr = df.YR.astype(int) + (df.SEAS == "DJF").astype(int) * 0
    df["date"] = pd.to_datetime(
        dict(year=yr, month=df.month.astype(int), day=1)
    )
    return df[["date", "ANOM"]].rename(columns={"ANOM": "oni"}).sort_values("date")


def load_dmi() -> pd.DataFrame:
    """Monthly Dipole Mode Index (HadISST). Columns: date, dmi."""
    txt = _cached(DMI_URL, "dmi.had.long.data")
    rows = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit():
            year = int(parts[0])
            for m, v in enumerate(parts[1:], start=1):
                val = float(v)
                rows.append((pd.Timestamp(year=year, month=m, day=1), val))
    df = pd.DataFrame(rows, columns=["date", "dmi"])
    # sentinel missing values in PSL files
    df.loc[df.dmi < -9, "dmi"] = np.nan
    return df.dropna().sort_values("date")


def monthly_to_weekly(df: pd.DataFrame, col: str, index: pd.DatetimeIndex) -> pd.Series:
    """Broadcast a monthly index onto a weekly index by forward-fill.

    No interpolation: the value in force during that week is the last published
    monthly value, which is what an operational forecaster would have had.
    """
    s = df.set_index("date")[col].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)
