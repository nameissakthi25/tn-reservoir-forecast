"""Ingest upper-Cauvery catchment rainfall and temperature from Open-Meteo Archive.

Biligundulu gauges Karnataka's outflow, so the physically relevant rainfall is
the KARNATAKA upper-Cauvery catchment (Kodagu headwaters, Hassan, Mysore,
Kabini and Hemavathy sub-basins), not Tamil Nadu's.

ERA5-derived, free, no API key. Every response is cached to data/raw/rainfall/
so reruns are offline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import urllib.parse
import urllib.request

RAW = Path("data/raw/rainfall")
API = "https://archive-api.open-meteo.com/v1/archive"

POINTS = [
    ("kodagu", 12.42, 75.74),
    ("hassan", 13.00, 76.10),
    ("mysore", 12.30, 76.65),
    ("kabini", 11.98, 76.35),
    ("hemavathy", 12.82, 76.10),
]


def _fetch_point(name: str, lat: float, lon: float,
                 start: str, end: str) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}_{start}_{end}.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "daily": "precipitation_sum,temperature_2m_mean",
            "timezone": "Asia/Kolkata",
        })
        req = urllib.request.Request(f"{API}?{q}",
                                     headers={"User-Agent": "Mozilla/5.0 (research)"})
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    payload = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(15 * (attempt + 1))   # free tier throttles hard
        if payload is None:
            raise RuntimeError(f"Open-Meteo rate-limited for {name}; retry later")
        path.write_text(json.dumps(payload))
        time.sleep(6)   # pace subsequent points

    d = payload["daily"]
    return pd.DataFrame({
        "day": pd.to_datetime(d["time"]),
        f"rain_{name}": d["precipitation_sum"],
        f"temp_{name}": d["temperature_2m_mean"],
    })


def points_daily(points: list[tuple[str, float, float]],
                 start: str = "2014-01-01",
                 end: str = "2026-09-01") -> pd.DataFrame:
    """Per-point daily rainfall/temperature, one column pair per named point.

    Used by the storage target, where each reservoir needs its OWN catchment
    rainfall -- a single basin average would wash out the differences between
    reservoirs 400 km apart.
    """
    out = None
    for name, lat, lon in points:
        df = _fetch_point(name, lat, lon, start, end)
        out = df if out is None else out.merge(df, on="day", how="outer")
    return out.sort_values("day")


def catchment_daily(start: str = "1990-01-01", end: str = "2026-09-01") -> pd.DataFrame:
    """Catchment-mean daily rainfall (mm) and temperature (C)."""
    out = None
    for name, lat, lon in POINTS:
        df = _fetch_point(name, lat, lon, start, end)
        out = df if out is None else out.merge(df, on="day", how="outer")

    rain_cols = [c for c in out.columns if c.startswith("rain_")]
    temp_cols = [c for c in out.columns if c.startswith("temp_")]
    return pd.DataFrame({
        "day": out.day,
        "rain_mm": out[rain_cols].mean(axis=1),
        "temp_c": out[temp_cols].mean(axis=1),
    }).sort_values("day")
