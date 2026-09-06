"""Parse all cached CWC bulletins into a tidy weekly storage panel.

Target 2 of the project. Reuses the Biligundulu harness for everything
downstream (baselines, metrics, backtest, plots); only ingest differs.

Unlike the discharge target, storage here is confounded by release policy --
CWC publishes no inflow, outflow or release field at all. That caveat belongs
in every result produced from this panel.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest import cwc_bulletin as cb  # noqa: E402

OUT = Path("data/processed")
CACHE = Path("data/interim/cwc_parsed.parquet")
MAX_CONSEC_INTERP_WEEKS = 2


def _one(path: str):
    try:
        df = cb.parse_pdf(path)
    except Exception:                                    # noqa: BLE001
        return None
    if df.empty:
        return None
    df["date"] = pd.Timestamp(Path(path).stem)
    return df


def parse_all(workers: int = 8) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    paths = sorted(str(p) for p in (cb.PDF_DIR).glob("*.pdf"))
    print(f"parsing {len(paths)} bulletins with {workers} workers ...")
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(_one, paths, chunksize=4), 1):
            if df is not None:
                frames.append(df)
            if i % 50 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


def build() -> pd.DataFrame:
    raw = parse_all()

    # cross-check: the date printed inside the row must match the filename date
    rd = pd.to_datetime(raw.row_date, format="%d.%m.%Y", errors="coerce")
    mismatch = (rd.notna() & (rd.dt.date != raw.date.dt.date)).sum()
    print(f"row-date vs filename-date mismatches: {mismatch} of {len(raw)}")

    # storage as a fraction of live capacity -- comparable across reservoirs
    raw["pct_full"] = 100 * raw.storage_bcm / raw.live_cap_bcm

    wide = raw.pivot_table(index="date", columns="reservoir",
                           values="storage_bcm", aggfunc="last").sort_index()

    # snap to a regular weekly grid (bulletins are issued each Thursday)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="W-THU")
    wide = wide.reindex(wide.index.union(grid)).interpolate(
        limit=MAX_CONSEC_INTERP_WEEKS, limit_area="inside").reindex(grid)

    panel = wide.reset_index().rename(columns={"index": "week"})
    panel = panel.rename(columns={panel.columns[0]: "week"})
    return panel, raw


def main() -> None:
    panel, raw = build()
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "panel_tn_storage_weekly.parquet", index=False)

    res = [c for c in panel.columns if c != "week"]
    print(f"\nwrote {OUT/'panel_tn_storage_weekly.parquet'}")
    print(f"weeks {len(panel)}  span {panel.week.min().date()} -> "
          f"{panel.week.max().date()}  reservoirs {len(res)}")

    print(f"\n{'reservoir':14} {'n_obs':>6} {'%cover':>7} {'first':>11} {'last':>11} "
          f"{'min':>7} {'median':>7} {'max':>7}  (BCM)")
    print("-" * 84)
    for c in res:
        s = panel[c]
        ok = s.notna()
        print(f"{c:14} {int(ok.sum()):6d} {100*ok.mean():6.1f}% "
              f"{str(panel.week[ok].min().date()):>11} "
              f"{str(panel.week[ok].max().date()):>11} "
              f"{s.min():7.3f} {s.median():7.3f} {s.max():7.3f}")

    print("\nobservations per year:")
    for y, g in panel.groupby(panel.week.dt.year):
        n = int(g[res].notna().any(axis=1).sum())
        print(f"  {y}: {n:2d} {'#' * n}")


if __name__ == "__main__":
    main()
