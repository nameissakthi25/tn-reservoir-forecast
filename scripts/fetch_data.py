#!/usr/bin/env python3
"""Download the raw sources into data/raw/. Idempotent — skips what exists.

The project treats data/raw as an immutable cache, so this is the only script
that touches the network for source data. Everything downstream runs offline.

  python scripts/fetch_data.py --discharge    # ~80 MB, needed for target 1
  python scripts/fetch_data.py --bulletins    # ~640 MB, needed for target 2
  python scripts/fetch_data.py --all

Reachability notes, verified 2026-09-06 — several published paths are dead:
  * nwdp.nwic.gov.in works;  the nwdp.nwic.in mirror does NOT.
  * indiawris.gov.in resolves but times out from every network tried.
  * The CWC bulletin index stops at 2025-05-08 even though bulletins are still
    issued; reservoir reporting moved to a new portal in April 2025.
If a source has moved again, this script fails loudly rather than substituting.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/128.0 Safari/537.36"}

NWDP = {
    "disch_tn_1950_2000.csv":
        "https://nwdp.nwic.gov.in/dataset/08fa3fd0-7861-471d-a295-27c1b239d1fa/"
        "resource/cba07162-d1da-4987-b52e-0c1e2173e287/download/"
        "riverdischarge_manual_daily_cwc_tn_1950_2000.csv",
    "disch_tn_2001_2025.csv":
        "https://nwdp.nwic.gov.in/dataset/08fa3fd0-7861-471d-a295-27c1b239d1fa/"
        "resource/fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73/download/"
        "river_discharge_manual_daily_cwc_tn_2001_2025.csv",
}
CWC_INDEX = "https://cwc.gov.in/en/reservoir-level-storage-bulletin"


def get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_discharge() -> None:
    out = RAW / "nwdp"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in NWDP.items():
        path = out / name
        if path.exists() and path.stat().st_size > 1_000_000:
            print(f"  have {name}")
            continue
        print(f"  downloading {name} ...", flush=True)
        path.write_bytes(get(url))
        print(f"    {path.stat().st_size/1e6:.0f} MB")


def fetch_bulletins(workers: int = 6) -> None:
    """Scrape the paginated index, then cache every weekly bulletin PDF."""
    out = RAW / "cwc" / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    links_file = RAW / "cwc" / "pdf_links.txt"

    if links_file.exists():
        links = [l for l in links_file.read_text().splitlines() if l.strip()]
    else:
        links = set()
        for page in range(22):
            html = get(f"{CWC_INDEX}?page={page}").decode("utf-8", "ignore")
            for m in re.finditer(r'href="([^"]+\.pdf)"', html, re.I):
                u = m.group(1)
                links.add(u if u.startswith("http") else "https://cwc.gov.in" + u)
            time.sleep(0.4)
        links = sorted(links)
        links_file.write_text("\n".join(links))
    print(f"  {len(links)} bulletin links")

    date_res = [re.compile(r"(\d{2})[-.](\d{2})[-.](\d{4})"),
                re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")]

    def one(url: str):
        fn = url.rsplit("/", 1)[-1]
        for rx in date_res:
            m = rx.search(fn)
            if not m:
                continue
            d, mo, y = m.groups()
            try:
                if not (2010 <= int(y) <= 2027):
                    continue
                stamp = f"{y}-{mo}-{d}"
            except ValueError:
                continue
            path = out / f"{stamp}.pdf"
            if path.exists() and path.stat().st_size > 10_000:
                return None
            try:
                path.write_bytes(get(url))
                time.sleep(0.2)
                return stamp
            except Exception as e:                        # noqa: BLE001
                return f"FAIL {stamp}: {type(e).__name__}"
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = [r for r in ex.map(one, links) if r]
    fails = [r for r in done if isinstance(r, str) and r.startswith("FAIL")]
    print(f"  fetched {len(done)-len(fails)} new, {len(fails)} failed")
    print(f"  cached total: {len(list(out.glob('*.pdf')))} PDFs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discharge", action="store_true", help="NWDP river discharge (~80 MB)")
    ap.add_argument("--bulletins", action="store_true", help="CWC weekly bulletins (~640 MB)")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.discharge or a.bulletins or a.all):
        ap.print_help()
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    if a.discharge or a.all:
        print("NWDP river discharge:")
        fetch_discharge()
    if a.bulletins or a.all:
        print("CWC weekly bulletins:")
        fetch_bulletins()
    print("\nRainfall and climate indices are fetched on demand and cached "
          "automatically by the pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
