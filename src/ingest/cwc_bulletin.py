"""Ingest CWC weekly reservoir bulletins (PDF) for Tamil Nadu reservoirs.

The bulletins are the only public weekly storage series for TN major reservoirs.
Caveats established in the data audit and carried into the README:

  * The index at cwc.gov.in stops at 2025-05-08 even though bulletins are still
    issued; reservoir reporting moved to the RSMS portal in April 2025. So this
    source gives 2015-04-16 -> 2025-05-08, ~503 weekly issues, 95.8% of slots.
  * The bulletin publishes LIVE STORAGE and LEVEL only. It contains no inflow,
    outflow, release or evaporation fields at all (verified by full-text search).
  * CWC reservoir names differ from the spec's: Bhavanisagar is LOWER BHAWANI
    and Papanasam appears as KARAYAR. Amaravathi and Krishnagiri are absent
    entirely. Sholayar is present as a bonus.

cwc.gov.in is unreachable from the tool sandbox, so downloads must run with the
sandbox disabled. Everything is cached under data/raw/cwc/.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import urllib.request

RAW = Path("data/raw/cwc")
PDF_DIR = RAW / "pdf"
INDEX = "https://cwc.gov.in/en/reservoir-level-storage-bulletin"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/128.0 Safari/537.36"}

# CWC name -> canonical project name used across this repo
NAME_MAP = {
    "METTUR": "Mettur",
    "LOWER BHAWANI": "Bhavanisagar",
    "VAIGAI": "Vaigai",
    "SATHANUR": "Sathanur",
    "KARAYAR": "Papanasam",
    "ALIYAR": "Aliyar",
    "MANIMUTHAR": "Manimuthar",
    "PARAMBIKULAM": "Parambikulam",
    "SHOLAYAR": "Sholayar",
}

DATE_PATTERNS = [
    re.compile(r"(\d{2})[-.](\d{2})[-.](\d{4})"),
    re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)"),
]


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def scrape_index(pages: int = 22) -> list[str]:
    """All bulletin PDF links across the paginated index."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / "pdf_links.txt"
    if cache.exists():
        return [l for l in cache.read_text().splitlines() if l.strip()]

    links: set[str] = set()
    for p in range(pages):
        html = _get(f"{INDEX}?page={p}").decode("utf-8", "ignore")
        for m in re.finditer(r'href="([^"]+\.pdf)"', html, re.I):
            u = m.group(1)
            if u.startswith("/"):
                u = "https://cwc.gov.in" + u
            links.add(u)
        time.sleep(0.4)
    out = sorted(links)
    cache.write_text("\n".join(out))
    return out


def date_from_url(url: str) -> pd.Timestamp | None:
    fn = url.rsplit("/", 1)[-1]
    for i, rx in enumerate(DATE_PATTERNS):
        m = rx.search(fn)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            ts = pd.Timestamp(year=int(c), month=int(b), day=int(a))
        except ValueError:
            continue
        if pd.Timestamp("2010-01-01") <= ts <= pd.Timestamp("2027-01-01"):
            return ts
    return None


def download_all(links: list[str], workers: int = 6) -> pd.DataFrame:
    """Cache every bulletin PDF locally. Returns a manifest."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def one(url: str):
        d = date_from_url(url)
        if d is None:
            return None
        path = PDF_DIR / f"{d.date()}.pdf"
        if not path.exists() or path.stat().st_size < 10_000:
            try:
                path.write_bytes(_get(url))
                time.sleep(0.2)
            except Exception as e:                       # noqa: BLE001
                return dict(date=d, url=url, path=str(path), ok=False, err=str(e)[:80])
        return dict(date=d, url=url, path=str(path), ok=True, err="")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, links):
            if r:
                rows.append(r)
    man = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    man.to_csv(RAW / "manifest.csv", index=False)
    return man


# --- parsing ------------------------------------------------------------------
# Row shape in the state-wise tables, after whitespace normalisation:
#   <sl> <NAME> Tamil Nadu <irr> <hydel> <FRL> <livecap> <date> <level>
#   <live storage> <pct> ...
_NUM = r"\d[\d,]*\.?\d*"
_DATE = r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
_NAME = r"[A-Z][A-Z .\-&()]{2,40}?"

# The bulletin layout changed over the decade. Two forms are supported.
#
# MODERN (seen 2025), state-wise table keyed on the "Tamil Nadu" suffix:
#   88 METTUR Tamil Nadu 122.000 360.000 240.79000 2.64700 08.05.2025
#      237.23500 2.15320 81.34 ...
#   order: <irr> <hydel> <FRL> <live_cap> <date> <level> <live_storage>
ROW_MODERN = re.compile(
    rf"({_NAME})\s+Tamil\s*Nadu\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+"
    rf"({_DATE})\s+({_NUM})\s+({_NUM})",
    re.I,
)

# LEGACY (seen 2015-2024), no state suffix, different column order and an
# optional * / # footnote marker before the serial number:
#   *87 METTUR(STANLEY) 240.79 226.82 2.647 1.030 6/10/2015 39 15 45 122 360
#   order: <FRL> <level> <live_cap> <live_storage> <date>
ROW_LEGACY = re.compile(
    rf"[*#]?\s*\d{{1,3}}\s+({_NAME})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+"
    rf"({_DATE})",
    re.I,
)


def normalise_name(raw: str) -> str | None:
    """CWC name -> canonical project name.

    Handles the parenthetical suffixes that come and go across years, e.g.
    METTUR(STANLEY) -> METTUR.
    """
    n = re.sub(r"\(.*?\)", " ", raw)
    n = re.sub(r"^[*#\d\s]+", "", n)
    n = re.sub(r"[^A-Z ]", " ", n.upper())
    n = re.sub(r"\s+", " ", n).strip()
    return NAME_MAP.get(n)


def parse_pdf(path: str | Path) -> pd.DataFrame:
    """Extract (reservoir, level_m, live_storage_bcm) rows from one bulletin."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = " ".join((pg.extract_text() or "") for pg in reader.pages)
    text = re.sub(r"\s+", " ", text)

    def f(x: str) -> float:
        return float(x.replace(",", ""))

    def valid(frl, live_cap, level, storage) -> bool:
        # capacities are BCM (small); levels are metres above MSL
        return (0 < live_cap < 100 and 0 <= storage <= live_cap * 1.25
                and 0 < level <= frl * 1.02 and frl > 10)

    out = []
    for m in ROW_MODERN.finditer(text):
        canon = normalise_name(m.group(1))
        if canon is None:
            continue
        frl, live_cap = f(m.group(4)), f(m.group(5))
        level, storage = f(m.group(7)), f(m.group(8))
        if not valid(frl, live_cap, level, storage):
            continue
        out.append(dict(reservoir=canon, frl_m=frl, live_cap_bcm=live_cap,
                        level_m=level, storage_bcm=storage,
                        row_date=m.group(6), layout="modern"))

    seen = {r["reservoir"] for r in out}
    for m in ROW_LEGACY.finditer(text):
        canon = normalise_name(m.group(1))
        if canon is None or canon in seen:
            continue
        frl, level = f(m.group(2)), f(m.group(3))
        live_cap, storage = f(m.group(4)), f(m.group(5))
        if not valid(frl, live_cap, level, storage):
            continue
        out.append(dict(reservoir=canon, frl_m=frl, live_cap_bcm=live_cap,
                        level_m=level, storage_bcm=storage,
                        row_date=m.group(6), layout="legacy"))

    if not out:
        return pd.DataFrame(columns=["reservoir", "frl_m", "live_cap_bcm",
                                     "level_m", "storage_bcm", "row_date",
                                     "layout"])
    return pd.DataFrame(out).drop_duplicates("reservoir")


def parse_all(manifest: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    def one(r):
        try:
            df = parse_pdf(r.path)
        except Exception:                                # noqa: BLE001
            return pd.DataFrame()
        df["date"] = r.date
        return df

    rows = [one(r) for r in manifest.itertuples() if Path(r.path).exists()]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["reservoir", "date"])
