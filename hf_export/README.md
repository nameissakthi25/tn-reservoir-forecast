---
license:
  - other
  - cc-by-4.0
license_name: godl-india
license_link: https://www.data.gov.in/Godl
language:
  - en
tags:
  - hydrology
  - time-series
  - india
  - water-resources
  - climate
pretty_name: Tamil Nadu Water Panels (Cauvery discharge and reservoir storage)
size_categories:
  - 10K<n<100K
configs:
  - config_name: biligundulu_daily
    data_files: data/biligundulu_daily.csv
  - config_name: biligundulu_weekly
    data_files: data/biligundulu_weekly.csv
  - config_name: tn_reservoir_storage_weekly
    data_files: data/tn_reservoir_storage_weekly.csv
---

# Tamil Nadu Water Panels

Cleaned, analysis-ready hydrological series for the Cauvery basin and Tamil Nadu's
major reservoirs, assembled from Indian government open data.

**Why this exists.** The underlying data is public but not usable as published. The
national water portal's CWC daily reservoir dataset covers only Odisha and Madhya
Pradesh, and enumerating its Tamil Nadu resources returned no reservoir file. The
archived reservoir bulletins are **weekly PDFs across two incompatible layouts**, and
that public index stops at 2025-05-08 — reservoir reporting moved to the RSMS portal
in April 2025. India-WRIS was unreachable from two independent networks during
collection (Aug–Sep 2026). These files are the tidy result of resolving that.

There is also a trap worth knowing about, documented below: **one inflow source
silently returns today's snapshot for dates it does not have.**

## Subsets

### `biligundulu_daily` — 19,848 rows, 1971-08-30 to 2025-12-31

Daily discharge of the Cauvery at **Biligundulu**, the gauging station on the
Karnataka–Tamil Nadu border about 40 km upstream of Mettur dam. This is the legally
designated inter-state measurement point for the Cauvery water dispute, which is why
the record is unusually long and complete.

Over 54 years and 19,848 calendar days: **19,664 observed**,
145 interpolated across gaps of ≤ 2 days (flagged), and 39
days left empty where gaps were longer. "Observed" here means measured — the
19,809 rows carrying a value include the interpolated ones.

| column | description |
|---|---|
| `date` | observation date |
| `discharge_cumecs` | daily discharge, m³/s |
| `is_spike` | flagged single-day outlier, value replaced (see QC) |
| `is_interpolated` | value filled across a gap of ≤ 2 days |

Median 100.3 m³/s; 99th percentile 2010.5 m³/s. Strongly right-skewed —
use a log transform or quantile methods, not RMSE.

**This is regulated flow, not natural flow.** Karnataka is required to deliver
177.25 TMC annually at this point on a tribunal-set monthly schedule. Treat it as
"water released across the border", not as a natural hydrograph.

### `biligundulu_weekly` — 2,836 rows, 1971-09-05 to 2026-01-04

Weekly aggregation (99.8% of weeks valid) with aligned covariates, ready
for modelling.

| column | description |
|---|---|
| `week_ending` | week ending Sunday |
| `discharge_cumecs` | weekly mean discharge, m³/s |
| `n_days` | observed days contributing (weeks with < 4 are invalidated) |
| `discharge_max_daily` | weekly maximum daily value |
| `valid` / `unusable` / `was_interpolated` | data-quality flags |
| `rain_mm`, `temp_c` | upper-Cauvery catchment mean (5 Karnataka points, ERA5) |
| `oni` | Oceanic Niño Index, forward-filled from monthly |
| `dmi` | Dipole Mode Index (IOD), forward-filled from monthly |
| `woy_sin`, `woy_cos` | week-of-year, cyclically encoded |
| `water_year` | June–May |
| `regime` | Cauvery governance regime (see below) |
| `monsoon_phase` | `SWM` / `NEM` / `inter` |
| `schedule_tmc` | tribunal-mandated monthly delivery, TMC (0 before the 2007 award) |

Rainfall is deliberately **Karnataka**, not Tamil Nadu: Biligundulu gauges Karnataka's
outflow, so the upper-catchment rainfall is the physical driver.

Monthly indices are **forward-filled, never interpolated** — a forecaster at time *t*
has the last published monthly value, not a smooth path through the next one.

### `tn_reservoir_storage_weekly` — 3,317 rows, 2015-04-16 to 2025-05-08

Weekly live storage, in billion cubic metres, parsed from **493 CWC weekly bulletin
PDFs**. Long format: `week_ending`, `reservoir`, `live_storage_bcm`.

| reservoir | weeks | first | last |
|---|---|---|---|
| Aliyar | 523 | 2015-04-16 | 2025-05-08 |
| Bhavanisagar | 523 | 2015-04-16 | 2025-05-08 |
| Mettur | 523 | 2015-04-16 | 2025-05-08 |
| Parambikulam | 523 | 2015-04-16 | 2025-05-08 |
| Sholayar | 523 | 2015-04-16 | 2025-05-08 |
| Vaigai | 523 | 2015-04-16 | 2025-05-08 |
| Sathanur | 167 | 2022-02-10 | 2025-05-08 |
| Manimuthar | 6 | 2025-04-03 | 2025-05-08 |
| Papanasam | 6 | 2025-04-03 | 2025-05-08 |

Only the first six have usable coverage for modelling. **Amaravathi and Krishnagiri
are absent from the CWC bulletin entirely.** Sathanur begins 2022; Manimuthar and
Papanasam appear only from 2025.

CWC names differ from common usage — `LOWER BHAWANI` is Bhavanisagar and `KARAYAR` is
Papanasam; both are normalised here.

## Important caveats

**Storage is not a natural process.** `storage change = inflow − release −
evaporation`, and releases are administrative decisions. CWC publishes **no** inflow,
outflow, release or evaporation field — verified by full-text search. Unexplained
variance in these series may be irrigation policy rather than hydrology.

**Governance regimes shift the seasonal shape.** The Cauvery regime changed at least
three times inside the discharge record: the CWDT interim order (1991), the final
award (2007), and the Supreme Court modification plus CWMA (2018). Annual volumes
barely differ across regimes, but **dry-season median flow roughly halves after 2007**
— a prescribed monthly schedule reallocates *when* water crosses the border, not how
much. Pooling regimes without an indicator treats several governance systems as one
process.

**ENSO signal is real but under-determined.** The six lowest annual volumes past
Biligundulu — 2016, 2003, 2023, 2012, 1987, 2002 — are close to the El Niño roll-call,
against a median near 260 TMC. But 54 years is only 54 monsoons and roughly ten strong
El Niño events.

**A source to avoid.** `tnagriculture.in` publishes reservoir inflow and outflow with
an apparently date-addressable archive. For dates it lacks it returns **today's
snapshot under the requested date's header** — 92 of 164 probed dates were fabricated
this way. A naive scrape yields today's value repeated through history, with no gap
and no error. Nothing from that source is included here.

## Quality control

Two decisions, both deliberate:

**Duplicate timestamps are collapsed by median.** They are genuine repeats (median
pair ratio 1.01), not the same day recorded in two units.

**The extreme tail is NOT percentile-clipped.** Values like 6,692 m³/s sit inside
coherent flood hydrographs (2,460 → 3,271 → 6,692 → 6,342 → 1,174) and are real
events — the ones that matter most. A narrow single-day spike filter (> 8× both
neighbours and > 10,000 m³/s) flags exactly **1 row**: 2018-08-19, recorded
74,713 between neighbours of 5,251 and 4,713.

That row is an entry error, not a unit error. Dividing by 35.31 — the cusecs-to-m³/s
hypothesis — yields 2,116, which would place the flood *peak below its own shoulders*.
It is interpolated rather than rescaled, and flagged in both `is_spike` and
`is_interpolated`.

Interpolation never runs off the end of a series: trailing and leading gaps are left
empty rather than carrying a neighbour's value forward. A partial final week is
therefore flagged `valid = false` **and** left null, not filled.

## Usage

```python
from datasets import load_dataset

daily = load_dataset("nameissakthi/tn-water-panels", "biligundulu_daily", split="train")
weekly = load_dataset("nameissakthi/tn-water-panels", "biligundulu_weekly", split="train")
storage = load_dataset("nameissakthi/tn-water-panels", "tn_reservoir_storage_weekly", split="train")
```

## Provenance and attribution

These files are **derivative works**. No source file is redistributed verbatim.

| source | data | terms |
|---|---|---|
| Central Water Commission via National Water Informatics Centre (`nwdp.nwic.gov.in`) | river discharge | open data, `other-open`; GoI open data is governed by [GODL-India](https://www.data.gov.in/Godl) |
| Central Water Commission | weekly reservoir bulletins | as above |
| NOAA Climate Prediction Center | ONI | public domain (US Government work) |
| NOAA Physical Sciences Laboratory | DMI (HadISST) | public domain |
| Open-Meteo (ERA5-derived) | rainfall, temperature | CC-BY 4.0 |

Attribution as required by GODL-India: contains information from the Central Water
Commission / National Water Informatics Centre, Government of India, sourced from
`https://nwdp.nwic.gov.in` and `https://cwc.gov.in`, used under the Government Open
Data License – India.

Note the licence label: the portal tags these records `other-open` without linking
specific licence text. GODL-India is the governing instrument for Government of India
open data, and is asserted here on that basis. Anyone with a stricter compliance
requirement should confirm with NWIC directly.

## Reproducing

Every file here is regenerated from primary sources by the pipeline at
[tn-reservoir-forecast](https://github.com/nameissakthi25/tn-reservoir-forecast):

```bash
python scripts/fetch_data.py --all
make panels
python scripts/export_hf_dataset.py
```

## Citation

```bibtex
@misc{tn_water_panels_2026,
  title  = {Tamil Nadu Water Panels: Cauvery discharge and reservoir storage},
  year   = {2026},
  note   = {Derived from Central Water Commission open data under GODL-India},
  url    = {https://huggingface.co/datasets/nameissakthi/tn-water-panels}
}
```
