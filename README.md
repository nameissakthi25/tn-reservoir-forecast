# Forecasting Tamil Nadu water with TimesFM-3

Weekly probabilistic forecasts, 1–13 weeks ahead, for two targets — and an honest
skill assessment against baselines that are genuinely hard to beat.

| target | what | span | status |
|---|---|---|---|
| **Cauvery discharge at Biligundulu** | the Karnataka→Tamil Nadu border gauge, 40 km above Mettur | 1971–2025, 54 years | complete |
| **Reservoir storage, 6 reservoirs** | Mettur, Bhavanisagar, Vaigai, Parambikulam, Aliyar, Sholayar | 2015–2025 | complete |

TimesFM-3 is used **zero-shot** — never trained on this data. The question is whether
generic pattern knowledge beats purpose-built simple methods. Sometimes it does.

> **A negative result, clearly reported, is a successful outcome for this project.**
> That was fixed before modelling, and it shaped every design choice below.

## Quickstart

```bash
pip install -e .
python scripts/fetch_data.py --discharge   # ~80 MB; --all for both targets
make panels                                # raw -> tidy weekly panels
make baselines                             # establish the bar, no model
make backtest                              # full backtest incl. TimesFM
make calibration figures
```

`make help` lists everything. Raw data is gitignored and rebuilt by the fetch script;
every network call is cached, so reruns are offline.

## Repository map

```
config/         reservoir definitions, source URLs, tribunal release schedule
scripts/        fetch_data.py — the only script that touches the network
src/ingest/     one module per source (discharge, bulletins, rainfall, indices)
src/            panels, baselines, metrics, backtests, calibration, plots, fine-tuning
docs/           pre-registered fine-tuning protocol, with outcomes recorded
tests/          API regression tests pinning four silent TimesFM-3 traps
data/processed/ result CSVs (tracked) and panels (gitignored, rebuildable)
figures/        fan charts and skill curves
```

## Headline result

**TimesFM-3 beats climatology at short horizons and does not beat it at long ones.**

CRPS skill vs smoothed climatology, 890 origins across 35 held-out water years
(1991–2025). Positive = beats climatology.

Canonical configuration: training window 1991+, external `log1p` transform (both
choices tested — see Sensitivity below).

| model | h=1 | h=4 | h=8 | h=13 |
|---|---|---|---|---|
| **TimesFM-3 + covariates** | **+0.254** | **+0.061** | +0.014 | −0.001 |
| TimesFM-3 univariate | +0.232 | +0.053 | −0.011 | −0.020 |
| persistence | +0.115 | −0.317 | −0.761 | −1.214 |
| seasonal naive | −0.397 | −0.427 | −0.423 | −0.388 |

At **1 week** TimesFM-3 is clearly better than climatology (23% CRPS reduction) and
better than persistence, the strongest naive baseline. At **4 weeks** the margin is
small but real (+6%). At **8 and 13 weeks it ties or loses** — climatology is as good
as or better than the foundation model.

That is the answer to the question the project asked, and the long-horizon result is
not a bug to be tuned away. Beyond about a month, the seasonal cycle is essentially
all the information there is in this series, and climatology already encodes it.

**Covariates add almost nothing**: +0.233 vs +0.223 at h=1, and the two modes are
within ~0.03 everywhere. Rainfall, ONI, DMI, the tribunal schedule and week-of-year
together buy roughly one percentage point of skill. This corroborates the TimesFM-3
claim that the model performs well without covariates, and it is a negative result
worth stating plainly: on this series the covariate machinery is not earning its
complexity.

## Target 2 — TN reservoir storage

Panel: 526 weeks, 2015-04-16 → 2025-05-08, parsed from 493 CWC bulletin PDFs
(3,088 rows, **zero** row-date/filename-date mismatches). Six reservoirs at 99.4%
coverage: Mettur, Bhavanisagar, Vaigai, Parambikulam, Aliyar, Sholayar. 882 scored
origin-reservoir pairs.

**Climatology is the wrong bar for storage, and reporting only it would flatter the
model badly.** Storage is a *stock* — the running integral of inflow minus release —
so it is strongly autocorrelated and persistence is very hard to beat, while
climatology is weak. Both references, therefore:

| horizon | CRPS (TimesFM) | vs climatology | vs persistence |
|---|---|---|---|
| 1 | 0.0160 | +0.897 | **+0.204** |
| 4 | 0.0516 | +0.696 | **+0.185** |
| 8 | 0.0866 | +0.506 | **+0.185** |
| 13 | 0.1161 | +0.334 | **+0.209** |

The +0.90 against climatology is nearly all baseline weakness, not model strength.
The honest number is the **+0.19 to +0.22 against persistence** — real, consistent,
and roughly flat across horizon.

The two targets invert each other, and the reason is physical:

| | strong baseline | TimesFM vs climatology | TimesFM vs persistence |
|---|---|---|---|
| Discharge (a flux) | climatology | +0.23 → −0.01 | +0.13 → +0.54 |
| Storage (a stock) | persistence | +0.90 → +0.34 | +0.20 → +0.22 |

For a flux the seasonal cycle carries the information and climatology already has it;
for a stock the current level carries it and persistence already has it. TimesFM
beats the *weak* reference easily in both cases and the *strong* reference modestly.
Quoting either result against its weak baseline alone would be misleading.

Covariates again add nothing — univariate is marginally **better** at h=8 and h=13
(0.1139 vs 0.1150). Across both targets, the covariate machinery has not earned its
complexity.

### Storage caveats that limit the result

- **The spec's ≥10 held-out years is unreachable here.** The panel is ~10 years
  total; with a 4-year context there are ~6 test years. Not relaxed quietly — stated.
- **Release policy is inside the target.** CWC publishes no inflow, outflow, release
  or evaporation field. Skill here is contaminated by irrigation decisions in a way
  the Biligundulu result is not.
- Only 6 of the spec's 10 reservoirs are usable. Sathanur begins 2022-02 (31.7%);
  Manimuthar and Papanasam appear only from 2025-04 (1.1%). **Amaravathi and
  Krishnagiri are absent from the CWC bulletin entirely.**
- The bulletin layout changed mid-archive: pre-2025 issues use a different column
  order and name form (`METTUR(STANLEY)`), and CWC monitored only 6 TN reservoirs in
  the early years. Both layouts are parsed; see `src/ingest/cwc_bulletin.py`.

## Sensitivity and ablations

Three checks run after the headline results, two of them robustness tests and one a
suspected bug.

### 1. Rainfall on the storage target — the suspected bug, and it wasn't one

The storage target originally ran with **no rainfall covariate at all**, only ENSO,
IOD and seasonality. Its covariate ablation was therefore not a fair test. Adding
per-reservoir catchment rainfall (6 channels, one per reservoir, Open-Meteo points
chosen per catchment rather than a single basin average) changes essentially nothing:

| horizon | CRPS univariate | CRPS + covariates | covariate gain |
|---|---|---|---|
| 1 | 0.0162 | 0.0160 | +1.3% |
| 4 | 0.0517 | 0.0516 | +0.2% |
| 8 | 0.0856 | 0.0866 | −1.1% |
| 13 | 0.1139 | 0.1161 | −1.9% |

**The flat ablation survives a fair test.** Including the single most causally direct
driver of reservoir storage buys nothing, and slightly hurts at long horizons. This
was the most likely bug in the project and it is not a bug — it is a finding. The
covariate result now stands on both targets.

### 2. External `log1p` transform — small CRPS gain, large calibration gain

Discharge is heavily right-skewed, but TimesFM already normalises per series
internally, so an external transform could plausibly have fought it. It doesn't. CRPS
improves slightly and consistently at every horizon (+2.8%, +0.2%, +1.0%, +0.8%).

The real gain is distributional. Calibration mad improves from 0.025–0.053 to
**0.005–0.024**, and at h=8 the PIT test returns p=0.97 — as close to perfectly
calibrated as this test can detect. `log1p` is therefore canonical.

**The left tail is the part that matters operationally, and it is the part that got
fixed.** Observations falling below P10 went from 7.3% to 10.8% against a nominal
10%. Before the transform, the model's lower bound was too optimistic: it claimed a
10% chance of flow below some level when the true chance was 7%, understating how bad
the bad case could be. For a water manager the left tail *is* the decision — nobody
plans around the median monsoon, they plan around the dry scenario, and reservoir
operating decisions are triggered by low-flow thresholds. A P10 that is quietly too
high is precisely the failure mode that matters, and it is now honest. This is a
larger practical improvement than the 0.2–2.8% CRPS gain suggests, and it would have
been invisible without the separate calibration check.

Worth noting *why* this is safe: quantiles commute exactly with monotone transforms,
so back-transforming the nine predictive quantiles by `expm1` is lossless. Doing the
same to a mean would not be, and is never done here.

### 3. Training-window sensitivity — conclusions are robust

The 1991+ window was chosen on evidence of a 2007 regime break. Restricting to 2007+
(19 test years, 471 origins vs 890) barely moves anything:

| horizon | skill, 1991+ | skill, 2007+ | vs persistence, 1991+ | vs persistence, 2007+ |
|---|---|---|---|---|
| 1 | +0.254 | +0.201 | +0.157 | +0.159 |
| 4 | +0.061 | +0.065 | +0.287 | +0.255 |
| 8 | +0.014 | +0.016 | +0.440 | +0.439 |
| 13 | −0.001 | −0.027 | +0.548 | +0.598 |

The h=1 figure is lower on the shorter window, but with half the origins. The shape
of the result — strong at 1 week, gone by 8, a tie at 13 — is unchanged. Stored as
`skill_discharge_wy2007_sensitivity.csv`.

## Fine-tuning: attempted, and it made the model worse

LoRA fine-tuning was run on the storage target under the pre-registered protocol in
`docs/finetuning-protocol.md` — a separate adapter per fold, trained only on data
strictly prior to that fold, with a deliberately contaminated negative control and a
stopping rule fixed in advance. **It failed every criterion at every horizon.**

| horizon | CRPS zero-shot | CRPS fine-tuned | gain | P10–P90 coverage | control gain |
|---|---|---|---|---|---|
| 1 | 0.0141 | 0.0139 | +1.2% | 0.74 | +3.0% ⚠ |
| 4 | 0.0537 | 0.0539 | −0.3% | 0.64 | +15.9% |
| 8 | 0.0883 | 0.0906 | −2.7% | 0.54 | +30.0% |
| 13 | 0.1262 | 0.1346 | −6.7% | 0.52 | +28.6% |

The damage is **calibration, not accuracy**. CRPS barely moves, but P10–P90 coverage
falls from zero-shot's ~0.79 to **0.52** against a nominal 0.80. The fine-tuned model
buys a little sharpness by lying about its uncertainty — the exact failure criterion 3
existed to catch, and precisely the wrong trade for a water-management application
where the low-flow tail is the decision.

### The confound was tested and eliminated

Training windows use a 96-week context while inference used 208, and that mismatch was
the obvious suspect. It was tested directly by re-running the entire harness with
inference context set to 96, on a **byte-identical origin set** (origin selection still
requires 208 weeks of history, so only the context length changes) and with identical
per-fold training losses to five decimals.

Coverage moved by **at most 0.02** — 0.732→0.741, 0.624→0.644, 0.518→0.536,
0.550→0.518. It remains far below the acceptable band either way. The mismatch is not
the cause; LoRA is genuinely collapsing the predictive spread.

Matching the context also *strengthened* the control (from +7/+16/+16% to
+15.9/+30.0/+28.6%), so at h=4, 8 and 13 the test is properly powered and the negative
result is trustworthy. Only h=1 still trips the lower trip-wire (+3.0% < 15%): there,
the honest statement is **not** "fine-tuning doesn't help" but "this experiment cannot
detect whether it does".

### The mechanism hypothesis was refuted

The pre-registered prediction was that gains would concentrate in heavily managed
reservoirs, because operating rules are learnable structure a generic model lacks. The
opposite happened, consistently across both runs:

| reservoir | ctx208, h=8 | ctx96, h=8 |
|---|---|---|
| Mettur (most policy-governed) | −9.6% | −5.9% |
| Parambikulam (rainfall-driven) | +6.2% | +16.0% |

The only consistent winner is a small rainfall-driven catchment; the most
release-governed reservoir is among the worst. Whatever LoRA is doing here, it is not
learning operating rules.

### What this does and does not license saying

Defensible: *LoRA fine-tuning at these settings degrades TimesFM-3 on this task,
primarily by destroying calibration, and this is not an artifact of the
train/inference context mismatch.*

Not defensible: *fine-tuning cannot help this problem.* Hyperparameters were never
tuned — 300 steps, lr 1e-4, r=8 — and a coverage collapse from 0.79 to 0.52 is the
signature of over-training against ~3k observations. Early stopping on validation
**coverage** rather than loss is the obvious next attempt, and has not been made.

### Discharge: the same protocol, and the test came back uninterpretable

Fine-tuning was then run on the discharge target — 25 folds (2001–2025), contexts
matched at 256 weeks by design rather than fixed afterwards. **The lower trip-wire
fired at all four horizons**: contaminated gains of +2.6%, +7.1%, +8.8%, +5.9%, all
under the 15% floor. Per the protocol every horizon is therefore **uninterpretable**.
Fine-tuning is *untested* on discharge, not disproven.

The cause is upstream of the science. The control's own training loss **rose**
(0.17906 → 0.18154), as did the largest fold's. LoRA at lr 1e-4 over 300 steps is not
fitting this target even when handed the test years outright.

Three things the run did establish:

| | discharge | storage |
|---|---|---|
| best contaminated gain | +8.8% | +30.0% |
| coverage at h=8 | **0.742** | **0.536** |
| coverage at h=13 | **0.717** | **0.518** |

1. **An upper bound on what is extractable.** Even cheating outright buys ≤8.8% here.
   Discharge is a spiky flux over 1,774 weeks; storage is six smooth curves. That bound
   corroborates the information-limit conclusion from an independent direction.
2. **The calibration collapse was storage-specific.** Discharge holds at 0.72–0.75
   against storage's collapse to 0.52 — which *reverses* the expectation that a repeat
   would indict LoRA generally.
3. **A directional signal survives the block.** At h=13, only 5 of 25 folds improved
   (sign p=0.9995) — strong evidence toward fine-tuning hurting at long horizon, even
   though the formal verdict is withheld.

### Why the discharge test was not rescued, and where it stops

A hyperparameter sweep against the contaminated control answered the blocking
question. **LoRA works on discharge — the original run simply under-fit on two axes at
once.** Raising the learning rate from 1e-4 to 1e-3 took the control from +8.8% to
+53.7% at h=8, and doubling steps to 600 cleared every horizon (+26 / +32 / +68 / +52%).

Two incidental corrections fall out of that: the "rising training loss" reported
earlier was an artifact of comparing a single step against a tail mean — properly
measured, loss falls monotonically with learning rate. And h=1 is not intrinsically
un-improvable; it responds to schedule length rather than learning rate.

**But the sweep was stopped part-way, because tuning against the control is the wrong
selection rule.** The control's job is to memorise the test years, so a config that
memorises well is by construction prone to overfitting. Carrying those settings to the
honest run would mean selecting the settings most likely to make it fail, then
reporting that failure as evidence about fine-tuning — a negative result that cannot
distinguish "fine-tuning does not help" from "I chose overfitting-prone settings".
Overshooting the floor also weakens the leak detector, since the upper trip-wire scales
with the contaminated gain.

**So fine-tuning is left FAILED on storage and UNTESTED on discharge.** The correct
design is recorded in `docs/finetuning-protocol.md` §4: keep the control as a pass/fail
validity gate, and select hyperparameters by nested validation inside each fold's
training slice. Nothing else in this study depends on the outcome.

Two bugs were caught before this result was believed, both mine. A single missing week
(2022-04-28) put NaN in the target windows of folds 2023+, silently destroying four of
eight adapters — the first run's clean-looking FAIL table was meaningless. The
protocol's lower trip-wire flagged it as uninterpretable before it could be reported.
Training windows now carry a validity mask, and a non-finite loss raises rather than
propagating.

## Calibration

CRPS rewards sharpness and calibration jointly, so a model can score well while being
systematically overconfident. `src/calibration.py` asks the separate question: when
the model says P90, is the truth below it 90% of the time?

Central 80% (P10–P90) coverage — nominal 0.80 — and mean absolute deviation of
coverage from nominal across the 9-quantile grid:

| | h=1 | h=4 | h=8 | h=13 | mad range |
|---|---|---|---|---|---|
| **Discharge** — TimesFM+cov (log1p) | 0.76 | **0.80** | 0.79 | 0.83 | 0.005–0.024 |
| Discharge — climatology | 0.66 | 0.65 | 0.65 | 0.65 | ~0.067 |
| **Storage** — TimesFM+cov | 0.79 | 0.80 | 0.79 | 0.79 | 0.012–0.029 |
| Storage — persistence | 0.80 | 0.77 | 0.77 | 0.76 | 0.017–0.031 |
| Storage — climatology | 0.58 | 0.56 | 0.54 | 0.55 | ~0.205 |

**TimesFM-3 is well calibrated on both targets.** On discharge, h=4 and h=8 return
PIT p-values of 0.69 and 0.97 with coverage deviations of 0.007 and 0.005 — as close
to perfect as this test can resolve. Storage sits at 0.79–0.80 throughout. The
weakest cells are discharge h=1 (0.76, mildly overconfident) and h=13 (0.83, mildly
wide).

**Climatology is badly calibrated on both, and catastrophically so on storage** —
0.55 coverage against a nominal 0.80, and 37–40% of observations above P90. It is not
merely imprecise, it is biased: reservoirs in the later years are much fuller than an
expanding-window climatology anticipates. This further undercuts the +0.90 headline:
the reference is not just weak, it is broken.

Caveat on the p-values: with n≈890 per cell, trivial deviations reach significance,
so the mad column is the more meaningful figure. The PIT is also coarse by
construction — 9 quantiles give 10 discrete bins, so a χ² on bin counts is used
rather than a KS test against a continuous uniform, which would be invalid here.

### A bug this check caught

The first calibration run showed persistence with P10–P90 coverage of **0.18** at
h=13. The cause was mine, not the model's: `Persistence.fit()` estimated error
quantiles at 1 step and reused them at every horizon, so its intervals never widened
with lead time. That made persistence look far worse than it is and inflated
TimesFM's apparent margin. Both naive baselines now fit error quantiles **per lead
time**. The correction moved storage skill-vs-persistence from a
horizon-widening +0.20→+0.29 to a flat +0.20→+0.22 — the numbers above are the
corrected ones. Discharge was barely affected.

## What the model cannot do

- **It cannot predict the monsoon.** Skill collapses to zero by 8 weeks. Anyone
  wanting a seasonal outlook should use climatology conditioned on ENSO, not this.
- **It cannot predict releases.** See the confounding section — the target is
  regulated flow.
- **It cannot forecast from today.** The NWDP discharge archive ends 2025-12-31, so
  the latest possible origin is 2026-01-04. The fan chart in `figures/` is issued
  from that date, not from the present.

## Data provenance

| Data | Source | Coverage | Verified |
|---|---|---|---|
| Daily discharge, Biligundulu | CWC via National Water Data Portal (CKAN) | 1971-08-30 → 2025-12-31, ~99% complete | 2026-09-04 |
| Catchment rainfall / temperature | Open-Meteo Archive (ERA5) | 1990 → 2026, 5 upper-Cauvery points in **Karnataka** | 2026-09-04 |
| ONI (ENSO) | NOAA CPC | 1950 → 2026 monthly | 2026-09-04 |
| DMI (IOD) | NOAA PSL (HadISST) | 1870 → 2026 monthly | 2026-09-04 |

Host notes, since several published paths are wrong or dead:

- `nwdp.nwic.gov.in` **works**; the `nwdp.nwic.in` mirror is **down**.
- `origin.cpc.ncep.noaa.gov` no longer resolves — use `www.cpc.ncep.noaa.gov`.
- `indiawris.gov.in` times out from every network tried. Treat as down.
- The CWC bulletin index at `cwc.gov.in` **stops at 2025-05-08** despite bulletins
  still being issued; reservoir data moved to the RSMS portal in April 2025.
- **NWDP holds no Tamil Nadu reservoir file.** All 114 TN resources were enumerated;
  zero match `reserv*`. Only Odisha and Madhya Pradesh have CWC reservoir data.

Raw data under `data/raw/` is immutable. Every network call is cached there, so
reruns are offline.

## Why Biligundulu, and why that is not the same as "inflow"

The spec preferred inflow over storage, because storage mixes hydrology with
irrigation policy. No public historical inflow archive exists for TN major
reservoirs: `tnagriculture.in` serves inflow and outflow but its archive is partial,
starts around 2017, and **silently returns the live snapshot for dates it lacks** —
92 of 164 probed dates were whole-page fallbacks. A naive scrape produces today's
value repeated through history with no gap and no error.

Biligundulu is the alternative: the Karnataka→Tamil Nadu border gauge, ~40 km above
Mettur, and the dominant inflow term for the reservoir.

**But it is regulated flow, not natural flow.** It is the legally designated
inter-state measurement point; Karnataka must deliver 177.25 TMC annually on a
tribunal-set monthly schedule, and CWRC issues directives measured there. Modelling
it removes Tamil Nadu's irrigation policy and substitutes **Karnataka's release
policy**. That is an improvement, not an escape.

## Governance regimes — the reason the window is 1991+

The Cauvery governance regime changed at least three times inside the record: the
CWDT interim order (1991), the final award (2007), and the Supreme Court
modification plus CWMA (2018). Before 1991 there was no binding allocation.

Tested empirically. **Annual volumes barely differ** (pre-1991 vs interim, KS
p=0.72). **The seasonal shape does**:

| month | pre-1991 | interim-91 | award-07 | cwma-18 |
|---|---|---|---|---|
| Jan | 73 | 82 | 43 | 47 |
| Feb | 48 | 55 | 30 | 21 |
| Mar | 45 | 49 | 34 | 23 |
| Apr | 53 | 56 | 47 | 25 |

(median m³/s). Dry-season baseflow roughly halved after 2007 — a prescribed monthly
schedule reallocates *when* water crosses the border, not how much.

So training uses **1991–2025** (35 water years): pre-1991 is a different generating
process, while 2007+ alone would give only 18 years and miss the ≥20-year
climatology the spec wants. Regime enters as an indicator with breaks at 2007 and
2018.

## Leakage handling

**Option 3 of the spec's three: past-covariates only for ENSO/IOD.** Observed ONI and
DMI enter as *past* covariates up to the forecast origin and nothing beyond. No
archived forecast plumes were used, and observed future values were never fed in.
This is the weaker but zero-leakage choice.

The past-**future** covariates are all genuinely knowable in advance and carry no
leakage risk: week-of-year, the regime indicator (historical fact), and the tribunal
monthly quantum (published, legally binding, fixed).

Monthly indices are broadcast to weeks by **forward-fill, never interpolation** — a
forecaster at time *t* has the last published monthly value, not a smooth path
through the next one.

## Confounding: how much is policy rather than hydrology?

Not fully quantified, and the honest answer is that a large share of dry-season
variance is governance. Two pieces of evidence:

1. The 2007 regime break halves dry-season flow without changing annual volume. That
   difference is entirely policy.
2. Urachikottai, immediately below Mettur, has median discharge of **exactly 0.0** in
   February, March, April and May — the dam shut. Release is a step function of
   administrative decisions, not a hydrological process.

The release ablation is limited by data: Urachikottai coverage is good 1979–2015 and
collapses after 2016 (94–381 obs/yr).

## ENSO

The six lowest years past Biligundulu are 2016 (69 TMC), 2003 (75), 2023 (81), 2012
(100), 1987 (107) and 2002 (110), against a median near 260 — essentially the El Niño
roll-call. The causal chain is ENSO → Karnataka SW monsoon inflow → Karnataka
releases → Biligundulu.

**But 54 years is only 54 monsoons and perhaps ten strong El Niños.** Climatology is
well estimated; the ENSO relationship is not. The covariate ablation above — ENSO
buying ~1 point of skill — should be read in that light rather than as proof the
signal is absent.

## Method notes

- **Whole-year holdout**, rolling origin, because adjacent weeks leak.
- **Climatology is refit per origin** on an expanding window, so the reference is
  always out-of-sample. It is Fourier-smoothed (3 harmonics) over week-of-year:
  raw empirical week-of-year quantiles are noisy and would be an artificially weak
  bar.
- Baselines are **probabilistic**, getting quantiles from their own historical error
  distribution per horizon, so they are scored on the same CRPS footing.
- All quantile work is done in `log1p` space and inverted by `expm1`. **Quantiles
  commute with monotone transforms exactly; means do not.** No mean is ever
  back-transformed.
- **RMSE is deliberately not a headline.** Discharge is heavily right-skewed and RMSE
  would be dominated by a few flood weeks.

## QC

Two traps, both documented in `src/ingest/discharge.py`:

- Duplicate timestamps are genuine repeats (median pair ratio 1.01), collapsed by
  median — *not* the same day in two units.
- The extreme tail is **not** percentile-clipped. Values like 6,692 m³/s sit inside
  coherent flood hydrographs and are real. A narrow single-day spike filter (>8×
  both neighbours and >10,000) flags exactly one row: 2018-08-19, recorded 74,713
  between neighbours of 5,251 and 4,713. The cusecs hypothesis fails on context —
  dividing by 35.31 gives 2,116, which would put the flood *peak below its own
  shoulders*. It is an entry error, interpolated, not rescaled.

## TimesFM-3 API notes

Verified against the installed package, because the documentation is misleading:

- Class is `TimesFM3Forecaster` in package `timesfm3`. `TimesFM3Evaluator` is the
  benchmark harness; `timesfm` is v2.5.
- The repo's `timesfm-forecasting/references/api_reference.md` documents the **2.5**
  API (10 columns including a mean) and is wrong for v3. v3 emits **9 quantiles**,
  `[0.1..0.9]`, median at index 4, no mean column.
- **`decode()` silently overrides the `horizon` argument** when past-future
  covariates are present, recomputing it as `pf.shape[-1] - context_len`. A
  covariate window one step short returns a shorter forecast with no warning.
  `timesfm_model.assert_pf_width()` makes this a hard failure.
- `predict_batch()` **silently linear-interpolates NaNs of any gap length**, in
  targets and covariates alike. Gap policy is enforced upstream in `build_panel.py`
  and must never be delegated to the model.
- `max_variates: 32` in the config is **not enforced** anywhere in the package; 50
  variates ran fine. Treat as a soft budget, not a limit.
- Weights are **non-commercial licence** (TimesFM Non-Commercial v1.0). The
  `timesfm3` source is Apache-2.0. This project is research use.

## Reproduce

```bash
pip install -e .

# target 1 - Biligundulu inflow
python src/build_panel.py                  # daily -> tidy weekly panel
python src/backtest.py --no-timesfm        # baselines only, the bar
python src/backtest.py --every 2           # full backtest incl. TimesFM
python src/plots.py discharge              # fan chart + skill figure

# target 2 - TN reservoir storage
python src/build_storage_panel.py          # 493 bulletin PDFs -> weekly panel
python src/backtest_storage.py --every 2
python src/plots.py storage

python src/calibration.py                  # coverage / PIT, both targets
```

Ingest for the storage target must run with the tool sandbox disabled —
`cwc.gov.in` is unreachable from inside it. NWDP, Open-Meteo, CPC and PSL are all
reachable sandboxed.

| output | what |
|---|---|
| `skill_overall.csv`, `skill_by_season.csv` | discharge skill, incl. NEM/SWM split |
| `skill_storage_overall.csv`, `skill_storage_by_reservoir.csv` | storage skill |
| `calibration_discharge.csv`, `calibration_storage.csv` | coverage and PIT |
| `forecast_next13.csv`, `forecast_storage_next13.csv` | quantile forecasts |
| `figures/fan_biligundulu.png` | discharge fan chart |
| `figures/fan_tn_storage.png` | six-reservoir fan charts, joint multi-target run |
| `figures/skill_vs_climatology.png`, `figures/skill_storage.png` | skill curves |

Both fan charts are issued from the **end of their archive**, not from today:
2026-01-04 for discharge, **2025-05-08** for storage (the CWC bulletin index stops
there). Neither is a current operational forecast.
