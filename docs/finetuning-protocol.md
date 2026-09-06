# Fine-tuning protocol — pre-registered

> **OUTCOME (recorded 2026-09-04, after running).** Every criterion failed at every
> horizon on the storage target. Coverage collapsed to 0.52–0.74 against the
> [0.75, 0.85] requirement; CRPS gains were −6.7% to +1.2%. The train/inference
> context confound was tested and eliminated (coverage moved <= 0.02 when matched).
> The mechanistic prediction below was **refuted** — gains concentrated in
> rainfall-driven Parambikulam, not release-governed Mettur — so per the rule
> written here, the operating-rule explanation is withdrawn rather than retrofitted.
> The lower trip-wire fired on the first run and correctly blocked a meaningless
> result caused by NaN-poisoned adapters. Results in README.md; raw output in
> `data/processed/finetune_overall.csv` and `finetune_overall_ctx96.csv`.
> Untested and therefore still open: hyperparameters, and early stopping on
> validation coverage rather than loss.
>
> **OUTCOME 2 — DISCHARGE (recorded 2026-09-06).** 25 folds (2001-2025), contexts
> matched at 256 by design. **The lower trip-wire fired at ALL FOUR horizons**
> (contaminated gain +2.6 / +7.1 / +8.8 / +5.9%, all below the 15% floor), so per
> this protocol every horizon is **UNINTERPRETABLE** — fine-tuning is untested on
> discharge, not disproven. The control's own training loss *rose*
> (0.17906 -> 0.18154), as did the largest fold's: LoRA at lr 1e-4 / 300 steps is
> not fitting this target even when handed the answers.
>
> Three things the run still established:
> 1. **An upper bound.** Even with full contamination LoRA extracted <= 8.8%,
>    versus +30% on storage. Discharge is a spiky flux over 1,774 weeks; storage is
>    six smooth curves. That bound independently corroborates the main study's
>    information-limit finding.
> 2. **The calibration collapse is storage-specific.** Discharge coverage held at
>    0.72-0.75 (from 0.76-0.83 zero-shot) against storage's collapse to 0.52. This
>    *reverses* the prior expectation that a repeat would indict LoRA generally.
> 3. **A directional signal survives.** h=13 consistency was 5/25 folds improving,
>    sign test against the null of no harm, p=0.002 — strong evidence toward
>    fine-tuning hurting at long horizon,
>    even though the formal verdict is blocked.
>
> **Next attempt has a concrete precondition, not a blind search: raise learning
> rate / steps / rank until the CONTAMINATED CONTROL clears 15%, then read the
> honest folds.** The control measures learning capacity with no leakage concern,
> so it can be tuned against freely. Results in
> `data/processed/finetune_discharge_overall.csv`.
>
> **CORRECTION (2026-09-06), issued the same day.** The sentence above is wrong as
> a *selection* rule, and the tuning run it authorised was stopped part-way for
> that reason. See section 4.

---

## 4. Where this stopped, and the design error to avoid repeating

**Status: fine-tuning is UNTESTED on discharge and FAILED on storage. Work stopped
here deliberately.** Nothing published depends on the discharge result.

### What the tuning run did establish (keep this)

Sweeping the contaminated control answered the question that blocked Outcome 2 —
*is LoRA broken on this target, or is the target unlearnable?*

| lr (300 steps, r=8) | loss drop | h=1 | h=4 | h=8 | h=13 |
|---|---|---|---|---|---|
| 3e-5 | 10.1% | +0.8% | +0.6% | +1.8% | +0.7% |
| 1e-4 *(original)* | 16.0% | +3.9% | +3.7% | +10.7% | +2.4% |
| 3e-4 | 32.5% | +11.0% | +13.2% | +40.5% | +20.7% |
| 1e-3 | 43.1% | +12.1% | +13.6% | +53.7% | +36.0% |

At lr=1e-3 with **600** steps every horizon clears (+26 / +32 / +68 / +52%).

So **LoRA works on discharge; the original run under-fit on two axes at once**
(learning rate and schedule length). Two incidental corrections: the "rising loss =
instability" reading in Outcome 2 was an artifact of comparing a single step against
a tail mean — properly measured, loss falls monotonically with lr. And h=1 is not
intrinsically un-improvable; it responds to *steps*, not lr.

### The design error (do not repeat)

**Tuning hyperparameters against the contaminated control, then carrying them to the
honest run, is close to circular.** The control's job is to memorise the test years.
A config that memorises well — high lr, long schedule, ample rank — is by
construction a config prone to overfitting. Selecting on it means choosing the
settings most likely to make the honest folds fail, and then reporting that failure
as evidence about fine-tuning. A negative result obtained that way cannot separate
"fine-tuning does not help" from "I picked overfitting-prone settings".

Two compounding problems: the 15% floor is arbitrary (it was written here without
justification), and **overshooting it weakens the leak detector** — the upper
trip-wire is `gain_correct > 0.5 * gain_contaminated`, so a control at +68% permits
an honest gain of +34% to pass unflagged.

### The correct design

1. **Demote the control back to a pass/fail validity gate.** That is all section 1
   ever claimed for it: an instrument that cannot detect cheating cannot certify
   honesty. It is not a selection criterion.
2. **Select hyperparameters by nested validation.** Inside each fold's training
   slice, hold out the most recent year as validation; choose the config with the
   best validation CRPS; evaluate on the test year. This selects for generalisation,
   never touches the test year, and is not circular.
3. Keep the grid small — the sweep already narrows it to lr 3e-4–1e-3, 450–600
   steps, rank 4–8.
4. Cost is roughly one training run per config per fold: ~3-4 configs x 25 folds.

Sweep results retained in `data/processed/control_lr.csv` and `control_min.csv` as a
*diagnostic* record of learning capacity — not as a selection.

**Status: written before any fine-tuning was run.** The stopping rule below is
committed in advance for the same reason the baselines were built before the
model: a bar decided after seeing results is not a bar.

Date: 2026-09-04. Applies to both targets.

---

## 1. The leakage constraint (non-negotiable)

**A single fine-tune performed before a rolling-origin backtest invalidates every
number the backtest produces.**

If the model is fine-tuned once on the full series and then backtested, every test
year is already inside the weights. Skill scores will jump substantially and mean
nothing.

This failure is *harder to detect* than the ENSO-covariate version of leakage. There,
the cheat is visible in an input array — future values sitting in a covariate row.
Here there is no suspicious array at all. The inputs look correct at every origin;
the contamination is in the checkpoint. Nothing in the backtest output distinguishes
it from a genuine result. It would simply look like a triumph.

### Required design

Fine-tuning must be repeated **per fold**, on data strictly prior to that fold:

```
for each held-out test year Y:
    train_slice = all weeks with water_year < Y
    ckpt_Y      = lora_finetune(base_model, train_slice)
    forecast all origins in Y using ckpt_Y   # never any other checkpoint
```

- ~35 fine-tune runs for discharge (1991–2025), ~6 for storage.
- No checkpoint may ever be used on a year at or before the one it saw.
- The expanding window must match the one the climatology baseline already uses, so
  model and reference see identical information at each origin.
- Cost is ~35x a naive run. That is the price of the result meaning anything.

### Verification before any result is reported

1. Assert, in code, that `max(train_slice.water_year) < Y` for every fold.
2. Persist the fold index inside each checkpoint filename and log which checkpoint
   produced each forecast row.
3. **Negative control, with a numeric floor.** Deliberately fine-tune one checkpoint
   on *all* data, backtest it, and record its CRPS. "Confirm the jump is large" is
   too vague to act on, so the reading is pre-committed in both directions.

   Define gain relative to zero-shot:
   `gain = 1 - CRPS_run / CRPS_zeroshot`

   - **Upper trip-wire.** If `gain_correct > 0.5 * gain_contaminated`, STOP and
     investigate the fold logic before reporting anything. A correctly-folded run
     recovering more than half the benefit of outright cheating is the signature of
     partial leakage, not of a good model.
   - **Lower trip-wire.** If `gain_contaminated < 0.15` (i.e. even full
     contamination buys under 15%), the control has failed as an instrument: either
     LoRA is barely training or the contamination did not take. Do not interpret any
     correctly-folded result until that is resolved — a control that cannot detect
     cheating cannot certify honesty.

   Both trip-wires are recorded per horizon, since contamination will inflate the
   long horizons most (those are where the model has least genuine signal and most
   to gain from having seen the answer).

---

## 2. Pre-registered stopping rule

Fine-tuning on ~2,836 weekly points (discharge) or ~3,100 (storage, 6 x 526) against
a 330M-parameter model will produce *some* movement in every cell. Most of it will be
fold-level noise. The following decides in advance what counts as a result.

### Primary comparison

**Fine-tuned vs zero-shot TimesFM**, same covariates, same folds, same origins. Not
against climatology or persistence — those answer a different question already
answered. The question here is solely: *does adapting the model to this data help?*

### Metric and horizons

CRPS, reported **separately** at h=1, h=4, h=8, h=13. No aggregate across horizons —
the zero-shot result already showed the horizons behave completely differently, and
averaging them would hide exactly the structure that matters.

### Test

Block bootstrap resampling **whole test years** with replacement (not individual
origins — adjacent origins within a year are correlated), 10,000 draws, on the
per-fold CRPS difference.

### Decision thresholds — all must hold

A fine-tuning gain at a given horizon is declared real only if:

1. **Significance:** the 95% bootstrap CI on `CRPS_zeroshot - CRPS_finetuned`
   excludes zero.
2. **Effect size:** median CRPS reduction >= **5%** relative to zero-shot. With
   enough folds, trivial effects reach significance; this floor is what makes the
   claim worth making.
3. **Calibration preserved:** P10-P90 coverage stays within **[0.75, 0.85]**
   (nominal 0.80). Zero-shot currently achieves 0.74-0.81. A fine-tune that buys
   CRPS by becoming overconfident is a regression, not an improvement, and CRPS
   alone will not reveal it.
4. **No long-horizon damage:** CRPS at h=8 and h=13 must not worsen by more than
   3% relative to zero-shot.
5. **Consistency across folds.** Criteria 1 and 2 can both be satisfied by a gain
   driven by two or three unusual years — a median across folds and a CI across
   whole years are both compatible with a handful of folds carrying everything.
   That is a materially different finding from a broad small improvement, and only
   the latter justifies deploying a fine-tuned model. So:
   - **>= 60% of folds must individually improve** (CRPS lower than zero-shot on
     that fold), with a binomial sign test at p < 0.05.
   - **Jackknife:** removing any single fold must not drop the median gain below
     the 5% floor in criterion 2.

   If the gain is significant and large but concentrated in a few folds, it is
   reported as "fine-tuning helps in specific years" — a hypothesis about *which*
   years, to be investigated — and explicitly **not** as a general improvement.

### Stated expectation, recorded in advance

Fine-tuning is expected to help at **h=1 and h=4** and to be **neutral at h=8 and
h=13**, because the long-horizon limit is informational rather than one of model
capacity: at 13 weeks the only available signal is the seasonal cycle, which
climatology already extracts optimally. Teaching the model Cauvery seasonality
teaches it what the reference already knows.

**Storage is the better prospect than discharge.** Reservoir operating rules are
idiosyncratic and repeat — drawdown patterns, fill behaviour — which is learnable
structure a generic pretrained model cannot have. Discharge has less
project-specific structure to learn.

If the outcome contradicts this expectation, the expectation was wrong and the
result stands. That is the point of writing it down first.

### Storage must be reported per reservoir, not pooled

The hypothesis being tested is specifically that fine-tuning learns **reservoir-
specific operating behaviour** — Mettur's drawdown schedule, Sholayar's fill
pattern. Pooled numbers would hide the very signal that confirms or refutes that
mechanism, so all four criteria above are evaluated per reservoir as well as pooled.

Pre-registered mechanistic prediction:

- Gain should **concentrate in the heavily managed reservoirs**. Mettur is the
  clearest case: its filling is governed by Karnataka's releases and its drawdown by
  the kuruvai irrigation calendar, so it carries the most learnable non-hydrological
  structure. Bhavanisagar next.
- Gain should be **near zero for the rainfall-driven reservoirs** — Sholayar, Aliyar,
  Parambikulam are small catchments that largely track rainfall, leaving little
  operating-rule structure to learn.
- **If the gain is uniform across all six, the mechanism is not operating-rule
  learning.** That would be evidence for something more generic — the model simply
  adapting to the scale, smoothness or noise characteristics of weekly BCM series —
  and the explanation in any writeup must change accordingly, even if the numbers
  themselves are good.

This is recorded now precisely because a uniform result would be easy to report
under the operating-rule story after the fact.

### Method

LoRA, not full fine-tuning — ~3k observations against 330M parameters would overfit
outright. Upstream ships `timesfm-forecasting/examples/finetuning/finetune_lora.py`.

---

## 3. Precondition

**Do not start fine-tuning until the storage rainfall-covariate gap is resolved.**

The storage target currently runs with no rainfall covariate at all — only ONI, DMI
and seasonality. Its covariate ablation was therefore never a fair test, and any
conclusion drawn from it is provisional. If adding catchment rainfall moves that
ablation materially, it changes what fine-tuning should be aimed at, and this
protocol should be revisited before the compute is spent.
