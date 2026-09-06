"""Per-fold LoRA fine-tuning backtest for the DISCHARGE target (Biligundulu).

Reuses the analysis machinery in backtest_finetune.py (bootstrap, the five
pre-registered criteria, the trip-wires) unchanged; only the panel, the target
shape and the transform differ.

Three design choices, all forced and all stated rather than buried:

  1. TRAINING IS RESTRICTED TO 1991+, like every other model in this study.
     Pre-1991 is the no-binding-allocation regime and a different generating
     process (dry-season flow roughly halves after the 2007 award while annual
     volume is unchanged). Fine-tuning IS training, so letting it see pre-1991
     data would contradict the window decision the rest of the project made.

  2. TEST FOLDS START AT 2001, not 1991. A fold trained only on 1991..1990 has
     no data at all; folds before 2001 would be data-starved to the point of
     meaninglessness. Starting at 2001 guarantees every fold at least ten years
     of same-regime training data, and still yields 25 held-out years -- well
     past the spec's ">= 10".

  3. TRAIN AND INFERENCE CONTEXTS ARE MATCHED AT 256 WEEKS. The main discharge
     backtest infers from 520, but a 520-week training window plus a 13-week
     horizon needs 533 weeks and the earliest folds do not have that, which
     would silently skip folds. 256 (a multiple of the 32-step input patch)
     leaves ~250 windows even at the first fold. Matching inference to it
     removes the train/inference confound from the start rather than testing it
     afterwards. Absolute CRPS therefore differs from the headline backtest;
     the fine-tuned-vs-zero-shot comparison, which is the actual question, is
     internally consistent.

Transform is log1p throughout, matching the canonical discharge configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import finetune as ft
import metrics as M
from backtest_finetune import HORIZONS, evaluate, score_rows, verdict

OUT = Path("data/processed")
TRAIN_CONTEXT = 256          # multiple of the 32-step input patch
INFER_CONTEXT = 256          # matched, by design -- see module docstring
TRAIN_START_WY = 1991
TEST_START_WY = 2001
MAX_H = bt.MAX_H
TRANSFORM = "log1p"


def build_origins(p: pd.DataFrame, y: np.ndarray, every: int):
    dates = pd.DatetimeIndex(p.week)
    out = []
    for i in range(INFER_CONTEXT, len(p) - MAX_H):
        if i % every:
            continue
        wy = int(p.water_year.iloc[i])
        if wy < TEST_START_WY:
            continue
        truth = y[i:i + MAX_H]
        if not np.isfinite(truth).all():
            continue
        ctx = y[i - INFER_CONTEXT:i]
        if not np.isfinite(ctx).all():
            ctx = bt._clean(ctx)
        out.append(dict(i=i, year=wy, ctx=np.atleast_2d(ctx),
                        truth=truth[:, None], dates=dates[i:i + MAX_H]))
    return out


def run(every: int, steps: int, lr: float, rank: int,
        device: str | None, control: bool) -> pd.DataFrame:
    p = bt.load_panel(TRAIN_START_WY)
    y = p[bt.TARGET].values.astype(float)
    wy = p.water_year.values.astype(int)

    origins = build_origins(p, y, every)
    years = sorted({o["year"] for o in origins})
    print(f"train context {TRAIN_CONTEXT} | infer context {INFER_CONTEXT} | "
          f"transform {TRANSFORM}")
    print(f"{len(origins)} origins across {len(years)} folds "
          f"{years[0]}-{years[-1]}", flush=True)

    names = ["Biligundulu"]
    rows: list[dict] = []

    # ---- zero-shot reference, identical origins and transform ---------------
    from timesfm_model import TimesFM3
    zs = TimesFM3(device=device or "cpu")
    q = zs.forecast([o["ctx"] for o in origins], MAX_H, transform=TRANSFORM)
    # TimesFM3.forecast squeezes the variate axis (it takes quantiles[0]), so it
    # returns (n, horizon, 9). score_rows indexes a variate axis, and ft.forecast
    # keeps one -- restore it here so both arms have the same shape.
    rows += score_rows("zeroshot", origins, list(q[:, None, :, :]), names,
                       lambda o: o["year"])
    del zs
    print("  zero-shot reference done", flush=True)

    # ---- per-fold fine-tuning ------------------------------------------------
    for Yr in years:
        tr = (wy >= TRAIN_START_WY) & (wy < Yr)
        n_train = int(tr.sum())
        assert wy[tr].max() < Yr, "training slice leaks into fold"
        assert wy[tr].min() >= TRAIN_START_WY, "training slice predates 1991"
        need = TRAIN_CONTEXT + MAX_H
        if n_train < need:
            print(f"  fold {Yr}: only {n_train} weeks (<{need}) — SKIPPED",
                  flush=True)
            continue
        print(f"  fold {Yr}: {n_train} weeks of 1991+ history", flush=True)

        model, hist = ft.train_fold(
            y[tr][None, :], context=TRAIN_CONTEXT, horizon=MAX_H,
            device=device, steps=steps, lr=lr, r=rank,
            transform=TRANSFORM, log_every=0)
        print(f"    loss {hist[0]:.5f} -> {np.mean(hist[-10:]):.5f}", flush=True)

        fo = [o for o in origins if o["year"] == Yr]
        qf = ft.forecast(model, [o["ctx"] for o in fo], MAX_H,
                         device=device, transform=TRANSFORM)
        rows += score_rows("finetuned", fo, list(qf), names,
                           lambda o: o["year"])
        del model

    # ---- negative control ----------------------------------------------------
    if control:
        allmask = wy >= TRAIN_START_WY
        print("  NEGATIVE CONTROL: training on ALL 1991+ data", flush=True)
        model, hist = ft.train_fold(
            y[allmask][None, :], context=TRAIN_CONTEXT, horizon=MAX_H,
            device=device, steps=steps, lr=lr, r=rank,
            transform=TRANSFORM, log_every=0)
        print(f"    loss {hist[0]:.5f} -> {np.mean(hist[-10:]):.5f}", flush=True)
        qc = ft.forecast(model, [o["ctx"] for o in origins], MAX_H,
                         device=device, transform=TRANSFORM)
        rows += score_rows("contaminated", origins, list(qc), names,
                           lambda o: o["year"])
        del model

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args()

    df = run(a.every, a.steps, a.lr, a.rank, a.device, not a.no_control)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "finetune_discharge_raw.parquet", index=False)
    overall, per_res = evaluate(df)
    overall.to_csv(OUT / "finetune_discharge_overall.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== DISCHARGE: fine-tuning vs zero-shot ===")
    print(overall.round(4).to_string(index=False))

    print("\n=== pre-registered verdict ===")
    for _, o in overall.iterrows():
        ok, fails = verdict(o)
        print(f"  h={int(o.horizon):<3} {'PASS' if ok else 'FAIL'}"
              + ("" if ok else "   " + "; ".join(fails)))

    print("\n=== negative control trip-wires ===")
    for _, o in overall.iterrows():
        if not np.isfinite(o.gain_contaminated):
            continue
        upper = o.gain_correct > 0.5 * o.gain_contaminated
        lower = o.gain_contaminated < 0.15
        print(f"  h={int(o.horizon):<3} contaminated {o.gain_contaminated:+.1%}, "
              f"correct {o.gain_correct:+.1%}"
              + ("   !! UPPER TRIP-WIRE" if upper else "")
              + ("   !! LOWER TRIP-WIRE (control uninformative)" if lower else ""))


if __name__ == "__main__":
    main()
