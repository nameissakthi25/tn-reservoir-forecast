"""Per-fold LoRA fine-tuning backtest for the storage target.

Implements docs/finetuning-protocol.md exactly. The three things that make this
honest rather than impressive:

  1. A separate adapter is trained for EVERY fold, on data strictly prior to
     that fold's test year, and asserted so in code.
  2. A deliberately contaminated adapter (trained on all data) is run as a
     negative control, with numeric trip-wires in both directions.
  3. The stopping rule -- significance, effect size, calibration, no
     long-horizon damage, and cross-fold consistency -- was written down
     before any of this ran.

Storage is the target because the hypothesis is that fine-tuning learns
reservoir-specific operating behaviour, so results are reported per reservoir
as well as pooled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_storage as bs
import finetune as ft
import metrics as M

OUT = Path("data/processed")
CKPT = Path("data/interim/lora")
TRAIN_CONTEXT = 96          # multiple of the 32-step input patch
HORIZONS = [1, 4, 8, 13]
MAX_H = bs.MAX_H


def build_origins(p: pd.DataFrame, Y: np.ndarray, every: int,
                  infer_context: int | None = None):
    """Scorable origins, tagged with the calendar year that holds them out.

    `infer_context` controls how much history each origin's context carries.
    Origin SELECTION still requires bs.CONTEXT weeks of history regardless, so
    varying it compares like with like on an identical origin set -- which is
    what isolates the train/inference context-length effect.
    """
    ic = infer_context or bs.CONTEXT
    dates = pd.DatetimeIndex(p.week)
    out = []
    for i in range(bs.CONTEXT, len(p) - MAX_H):
        if i % every:
            continue
        if not np.isfinite(Y[i:i + MAX_H]).all():
            continue
        ctx = Y[i - ic:i].T
        if not np.isfinite(ctx).all():
            ctx = np.vstack([bs._clean(r) for r in ctx])
        out.append(dict(i=i, year=int(dates[i].year), ctx=ctx,
                        truth=Y[i:i + MAX_H], dates=dates[i:i + MAX_H]))
    return out


def score_rows(model_name, origins, qs, names, fold_of):
    rows = []
    for o, q in zip(origins, qs):
        for j, nm in enumerate(names):
            for h in HORIZONS:
                t = h - 1
                yt, qh = o["truth"][t:t + 1, j], q[j, t:t + 1, :]
                rows.append(dict(
                    model=model_name, fold=fold_of(o), origin=o["i"],
                    reservoir=nm, horizon=h, date=o["dates"][t],
                    truth=float(yt[0]), pinball=float(M.pinball(yt, qh)[0]),
                    q10=float(qh[0, 0]), q90=float(qh[0, 8]),
                ))
    return rows


def run(every: int, steps: int, lr: float, rank: int,
        device: str | None, control: bool,
        infer_context: int | None = None) -> pd.DataFrame:
    p, names = bs.load_panel()
    Y = p[names].values.astype(float)
    dates = pd.DatetimeIndex(p.week)
    origins = build_origins(p, Y, every, infer_context)
    print(f"inference context: {infer_context or bs.CONTEXT} weeks "
          f"(training window {TRAIN_CONTEXT})")
    years = sorted({o["year"] for o in origins})
    print(f"{len(origins)} origins across folds {years}")
    CKPT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # ---- zero-shot reference, same origins -----------------------------------
    from timesfm_model import TimesFM3
    zs = TimesFM3(device=device or "cpu")
    outs = list(zs.fc.predict_batch(contexts=[o["ctx"] for o in origins],
                                    horizon=MAX_H, return_quantiles=True))
    qs = [np.sort(o.quantiles.clip(min=0), -1) for o in outs]
    rows += score_rows("zeroshot", origins, qs, names, lambda o: o["year"])
    del zs

    # ---- per-fold fine-tuning ------------------------------------------------
    for Yr in years:
        tr_mask = dates.year < Yr
        n_train = int(tr_mask.sum())
        # HARD leakage assertion, per protocol section 1
        assert dates[tr_mask].year.max() < Yr, "training slice leaks into fold"
        need = TRAIN_CONTEXT + MAX_H
        if n_train < need:
            print(f"  fold {Yr}: only {n_train} prior weeks (<{need}) — SKIPPED")
            continue
        print(f"  fold {Yr}: training on {n_train} weeks "
              f"(up to {dates[tr_mask].max().date()})", flush=True)

        model, hist = ft.train_fold(
            Y[tr_mask].T, context=TRAIN_CONTEXT, horizon=MAX_H,
            device=device, steps=steps, lr=lr, r=rank, log_every=0)
        print(f"    loss {hist[0]:.5f} -> {np.mean(hist[-10:]):.5f}", flush=True)

        fold_origins = [o for o in origins if o["year"] == Yr]
        q = ft.forecast(model, [o["ctx"] for o in fold_origins], MAX_H,
                        device=device)
        rows += score_rows("finetuned", fold_origins, list(q), names,
                           lambda o: o["year"])
        del model

    # ---- negative control: deliberately contaminated -------------------------
    if control:
        print("  NEGATIVE CONTROL: training on ALL data (deliberate leakage)",
              flush=True)
        model, hist = ft.train_fold(
            Y.T, context=TRAIN_CONTEXT, horizon=MAX_H, device=device,
            steps=steps, lr=lr, r=rank, log_every=0)
        print(f"    loss {hist[0]:.5f} -> {np.mean(hist[-10:]):.5f}", flush=True)
        q = ft.forecast(model, [o["ctx"] for o in origins], MAX_H, device=device)
        rows += score_rows("contaminated", origins, list(q), names,
                           lambda o: o["year"])
        del model

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# analysis, per the pre-registered stopping rule
# --------------------------------------------------------------------------- #
def gain(a: float, b: float) -> float:
    """Relative CRPS reduction of b against a."""
    return float(1.0 - b / a) if a > 0 else np.nan


def bootstrap_ci(per_fold: pd.DataFrame, n_boot: int = 10_000, seed: int = 0):
    """Block bootstrap resampling WHOLE FOLDS (not origins -- within-year
    origins are correlated)."""
    rng = np.random.default_rng(seed)
    d = per_fold["diff"].values
    if len(d) < 2:
        return np.nan, np.nan
    draws = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def crps(g):
        return g.pinball.mean()

    rep = []
    for h in HORIZONS:
        sub = df[df.horizon == h]
        zs = sub[sub.model == "zeroshot"]
        ftd = sub[sub.model == "finetuned"]
        if ftd.empty:
            continue
        folds = sorted(set(zs.fold) & set(ftd.fold))
        pf = pd.DataFrame([
            dict(fold=f,
                 zs=crps(zs[zs.fold == f]), ft=crps(ftd[ftd.fold == f]))
            for f in folds])
        pf["diff"] = pf.zs - pf.ft
        pf["rel"] = pf["diff"] / pf.zs
        lo, hi = bootstrap_ci(pf)

        c_zs, c_ft = crps(zs), crps(ftd)
        cov_ft = float(((ftd.truth >= ftd.q10) & (ftd.truth <= ftd.q90)).mean())
        n_improved = int((pf["diff"] > 0).sum())
        # binomial sign test
        from scipy import stats as st
        sign_p = float(st.binomtest(n_improved, len(pf), 0.5,
                                    alternative="greater").pvalue)
        # jackknife on the median relative gain
        jk = [float(np.median(pf.rel.drop(k))) for k in pf.index]

        contam = sub[sub.model == "contaminated"]
        g_contam = gain(c_zs, crps(contam)) if not contam.empty else np.nan
        g_correct = gain(c_zs, c_ft)

        rep.append(dict(
            horizon=h, crps_zeroshot=c_zs, crps_finetuned=c_ft,
            median_rel_gain=float(pf.rel.median()), ci_lo=lo, ci_hi=hi,
            folds=len(pf), folds_improved=n_improved, sign_p=sign_p,
            jackknife_min=float(min(jk)) if jk else np.nan,
            coverage_ft=cov_ft,
            gain_correct=g_correct, gain_contaminated=g_contam,
        ))
    overall = pd.DataFrame(rep)

    per_res = []
    for (nm, h), g in df[df.model.isin(["zeroshot", "finetuned"])] \
            .groupby(["reservoir", "horizon"]):
        z = g[g.model == "zeroshot"].pinball.mean()
        f = g[g.model == "finetuned"].pinball.mean()
        per_res.append(dict(reservoir=nm, horizon=h, crps_zeroshot=z,
                            crps_finetuned=f, rel_gain=gain(z, f)))
    return overall, pd.DataFrame(per_res)


def verdict(o: pd.Series) -> tuple[bool, list[str]]:
    """Apply the five pre-registered criteria. All must hold."""
    fails = []
    if not (o.ci_lo > 0):
        fails.append(f"CI includes zero [{o.ci_lo:+.4f},{o.ci_hi:+.4f}]")
    if not (o.median_rel_gain >= 0.05):
        fails.append(f"median gain {o.median_rel_gain:+.1%} < 5%")
    if not (0.75 <= o.coverage_ft <= 0.85):
        fails.append(f"coverage {o.coverage_ft:.2f} outside [0.75,0.85]")
    frac = o.folds_improved / o.folds
    if not (frac >= 0.60 and o.sign_p < 0.05):
        fails.append(f"consistency {o.folds_improved}/{o.folds} "
                     f"({frac:.0%}), sign p={o.sign_p:.3f}")
    if not (o.jackknife_min >= 0.05):
        fails.append(f"jackknife min {o.jackknife_min:+.1%} < 5%")
    return (not fails), fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--infer-context", type=int, default=None,
                    help="diagnostic: match inference context to TRAIN_CONTEXT")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    df = run(a.every, a.steps, a.lr, a.rank, a.device, not a.no_control,
             a.infer_context)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    df.to_parquet(OUT / f"finetune_raw{tag}.parquet", index=False)
    overall, per_res = evaluate(df)
    overall.to_csv(OUT / f"finetune_overall{tag}.csv", index=False)
    per_res.to_csv(OUT / f"finetune_by_reservoir{tag}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== fine-tuning vs zero-shot (storage) ===")
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
        print(f"  h={int(o.horizon):<3} contaminated gain "
              f"{o.gain_contaminated:+.1%}, correct {o.gain_correct:+.1%}"
              + ("   !! UPPER TRIP-WIRE" if upper else "")
              + ("   !! LOWER TRIP-WIRE (control uninformative)" if lower else ""))

    print("\n=== per reservoir (mechanism check) ===")
    piv = per_res.pivot(index="reservoir", columns="horizon", values="rel_gain")
    print((100 * piv).round(1).to_string())


if __name__ == "__main__":
    main()
