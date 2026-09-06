"""Tune LoRA hyperparameters against the CONTAMINATED control only.

Rationale (docs/finetuning-protocol.md, Outcome 2): the discharge fine-tuning
test came back uninterpretable because the contaminated control never cleared
its 15% floor -- LoRA was not fitting the target even when handed the answers.
Until that is fixed, no honest fold can be read.

The control is the correct thing to tune against because it carries NO leakage
concern: it trains on all data and is scored on that same data, so it is a pure
measure of learning CAPACITY. Optimising it cannot contaminate the real
experiment -- the honest per-fold run is a separate execution that never sees
these weights.

Once a config clears 15%, that config (and only that config) is carried over to
the per-fold run, which is where leakage discipline applies.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import backtest_finetune_discharge as fd
import finetune as ft
import metrics as M

OUT = Path("data/processed")
HORIZONS = fd.HORIZONS
MAX_H = fd.MAX_H
FLOOR = 0.15


def crps_by_h(origins, qs) -> dict[int, float]:
    out = {}
    for h in HORIZONS:
        t = h - 1
        vals = [float(M.pinball(o["truth"][t:t + 1, 0], q[0, t:t + 1, :])[0])
                for o, q in zip(origins, qs)]
        out[h] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=8, help="origin spacing (sweep uses coarse)")
    ap.add_argument("--lrs", type=float, nargs="*", default=[1e-4])
    ap.add_argument("--steps", type=int, nargs="*", default=[300])
    ap.add_argument("--ranks", type=int, nargs="*", default=[8])
    ap.add_argument("--batch", type=int, nargs="*", default=[8])
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()

    p = bt.load_panel(fd.TRAIN_START_WY)
    y = p[bt.TARGET].values.astype(float)
    wy = p.water_year.values.astype(int)
    origins = fd.build_origins(p, y, a.every)
    allmask = wy >= fd.TRAIN_START_WY
    print(f"{len(origins)} origins | control trains on {int(allmask.sum())} weeks "
          f"| floor {FLOOR:.0%}", flush=True)

    # ---- zero-shot reference, computed once -------------------------------
    from timesfm_model import TimesFM3
    zs = TimesFM3(device=a.device or "cpu")
    q0 = zs.forecast([o["ctx"] for o in origins], MAX_H, transform=fd.TRANSFORM)
    base = crps_by_h(origins, list(q0[:, None, :, :]))
    del zs
    print("zero-shot CRPS:", {h: round(v, 2) for h, v in base.items()}, flush=True)

    rows = []
    grid = list(itertools.product(a.lrs, a.steps, a.ranks, a.batch))
    for i, (lr, steps, rank, batch) in enumerate(grid, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(grid)}] lr={lr:g} steps={steps} r={rank} bs={batch}",
              flush=True)
        try:
            model, hist = ft.train_fold(
                y[allmask][None, :], context=fd.TRAIN_CONTEXT, horizon=MAX_H,
                device=a.device, steps=steps, lr=lr, r=rank,
                batch_size=batch, transform=fd.TRANSFORM, log_every=0)
        except FloatingPointError as e:
            print(f"    DIVERGED: {e}", flush=True)
            rows.append(dict(lr=lr, steps=steps, rank=rank, batch=batch,
                             diverged=True))
            continue

        # honest loss trend: mean of first 20 vs last 20 steps, not step 0 vs tail
        k = min(20, len(hist) // 4) or 1
        first, last = float(np.mean(hist[:k])), float(np.mean(hist[-k:]))
        qf = ft.forecast(model, [o["ctx"] for o in origins], MAX_H,
                         device=a.device, transform=fd.TRANSFORM)
        cur = crps_by_h(origins, list(qf))
        del model

        gains = {h: 1.0 - cur[h] / base[h] for h in HORIZONS}
        best = max(gains.values())
        row = dict(lr=lr, steps=steps, rank=rank, batch=batch, diverged=False,
                   loss_first=first, loss_last=last,
                   loss_drop=(first - last) / first,
                   **{f"gain_h{h}": gains[h] for h in HORIZONS},
                   best_gain=best, clears=bool(best >= FLOOR),
                   secs=round(time.time() - t0))
        rows.append(row)
        print(f"    loss {first:.5f} -> {last:.5f} ({100*(first-last)/first:+.1f}%)"
              f" | gains " + " ".join(f"h{h}:{gains[h]:+.1%}" for h in HORIZONS)
              + f" | {'CLEARS' if best >= FLOOR else 'below floor'}"
              f" | {row['secs']}s", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"control_{a.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    ok = df[df.get("clears", False) == True] if "clears" in df else df.iloc[0:0]
    print(f"\n=== sweep {a.tag}: {len(ok)} of {len(df)} configs clear {FLOOR:.0%} ===")
    cols = [c for c in ["lr", "steps", "rank", "batch", "loss_drop",
                        "gain_h1", "gain_h4", "gain_h8", "gain_h13",
                        "best_gain", "clears", "secs"] if c in df]
    print(df[cols].sort_values("best_gain", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
