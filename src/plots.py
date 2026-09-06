"""Phase 5 outputs: fan chart and skill table."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import backtest as bt
from timesfm_model import TimesFM3

OUT = Path("data/processed")
FIG = Path("figures")


def fan_chart(start_wy: int = 1991, device: str = "cpu") -> None:
    p = bt.load_panel(start_wy)
    y = p[bt.TARGET].values.astype(float)
    dates = pd.DatetimeIndex(p.week)

    i = len(p)                       # forecast from the end of the record
    ctx = bt._clean(y[i - bt.CONTEXT:i])
    po = np.vstack([bt._clean(p[c].values[i - bt.CONTEXT:i]) for c in bt.PAST_ONLY])

    # future covariate rows must extend exactly MAX_H beyond the context
    fut_idx = pd.date_range(dates[-1] + pd.Timedelta(weeks=1),
                            periods=bt.MAX_H, freq="W-SUN")
    woy = fut_idx.isocalendar().week.values.astype(float)
    sched = {6: 9.19, 7: 31.24, 8: 45.95, 9: 36.76, 10: 20.22, 11: 13.78,
             12: 7.35, 1: 2.76, 2: 2.20, 3: 2.20, 4: 2.20, 5: 2.20}
    fut = {
        "woy_sin": np.sin(2 * np.pi * woy / 52.0),
        "woy_cos": np.cos(2 * np.pi * woy / 52.0),
        "schedule_tmc": np.array([sched[m] for m in fut_idx.month], float),
        "regime_idx": np.full(bt.MAX_H, 3.0),
    }
    pf = np.vstack([np.r_[bt._clean(p[c].values[i - bt.CONTEXT:i]), fut[c]]
                    for c in bt.PAST_FUTURE])

    m = TimesFM3(device=device)
    q = m.forecast([ctx], bt.MAX_H, past_only=[po], past_future=[pf])[0]

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = slice(-104, None)
    ax.plot(dates[hist], y[hist], color="#222", lw=1.2, label="observed")
    ax.plot(fut_idx, q[:, 4], color="#c1121f", lw=2, label="median forecast")
    for lo, hi, a in [(0, 8, 0.15), (1, 7, 0.2), (2, 6, 0.25), (3, 5, 0.3)]:
        ax.fill_between(fut_idx, q[:, lo], q[:, hi], color="#c1121f", alpha=a, lw=0)
    ax.set_title("Cauvery discharge at Biligundulu — 13-week forecast\n"
                 "median with P10–P90 band (TimesFM-3, covariates)")
    ax.set_ylabel("discharge (m³/s)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fan_biligundulu.png", dpi=150)
    print(f"wrote {FIG/'fan_biligundulu.png'}  origin={dates[-1].date()}")
    pd.DataFrame(q, index=fut_idx,
                 columns=[f"q{int(x*100)}" for x in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
                 ).to_csv(OUT / "forecast_next13.csv")


def skill_plot() -> None:
    df = pd.read_csv(OUT / "skill_overall.csv")
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in df.groupby("model"):
        if name == "climatology":
            continue
        ax.plot(g.horizon, g.skill_vs_clim, marker="o", label=name)
    ax.axhline(0, color="#444", lw=1.2, ls="--")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("CRPS skill vs climatology")
    ax.set_title("Skill against smoothed climatology (0 = ties, >0 = beats)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "skill_vs_climatology.png", dpi=150)
    print(f"wrote {FIG/'skill_vs_climatology.png'}")


def storage_fan_charts(device: str = "cpu") -> None:
    """Small-multiple fan charts for the six usable TN reservoirs.

    All six are forecast in a SINGLE multi-target TimesFM call, the way the
    backtest scores them -- they co-evolve through shared monsoon forcing, and
    forecasting them jointly is the point of the multivariate mode.
    """
    import yaml
    import backtest_storage as bs

    p, names = bs.load_panel()
    Y = p[names].values.astype(float)
    dates = pd.DatetimeIndex(p.week)
    i = len(p)

    ctx = np.vstack([bs._clean(r) for r in Y[i - bs.CONTEXT:i].T])
    po = np.vstack([bs._clean(p[c].values[i - bs.CONTEXT:i]) for c in bs.PAST_ONLY])

    fut_idx = pd.date_range(dates[-1] + pd.Timedelta(weeks=1),
                            periods=bs.MAX_H, freq="W-THU")
    woy = fut_idx.isocalendar().week.values.astype(float)
    month = fut_idx.month.values
    fut = {
        "woy_sin": np.sin(2 * np.pi * woy / 52.0),
        "woy_cos": np.cos(2 * np.pi * woy / 52.0),
        "is_swm": ((month >= 6) & (month <= 9)).astype(float),
        "is_nem": ((month >= 10) & (month <= 12)).astype(float),
    }
    pf = np.vstack([np.r_[bs._clean(p[c].values[i - bs.CONTEXT:i]), fut[c]]
                    for c in bs.PAST_FUTURE])

    from timesfm_model import TimesFM3, assert_pf_width
    assert_pf_width(pf, ctx.shape[-1], bs.MAX_H)
    m = TimesFM3(device=device)
    out = list(m.fc.predict_batch(contexts=[ctx], horizon=bs.MAX_H,
                                  past_only_covariates=[po],
                                  past_future_covariates=[pf],
                                  return_quantiles=True))[0]
    Q = np.sort(out.quantiles.clip(min=0), axis=-1)     # (6, 13, 9)

    caps = {r["name"]: r["live_cap_bcm"]
            for r in yaml.safe_load(Path("config/reservoirs.yaml").read_text())
            ["reservoirs"]}

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    hist = slice(-104, None)
    for ax, nm, j in zip(axes.ravel(), names, range(len(names))):
        ax.plot(dates[hist], Y[hist, j], color="#222", lw=1.2)
        ax.plot(fut_idx, Q[j, :, 4], color="#1d4e89", lw=2)
        for lo, hi, a in [(0, 8, 0.15), (1, 7, 0.2), (2, 6, 0.25), (3, 5, 0.3)]:
            ax.fill_between(fut_idx, Q[j, :, lo], Q[j, :, hi],
                            color="#1d4e89", alpha=a, lw=0)
        cap = caps.get(nm)
        if cap:
            ax.axhline(cap, color="#c1121f", ls=":", lw=1)
            ax.text(dates[hist].start if False else dates[-104], cap,
                    " live capacity", va="bottom", ha="left",
                    fontsize=7, color="#c1121f")
        ax.set_title(nm, fontsize=11)
        ax.set_ylabel("BCM", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel():
        for lb in ax.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha("right")
    fig.suptitle("TN reservoir storage — 13-week forecast from "
                 f"{dates[-1].date()}  (median, P10–P90; joint multi-target run)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fan_tn_storage.png", dpi=150)
    print(f"wrote {FIG/'fan_tn_storage.png'}  origin={dates[-1].date()}")

    rows = []
    for j, nm in enumerate(names):
        for t, d in enumerate(fut_idx):
            rows.append(dict(reservoir=nm, week=d,
                             **{f"q{int(l*100)}": Q[j, t, k]
                                for k, l in enumerate([.1,.2,.3,.4,.5,.6,.7,.8,.9])}))
    pd.DataFrame(rows).to_csv(OUT / "forecast_storage_next13.csv", index=False)


def storage_skill_plot() -> None:
    df = pd.read_csv(OUT / "skill_storage_overall.csv")
    piv = df.pivot(index="horizon", columns="model", values="crps")
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ref, style in (("climatology", "-o"), ("persistence", "--s")):
        ax.plot(piv.index, 1 - piv["timesfm_covariates"] / piv[ref], style,
                label=f"TimesFM vs {ref}")
    ax.axhline(0, color="#444", lw=1.2, ls=":")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("CRPS skill")
    ax.set_title("Storage: skill depends entirely on which baseline you pick")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "skill_storage.png", dpi=150)
    print(f"wrote {FIG/'skill_storage.png'}")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "discharge"):
        skill_plot()
        fan_chart()
    if which in ("all", "storage"):
        storage_skill_plot()
        storage_fan_charts()
