"""LoRA fine-tuning for TimesFM-3.

Upstream ships a LoRA example only for TimesFM **2.5**, via HF Transformers +
PEFT. It does not apply to `timesfm3`, whose torch module is a plain nn.Module
with a `decode()` entry point, so the training loop here is bespoke. PEFT's
LoraConfig still wraps it directly: 204 nn.Linear layers, 4.16M trainable
parameters at r=8 (1.26% of 330.7M).

Two design choices worth stating:

  * Loss is the same pinball/quantile loss the model was pretrained with, over
    all 9 quantiles. Training on MSE would break the quantile heads and destroy
    the calibration that the zero-shot model currently has.

  * Training windows use a SHORTER context than inference. With a 208-week
    inference context the earliest storage folds have too little prior history
    to form any window at all, which would silently skip folds the protocol
    requires. TimesFM handles variable context natively (left-pad + mask), so
    training at 104 weeks and inferring at 208 is safe. Flagged rather than
    hidden because it is a train/inference mismatch.

See docs/finetuning-protocol.md for the leakage constraint and the
pre-registered stopping rule. Nothing here may be run on a fold's own years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WindowDataset(Dataset):
    """Sliding (context, horizon) windows over a multivariate training slice.

    y: (n_variates, n_timesteps) -- the slice must already be restricted to
    data strictly prior to the fold being trained for.
    """

    def __init__(self, y: np.ndarray, context: int, horizon: int, stride: int = 1):
        raw = np.atleast_2d(np.asarray(y, dtype=np.float32))
        # A single NaN anywhere in a target window makes the pinball loss NaN,
        # which propagates into the gradients and silently destroys the
        # adapter -- observed as folds 2023+ diverging on one missing week
        # (2022-04-28). Inputs are filled; targets keep a validity mask so
        # imputed points are never trained against.
        self.valid = np.isfinite(raw)
        filled = (pd.DataFrame(raw.T)
                  .interpolate(limit_direction="both").ffill().bfill()
                  .fillna(0.0).values.T.astype(np.float32))
        self.y = filled
        self.context, self.horizon = context, horizon
        n = self.y.shape[1]
        last = n - context - horizon
        if last < 0:
            raise ValueError(
                f"training slice too short: {n} steps, need "
                f">= {context + horizon} for one window")
        self.starts = list(range(0, last + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        a, b = s + self.context, s + self.context + self.horizon
        ctx = self.y[:, s:a]
        fut = self.y[:, a:b]
        msk = self.valid[:, a:b]
        return (torch.from_numpy(ctx.copy()),
                torch.from_numpy(fut.copy()),
                torch.from_numpy(msk.copy()))


def pinball_loss(pred: torch.Tensor, truth: torch.Tensor,
                 valid: torch.Tensor | None = None) -> torch.Tensor:
    """pred (b, u, h, q); truth (b, u, h); valid (b, u, h) bool or None.

    Masked mean, so imputed timesteps contribute no gradient.
    """
    q = QUANTILES.to(pred.device).view(1, 1, 1, -1)
    diff = truth.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    if valid is None:
        return loss.mean()
    w = valid.unsqueeze(-1).to(loss.dtype)
    denom = w.sum() * loss.shape[-1]
    if denom == 0:
        return loss.sum() * 0.0
    return (loss * w).sum() / denom


def decode_with_grad(model, **kw):
    """Call TimesFM3Torch.decode WITH gradients enabled.

    `decode` is hard-decorated `@torch.no_grad()`, so calling it normally
    yields a graph-less tensor and `loss.backward()` raises "element 0 of
    tensors does not require grad". torch's decorator preserves the original
    via functools.wraps, so `__wrapped__` is the undecorated function.

    Reusing decode rather than reimplementing the patching means training and
    inference share one code path -- input padding, masking and the CPM/RevIN
    handling are guaranteed identical, which reimplementing `forward()`'s
    patched interface by hand would not guarantee.
    """
    from timesfm3 import TimesFM3Torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return TimesFM3Torch.decode.__wrapped__(base, **kw)


def build_lora(base_model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules="all-linear",
                     lora_dropout=dropout)
    return get_peft_model(base_model, cfg)


def train_fold(
    y_train: np.ndarray,
    *,
    context: int,
    horizon: int,
    device: str | None = None,
    steps: int = 200,
    batch_size: int = 8,
    lr: float = 1e-4,
    r: int = 8,
    transform: str | None = None,
    seed: int = 0,
    log_every: int = 50,
):
    """Fine-tune a LoRA adapter on ONE fold's training slice.

    Returns (peft_model, history). The caller is responsible for guaranteeing
    that y_train contains no data at or after the fold's test year.
    """
    from timesfm3 import TimesFM3Torch

    torch.manual_seed(seed)
    dev = pick_device(device)

    y = np.atleast_2d(np.asarray(y_train, dtype=np.float32))
    if transform == "log1p":
        y = np.log1p(np.clip(y, 0, None))

    ds = WindowDataset(y, context, horizon)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    base = TimesFM3Torch.from_pretrained("google/timesfm-3.0-pytorch")
    model = build_lora(base, r=r).to(dev)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    hist, step = [], 0
    while step < steps:
        for ctx, fut, msk in dl:
            if step >= steps:
                break
            ctx, fut, msk = ctx.to(dev), fut.to(dev), msk.to(dev)
            # PEFT swaps the nn.Linear modules in place, so this still
            # routes through the LoRA adapters.
            out = decode_with_grad(model, target=ctx, horizon=horizon)
            loss = pinball_loss(out[:, :, :horizon, :], fut, msk)
            if not torch.isfinite(loss):
                # fail loudly: a diverged adapter produces plausible-looking
                # but meaningless forecasts downstream
                raise FloatingPointError(
                    f"non-finite loss at step {step}; adapter is invalid")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            hist.append(float(loss.item()))
            if log_every and step % log_every == 0:
                print(f"    step {step:4d}  loss {loss.item():.5f}", flush=True)
            step += 1

    model.eval()
    return model, hist


@torch.no_grad()
def forecast(model, contexts: list[np.ndarray], horizon: int,
             device: str | None = None,
             transform: str | None = None) -> np.ndarray:
    """Quantile forecasts from a fine-tuned model. Returns (n, u, horizon, 9)."""
    dev = pick_device(device)
    out = []
    for c in contexts:
        c = np.atleast_2d(np.asarray(c, dtype=np.float32))
        if transform == "log1p":
            c = np.log1p(np.clip(c, 0, None))
        t = torch.from_numpy(c).unsqueeze(0).to(dev)
        q = decode_with_grad(
            model, target=t, horizon=horizon
        )[0, :, :horizon, :].float().cpu().numpy()
        if transform == "log1p":
            q = np.expm1(q)
        out.append(np.sort(np.clip(q, 0, None), axis=-1))
    return np.stack(out, axis=0)
