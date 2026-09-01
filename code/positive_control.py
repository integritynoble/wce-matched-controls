"""Positive control: can the framework detect a physics prior that genuinely helps?

This is the experiment the MedIA paper is blocked on. Its purpose is not to show
that priors work. It is to establish that the control hierarchy and the
equivalence machinery are capable of returning a POSITIVE result when one is
warranted -- because a framework that only ever produces nulls is not a
measuring instrument, and a reviewer is entitled to ask.

PRE-COMMITTED STOPPING CONDITION. If no (N, capacity) regime shows the correct
prior beating its matched controls, the framework's sensitivity is unproven,
every null it produces elsewhere is uninterpretable, and the MedIA paper is held
rather than submitted. Recorded in TWO_PAPER_DESIGN_2026-08-26.md before running.

--------------------------------------------------------------------------
THE FORWARD MODEL

Images are formed by Beer-Lambert attenuation of a spatially varying
illumination field:

    I_c(x) = I0_c * Phi(x) * exp(-eps_c * C(x))

  C(x)    absorber concentration map (blobs on a background)
  Phi(x)  illumination, a smooth radial falloff, unknown to the learner
  eps_c   per-channel extinction coefficients (the "physics")

The label depends only on C: a sample is positive if its peak absorber
concentration exceeds a threshold over a minimum area. So the task is decidable
from C alone, and the illumination Phi is pure nuisance.

WHY THE CORRECT PRIOR SHOULD HELP. Taking a log-ratio of two channels cancels
both I0 and Phi exactly:

    log(I_a / I_b) = log(I0_a/I0_b) - (eps_a - eps_b) * C(x)

so C is recoverable up to a known affine map, with the illumination gone. A
network given only RGB must learn to perform that cancellation from data. A
network handed the recovered map does not. That is an ACCESSIBILITY benefit, not
an informational one -- consistent with I(Y; X, f(X)) = I(Y; X) -- and it is
exactly the kind of effect the framework claims to be able to detect. It should
be large when data are scarce and vanish as data grow.

ARMS
  prior         the correct inversion: log-ratio with the true channel pair
  wrong_physics same family, but a log-product rather than a log-ratio: it
                retains the signal while leaving the illumination attached, so it
                is plausible and structured but does not isolate C
  cross_image   the correct inversion computed from a DIFFERENT sample (C5)
  shuffled      the correct inversion, pixels permuted (C4)
  gauss         noise matched to the prior's moments (C2)
  zeros         null path (C1)
  rgb           no extra channel (C0)

USAGE
    python positive_control.py --smoke          # tiny, checks it runs
    python positive_control.py --out results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 48
EPS = np.array([0.9, 0.45, 0.15], dtype=np.float32)   # per-channel extinction
I0 = np.array([1.0, 0.95, 0.9], dtype=np.float32)


def illumination(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth radial falloff with a randomly jittered centre: nuisance."""
    ys, xs = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    out = np.empty((n, IMG, IMG), dtype=np.float32)
    for i in range(n):
        cy, cx = rng.uniform(0.35, 0.65, 2) * IMG
        lam = rng.uniform(0.35, 0.55) * IMG
        r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        out[i] = np.exp(-r / lam)
    return out


def make_dataset(n: int, rng: np.random.Generator):
    """Returns rgb [n,3,H,W], concentration [n,H,W], labels [n]."""
    ys, xs = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    C = np.zeros((n, IMG, IMG), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # Background texture: low-level absorber everywhere.
        C[i] = rng.uniform(0.05, 0.15) * rng.random((IMG, IMG)).astype(np.float32)
        C[i] = np.asarray(
            torch.nn.functional.avg_pool2d(
                torch.from_numpy(C[i])[None, None], 5, 1, 2)[0, 0])
        positive = bool(rng.random() < 0.5)
        y[i] = int(positive)
        # Blobs. Positives get a denser one; negatives get a fainter/smaller one,
        # so the classes differ in absorber concentration, not in blob presence.
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.uniform(0.2, 0.8, 2) * IMG
            sd = rng.uniform(3.5, 6.0)
            amp = rng.uniform(0.9, 1.4) if positive else rng.uniform(0.25, 0.5)
            C[i] += amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sd ** 2))
    phi = illumination(rng, n)
    rgb = (I0[None, :, None, None]
           * phi[:, None] * np.exp(-EPS[None, :, None, None] * C[:, None]))
    rgb = rgb.astype(np.float32)
    return rgb, C.astype(np.float32), y


def inversion(rgb: np.ndarray, a: int, b: int) -> np.ndarray:
    """log(I_a/I_b): cancels illumination exactly when (a,b) are the true pair."""
    z = np.log(rgb[:, a] + 1e-6) - np.log(rgb[:, b] + 1e-6)
    m = z.mean(axis=(1, 2), keepdims=True)
    s = z.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((z - m) / s).astype(np.float32)


def build_channel(arm: str, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    if arm == "rgb":
        return None
    n = rgb.shape[0]
    if arm == "zeros":
        return np.zeros((n, IMG, IMG), dtype=np.float32)
    if arm == "prior":
        return inversion(rgb, 0, 2)                      # true extinction pair
    if arm == "wrong_physics":
        # A transform of the same functional family that does NOT isolate the
        # signal. Any log-RATIO cancels the illumination exactly, because Phi
        # multiplies every channel equally -- so "the wrong channel pair" is not
        # a wrong-physics control at all, it recovers C up to a scale factor.
        # (Caught by the first smoke test: prior - wrong_physics was +0.0002.)
        # A log-PRODUCT keeps C but leaves 2*log(Phi) attached to it, so the map
        # is plausible, structured, and confounded with the nuisance field.
        z = np.log(rgb[:, 0] + 1e-6) + np.log(rgb[:, 2] + 1e-6)
        m = z.mean(axis=(1, 2), keepdims=True)
        sd = z.std(axis=(1, 2), keepdims=True) + 1e-6
        return ((z - m) / sd).astype(np.float32)
    if arm == "shuffled":
        z = inversion(rgb, 0, 2).reshape(n, -1)
        idx = np.argsort(rng.random(z.shape), axis=1)
        return np.take_along_axis(z, idx, axis=1).reshape(n, IMG, IMG)
    if arm == "cross_image":
        z = inversion(rgb, 0, 2)
        perm = (np.arange(n) + 1 + rng.integers(0, n - 1)) % n   # never itself
        return z[perm]
    if arm == "gauss":
        z = inversion(rgb, 0, 2)
        return rng.normal(z.mean(), z.std(), z.shape).astype(np.float32)
    raise ValueError(arm)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(width * 4, 2)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


_TEST_CACHE: dict = {}


def _test_set():
    """One test set, generated once and shared by every arm and seed.

    Regenerating it per arm was both slow and wrong in spirit: arms must be
    compared on identical images, and a freshly drawn test set per arm adds
    sampling noise to every contrast.
    """
    if "d" not in _TEST_CACHE:
        _TEST_CACHE["d"] = make_dataset(1500, np.random.default_rng(999))
    return _TEST_CACHE["d"]


def train_eval(xtr, ytr, xte, yte, width, seed, dev, epochs=40, return_probs=False):
    torch.manual_seed(seed)
    model = SmallCNN(xtr.shape[1], width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    bs = 64
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(xtr[j]), ytr[j]).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(xte.to(dev)), 1)[:, 1].cpu().numpy()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yte.numpy(), p))
    return (auc, p) if return_probs else auc


def run_seed(arms, n_train, width, seed, dev, probs_sink=None):
    """All arms on ONE draw of data, so contrasts are paired within seed."""
    rng = np.random.default_rng(10_000 + seed)
    rgb_tr, _, y_tr = make_dataset(n_train, rng)
    rgb_te, _, y_te = _test_set()
    ytr, yte = torch.from_numpy(y_tr), torch.from_numpy(y_te)
    out, probs_out = {}, {}
    for arm in arms:
        r1 = np.random.default_rng(20_000 + seed)
        r2 = np.random.default_rng(30_000 + seed)
        ctr = build_channel(arm, rgb_tr, r1)
        cte = build_channel(arm, rgb_te, r2)
        xtr = torch.from_numpy(rgb_tr if ctr is None
                               else np.concatenate([rgb_tr, ctr[:, None]], 1))
        xte = torch.from_numpy(rgb_te if cte is None
                               else np.concatenate([rgb_te, cte[:, None]], 1))
        # Predictions are kept so the influence ratio R can be computed on the
        # same runs the AUC comes from. The WCE corpora supply only null effects,
        # so they cannot on their own show whether R tracks label-based
        # attribution; the simulation spans Delta from ~0.25 down to ~0.0006 and
        # is what makes the calibration answerable.
        auc, probs = train_eval(xtr, ytr, xte, yte, width, seed, dev,
                                return_probs=True)
        out[arm] = auc
        probs_out[arm] = probs
    if probs_sink is not None:
        probs_sink[(n_train, width, seed)] = probs_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[150, 400, 1200, 4000])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--probs_out", type=Path, default=None,
                    help="write per-sample test predictions, so the influence "
                         "ratio can be computed on the same runs as the AUCs")
    a = ap.parse_args()
    arms = ["rgb", "zeros", "gauss", "shuffled", "cross_image", "wrong_physics", "prior"]
    if a.smoke:
        a.sizes, a.widths, a.seeds = [150], [8], 2

    res, probs_sink = {}, ({} if a.probs_out else None)
    for n, w in itertools.product(a.sizes, a.widths):
        print(f"\n=== N={n}  width={w} ===")
        per_seed = [run_seed(arms, n, w, s, a.device, probs_sink) for s in range(a.seeds)]
        for arm in arms:
            v = [d[arm] for d in per_seed]
            res[f"{n}|{w}|{arm}"] = v
            print(f"  {arm:14s} AUC {st.mean(v):.4f}"
                  f"{' +/- ' + format(st.stdev(v), '.4f') if len(v) > 1 else ''}")
        # The question the framework asks, not just the utility question.
        for ctrl in ("cross_image", "wrong_physics", "zeros"):
            d = [p - c for p, c in zip(res[f"{n}|{w}|prior"], res[f"{n}|{w}|{ctrl}"])]
            print(f"    prior - {ctrl:14s} {st.mean(d):+.4f}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {a.out}")
    if a.probs_out:
        np.savez_compressed(a.probs_out, **{
            f"{n}|{w}|{s}|{arm}": pr
            for (n, w, s), d in probs_sink.items() for arm, pr in d.items()})
        print(f"wrote {a.probs_out}  ({len(probs_sink)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
