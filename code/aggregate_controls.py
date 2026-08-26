#!/usr/bin/env python3
"""Aggregate the control-prior matrix into the comparison the paper needs.

The question is narrow: does the published analytic prior beat channels that
carry no optical information? Reports, per arm, the cross-seed mean macro-AUC
and the paired delta against two references -- the RGB baseline (does the arm
help at all?) and the real prior (does the prior beat this control?).

Paired tests are on cross-seed deltas, because seed is the unit of replication
here: the same seed trains every arm, so pairing removes seed-level variance
that would otherwise swamp a 0.02 effect. Reports a sign test (exact, no
distributional assumption, appropriate at n=6) alongside the mean, and the
per-seed deltas themselves, because with this few seeds the spread matters
more than any summary of it.

    python aggregate_controls.py --runs_dir /home/S248103/biohpc/tmi_runs
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ARM_ORDER = ["rgb", "prior", "zeros", "shuffled", "random_fixed", "phi_dup", "gauss"]

ARM_BLURB = {
    "rgb": "3-channel baseline",
    "prior": "published analytic prior",
    "zeros": "two zero channels (input-width control)",
    "shuffled": "prior pixels permuted (marginals kept, structure destroyed)",
    "random_fixed": "fixed smooth random fields (scale-matched)",
    "phi_dup": "bare fluence map Phi (content-free geometry)",
    "gauss": "per-frame noise (scale-matched)",
}


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test at p=0.5, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(runs_dirs, model: str):
    """-> {arm: {seed: macro_auc}}

    Accepts one directory or several. Several is the normal case once a
    matrix has been extended: tmi_runs/ holds seeds 41-52 for all arms and
    tmi_runs_selAUC/ holds 53-84, kept apart so neither directory is ever a
    matrix with unequal per-arm seed counts. A seed appearing for the same
    arm in two directories is an error rather than something to silently
    resolve -- it would mean two different runs claim the same cell.
    """
    if isinstance(runs_dirs, (str, Path)):
        runs_dirs = [runs_dirs]
    pat = re.compile(rf"^{re.escape(model)}_(?P<arm>.+)_seed(?P<seed>\d+)$")
    out: dict[str, dict[int, float]] = {}
    origin: dict[tuple[str, int], Path] = {}
    for runs_dir in runs_dirs:
        runs_dir = Path(runs_dir)
        if not runs_dir.is_dir():
            raise SystemExit(f"no such runs_dir: {runs_dir}")
        for d in sorted(runs_dir.iterdir()):
            if not d.is_dir():
                continue
            m = pat.match(d.name)
            if not m:
                continue
            metrics = d / "test_metrics.json"
            if not (d / "test_predictions.npz").exists() or not metrics.exists():
                continue                              # incomplete run
            auc = json.loads(metrics.read_text()).get("macro_auc")
            if auc is None:
                continue
            arm, seed = m.group("arm"), int(m.group("seed"))
            if (arm, seed) in origin:
                raise SystemExit(
                    f"duplicate run for {arm} seed {seed}: "
                    f"{origin[(arm, seed)]} and {runs_dir}"
                )
            origin[(arm, seed)] = runs_dir
            out.setdefault(arm, {})[seed] = float(auc)
    return out


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan")
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def paired(a: dict[int, float], b: dict[int, float]):
    """Deltas a-b over seeds present in both."""
    seeds = sorted(set(a) & set(b))
    return seeds, [a[s] - b[s] for s in seeds]


def report(data, ref: str, label: str):
    if ref not in data:
        print(f"\n(no {ref} arm yet -- skipping {label} comparison)")
        return
    print(f"\n{label}")
    print("-" * 96)
    print(f"{'arm':14s} {'macro-AUC':>16s} {'paired d':>10s} {'sign':>7s} "
          f"{'p':>7s}  per-seed deltas")
    print("-" * 96)
    for arm in ARM_ORDER:
        if arm not in data or arm == ref:
            continue
        seeds, deltas = paired(data[arm], data[ref])
        if not deltas:
            continue
        mu, sd = mean_sd(list(data[arm].values()))
        dmu, _ = mean_sd(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        p = sign_test_p(wins, losses)
        ds = " ".join(f"{d:+.3f}" for d in deltas)
        star = "*" if p < 0.05 else " "
        print(f"{arm:14s} {mu:9.4f}+-{sd:5.4f} {dmu:+10.4f} "
              f"{wins:>3d}/{len(deltas):<3d} {p:7.3f}{star} {ds}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", nargs="+",
                    default=["/home/S248103/biohpc/tmi_runs"],
                    help="one or more run directories; seeds are pooled across them")
    ap.add_argument("--model", default="efficientnet_b0")
    a = ap.parse_args()

    data = load(a.runs_dir, a.model)
    if not data:
        print("no completed runs found")
        return 1

    print(f"=== {a.model}: cross-seed test macro-AUC ===\n")
    print(f"{'arm':14s} {'n':>3s} {'mean':>8s} {'sd':>8s}   what it is")
    print("-" * 96)
    for arm in ARM_ORDER:
        if arm not in data:
            continue
        mu, sd = mean_sd(list(data[arm].values()))
        print(f"{arm:14s} {len(data[arm]):3d} {mu:8.4f} {sd:8.4f}   {ARM_BLURB.get(arm,'')}")

    report(data, "rgb", "vs RGB baseline  (positive = the extra channels help at all)")
    report(data, "prior", "vs PUBLISHED PRIOR  (positive = this control BEATS the prior)")

    seed_counts = {arm: len(s) for arm, s in data.items()}
    if len(set(seed_counts.values())) > 1:
        print(f"\nNOTE: unequal seed counts {seed_counts} -- matrix still running; "
              f"means are not yet comparable across arms.")
    print("\nInterpretation guide: the manuscript's claim requires 'prior' to beat "
          "\nshuffled/random_fixed/phi_dup/gauss. A control that matches or beats it "
          "\nfalsifies the optical-specificity claim, though not the training-time one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
