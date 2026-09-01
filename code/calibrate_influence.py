"""Does the label-free influence ratio predict label-based attribution?

R is cheap and needs no ground truth. Delta_attr needs labels. If R predicted
Delta_attr, R would be a screening statistic: a way to tell, on an unlabelled
cohort, whether a representation is doing anything at all. That would make it a
method. If it does not predict Delta_attr, R measures behavioural sensitivity
and nothing more, and the paper must say so.

The WCE corpora cannot answer this on their own. Every contrast there is null --
Delta_attr is within noise of zero for all seven arms on all four corpora -- so
there is no variation in the predictor's target to correlate against. The
simulation supplies that variation: Delta_attr spans roughly 0.25 down to 0.0006
as training data grows, by construction.

For each (N, capacity, control arm) cell:

    Delta_attr = mean_s AUC(prior, s) - mean_s AUC(control, s)     label-based
    R          = mean_{i!=j} D(p_prior_i, p_ctrl_j)                label-free
                 -------------------------------------
                 mean_{i!=j} D(p_arm_i,   p_arm_j)

with D the mean absolute difference in predicted probability (for a binary
task this is the total-variation distance).

USAGE
    python calibrate_influence.py --probs pc_probs.npz --auc pc_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

CONTROLS = ["cross_image", "wrong_physics", "shuffled", "gauss", "zeros", "rgb"]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _split_stat(models, group, d) -> float:
    """Between-group over within-group mean distance for one labelling of the
    2n models. With the true labelling this is R; over all labellings it is the
    null distribution R is being compared against."""
    g0 = list(group)
    g1 = [i for i in range(len(models)) if i not in set(g0)]
    within = ([d(models[i], models[j]) for i, j in itertools.combinations(g0, 2)]
              + [d(models[i], models[j]) for i, j in itertools.combinations(g1, 2)])
    cross = [d(models[i], models[j]) for i in g0 for j in g1]
    w = st.mean(within)
    return st.mean(cross) / w if w > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, default=Path("/home/S248103/biohpc/pc_probs.npz"))
    ap.add_argument("--auc", type=Path, default=Path("/home/S248103/biohpc/pc_results.json"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    P = np.load(a.probs)
    A = json.loads(a.auc.read_text())
    cells = sorted({(int(k.split("|")[0]), int(k.split("|")[1])) for k in A})
    seeds = sorted({int(k.split("|")[2]) for k in P.files})

    rows = []
    print(f"{'N':>6} {'width':>5} {'control':>14}   {'Delta_attr':>10} {'R':>7}")
    for n, w in cells:
        for c in CONTROLS:
            if f"{n}|{w}|{c}" not in A:
                continue
            delta = st.mean(A[f"{n}|{w}|prior"]) - st.mean(A[f"{n}|{w}|{c}"])
            pr = {s: P[f"{n}|{w}|{s}|prior"] for s in seeds}
            ct = {s: P[f"{n}|{w}|{s}|{c}"] for s in seeds}
            within = st.mean([dist(pr[i], pr[j]) for i, j in itertools.combinations(seeds, 2)]
                             + [dist(ct[i], ct[j]) for i, j in itertools.combinations(seeds, 2)])
            cross = st.mean([dist(pr[i], ct[j]) for i in seeds for j in seeds if i != j])
            R = cross / within if within > 0 else float("nan")
            rows.append({"N": n, "width": w, "control": c, "delta": delta, "R": R,
                         "within": within, "cross": cross})
            print(f"{n:>6} {w:>5} {c:>14}   {delta:>+10.4f} {R:>7.3f}")

    good = [r for r in rows if np.isfinite(r["R"])]
    d = [r["delta"] for r in good]
    R = [r["R"] for r in good]
    print(f"\nn = {len(good)} cells\n")

    pear = stats.pearsonr(d, R)
    spear = stats.spearmanr(d, R)
    print(f"  Pearson  r = {pear[0]:+.3f}  p = {pear[1]:.2e}")
    print(f"  Spearman rho = {spear[0]:+.3f}  p = {spear[1]:.2e}")

    # Pooling across control arms may be what destroys the rank correlation:
    # each control sits at its own baseline distance from the prior regardless
    # of whether that distance matters for accuracy. Condition on the arm.
    print("\n  within control arm (across the 8 regimes):")
    for c in CONTROLS:
        sub = [r for r in good if r["control"] == c]
        if len(sub) >= 4:
            rho = stats.spearmanr([r["delta"] for r in sub], [r["R"] for r in sub])
            print(f"    {c:>14}  n={len(sub)}  rho = {rho[0]:+.3f}  p = {rho[1]:.3f}")

    # And conditioning the other way: within one regime, does R rank the arms?
    print("\n  within (N, width) regime (across the 6 arms):")
    for n, w in cells:
        sub = [r for r in good if r["N"] == n and r["width"] == w]
        if len(sub) >= 4:
            rho = stats.spearmanr([r["delta"] for r in sub], [r["R"] for r in sub])
            print(f"    N={n:<5} width={w:<3} n={len(sub)}  rho = {rho[0]:+.3f}  p = {rho[1]:.3f}")

    # The mechanism to check: R is a ratio, and its denominator is seed spread.
    # Where training converges, seeds agree, the denominator collapses, and R
    # inflates on a numerator that is itself tiny. If so, R is confounded by
    # convergence -- which is a property of the regime, not of the prior.
    print("\n  denominator (within-arm seed distance) by regime:")
    for n, w in cells:
        sub = [r for r in good if r["N"] == n and r["width"] == w]
        if sub:
            dens = st.mean([r["within"] for r in sub])
            nums = st.mean([r["cross"] for r in sub])
            print(f"    N={n:<5} width={w:<3}  within = {dens:.5f}  cross = {nums:.5f}")
    wr = stats.spearmanr([r["within"] for r in good], [r["R"] for r in good])
    print(f"\n  Spearman(within-arm distance, R) = {wr[0]:+.3f}  p = {wr[1]:.2e}")

    # R's scale is confounded, but the permutation test that accompanied it in
    # the WCE study is not obviously so: it recalibrates against the observed
    # seed spread in each cell, which is exactly the quantity that shifts. If
    # the permutation p separates real-effect cells from null cells in EVERY
    # regime, the test -- not the ratio -- is the calibrated statistic, and the
    # WCE conclusion rests on the part that survives.
    print("\n  permutation test on arm labels (exact over all 126 splits):")
    print(f"    {'N':>6} {'width':>5} {'control':>14} {'Delta':>9} {'R':>6} {'perm p':>8}")
    for r in good:
        n, w, c = r["N"], r["width"], r["control"]
        models = ([P[f"{n}|{w}|{s}|prior"] for s in seeds]
                  + [P[f"{n}|{w}|{s}|{c}"] for s in seeds])
        k = len(seeds)
        obs = _split_stat(models, list(range(k)), dist)
        null = [_split_stat(models, list(g), dist)
                for g in itertools.combinations(range(2 * k), k)]
        p = sum(1 for v in null if v >= obs) / len(null)
        r["perm_p"] = p
        print(f"    {n:>6} {w:>5} {c:>14} {r['delta']:>+9.4f} {r['R']:>6.2f} {p:>8.4f}")

    real = [r for r in good if abs(r["delta"]) >= 0.05]
    null_ = [r for r in good if abs(r["delta"]) < 0.005]
    if real and null_:
        print(f"\n    cells with |Delta| >= 0.05 (real effect):  n={len(real):>2}  "
              f"median perm p = {st.median([r['perm_p'] for r in real]):.4f}  "
              f"frac p<0.05 = {sum(1 for r in real if r['perm_p'] < 0.05)/len(real):.2f}")
        print(f"    cells with |Delta| <  0.005 (no effect):   n={len(null_):>2}  "
              f"median perm p = {st.median([r['perm_p'] for r in null_]):.4f}  "
              f"frac p<0.05 = {sum(1 for r in null_ if r['perm_p'] < 0.05)/len(null_):.2f}")

    if a.out:
        a.out.write_text(json.dumps(
            {"cells": rows, "pearson_r": pear[0], "pearson_p": pear[1],
             "spearman_rho": spear[0], "spearman_p": spear[1]}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
