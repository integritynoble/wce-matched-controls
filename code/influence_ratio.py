"""Influence ratio: does swapping the analytic channels move the model at all?

Every performance comparison needs labels. This one does not.

If the analytic channels carry information the network uses, then replacing them
with a control must change what the network predicts. If predictions are
unchanged, the channels are inert -- and that is true whatever the labels say,
on any corpus, including one with no ground truth.

The difficulty is the baseline. Two runs of the *same* arm at different seeds
already disagree, because training is stochastic. So a cross-arm distance is
uninterpretable on its own; it has to be read against the distance ordinary
retraining produces. Hence a ratio:

    R = mean D(p_A_i, p_B_j)  /  mean D(p_A_i, p_A_j)        i != j

    R ~ 1   the representation moves predictions no more than reseeding does
    R >> 1  the representation materially changes model behaviour

R says nothing about whether performance improves. It is an influence
diagnostic, not an attribution test -- the name matters, because without labels
the question of benefit cannot be asked.

WHAT THIS REPORTS
  * R with a bootstrap interval over seeds
  * a permutation p-value: under the null that the arm label is uninformative,
    the 2n models are exchangeable, so relabelling them at random should give
    the same R. This is the honest null for the statistic.
  * per-class R, since a representation may move one class and not others
  * saturation diagnostics. If the softmax is near one-hot, total-variation
    distance is dominated by argmax flips and is insensitive to real change;
    a log-probability variant is reported alongside for that reason.
  * argmax agreement, which is the most interpretable number of the set.

USAGE
    python influence_ratio.py --corpus in_domain --arm_a prior --arm_b zeros
    python influence_ratio.py --corpus galar_all --arm_a prior --arm_b cross_image
    python influence_ratio.py --corpus in_domain --arm_a prior --arm_b zeros --per_class
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
from pathlib import Path

import numpy as np

BIOHPC = Path("/home/S248103/biohpc")
ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]


def _run_roots(arm: str) -> list[tuple[int, Path]]:
    """(seed, run_dir) for the canonical macro_auc selection-rule set."""
    out = []
    for seed in range(41, 85):
        if arm == "cross_image":
            d = BIOHPC / "tmi_runs_crossimage" / f"efficientnet_b0_cross_image_seed{seed}"
        else:
            root = "tmi_runs" if seed <= 52 else "tmi_runs_selAUC"
            d = BIOHPC / root / f"efficientnet_b0_{arm}_seed{seed}"
        if d.is_dir():
            out.append((seed, d))
    return out


def load_arm(arm: str, corpus: str) -> dict[int, np.ndarray]:
    """seed -> [n_frames, n_classes] softmax outputs on the chosen corpus."""
    fname = "test_predictions.npz" if corpus == "in_domain" \
        else f"galar_{corpus.removeprefix('galar_')}_test_predictions.npz"
    out = {}
    for seed, d in _run_roots(arm):
        f = d / fname
        if f.exists():
            out[seed] = np.load(f, allow_pickle=True)["probs"].astype(np.float32)
    return out


def tv(a: np.ndarray, b: np.ndarray) -> float:
    """Mean total-variation distance between two sets of predicted distributions."""
    return float(np.abs(a - b).sum(axis=1).mean() / 2.0)


def logp_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    """Mean L1 distance in log-probability space.

    Total variation saturates when the softmax is near one-hot: two confidently
    wrong models can differ enormously in the tail and barely at all in TV. The
    log-space distance stays sensitive there, so a disagreement between the two
    is itself informative.
    """
    return float(np.abs(np.log(a + eps) - np.log(b + eps)).sum(axis=1).mean())


def argmax_agree(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.argmax(1) == b.argmax(1)).mean())


def pair_stats(A: dict, B: dict, metric, rng: np.random.Generator,
               max_pairs: int = 200) -> tuple[float, float, float]:
    """(within_A, within_B, cross) mean distances, subsampling pairs if needed."""
    sa, sb = sorted(A), sorted(B)

    def sub(pairs):
        pairs = list(pairs)
        if len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]
        return pairs

    wa = [metric(A[i], A[j]) for i, j in sub(itertools.combinations(sa, 2))]
    wb = [metric(B[i], B[j]) for i, j in sub(itertools.combinations(sb, 2))]
    cr = [metric(A[i], B[j]) for i, j in sub([(i, j) for i in sa for j in sb if i != j])]
    return float(np.mean(wa)), float(np.mean(wb)), float(np.mean(cr))


def permutation_p(A: dict, B: dict, metric, observed_R: float,
                  n_perm: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Null: the arm label is uninformative, so the 2n models are exchangeable."""
    pooled = [A[s] for s in sorted(A)] + [B[s] for s in sorted(B)]
    n = len(A)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(pooled))
        g1 = {i: pooled[perm[i]] for i in range(n)}
        g2 = {i: pooled[perm[n + i]] for i in range(len(pooled) - n)}
        w1, w2, cr = pair_stats(g1, g2, metric, rng, max_pairs=60)
        null[k] = cr / ((w1 + w2) / 2)
    # two-sided
    p = float((np.abs(null - 1.0) >= abs(observed_R - 1.0)).mean())
    return p, null


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="in_domain",
                    help="in_domain | galar_all | galar_pillcam | galar_olympus")
    ap.add_argument("--arm_a", default="prior")
    ap.add_argument("--arm_b", default="zeros")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--per_class", action="store_true")
    ap.add_argument("--max_frames", type=int, default=25000,
                    help="Subsample frames before computing distances. The "
                         "statistic is a MEAN over frames, so 25k estimates it to "
                         "well past three decimals; Galar's 190k frames make every "
                         "distance 30x heavier than in-domain for no gain in "
                         "precision. The same frame indices are used for every "
                         "model, which is required for the distances to be "
                         "comparable. 0 disables subsampling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A, B = load_arm(a.arm_a, a.corpus), load_arm(a.arm_b, a.corpus)
    shared = sorted(set(A) & set(B))
    if len(shared) < 4:
        raise SystemExit(f"need >=4 shared seeds, found {len(shared)} "
                         f"({a.arm_a}: {len(A)}, {a.arm_b}: {len(B)})")
    A = {s: A[s] for s in shared}
    B = {s: B[s] for s in shared}
    n_total = next(iter(A.values())).shape[0]
    if a.max_frames and n_total > a.max_frames:
        # One index set, shared by every model. Subsampling per model would
        # compare different frames and the distances would be meaningless.
        sel = np.random.default_rng(12345).choice(n_total, a.max_frames,
                                                  replace=False)
        sel.sort()
        A = {s: v[sel] for s, v in A.items()}
        B = {s: v[sel] for s, v in B.items()}
        print(f"subsampled {a.max_frames} of {n_total} frames "
              f"(fixed indices, identical for every model)")
    n_frames = next(iter(A.values())).shape[0]
    print(f"corpus={a.corpus}  {a.arm_a} vs {a.arm_b}  "
          f"seeds={len(shared)}  frames={n_frames}\n")

    # --- saturation diagnostics ------------------------------------------
    p0 = next(iter(A.values()))
    maxp = p0.max(axis=1)
    ent = -(p0 * np.log(p0 + 1e-12)).sum(axis=1)
    print("saturation (arm A, first seed):")
    print(f"  mean max prob = {maxp.mean():.4f}   frac > 0.99 = {(maxp > 0.99).mean():.4f}")
    print(f"  mean entropy  = {ent.mean():.4f}  (uniform over 14 = {np.log(14):.4f})")
    if (maxp > 0.99).mean() > 0.5:
        print("  WARNING: predictions are largely saturated; read the log-space "
              "variant, not TV")

    results = {"corpus": a.corpus, "arm_a": a.arm_a, "arm_b": a.arm_b,
               "n_seeds": len(shared), "n_frames": int(n_frames),
               "saturation": {"mean_max_prob": float(maxp.mean()),
                              "frac_gt_0.99": float((maxp > 0.99).mean()),
                              "mean_entropy": float(ent.mean())}}

    # --- the ratio, under two metrics ------------------------------------
    for name, metric in (("total_variation", tv), ("log_prob_L1", logp_l1)):
        wa, wb, cr = pair_stats(A, B, metric, rng)
        within = (wa + wb) / 2
        R = cr / within
        boot = np.empty(a.n_boot)
        for k in range(a.n_boot):
            pick = rng.choice(shared, len(shared), replace=True)
            Ab = {i: A[s] for i, s in enumerate(pick)}
            Bb = {i: B[s] for i, s in enumerate(pick)}
            w1, w2, c = pair_stats(Ab, Bb, metric, rng, max_pairs=60)
            boot[k] = c / ((w1 + w2) / 2)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p, _null = permutation_p(A, B, metric, R, a.n_perm, rng)
        print(f"\n{name}:")
        print(f"  within {a.arm_a:12s} = {wa:.5f}")
        print(f"  within {a.arm_b:12s} = {wb:.5f}")
        print(f"  cross-arm            = {cr:.5f}")
        print(f"  R = {R:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  permutation p = {p:.3f}")
        results[name] = {"within_a": wa, "within_b": wb, "cross": cr,
                         "R": R, "ci": [float(lo), float(hi)], "perm_p": p}

    print(f"\nargmax agreement:")
    wa_ag, wb_ag, cr_ag = pair_stats(A, B, argmax_agree, rng)
    print(f"  within {a.arm_a:12s} = {wa_ag:.4f}")
    print(f"  within {a.arm_b:12s} = {wb_ag:.4f}")
    print(f"  cross-arm            = {cr_ag:.4f}")
    results["argmax_agreement"] = {"within_a": wa_ag, "within_b": wb_ag, "cross": cr_ag}

    # --- per class --------------------------------------------------------
    if a.per_class:
        print("\nper-class R (mean |Δp| cross / within):")
        results["per_class"] = {}
        for c, cname in enumerate(ALL_CLASSES):
            m = lambda x, y, _c=c: float(np.abs(x[:, _c] - y[:, _c]).mean())
            wa_c, wb_c, cr_c = pair_stats(A, B, m, rng, max_pairs=80)
            within_c = (wa_c + wb_c) / 2
            Rc = cr_c / within_c if within_c > 0 else float("nan")
            print(f"  {cname:24s} R={Rc:6.3f}   within={within_c:.5f}")
            results["per_class"][cname] = {"R": Rc, "within": within_c, "cross": cr_c}

    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
