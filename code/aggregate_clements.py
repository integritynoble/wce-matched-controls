"""Study-level analysis of the Clements cohort.

Implements paper/Capsule-Endoscopy/CLEMENTS_PREREGISTRATION_2026-08-31.md and
nothing else. Every choice below -- the aggregation rule, the six findings, the
metric, the contrasts, the margin, the validity gate -- was fixed in a commit
that predates the existence of any Clements prediction file.

Deliberately kept separate from eval_clements.py so the aggregation cannot be
adjusted while watching the GPU pass, and so a questioned analysis choice never
requires re-running inference.

USAGE
    python aggregate_clements.py --sheet .../clements_study_review_sheet.csv \
        --scan_root /home/S248103/biohpc/tmi_runs \
        --scan_root /home/S248103/biohpc/tmi_runs_selAUC
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Pre-registered mapping, sheet column -> Kvasir class. Fixed in advance.
FINDINGS = [
    ("normal_no_significant_finding", "Normal clean mucosa"),
    ("angioectasia_AVM", "Angiectasia"),
    ("active_bleeding", "Blood - fresh"),
    ("blood_or_hematin_no_active_bleeding", "Blood - hematin"),
    ("ulcer", "Ulcer"),
    ("erosion", "Erosion"),
]
ARMS = ["prior", "rgb", "gauss", "random_fixed", "phi_dup", "shuffled", "zeros"]
MARGIN = 0.023          # pre-registered, the paper's standing equivalence margin
FLOOR = 0.60            # pre-registered validity gate
RUN = re.compile(r"^efficientnet_b0_(?P<arm>.+)_seed(?P<seed>\d+)$")


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Rank AUC with average ranks for ties."""
    if pos.sum() == 0 or pos.sum() == len(pos):
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    n1 = int(pos.sum())
    n0 = len(score) - n1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tost(d: np.ndarray, margin: float) -> tuple:
    """Two one-sided tests. Returns (mean, lo, hi, p_tost, equivalent)."""
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return m, m, m, 0.0, abs(m) < margin
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    crit = stats.t.ppf(0.95, n - 1)          # 90% CI is the TOST-consistent one
    return m, m - crit * se, m + crit * se, float(p), bool(p < 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--scan_root", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # ---- labels -------------------------------------------------------
    rows = list(csv.DictReader(open(a.sheet)))
    cols = [c for c, _ in FINDINGS]
    studies, labels = [], {}
    for r in rows:
        if not any((r[c] or "").strip() for c in cols):
            continue                                  # S04, pre-registered drop
        sid = r["study_file"].replace(".gvf", "")
        studies.append(sid)
        labels[sid] = {c: int((r[c] or "0").strip() or 0) for c in cols}
    studies = sorted(studies)
    print(f"[clements] {len(studies)} reviewed studies "
          f"({len(rows) - len(studies)} excluded as blank)")
    for c, k in FINDINGS:
        print(f"    {c:38} -> {k:22} positives = "
              f"{sum(labels[s][c] for s in studies)}")

    # ---- per-run study-level macro AUC --------------------------------
    scores = defaultdict(dict)                        # arm -> {seed: macro auc}
    per_class = defaultdict(lambda: defaultdict(list))
    n_runs = 0
    for root in a.scan_root:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = RUN.match(d.name)
            if not m or m["arm"] not in ARMS:
                continue
            f = d / "clements_predictions.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            probs = z["probs"]
            sids = np.array([str(s) for s in z["studies"]])
            names = [str(x) for x in z["class_names"]]

            # Pre-registered aggregation: MAX frame probability per study.
            idx = {s: np.where(sids == s)[0] for s in studies}
            aucs = []
            for col, cls in FINDINGS:
                ci = names.index(cls)
                sc = np.array([probs[idx[s], ci].max() if len(idx[s]) else np.nan
                               for s in studies])
                y = np.array([labels[s][col] for s in studies], dtype=bool)
                v = auc(sc, y)
                if np.isfinite(v):
                    aucs.append(v)
                    per_class[m["arm"]][cls].append(v)
            if aucs:
                scores[m["arm"]][int(m["seed"])] = float(np.mean(aucs))
                n_runs += 1
    print(f"[clements] {n_runs} runs aggregated")
    if not scores:
        print("no predictions found -- run eval_clements.py first")
        return 1

    print(f"\n{'arm':>14} {'n':>3} {'study-level macro AUC':>22}")
    for arm in ARMS:
        v = list(scores[arm].values())
        if v:
            print(f"{arm:>14} {len(v):>3}   {np.mean(v):.4f} +- {np.std(v):.4f}")

    # ---- validity gate, read BEFORE any contrast ----------------------
    best = max((np.mean(list(v.values())) for v in scores.values() if v),
               default=float("nan"))
    print(f"\n=== validity gate (pre-registered floor {FLOOR}) ===")
    print(f"  best arm study-level macro AUC = {best:.4f}")
    if not np.isfinite(best) or best < FLOOR:
        print("  ** GATE FAILED: cohort reported as uninformative for "
              "attribution; no contrast is interpreted, per pre-registration.")
        if a.out:
            a.out.write_text(json.dumps({"gate": "failed", "best": best}, indent=2))
        return 0
    print("  gate passed -- contrasts below are interpretable")

    # ---- contrasts ----------------------------------------------------
    out = {"gate": "passed", "best_auc": best, "n_studies": len(studies),
           "contrasts": []}
    print(f"\n=== contrasts vs prior, paired by seed, TOST at +-{MARGIN} ===")
    print(f"{'contrast':>24} {'n':>3} {'delta':>9} {'90% CI':>20} {'p_TOST':>8}  verdict")
    order = ["rgb"] + [x for x in ARMS if x not in ("prior", "rgb")]
    for other in order:
        seeds = sorted(set(scores["prior"]) & set(scores[other]))
        if len(seeds) < 3:
            continue
        d = np.array([scores["prior"][s] - scores[other][s] for s in seeds])
        m, lo, hi, p, eq = tost(d, MARGIN)
        tag = "PRIMARY" if other == "rgb" else ""
        print(f"{'prior - ' + other:>24} {len(seeds):>3} {m:>+9.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}] {p:>8.4f}  "
              f"{'equivalent' if eq else 'not established'} {tag}")
        out["contrasts"].append({"contrast": f"prior - {other}", "n": len(seeds),
                                 "delta": m, "ci90": [lo, hi], "p_tost": p,
                                 "equivalent": eq})

    print("\n=== per-finding AUC (prior arm) -- context only, not a contrast ===")
    for _, cls in FINDINGS:
        v = per_class["prior"].get(cls, [])
        if v:
            print(f"  {cls:22} {np.mean(v):.4f} +- {np.std(v):.4f}")

    if a.out:
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
