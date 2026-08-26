#!/usr/bin/env python3
"""Paired comparison of the two checkpoint-selection rules.

The released checkpoints were early-stopped on validation
macro-F1-evaluable; the manuscript states early stopping is on validation
macro-AUC. This aggregates the prior-minus-rgb delta under each rule and
tests whether the rule itself moves the delta.

Seed is the unit of replication and every contrast is paired within seed,
including the between-rule contrast: the same seed under both rules is the
only comparison that removes seed-level variance, which is large here
relative to every effect being measured.

Run directories:
    macro_auc          tmi_runs/ (seeds 41-52) + tmi_runs_selAUC/ (53-84)
    macro_f1_evaluable tmi_runs_selF1/ (seeds 41-84)

tmi_runs/ holds the 7-arm control matrix and is left untouched; the
macro_auc seed extension lives in tmi_runs_selAUC/ so that matrix keeps
equal per-arm seed counts.

Only seeds present under BOTH rules are used, so a partial run reports a
smaller n rather than an unbalanced comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

try:
    from scipy import stats
except ImportError:                                            # pragma: no cover
    stats = None

AUC_SEL = [("/home/S248103/biohpc/tmi_runs", range(41, 53)),
           ("/home/S248103/biohpc/tmi_runs_selAUC", range(53, 85))]
F1_SEL = [("/home/S248103/biohpc/tmi_runs_selF1", range(41, 85))]


def _auc(base: str, arm: str, seed: int):
    p = Path(base) / f"efficientnet_b0_{arm}_seed{seed}" / "test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("macro_auc")


def _deltas(spec):
    """-> {seed: prior_auc - rgb_auc} for every complete pair."""
    out = {}
    for base, seeds in spec:
        for s in seeds:
            r, p = _auc(base, "rgb", s), _auc(base, "prior", s)
            if r is not None and p is not None:
                out[s] = p - r
    return out


def _summary(name, d):
    n = len(d)
    if n < 2:
        print(f"{name}: n={n}, too few to summarize")
        return
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(n)
    pos = sum(1 for x in d if x > 0)
    if stats is not None:
        t = m / se
        p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
        tc = stats.t.ppf(0.975, n - 1)
        ci = f"[{m - tc * se:+.4f}, {m + tc * se:+.4f}]"
        extra = f"  t={t:+.2f}  p={p:.4f}"
    else:
        ci, extra = "(scipy unavailable)", ""
    print(f"{name}")
    print(f"   n={n}  delta={m:+.4f}  sd={sd:.4f}  95% CI {ci}{extra}  sign {pos}/{n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="omit the per-seed table")
    args = ap.parse_args()

    a, b = _deltas(AUC_SEL), _deltas(F1_SEL)
    shared = sorted(set(a) & set(b))

    print("=== checkpoint-selection rule: effect on the prior-vs-rgb delta ===\n")
    print(f"seeds under macro_auc:          {len(a)}")
    print(f"seeds under macro_f1_evaluable: {len(b)}")
    print(f"seeds under both (paired n):    {len(shared)}\n")

    if not args.quiet and shared:
        print(f"{'seed':>5} {'macro_auc':>12} {'macro_f1':>12} {'diff':>10}")
        for s in shared:
            print(f"{s:>5} {a[s]:+12.4f} {b[s]:+12.4f} {b[s] - a[s]:+10.4f}")
        print()

    _summary("selection = macro_auc  (manuscript's stated protocol)",
             [a[s] for s in shared])
    _summary("selection = macro_f1_evaluable  (rule used for the artifacts)",
             [b[s] for s in shared])
    print()
    _summary("EFFECT OF THE SELECTION RULE (paired within seed)",
             [b[s] - a[s] for s in shared])

    print("\nA delta that is near zero under the stated protocol but positive")
    print("under the rule actually used means the reported effect depends on")
    print("selecting checkpoints by a quantity other than the one reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
