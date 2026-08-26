#!/usr/bin/env python3
"""Equivalence (TOST) on paired cross-seed macro-AUC deltas.

The manuscript's central claim is an absence: the analytic prior does not
outperform its baseline by the published +0.023. An absence needs an
equivalence test, not a failed difference test -- "p > 0.05" would only mean
underpowered. TOST runs two one-sided t-tests against +-margin and reports the
larger p; small p means the effect is bounded inside the margin.

Pairing is on seed: the same seed trains both arms, so pairing removes
seed-level variance that would otherwise swamp a 0.02 effect.

This existed only as ad-hoc code until 2026-08-11, which is how the n=12
ConvNeXt p-value survived next to an n=44 delta in the handoff. It is a script
now so the numbers in the paper can be regenerated in one command.

Validation: EfficientNet-B0 prior-rgb at n=44 must give p = 0.0096. The
--validate flag asserts it; run it before trusting any new number.

    python compute_tost.py --runs_dir ~/biohpc/tmi_runs ~/biohpc/tmi_runs_selAUC
    python compute_tost.py --runs_dir ~/biohpc/tmi_runs ~/biohpc/tmi_runs_convnext_ext \
        --model convnext_tiny
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from statistics import mean, stdev


def load(runs_dirs, model):
    pat = re.compile(rf"^{re.escape(model)}_(?P<arm>.+)_seed(?P<seed>\d+)$")
    out: dict[str, dict[int, float]] = {}
    for rd in runs_dirs:
        for d in sorted(glob.glob(os.path.join(os.path.expanduser(rd), "*"))):
            m = pat.match(os.path.basename(d))
            if not m or not os.path.exists(os.path.join(d, "test_predictions.npz")):
                continue
            mj = os.path.join(d, "test_metrics.json")
            if not os.path.exists(mj):
                continue
            auc = json.load(open(mj)).get("macro_auc")
            if auc is None:
                continue
            out.setdefault(m.group("arm"), {})[int(m.group("seed"))] = float(auc)
    return out


# Student t CDF via the regularized incomplete beta, so this has no dependency
# beyond the stdlib and can run in any env the trainer runs in.
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (FPMIN if abs(d) < FPMIN else d)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (FPMIN if abs(d) < FPMIN else d)
        c = FPMIN if abs(c) < FPMIN else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (FPMIN if abs(d) < FPMIN else d)
        c = FPMIN if abs(c) < FPMIN else c
        de = d * c
        h *= de
        if abs(de - 1) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) \
        else 1 - bt * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    p = 0.5 * _betai(df / 2, 0.5, df / (df + t * t))
    return p if t <= 0 else 1 - p


def tost(deltas, margin):
    """-> (n, mean, sd, p) where p is the larger of the two one-sided tests."""
    n = len(deltas)
    d, s = mean(deltas), stdev(deltas)
    se, df = s / math.sqrt(n), n - 1
    p_lo = 1 - t_cdf((d + margin) / se, df)     # H0: d <= -margin
    p_hi = t_cdf((d - margin) / se, df)         # H0: d >=  margin
    return n, d, s, max(p_lo, p_hi)


def paired(data, a, b):
    seeds = sorted(set(data[a]) & set(data[b]))
    return [data[a][s] - data[b][s] for s in seeds]


def tightest_margin(deltas, alpha=0.05, hi=0.06, step=0.0005):
    """Smallest margin at which equivalence still holds -- the honest summary,
    since the pre-specified +-0.023 is the claim being tested, not the limit."""
    m = step
    while m < hi:
        if tost(deltas, m)[3] < alpha:
            return m
        m += step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", nargs="+", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--arm", default="prior")
    ap.add_argument("--ref", default="rgb")
    ap.add_argument("--margin", type=float, default=0.023)
    ap.add_argument("--validate", action="store_true",
                    help="assert the known EffB0 n=44 result reproduces")
    a = ap.parse_args()

    data = load(a.runs_dir, a.model)
    if a.arm not in data or a.ref not in data:
        raise SystemExit(f"missing arm: have {sorted(data)}")
    deltas = paired(data, a.arm, a.ref)
    n, d, s, p = tost(deltas, a.margin)
    tm = tightest_margin(deltas)

    print(f"{a.model}: {a.arm} - {a.ref}")
    print(f"  n = {n}   mean delta = {d:+.4f}   sd = {s:.4f}")
    print(f"  TOST at +-{a.margin}: p = {p:.4f}"
          f"   {'EQUIVALENT' if p < 0.05 else 'not established'}")
    print(f"  tightest margin with p < 0.05: "
          + (f"+-{tm:.4f}" if tm else "none below 0.06"))

    if a.validate:
        assert a.model == "efficientnet_b0", "validate applies to the EffB0 case"
        assert n == 44, f"expected n=44, got {n}"
        assert abs(p - 0.0096) < 0.0005, f"expected p=0.0096, got {p:.4f}"
        print("  VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
