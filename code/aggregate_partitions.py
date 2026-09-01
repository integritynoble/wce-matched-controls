"""Variance decomposition across patient partitions and training seeds.

The 616-run matrix uses 44 seeds on ONE patient partition. It therefore
estimates how much a contrast moves when training is repeated, and says nothing
about how much it moves when different patients land in the test set. Those are
different estimands and a reviewer is entitled to ask why the effect is inferred
from seeds rather than patients.

This fits the one-way random-effects model

    Delta_{p,s} = mu + b_p + eps_{p,s},     b_p ~ (0, sigma^2_partition)
                                            eps  ~ (0, sigma^2_seed)

to the paired contrast Delta over 10 patient-disjoint partitions x 5 seeds, and
reports the two variance components separately. The deliverable sentence is
"training randomness contributes X, the choice of patient partition contributes
Y" -- which no number of additional seeds on a single partition can produce.

Components are estimated by the unbalanced one-way ANOVA estimator, since runs
may be missing:

    sigma^2_seed      = MS_within
    sigma^2_partition = (MS_between - MS_within) / n0,
    n0 = (N - sum_p n_p^2 / N) / (k - 1)

A negative variance estimate is reported as such and truncated to zero for the
derived quantities, rather than silently clamped: it means the between-partition
signal is not resolved above seed noise at this sample size, which is itself the
answer to the question.

USAGE
    python aggregate_partitions.py --arm_a prior --arm_b rgb
    python aggregate_partitions.py --arm_a prior --arm_b zeros --out vc.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path("/home/S248103/biohpc/tmi_split_runs")


def load(arm: str) -> dict[tuple[str, int], float]:
    out = {}
    for f in glob.glob(str(ROOT / "split*" / f"effb0_{arm}_seed*" / "test_metrics.json")):
        m = re.search(r"(split\d+)/effb0_.+_seed(\d+)/", f)
        if m:
            out[(m.group(1), int(m.group(2)))] = json.loads(Path(f).read_text())["macro_auc"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_a", default="prior")
    ap.add_argument("--arm_b", default="rgb")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    A, B = load(a.arm_a), load(a.arm_b)
    pairs = sorted(set(A) & set(B))
    if not pairs:
        raise SystemExit(f"no paired runs for {a.arm_a} vs {a.arm_b}")

    by_p: dict[str, list[float]] = {}
    for p, s in pairs:
        by_p.setdefault(p, []).append(A[(p, s)] - B[(p, s)])

    k = len(by_p)
    N = sum(len(v) for v in by_p.values())
    print(f"{a.arm_a} - {a.arm_b}: {N} paired runs over {k} partitions")
    for p in sorted(by_p):
        v = by_p[p]
        sd = f" +/- {st.stdev(v):.4f}" if len(v) > 1 else ""
        print(f"  {p}  n={len(v)}  mean {st.mean(v):+.4f}{sd}")
    if k < 2 or N <= k:
        raise SystemExit("need >=2 partitions and >1 run per partition overall")

    grand = sum(sum(v) for v in by_p.values()) / N
    ss_between = sum(len(v) * (st.mean(v) - grand) ** 2 for v in by_p.values())
    ss_within = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in by_p.values())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    n0 = (N - sum(len(v) ** 2 for v in by_p.values()) / N) / (k - 1)

    var_seed = ms_within
    var_part_raw = (ms_between - ms_within) / n0
    var_part = max(var_part_raw, 0.0)

    print(f"\ngrand mean Delta = {grand:+.4f}")
    print(f"  sigma_seed      = {math.sqrt(var_seed):.4f}   (training stochasticity)")
    if var_part_raw < 0:
        print(f"  sigma_partition = 0 (raw estimate {var_part_raw:+.6f} < 0)")
        print("    between-partition variation is not resolved above seed noise here")
    else:
        print(f"  sigma_partition = {math.sqrt(var_part):.4f}   (which patients are in test)")
    tot = var_seed + var_part
    if tot > 0:
        print(f"  partition share of total variance = {var_part / tot:.1%}")

    # An effect must clear the variation a reader cares about. If partition
    # variance is real, a single-split confidence interval understates it.
    # SE of the GRAND mean under the (false) assumption that every run is an
    # independent draw -- i.e. what a single-partition analysis implicitly
    # claims. Using var_seed / n0 here would give the SE of a partition mean,
    # which is a different quantity and made the comparison come out backwards.
    se_single = math.sqrt(var_seed / N)
    se_multi = math.sqrt(var_part / k + var_seed / N)
    print(f"\n  SE of the mean if partition variance is ignored: {se_single:.4f}")
    print(f"  SE accounting for partition variance:            {se_multi:.4f}")
    if se_single > 0:
        print(f"  understatement factor: {se_multi / se_single:.2f}x")

    if a.out:
        a.out.write_text(json.dumps({
            "arm_a": a.arm_a, "arm_b": a.arm_b, "n_runs": N, "n_partitions": k,
            "grand_mean": grand, "sigma_seed": math.sqrt(var_seed),
            "sigma_partition": math.sqrt(var_part),
            "sigma_partition_raw_variance": var_part_raw,
            "se_ignoring_partition": se_single, "se_accounting": se_multi,
            "per_partition": {p: {"n": len(v), "mean": st.mean(v)} for p, v in by_p.items()},
        }, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
