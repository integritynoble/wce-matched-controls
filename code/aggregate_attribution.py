"""The control hierarchy, in-domain and on the external cohort.

Reproduces the attribution results in
`paper/Capsule-Endoscopy/ATTRIBUTION_RESULTS_2026-08-27.md`.

The hierarchy, weakest falsification to strongest:

    C0 rgb           no added representation
    C1 zeros         the 5-channel path; extra kernels train to exactly zero,
                     so this model is functionally 3-channel at inference
    C2 gauss,
       random_fixed  active channels carrying no image-specific content
    C3 phi_dup       geometry without chromophore content
    C4 shuffled      the frame's own marginals, spatial structure destroyed
    C5 cross_image   a REAL prior, from the wrong frame: correct marginals,
                     correct radial geometry, realistic texture, wrong patient
    P  prior         the correct image-specific representation

C5 is the primary contrast. Every other control is recognisably not a prior, so
a network could discriminate prior-like from not-prior-like instead of using
content; a donor prior removes that escape route.

Absence claims use TOST at +/-0.023, the effect size the earlier preprint
reported, so "no effect" means an effect that large is excluded rather than
merely unproven.

USAGE
    python aggregate_attribution.py                    # in-domain
    python aggregate_attribution.py --corpus galar_all # external cohort
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

from scipy import stats

BIOHPC = Path("/home/S248103/biohpc")
HIERARCHY = ["rgb", "zeros", "gauss", "random_fixed", "phi_dup", "shuffled",
             "cross_image", "prior"]
TOST_MARGIN = 0.023


def load(corpus: str) -> dict[tuple[str, int], float]:
    """(arm, seed) -> macro-AUC on the chosen corpus.

    In-domain reads test_metrics.json's macro_auc. The external corpora read
    macro_auc_common, the pinned in-domain/Galar class intersection -- the wider
    per-dataset macros average over different class sets and are not comparable.
    """
    out = {}
    for seed in range(41, 85):
        root = BIOHPC / ("tmi_runs" if seed <= 52 else "tmi_runs_selAUC")
        for arm in HIERARCHY:
            if arm == "cross_image":
                d = BIOHPC / "tmi_runs_crossimage" / f"efficientnet_b0_cross_image_seed{seed}"
            else:
                d = root / f"efficientnet_b0_{arm}_seed{seed}"
            if corpus == "in_domain":
                f, key = d / "test_metrics.json", "macro_auc"
            else:
                f, key = d / f"galar_{corpus.removeprefix('galar_')}_test_auc.json", \
                         "macro_auc_common"
            if f.exists():
                v = json.loads(f.read_text()).get(key)
                if v is not None:
                    out[(arm, seed)] = v
    return out


def paired(R, a, b, seeds):
    d = [R[(a, s)] - R[(b, s)] for s in seeds if (a, s) in R and (b, s) in R]
    if len(d) < 3:
        return None
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    ci = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se)
    p = float(stats.ttest_1samp(d, 0).pvalue)
    tost = max(stats.t.sf((m + TOST_MARGIN) / se, len(d) - 1),
               stats.t.cdf((m - TOST_MARGIN) / se, len(d) - 1))
    return m, ci, p, tost, len(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="in_domain",
                    help="in_domain | galar_all | galar_pillcam | galar_olympus")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    R = load(a.corpus)
    seeds = sorted({s for arm, s in R if arm == "prior"})
    have = [arm for arm in HIERARCHY
            if sum(1 for s in seeds if (arm, s) in R) == len(seeds)]
    print(f"corpus={a.corpus}  seeds={len(seeds)}  arms complete: {len(have)}/{len(HIERARCHY)}")
    missing = [x for x in HIERARCHY if x not in have]
    if missing:
        print(f"  incomplete, excluded: {missing}")

    print("\narm means, ranked:")
    means = sorted(((st.mean([R[(arm, s)] for s in seeds]),
                     st.stdev([R[(arm, s)] for s in seeds]), arm) for arm in have),
                   reverse=True)
    for m, sd, arm in means:
        print(f"  {arm:13s} {m:.4f} +/- {sd:.4f}"
              f"{'   <- the analytic prior' if arm == 'prior' else ''}")
    rank = [arm for _, _, arm in means].index("prior") + 1
    print(f"\nprior ranks {rank} of {len(means)}")

    results = {"corpus": a.corpus, "n_seeds": len(seeds),
               "arm_means": {arm: {"mean": m, "sd": sd} for m, sd, arm in means},
               "prior_rank": rank, "contrasts": {}}

    print(f"\nprior against each control, paired within seed, TOST at "
          f"+/-{TOST_MARGIN}:")
    order = ["cross_image"] + [x for x in have if x not in ("prior", "cross_image")]
    for c in order:
        r = paired(R, "prior", c, seeds)
        if r is None:
            continue
        m, ci, p, tost, n = r
        tag = "  <- PRIMARY" if c == "cross_image" else ""
        print(f"  prior - {c:13s} {m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={p:.3f}  "
              f"{'EQUIV' if tost < 0.05 else 'not est.':9s}{tag}")
        results["contrasts"][f"prior-{c}"] = {
            "delta": m, "ci": list(ci), "p": p, "tost_p": tost,
            "equivalent": bool(tost < 0.05), "n": n}

    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
