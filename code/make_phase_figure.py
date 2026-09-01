"""The phase diagram: simulation against real images.

The paper's central empirical claim is regime dependence, and the figure has to
carry two things at once -- what the simulation shows and how much of it
survives on clinical data. Plotting only the simulation would oversell; plotting
only the real data would lose the calibration that makes the null interpretable.

Panel A: simulation, Delta_attr vs N at two capacities. The effect that decays
~400x, which is what establishes the instrument can detect a real effect.

Panel B: Kvasir-Capsule, Delta_attr vs training fraction at two capacities,
with floor-gated cells drawn open and unfilled. Those cells are NOT dropped
silently -- a reader must see that the corner where panel A is largest is the
corner where panel B cannot measure anything.

Error bars are +-1 SE throughout. The y-axes are deliberately NOT shared: panel
A spans 0.25 and panel B spans 0.06, and forcing a common axis would render
panel B a flat line at zero, which would misrepresent a measured contrast as an
absence of one.

USAGE
    python make_phase_figure.py --sim positive_control_results.json \
        --real regime_results_n25.json --out figures/phase_diagram.pdf
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTROL = "cross_image"      # the primary control, both panels


def sim_points(path: Path):
    """Delta_attr = AUC(prior) - AUC(cross_image), per (N, width) cell."""
    A = json.loads(path.read_text())
    out = {}
    for k in A:
        n, w, arm = k.split("|")
        if arm != "prior":
            continue
        ck = f"{n}|{w}|{CONTROL}"
        if ck not in A:
            continue
        d = st.mean(A[k]) - st.mean(A[ck])
        se = float(np.hypot(np.std(A[k], ddof=1), np.std(A[ck], ddof=1))
                   / np.sqrt(len(A[k])))
        out.setdefault(int(w), []).append((int(n), d, se))
    for w in out:
        out[w].sort()
    return out


def real_points(path: Path):
    rows = json.loads(path.read_text())
    out = {}
    for r in rows:
        out.setdefault(r["cap"], []).append(
            (r["frac"], r["delta_attr"], r["se_attr"], r["gate"]))
    for c in out:
        out[c].sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sim = sim_points(a.sim)
    real = real_points(a.real)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    styles = {8: ("o-", "#1b4965", "width 8"), 32: ("s--", "#5fa8d3", "width 32")}
    for w, pts in sorted(sim.items()):
        mk, col, lab = styles.get(w, ("^-", "gray", f"width {w}"))
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        ax1.errorbar(x, y, yerr=e, fmt=mk, color=col, label=lab,
                     capsize=2.5, lw=1.4, ms=5)
    ax1.set_xscale("log")
    # Explicit ticks at the sampled values. Matplotlib's default log minor ticks
    # collide into unreadable overlap at this figure width.
    xs = sorted({p[0] for pts in sim.values() for p in pts})
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(v) for v in xs])
    ax1.minorticks_off()
    ax1.axhline(0, color="k", lw=0.7, alpha=0.5)
    ax1.set_xlabel("training samples $N$")
    ax1.set_ylabel(r"$\Delta_{\mathrm{attr}}$ (AUC)")
    ax1.set_title("A. Simulation: a correct prior", fontsize=10, loc="left", pad=26)
    ax1.legend(frameon=False, fontsize=8)

    cstyle = {"scr": ("o-", "#9b2226", "from scratch"),
              "pre": ("s--", "#e09f3e", "pretrained")}
    for cap, pts in sorted(real.items()):
        mk, col, lab = cstyle.get(cap, ("^-", "gray", cap))
        ok = [p for p in pts if p[3] == "ok"]
        bad = [p for p in pts if p[3] != "ok"]
        if ok:
            ax2.errorbar([p[0] for p in ok], [p[1] for p in ok],
                         yerr=[p[2] for p in ok], fmt=mk, color=col,
                         label=lab, capsize=2.5, lw=1.4, ms=5)
        # Gated cells: shown, hollow, excluded from the line. Visible so the
        # reader sees where measurement fails rather than where effect vanishes.
        if bad:
            ax2.errorbar([p[0] for p in bad], [p[1] for p in bad],
                         yerr=[p[2] for p in bad], fmt="o", color=col,
                         mfc="none", capsize=2.5, lw=1.0, ms=6, alpha=0.75,
                         label=f"{lab} (floor: not interpretable)")
    ax2.set_xscale("log")
    fx = sorted({p[0] for pts in real.values() for p in pts})
    ax2.set_xticks(fx)
    ax2.set_xticklabels([f"{v:g}" for v in fx])
    ax2.minorticks_off()
    ax2.axhline(0, color="k", lw=0.7, alpha=0.5)
    # Second axis in absolute frames, so a reader can place panel B against
    # panel A's N without doing arithmetic: 31,820 training frames at 100%.
    sec = ax2.secondary_xaxis("top", functions=(lambda f: f * 31820,
                                                lambda n: n / 31820))
    sec.set_xlabel("training frames", fontsize=8)
    sec.set_xticks([318, 955, 3182, 9546, 31820])
    sec.set_xticklabels(["318", "955", "3.2k", "9.5k", "32k"], fontsize=7.5)
    ax2.set_xlabel("fraction of training data")
    ax2.set_ylabel(r"$\Delta_{\mathrm{attr}}$ (macro-AUC)")
    ax2.set_title("B. Kvasir-Capsule: the analytic prior", fontsize=10, loc="left", pad=26)
    ax2.legend(frameon=False, fontsize=7.5)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} and {a.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
