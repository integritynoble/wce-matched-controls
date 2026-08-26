#!/usr/bin/env python3
"""Regenerate every number in Results of the TMI control paper.

Written 2026-08-11, when both control matrices reached n=44 and the section
still carried n=12 tables. The point is that the manuscript's numbers should
come from one command against the run directories, not from notes.

Conventions matched to the existing tables: arm-vs-arm and arm-vs-baseline
p-values are paired two-sided t-tests on within-seed deltas (the sign tests in
aggregate_controls.py are a separate, more conservative check and are printed
alongside). Power is exact non-central t, not the normal approximation --
the latter is optimistic at these sample sizes and gave 29/152 where the exact
calculation gives 31/155.

--validate re-derives published EfficientNet-B0 figures and asserts they match
before any new number is printed.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
from scipy import stats

ARMS = ["rgb","prior","zeros","shuffled","random_fixed","phi_dup","gauss"]

def load(runs_dirs, model):
    pat = re.compile(rf"^{re.escape(model)}_(?P<arm>.+)_seed(?P<seed>\d+)$")
    out={}
    for rd in runs_dirs:
        for d in sorted(glob.glob(os.path.join(os.path.expanduser(rd),"*"))):
            m=pat.match(os.path.basename(d))
            if not m or not os.path.exists(os.path.join(d,"test_predictions.npz")): continue
            mj=os.path.join(d,"test_metrics.json")
            if not os.path.exists(mj): continue
            a=json.load(open(mj)).get("macro_auc")
            if a is None: continue
            out.setdefault(m.group("arm"),{})[int(m.group("seed"))]=float(a)
    return out

def deltas(data,a,b):
    s=sorted(set(data[a])&set(data[b]))
    return np.array([data[a][x]-data[b][x] for x in s])

def paired_t(d):
    n=len(d); mu=d.mean(); sd=d.std(ddof=1); se=sd/np.sqrt(n)
    t,p=stats.ttest_1samp(d,0.0)
    lo,hi=stats.t.interval(0.95,n-1,loc=mu,scale=se)
    return dict(n=n,mean=mu,sd=sd,p=p,ci=(lo,hi))

def tost(d,margin):
    n=len(d); mu=d.mean(); se=d.std(ddof=1)/np.sqrt(n); df=n-1
    p_lo=stats.t.sf((mu+margin)/se,df)      # H0: mu <= -margin
    p_hi=stats.t.cdf((mu-margin)/se,df)     # H0: mu >= +margin
    return max(p_lo,p_hi)

def tightest(d,alpha=0.05,hi=0.08,step=0.0001):
    m=step
    while m<hi:
        if tost(d,m)<alpha: return m
        m+=step
    return None

def power_n(effect,sd,power=0.80,alpha=0.05,nmax=1000):
    """Smallest n whose exact non-central t power reaches `power`."""
    for n in range(3,nmax+1):
        df=n-1; lam=effect/(sd/np.sqrt(n)); crit=stats.t.ppf(1-alpha/2,df)
        pw=stats.nct.sf(crit,df,lam)+stats.nct.cdf(-crit,df,lam)
        if pw>=power: return n
    return None

def arm_table(data,ref="rgb"):
    rows=[]
    for arm in ARMS:
        if arm not in data: continue
        vals=np.array([data[arm][s] for s in sorted(data[arm])])
        row=dict(arm=arm,n=len(vals),mean=vals.mean(),sd=vals.std(ddof=1))
        if arm!=ref:
            d=deltas(data,arm,ref); r=paired_t(d)
            row.update(delta=r["mean"],p=r["p"],ci=r["ci"],
                       sign_p=stats.binomtest(int((d>0).sum()),len(d),0.5).pvalue)
        rows.append(row)
    return rows

def show(title,rows,ref="rgb"):
    print(f"\n{title}")
    print(f"  {'arm':14s} {'n':>3s} {'mean':>8s} {'sd':>7s} {'D vs '+ref:>9s} "
          f"{'95% CI':>22s} {'t-p':>7s} {'sign-p':>7s}")
    for r in sorted(rows,key=lambda x:-x["mean"]):
        if "delta" in r:
            print(f"  {r['arm']:14s} {r['n']:3d} {r['mean']:8.4f} {r['sd']:7.4f} "
                  f"{r['delta']:+9.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                  f"{r['p']:7.3f} {r['sign_p']:7.3f}")
        else:
            print(f"  {r['arm']:14s} {r['n']:3d} {r['mean']:8.4f} {r['sd']:7.4f} "
                  f"{'---':>9s} {'---':>22s} {'---':>7s} {'---':>7s}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--validate",action="store_true")
    a=ap.parse_args()
    E=load(["~/biohpc/tmi_runs","~/biohpc/tmi_runs_selAUC"],"efficientnet_b0")
    C=load(["~/biohpc/tmi_runs","~/biohpc/tmi_runs_convnext_ext"],"convnext_tiny")
    F=load(["~/biohpc/tmi_runs_selF1"],"efficientnet_b0")

    if a.validate:
        r=paired_t(deltas(E,"prior","rgb"))
        assert r["n"]==44 and abs(r["mean"]-0.0068)<5e-4, r
        assert abs(r["ci"][0]+0.0066)<1e-3 and abs(r["ci"][1]-0.0202)<1e-3, r["ci"]
        assert abs(r["p"]-0.31)<0.02, r["p"]
        assert abs(tost(deltas(E,"prior","rgb"),0.023)-0.0096)<5e-4
        assert abs(tightest(deltas(E,"prior","rgb"))-0.018)<6e-4
        rf=paired_t(deltas(F,"prior","rgb"))
        assert abs(rf["mean"]+0.0023)<5e-4, rf
        assert abs(tost(deltas(F,"prior","rgb"),0.023)-0.0018)<5e-4
        assert power_n(0.023,0.0441)==31 and power_n(0.010,0.0441)==155
        print("VALIDATION OK -- published EffB0 figures reproduce "
              "(delta, CI, p, TOST, tightest margin, power 31/155)")

    show("EfficientNet-B0, seven arms, n=44",arm_table(E))
    show("ConvNeXt-Tiny, seven arms, n=44",arm_table(C))

    print("\nEquivalence, prior - rgb")
    for lab,d in [("EffB0 macro_auc",deltas(E,"prior","rgb")),
                  ("EffB0 macro_f1 ",deltas(F,"prior","rgb")),
                  ("ConvNeXt       ",deltas(C,"prior","rgb"))]:
        r=paired_t(d)
        print(f"  {lab}: n={r['n']} D={r['mean']:+.4f} sd={r['sd']:.4f} "
              f"CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2f}")
        print("      TOST  " + "  ".join(
            f"{m}:{tost(d,m):.4f}" for m in (0.023,0.020,0.015)) +
            f"   tightest=+-{tightest(d):.4f}")

    print("\nPower (exact non-central t, 80%)")
    for lab,d in [("EffB0   ",deltas(E,"prior","rgb")),("ConvNeXt",deltas(C,"prior","rgb"))]:
        sd=d.std(ddof=1)
        print(f"  {lab} sd={sd:.4f}  n(0.023)={power_n(0.023,sd)}  n(0.010)={power_n(0.010,sd)}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
