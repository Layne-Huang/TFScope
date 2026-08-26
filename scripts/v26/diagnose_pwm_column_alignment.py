#!/usr/bin/env python
"""Diagnostic: are assigned PWM columns concentrated on informative positions?

Validation logic: if the strand->PWM alignment is sound, contacted bases should sit on HIGH
information-content columns more often than chance. If they scatter uniformly, the placement is
noise and must not be used as supervision.

The first (unweighted, no-margin) version scored an IC-enrichment ratio of 1.11 -- statistically
significant only because n=75,889, effect size negligible. This version sweeps the runner-up
MARGIN threshold so the retention/enrichment trade-off is chosen from the data.

Null: for each retained contact, the IC of a uniformly random column of the SAME PWM. That
controls for PWMs differing in length and overall information.

  python scripts/v26/diagnose_pwm_column_alignment.py
"""
from __future__ import annotations
import json, numpy as np, pandas as pd

CD="data/contacts_v26"; V26D="data/processed/v26"; RESD="results/v26"
MARGINS=[0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0]

def ic(pwm):
    p=np.clip(pwm,1e-9,1.0); return 2.0+(p*np.log2(p)).sum(0)

df=pd.read_parquet(f"{CD}/contacts2d_core.parquet")
ex=pd.read_parquet(f"{V26D}/v26_core.parquet")

ic_of={}; L_of={}
for r in ex.itertuples():
    b=r.pwm
    if not isinstance(b,(bytes,bytearray)): continue
    a=np.frombuffer(b,dtype=np.float32)
    if a.size%4: continue
    m=a.reshape(4,-1).astype(float)[:, :int(r.motif_length)]
    if m.shape[1]==0: continue
    ic_of[r.example_id]=ic(m); L_of[r.example_id]=m.shape[1]

base=df[(df.column_status=="assigned")&df.in_crop].copy()
base=base[base.example_id.isin(ic_of)]
base["col_ic"]=[ic_of[e][int(c)] if int(c)<len(ic_of[e]) else np.nan
                for e,c in zip(base.example_id,base.pwm_column)]
base=base.dropna(subset=["col_ic"])
n_all=len(base)
rng=np.random.default_rng(0)

def evaluate(sub):
    obs=sub.col_ic.to_numpy()
    null=np.array([ic_of[e][rng.integers(0,L_of[e])] for e in sub.example_id])
    d={"contacts":int(len(sub)),
       "retention":round(len(sub)/max(n_all,1),4),
       "mean_IC_obs":round(float(obs.mean()),4),
       "mean_IC_null":round(float(null.mean()),4),
       "enrichment_ratio":round(float(obs.mean()/max(null.mean(),1e-9)),4),
       "frac_obs_IC_gt_1bit":round(float((obs>1.0).mean()),4),
       "frac_null_IC_gt_1bit":round(float((null>1.0).mean()),4),
       "examples":int(sub.example_id.nunique())}
    try:
        from scipy.stats import mannwhitneyu
        d["mw_p"]=float(mannwhitneyu(obs,null,alternative="greater").pvalue)
    except Exception: d["mw_p"]=None
    return d

has_margin = "align_margin" in base.columns and base.align_margin.notna().any()
sweep=[]
if has_margin:
    for m in MARGINS:
        sub=base[base.align_margin.fillna(0)>=m]
        if len(sub)<100: continue
        sweep.append({"margin":m, **evaluate(sub)})
else:
    print("WARNING: align_margin absent -- rebuild with the updated builder")

overall=evaluate(base)
res={"n_assigned_in_crop":n_all,"overall":overall,"margin_sweep":sweep}
json.dump(res,open(f"{RESD}/pwm_column_alignment_diagnostic.json","w"),indent=2)

print(json.dumps({"overall":overall},indent=2))
if sweep:
    print("\nmargin sweep (retention vs IC enrichment):")
    print(f"{'margin':>7} {'contacts':>9} {'retain':>7} {'meanIC':>7} {'null':>6} {'enrich':>7} {'>1bit':>6}")
    for r in sweep:
        print(f"{r['margin']:7.2f} {r['contacts']:9d} {r['retention']:7.3f} "
              f"{r['mean_IC_obs']:7.3f} {r['mean_IC_null']:6.3f} "
              f"{r['enrichment_ratio']:7.3f} {r['frac_obs_IC_gt_1bit']:6.3f}")
    ok=[r for r in sweep if r["enrichment_ratio"]>=1.25 and r["retention"]>=0.25]
    if ok:
        best=max(ok,key=lambda r:(r["retention"]))
        print(f"\nRECOMMENDED margin >= {best['margin']}: enrichment "
              f"{best['enrichment_ratio']}, retains {100*best['retention']:.0f}% "
              f"({best['contacts']} contacts, {best['examples']} examples)")
    else:
        b=max(sweep,key=lambda r:r["enrichment_ratio"])
        print(f"\nNO THRESHOLD reaches enrichment>=1.25 at >=25% retention. "
              f"Best is margin {b['margin']} -> enrichment {b['enrichment_ratio']} "
              f"(retains {100*b['retention']:.0f}%). 2-D supervision should stay DISABLED.")
