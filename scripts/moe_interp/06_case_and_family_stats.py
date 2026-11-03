#!/usr/bin/env python
"""Positional and cross-family statistics for panels d, e and f.

d  C2H2 recognition-helix register. Every C-x(2,4)-C-x(12)-H-x(3,5)-H module in the
   routing table is located by motif, anchored on the first His (canonical helix
   position +7), and the top-1 expert is tallied per register position. Aggregating
   over thousands of fingers turns the single-protein cartoon into a real distribution.

e  bZIP / bHLH basic-region-to-dimerisation register. Crops differ in what they cover,
   so the anchor is structural rather than sequence-based: position 0 is the last
   DNA-base-contacting residue of the crop, i.e. the end of the basic region. Negative
   positions are the DNA-binding basic region, positive positions the helix-1 /
   coiled-coil dimerisation element.

f  Cross-family reuse. The question is not how often a family uses an expert but
   whether the expert performs the *same chemical operation* wherever it is used. So the
   headline quantity is the cosine between an expert's family-controlled amino-acid
   profile computed inside one family and its global profile: ~1 means the same program
   is being reused in that clade, ~0 means the expert is doing something unrelated
   there. Usage enrichment P(expert | family) / P(expert) is reported alongside.
   Both carry a protein-level bootstrap CI.

Usage:
  python scripts/moe_interp/06_case_and_family_stats.py [--indir results/moe_biochem]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

C2H2_MOTIF = re.compile(r"C.{2,4}C.{12}H.{3,5}H")
# register window around the canonical recognition helix (first His = +7)
D_LO, D_HI = -13, 13
# bZIP/bHLH window around the end of the basic region (last base contact = 0)
E_LO, E_HI = -22, 24
FAMILIES_F = ["C2H2_long", "bHLH", "bZIP", "Homeodomain", "ETS"]
FAMILY_LABEL = {"C2H2_long": "C2H2", "bHLH": "bHLH", "bZIP": "bZIP",
                "Homeodomain": "HD", "ETS": "ETS"}


def log(m):
    print(m, flush=True)


# ── d ───────────────────────────────────────────────────────────────────────────
def c2h2_register(tok, ptab, E):
    rows, n_fing, n_prot = [], 0, 0
    e1_by_prot = {p: g.sort_values("pos").e1.values
                  for p, g in tok.groupby("prot")}
    for r in ptab[ptab.family_name.astype(str).str.startswith("C2H2")].itertuples():
        seq = str(r.seq)
        experts = e1_by_prot.get(int(r.prot))
        if experts is None or len(experts) != len(seq):
            continue
        hits = list(C2H2_MOTIF.finditer(seq))
        if not hits:
            continue
        n_prot += 1
        for m in hits:
            h1 = m.start() + m.group().index("H", 4)   # first His of the HxxxH pair
            n_fing += 1
            for p in range(D_LO, D_HI + 1):
                i = h1 - 7 + p
                if 0 <= i < len(seq):
                    rows.append((p, int(experts[i]), seq[i]))
    d = pd.DataFrame(rows, columns=["position", "expert", "aa"])
    log(f"  panel d: {n_fing} C2H2 fingers from {n_prot} proteins")

    out = []
    for p, g in d.groupby("position"):
        cnt = np.bincount(g.expert.values, minlength=E)
        top_aa = g.aa.value_counts()
        for e in range(E):
            out.append(dict(position=p, expert=e, n=int(cnt[e]),
                            frac=round(float(cnt[e] / cnt.sum()), 4),
                            n_fingers=int(cnt.sum()),
                            modal_aa=top_aa.index[0],
                            modal_aa_frac=round(float(top_aa.iloc[0] / len(g)), 3)))
    return pd.DataFrame(out)


# ── e ───────────────────────────────────────────────────────────────────────────
def bzip_bhlh_register(tok, ann, ptab, E):
    fam = ptab.set_index("prot").family_name
    sub = ann.merge(tok[["prot", "pos", "e1"]], on=["prot", "pos"])
    rows, n_dom = [], 0
    for p, g in sub.groupby("prot"):
        if fam.get(p) not in ("bZIP", "bHLH"):
            continue
        g = g.sort_values("pos")
        bc = g.pos.values[g.base_contact.values]
        if not len(bc):
            continue
        anchor = int(bc.max())          # end of the basic / DNA-reading region
        n_dom += 1
        for pos, e in zip(g.pos.values, g.e1.values):
            k = int(pos) - anchor
            if E_LO <= k <= E_HI:
                rows.append((k, int(e), fam.get(p)))
    d = pd.DataFrame(rows, columns=["position", "expert", "family"])
    log(f"  panel e: {n_dom} bZIP/bHLH domains anchored on the end of the basic region "
        f"({dict(d.groupby('family').apply(lambda g: g.position.nunique(), include_groups=False))} "
        f"positions covered per family)")
    # bZIP and bHLH behave differently here (bZIP crops rarely reach the zipper, and its
    # basic/dimerisation contrast reverses), so emit each subset separately as well as
    # the pool, and let the figure choose.
    out = []
    for subset, dd in [("bHLH", d[d.family == "bHLH"]), ("bZIP", d[d.family == "bZIP"]),
                       ("both", d)]:
        for k, g in dd.groupby("position"):
            cnt = np.bincount(g.expert.values, minlength=E)
            for e in range(E):
                out.append(dict(subset=subset, position=k, expert=e, n=int(cnt[e]),
                                frac=round(float(cnt[e] / cnt.sum()), 4),
                                n_domains=int(cnt.sum())))
    return pd.DataFrame(out)


# ── f ───────────────────────────────────────────────────────────────────────────
AAS = list("ACDEFGHIKLMNPQRSTVWY")
MIN_RES = 200          # residues an expert needs inside a family to get a profile


def _profiles(aai, e1, E):
    """Family-controlled log2 chemistry profile per expert on one residue subset."""
    bg = np.bincount(aai, minlength=20) / max(len(aai), 1) + 1e-9
    out = {}
    for e in range(E):
        s = e1 == e
        if s.sum() < MIN_RES:
            out[e] = None
            continue
        p = np.bincount(aai[s], minlength=20) / s.sum() + 1e-9
        out[e] = np.log2(p / bg)
    return out


def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def family_reuse(tok, E, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    tok = tok[tok.aa.isin(AAS)]
    aai_all = np.searchsorted(AAS, tok.aa.values)
    e1_all = tok.e1.values
    glob_usage = np.bincount(e1_all, minlength=E) / len(e1_all)
    glob_prof = _profiles(aai_all, e1_all, E)

    rows = []
    for fam in FAMILIES_F:
        g = tok[tok.family_name == fam]
        if not len(g):
            continue
        aai = np.searchsorted(AAS, g.aa.values)
        e1 = g.e1.values
        prots = g.prot.unique()
        idx_by_prot = {p: np.where(g.prot.values == p)[0] for p in prots}

        obs_usage = np.bincount(e1, minlength=E) / len(e1)
        obs_prof = _profiles(aai, e1, E)

        bu = np.empty((n_boot, E))
        bc = np.full((n_boot, E), np.nan)
        for b in range(n_boot):
            pick = np.concatenate([idx_by_prot[p]
                                   for p in rng.choice(prots, len(prots), replace=True)])
            bu[b] = np.bincount(e1[pick], minlength=E) / len(pick)
            pr = _profiles(aai[pick], e1[pick], E)
            for e in range(E):
                if pr[e] is not None and glob_prof[e] is not None:
                    bc[b, e] = _cos(glob_prof[e], pr[e])

        for e in range(E):
            ok = obs_prof[e] is not None and glob_prof[e] is not None
            cs = bc[:, e][~np.isnan(bc[:, e])]
            rows.append(dict(
                family=fam, family_label=FAMILY_LABEL.get(fam, fam), expert=e,
                profile_cosine=round(_cos(glob_prof[e], obs_prof[e]), 4) if ok else None,
                cos_ci_lo=round(float(np.percentile(cs, 2.5)), 4) if len(cs) > 50 else None,
                cos_ci_hi=round(float(np.percentile(cs, 97.5)), 4) if len(cs) > 50 else None,
                usage_in_family=round(float(obs_usage[e]), 4),
                usage_global=round(float(glob_usage[e]), 4),
                usage_enrichment=round(float(obs_usage[e] / glob_usage[e]), 4),
                usage_ci_lo=round(float(np.percentile(bu[:, e] / glob_usage[e], 2.5)), 4),
                usage_ci_hi=round(float(np.percentile(bu[:, e] / glob_usage[e], 97.5)), 4),
                n_residues_expert_family=int((e1 == e).sum()),
                n_residues=int(len(g)), n_proteins=int(len(prots))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    tok = pd.read_parquet(os.path.join(a.indir, "routing_seed42.parquet"))
    ptab = pd.read_parquet(os.path.join(a.indir, "protein_table.parquet"))
    ann = pd.read_parquet(os.path.join(a.indir, "structural_annotations.parquet"))
    tok = tok.merge(ptab[["prot", "family_name"]], on="prot", how="left")
    E = int(tok.e1.max()) + 1

    d = c2h2_register(tok, ptab, E)
    d.to_csv(os.path.join(a.indir, "c2h2_register.csv"), index=False)
    piv = d.pivot(index="position", columns="expert", values="frac")
    log("  dominant expert by C2H2 register position "
        "(C2 = -6, recognition = -1/+2/+3/+6, His = +7 and +10..12):")
    aa = d.drop_duplicates("position").set_index("position").modal_aa
    for p in range(D_LO, D_HI + 1):
        if p in piv.index:
            e = int(piv.loc[p].idxmax())
            log(f"    {p:+3d}  {aa[p]}  e{e} ({piv.loc[p, e]:.2f})")

    e_df = bzip_bhlh_register(tok, ann, ptab, E)
    e_df.to_csv(os.path.join(a.indir, "bzip_bhlh_register.csv"), index=False)

    f = family_reuse(tok, E, a.n_boot)
    f.to_csv(os.path.join(a.indir, "family_reuse.csv"), index=False)
    log("  panel f: chemistry-profile cosine vs global  (usage enrichment in brackets)")
    for e in sorted(f.expert.unique()):
        g = f[f.expert == e]
        log(f"    e{e}: " + "  ".join(
            (f"{r.family_label}={r.profile_cosine:+.2f}[{r.usage_enrichment:.2f}]"
             if r.profile_cosine is not None else f"{r.family_label}=n/a")
            for r in g.itertuples()))
    log("done")


if __name__ == "__main__":
    main()
