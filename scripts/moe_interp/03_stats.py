#!/usr/bin/env python
"""Statistics behind the MoE biochemical-interpretability figure.

Produces four tidy tables:

  expert_residue_stats.csv   panel A. Family-composition-controlled amino-acid
      enrichment (the statistic already used for the v24 expert analysis:
      log2 P(aa|e) / sum_f P(f|e) P(aa|f)), accompanied by a Cochran-Mantel-Haenszel
      odds ratio stratified by TF family so the same claim carries a CI and a p-value.

  expert_role_stats.csv      panel B. For every (expert, structural role) pair, a
      logistic regression  role ~ 1{expert} + C(amino acid) + C(family)  on the
      co-crystal subset, with SEs clustered by gene. Amino-acid identity is a
      covariate, so a surviving effect is *not* explainable by composition.

  aa_matched_controls.csv    the amino-acid-matched controls: e.g. among Cys residues
      only, are e6 cysteines more often Zn ligands than other cysteines?

  corouting_stats.csv        panel C. Top-2 co-routing frequency against an
      independence expectation that preserves each expert's slot marginals, with a
      permutation p-value.

Usage:
  python scripts/moe_interp/03_stats.py [--indir results/moe_biochem]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

AAS = list("ACDEFGHIKLMNPQRSTVWY")
ROLES = ["base_contact", "major_groove", "minor_groove", "backbone_contact",
         "zn_coord", "buried_core", "helix", "strand", "loop", "ppi", "exposed_nondna"]
ROLE_LABEL = {
    "base_contact": "DNA base contact",
    "major_groove": "major-groove contact",
    "minor_groove": "minor-groove contact",
    "backbone_contact": "DNA backbone contact",
    "zn_coord": "Zn coordination",
    "buried_core": "buried core",
    "helix": "α-helix",
    "strand": "β-strand",
    "loop": "loop / turn",
    "ppi": "protein–protein interface",
    "exposed_nondna": "exposed non-DNA surface",
}
ENS_JSON = "results/moe_expert_interpretation/v24_ens_expert_residues.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── panel A ─────────────────────────────────────────────────────────────────────
def panel_a(tok, E, contact_enr):
    """Family-controlled AA enrichment + family-stratified CMH odds ratio."""
    aa = tok.aa.values
    fam = tok.fam_code.values
    e1 = tok.e1.values
    prot = tok.prot.values
    nfam = fam.max() + 1
    keep = np.isin(aa, AAS)
    aa, fam, e1, prot = aa[keep], fam[keep], e1[keep], prot[keep]
    aai = np.searchsorted(AAS, aa)

    N = len(aai)
    usage = np.bincount(e1, minlength=E) / N
    p_aa_f = np.zeros((nfam, 20))
    for f in range(nfam):
        s = fam == f
        if s.sum():
            p_aa_f[f] = np.bincount(aai[s], minlength=20) / s.sum()

    # one pass for the full (expert x family x amino-acid) count tensor; every
    # family-stratified 2x2 table below is a slice of it
    C = np.zeros((E, nfam, 20))
    np.add.at(C, (e1, fam, aai), 1.0)
    n_ef = C.sum(2)          # (E, nfam) tokens of expert e in family f
    n_fi = C.sum(0)          # (nfam, 20) tokens of amino acid i in family f
    n_f = C.sum((0, 2))      # (nfam,)

    rows, mat = [], np.zeros((E, 20))
    for e in range(E):
        s = e1 == e
        p_aa = np.bincount(aai[s], minlength=20) / s.sum() + 1e-9
        fam_mix = np.bincount(fam[s], minlength=nfam) / s.sum()
        exp_aa = fam_mix @ p_aa_f + 1e-9
        enr = np.log2(p_aa / exp_aa)
        mat[e] = enr
        for i, c in enumerate(AAS):
            # family-stratified 2x2: rows = {this expert, others}, cols = {this aa, other aa}
            tabs = []
            for f in range(nfam):
                if n_f[f] < 20:
                    continue
                a_ = C[e, f, i]
                b_ = n_ef[e, f] - a_
                c_ = n_fi[f, i] - a_
                d_ = n_f[f] - n_ef[e, f] - n_fi[f, i] + a_
                t = np.array([[a_, b_], [c_, d_]], float)
                if t.sum(0).min() == 0 or t.sum(1).min() == 0:
                    continue
                tabs.append(t)
            lor, se, pv = np.nan, np.nan, np.nan
            if tabs:
                try:
                    st = StratifiedTable(np.dstack(tabs))
                    lor = float(st.logodds_pooled)
                    se = float(st.logodds_pooled_se)
                    pv = float(st.test_null_odds().pvalue)
                except Exception:
                    pass
            sel = s & (aai == i)
            rows.append(dict(
                expert=e, expert_usage=round(float(usage[e]), 4),
                contact_enrichment=round(float(contact_enr[e]), 4),
                feature_type="amino_acid", feature=c,
                effect_size=round(float(enr[i]), 4),
                log2_odds_ratio=None if np.isnan(lor) else round(lor / np.log(2), 4),
                standard_error=None if np.isnan(se) else round(se / np.log(2), 4),
                p_value=pv,
                n_residues=int(sel.sum()),
                n_proteins=int(np.unique(prot[sel]).size),
                n_families=int(len(tabs)),
            ))
    df = pd.DataFrame(rows)
    ok = df.p_value.notna()
    df["fdr"] = np.nan
    df.loc[ok, "fdr"] = multipletests(df.p_value[ok], method="fdr_bh")[1]
    return df, mat, usage


# ── panel B ─────────────────────────────────────────────────────────────────────
def logit_effect(y, x, aa, fam, groups):
    """log OR for x=1 in  y ~ x + C(aa) + C(family), SEs clustered by `groups`."""
    D = [x.astype(float).reshape(-1, 1), np.ones((len(y), 1))]
    for lab in (aa, fam):
        u = pd.get_dummies(pd.Series(lab), drop_first=True).values.astype(float)
        if u.shape[1]:
            D.append(u)
    X = np.hstack(D)
    # drop covariate columns that are constant or collinear within this subset
    keep = np.ones(X.shape[1], bool)
    keep[2:] = X[:, 2:].std(axis=0) > 1e-9
    X = X[:, keep]
    try:
        res = sm.GLM(y.astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200)
        return float(res.params[0]), float(res.bse[0]), float(res.pvalues[0])
    except Exception:
        return np.nan, np.nan, np.nan


def panel_b(sub, E, usage, contact_enr):
    rows = []
    aa = sub.aa.values
    fam = sub.family_name.values
    grp = pd.factorize(sub.gene_symbol.values)[0]
    for role in ROLES:
        y = sub[role].values.astype(int)
        if y.sum() < 20 or y.sum() == len(y):
            log(f"  skip role {role}: n_positive={y.sum()}")
            continue
        for e in range(E):
            x = (sub.e1.values == e)
            npos = int(y[x].sum())
            if x.sum() < 30:
                continue
            if npos == 0 or npos == int(x.sum()):
                lor, se, pv = np.nan, np.nan, np.nan     # complete separation
            else:
                lor, se, pv = logit_effect(y, x, aa, fam, grp)
            rate = float(y[x].mean())
            bg = float(y[~x].mean())
            # a huge coefficient with a huge SE is quasi-separation, not an effect
            unstable = (not np.isfinite(lor)) or (np.isfinite(se) and se / np.log(2) > 3.0)
            rows.append(dict(
                expert=e, expert_usage=round(float(usage[e]), 4),
                contact_enrichment=round(float(contact_enr[e]), 3),
                feature_type="structural_role", feature=role,
                role_label=ROLE_LABEL[role],
                effect_size=None if np.isnan(lor) else round(lor / np.log(2), 4),
                log2_odds_ratio=None if np.isnan(lor) else round(lor / np.log(2), 4),
                standard_error=None if np.isnan(se) else round(se / np.log(2), 4),
                p_value=pv,
                rate_in_expert=round(rate, 4), rate_other=round(bg, 4),
                unstable=bool(unstable),
                n_residues=int(x.sum()), n_positive=npos,
                n_proteins=int(sub.prot[x].nunique()),
                n_families=int(sub.family_name[x].nunique()),
            ))
    df = pd.DataFrame(rows)
    ok = df.p_value.notna()
    df["fdr"] = np.nan
    df.loc[ok, "fdr"] = multipletests(df.p_value[ok], method="fdr_bh")[1]
    return df


# ── amino-acid-matched controls ────────────────────────────────────────────────
CONTROLS = [
    ("e6 Cys are Zn ligands", 6, ["C"], "zn_coord"),
    ("e6 Cys are DNA base contacts", 6, ["C"], "base_contact"),
    ("e1 Arg/Lys are DNA base contacts", 1, ["R", "K"], "base_contact"),
    ("e1 Arg/Lys are DNA backbone contacts", 1, ["R", "K"], "backbone_contact"),
    ("e7 Phe/Tyr are DNA-interface", 7, ["F", "Y"], "dna_any"),
    ("e7 Phe/Tyr are buried core", 7, ["F", "Y"], "buried_core"),
    ("e4 His are Zn ligands", 4, ["H"], "zn_coord"),
    ("e4 His are DNA base contacts", 4, ["H"], "base_contact"),
    ("e3 Leu are buried core", 3, ["L"], "buried_core"),
    ("e3 Lys/Arg are DNA base contacts", 3, ["K", "R"], "base_contact"),
    ("e5 Gly are loop/turn", 5, ["G"], "loop"),
    ("e2 Pro/Ala/Thr are loop/turn", 2, ["P", "A", "T"], "loop"),
    ("e0 Gln are DNA base contacts", 0, ["Q"], "base_contact"),
]


def aa_matched(sub):
    rows = []
    for name, e, aas, role in CONTROLS:
        m = sub[sub.aa.isin(aas)]
        if len(m) < 40:
            rows.append(dict(test=name, expert=e, amino_acids="/".join(aas), role=role,
                             note="too few residues", n_residues=len(m)))
            continue
        y = (m.base_contact | m.backbone_contact).values.astype(int) if role == "dna_any" \
            else m[role].values.astype(int)
        x = (m.e1.values == e)
        if y.sum() < 5 or x.sum() < 10 or y[x].sum() in (0, int(x.sum())):
            rows.append(dict(test=name, expert=e, amino_acids="/".join(aas), role=role,
                             note="degenerate", n_residues=len(m),
                             rate_in_expert=round(float(y[x].mean()), 4) if x.sum() else None,
                             rate_other=round(float(y[~x].mean()), 4) if (~x).sum() else None))
            continue
        grp = pd.factorize(m.gene_symbol.values)[0]
        lor, se, pv = logit_effect(y, x, np.array(["_"] * len(m)), m.family_name.values, grp)
        # Fisher exact on the pooled table as a distribution-free companion
        tab = [[int(y[x].sum()), int((~y.astype(bool))[x].sum())],
               [int(y[~x].sum()), int((~y.astype(bool))[~x].sum())]]
        rows.append(dict(
            test=name, expert=e, amino_acids="/".join(aas), role=role,
            n_residues=int(len(m)), n_in_expert=int(x.sum()),
            rate_in_expert=round(float(y[x].mean()), 4),
            rate_other=round(float(y[~x].mean()), 4),
            log2_odds_ratio=None if np.isnan(lor) else round(lor / np.log(2), 4),
            standard_error=None if np.isnan(se) else round(se / np.log(2), 4),
            p_value=pv,
            fisher_p=float(sps.fisher_exact(tab)[1]),
            n_proteins=int(m.prot[x].nunique()),
        ))
    df = pd.DataFrame(rows)
    if "p_value" in df:
        ok = df.p_value.notna()
        df["fdr"] = np.nan
        df.loc[ok, "fdr"] = multipletests(df.p_value[ok], method="fdr_bh")[1]
    return df


# ── panel C ─────────────────────────────────────────────────────────────────────
def corouting(e1, e2, E, n_perm=2000, seed=0):
    """Observed vs independence-expected top-2 pair frequencies + permutation p."""
    lo = np.minimum(e1, e2).astype(np.int64)
    hi = np.maximum(e1, e2).astype(np.int64)
    assert (lo != hi).all(), "top-2 slots should never select the same expert"
    code = lo * E + hi
    obs = np.bincount(code, minlength=E * E).astype(float)
    N = len(e1)

    p1 = np.bincount(e1, minlength=E) / N
    p2 = np.bincount(e2, minlength=E) / N
    exp = np.zeros(E * E)
    for i in range(E):
        for j in range(i + 1, E):
            exp[i * E + j] = p1[i] * p2[j] + p1[j] * p2[i]
    exp = exp / exp.sum() * N

    rng = np.random.RandomState(seed)
    ge = np.zeros(E * E)
    for _ in range(n_perm):
        perm = e2[rng.permutation(N)]
        bad = perm == e1
        while bad.any():                       # top-2 pairs cannot be self-pairs
            perm[bad] = e2[rng.randint(0, N, bad.sum())]
            bad = perm == e1
        l = np.minimum(e1, perm).astype(np.int64)
        h = np.maximum(e1, perm).astype(np.int64)
        ge += np.bincount(l * E + h, minlength=E * E) >= obs
    rows = []
    for i in range(E):
        for j in range(i + 1, E):
            k = i * E + j
            rows.append(dict(
                expert_i=i, expert_j=j,
                observed_count=int(obs[k]),
                observed_frequency=round(float(obs[k] / N), 5),
                expected_frequency=round(float(exp[k] / N), 5),
                log2_observed_expected=round(float(np.log2((obs[k] + .5) / (exp[k] + .5))), 4),
                permutation_p=float(min(1.0, (min(ge[k], n_perm - ge[k]) * 2 + 1) / (n_perm + 1))),
            ))
    df = pd.DataFrame(rows)
    df["fdr"] = multipletests(df.permutation_p, method="fdr_bh")[1]
    return df


# ── supplementary: cross-seed reproducibility as cosine similarity ─────────────
def cross_seed_cosine():
    if not os.path.exists(ENS_JSON):
        return None
    from scipy.optimize import linear_sum_assignment
    R = json.load(open(ENS_JSON))
    seeds = list(R["per_seed"])

    def prof(tag):
        ex = R["per_seed"][tag]["experts"]
        E = len(R["per_seed"][tag]["expert_usage"])
        return np.array([[ex[str(e)].get("aa_enrichment_within_family", {}).get(c, 0.0)
                          for c in AAS] for e in range(E)])

    ref = prof(seeds[0])
    E = ref.shape[0]
    cos = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    rng = np.random.RandomState(0)
    rows = []
    for s in seeds[1:]:
        P = prof(s)
        M = np.array([[cos(ref[i], P[j]) for j in range(E)] for i in range(E)])
        ri, ci = linear_sum_assignment(-M)
        matched = M[ri, ci]
        null = np.array([M[np.arange(E), rng.permutation(E)].mean() for _ in range(2000)])
        for i, j in zip(ri, ci):
            rows.append(dict(seed=s, ref_expert=int(i), matched_expert=int(j),
                             cosine=round(float(M[i, j]), 4)))
        rows.append(dict(seed=s, ref_expert=-1, matched_expert=-1,
                         cosine=round(float(matched.mean()), 4),
                         median_cosine=round(float(np.median(matched)), 4),
                         null_mean_cosine=round(float(null.mean()), 4),
                         p_vs_null=float((null >= matched.mean()).mean())))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()

    tok = pd.read_parquet(os.path.join(a.indir, "routing_seed42.parquet"))
    ptab = pd.read_parquet(os.path.join(a.indir, "protein_table.parquet"))
    ann = pd.read_parquet(os.path.join(a.indir, "structural_annotations.parquet"))
    E = int(tok.e1.max()) + 1
    tok = tok.merge(ptab[["prot", "gene_symbol", "family_name"]], on="prot", how="left")
    tok["fam_code"] = pd.factorize(tok.family_name)[0]
    log(f"{len(tok)} routed residues, {E} experts, {tok.prot.nunique()} proteins")

    # DNA-contact enrichment on the v23 supervision set (the Panel A right sidebar)
    s = tok[tok.has_contact_label]
    bg = s.base_contact_v23.mean()
    contact_enr = np.array([np.log2(s.base_contact_v23[s.e1 == e].mean() / bg) for e in range(E)])
    log("contact enrichment (v23 any-contact set): " +
        " ".join(f"e{e}={contact_enr[e]:+.3f}" for e in range(E)))

    log("panel A: family-controlled AA enrichment + CMH odds ratios ...")
    a_df, a_mat, usage = panel_a(tok, E, contact_enr)

    # independent cross-check of the contact axis: recomputed co-crystal contacts
    st = ann.merge(tok[["prot", "pos", "e1"]], on=["prot", "pos"], how="inner")
    st_any = (st.base_contact | st.backbone_contact)
    bg_s = st_any.mean()
    st_enr = np.array([np.log2(st_any[st.e1 == e].mean() / bg_s) for e in range(E)])
    st_base = np.array([np.log2(st.base_contact[st.e1 == e].mean() / st.base_contact.mean())
                        for e in range(E)])
    log("structural cross-check, any-contact enrichment: " +
        " ".join(f"e{e}={st_enr[e]:+.3f}" for e in range(E)))
    log("  Spearman(v23 axis, structural axis) = "
        f"{sps.spearmanr(contact_enr, st_enr).statistic:.3f}")

    pd.DataFrame({
        "expert": range(E),
        "expert_usage": [round(float(usage[e]), 4) for e in range(E)],
        "contact_enrichment": [round(float(contact_enr[e]), 4) for e in range(E)],
        "contact_enrichment_structural_any": [round(float(st_enr[e]), 4) for e in range(E)],
        "contact_enrichment_structural_base": [round(float(st_base[e]), 4) for e in range(E)],
        "n_residues": [int((tok.e1 == e).sum()) for e in range(E)],
        "n_top2_slots": [int(((tok.e1 == e) | (tok.e2 == e)).sum()) for e in range(E)],
    }).to_csv(os.path.join(a.indir, "expert_summary.csv"), index=False)
    a_df.to_csv(os.path.join(a.indir, "expert_residue_stats.csv"), index=False)
    np.savetxt(os.path.join(a.indir, "panelA_matrix.csv"), a_mat, delimiter=",",
               header=",".join(AAS), comments="")

    log("panel B: structural-role logistic regressions ...")
    sub = ann.merge(tok[["prot", "pos", "e1", "e2", "w1"]], on=["prot", "pos"], how="inner")
    sub = sub.merge(ptab[["prot", "gene_symbol", "family_name"]], on="prot", how="left")
    log(f"  co-crystal subset: {len(sub)} residues, {sub.prot.nunique()} rows, "
        f"{sub.gene_symbol.nunique()} genes, {sub.pdb_id.nunique()} PDB entries")
    b_df = panel_b(sub, E, usage, contact_enr)
    b_df.to_csv(os.path.join(a.indir, "expert_role_stats.csv"), index=False)

    # sensitivity: many genes contribute several near-identical crops, so repeat the
    # regressions on one row per gene (the longest crop)
    keep = (sub.groupby(["gene_symbol", "prot"]).size().reset_index(name="n")
            .sort_values("n", ascending=False).groupby("gene_symbol").head(1).prot)
    dedup = sub[sub.prot.isin(set(keep))]
    log(f"  dedup sensitivity: {len(dedup)} residues from {dedup.prot.nunique()} rows "
        f"({dedup.gene_symbol.nunique()} genes)")
    panel_b(dedup, E, usage, contact_enr).to_csv(
        os.path.join(a.indir, "expert_role_stats_dedup.csv"), index=False)

    log("amino-acid-matched controls ...")
    c_df = aa_matched(sub)
    c_df.to_csv(os.path.join(a.indir, "aa_matched_controls.csv"), index=False)
    for r in c_df.itertuples():
        log(f"  {r.test:42s} lor2={getattr(r, 'log2_odds_ratio', None)} "
            f"p={getattr(r, 'p_value', None)} "
            f"rate {getattr(r, 'rate_in_expert', None)} vs {getattr(r, 'rate_other', None)}")

    log(f"panel C: top-2 co-routing ({a.n_perm} permutations) ...")
    d_df = corouting(tok.e1.values.astype(int), tok.e2.values.astype(int), E, a.n_perm)
    d_df.to_csv(os.path.join(a.indir, "corouting_stats.csv"), index=False)
    top = d_df.reindex(d_df.log2_observed_expected.abs().sort_values(ascending=False).index)
    for r in top.head(6).itertuples():
        log(f"  e{r.expert_i}-e{r.expert_j}: log2(obs/exp)={r.log2_observed_expected:+.3f} "
            f"obs={r.observed_frequency:.4f} exp={r.expected_frequency:.4f} p={r.permutation_p:.4f}")

    # single tidy table over both feature types, in the requested column order
    COLS = ["expert", "expert_usage", "contact_enrichment", "feature_type", "feature",
            "effect_size", "log2_odds_ratio", "standard_error", "p_value", "fdr",
            "n_residues", "n_proteins", "n_families"]
    tidy = pd.concat([a_df, b_df], ignore_index=True)
    for c in COLS:
        if c not in tidy:
            tidy[c] = np.nan
    tidy[COLS].to_csv(os.path.join(a.indir, "expert_feature_stats.csv"), index=False)
    log(f"combined tidy table: {len(tidy)} rows -> expert_feature_stats.csv")

    cs = cross_seed_cosine()
    if cs is not None:
        cs.to_csv(os.path.join(a.indir, "cross_seed_cosine.csv"), index=False)
        summ = cs[cs.ref_expert == -1]
        log("cross-seed matched cosine (supplementary): " +
            " ".join(f"{r.seed}={r.cosine:.3f}" for r in summ.itertuples()))

    # per-family co-routing for the supplement
    fam_rows = []
    for fam, g in tok.groupby("family_name"):
        if len(g) < 3000:
            continue
        m = corouting(g.e1.values.astype(int), g.e2.values.astype(int), E, n_perm=500, seed=1)
        m["family_name"] = fam
        m["n_residues"] = len(g)
        fam_rows.append(m)
    if fam_rows:
        pd.concat(fam_rows, ignore_index=True).to_csv(
            os.path.join(a.indir, "corouting_by_family.csv"), index=False)
        log(f"per-family co-routing written for {len(fam_rows)} families")
    log("done")


if __name__ == "__main__":
    main()
