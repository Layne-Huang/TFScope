#!/usr/bin/env python
"""Stage 2 (refined): cluster-aware DBD cropping.

Three refinements over cluster_crop.py, each driven by a failure found when
the first version was run across all 208 genes:

 1. FAMILY-AWARE cluster choice. "Largest cluster" silently picks whichever
    domain type spans more residues, which is confidently wrong for the 40
    mixed-family genes (ZFHX3/ZEB1/ZEB2 C2H2+Homeodomain, TRPS1 C2H2+GATA,
    CREB5 C2H2+bZIP). We now prefer the cluster whose domain family matches
    the family the MOTIF DATABASE assigns to that gene (CIS-BP DBDs /
    HOCOMOCO tfclass) -- an independent curated source -- and only fall back
    to "largest" when the motif DB gives no usable family.

 2. AT-hook special case. AT-hooks are ~9-residue minor-groove motifs; taking
    the largest single cluster gave SETBP1 a 13-residue "DBD". For AT_hook-only
    genes we span ALL hooks (they function collectively), never one hook.

 3. Threshold set from BIOLOGY, not auto-calibration. Two data-driven attempts
    were tried and both rejected:
      - MAE of crop length vs deeppbs_v2 structural length: DEGENERATE. It
        falls monotonically as the threshold -> 1, because shorter is always
        "closer" to the structural target; at gap=1, 194/208 genes split into
        one cluster per fragment, which would shred genuine tandem arrays.
      - domain density: CONFOUNDED. InterPro returns overlapping family and
        superfamily annotations over the same residues, so density exceeds 1
        and is not comparable across genes.
    What the density/retention sweep does show is a real tradeoff: raising the
    threshold from 40 to 100 lifts fragment retention 0.79 -> 0.88 while
    tightness degrades only slowly. Absent a clean optimum we use the
    biological argument: the canonical C2H2 linker (TGEKP) is ~5-7 residues
    and loose arrays rarely exceed ~20-30, so GAP_THRESHOLD=40 keeps real
    arrays intact while splitting clearly separated clusters. This is a
    judgment call; crop_ambiguous records where it matters.
"""
import json
import numpy as np
import pandas as pd

FRAGS = "/tmp/domain_fragments_v2.jsonl"
GAP_THRESHOLD = 40
AMBIG_FRAC = 0.20
MIN_SENSIBLE_LEN = 30


def load_fragments(path=FRAGS):
    out = {}
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        if "error" in r or not r.get("frags"):
            continue
        out[r["gene"]] = r
    return out


def cluster(frags, gap):
    """frags: [[start,end,family],...] sorted -> [(start,end,n,families_set)]"""
    if not frags:
        return []
    cl = [[frags[0][0], frags[0][1], 1, {frags[0][2]}]]
    for a, b, fam in frags[1:]:
        if a - cl[-1][1] <= gap:
            cl[-1][1] = max(cl[-1][1], b); cl[-1][2] += 1; cl[-1][3].add(fam)
        else:
            cl.append([a, b, 1, {fam}])
    return [(c[0], c[1], c[2], c[3]) for c in cl]


def motif_db_family(genes):
    """Per-gene DBD family from the motif databases (independent of InterPro)."""
    out = {}
    cis = pd.read_csv("data/raw/cisbp_v3_10/TF_Information.txt", sep="\t", low_memory=False)
    cis = cis[cis["TF_Species"] == "Homo_sapiens"]
    for g, sub in cis.groupby(cis["TF_Name"].astype(str).str.upper()):
        d = str(sub.iloc[0]["DBDs"])
        if d and d.upper() != "UNKNOWN":
            out[g] = d
    h = pd.read_csv("data/raw/hocomoco_v14/tf_masterlist.tsv", sep="\t", low_memory=False)
    h = h[h["curated:uniprot_id"].astype(str).str.endswith("_HUMAN")]
    for g, sub in h.groupby(h["auto:gene_symbol"].astype(str).str.upper()):
        fam = str(sub.iloc[0]["tfclass:family"])
        if g not in out and fam and fam.lower() not in ("unannotated", "nan"):
            out[g] = fam
    return out


# map motif-DB family strings onto our internal family tokens
DB_TO_TOKEN = [
    ("zf-C2H2", "C2H2"), ("C2H2", "C2H2"), ("Krueppel", "C2H2"),
    ("Homeo", "Homeodomain"), ("HD-", "Homeodomain"), ("Prospero", "Prospero"),
    ("bZIP", "bZIP"), ("Leucine zipper", "bZIP"),
    ("bHLH", "bHLH"), ("Helix-loop-helix", "bHLH"),
    ("Forkhead", "Forkhead"), ("FOX", "Forkhead"),
    ("Ets", "ETS"), ("ETS", "ETS"),
    ("GATA", "GATA"), ("Nuclear receptor", "Nuclear_Receptor"), ("zf-C4", "Nuclear_Receptor"),
    ("HMG", "HMG_box"), ("Sox", "HMG_box"),
    ("T-box", "T_box"), ("Rel", "Rel_homology"), ("STAT", "STAT"),
    ("IRF", "IRF"), ("MADS", "MADS_box"), ("p53", "p53"), ("Runt", "Runt"),
    ("SMAD", "SMAD"), ("Paired", "Paired_domain"), ("TEA", "TEA"),
    ("ARID", "ARID"), ("AP-2", "AP2_TF"), ("HSF", "HSF"), ("BED", "BED_zf"),
    ("CG-1", "CG1"), ("CAMTA", "CG1"), ("Myb", "Myb_SANT"), ("SANT", "Myb_SANT"),
    ("MADF", "Myb_SANT_like"), ("C2HC", "C2H2C_type"), ("MYT", "C2H2C_type"),
    ("BEN", "BEN"), ("CXXC", "CXXC"), ("CCCH", "CCCH"), ("AT hook", "AT_hook"),
    ("AT_hook", "AT_hook"), ("FLYWCH", "FLYWCH"), ("GTF2I", "GTF2I"),
    ("Ndt80", "NDT80"), ("NDT80", "NDT80"), ("SAND", "SAND"), ("E2F", "E2F_DP"),
    ("DM ", "DM_domain"), ("DMRT", "DM_domain"), ("RFX", "RFX"),
    ("CP2", "CP2_GRHL"), ("CenpB", "CenpB_HTH"), ("THAP", "THAP"),
]


def db_family_tokens(s):
    """Return tokens in the order the motif DB lists them.

    CIS-BP's DBDs field can name MORE THAN ONE domain, e.g. ATF7 is
    'bZIP_1,zf-C2H2'. A first-match-wins scan over a fixed-order table picked
    C2H2 for ATF7 and cropped a 29aa zinc finger instead of its real 66aa
    bZIP domain -- so we preserve the DB's own ordering (primary DBD first)
    and let the caller try each in turn.
    """
    if not s: return []
    toks = []
    for part in str(s).split(","):
        for pat, tok in DB_TO_TOKEN:
            if pat.lower() in part.lower():
                if tok not in toks: toks.append(tok)
                break
    return toks


def db_family_token(s):
    t = db_family_tokens(s)
    return t[0] if t else None


def choose(clusters, preferred_family=None, target_len=None):
    """-> (start, end, ambiguous, reason)"""
    if len(clusters) == 1:
        c = clusters[0]
        return c[0], c[1], False, "single"

    pool = clusters
    reason = "largest"
    prefs = preferred_family if isinstance(preferred_family, list) else (
        [preferred_family] if preferred_family else [])
    for pf in prefs:                      # try DB-listed families in order
        matching = [c for c in clusters if pf in c[3]]
        if matching:
            pool, reason = matching, f"family:{pf}"
            break

    sizes = sorted(((c[1] - c[0], c) for c in pool), key=lambda x: -x[0])
    best_size, best = sizes[0]
    ambiguous = False
    if len(sizes) > 1:
        ambiguous = sizes[1][0] >= best_size * (1 - AMBIG_FRAC)
        if ambiguous and target_len:
            cand = [c for s, c in sizes if s >= best_size * (1 - AMBIG_FRAC)]
            best = min(cand, key=lambda c: abs((c[1] - c[0]) - target_len))
            reason += "+len_tiebreak"
    return best[0], best[1], ambiguous, reason


def family_targets():
    st = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2_deduped.parquet")
    main = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim.parquet")
    g2f = {}
    for _, r in main.iterrows():
        g2f.setdefault(str(r["gene_symbol"]).upper(), r["family_name"])
    st["family"] = st["gene"].str.upper().map(g2f)
    return st.groupby("family")["seq_length"].median().to_dict(), st["seq_length"].median()


def crop_all(data, gap, dbfam, targets, global_target, aug_fam):
    rows = []
    for gene, r in data.items():
        frags = [tuple(f) for f in r["frags"]]
        fams_present = {f[2] for f in frags}
        # refinement 2: AT-hooks act collectively; never crop to one 9aa hook
        if fams_present == {"AT_hook"}:
            s, e = frags[0][0], frags[-1][1]
            rows.append(dict(gene=gene, uniprot_id=r["uniprot_id"], n_clusters=1,
                             start=s, end=e, new_len=e - s,
                             old_span=frags[-1][1] - frags[0][0],
                             crop=r["seq"][s:e], families="AT_hook",
                             crop_ambiguous=False, reason="at_hook_span"))
            continue
        cl = cluster(frags, gap)
        pref = db_family_tokens(dbfam.get(gene.upper()))
        tgt = targets.get(aug_fam.get(gene.upper()), global_target)
        s, e, amb, why = choose(cl, pref, tgt)
        rows.append(dict(gene=gene, uniprot_id=r["uniprot_id"], n_clusters=len(cl),
                         start=s, end=e, new_len=e - s,
                         old_span=frags[-1][1] - frags[0][0],
                         crop=r["seq"][s:e], families="_".join(sorted(fams_present)),
                         crop_ambiguous=amb, reason=why))
    return pd.DataFrame(rows)


def main():
    data = load_fragments()
    dbfam = motif_db_family(list(data))
    targets, global_target = family_targets()
    aug = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
    aug_fam = dict(zip(aug["gene_symbol"].str.upper(), aug["family_name"]))
    print(f"genes: {len(data)}   motif-DB family available for: "
          f"{sum(1 for g in data if db_family_tokens(dbfam.get(g.upper())))}", flush=True)

    # threshold fixed from biology (see module docstring); the auto-calibration
    # attempts were degenerate/confounded and are deliberately NOT used.
    best = GAP_THRESHOLD
    print(f"\nusing GAP_THRESHOLD = {best} (biological, not auto-calibrated)")

    df = crop_all(data, best, dbfam, targets, global_target, aug_fam)
    df["gap_threshold"] = best
    df["saved"] = df["old_span"] - df["new_len"]
    print(f"\nmulti-cluster={int((df['n_clusters']>1).sum())}  "
          f"ambiguous={int(df['crop_ambiguous'].sum())}  "
          f"short(<30aa)={int((df['new_len']<MIN_SENSIBLE_LEN).sum())}")
    print("\nchoice reason breakdown:")
    print(df["reason"].str.replace(r"family:.*", "family-matched", regex=True).value_counts().to_string())
    print("\nbiggest reductions:")
    print(df.sort_values("saved", ascending=False)
            [["gene","n_clusters","old_span","new_len","saved","families","reason","crop_ambiguous"]]
            .head(15).to_string(index=False))
    df.to_parquet("/tmp/cluster_crops_v2.parquet")
    print("\nsaved /tmp/cluster_crops_v2.parquet")


if __name__ == "__main__":
    main()
