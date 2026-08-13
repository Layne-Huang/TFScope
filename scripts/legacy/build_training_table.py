#!/usr/bin/env python
"""Merge the final sequence-only + structure datasets into ONE training table
that the existing TFDataset/train.py can consume, and rebuild the 40%
component split on THIS final data (the old clu40_structtest split used
positional _rids from the pre-QC 6086-row data and no longer aligns after the
74-row PWM-quality/outlier drops -- reusing it would leak test into train).

Outputs:
  data/processed/tf_pwm_training_v22.parquet          (merged, TFDataset schema)
  data/processed/splits/train_v22/split.json          ({split: [filename,...]})
  data/processed/splits/train_v22/assignments.parquet (audit trail)
"""
import json, os, shutil, subprocess
import numpy as np
import pandas as pd
from reclassify_tf_families import classify

AUG = "data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet"
STR = (
    "data/processed/tf_pwm_deeppbs_v2_partner.parquet"
    if os.path.exists("data/processed/tf_pwm_deeppbs_v2_partner.parquet")
    else "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
)
OUT_PARQUET = "data/processed/tf_pwm_training_v22.parquet"
OUT_SPLIT = "data/processed/splits/train_v22"
WORK = "/tmp/train_v2_clu"
BASE_DATA = "data/processed/tf_pwm_training_v2.parquet"
BASE_SPLIT = "data/processed/splits/train_v2"
MIN_ID, COV, VAL_FRAC, SEED = 0.4, 0.8, 0.12, 42
TEST_TARGET = 200

# canonical 10-family scheme the residue-MoE embedding expects
FID = {"C2H2_short":0,"C2H2_medium":1,"C2H2_long":2,"bHLH":3,"Homeodomain":4,
       "bZIP":5,"Nuclear_Receptor":6,"Forkhead":7,"ETS":8,"Other":9}


def canon_family(name):
    """Map any fine-grained / composite family_name to a canonical id 0-9.
    Composite names (e.g. C2H2_Homeodomain) resolve on their FIRST component,
    which is the domain the crop was actually taken from (see cluster_crop)."""
    s = str(name)
    first = s.split("_")[0] if s not in FID else s
    for key in (s, first):
        if key in FID:
            return FID[key], key
    # bare "C2H2" / any C2H2 variant not sub-typed -> C2H2_long bucket
    if s.startswith("C2H2") or first == "C2H2":
        return FID["C2H2_long"], "C2H2_long"
    return FID["Other"], "Other"


def main():
    os.makedirs(OUT_SPLIT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    aug = pd.read_parquet(AUG); st = pd.read_parquet(STR)

    # gene -> canonical family, learned from aug (has family_name for every row)
    gene_fam = {}
    for g, sub in aug.groupby(aug["gene_symbol"].str.upper()):
        gene_fam[g] = sub["family_name"].mode().iat[0]
    c2h2_fam = {
        str(g).upper(): str(f)
        for g, f in zip(aug["gene_symbol"], aug["family_name"])
        if str(f) in {"C2H2_short", "C2H2_medium", "C2H2_long"}
    }

    # ---- normalise AUG rows ----
    a = pd.DataFrame()
    a["filename"] = ["seq_%d" % i for i in range(len(aug))]
    a["gene_symbol"] = aug["gene_symbol"].astype(str).values
    a["sequence"] = aug["sequence"].astype(str).values
    a["pwm"] = aug["pwm"].values
    a["motif_length"] = aug["motif_length"].values
    a["seq_length"] = aug["sequence"].astype(str).str.len().values
    a["dbd_start"] = 0
    a["dbd_end"] = a["seq_length"].values
    fam = aug["family_name"].map(lambda n: canon_family(n))
    a["family_id"] = [f[0] for f in fam]
    a["family_name"] = [f[1] for f in fam]
    a["family_source"] = "sequence_annotation"
    a["motif_source"] = aug.get("source", pd.Series(["unknown"] * len(aug))).astype(str).values
    a["partner_sequence"] = aug.get(
        "partner_sequence", pd.Series([""] * len(aug))
    ).fillna("").astype(str).values
    a["partner_gene"] = aug.get(
        "partner_gene", pd.Series([""] * len(aug))
    ).fillna("").astype(str).values
    a["is_dimer"] = aug.get(
        "dimer_required", pd.Series([False] * len(aug))
    ).fillna(False).astype(bool).values
    a["_set"] = "seq"

    # ---- normalise STRUCT rows (no family_id/dbd here) ----
    s = pd.DataFrame()
    s["filename"] = ["str_%d" % i for i in range(len(st))]
    s["gene_symbol"] = st["gene"].astype(str).values
    s["sequence"] = st["sequence"].astype(str).values
    s["pwm"] = st["pwm"].values
    s["motif_length"] = st["motif_length"].values
    s["seq_length"] = st["sequence"].astype(str).str.len().values
    s["dbd_start"] = 0
    s["dbd_end"] = s["seq_length"].values
    raw_family = []
    family_source = []
    for gene in st["gene"].astype(str).str.upper():
        if gene in gene_fam:
            raw_family.append(gene_fam[gene])
            family_source.append("sequence_gene_lookup")
        else:
            inferred = classify(gene, c2h2_fam)
            raw_family.append(inferred)
            family_source.append(
                "curated_gene_rule" if inferred != "Other" else "fallback_other"
            )
    famc = pd.Series(raw_family).map(canon_family)
    s["family_id"] = [f[0] for f in famc]
    s["family_name"] = [f[1] for f in famc]
    s["family_source"] = family_source
    s["motif_source"] = st.get(
        "pwm_source", pd.Series(["unknown"] * len(st))
    ).fillna("unknown").astype(str).values
    s["partner_sequence"] = st.get(
        "partner_sequence", pd.Series([""] * len(st))
    ).fillna("").astype(str).values
    s["partner_gene"] = st.get(
        "partner_gene_used", pd.Series([""] * len(st))
    ).fillna("").astype(str).values
    s["is_dimer"] = st.get(
        "is_dimer", pd.Series([False] * len(st))
    ).fillna(False).astype(bool).values
    s["_set"] = "struct"

    merged = pd.concat([a, s], ignore_index=True)
    merged["gene_key"] = merged["gene_symbol"].str.strip().str.upper()
    merged["group_id"] = (
        merged["gene_key"] + "|" + merged["sequence"].astype(str)
        + "|" + merged["motif_source"].astype(str).str.upper()
    )
    repeat_families = {"bHLH", "bZIP", "Nuclear_Receptor"}
    merged["multichain_eligible"] = (
        merged["is_dimer"].astype(bool)
        & (
            merged["family_name"].isin(repeat_families)
            | merged["gene_key"].str.match(r"^(TP53|TP63|TP73|P53)$")
        )
        & merged["partner_sequence"].astype(str).str.len().gt(0)
    )
    merged.to_parquet(OUT_PARQUET)
    print(f"merged table: {len(merged)} rows ({len(a)} seq + {len(s)} struct), "
          f"{merged['gene_symbol'].str.upper().nunique()} genes")
    print("family_id distribution:", merged["family_id"].value_counts().sort_index().to_dict())

    # Metadata repair does not alter filename/sequence identity. If MMseqs2 is
    # unavailable, safely reuse the already leakage-audited v2 partition after
    # proving row identity instead of silently creating an unclustered split.
    if shutil.which("mmseqs") is None:
        if not (os.path.exists(BASE_DATA) and os.path.exists(f"{BASE_SPLIT}/split.json")):
            raise FileNotFoundError(
                "mmseqs is unavailable and no compatible base split exists"
            )
        base = pd.read_parquet(BASE_DATA)
        identity = (
            len(base) == len(merged)
            and base["filename"].astype(str).tolist()
            == merged["filename"].astype(str).tolist()
            and base["sequence"].astype(str).tolist()
            == merged["sequence"].astype(str).tolist()
        )
        if not identity:
            raise RuntimeError(
                "Cannot reuse train_v2 split: filename/sequence identity changed"
            )
        with open(f"{BASE_SPLIT}/split.json") as handle:
            split_data = json.load(handle)
        with open(f"{OUT_SPLIT}/split.json", "w") as handle:
            json.dump(split_data, handle)
        split_of = {
            filename: split
            for split, filenames in split_data.items()
            for filename in filenames
        }
        audit = merged[
            ["filename", "_set", "group_id", "family_name", "family_source",
             "motif_source", "multichain_eligible"]
        ].copy()
        audit["split"] = audit["filename"].map(split_of)
        base_assignments = f"{BASE_SPLIT}/assignments.parquet"
        if os.path.exists(base_assignments):
            old = pd.read_parquet(base_assignments)
            keep = [c for c in ["filename", "_g", "_c", "_comp"] if c in old]
            audit = audit.merge(old[keep], on="filename", how="left")
        audit.to_parquet(f"{OUT_SPLIT}/assignments.parquet")
        print(f"mmseqs unavailable; reused verified split from {BASE_SPLIT}")
        print(f"saved {OUT_PARQUET} and {OUT_SPLIT}/")
        return

    # ---- rebuild 40% component split on THIS data ----
    fa = os.path.join(WORK, "seqs.fasta")
    with open(fa, "w") as f:
        for fn, seq in zip(merged["filename"], merged["sequence"]):
            f.write(f">{fn}\n{seq}\n")
    print(f"clustering {len(merged)} sequences at {int(MIN_ID*100)}% ...", flush=True)
    subprocess.run(["mmseqs", "easy-cluster", fa, os.path.join(WORK, "clu"),
                    os.path.join(WORK, "tmp"), "--min-seq-id", str(MIN_ID),
                    "-c", str(COV), "--cov-mode", "1", "-v", "1"],
                   check=True, capture_output=True)
    clu = pd.read_csv(os.path.join(WORK, "clu_cluster.tsv"), sep="\t", names=["rep", "member"])
    merged["_c"] = merged["filename"].map(dict(zip(clu["member"], clu["rep"]))).fillna(merged["filename"])
    merged["_g"] = merged["gene_symbol"].str.upper()

    # connected components of gene<->cluster graph (gene+cluster disjointness)
    from collections import defaultdict, deque
    adj = defaultdict(set)
    for g, c in zip(merged["_g"], merged["_c"]):
        adj[("g", g)].add(("c", c)); adj[("c", c)].add(("g", g))
    comp_of, seen, cid = {}, set(), 0
    for node in adj:
        if node in seen: continue
        stack, comp = [node], []
        seen.add(node)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        for u in comp: comp_of[u] = cid
        cid += 1
    merged["_comp"] = merged["_g"].map(lambda g: comp_of[("g", g)])

    # pick test components: family-diverse, cheapest collateral first
    stats = (merged.groupby("_comp")
             .agg(n=("filename", "size"), n_str=("_set", lambda x: (x == "struct").sum()),
                  fam=("family_name", lambda x: x.mode().iat[0])).query("n_str>0"))
    stats["coll"] = stats["n"] - stats["n_str"]; stats["eff"] = stats["n_str"] / stats["n"]
    rng = np.random.default_rng(SEED)
    chosen, ntest = [], 0
    for fam, grp in stats.sort_values("eff", ascending=False).groupby("fam"):
        b = grp.sort_values(["eff", "n_str"], ascending=[False, False]).index[0]
        if stats.loc[b, "n_str"] >= 3: chosen.append(b); ntest += int(stats.loc[b, "n_str"])
    for c in stats.sort_values(["eff", "n_str"], ascending=[False, False]).index:
        if ntest >= TEST_TARGET: break
        if c in chosen or stats.loc[c, "coll"] > 3 * stats.loc[c, "n_str"]: continue
        chosen.append(c); ntest += int(stats.loc[c, "n_str"])
    test_comp = set(chosen)

    merged["split"] = "train"
    inc = merged["_comp"].isin(test_comp)
    merged.loc[inc & (merged._set == "struct"), "split"] = "test"
    merged.loc[inc & (merged._set != "struct"), "split"] = "excluded"
    rest = merged[merged.split == "train"]
    cl = rest.groupby("_comp").size(); order = rng.permutation(cl.index.to_numpy())
    tgt, acc, valc = int(len(rest) * VAL_FRAC), 0, []
    for c in order:
        if acc >= tgt: break
        valc.append(c); acc += int(cl[c])
    merged.loc[merged["_comp"].isin(valc), "split"] = "val"

    print("\n=== split ===")
    print(merged.groupby(["split", "_set"]).size().unstack(fill_value=0).to_string())
    print("\n=== leakage (must be 0) ===")
    for x, y in [("train", "test"), ("train", "val"), ("val", "test")]:
        cx, cy = set(merged[merged.split == x]["_comp"]), set(merged[merged.split == y]["_comp"])
        gx, gy = set(merged[merged.split == x]["_g"]), set(merged[merged.split == y]["_g"])
        print(f"  {x}/{y}: shared_comp={len(cx & cy)} shared_gene={len(gx & gy)}")

    json.dump({sp: merged[merged.split == sp]["filename"].tolist()
               for sp in ["train", "val", "test", "excluded"]},
              open(f"{OUT_SPLIT}/split.json", "w"))
    merged[["filename", "_set", "_g", "_c", "_comp", "group_id",
            "family_name", "family_source", "motif_source",
            "multichain_eligible", "split"]].to_parquet(
        f"{OUT_SPLIT}/assignments.parquet")
    print(f"\nsaved {OUT_PARQUET} and {OUT_SPLIT}/")


if __name__ == "__main__":
    main()
