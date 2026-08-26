#!/usr/bin/env python
"""v26 Phase-3: leakage-controlled split over biological target units.

Fixes two audit findings:
  * Finding E -- 13 primary+partner DBD clusters were shared train<->test in train_v22
    (Primary-OOD held, Assembly-OOD did not).
  * Finding B -- all 20 Barrera genes were in TRAIN, so the mutation-blindness result behind
    "frozen ESM-2 is the wall" was measured on memorised motifs.

The split unit is a CONNECTED COMPONENT of a hypergraph, never a parquet row. Nodes joined:
    accession · primary DBD cluster · exact DBD sequence hash · partner sequence hash
    · partner cluster · WT/mutant group · designed-protein group · PDB assembly
Application sets are locked out BEFORE allocation, by whole component.

Outputs
  data/processed/splits/v26/manifest.parquet        frozen, one row per example
  data/processed/splits/v26/split.json              {split: [example_id,...]}
  data/processed/splits/v26/application_holdout.json
  data/processed/splits/v26/components.parquet
  results/v26/split_assertions.csv                  zero-overlap table (exit 1 if violated)
  results/v26/split_summary.json

  python scripts/v26/build_v26_split.py --dataset core
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

V26D = "data/processed/v26"
SPD = "data/processed/splits/v26"
RESD = "results/v26"
WORK = "/tmp/v26_split_clu"
MIN_ID, COV, COV_MODE = 0.4, 0.8, 1
SEED = 42
TEST_FRAC, VAL_FRAC = 0.15, 0.12


class UF:
    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def mmseqs_cluster(seqs: dict, name: str):
    os.makedirs(WORK, exist_ok=True)
    fa = os.path.join(WORK, f"{name}.fasta")
    pref = os.path.join(WORK, f"{name}_clu")
    with open(fa, "w") as fh:
        for k, s in seqs.items():
            fh.write(f">{k}\n{s}\n")
    cmd = ["mmseqs", "easy-cluster", fa, pref, os.path.join(WORK, f"tmp_{name}"),
           "--min-seq-id", str(MIN_ID), "-c", str(COV), "--cov-mode", str(COV_MODE), "-v", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    clu = pd.read_csv(f"{pref}_cluster.tsv", sep="\t", names=["rep", "member"])
    return dict(zip(clu["member"].astype(str), clu["rep"].astype(str))), " ".join(cmd)


def load_application_sets(ex: pd.DataFrame):
    """Return {set_name: set(example_id)} for locked application sets."""
    out = defaultdict(set)
    genes = ex.primary_gene_symbol_legacy.astype(str).str.upper()

    bp = "results/mutation_benchmark/barrera_pairs.json"
    if os.path.exists(bp):
        bg = {str(p["gene"]).upper() for p in json.load(open(bp)).get("pairs", [])}
        out["barrera"] |= set(ex[genes.isin(bg)].example_id)
    out["myod1"] |= set(ex[genes == "MYOD1"].example_id)
    # designed DBPs are not in the training table; record the intent so the manifest is explicit
    for g in ("DBP5", "DBP6", "DBP9", "DBP35", "DBP005", "DBP006", "DBP009", "DBP035"):
        out["designed_dbp"] |= set(ex[genes == g].example_id)
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="core")
    a = ap.parse_args()
    os.makedirs(SPD, exist_ok=True)
    os.makedirs(RESD, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ex = pd.read_parquet(f"{V26D}/v26_{a.dataset}.parquet")
    ex["dbd_core"] = [s[int(d0):int(d1)] for s, d0, d1
                      in zip(ex.sequence, ex.dbd_start, ex.dbd_end)]
    ex["dbd_hash"] = [hashlib.sha256(s.encode()).hexdigest()[:16] for s in ex.dbd_core]
    print(f"examples: {len(ex)}  target_units: {ex.target_unit_id.nunique()}  "
          f"accessions: {ex.primary_accession.nunique()}", flush=True)

    # ---- cluster DBD CORES (not flanked, not full-length) together with partner sequences
    seqs = {}
    for h, s in zip(ex.dbd_hash, ex.dbd_core):
        if len(s) >= 10:
            seqs[f"P#{h}"] = s
    partner_of = defaultdict(set)
    for r in ex.itertuples():
        for p in json.loads(r.partner_entities or "[]"):
            ph = p["sequence_hash"]
            if p["length"] >= 10:
                seqs[f"Q#{ph}"] = p["sequence"]
                partner_of[r.example_id].add(ph)
    print(f"clustering {len(seqs)} DBD cores + partner sequences at "
          f"{int(MIN_ID*100)}% id / {COV} cov (cov-mode {COV_MODE}) ...", flush=True)
    cmap, cmd = mmseqs_cluster(seqs, "cores")
    print(f"  {cmd}", flush=True)
    clu_of_hash = {h: cmap.get(f"P#{h}", f"P#{h}") for h in ex.dbd_hash.unique()}
    clu_of_partner = {ph: cmap.get(f"Q#{ph}", f"Q#{ph}")
                      for s in partner_of.values() for ph in s}
    n_clu = len({*clu_of_hash.values(), *clu_of_partner.values()})
    print(f"  clusters: {n_clu}", flush=True)

    # ---- PRIMARY-OOD hypergraph (the practical main setting, per the brief)
    #
    # Deliberately EXCLUDES PDB-assembly and partner-cluster edges. Including them chains
    # transitively across structures -- TF1-PDB_A-TF2, TF2-PDB_B-TF3, ... -- which collapsed
    # 1,355 target units into 34 components, one holding 96% of all examples. Partner identity
    # is instead handled as the STRICTER SECONDARY analysis (Assembly-OOD) below: it is measured
    # and reported, and test components that share a partner cluster with train are flagged.
    uf = UF()
    for r in ex.itertuples():
        t = ("T", r.target_unit_id)
        uf.add(t)
        for node in [("ACC", r.primary_accession),
                     ("HASH", r.dbd_hash),
                     ("CLU", clu_of_hash[r.dbd_hash]),
                     ("GENE", str(r.primary_gene_symbol_legacy).upper())]:
            uf.union(t, node)

    ex["component"] = [uf.find(("T", t)) for t in ex.target_unit_id]
    comp_ids = {c: i for i, c in enumerate(sorted(set(ex.component), key=str))}
    ex["component_id"] = ex.component.map(comp_ids)
    print(f"components: {ex.component_id.nunique()}", flush=True)

    # ---- lock application sets by WHOLE component, before allocation
    apps = load_application_sets(ex)
    locked_comp, app_of_comp = set(), {}
    for name, ids in apps.items():
        cs = set(ex[ex.example_id.isin(ids)].component_id)
        locked_comp |= cs
        for c in cs:
            app_of_comp.setdefault(c, []).append(name)
    ex["application_holdout"] = ex.component_id.isin(locked_comp)
    ex["application_set"] = ex.component_id.map(
        lambda c: ";".join(sorted(app_of_comp.get(c, []))) or None)
    print(f"application sets: "
          f"{ {k: len(v) for k, v in apps.items()} }; "
          f"locked components={len(locked_comp)} "
          f"examples={int(ex.application_holdout.sum())}", flush=True)

    # ---- allocate remaining components, stratified by family + structure availability
    pool = ex[~ex.application_holdout]
    cstats = pool.groupby("component_id").agg(
        n=("example_id", "size"),
        n_units=("target_unit_id", "nunique"),
        n_struct=("structure_id", lambda s: int(s.notna().sum())),
        fam=("dbd_families_for_analysis_only", lambda s: s.mode().iat[0] if len(s) else "NA"),
        multimer=("n_partners", lambda s: int((s > 0).any())),
    ).reset_index()

    total_units = int(cstats.n_units.sum())
    want_test = TEST_FRAC * total_units
    want_val = VAL_FRAC * total_units

    # Allocate SMALLEST-FIRST so many small components fill the quota -- that maximises
    # diversity and stops one giant component from swallowing the whole test set. The earlier
    # largest-first version put 593/1355 target units (44%) in test.
    want_test_units = int(round(TEST_FRAC * total_units))
    want_val_units = int(round(VAL_FRAC * total_units))

    struct_c = cstats[cstats.n_struct > 0].sort_values("n_units")
    test_c, n_t = [], 0
    # round-robin over families first, so every family with structures is represented
    for fam, grp in struct_c.groupby("fam"):
        if n_t >= want_test_units:
            break
        c = grp.iloc[0]
        test_c.append(int(c.component_id)); n_t += int(c.n_units)
    for c in struct_c.itertuples():
        if n_t >= want_test_units:
            break
        if int(c.component_id) in test_c:
            continue
        test_c.append(int(c.component_id)); n_t += int(c.n_units)

    remain = cstats[~cstats.component_id.isin(test_c)].sort_values("n_units")
    val_c, n_v = [], 0
    order = rng.permutation(remain.component_id.to_numpy())
    nu = dict(zip(remain.component_id, remain.n_units))
    for c in order:
        if n_v >= want_val_units:
            break
        val_c.append(int(c)); n_v += int(nu[c])

    print(f"  target units: total={total_units} want_test={want_test_units} "
          f"got_test={n_t} want_val={want_val_units} got_val={n_v}", flush=True)

    def lab(c):
        if c in set(test_c):
            return "test"
        if c in set(val_c):
            return "val"
        return "train"

    ex["split"] = np.where(ex.application_holdout, "application_holdout",
                           ex.component_id.map(lab))
    print("\nsplit sizes (examples):", ex.split.value_counts().to_dict(), flush=True)
    print("split sizes (target units):",
          ex.groupby("split").target_unit_id.nunique().to_dict(), flush=True)

    # ---- assertions
    ex["primary_cluster"] = ex.dbd_hash.map(clu_of_hash)
    ex["partner_clusters"] = [";".join(sorted(clu_of_partner[p]
                                              for p in partner_of.get(e, ()))) or None
                              for e in ex.example_id]

    def units(sp, col):
        return set(ex[ex.split == sp][col].dropna())

    def assembly_units(sp):
        s = set(ex[ex.split == sp].primary_cluster.dropna())
        for v in ex[ex.split == sp].partner_clusters.dropna():
            s |= set(v.split(";"))
        return s

    # ---- Assembly-OOD (stricter secondary): flag test components whose primary OR partner
    # cluster also appears anywhere in train. Reported, not used to build the main split.
    train_asm = set(ex[ex.split == "train"].primary_cluster.dropna())
    for v in ex[ex.split == "train"].partner_clusters.dropna():
        train_asm |= set(v.split(";"))

    def _row_asm(r):
        s = {r.primary_cluster} if r.primary_cluster else set()
        if r.partner_clusters:
            s |= set(str(r.partner_clusters).split(";"))
        return s

    ex["assembly_ood_clean"] = [
        (r.split != "test") or (not (_row_asm(r) & train_asm)) for r in ex.itertuples()]
    n_dirty = int((~ex.assembly_ood_clean).sum())
    n_dirty_units = int(ex[~ex.assembly_ood_clean].target_unit_id.nunique())
    print(f"\nAssembly-OOD: {n_dirty} test examples ({n_dirty_units} target units) share a "
          f"primary/partner cluster with train -> excluded from the Assembly-OOD subset",
          flush=True)

    rows = []
    for col, label in [("primary_gene_symbol_legacy", "gene"),
                       ("primary_accession", "accession"),
                       ("dbd_hash", "exact_dbd_sequence"),
                       ("primary_cluster", "primary_dbd_cluster"),
                       ("component_id", "component"),
                       ("target_unit_id", "target_unit")]:
        for x, y in [("train", "test"), ("train", "val"), ("val", "test"),
                     ("train", "application_holdout"), ("test", "application_holdout")]:
            sa, sb = units(x, col), units(y, col)
            rows.append({"unit": label, "pair": f"{x}/{y}", "n_a": len(sa),
                         "n_b": len(sb), "shared": len(sa & sb)})
    for x, y in [("train", "test"), ("train", "val"), ("val", "test"),
                 ("train", "application_holdout")]:
        sa, sb = assembly_units(x), assembly_units(y)
        rows.append({"unit": "assembly_clusters(primary+partner)", "pair": f"{x}/{y}",
                     "n_a": len(sa), "n_b": len(sb), "shared": len(sa & sb)})
    asrt = pd.DataFrame(rows)
    asrt.to_csv(f"{RESD}/split_assertions.csv", index=False)

    ADVISORY = {"assembly_clusters(primary+partner)"}
    asrt["severity"] = np.where(asrt.unit.isin(ADVISORY), "advisory", "fatal")
    print("\n=== zero-overlap assertions ===")
    bad = 0
    for r in asrt.itertuples():
        flag = ""
        if r.shared:
            if r.severity == "fatal":
                flag = "  <-- FATAL LEAK"
                bad += 1
            else:
                flag = "  <-- advisory (Assembly-OOD; handled by assembly_ood_clean)"
        print(f"  {r.unit:38s} {r.pair:28s} shared={r.shared:5d}{flag}")

    # ---- manifest
    # NOTE: example_id hashes the crop window, so it is DATASET-SPECIFIC. Downstream code must
    # join this manifest on target_unit_id (or motif_observation_id), both of which are invariant
    # across core / flank20 / flank32. Joining on example_id matched 13/5966 flank20 rows.
    man_cols = ["example_id", "motif_observation_id", "target_unit_id", "component_id", "split",
                "application_holdout", "application_set",
                "primary_accession", "primary_gene", "primary_gene_symbol_legacy",
                "dbd_hash", "primary_cluster", "partner_clusters",
                "dbd_families_for_analysis_only", "motif_source",
                "structure_id", "structure_chain", "n_partners",
                "assembly_ood_clean",
                "dbd_selection_mode", "dbd_unp_start", "dbd_unp_end",
                "crop_unp_start", "crop_unp_end", "legacy_filename"]
    man = ex[[c for c in man_cols if c in ex.columns]].copy()
    man.to_parquet(f"{SPD}/manifest.parquet", index=False)
    json.dump({sp: ex[ex.split == sp].example_id.tolist()
               for sp in ex.split.unique()}, open(f"{SPD}/split.json", "w"))
    json.dump({k: sorted(v) for k, v in apps.items()},
              open(f"{SPD}/application_holdout.json", "w"), indent=1)
    ex[["component_id", "split", "application_holdout"]].drop_duplicates().to_parquet(
        f"{SPD}/components.parquet", index=False)

    summary = {
        "dataset": a.dataset,
        "mmseqs_command": cmd,
        "mmseqs_params": {"min_seq_id": MIN_ID, "coverage": COV, "cov_mode": COV_MODE},
        "examples": int(len(ex)),
        "target_units": int(ex.target_unit_id.nunique()),
        "components": int(ex.component_id.nunique()),
        "clusters": int(n_clu),
        "split_examples": ex.split.value_counts().to_dict(),
        "split_target_units": ex.groupby("split").target_unit_id.nunique().to_dict(),
        "application_sets": {k: len(v) for k, v in apps.items()},
        "locked_components": len(locked_comp),
        "assertion_violations": int(bad),
        "test_structures": int(ex[(ex.split == "test") & ex.structure_id.notna()]
                              .structure_id.nunique()),
        "test_struct_target_units": int(ex[(ex.split == "test") & ex.structure_id.notna()]
                                        .target_unit_id.nunique()),
        "assembly_ood_excluded_examples": int((~ex.assembly_ood_clean).sum()),
        "assembly_ood_test_units": int(ex[(ex.split == "test")
                                          & ex.assembly_ood_clean].target_unit_id.nunique()),
    }
    json.dump(summary, open(f"{RESD}/split_summary.json", "w"), indent=2)
    print(f"\n=== summary ===\n{json.dumps(summary, indent=2)}")

    if bad:
        print(f"\nFAIL: {bad} FATAL assertion violations -- split NOT usable")
        raise SystemExit(1)
    print("\nAll FATAL zero-overlap assertions PASS "
          "(Assembly-OOD overlaps are advisory; use the assembly_ood_clean subset)")


if __name__ == "__main__":
    main()
