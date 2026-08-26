#!/usr/bin/env python
"""v26 Phase-0 audit: READ-ONLY inventory of the v23/v24/v25 data pipeline.

Produces the quantitative tables behind docs/v26_audit.md. Touches nothing under
data/, checkpoints/ or results/ except its own output directory.

  OUT = results/v26_audit/

Sections
  1. dataset inventory ......... rows/genes/seqs/DBD stats for v23, v25flank, v25xtal
  2. index validity ............ recognition + contact indices vs each version's crop
  3. split overlap ............. gene / exact-seq / group / mmseqs-cluster / partner
  4. contact-defined crops ..... str_ rows whose input boundary came from DNA contacts
  5. application-set leakage ... Barrera, MyoD1, designed DBPs vs train split
  6. multi-domain crops ........ genes re-cropped using motif-DB family knowledge

Run (repo root, tfscope env):
  python scripts/v26/audit_phase0.py
  python scripts/v26/audit_phase0.py --skip-mmseqs      # reuse cached cluster tsv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = "results/v26_audit"
WORK = "/tmp/v26_audit_clu"

V23 = "data/processed/tf_pwm_training_v23.parquet"
V25F = "data/processed/tf_pwm_training_v25flank.parquet"
V25X = "data/processed/tf_pwm_training_v25xtal.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
ASSIGN = "data/processed/splits/train_v22/assignments.parquet"

VERSIONS = [
    ("v23", V23, "data/contact_maps/recognition_residues_v23.json",
     "data/contact_maps/contact_targets_v23.json"),
    ("v25flank", V25F, "data/contact_maps/recognition_residues_v25flank.json",
     "data/contact_maps/contact_targets_v25flank.json"),
    ("v25xtal", V25X, "data/contact_maps/recognition_residues_v25xtal.json",
     "data/contact_maps/contact_targets_v25xtal.json"),
]

MAX_MOTIF_LENGTH = 42          # config.max_motif_length; contact cols >= this are dropped
MAX_SEQ_LEN = 1024             # TFDataset truncation at eval time


# --------------------------------------------------------------------------- utils
def _load(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _primary(entry):
    """Recognition-prior entries are either a bare list or {'primary': [...], 'partner': [...]}."""
    if entry is None:
        return []
    return entry.get("primary", []) if isinstance(entry, dict) else list(entry)


# ------------------------------------------------------------------- 1. inventory
def section_inventory():
    rows = []
    for tag, path, _, _ in VERSIONS:
        d = _load(path)
        if d is None:
            rows.append({"version": tag, "status": "MISSING", "path": path})
            continue
        L = d.sequence.str.len()
        span = d.dbd_end - d.dbd_start
        is_str = d.filename.astype(str).str.startswith("str_")
        rows.append({
            "version": tag, "status": "ok", "path": path,
            "rows": len(d),
            "genes": int(d.gene_symbol.astype(str).str.upper().nunique()),
            "unique_sequences": int(d.sequence.nunique()),
            "seq_rows": int((~is_str).sum()), "str_rows": int(is_str.sum()),
            "seqlen_median": int(L.median()), "seqlen_max": int(L.max()),
            "seqlen_over_1024": int((L > MAX_SEQ_LEN).sum()),
            "dbd_span_median": int(span.median()),
            "dbd_start_gt0_frac": round(float((d.dbd_start > 0).mean()), 4),
            "dbd_fraction_of_input": round(float((span / L).mean()), 4),
            "flank_source": (d.flank_source.value_counts().to_dict()
                             if "flank_source" in d.columns else None),
        })
    return rows


# --------------------------------------------------------------- 2. index validity
def section_index_validity():
    """Classify every recognition/contact index against the crop it will be applied to.

    Mirrors TFDataset.__getitem__ exactly:
      recog  : kept iff 0 <= p < len(sequence_tokens)          (dataset.py:630-633)
      contact: column kept iff 0 <= c < max_motif_length       (dataset.py:645-647)
               residue kept iff 0 <= ridx < len(seq)           (dataset.py:648-650)
    Anything outside is dropped with no warning -> "silently_clipped".
    """
    recog_rows, contact_rows, detail = [], [], []
    for tag, path, recog_p, con_p in VERSIONS:
        d = _load(path)
        if d is None or not os.path.exists(recog_p):
            continue
        meta = {r.filename: (int(r.dbd_start), int(r.dbd_end), len(r.sequence))
                for r in d.itertuples()}

        rec = json.load(open(recog_p))
        n_entries = n_matched = 0
        n_idx = n_in_dbd = n_in_seq_outside_dbd = n_dropped = 0
        for fn, entry in rec.items():
            n_entries += 1
            if fn not in meta:
                continue
            n_matched += 1
            ds, de, L = meta[fn]
            for p in _primary(entry):
                n_idx += 1
                if 0 <= p < L:
                    if ds <= p < de:
                        n_in_dbd += 1
                    else:
                        n_in_seq_outside_dbd += 1
                        detail.append({"version": tag, "kind": "recog_outside_dbd",
                                       "filename": fn, "index": int(p),
                                       "dbd_start": ds, "dbd_end": de, "seq_len": L})
                else:
                    n_dropped += 1
                    detail.append({"version": tag, "kind": "recog_silently_clipped",
                                   "filename": fn, "index": int(p),
                                   "dbd_start": ds, "dbd_end": de, "seq_len": L})
        recog_rows.append({
            "version": tag, "json": recog_p, "entries": n_entries,
            "entries_matched_to_row": n_matched, "indices_total": n_idx,
            "inside_dbd": n_in_dbd, "in_seq_outside_dbd": n_in_seq_outside_dbd,
            "silently_clipped": n_dropped,
        })

        if not os.path.exists(con_p):
            continue
        con = json.load(open(con_p))
        c_entries = c_matched = 0
        col_total = col_dropped = 0
        res_total = res_dropped = res_outside_dbd = 0
        cols_emptied = 0
        for fn, entry in con.items():
            c_entries += 1
            if fn not in meta:
                continue
            c_matched += 1
            ds, de, L = meta[fn]
            for col, rlist in entry.get("cols", {}).items():
                col_total += 1
                c = int(col)
                if not (0 <= c < MAX_MOTIF_LENGTH):
                    col_dropped += 1
                    detail.append({"version": tag, "kind": "contact_col_dropped",
                                   "filename": fn, "index": c,
                                   "dbd_start": ds, "dbd_end": de, "seq_len": L})
                    continue
                kept = 0
                for ridx, _w in rlist:
                    res_total += 1
                    if 0 <= ridx < L:
                        kept += 1
                        if not (ds <= ridx < de):
                            res_outside_dbd += 1
                    else:
                        res_dropped += 1
                        detail.append({"version": tag, "kind": "contact_res_clipped",
                                       "filename": fn, "index": int(ridx),
                                       "dbd_start": ds, "dbd_end": de, "seq_len": L})
                if rlist and kept == 0:
                    cols_emptied += 1
        contact_rows.append({
            "version": tag, "json": con_p, "entries": c_entries,
            "entries_matched_to_row": c_matched,
            "columns_total": col_total, "columns_dropped_out_of_range": col_dropped,
            "residue_links_total": res_total,
            "residue_links_silently_clipped": res_dropped,
            "residue_links_outside_dbd": res_outside_dbd,
            "columns_emptied_by_clipping": cols_emptied,
        })
    return recog_rows, contact_rows, detail


# ------------------------------------------------------------------ 3. split audit
def _mmseqs_cluster(seqs: dict, name: str, min_id=0.4, cov=0.8, cov_mode=1, skip=False):
    """seqs: {id: sequence} -> {id: cluster_rep}. Returns (mapping, command_string)."""
    os.makedirs(WORK, exist_ok=True)
    fa = os.path.join(WORK, f"{name}.fasta")
    pref = os.path.join(WORK, f"{name}_clu")
    tsv = f"{pref}_cluster.tsv"
    cmd = ["mmseqs", "easy-cluster", fa, pref, os.path.join(WORK, f"tmp_{name}"),
           "--min-seq-id", str(min_id), "-c", str(cov), "--cov-mode", str(cov_mode), "-v", "1"]
    if not (skip and os.path.exists(tsv)):
        with open(fa, "w") as fh:
            for k, s in seqs.items():
                fh.write(f">{k}\n{s}\n")
        subprocess.run(cmd, check=True, capture_output=True)
    clu = pd.read_csv(tsv, sep="\t", names=["rep", "member"])
    return dict(zip(clu["member"].astype(str), clu["rep"].astype(str))), " ".join(cmd)


def section_split(skip_mmseqs=False):
    d = _load(V23)
    split = json.load(open(SPLIT))
    split_of = {fn: sp for sp, fns in split.items() for fn in fns}
    d = d.copy()
    d["split"] = d.filename.map(split_of)
    d["gene"] = d.gene_symbol.astype(str).str.upper()

    assign = _load(ASSIGN)
    comp_of = dict(zip(assign.filename, assign._comp)) if assign is not None else {}
    clus_of = dict(zip(assign.filename, assign._c)) if assign is not None else {}
    d["_comp"] = d.filename.map(comp_of)
    d["_c"] = d.filename.map(clus_of)

    # partner sequences -> one cluster space, shared with primaries
    partner_seqs = {}
    for r in d.itertuples():
        ps = r.partner_seqs
        if ps is None:
            continue
        for j, p in enumerate(list(ps)):
            p = str(p)
            if len(p) >= 10:
                partner_seqs[f"{r.filename}#p{j}"] = p
    all_seqs = {f"P#{r.filename}": str(r.sequence) for r in d.itertuples()}
    all_seqs.update({f"Q#{k}": v for k, v in partner_seqs.items()})
    cmap, cmd = _mmseqs_cluster(all_seqs, "primary_partner", skip=skip_mmseqs)

    prim_clu = {r.filename: cmap.get(f"P#{r.filename}") for r in d.itertuples()}
    part_clu = defaultdict(set)
    for k in partner_seqs:
        fn = k.split("#p")[0]
        c = cmap.get(f"Q#{k}")
        if c:
            part_clu[fn].add(c)
    d["_pc"] = d.filename.map(prim_clu)

    SPL = ["train", "val", "test"]
    sets = {sp: d[d.split == sp] for sp in SPL}

    def sets_of(col, sp):
        return set(sets[sp][col].dropna().tolist())

    overlaps = []
    for col, label in [("gene", "gene_symbol"), ("sequence", "exact_sequence"),
                       ("group_id", "group_id"), ("_c", "legacy_mmseqs_cluster(_c)"),
                       ("_pc", "primary_dbd_cluster(recomputed)"), ("_comp", "component(_comp)")]:
        for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
            sa, sb = sets_of(col, a), sets_of(col, b)
            overlaps.append({"unit": label, "pair": f"{a}/{b}",
                             "n_a": len(sa), "n_b": len(sb), "shared": len(sa & sb)})

    # Assembly-OOD: primary cluster of one split vs ALL clusters (primary+partner) of the other
    def all_clu(sp):
        out = set(sets[sp]["_pc"].dropna())
        for fn in sets[sp].filename:
            out |= part_clu.get(fn, set())
        return out

    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        sa, sb = all_clu(a), all_clu(b)
        overlaps.append({"unit": "assembly_clusters(primary+partner)", "pair": f"{a}/{b}",
                         "n_a": len(sa), "n_b": len(sb), "shared": len(sa & sb)})

    # components crossing train/val/test (ignore the by-design 'excluded' label)
    core = d[d.split.isin(SPL)]
    crossing = (core.groupby("_comp").split.nunique() > 1)
    cross_ids = sorted(int(c) for c in crossing[crossing].index)

    comp_sizes = core.groupby("_comp").size().sort_values(ascending=False)
    return {
        "mmseqs_command": cmd,
        "overlaps": overlaps,
        "components_crossing_train_val_test": cross_ids,
        "n_components": int(core._comp.nunique()),
        "largest_components": {int(k): int(v) for k, v in comp_sizes.head(10).items()},
        "largest_component_genes": int(core[core._comp == comp_sizes.index[0]].gene.nunique()),
        "split_sizes": {sp: int(len(sets[sp])) for sp in SPL},
    }, d, part_clu


# ------------------------------------------------ 4. contact-defined input boundaries
def section_contact_defined_crops(d):
    """str_ rows get dbd = [min,max] residue within 4.5 A of DNA
    (build_deeppbs_structural_v2.py). For rows in TEST that means the model INPUT
    boundary was chosen using the held-out co-crystal."""
    is_str = d.filename.astype(str).str.startswith("str_")
    tbl = (d[is_str].groupby("split").size().to_dict())
    test_rows = d[is_str & (d.split == "test")][
        ["filename", "gene_symbol", "family_name", "seq_length"]].copy()
    return tbl, test_rows


# -------------------------------------------------- 5. application-set leakage check
def section_applications(d):
    out = {}
    bp = "results/mutation_benchmark/barrera_pairs.json"
    if os.path.exists(bp):
        pairs = json.load(open(bp)).get("pairs", [])
        genes = sorted({str(p["gene"]).upper() for p in pairs})
        wt_seqs = {str(p["wt_seq"]) for p in pairs if p.get("wt_seq")}
        in_train = sorted(set(genes) & set(d[d.split == "train"].gene))
        seq_hits = int(d[d.sequence.isin(wt_seqs)].shape[0])
        out["barrera"] = {"n_pairs": len(pairs), "n_genes": len(genes),
                          "genes_in_train": in_train, "n_genes_in_train": len(in_train),
                          "exact_wt_sequence_rows_in_table": seq_hits,
                          "split_of_those_rows": d[d.sequence.isin(wt_seqs)]
                          .split.value_counts().to_dict()}
    my = d[d.gene == "MYOD1"]
    out["myod1"] = {"rows": len(my),
                    "splits": my.split.value_counts().to_dict(),
                    "sequences": {r.filename: r.sequence for r in my.itertuples()}}
    return out


# ------------------------------------------------------ 6. multi-domain crop choice
def section_multidomain():
    """cluster_crop_v2.py picks the domain cluster matching the MOTIF-DB family.
    Its audit table is written to /tmp (non-durable); copy it if still present."""
    src = "/tmp/cluster_crops_v2.parquet"
    if not os.path.exists(src):
        return {"status": "MISSING", "path": src,
                "note": "cluster_crop_v2.py writes only to /tmp; artifact not durable"}
    t = pd.read_parquet(src)
    dst = os.path.join(OUT, "cluster_crops_v2_recovered.parquet")
    t.to_parquet(dst)
    multi = t[t.n_clusters > 1] if "n_clusters" in t.columns else t
    return {"status": "recovered", "path": src, "saved_to": dst,
            "rows": len(t), "multi_cluster_genes": int(len(multi)),
            "ambiguous": int(t.crop_ambiguous.sum()) if "crop_ambiguous" in t else None,
            "reasons": t.reason.value_counts().to_dict() if "reason" in t else None,
            "genes_multi": sorted(multi.gene.astype(str).tolist())[:400]}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mmseqs", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("[1/6] dataset inventory ...", flush=True)
    inv = section_inventory()
    pd.DataFrame(inv).to_csv(f"{OUT}/01_inventory.csv", index=False)

    print("[2/6] recognition / contact index validity ...", flush=True)
    rec_t, con_t, detail = section_index_validity()
    pd.DataFrame(rec_t).to_csv(f"{OUT}/02_recognition_index_validity.csv", index=False)
    pd.DataFrame(con_t).to_csv(f"{OUT}/02_contact_index_validity.csv", index=False)
    pd.DataFrame(detail).to_csv(f"{OUT}/02_invalid_index_detail.csv", index=False)

    print("[3/6] split overlap (mmseqs may take ~1 min) ...", flush=True)
    split_res, d, part_clu = section_split(skip_mmseqs=a.skip_mmseqs)
    pd.DataFrame(split_res["overlaps"]).to_csv(f"{OUT}/03_split_overlaps.csv", index=False)

    print("[4/6] contact-defined input boundaries ...", flush=True)
    str_tbl, test_rows = section_contact_defined_crops(d)
    test_rows.to_csv(f"{OUT}/04_test_rows_with_contact_defined_dbd.csv", index=False)

    print("[5/6] application-set leakage ...", flush=True)
    apps = section_applications(d)

    print("[6/6] multi-domain crop selection ...", flush=True)
    md = section_multidomain()

    summary = {
        "inventory": inv,
        "recognition_index_validity": rec_t,
        "contact_index_validity": con_t,
        "split": split_res,
        "structure_rows_by_split": {k: int(v) for k, v in str_tbl.items()},
        "n_test_rows_with_contact_defined_dbd": int(len(test_rows)),
        "applications": apps,
        "multidomain": md,
    }
    with open(f"{OUT}/audit_phase0.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # ---- console digest
    print("\n================ DIGEST ================")
    for r in inv:
        print(f"  {r['version']:10s} rows={r.get('rows')} str={r.get('str_rows')} "
              f"medlen={r.get('seqlen_median')} dbd_frac={r.get('dbd_fraction_of_input')}")
    print("\n  recognition indices:")
    for r in rec_t:
        print(f"    {r['version']:10s} total={r['indices_total']:5d} "
              f"in_dbd={r['inside_dbd']:5d} outside_dbd={r['in_seq_outside_dbd']:4d} "
              f"CLIPPED={r['silently_clipped']:4d}")
    print("\n  contact indices:")
    for r in con_t:
        print(f"    {r['version']:10s} links={r['residue_links_total']:6d} "
              f"CLIPPED={r['residue_links_silently_clipped']:5d} "
              f"outside_dbd={r['residue_links_outside_dbd']:5d} "
              f"cols_dropped={r['columns_dropped_out_of_range']:4d} "
              f"cols_emptied={r['columns_emptied_by_clipping']:4d}")
    print("\n  split overlaps (shared units):")
    for r in split_res["overlaps"]:
        flag = "  <-- LEAK" if r["shared"] else ""
        print(f"    {r['unit']:38s} {r['pair']:12s} shared={r['shared']:5d}{flag}")
    print(f"\n  components crossing train/val/test: {split_res['components_crossing_train_val_test']}")
    print(f"  largest component: {split_res['largest_components']} "
          f"({split_res['largest_component_genes']} genes in the biggest)")
    print(f"\n  structure rows by split: {str_tbl}")
    print(f"  TEST rows whose DBD boundary came from the held-out co-crystal: {len(test_rows)}")
    if "barrera" in apps:
        b = apps["barrera"]
        print(f"\n  Barrera: {b['n_pairs']} pairs / {b['n_genes']} genes; "
              f"{b['n_genes_in_train']} genes ALSO IN TRAIN -> {b['genes_in_train']}")
    print(f"  MYOD1 rows: {apps['myod1']['splits']}")
    print(f"\n  wrote {OUT}/")


if __name__ == "__main__":
    main()
