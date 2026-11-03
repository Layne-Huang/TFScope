#!/usr/bin/env python
"""A1 — train TFScope on a byte-identical copy of DeepPBS's folds, with DeepPBS's labels.

The question this answers: does TFScope's edge survive when it is given *exactly* the
data DeepPBS is given — the same structures, the same train/valid partition, and the
same masked PWM targets — instead of its own 4 794-row corpus?

Two things are taken verbatim from DeepPBS and nothing is substituted:

1. **The fold membership.** `iclr_folds/train0.txt` (396) and `valid0.txt` (81) are
   read as-is. These are the gene-disjoint retrain folds behind the published
   DeepPBS-vs-v24 tie, so the comparison is against the same data the retrained
   DeepPBS ensemble saw. Each NPZ is `<pdb>_<chain>_<motifid>.npz`; the (pdb, chain)
   key resolves to exactly one row of `tf_pwm_deeppbs_v2.parquet` for all 477 entries
   (1:1, no ambiguity, no drops). That parquet is the pre-dedup structural table
   TFScope's own `str_*` rows were carved from, so the DBD crop the model sees is the
   same crop TFScope uses everywhere else — only the row *set* changes.

   (The deduped table `tf_pwm_deeppbs_v2_deduped.parquet`, whose row i is exactly
   `str_i`, covers only 350/477. The pre-dedup table is what makes the copy exact.)

2. **The labels.** Target = `Y_pwm[0]` restricted to `pwm_mask[0]`, i.e. DeepPBS's own
   masked PWM: only the columns covered by DNA in the co-crystal are supervised.
   `Y_pwm` is (2, L, 4) with strand 1 the exact reverse complement of strand 0 (verified
   477/477), so strand 0 is the whole label. The valid region is a contiguous block in
   all 477 cases, so it slices cleanly onto TFScope's prefix-mask convention; mean valid
   length 9.6 columns (range 5-19). This is a real change, not padding removal — in 303
   of 477 rows the masked-out columns carry non-zero probability that TFScope's own
   parquet PWM would have supervised.

Test set is untouched: the 291 canonical `str_*` rows are copied verbatim (v23 PWMs,
v23 partners) into the same parquet so `scripts/train.py` can read one file, and the
split's `test` list is the canonical 291. Only training and validation change.

**Leakage, stated up front.** DeepPBS's folds were made disjoint from the 20 test genes
that have a co-crystal, not from all 51 genes of the 291-row benchmark. 16 test genes
(65 of 291 rows: ERF, FOXC2/D3/G1/K2/N3, GCN4, POU1F1/2F1/3F1/5F1/5F1::SOX2/6F1, SKN-1,
SPDEF, THRB) do appear in fold 0, plus 29 shared PDBs. That leakage flatters A1, so
`make_ablation_table.py --leak-free` also reports the 226-row / 35-gene clean subset.
Keeping the folds byte-identical was the explicit requirement; the clean subset is the
honest read.

**Contact supervision follows the data.** The v23 contact files are keyed by `str_*`, so
without a remap the fold rows would silently receive no contact supervision at all and A1
would secretly become "DeepPBS data *and* no contacts". This script therefore re-keys them
onto the new `dpb_*` ids, transferring only what transfers exactly:

  * the 1-D recognition prior is in *residue* space and is column-independent, so it
    transfers verbatim for every fold row with a `str_*` twin — 164 rows;
  * the 2-D distillation target is indexed by *PWM column*, and DeepPBS's masked label
    has a different column extent than the v23 PWM (e.g. FOS 8 vs 11 columns) with no
    recoverable register between them, so it transfers only where the two label lengths
    agree 1:1 — 54 rows.

For scale, v24's own run has 254 recognition priors and 213 2-D targets over 4 794 train
rows; A1 has 164 and 54 over 477. Thinner in absolute terms, denser per row. This is
inherent to the ablation — DeepPBS's labels do not come with TFScope's column annotation.

Out:
    data/processed/tf_pwm_deeppbs_fold0.parquet
    data/processed/splits/ablation_v24/deeppbs_fold0.json
    data/processed/splits/ablation_v24/deeppbs_fold0_leaky_genes.json
    data/contact_maps/recognition_residues_deeppbsfold0.json
    data/contact_maps/contact_targets_deeppbsfold0.json

    scripts/ablation_v24/build_deeppbs_fold_dataset.py [--fold 0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

FOLDS = "/data1/leihuang/DeepPBS/deeppbsmar24/run/iclr_folds"
NPZ_DIR = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
V23 = "data/processed/tf_pwm_training_v23.parquet"
V23_SPLIT = "data/processed/splits/train_v22/split.json"
STRUCT_ALL = "data/processed/tf_pwm_deeppbs_v2.parquet"          # pre-dedup: covers 477/477
STRUCT_DEDUP = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"  # row i == str_i
OUT_PARQUET = "data/processed/tf_pwm_deeppbs_fold0.parquet"
OUT_SPLIT = "data/processed/splits/ablation_v24/deeppbs_fold0.json"
OUT_LEAK = "data/processed/splits/ablation_v24/deeppbs_fold0_leaky_genes.json"
RECOG_IN = "data/contact_maps/recognition_residues_v23.json"
CT_IN = "data/contact_maps/contact_targets_v23.json"
RECOG_OUT = "data/contact_maps/recognition_residues_deeppbsfold0.json"
CT_OUT = "data/contact_maps/contact_targets_deeppbsfold0.json"

MAX_PARTNERS = 3      # + self = 4 chains, matching v24's --max-chains 4
MAX_PLEN = 150        # per-partner DBD crop cap, matching build_nchain_v23.py


def parse_npz(name: str) -> tuple[str, str]:
    pdb, chain, _rest = name[:-4].split("_", 2)
    return pdb.lower(), chain


def masked_pwm(npz_path: str) -> tuple[np.ndarray, int, int]:
    """DeepPBS label: Y_pwm[0] restricted to pwm_mask[0]. Returns (4, Lv), Lv, L_full."""
    z = np.load(npz_path, allow_pickle=True)
    Y, M = z["Y_pwm"][0], z["pwm_mask"][0]
    idx = np.where(M)[0]
    if idx.size == 0:
        raise ValueError(f"{npz_path}: empty pwm_mask")
    if idx[-1] - idx[0] + 1 != idx.size:
        raise ValueError(f"{npz_path}: non-contiguous pwm_mask, cannot use a prefix mask")
    core = np.asarray(Y[idx[0]:idx[-1] + 1], dtype=np.float32).T      # (4, Lv)
    s = core.sum(0, keepdims=True)
    core = np.divide(core, s, out=np.full_like(core, 0.25), where=s > 0)
    return np.ascontiguousarray(core, dtype=np.float32), int(idx.size), int(Y.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    for p in (FOLDS, NPZ_DIR, V23, STRUCT_ALL, STRUCT_DEDUP):
        if not os.path.exists(p):
            sys.exit(f"ABORT: missing input {p}")

    def fold_names(which):
        f = os.path.join(FOLDS, f"{which}{args.fold}.txt")
        return [l.strip() for l in open(f) if l.strip()]

    tr_names, va_names = fold_names("train"), fold_names("valid")
    print(f"=== DeepPBS fold {args.fold}: {len(tr_names)} train + {len(va_names)} valid NPZ ===")

    sa = pd.read_parquet(STRUCT_ALL)
    key = {}
    for i, r in sa.iterrows():
        key.setdefault((str(r.pdb_id).lower(), str(r.chain_id)), []).append(i)
    dup = [k for k, v in key.items() if len(v) > 1]

    # (pdb, chain) -> str_N, for reusing v23's family/partner annotation where it exists
    sd = pd.read_parquet(STRUCT_DEDUP).reset_index(drop=True)
    to_str = {(str(r.pdb_id).lower(), str(r.chain_id)): f"str_{i}" for i, r in sd.iterrows()}

    v23 = pd.read_parquet(V23)
    v23["filename"] = v23.filename.astype(str)
    v23i = v23.set_index("filename")
    # gene -> (family_id, family_name, family_source), for fold rows with no str_ twin
    fam_by_gene = {}
    for r in v23.itertuples():
        fam_by_gene.setdefault(str(r.gene_key), (int(r.family_id), r.family_name, r.family_source))
    OTHER = (9, "Other", "fallback_other")

    # partner DBD crops for rows with no str_ twin: partner_chains -> same-PDB v2 sequence
    seq_by_pc = {(str(r.pdb_id).lower(), str(r.chain_id)): str(r.sequence) for _, r in sa.iterrows()}

    rows, fold_of, n_from_v23, n_derived = [], {}, 0, 0
    for which, names in (("train", tr_names), ("val", va_names)):
        for name in names:
            pc = parse_npz(name)
            if pc not in key:
                sys.exit(f"ABORT: no structural row for {name} {pc}")
            src = sa.loc[key[pc][0]]
            pwm, mlen, lfull = masked_pwm(os.path.join(NPZ_DIR, name))

            gene = str(src.gene)
            gkey = gene.upper()
            twin = to_str.get(pc)
            if twin is not None:
                t = v23i.loc[twin]
                fam = (int(t.family_id), t.family_name, t.family_source)
                partner_seqs = list(t.partner_seqs)
                partner_sequence, partner_gene = str(t.partner_sequence), str(t.partner_gene)
                is_dimer, mce = bool(t.is_dimer), bool(t.multichain_eligible)
                n_from_v23 += 1
            else:
                fam = fam_by_gene.get(gkey, OTHER)
                partner_seqs = []
                for ch in list(src.partner_chains)[:MAX_PARTNERS]:
                    ps = seq_by_pc.get((pc[0], str(ch)), "")
                    if ps:
                        partner_seqs.append(ps[:MAX_PLEN])
                partner_sequence = partner_seqs[0] if partner_seqs else ""
                pg = list(src.partner_genes)
                partner_gene = str(pg[0]) if pg else ""
                is_dimer, mce = bool(src.is_dimer), bool(partner_seqs)
                n_derived += 1

            fn = f"dpb_{len(rows)}"
            fold_of[fn] = which
            seq = str(src.sequence)
            rows.append({
                "filename": fn, "gene_symbol": gene, "sequence": seq,
                "pwm": pwm.tobytes(), "motif_length": mlen,
                "seq_length": len(seq), "dbd_start": 0, "dbd_end": len(seq),
                "family_id": fam[0], "family_name": fam[1], "family_source": fam[2],
                "motif_source": str(src.pwm_source),
                "partner_sequence": partner_sequence, "partner_gene": partner_gene,
                "is_dimer": is_dimer, "_set": "struct", "gene_key": gkey,
                "group_id": f"{gkey}|{seq}|{src.pwm_source}",
                "multichain_eligible": mce,
                "partner_seqs": np.array(partner_seqs, dtype=object),
                "n_chains": 1 + len(partner_seqs),
                # provenance, dropped before writing
                "_npz": name, "_lfull": lfull, "_twin": twin or "",
            })

    if dup:
        print(f"  note: {len(dup)} (pdb,chain) keys are non-unique in the pre-dedup table; "
              f"none of them are hit by fold {args.fold}"
              if not any(parse_npz(n) in set(dup) for n in tr_names + va_names)
              else f"  WARNING: {len(dup)} ambiguous (pdb,chain) keys, first candidate used")
    df = pd.DataFrame(rows)
    prov = df[["filename", "_npz", "_lfull", "_twin"]].rename(
        columns={"_npz": "npz", "_lfull": "label_len_full", "_twin": "v23_twin"}).copy()
    df = df.drop(columns=["_npz", "_lfull", "_twin"])
    print(f"  built {len(df)} fold rows | annotation from the v23 str_ twin: {n_from_v23}, "
          f"derived: {n_derived}")
    print(f"  masked label length: mean {df.motif_length.mean():.2f} "
          f"(min {df.motif_length.min()}, max {df.motif_length.max()}); "
          f"columns dropped by pwm_mask: {int((prov.label_len_full - df.motif_length).sum())}")

    # canonical 291 test rows, verbatim
    split23 = json.load(open(V23_SPLIT))
    test = v23i.loc[split23["test"]].reset_index()
    test = test[df.columns]
    out = pd.concat([df, test], ignore_index=True)

    # leakage report
    fold_genes = set(df.gene_key)
    tl = test[test.gene_key.isin(fold_genes)]
    leaky_genes = sorted(tl.gene_key.unique())
    print(f"  LEAKAGE: {len(tl)}/{len(test)} test rows in {len(leaky_genes)} genes appear in "
          f"fold {args.fold}; clean subset = {len(test) - len(tl)} rows / "
          f"{test.gene_key.nunique() - len(leaky_genes)} genes")

    # ── re-key contact supervision onto the dpb_* ids ──────────────────────────────
    # Without this the fold rows get no contact supervision at all and A1 silently
    # becomes "DeepPBS data AND no contacts". Transfer only what is exact:
    #   1-D recognition prior -> residue space, column-independent: always transferable
    #   2-D distillation      -> PWM-column space: only when the two label lengths agree
    recog_in, ct_in = json.load(open(RECOG_IN)), json.load(open(CT_IN))
    mlen_dpb = dict(zip(df.filename, df.motif_length))
    recog_out, ct_out, n_rp, n_ct, n_ct_skip = {}, {}, 0, 0, 0
    for r in prov.itertuples():
        twin = r.v23_twin
        if not twin:
            continue
        if recog_in.get(twin):
            recog_out[r.filename] = recog_in[twin]
            n_rp += 1
        e = ct_in.get(twin)
        if e is not None:
            if int(v23i.loc[twin].motif_length) == int(mlen_dpb[r.filename]):
                ct_out[r.filename] = e
                n_ct += 1
            else:
                n_ct_skip += 1
    # keep the canonical test keys so the files are drop-in replacements
    for fn in split23["test"]:
        if recog_in.get(fn):
            recog_out[fn] = recog_in[fn]
        if fn in ct_in:
            ct_out[fn] = ct_in[fn]
    json.dump(recog_out, open(RECOG_OUT, "w"))
    json.dump(ct_out, open(CT_OUT, "w"))
    print(f"  contact remap: {n_rp} recognition priors, {n_ct} 2-D targets transferred "
          f"({n_ct_skip} 2-D dropped -- label column count differs, no recoverable register)")

    os.makedirs(os.path.dirname(OUT_SPLIT), exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    split = {
        "train": [f for f, w in fold_of.items() if w == "train"],
        "val": [f for f, w in fold_of.items() if w == "val"],
        "test": list(split23["test"]),
        "excluded": [],
        "metadata": {
            "source_folds": f"{FOLDS}/train{args.fold}.txt + valid{args.fold}.txt",
            "labels": "DeepPBS Y_pwm[0] restricted to pwm_mask[0] (masked PWM)",
            "n_train_rows": sum(1 for w in fold_of.values() if w == "train"),
            "n_val_rows": sum(1 for w in fold_of.values() if w == "val"),
            "n_test_rows": len(split23["test"]),
            "test_labels": "canonical v23 PWMs, unchanged",
            "leaky_test_genes": leaky_genes,
            "n_leaky_test_rows": int(len(tl)),
        },
    }
    json.dump(split, open(OUT_SPLIT, "w"), indent=1)
    json.dump({"leaky_test_genes": leaky_genes,
               "n_leaky_test_rows": int(len(tl)),
               "n_clean_test_rows": int(len(test) - len(tl))},
              open(OUT_LEAK, "w"), indent=1)
    prov.to_csv("data/processed/splits/ablation_v24/deeppbs_fold0_provenance.csv", index=False)
    print(f"  -> {OUT_PARQUET} ({len(out)} rows)\n  -> {OUT_SPLIT}\n  -> {OUT_LEAK}")


if __name__ == "__main__":
    main()
