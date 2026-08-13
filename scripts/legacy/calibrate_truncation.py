#!/usr/bin/env python
"""Steps 1-2 of the truncation plan: learn where the real DNA-contacting
region sits inside an annotated DBD, then validate it on held-out genes.

Calibration set = the 298 genes that have BOTH an annotation-based crop
(tf_pwm_aug_dbd_canon_trim.parquet, pre-v2) and a real 5A DNA-contact crop
(tf_pwm_deeppbs_v2_deduped.parquet).

For each gene we locate the structural crop inside the annotation crop and
record the N-/C-terminal trims. Per family we take the median trim, then test
on held-out genes whether applying those medians reproduces the true
structural crop (boundary error + IoU).

This is a GO/NO-GO gate: if held-out IoU is poor, family-average trimming is
not justified and we should fall back to the recognition-region prior instead.
"""
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

MIN_FAM_N = 5          # families with fewer calibration genes are not trusted


def locate(struct_seq, ann_seq):
    """Where does struct_seq sit inside ann_seq? -> (start, end) or None."""
    i = ann_seq.find(struct_seq)
    if i >= 0:
        return i, i + len(struct_seq)
    # PDB vs UniProt differences (tags, point mutations, isoforms): fall back
    # to the longest common block and extend it to the structural length
    sm = SequenceMatcher(None, ann_seq, struct_seq, autojunk=False)
    m = sm.find_longest_match(0, len(ann_seq), 0, len(struct_seq))
    if m.size < 15:
        return None
    start = max(0, m.a - m.b)
    end = min(len(ann_seq), start + len(struct_seq))
    if end - start < 15:
        return None
    return start, end


def build_pairs():
    ann = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim.parquet")
    st = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2_deduped.parquet")
    ann = ann.drop_duplicates("gene_symbol").copy()
    ann["g"] = ann["gene_symbol"].str.upper()
    fam = dict(zip(ann["g"], ann["family_name"]))
    ann_seq = dict(zip(ann["g"], ann["sequence"].astype(str)))

    rows = []
    for g, sub in st.groupby(st["gene"].str.upper()):
        a = ann_seq.get(g)
        if not a:
            continue
        for s in sub["sequence"].astype(str):
            loc = locate(s, a)
            if loc is None:
                rows.append(dict(gene=g, family=fam.get(g), ok=False))
                continue
            s0, s1 = loc
            rows.append(dict(gene=g, family=fam.get(g), ok=True,
                             ann_len=len(a), struct_len=len(s),
                             n_trim=s0, c_trim=len(a) - s1,
                             frac_retained=(s1 - s0) / len(a)))
    return pd.DataFrame(rows)


def learn(df):
    """median N/C trim per family, as a FRACTION of annotation length."""
    d = df[df["ok"]].copy()
    d["n_frac"] = d["n_trim"] / d["ann_len"]
    d["c_frac"] = d["c_trim"] / d["ann_len"]
    g = d.groupby("family").agg(n=("gene", "nunique"),
                                 n_frac=("n_frac", "median"),
                                 c_frac=("c_frac", "median"),
                                 struct_len=("struct_len", "median"),
                                 struct_p10=("struct_len", lambda x: np.percentile(x, 10)))
    return g


def apply_trim(ann_len, rule):
    s = int(round(ann_len * rule["n_frac"]))
    e = ann_len - int(round(ann_len * rule["c_frac"]))
    if e - s < 10:                       # degenerate -> keep whole
        return 0, ann_len
    return s, e


def main():
    pairs = build_pairs()
    ok = pairs[pairs["ok"]]
    print(f"structural crops located inside the annotation crop: {len(ok)}/{len(pairs)} "
          f"({ok['gene'].nunique()} genes)")
    print("\nper-family calibration (all data):")
    print(learn(pairs).round(3).to_string())

    # ---- holdout validation ----
    genes = sorted(ok["gene"].unique())
    rng = np.random.default_rng(0)
    rng.shuffle(genes)
    cut = int(len(genes) * 0.7)
    train_g, test_g = set(genes[:cut]), set(genes[cut:])
    rules = learn(pairs[pairs["gene"].isin(train_g)])
    print(f"\nholdout: train={len(train_g)} genes, test={len(test_g)} genes")

    recs = []
    for _, r in ok[ok["gene"].isin(test_g)].iterrows():
        fam = r["family"]
        if fam not in rules.index or rules.loc[fam, "n"] < MIN_FAM_N:
            continue
        ps, pe = apply_trim(int(r["ann_len"]), rules.loc[fam])
        ts, te = int(r["n_trim"]), int(r["ann_len"] - r["c_trim"])
        inter = max(0, min(pe, te) - max(ps, ts))
        union = max(pe, te) - min(ps, ts)
        recs.append(dict(family=fam, iou=inter / union if union else 0,
                         start_err=abs(ps - ts), end_err=abs(pe - te),
                         pred_len=pe - ps, true_len=te - ts,
                         # baseline: keep the whole annotation crop, no trimming
                         iou_base=(te - ts) / int(r["ann_len"])))
    v = pd.DataFrame(recs)
    print(f"\nvalidated on {len(v)} held-out structural crops")
    print(f"  mean IoU (learned trim) : {v['iou'].mean():.3f}   median {v['iou'].median():.3f}")
    print(f"  mean IoU (no trim, base): {v['iou_base'].mean():.3f}   median {v['iou_base'].median():.3f}")
    print(f"  median |start error|: {v['start_err'].median():.0f} aa")
    print(f"  median |end error|  : {v['end_err'].median():.0f} aa")
    print(f"  median pred_len {v['pred_len'].median():.0f} vs true_len {v['true_len'].median():.0f}")
    print("\nper-family IoU (learned vs baseline):")
    print(v.groupby("family").agg(n=("iou", "size"), iou=("iou", "mean"),
                                   iou_base=("iou_base", "mean"),
                                   start_err=("start_err", "median"),
                                   end_err=("end_err", "median")).round(3).to_string())
    learn(pairs).to_parquet("/tmp/truncation_rules.parquet")
    print("\nsaved /tmp/truncation_rules.parquet")


if __name__ == "__main__":
    main()
