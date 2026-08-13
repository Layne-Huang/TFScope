"""Dimer-duplication dataset: for DIMERIC families (bHLH, bZIP, Nuclear_Receptor)
the input sequence becomes DBD–linker–DBD (homodimer = two copies, so the model
sees both monomers). MONOMERIC families are left unchanged. Fair test of whether
dimer context helps (esp. bZIP), with no architecture change.
"""
import pandas as pd, numpy as np

DIMERIC = {"bHLH", "bZIP", "Nuclear_Receptor"}
LINKER = "GGGGSGGGGS"   # flexible GS linker between the two DBD copies
SRC = {
    "data/processed/tf_pwm_combined_fm_deeppbs.parquet":
        "data/processed/tf_pwm_combined_fm_deeppbs_dimerdup.parquet",      # train corpus
    "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet":
        "data/processed/tf_pwm_deeppbs_only_canon_trim_dimerdup.parquet",  # benchmark test
}

for src, dst in SRC.items():
    d = pd.read_parquet(src).copy()
    n_dup = 0
    seqs, slens, dends = [], [], []
    for r in d.itertuples():
        seq = str(r.sequence); s, e = int(r.dbd_start), int(r.dbd_end)
        if str(r.family_name) in DIMERIC:
            dbd = seq[s:e]
            new = dbd + LINKER + dbd                 # DBD–linker–DBD (whole = DBD)
            seqs.append(new); slens.append(len(new)); dends.append(len(new)); n_dup += 1
        else:
            seqs.append(seq); slens.append(int(r.seq_length)); dends.append(e)
    d["sequence"] = seqs
    d["seq_length"] = slens
    d["dbd_start"] = 0
    d["dbd_end"] = dends
    d.to_parquet(dst)
    print(f"{src.split('/')[-1]:45s} -> dup {n_dup}/{len(d)} dimeric records  "
          f"(dimer seq len: {min(l for l,f in zip(slens,d.family_name) if f in DIMERIC)}-"
          f"{max(l for l,f in zip(slens,d.family_name) if f in DIMERIC)})")
    print(f"  saved {dst}")
print("done")
