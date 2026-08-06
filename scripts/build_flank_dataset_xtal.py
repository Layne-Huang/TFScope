#!/usr/bin/env python
"""Second flank variant (user choice): PDB rows use the FULL resolved crystal
chain (not the DNA-contact crop, not UniProt-flank), with the DBD marked at the
contact-crop location; sequence-only rows keep DBD+flanks (reused from v25flank).

  str_ rows: full protein_chains[chain] from the co-crystal CIF; dbd=[loc of the
             v23 contact-crop within that chain]. Fallback: contact-crop (dbd_start=0).
  seq_ rows: unchanged from tf_pwm_training_v25flank.parquet (DBD +/-20aa).

Out: data/processed/tf_pwm_training_v25xtal.parquet
"""
import sys, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

V25FLANK = "data/processed/tf_pwm_training_v25flank.parquet"   # seq_ rows already DBD+flank
V23 = "data/processed/tf_pwm_training_v23.parquet"             # original contact-crops
DV2 = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"       # index-aligned pdb_id/chain_id
CIFDIR = "data/raw/pdb_cif_cache"
OUT = "data/processed/tf_pwm_training_v25xtal.parquet"


def full_chain_seq(pdb_id, chain_id):
    import os
    p = os.path.join(CIFDIR, f"{pdb_id.lower()}.cif")
    if not os.path.exists(p):
        return None
    try:
        chains, _, _ = bd.load_chains(p)
    except Exception:
        return None
    ent = chains.get(chain_id) or chains.get(chain_id.upper()) or chains.get(chain_id.lower())
    if not ent:
        return None
    return "".join(e[2] for e in ent)


def main():
    out = pd.read_parquet(V25FLANK).reset_index(drop=True)   # start from v25flank (seq_ done)
    v23 = pd.read_parquet(V23).set_index("filename")
    dv2 = pd.read_parquet(DV2).reset_index(drop=True)

    seq = out["sequence"].tolist(); ds = out["dbd_start"].tolist()
    de = out["dbd_end"].tolist(); src = out["flank_source"].tolist()
    is_str = out["filename"].str.startswith("str_").tolist()
    n_full = n_fallback = 0
    str_rows = [i for i, s in enumerate(is_str) if s]
    print(f"processing {len(str_rows)} str_ rows (CIF parse) ...", flush=True)
    for c, i in enumerate(str_rows):
        fn = out.at[i, "filename"]; idx = int(fn.replace("str_", ""))
        crop = str(v23.at[fn, "sequence"]) if fn in v23.index else None
        pdb = str(dv2.iloc[idx]["pdb_id"]) if idx < len(dv2) else None
        chain = str(dv2.iloc[idx]["chain_id"]) if idx < len(dv2) else None
        full = full_chain_seq(pdb, chain) if (pdb and chain) else None
        pos = full.find(crop) if (full and crop) else -1
        if pos >= 0:
            seq[i] = full; ds[i] = pos; de[i] = pos + len(crop); src[i] = "xtal_full"; n_full += 1
        else:
            # fallback: original contact-crop, DBD = whole crop
            seq[i] = crop if crop else seq[i]; ds[i] = 0; de[i] = len(seq[i]); src[i] = "dbd_only"; n_fallback += 1
        if (c + 1) % 200 == 0:
            print(f"  {c+1}/{len(str_rows)}  full={n_full} fallback={n_fallback}", flush=True)

    out["sequence"] = seq; out["dbd_start"] = ds; out["dbd_end"] = de
    out["seq_length"] = [len(s) for s in seq]; out["flank_source"] = src
    out.to_parquet(OUT, index=False)
    print(f"\nstr_ rows: full_crystal_chain={n_full}  fallback_dbd_only={n_fallback}")
    print(f"median str_ len: {int(np.median([len(seq[i]) for i in str_rows]))} (contact-crop was ~73)")
    print("wrote", OUT)
    # sanity on a few
    for g in ["ETS1", "P53", "CLOCK"]:
        r = out[out.gene_symbol.astype(str).str.upper() == g]
        r = r[r.filename.str.startswith("str_")]
        if len(r):
            x = r.iloc[0]; print(f"  {g}: len={len(x.sequence)} dbd=[{x.dbd_start}:{x.dbd_end}] src={x.flank_source}")


if __name__ == "__main__":
    main()
