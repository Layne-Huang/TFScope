#!/usr/bin/env python
"""Phase 3 prep: assemble native bHLH seq->PWM subset (MyoD1 cluster held out) and
precompute RAW frozen ESM-2 650M per-residue embeddings (no v24 LoRA -> independent
decoder). Also embeds MyoD1 WT + L112R for the go/no-go. Saves one npz.
"""
import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
dev = "cuda"
OUT = "/data1/leihuang/TFScope/phase3_bhlh.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

tr = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet")
bh = tr[tr.family_name == "bHLH"].copy()
# hold out MyoD1 AND its cluster (group_id) entirely -> zero-shot mutation target
myo_groups = set(tr[tr.gene_symbol.str.upper() == "MYOD1"]["group_id"])
bh = bh[~bh["group_id"].isin(myo_groups)]
bh = bh[~bh.gene_symbol.str.upper().eq("MYOD1")]
def dec(b): return np.frombuffer(b, np.float32).reshape(4, -1)
bh = bh[bh["pwm"].apply(lambda b: isinstance(b, (bytes, bytearray)) and len(b) >= 4 * 4 * 4)]
print(f"bHLH training rows (MyoD1 cluster {sorted(myo_groups)} held out): {len(bh)} / {bh.gene_symbol.nunique()} genes")

# ── raw frozen ESM-2 650M ──
import esm as esm_lib
model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(dev)
for p in model.parameters(): p.requires_grad = False
bc = alphabet.get_batch_converter()

@torch.no_grad()
def embed(seq):
    _, _, toks = bc([("x", seq)])
    rep = model(toks.to(dev), repr_layers=[33])["representations"][33][0]
    return rep[1:1 + len(seq)].float().cpu().numpy()          # (L,1280) strip cls/eos

rows = []
for i, (_, r) in enumerate(bh.iterrows()):
    seq = r["sequence"]; pwm = dec(r["pwm"])
    rows.append(dict(gene=r["gene_symbol"], seq=seq, emb=embed(seq), pwm=pwm,
                     motif_len=pwm.shape[1]))
    if (i + 1) % 50 == 0: print(f"  embedded {i+1}/{len(bh)}")

# MyoD1 WT + L112R (eval only, never trained)
WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"; MUTPOS = 11
MUT = WT[:MUTPOS] + "R" + WT[MUTPOS + 1:]
myo = dict(wt_seq=WT, mut_seq=MUT, mutpos=MUTPOS, wt_emb=embed(WT), mut_emb=embed(MUT))

def to_obj(lst):
    a = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst): a[i] = x
    return a
np.savez_compressed(OUT,
                    genes=np.array([r["gene"] for r in rows]),
                    seqs=to_obj([r["seq"] for r in rows]),
                    embs=to_obj([r["emb"] for r in rows]),
                    pwms=to_obj([r["pwm"] for r in rows]),
                    myo=to_obj([myo]))
print(f"saved {OUT}  ({len(rows)} train rows, MyoD1 WT/MUT embedded for go/no-go)")
