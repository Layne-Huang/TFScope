#!/usr/bin/env python
"""Phase 7 prep (homeodomain): native HD seq->PWM subset with ALL Barrera-gene clusters
held out, RAW frozen ESM-2 embeddings; plus the 55 Barrera WT/MUT DBD-crop pairs
(measured PWMs + spec.change) embedded for the zero-shot mutation eval.
"""
import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
dev = "cuda"; OUT = "/data1/leihuang/TFScope/phase7_hd.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

tr = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet")
hd = tr[tr.family_name == "Homeodomain"].copy()
pairs = json.load(open("results/mutation_benchmark/barrera_pairs.json"))["pairs"]
barrera_genes = {p["gene"].upper() for p in pairs}
# hold out every cluster that contains a Barrera gene
bar_groups = set(tr[tr.gene_symbol.str.upper().isin(barrera_genes)]["group_id"])
hd = hd[~hd["group_id"].isin(bar_groups)]
hd = hd[~hd.gene_symbol.str.upper().isin(barrera_genes)]
def dec(b): return np.frombuffer(b, np.float32).reshape(4, -1)
hd = hd[hd["pwm"].apply(lambda b: isinstance(b, (bytes, bytearray)) and len(b) >= 4 * 4 * 4)]
print(f"native HD training rows (Barrera clusters held out): {len(hd)} / {hd.gene_symbol.nunique()} genes")

# S6 spec.change per (gene, mut)
s6 = pd.read_csv("/data1/leihuang/rCLAMPS/barrera2016_SuppTable_S6_combined.csv")
spec = {}
for _, r in s6.iterrows():
    k = (r["prot"], r["sub"]); spec[k] = spec.get(k, False) or (str(r["spec.change"]).strip() == "Yes")

import esm as esm_lib
model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D(); model = model.eval().to(dev)
for p in model.parameters(): p.requires_grad = False
bc = alphabet.get_batch_converter()
@torch.no_grad()
def embed(seq):
    _, _, t = bc([("x", seq)]); r = model(t.to(dev), repr_layers=[33])["representations"][33][0]
    return r[1:1 + len(seq)].float().cpu().numpy()

rows = []
for i, (_, r) in enumerate(hd.iterrows()):
    rows.append(dict(gene=r["gene_symbol"], seq=r["sequence"], emb=embed(r["sequence"]), pwm=dec(r["pwm"])))
    if (i + 1) % 200 == 0: print(f"  HD embedded {i+1}/{len(hd)}")

bar = []
for p in pairs:
    k = (p["gene"], p["mut"])
    bar.append(dict(gene=p["gene"], mut=p["mut"], wt_seq=p["wt_seq"], mut_seq=p["mut_seq"],
                    wt_emb=embed(p["wt_seq"]), mut_emb=embed(p["mut_seq"]),
                    wt_pwm=np.array(p["wt_pwm"], np.float32), mut_pwm=np.array(p["mut_pwm"], np.float32),
                    spec_change=bool(spec.get(k, False))))
print(f"Barrera pairs embedded: {len(bar)} | spec.change Yes {sum(b['spec_change'] for b in bar)}")

def to_obj(lst):
    a = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst): a[i] = x
    return a
np.savez_compressed(OUT,
                    genes=np.array([r["gene"] for r in rows]),
                    seqs=to_obj([r["seq"] for r in rows]), embs=to_obj([r["emb"] for r in rows]),
                    pwms=to_obj([r["pwm"] for r in rows]), bar=to_obj(bar))
print(f"saved {OUT} ({len(rows)} HD train rows, {len(bar)} Barrera pairs)")
