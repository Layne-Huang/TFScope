"""Build a SPARSE recognition-prior JSON for the contact-bias ablation.

- keys = parquet filenames (matched by pdb_id + protein_chain to the probe's complexes)
- train filenames  -> TRUE 4.5A contacts (label==1 in the probe CSV)
- val/test filenames -> PROBE-PREDICTED contacts (score>thr), i.e. what deployment uses
- residue indices are DBD-relative (position - dbd_start), matching the dataset convention.

Verifies frame alignment against the existing broad recognition_residues.json before writing.
"""
import json, numpy as np, pandas as pd
D = "/data1/leihuang/TFScope/esm_contact_diagnostic"
PRED = pd.read_csv(f"{D}/residue_probe_predictions.csv")
META = pd.read_csv(f"{D}/contact_diagnostic_metadata.csv")
meta = {r.complex_id: r for r in META.itertuples()}

parq = pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet")
split = json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))
train_fn = set(split["train"]); heldout_fn = set(split.get("val", [])) | set(split.get("test", []))
broad = json.load(open("data/contact_maps/recognition_residues.json"))

# map (pdb_lower, chain) -> complex_id
key2cx = {}
for cx, r in meta.items():
    key2cx[(str(r.pdb_id).lower(), str(r.protein_chain))] = cx

def fn_key(fn):
    p = fn.split("_")
    return (p[0].lower(), p[1]) if len(p) >= 2 else None

# per-complex: DBD-relative true contacts + predicted contacts
THR = 0.5
cx_true, cx_pred = {}, {}
for cx, g in PRED.groupby("complex_id"):
    ds = int(meta[cx].dbd_start) if cx in meta else 0
    pos = g.position.values - ds
    tru = sorted(int(p) for p, l in zip(pos, g.label.values) if l == 1 and p >= 0)
    prd = sorted(int(p) for p, s in zip(pos, g.score.values) if s > THR and p >= 0)
    cx_true[cx] = tru; cx_pred[cx] = prd

# ── frame verification: true contacts should sit within the broad DBD prior ──
checked = 0; inside = 0
for fn in list(train_fn)[:200]:
    k = fn_key(fn); cx = key2cx.get(k)
    if cx is None or fn not in broad or not cx_true.get(cx): continue
    bset = set(broad[fn])
    frac = np.mean([t in bset for t in cx_true[cx]])
    checked += 1; inside += (frac > 0.6)
print(f"FRAME CHECK: {inside}/{checked} train TFs have >60% of true contacts inside the broad DBD prior "
      f"(high = frames aligned)")

# ── build sparse prior JSON ──
out = {}; n_tr = n_ho = miss = 0
allfn = set(parq["filename"].astype(str))
for fn in allfn:
    k = fn_key(fn); cx = key2cx.get(k)
    if cx is None: miss += 1; continue
    if fn in train_fn and cx_true.get(cx):
        out[fn] = cx_true[cx]; n_tr += 1
    elif fn in heldout_fn and cx_pred.get(cx):
        out[fn] = cx_pred[cx]; n_ho += 1
    elif cx_true.get(cx):                       # train fallback / unclassified -> true
        out[fn] = cx_true[cx]; n_tr += 1
print(f"prior built: {len(out)} entries | train(true)={n_tr} heldout(pred)={n_ho} | unmapped filenames={miss}")
lens = [len(v) for v in out.values()]
print(f"sparsity: median {int(np.median(lens))} contacts/TF (broad prior was ~21) | min {min(lens)} max {max(lens)}")
json.dump(out, open("data/contact_maps/recognition_residues_sparse_biasablation.json", "w"))
print("saved data/contact_maps/recognition_residues_sparse_biasablation.json")

# show a couple examples
for fn in list(out)[:3]:
    print(f"  {fn[:40]:40} true/pred contacts -> {out[fn][:12]}{'...' if len(out[fn])>12 else ''}")
