#!/usr/bin/env python
"""Inspect selected DPAC proteins: do they look like sequence-specific DBDs?

For sampled DPAC proteins show: protein length, bound DNA + pseudo consensus,
and the NEAREST real TF (ESM2 cosine) with its gene/family/motif consensus.
Focus on the proteins that displaced a real TF for top-1 in the diagnostic.
"""
import sys, json, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm

DATA = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"
EMB_REAL = "data/processed/tf_dbd_embeddings_aug.npz"
EMB_DPAC = "data/processed/dpac_dbd_embeddings.npz"
PWM_DPAC = "data/processed/dpac_pseudo_pwms.npz"
PROT_DPAC = "data/processed/dpac_proteins.parquet"
BASES = np.array(list("ACGT"))
IC_THRESH, MIN_POS = 0.25, 4


def decode(b, L): return np.frombuffer(b, np.float32).reshape(4, L).copy()
def consensus(p): return "".join(BASES[p.argmax(0)])
def ic_cols(p): p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)
def trim(p):
    inf = np.where(ic_cols(p) >= IC_THRESH)[0]
    if len(inf) == 0: return None
    c = p[:, inf[0]:inf[-1] + 1]; return c / c.sum(0, keepdims=True)


split = json.load(open(SPLIT))
donor_set = set(split["train"]) | set(split["val"])
df = pd.read_parquet(DATA)
fn_pwm = {r.filename: decode(r.pwm, int(r.motif_length)) for r in df.itertuples()}
fn_gene = dict(zip(df.filename, df.gene_symbol)); fn_fam = dict(zip(df.filename, df.family_name))
dprot = pd.read_parquet(PROT_DPAC).set_index("filename")

er = np.load(EMB_REAL); ed = np.load(EMB_DPAC); pdw = np.load(PWM_DPAC)
real_fns = [fn for fn in donor_set if fn in er.files]
R = np.stack([er[fn] for fn in real_fns]).astype(np.float32)
Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)

# re-find DPAC top-1 winners over test queries
test = [fn for fn in split["test"] if fn in er.files and fn in fn_pwm]
D = np.stack([ed[fn] for fn in pdw.files]).astype(np.float32)
Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
dpac_fns = list(pdw.files)

winners = []   # (query_fn, dpac_fn, cos, oracle_r_of_dpac, oracle_r_of_best_real)
for fn in test:
    core = trim(fn_pwm[fn])
    if core is None or core.shape[1] < MIN_POS: continue
    q = er[fn].astype(np.float32); q = q / (np.linalg.norm(q) + 1e-8)
    sd = Dn @ q; sr = Rn @ q
    if fn in real_fns: sr[real_fns.index(fn)] = -2.0
    bd = int(sd.argmax()); br = int(sr.argmax())
    if sd[bd] > sr[br]:    # DPAC won top-1
        dc = trim(pdw[dpac_fns[bd]]); rr_d = align_pwm(dc, core, 10, True)[3] if dc is not None else -1
        rc = trim(fn_pwm[real_fns[br]]); rr_r = align_pwm(rc, core, 10, True)[3] if rc is not None else -1
        winners.append((fn, dpac_fns[bd], float(sd[bd]), float(rr_d), float(rr_r)))

winners.sort(key=lambda x: x[3] - x[4])   # worst DPAC-vs-real motif gap first
print(f"DPAC won top-1 for {len(winners)} test TFs. Showing 8 worst-mismatch cases:\n")
for fn, dfn, cos, rd, rr in winners[:8]:
    row = dprot.loc[dfn]; seq = row.sequence
    pc = consensus(pdw[dfn])
    tc = consensus(trim(fn_pwm[fn]))
    print(f"QUERY {fn_gene.get(fn,'?')} [{fn_fam.get(fn,'?')}]  target motif={tc}")
    print(f"  retrieved DPAC {dfn}: cos={cos:.3f}  motif(consensus)={pc}  n_motifs={int(row.n_motifs)}")
    print(f"    -> DPAC oracle-r={rd:.2f}  vs best-real oracle-r={rr:.2f}  (gap {rd-rr:+.2f})")
    print(f"    DPAC protein len={len(seq)}aa : {seq[:90]}{'...' if len(seq)>90 else ''}")
    print()

# overall DPAC protein-length profile + similarity-to-real-DBD profile
ln = dprot.sequence.str.len()
maxcos = (Dn @ Rn.T).max(1)
print("DPAC protein length:   median=%d  <=120aa(DBD-like)=%.0f%%  >300aa(full prot)=%.0f%%"
      % (ln.median(), 100*(ln<=120).mean(), 100*(ln>300).mean()))
print("DPAC max cosine to ANY real TF DBD:  median=%.3f  >=0.9(DBD-homolog)=%.0f%%  <0.7(off-fold)=%.0f%%"
      % (np.median(maxcos), 100*(maxcos>=0.9).mean(), 100*(maxcos<0.7).mean()))
