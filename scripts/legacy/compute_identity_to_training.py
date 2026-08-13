"""For each held-out factor: per-record motif recovery r + % DBD identity to the nearest
training factor (max over the 16 ESM-nearest training neighbours; Biopython global, BLOSUM62).
Out: results/fig3a_heldout/recovery_vs_identity.npz (fn, r, maxid)
"""
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from Bio import Align
from eval_full_metrics import trimmed_core, aligned_cols

idx = json.load(open("data/processed/tf_nn_index_cluster40_clean.json"))
d = pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet", columns=["filename", "sequence", "dbd_start", "dbd_end"])
d["fn"] = d.filename.astype(str)
dbd_by = {r.fn: str(r.sequence)[int(r.dbd_start):int(r.dbd_end)] for r in d.itertuples()}

aligner = Align.PairwiseAligner(); aligner.mode = "global"
aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -11; aligner.extend_gap_score = -1
def pid(a, b):
    if not a or not b: return 0.0
    aln = aligner.align(a, b)[0]; A, B = aln[0], aln[1]
    nid = sum(x == y and x != "-" for x, y in zip(A, B))
    return 100.0 * nid / min(len(a), len(b))

def colr(A, B):
    rs = [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])
          if A[:, j].std() > 1e-8 and B[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan

dd = np.load("results/fig3a_heldout/combined_heldout_predictions.npz", allow_pickle=True)
fns, rs, ids = [], [], []
for i in range(len(dd["filename"])):
    fn = str(dd["filename"][i])
    gt = trimmed_core(dd["target"][i], dd["mask"][i] > 0.5)
    if gt is None or gt.shape[1] < 4: continue
    al, cols, _ = aligned_cols(dd["prediction"][i], gt)
    if len(cols) < 4: continue
    G = gt[:, cols]; P = np.clip(al[:, cols], 1e-8, 1); P = P / P.sum(0, keepdims=True)
    r = colr(P, G)
    if r != r: continue
    qa = dbd_by.get(fn, "")
    nn = [pid(qa, dbd_by.get(n["nn_filename"], "")) for n in idx.get(fn, [])[:16]]
    fns.append(fn); rs.append(r); ids.append(max(nn) if nn else np.nan)
np.savez("results/fig3a_heldout/recovery_vs_identity.npz",
         fn=np.array(fns), r=np.array(rs), maxid=np.array(ids))
R, I = np.array(rs), np.array(ids); ok = ~np.isnan(I)
from scipy.stats import spearmanr
print(f"n={len(rs)}  median r={np.median(R):.3f}  median id={np.nanmedian(I):.1f}%  "
      f"frac<40%={(I[ok] < 40).mean():.2f}  spearman(r,id)={spearmanr(R[ok], I[ok])[0]:.3f} "
      f"p={spearmanr(R[ok], I[ok])[1]:.3f}")
print(f"median r <40%id={np.median(R[ok & (I < 40)]):.3f}  >=40%id={np.median(R[ok & (I >= 40)]):.3f}")
