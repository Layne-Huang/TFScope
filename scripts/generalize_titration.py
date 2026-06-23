"""Generalize the multi-mutant titration across families: for each family, automatically pick the
MOST motif-divergent well-predicted TF pair (lowest predicted-PWM corr), then titrate a recognition-
module swap (progressively substitute target residues into source) and measure how far the predicted
motif moves to the target. Tests whether 'resolution scales with determinant size' generalizes.
"""
import os, sys, json, pickle
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from Bio import Align
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import trimmed_core, aligned_cols
CK = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
cache = pickle.load(open("results/specificity_design/pwm_cache.pkl", "rb"))
d = pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet")
gene2row = {r.gene_symbol: r for r in d.drop_duplicates("gene_symbol").itertuples()}
FAMILIES = ["Nuclear_Receptor", "bZIP", "Homeodomain", "bHLH", "Forkhead", "ETS", "C2H2_medium"]
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CK) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to("cuda:0").eval(); m.load_state_dict(torch.load(CK, map_location="cuda:0", weights_only=False)["model"], strict=False)
COMP = np.array([3, 2, 1, 0])
@torch.no_grad()
def predP(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device="cuda:0"); dm = torch.ones(1, len(seq), dtype=torch.bool, device="cuda:0"); fi = torch.tensor([fid], device="cuda:0")
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    g = gl.sigmoid()[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy(); c = np.where(g > 0.5)[0]
    if len(c) < 4: ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    return p[:, c.min():c.max() + 1]
def ic_of(P): return float((2 + (np.clip(P, 1e-9, 1) * np.log2(np.clip(P, 1e-9, 1))).sum(0)).mean())
def pcorr(A, B):
    best = -1
    for Bx in (B, B[COMP][:, ::-1]):
        for off in range(-(B.shape[1] - 1), A.shape[1]):
            a0, b0 = max(0, off), max(0, -off); ov = min(A.shape[1] - a0, Bx.shape[1] - b0)
            if ov < 4: continue
            a = A[:, a0:a0 + ov].ravel(); b = Bx[:, b0:b0 + ov].ravel()
            if a.std() > 1e-9 and b.std() > 1e-9: best = max(best, np.corrcoef(a, b)[0, 1])
    return best
def corr_aln(A, B):
    co = trimmed_core(B, np.ones(B.shape[1], bool)); al, cols, _ = aligned_cols(A, co)
    if len(cols) < 4: return np.nan
    G = co[:, cols]; P = np.clip(al[:, cols], 1e-8, 1); P /= P.sum(0, keepdims=True)
    rs = [np.corrcoef(P[:, j], G[:, j])[0, 1] for j in range(len(cols)) if P[:, j].std() > 1e-8 and G[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan
def dbd_fid(g): r = gene2row[g]; return str(r.sequence)[int(r.dbd_start):int(r.dbd_end)], int(r.family_id)

aligner = Align.PairwiseAligner(); aligner.mode = "global"; aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -11; aligner.extend_gap_score = -1
def titrate(src, tgt):
    s_seq, fid = dbd_fid(src); t_seq, _ = dbd_fid(tgt)
    if not (15 <= len(s_seq) <= 200 and 15 <= len(t_seq) <= 200): return None
    s_pwm, t_pwm = predP(s_seq, fid), predP(t_seq, fid)
    aln = aligner.align(s_seq, t_seq)[0]; A, B = aln[0], aln[1]
    si = -1; diffs = []
    for a, b in zip(A, B):
        if a != "-": si += 1
        if a != "-" and b != "-" and a != b: diffs.append((si, b))
    nd = len(diffs); tit = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k = int(round(frac * nd)); chim = list(s_seq)
        for si_, b in diffs[:k]: chim[si_] = b
        P = predP("".join(chim), fid)
        tit.append(dict(frac=frac, corr_src=corr_aln(P, s_pwm), corr_tgt=corr_aln(P, t_pwm)))
    return dict(src=src, tgt=tgt, ndiff=nd, titration=tit)

out = []
for fam in FAMILIES:
    genes = [g for g in cache if cache[g]["family"] == fam and ic_of(cache[g]["pred"]) > 0.8]
    if len(genes) < 2: print(f"{fam}: <2 well-predicted genes"); continue
    # most-divergent pair by predicted-PWM corr (sample to bound cost)
    import itertools
    genes = genes[:40]
    best = None
    for a, b in itertools.combinations(genes, 2):
        c = pcorr(cache[a]["pred"], cache[b]["pred"])
        if best is None or c < best[0]: best = (c, a, b)
    c0, a, b = best
    if c0 > 0.6: print(f"{fam}: most-divergent pair {a}/{b} still similar (corr {c0:.2f}) — no meaningful switch"); continue
    r = titrate(a, b)
    if r is None: print(f"{fam}: {a}/{b} dbd length issue"); continue
    r["family"] = fam; r["wt_corr"] = round(float(c0), 2); out.append(r)
    cr = [t["corr_tgt"] for t in r["titration"]]
    cross = next((int(t["frac"] * 100) for t in r["titration"] if t["corr_tgt"] >= 0.7), None)
    print(f"{fam}: {a}->{b} (WT corr {c0:.2f}, {r['ndiff']} diffs) corr-to-target 0%={cr[0]:.2f} -> 100%={cr[-1]:.2f}  cross0.7@{cross}%")
json.dump(out, open("results/myod1_mut/generalize_titration.json", "w"), indent=1)
print(f"\nsaved results/myod1_mut/generalize_titration.json ({len(out)} families with a meaningful switch)")
