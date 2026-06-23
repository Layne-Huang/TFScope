"""Does TFScope resolve specificity switches better as MORE specificity-determining residues are
swapped? Titration: progressively substitute target-receptor residues into a source receptor and
measure how far the predicted motif moves from source toward target. Pairs with clearly different
motifs (GR/AR GRE 'AGAACA' <-> ER ERE 'AGGTCA'). Combined no-RAG model.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from Bio import Align
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import trimmed_core, aligned_cols
CK = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CK) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to("cuda:0").eval(); m.load_state_dict(torch.load(CK, map_location="cuda:0", weights_only=False)["model"], strict=False)
@torch.no_grad()
def predP(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device="cuda:0"); dm = torch.ones(1, len(seq), dtype=torch.bool, device="cuda:0"); fi = torch.tensor([fid], device="cuda:0")
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    g = gl.sigmoid()[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy(); c = np.where(g > 0.5)[0]
    if len(c) < 4: ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    return p[:, c.min():c.max() + 1]
def cons(P): return "".join("ACGT"[i] for i in P.argmax(0))
def corr(A, B):
    co = trimmed_core(B, np.ones(B.shape[1], bool)); al, cols, _ = aligned_cols(A, co)
    if len(cols) < 4: return np.nan
    G = co[:, cols]; P = np.clip(al[:, cols], 1e-8, 1); P /= P.sum(0, keepdims=True)
    rs = [np.corrcoef(P[:, j], G[:, j])[0, 1] for j in range(len(cols)) if P[:, j].std() > 1e-8 and G[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan
d = pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet")
def get(g): r = d[d.gene_symbol == g].iloc[0]; return str(r.sequence)[int(r.dbd_start):int(r.dbd_end)], int(r.family_id)

al = Align.PairwiseAligner(); al.mode = "global"; al.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
al.open_gap_score = -11; al.extend_gap_score = -1
def titrate(src, tgt):
    s_seq, fid = get(src); t_seq, _ = get(tgt)
    s_pwm, t_pwm = predP(s_seq, fid), predP(t_seq, fid)
    aln = al.align(s_seq, t_seq)[0]; A, B = aln[0], aln[1]      # aligned strings w/ gaps
    # differing aligned columns -> map to source index + target residue
    si = -1; diffs = []
    for a, b in zip(A, B):
        if a != "-": si += 1
        if a != "-" and b != "-" and a != b: diffs.append((si, b))
    ndiff = len(diffs)
    print(f"\n=== {src}->{tgt}: {ndiff} differing residues; src={cons(s_pwm)} tgt={cons(t_pwm)} ===")
    rows = []
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        k = int(round(frac * ndiff)); chim = list(s_seq)
        for si_, b in diffs[:k]: chim[si_] = b
        P = predP("".join(chim), fid)
        rs, rt = corr(P, s_pwm), corr(P, t_pwm)
        rows.append((frac, k, cons(P), rs, rt))
        print(f"  swap {int(frac*100):>3}% ({k:>2}/{ndiff} res): {cons(P):<14} corr(src)={rs:+.2f} corr(tgt)={rt:+.2f}  {'->TARGET' if rt>rs+0.1 else '~src'}")
    return dict(src=src, tgt=tgt, ndiff=ndiff, src_cons=cons(s_pwm), tgt_cons=cons(t_pwm),
                titration=[dict(frac=f, k=k, cons=c, corr_src=rs, corr_tgt=rt) for f, k, c, rs, rt in rows])

res = [titrate("NR3C1", "ESR1"), titrate("AR", "ESR1"), titrate("ESR1", "NR3C1")]
json.dump(res, open("results/myod1_mut/multimutant_titration.json", "w"), indent=1)
print("\nsaved results/myod1_mut/multimutant_titration.json")
