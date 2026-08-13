"""Scoped bZIP dimer probe (half-site decomposition).
Does TFScope fail bZIP at the HALF-SITE (recognition) or the FULL PALINDROME
(dimeric assembly)? For each bZIP test TF: align predicted vs GT motif, then
compare half-site r vs full-site r, and predicted vs GT palindromicity.
  half-site ✅ / full ❌  -> dimer assembly is the problem (two-seq input worth it)
  half-site ❌ too        -> recognition/motif-incoherence (dimer rewrite won't help)
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import trimmed_core, aligned_cols

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def colr(A, B):
    rs = [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])
          if A[:, j].std() > 1e-8 and B[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else float("nan")

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

sp = set(json.load(open(SPLIT))["test"])
df = pd.read_parquet(PARQ)
df = df[df.filename.astype(str).isin(sp) & (df.family_name == "bZIP")].reset_index(drop=True)

print(f"{'TF':<22} {'full_r':>7} {'half_r':>7} {'gt_pal':>7} {'pred_pal':>8}  verdict")
print("-" * 75)
rows = []
for r in df.itertuples():
    pwm_gt = np.frombuffer(r.pwm, dtype=np.float32).reshape(4, -1).astype(float)
    mask = np.ones(pwm_gt.shape[1], bool)
    gt = trimmed_core(pwm_gt, mask)
    if gt is None or gt.shape[1] < 4: continue
    pv = predict(r.sequence, int(r.family_id))
    aligned, cols, _ = aligned_cols(pv, gt)
    if len(cols) < 4: continue
    G = gt[:, cols]; P = np.clip(aligned[:, cols], 1e-8, 1); P = P / P.sum(0, keepdims=True)
    L = G.shape[1]; h = L // 2
    full_r = colr(P, G)
    # half-site r: best of the two halves (does it get at least one half-site?)
    lr = colr(P[:, :h], G[:, :h]); rr = colr(P[:, L - h:], G[:, L - h:])
    half_r = np.nanmax([lr, rr])
    gt_pal = colr(G[:, :h], revcomp_pwm_np(G[:, L - h:]))      # GT palindromicity
    pred_pal = colr(P[:, :h], revcomp_pwm_np(P[:, L - h:]))    # pred palindromicity
    v = "DIMER-assembly" if (half_r - full_r > 0.15 and pred_pal < gt_pal - 0.15) else \
        ("recognition" if half_r < 0.45 else "ok-ish")
    print(f"{r.gene_symbol+' '+r.filename[:6]:<22} {full_r:>7.3f} {half_r:>7.3f} {gt_pal:>7.3f} {pred_pal:>8.3f}  {v}")
    rows.append(dict(tf=r.gene_symbol, full_r=full_r, half_r=half_r, gt_pal=gt_pal, pred_pal=pred_pal))

import numpy as np
agg = lambda k: round(float(np.nanmean([x[k] for x in rows])), 3)
print("\n=== mean over bZIP test TFs ===")
print(f"full_r={agg('full_r')}  half_r={agg('half_r')}  gt_palindromicity={agg('gt_pal')}  pred_palindromicity={agg('pred_pal')}")
print("\nDECISION: half_r >> full_r AND pred_pal << gt_pal  -> dimer-assembly failure (two-seq worth it)")
print("          half_r also low                            -> recognition/motif-incoherence (dimer won't help)")
json.dump(rows, open("results/per_family/bzip_dimer_probe.json", "w"), indent=1)
print("saved results/per_family/bzip_dimer_probe.json")
