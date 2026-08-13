"""Fig 2b — in-silico alanine scan over the WHOLE cluster40 test set (84 TFs).

For each test TF: predict baseline PWM, then mutate each DBD residue -> Ala and
recompute the PWM; residue importance = L1(PWM_wt - PWM_mut) over the 4x20 motif.
Ground-truth recognition residues (structure-derived) from recognition_residues.json.

Population statistic: per-TF AUROC(importance -> is-recognition-residue), plus pooled
importance at recognition vs background positions. Saves per-TF tracks for example panels.
"""
import os, sys, json, re
import numpy as np, torch, torch.nn.functional as F
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"   # /n/holylabs unreachable on this host
sys.path.insert(0, "src")
import pandas as pd
from sklearn.metrics import roc_auc_score
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
RECOG = "data/contact_maps/true_recognition_residues.json"   # structural base contacts (Fig 2b GT)
OUT = "results/per_family_v24_ens"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ALA = AA_TO_TOKEN.get("A", 4)

# ---- model ----
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
_ENSC=[CKPT]+[f"checkpoints/iclr_phase1/v24_ens/seed{s}/ckpt_best.pt" for s in (1,7,13,23)]
MODELS=[]
for _ck in _ENSC:
    _mm=TFScopeModel(cfg).to(dev).eval(); _mm.load_state_dict(torch.load(_ck,map_location=dev,weights_only=False)["model"],strict=False); MODELS.append(_mm)
print(f"ensemble members: {len(MODELS)}")

# ---- data ----
test = set(json.load(open(SPLIT))["test"])
df = pd.read_parquet(PARQ)
df = df[df.filename.astype(str).isin(test)].reset_index(drop=True)
recog = json.load(open(RECOG))
base = lambda fn: re.sub(r"\.txt(__os\d+)?$", "", str(fn))

@torch.no_grad()
def pwm_batch(seqs, fid):
    L = len(seqs[0])
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in s] for s in seqs],
                       dtype=torch.long, device=dev)
    dm = torch.ones(len(seqs), L, dtype=torch.bool, device=dev)
    fi = torch.full((len(seqs),), fid, dtype=torch.long, device=dev)
    out = []
    for i in range(0, len(seqs), 48):           # chunk for memory
        _acc=None
        for _m in MODELS:
            gl, pl, _ = _m(tok[i:i+48], dm[i:i+48], fi[i:i+48],
                           retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
            _p=F.softmax(pl,1).cpu().numpy(); _acc=_p if _acc is None else _acc+_p
        out.append(_acc/len(MODELS))   # (b,4,20) ensemble mean
    return np.concatenate(out, 0)

rows = []
for r in df.itertuples():
    seq = r.sequence
    L = len(seq)
    fid = int(r.family_id)
    rr = [i for i in recog.get(base(r.filename), []) if i < L]
    # compute importance for EVERY test TF (GT may be empty for the 6 missing-PDB cases)
    # WT + L alanine mutants in one batch
    seqs = [seq] + [seq[:i] + "A" + seq[i+1:] for i in range(L)]
    pwms = pwm_batch(seqs, fid)
    wt = pwms[0]
    imp = np.abs(pwms[1:] - wt[None]).reshape(L, -1).sum(1)   # (L,) importance
    # alanine-already positions -> undefined (set nan, exclude)
    is_ala = np.array([seq[i] == "A" for i in range(L)])
    imp[is_ala] = np.nan
    y = np.zeros(L, int); y[rr] = 1
    valid = ~np.isnan(imp)
    auc = np.nan
    if valid.sum() > 1 and 0 < y[valid].sum() < valid.sum():
        auc = roc_auc_score(y[valid], imp[valid])
    rows.append(dict(filename=str(r.filename), gene=str(r.gene_symbol),
                     family=str(r.family_name), L=L, recog=rr,
                     imp=imp.tolist(), auc=float(auc) if auc==auc else None))
    print(f"{r.gene_symbol:12s} {r.family_name:16s} L={L:3d} recog={len(rr):2d} AUROC={auc:.3f}", flush=True)

# ---- population stats ----
aucs = np.array([x["auc"] for x in rows if x["auc"] is not None])
pooled_rec, pooled_bg = [], []
for x in rows:
    imp = np.array(x["imp"], float); y = np.zeros(x["L"], int)
    for i in x["recog"]: y[i] = 1
    v = ~np.isnan(imp)
    pooled_rec += imp[v & (y == 1)].tolist()
    pooled_bg += imp[v & (y == 0)].tolist()
from scipy.stats import mannwhitneyu
U, p = mannwhitneyu(pooled_rec, pooled_bg, alternative="greater")
summary = dict(n_tf=len(rows), n_auc=len(aucs),
               auc_mean=float(aucs.mean()), auc_median=float(np.median(aucs)),
               auc_frac_above_0p5=float((aucs > 0.5).mean()),
               pooled_recog_median=float(np.median(pooled_rec)),
               pooled_bg_median=float(np.median(pooled_bg)),
               mannwhitney_p=float(p))
json.dump(dict(summary=summary, rows=rows),
          open(f"{OUT}/alascan_population.json", "w"))
print("\n=== POPULATION ===")
print(json.dumps(summary, indent=1))
print(f"saved {OUT}/alascan_population.json")
