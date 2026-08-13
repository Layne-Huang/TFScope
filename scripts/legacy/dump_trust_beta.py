#!/usr/bin/env python
"""Dump the retrieval trust scores + β gate per test TF (v17 = v18a's frozen gate).

For each test TF: per-neighbour trust (sigmoid of trust_logits), max trust, and the
per-sample β_gated (how much the retrieval log-prior is weighted vs de-novo).
Also reports whether β tracks actual retrieval quality (top-1 neighbour r_gt).
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_200ep/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

df = pd.read_parquet(DATA); fn2gene = dict(zip(df["filename"], df["gene_symbol"].astype(str)))
idx = json.load(open(cfg.retrieval_index_path))
fn2pwm = {}
for _, r in df.iterrows():
    b = r["pwm"]; p = np.frombuffer(b, np.float32).reshape(4, -1) if isinstance(b, bytes) else np.full((4, 4), .25, np.float32)
    L = min(p.shape[1], 20); a = np.full((4, 20), .25, np.float32); a[:, :L] = p[:, :L]; fn2pwm[r["filename"]] = (a, L)

rows = []
with torch.no_grad():
    bi = 0
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
        _, _, aux = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                      retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                      retrieved_sims=b.get("retrieved_sims"))
        trust = torch.sigmoid(aux["trust_logits"]).cpu().numpy()        # (B,K)
        beta = aux["beta_gated"].cpu().numpy()                          # (B,)
        for j in range(trust.shape[0]):
            fn = ds.filenames[bi]; bi += 1
            tgt = b["target_pwm"][j].cpu().numpy(); L = int(b["pwm_mask"][j].sum())
            # top-1 neighbour aligned r vs truth (retrieval quality)
            nbrs = idx.get(fn, [])
            rq = np.nan
            if nbrs:
                npw, nL = fn2pwm[nbrs[0]["nn_filename"]]
                _, _, _, rq = align_pwm(npw[:, :nL], tgt[:, :L], max_shift=10)
            rows.append((fn, fn2gene.get(fn, "?"), trust[j], float(beta[j]), float(rq)))

# ---- per-TF table for the highlighted cases ----
def show(name):
    print(f"\n{name}:")
    for fn, g, tr, be, rq in rows:
        if name.upper() in g.upper():
            ts = " ".join(f"{x:.2f}" for x in tr)
            print(f"  {fn:42s} trust=[{ts}] maxT={tr.max():.2f}  β={be:.3f}  top1_r_gt={rq:.2f}")

for nm in ["Egr1", "MEF2A", "CTCF"]:
    show(nm)

# ---- overall stats ----
betas = np.array([r[3] for r in rows]); rqs = np.array([r[4] for r in rows]); maxT = np.array([r[2].max() for r in rows])
ok = ~np.isnan(rqs)
print(f"\n=== overall ({len(rows)} test TFs) ===")
print(f"  β: mean={betas.mean():.3f} median={np.median(betas):.3f} min={betas.min():.3f} max={betas.max():.3f}")
print(f"  frac β<0.05 (retrieval effectively off): {(betas<0.05).mean():.2f}")
print(f"  max-trust: mean={maxT.mean():.3f}")
print(f"  corr(β, top1_r_gt) = {pearsonr(betas[ok], rqs[ok])[0]:.3f}   (does β track retrieval quality?)")
print(f"  corr(max-trust, top1_r_gt) = {pearsonr(maxT[ok], rqs[ok])[0]:.3f}")
