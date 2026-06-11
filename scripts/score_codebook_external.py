#!/usr/bin/env python
"""Honest external validation against Codebook expert-curated motifs.

Clean subset only: Codebook genes that fall in cluster40's HELD-OUT test
partition (held out at 40% identity; never seen by the cluster40 checkpoint).
For each, score TFScope's sequence-only prediction against the INDEPENDENT
Codebook experimental motif, and also score our own parquet target vs Codebook
(database-concordance ceiling) to show how much agreement is real.

Out: results/codebook_external/scores.json + per-gene arrays for figures.
"""
import os, sys, glob, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.alignment import align_pwm

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
CKPT  = f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_best.pt"
DATA  = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"
MDIR  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/codebook/Expert_curated_PWMs_Hugheslab/motifs"
OUT   = "results/codebook_external"; os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
MAXL = 20

# ── parse Codebook motifs ─────────────────────────────────────────────────────
def parse_codebook(path):
    rows = []
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if not p or p[0] in ("TF", "Motif", "Pos") or len(p) < 5:
            continue
        try: rows.append([float(x) for x in p[1:5]])
        except ValueError: continue
    if not rows: return None
    m = np.array(rows).T            # (4, L)  rows A,C,G,T
    m = np.clip(m, 1e-8, 1); m = m / m.sum(0, keepdims=True)
    return m

cb_motifs = {}
for f in glob.glob(MDIR + "/*.txt"):
    g = os.path.basename(f).split("_PWM")[0].upper()
    m = parse_codebook(f)
    if m is not None: cb_motifs[g] = m
print(f"parsed {len(cb_motifs)} Codebook motifs")

# ── clean subset: cluster40 test genes ∩ Codebook, NOT in cluster40 train ─────
df = pd.read_parquet(DATA); df['g'] = df['gene_symbol'].astype(str).str.upper()
fn2gene = dict(zip(df['filename'], df['g']))
sp = json.load(open(SPLIT))
train_genes = {fn2gene.get(f, '') for f in sp['train']}
test_fns = sp['test']
clean_genes = {fn2gene.get(f, '') for f in test_fns} & set(cb_motifs) - train_genes
print(f"clean held-out Codebook genes: {len(clean_genes)}")

# ── run cluster40 checkpoint on the cluster40 test set ────────────────────────
def load_model(ckpt):
    cfg = TFScopeConfig()
    cp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.isfile(cp):
        for k, v in json.load(open(cp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval(); return m, cfg

m, cfg = load_model(CKPT)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
P, T, G = [], [], []
gate_all = []
with torch.no_grad():
    for b in ld:
        b = {k: v.to(device, dtype=(torch.float32 if v.is_floating_point() else torch.long))
             for k, v in b.items()}
        gl, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                      retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                      retrieved_sims=b.get("retrieved_sims"), recog_prior=b.get("recog_prior"))
        P.append(F.softmax(pw, 1).cpu().numpy())
        T.append(b["target_pwm"].cpu().numpy())
        gate_all.append(torch.sigmoid(gl).cpu().numpy())
P = np.concatenate(P); T = np.concatenate(T); gate_all = np.concatenate(gate_all)
fns = list(ds.filenames)

# ── score each clean gene vs Codebook ─────────────────────────────────────────
def score(pred_full, gate, target_cb):
    gm = gate > 0.5
    if gm.sum() < 2: gm = np.ones(pred_full.shape[1], bool)
    pv = pred_full[:, gm]
    L = target_cb.shape[1]
    al, _, _, _ = align_pwm(pv, target_cb, max_shift=10, consider_revcomp=True)
    Lc = min(al.shape[1], L)
    return float(np.nanmean([pearsonr(target_cb[:, j], al[:, j])[0] for j in range(Lc)]))

rows = []
arrays = {}
seen = set()
for i, fn in enumerate(fns):
    g = fn2gene.get(fn, '')
    if g not in clean_genes or g in seen: continue
    seen.add(g)
    cb = cb_motifs[g]
    r_pred = score(P[i], gate_all[i], cb)
    # database-concordance ceiling: our parquet target vs Codebook
    Lt = int((T[i].sum(0) > 1e-6).sum())
    tt = T[i][:, :Lt] if Lt >= 2 else T[i]
    al_t, _, _, _ = align_pwm(tt, cb, max_shift=10, consider_revcomp=True)
    Lc = min(al_t.shape[1], cb.shape[1])
    r_ceil = float(np.nanmean([pearsonr(cb[:, j], al_t[:, j])[0] for j in range(Lc)]))
    rows.append({"gene": g, "r_tfscope_vs_codebook": r_pred,
                 "r_dbtarget_vs_codebook": r_ceil, "cb_len": int(cb.shape[1])})
    arrays[f"{g}_codebook"] = cb
    arrays[f"{g}_pred"] = P[i]
    arrays[f"{g}_gate"] = gate_all[i]

rows.sort(key=lambda r: -r["r_tfscope_vs_codebook"])
rp = np.array([r["r_tfscope_vs_codebook"] for r in rows])
rc = np.array([r["r_dbtarget_vs_codebook"] for r in rows])
summary = {
    "n_clean_genes": len(rows),
    "tfscope_vs_codebook_mean": float(np.nanmean(rp)),
    "tfscope_vs_codebook_median": float(np.nanmedian(rp)),
    "db_concordance_ceiling_mean": float(np.nanmean(rc)),
    "db_concordance_ceiling_median": float(np.nanmedian(rc)),
    "frac_tfscope_r_gt_0.5": float(np.mean(rp > 0.5)),
    "per_gene": rows,
}
json.dump(summary, open(f"{OUT}/scores.json", "w"), indent=2)
np.savez(f"{OUT}/arrays.npz", **arrays)

print("\n=== Honest external validation (cluster40-held-out ∩ Codebook) ===")
print(f"clean genes: {len(rows)}")
print(f"TFScope vs Codebook:        mean r = {summary['tfscope_vs_codebook_mean']:.3f}  "
      f"median = {summary['tfscope_vs_codebook_median']:.3f}")
print(f"DB-concordance ceiling:     mean r = {summary['db_concordance_ceiling_mean']:.3f}  "
      f"median = {summary['db_concordance_ceiling_median']:.3f}")
print(f"frac TFScope r > 0.5: {summary['frac_tfscope_r_gt_0.5']*100:.0f}%")
print("\ntop 12 genes (TFScope vs Codebook):")
for r in rows[:12]:
    print(f"  {r['gene']:<12} pred={r['r_tfscope_vs_codebook']:+.3f}  ceiling={r['r_dbtarget_vs_codebook']:+.3f}")
print(f"\nsaved {OUT}/scores.json + arrays.npz")
