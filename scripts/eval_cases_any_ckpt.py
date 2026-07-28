"""Test any TFScope checkpoint on the two case-study analyses:
  (1) MyoD1 L112R directional specificity-switch score  Δ_switch  (Fig 4a metric)
  (2) 4 designed DBPs (DBP005/009/006/035): Pearson r between TFScope predicted
      per-base relative-binding matrix and the experimental SELEX matrix
      (the corrected-sign metric used in figures/figure_dbp_heatmap).

Usage:  python scripts/eval_cases_any_ckpt.py <CKPT_DIR> [ckpt_name]
Builds the model from <CKPT_DIR>/config.json, so it works for both the pooled
(protein-mode) and the per-residue MoE checkpoints.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKDIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42"
CKNAME = sys.argv[2] if len(sys.argv) > 2 else "ckpt_best.pt"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKDIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
print(f"[ckpt] {CKDIR}/{CKNAME}  moe_granularity={getattr(cfg,'moe_granularity','protein')}")
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKDIR, CKNAME), map_location=dev, weights_only=False)["model"], strict=False)

B = {"A": 0, "C": 1, "G": 2, "T": 3}; BA = np.array(list("ACGT")); VMAX = 2.5

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()

def rc(s):
    if isinstance(s, str): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
    return s[[3, 2, 1, 0]][:, ::-1]           # reverse-complement a (4,W) PWM

# ── (1) MyoD1 L112R Δ_switch ──────────────────────────────────────────────────
WT_DBD = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MUT_DBD = WT_DBD[:11] + "R" + WT_DBD[12:]
assert WT_DBD[11] == "L" and MUT_DBD[11] == "R"
FID_BHLH = 3

def score(P, seq):
    lo = np.log2(np.clip(P, 1e-6, 1) / 0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B[c] for c in s]
        for off in range(0, W - L + 1):
            best = max(best, float(sum(lo[idx[j], off + j] for j in range(L))))
    return best

_, pw = predict(WT_DBD, FID_BHLH); _, pm = predict(MUT_DBD, FID_BHLH)
S = {"WT": {"CACGTG": score(pw, "CACGTG"), "CACCTG": score(pw, "CACCTG")},
     "mut": {"CACGTG": score(pm, "CACGTG"), "CACCTG": score(pm, "CACCTG")}}
dWT = S["WT"]["CACGTG"] - S["WT"]["CACCTG"]; dMUT = S["mut"]["CACGTG"] - S["mut"]["CACCTG"]
d_switch = dMUT - dWT
print("\n=== (1) MyoD1 L112R directional switch ===")
print(f"WT : S(CACGTG)={S['WT']['CACGTG']:.2f} S(CACCTG)={S['WT']['CACCTG']:.2f}  d={dWT:+.2f}")
print(f"mut: S(CACGTG)={S['mut']['CACGTG']:.2f} S(CACCTG)={S['mut']['CACCTG']:.2f}  d={dMUT:+.2f}")
print(f"Δ_switch = {d_switch:+.2f}  -> {'switch reproduced (>0)' if d_switch > 0 else 'NOT reproduced'}")

# ── (2) designed DBPs: r(TFScope, experimental) ───────────────────────────────
WT = "GCAGATCTGCACAT"; L = len(WT); T = np.eye(4)[[B[c] for c in WT]].T
ORDER = ["DBP005", "DBP009", "DBP006", "DBP035"]
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP009": "Extended_Data_Figure_1_E_DBP009",
         "DBP006": "Extended_Data_Figure_1_D_DBP006", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
VAL = "Median PE/FITC (Normalized)"; WTOVERRIDE = {"DBP006": 0.1202}
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
dff = pd.read_parquet("data/processed/tf_pwm_deeppbs_only_canon_trim.parquet")
fn = dff["filename"].astype(str).str.upper()
def fidf(g):
    s = dff[fn.str.contains(g.upper())]; return int(s["family_id"].mode().iloc[0]) if len(s) else 4

def exp_rel(d):
    df = pd.ExcelFile(XLS).parse(SHEET[d]); wt = df[df.position.astype(str) == "WT"]
    wtv = WTOVERRIDE.get(d) or (float(wt[VAL].iloc[0]) if len(wt) and not pd.isna(wt[VAL].iloc[0]) else None)
    dd = df[df.position.astype(str).str.isdigit()].copy(); dd["position"] = dd.position.astype(int)
    R = np.zeros((4, L))
    for p in range(1, L + 1):
        sub = dd[dd.position == p]; wv = wtv if wtv is not None else float(sub[VAL].median())
        for _, r in sub.iterrows(): R[B[str(r.new_base)], p - 1] = np.log2(float(r[VAL]) / wv)
    return np.clip(R, -VMAX, VMAX)

def tf_rel(seq, fid):
    gate, P = predict(seq, fid); W = P.shape[1]
    gcols = np.where(gate > 0.5)[0]
    if len(gcols) < 4:
        ic = (P * np.log2(P + 1e-9)).sum(0) + 2; c = ic.argmax(); gcols = np.arange(max(0, c - 4), min(W, c + 5))
    lo, hi = gcols.min(), gcols.max() + 1; klen = hi - lo
    best = (-1e9, None, None, None, None)
    for strand in ("+", "-"):
        Q = P if strand == "+" else rc(P); clo = lo if strand == "+" else (W - hi)
        core = Q[:, clo:clo + klen]
        for coff in range(-(klen - 1), L):
            sc = sum(float(core[:, j] @ T[:, coff + j]) for j in range(klen) if 0 <= coff + j < L)
            if sc > best[0]: best = (sc, strand, coff, Q, clo)
    _, strand, coff, Q, clo = best
    full = np.full((4, L), 0.25)
    for p in range(L):
        c = clo + (p - coff)
        if 0 <= c < W: full[:, p] = Q[:, c]
    R = np.zeros((4, L))
    for p in range(L):
        wb = B[WT[p]]; R[:, p] = np.log2((full[wb, p] + 1e-6) / (full[:, p] + 1e-6))
    return np.clip(R, -VMAX, VMAX)

print("\n=== (2) designed DBPs — r(TFScope vs experimental SELEX) ===")
rs = []
for d in ORDER:
    E = exp_rel(d); Rtf = tf_rel(e2[d]["prot_seq"], fidf(str(e2[d].get("top_donor", "POU2F1"))))
    # correlate off-WT-base entries (WT entries are 0 by construction)
    mask = np.ones((4, L), bool)
    for p in range(L): mask[B[WT[p]], p] = False
    r = float(np.corrcoef(E[mask], Rtf[mask])[0, 1]); rs.append(r)
    print(f"  {d}: r = {r:+.3f}")
print(f"  mean r = {np.mean(rs):+.3f}")

out = {"ckpt": f"{CKDIR}/{CKNAME}", "moe_granularity": getattr(cfg, "moe_granularity", "protein"),
       "myod1_delta_switch": round(float(d_switch), 3),
       "dbp_r": {d: round(r, 3) for d, r in zip(ORDER, rs)}, "dbp_r_mean": round(float(np.mean(rs)), 3)}
os.makedirs("results/residue_moe_cases", exist_ok=True)
tag = "residue" if out["moe_granularity"] == "residue" else "combined"
json.dump(out, open(f"results/residue_moe_cases/cases_{tag}.json", "w"), indent=1)
print(f"\nsaved results/residue_moe_cases/cases_{tag}.json")
