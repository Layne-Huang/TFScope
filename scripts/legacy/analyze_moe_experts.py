"""Deeper analysis of the per-residue MoE experts.

Two questions the basic routing diagnostic can't answer:
  (A) WHAT does each expert recognize?  -> per-expert amino-acid enrichment:
      for tokens routed to expert e, log2( P(aa | e) / P(aa) ). If e7 is enriched
      for C/H (zinc-coordinating) that is genuine recognition chemistry, not a label.
  (B) Is specialization REAL chemistry or just the family label re-encoded?
      The router input is [token_feat || family_emb] + cos(family_emb, prototypes),
      so it is HANDED the family. Ablation: re-route every protein with family_id
      held CONSTANT (all -> "Other"). If experts still fire differentially by residue
      content, routing is token/chemistry-driven. If NMI(expert;true_family) collapses
      toward 0, it was the family-embedding bias doing the work.

Usage: python scripts/analyze_moe_experts.py <CKPT_DIR> [ckpt] [n_per_family]
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKDIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42"
CKNAME = sys.argv[2] if len(sys.argv) > 2 else "ckpt_best.pt"
N_PER_FAM = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKDIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
E = cfg.num_experts
assert getattr(cfg, "moe_granularity", "protein") == "residue"
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKDIR, CKNAME), map_location=dev, weights_only=False)["model"], strict=False)

TOK2AA = {v: k for k, v in AA_TO_TOKEN.items()}
df = pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet")
if "dbd_start" in df.columns:
    df = df.assign(dbdseq=[str(s)[int(a):int(b)] for s, a, b in zip(df.sequence, df.dbd_start, df.dbd_end)])
else:
    df["dbdseq"] = df.sequence.astype(str)
df = df.drop_duplicates("dbdseq")
samp = pd.concat([g.sample(min(N_PER_FAM, len(g)), random_state=0) for _, g in df.groupby("family_name")])
fams = sorted(samp.family_name.unique()); fam2i = {f: i for i, f in enumerate(fams)}
print(f"proteins {len(samp)}  experts {E}")

@torch.no_grad()
def route(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([int(fid)], device=dev)
    _, _, aux = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return aux["top_indices"][:, 0].cpu().numpy()   # (Ntok,) top-1 expert, in residue order

# collect (expert, aa, family) for true-family and constant-family routing
AAs = list("ACDEFGHIKLMNPQRSTVWY")
CONST_FID = int(df[df.family_name == "Other"].family_id.mode().iloc[0])   # route all as "Other"
rec = {"true": [], "const": []}
aa_of, fam_of = [], []
for r in samp.itertuples():
    seq = r.dbdseq
    if not isinstance(seq, str) or len(seq) < 4: continue
    e_true = route(seq, r.family_id)
    e_const = route(seq, CONST_FID)
    n = min(len(e_true), len(seq))
    rec["true"].append(e_true[:n]); rec["const"].append(e_const[:n])
    aa_of.append(np.array([c for c in seq[:n]])); fam_of.append(np.full(n, fam2i[r.family_name]))
et = np.concatenate(rec["true"]); ec = np.concatenate(rec["const"])
aa = np.concatenate(aa_of); fam = np.concatenate(fam_of); N = len(et)

def nmi(expert, label, nlab):
    J = np.zeros((nlab, E))
    for l in range(nlab): J[l] = np.bincount(expert[label == l], minlength=E)
    J = J / J.sum(); pl = J.sum(1, keepdims=True); pe = J.sum(0, keepdims=True); nz = J > 0
    I = (J[nz] * np.log(J[nz] / (pl @ pe)[nz])).sum()
    H = -(pl[pl > 0] * np.log(pl[pl > 0])).sum()
    return I / max(H, 1e-9), I

# ── (B) family-bias ablation ──────────────────────────────────────────────────
nmi_true_fam, _ = nmi(et, fam, len(fams))
nmi_const_fam, _ = nmi(ec, fam, len(fams))
# AA index labels for NMI(expert; aa)
aai = {c: i for i, c in enumerate(AAs)}
aa_lab = np.array([aai.get(c, 0) for c in aa])
nmi_true_aa, _ = nmi(et, aa_lab, len(AAs))
nmi_const_aa, _ = nmi(ec, aa_lab, len(AAs))
frac_same = float((et == ec).mean())
print("\n=== (B) Is routing chemistry or family-label? ===")
print(f"  NMI(expert; family): true-family input = {nmi_true_fam:.3f}  |  CONSTANT-family input = {nmi_const_fam:.3f}")
print(f"  NMI(expert; amino-acid): true = {nmi_true_aa:.3f}  |  constant-family = {nmi_const_aa:.3f}")
print(f"  routing unchanged when family label removed: {frac_same:.2f} of tokens keep the same top-1 expert")
print("  -> high const-family NMI(family) & NMI(aa), high frac_same => specialization is TOKEN/CHEMISTRY-driven,")
print("     not merely the family embedding being re-encoded.")

# ── (A) per-expert amino-acid enrichment (true-family routing) ─────────────────
bg = np.array([(aa == c).mean() for c in AAs]) + 1e-9
print("\n=== (A) what each expert recognizes — top log2 AA enrichment (true routing) ===")
prof = {}
for e in range(E):
    sel = et == e
    if sel.sum() < 30:
        print(f"  e{e}: (n={sel.sum()}) too few"); continue
    pe = np.array([(aa[sel] == c).mean() for c in AAs]) + 1e-9
    enr = np.log2(pe / bg)
    order = np.argsort(-enr)[:5]
    prof[e] = {AAs[i]: round(float(enr[i]), 2) for i in order}
    top = "  ".join(f"{AAs[i]}{enr[i]:+.1f}" for i in order)
    print(f"  e{e} (n={sel.sum():5d}, usage {sel.mean():.3f}): {top}")

out = {"ckpt": f"{CKDIR}/{CKNAME}", "n_tokens": int(N),
       "nmi_true_family": round(float(nmi_true_fam), 3), "nmi_const_family": round(float(nmi_const_fam), 3),
       "nmi_true_aa": round(float(nmi_true_aa), 3), "nmi_const_aa": round(float(nmi_const_aa), 3),
       "frac_routing_unchanged_no_family": round(frac_same, 3),
       "expert_aa_enrichment": prof}
os.makedirs("results/residue_moe_cases", exist_ok=True)
json.dump(out, open("results/residue_moe_cases/expert_analysis.json", "w"), indent=1)
print("\nsaved results/residue_moe_cases/expert_analysis.json")
