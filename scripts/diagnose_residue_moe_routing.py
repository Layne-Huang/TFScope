"""Does the per-residue MoE collapse (uniform routing) or specialize?

Runs the residue-MoE over proteins, collects per-DBD-token routing (top-1/top-2
expert + gate distribution) from model aux, and reports:
  - marginal expert usage        (uniform collapse => all == 1/num_experts)
  - mean per-token routing entropy vs max (low => confident routing; high => flat)
  - fraction of tokens whose top-1 gate weight >> uniform (decisiveness)
  - expert x family contingency  (does an expert prefer certain families?)
  - normalized mutual information I(expert; family) / H(family)  (emergent specialization)

Usage: python scripts/diagnose_residue_moe_routing.py <CKPT_DIR> [ckpt] [n_per_family]
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
gran = getattr(cfg, "moe_granularity", "protein")
E = cfg.num_experts; K = cfg.top_k
print(f"[ckpt] {CKDIR}/{CKNAME}  granularity={gran}  num_experts={E} top_k={K}")
assert gran == "residue", "this diagnostic targets the per-residue MoE"
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKDIR, CKNAME), map_location=dev, weights_only=False)["model"], strict=False)

# proteins across families (dedup by sequence), balanced sample
df = pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet")
seqcol = "sequence"; fcol = "family_name"; fidcol = "family_id"
if "dbd_start" in df.columns:
    df = df.assign(dbdseq=[str(s)[int(a):int(b)] if not pd.isna(a) else str(s)
                         for s, a, b in zip(df[seqcol], df.get("dbd_start", 0), df.get("dbd_end", 0))])
else:
    df["dbdseq"] = df[seqcol].astype(str)
df = df.drop_duplicates("dbdseq")
rows = []
for fam, g in df.groupby(fcol):
    rows.append(g.sample(min(N_PER_FAM, len(g)), random_state=0))
samp = pd.concat(rows)
print(f"proteins sampled: {len(samp)} across {samp[fcol].nunique()} families")

top1 = []; fam_of_tok = []; gate_max = []; entropies = []
fams = sorted(samp[fcol].unique())
fam2i = {f: i for i, f in enumerate(fams)}

@torch.no_grad()
def run(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([int(fid)], device=dev)
    _, _, aux = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return aux

for r in samp.itertuples():
    seq = getattr(r, "dbdseq")
    if not isinstance(seq, str) or len(seq) < 4: continue
    aux = run(seq, getattr(r, fidcol))
    gl = aux["gate_logits"]                         # (Ntok, E)
    ti = aux["top_indices"][:, 0]                   # (Ntok,) top-1 expert
    p = F.softmax(gl.float(), dim=-1)               # (Ntok, E)
    ent = -(p * (p + 1e-9).log()).sum(-1)           # (Ntok,)
    top1.append(ti.cpu().numpy())
    gate_max.append(p.max(-1).values.cpu().numpy())
    entropies.append(ent.cpu().numpy())
    fam_of_tok.append(np.full(ti.shape[0], fam2i[getattr(r, fcol)]))

top1 = np.concatenate(top1); gate_max = np.concatenate(gate_max)
entropies = np.concatenate(entropies); fam_of_tok = np.concatenate(fam_of_tok)
Ntok = len(top1)

usage = np.bincount(top1, minlength=E) / Ntok
maxent = np.log(E)
print(f"\n=== ROUTING over {Ntok} DBD tokens ===")
print(f"marginal top-1 expert usage (uniform={1/E:.3f}):")
print("  " + "  ".join(f"e{i}:{usage[i]:.3f}" for i in range(E)))
print(f"  usage std = {usage.std():.3f}  (0 = perfect collapse; higher = uneven)")
print(f"  max/min usage ratio = {usage.max()/max(usage.min(),1e-9):.1f}x")
print(f"mean per-token routing entropy = {entropies.mean():.3f} / max {maxent:.3f}  "
      f"({entropies.mean()/maxent:.2f} of max; 1.0 = flat/collapsed)")
print(f"mean top-1 gate weight = {gate_max.mean():.3f}  (uniform={1/E:.3f}; higher = decisive)")

# expert x family contingency (row-normalized: P(expert | family))
Fn = len(fams)
C = np.zeros((Fn, E))
for f in range(Fn):
    sel = fam_of_tok == f
    C[f] = np.bincount(top1[sel], minlength=E) / max(sel.sum(), 1)
print("\n=== P(top-1 expert | family) — rows sum to 1 ===")
print("family".ljust(18) + "".join(f"  e{i}" for i in range(E)))
for f in range(Fn):
    print(fams[f][:17].ljust(18) + "".join(f" {C[f,i]:.2f}" for i in range(E)))

# normalized mutual information I(expert;family)/H(family)
joint = np.zeros((Fn, E))
for f in range(Fn):
    sel = fam_of_tok == f; joint[f] = np.bincount(top1[sel], minlength=E)
joint = joint / joint.sum()
pf = joint.sum(1, keepdims=True); pe = joint.sum(0, keepdims=True)
nz = joint > 0
MI = (joint[nz] * np.log(joint[nz] / (pf @ pe)[nz])).sum()
Hf = -(pf[pf > 0] * np.log(pf[pf > 0])).sum()
He = -(pe[pe > 0] * np.log(pe[pe > 0])).sum()
print(f"\nI(expert;family) = {MI:.3f} nats | H(family)={Hf:.3f} H(expert)={He:.3f}")
print(f"NMI = I/H(family) = {MI/max(Hf,1e-9):.3f}   (0 = routing independent of family/collapse; "
      f"1 = expert determined by family)")

verdict = "COLLAPSED (uniform)" if usage.std() < 0.02 and entropies.mean()/maxent > 0.9 else \
          ("SPECIALIZED" if MI/max(Hf,1e-9) > 0.1 else "PARTIAL / weak structure")
print(f"\nVERDICT: {verdict}")
out = {"ckpt": f"{CKDIR}/{CKNAME}", "n_tokens": int(Ntok), "usage": usage.round(3).tolist(),
       "usage_std": round(float(usage.std()), 4), "mean_entropy": round(float(entropies.mean()), 3),
       "max_entropy": round(float(maxent), 3), "mean_top1_gate": round(float(gate_max.mean()), 3),
       "NMI_expert_family": round(float(MI/max(Hf, 1e-9)), 3), "verdict": verdict,
       "P_expert_given_family": {fams[f]: C[f].round(3).tolist() for f in range(Fn)}}
os.makedirs("results/residue_moe_cases", exist_ok=True)
json.dump(out, open("results/residue_moe_cases/routing_diagnostic.json", "w"), indent=1)
print("saved results/residue_moe_cases/routing_diagnostic.json")
