"""Sweep literature-documented DNA-specificity-switch point mutations / P-box swaps through the
combined no-RAG model; report which ones TFScope responds to (consensus change + L1 localization).
Goal: find good Fig 4a case studies. Mutations specified as DBD substring find->replace (robust).
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
CK = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CK) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to("cuda:0").eval(); m.load_state_dict(torch.load(CK, map_location="cuda:0", weights_only=False)["model"], strict=False)
@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device="cuda:0")
    dm = torch.ones(1, len(seq), dtype=torch.bool, device="cuda:0"); fi = torch.tensor([fid], device="cuda:0")
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    g = gl.sigmoid()[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy(); c = np.where(g > 0.5)[0]
    if len(c) < 4: ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    lo, hi = c.min(), c.max() + 1
    return p[:, lo:hi]
def cons(P): return "".join("ACGT"[i] for i in P.argmax(0))

d = pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet")
def dbd_fid(gene, fam=None):
    r = d[(d.gene_symbol == gene)]
    if fam: r = r[r.family_name == fam]
    if len(r) == 0: return None, None
    r = r.iloc[0]; return str(r.sequence)[int(r.dbd_start):int(r.dbd_end)], int(r.family_id)

# curated literature cases: (gene, find, replace, label, known)
CASES = [
    ("MYOD1", None, ("L112R_pos11"), "MyoD1 L112R (bHLH basic)", "CACCTG->CACGTG"),
    ("NR3C1", "GSCKV", "EGCKA", "GR P-box->ER (NR)", "GRE->ERE"),
    ("AR", "GSCKV", "EGCKA", "AR P-box->ER (NR)", "GRE->ERE"),
    ("PGR", "GSCKV", "EGCKA", "PGR P-box->ER (NR)", "GRE->ERE"),
    ("NR3C2", "GSCKV", "EGCKA", "MR P-box->ER (NR)", "GRE->ERE"),
    ("ESR1", "EGCKA", "GSCKV", "ER P-box->GR (NR)", "ERE->GRE"),
    ("ESR2", "EGCKA", "GSCKV", "ERbeta P-box->GR (NR)", "ERE->GRE"),
    ("ESRRA", "EGCKG", "GSCKV", "ERRalpha P-box->GR (NR)", "ERE->GRE"),
]
# add Q50K for several Q50 homeodomains (WFQN -> WFKN; Bicoid-type 3' switch)
hd = d[d.family_name == "Homeodomain"].drop_duplicates("gene_symbol")
nq = 0
for r in hd.itertuples():
    dbd = str(r.sequence)[int(r.dbd_start):int(r.dbd_end)]
    if "WFQN" in dbd and nq < 8:
        CASES.append((r.gene_symbol, "WFQN", "WFKN", f"{r.gene_symbol} Q50K (HD)", "TAATGG->TAATCC")); nq += 1

print(f"{'case':<28}{'WT':<13}{'MUT':<13}{'L1pk':>6}  verdict")
hits = []
for gene, find, repl, label, known in CASES:
    dbd, fid = dbd_fid(gene)
    if dbd is None: print(f"{label:<28} (absent)"); continue
    if gene == "MYOD1":
        if len(dbd) < 12: print(f"{label:<28} (dbd short)"); continue
        mut = dbd[:11] + "R" + dbd[12:]
    else:
        if find not in dbd: print(f"{label:<28} (motif '{find}' not found)"); continue
        mut = dbd.replace(find, repl, 1)
    Pw, Pm = predict(dbd, fid), predict(mut, fid)
    W = min(Pw.shape[1], Pm.shape[1]); l1 = np.abs(Pw[:, :W] - Pm[:, :W]).sum(0)
    cw, cm = cons(Pw), cons(Pm)
    changed = cw != cm
    verdict = "RESPONDS" if (changed and l1.max() > 0.5) else ("weak" if l1.max() > 0.3 else "insensitive")
    print(f"{label:<28}{cw:<13}{cm:<13}{l1.max():>6.2f}  {verdict}  ({known})")
    hits.append(dict(gene=gene, label=label, wt=cw, mut=cm, l1peak=round(float(l1.max()), 2), verdict=verdict, known=known))
json.dump(hits, open("results/myod1_mut/mutation_sweep.json", "w"), indent=1)
print("\nRESPONDS cases:", [h["label"] for h in hits if h["verdict"] == "RESPONDS"])
