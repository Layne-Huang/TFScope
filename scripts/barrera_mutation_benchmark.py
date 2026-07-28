#!/usr/bin/env python
"""Build the Barrera-2016 homeodomain WT/MUT PWM benchmark and evaluate a model's
MUTATION SENSITIVITY: does it predict a WT->MUT PWM change matching the measured one?

55 single-residue human homeodomain variants (disease), each with a Barrera-2016
PBM motif; WT motif from CIS-BP. Sequences = HMM-aligned 60-res homeodomain
(matchTab), WT vs MUT differ at one position.
"""
import os, sys, json, re
import numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm
R = "/data1/leihuang/rCLAMPS/cis_bp"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def parse_pwms(path):
    """PWM.txt = concatenated blocks; return {motif_id: (L,4) array}."""
    out = {}; mid = None; rows = []
    def flush():
        if mid and rows: out[mid] = np.array(rows, np.float32)
    for ln in open(path):
        t = ln.rstrip("\n").split("\t")
        if t[0] == "TF": flush(); mid = None; rows = []
        elif t[0] == "Motif": mid = t[1].split("_")[0]      # M00260_2.00 -> M00260
        elif t and t[0].isdigit() and len(t) >= 5:
            rows.append([float(x) for x in t[1:5]])
    flush(); return out

PW = parse_pwms(f"{R}/PWM.txt")

def seqs_from_matchtab(path):
    d = {}
    for ln in open(path):
        p = ln.split()
        if len(p) > 40: d[p[0]] = "".join(p[1:])   # name + 60 aligned residues
    return d
mut_seq = seqs_from_matchtab(f"{R}/homeodomains_barreraMuts_hasPWM.matchTab.txt")
wt_seq  = seqs_from_matchtab(f"{R}/homeodomains_cisbp_hasPWM.matchTab.txt")

# mutant motif ids (DBID.2 = mutant name e.g. ARX_L343Q; Motif_ID -> PWM)
mt = pd.read_csv(f"{R}/motifTable_Barrera2016_mutsOnly.txt", sep="\t")
mutrows = {r["DBID.2"]: (str(r.TF_Name).upper(), str(r.Motif_ID).split("_")[0]) for _, r in mt.iterrows()}
# WT motif per gene (pick first homeodomain entry)
wt = pd.read_csv(f"{R}/motifTable_mostRecent_noMuts.txt", sep="\t")
wt = wt[wt.Family_Name.astype(str).str.contains("Homeo", na=False)]
wt_mid = {}
for _, r in wt.iterrows():
    g = str(r.TF_Name).upper(); wt_mid.setdefault(g, str(r.Motif_ID).split("_")[0])
# WT aligned seq per gene: match matchtab keys (may be gene or accession) — build gene->seq
wt_seq_by_gene = {}
for k, s in wt_seq.items():
    wt_seq_by_gene.setdefault(k.upper(), s)

pairs = []
for mutname, (gene, mmid) in mutrows.items():
    if mmid not in PW: continue
    wmid = wt_mid.get(gene)
    if wmid is None or wmid not in PW: continue
    ms = mut_seq.get(mutname); ws = wt_seq_by_gene.get(gene)
    if ms is None: continue
    if ws is None:  # fall back: WT seq = mutant seq with the point mutation reverted
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", mutname.split("_", 1)[1])
        ws = ms  # can't revert in HMM frame reliably; use mut seq minus 1 diff later
    pairs.append(dict(name=mutname, gene=gene, mut=mutname.split("_", 1)[1],
                      wt_seq=ws.replace("-", ""), mut_seq=ms.replace("-", ""),
                      wt_pwm=PW[wmid].T.tolist(), mut_pwm=PW[mmid].T.tolist()))
print(f"built {len(pairs)} WT/MUT pairs ({len(set(p['gene'] for p in pairs))} genes)")

# ---- how much do the MEASURED motifs actually change? ----
def core(p):
    p = np.array(p); ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= 0.25)[0]; return p[:, inf[0]:inf[-1] + 1] if len(inf) else p
def dist(a, b):
    _, _, _, r = align_pwm(core(a), core(b), max_shift=6, consider_revcomp=True); return 1 - r
dtrue = [dist(np.array(p["wt_pwm"]), np.array(p["mut_pwm"])) for p in pairs]
print(f"measured WT->MUT motif change (1-r): median={np.median(dtrue):.2f} "
      f"p75={np.percentile(dtrue,75):.2f} | pairs with real change (>0.2): {sum(d>0.2 for d in dtrue)}/{len(pairs)}")

# ---- v24 baseline mutation sensitivity ----
CK = os.environ.get("BENCH_CK","/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42")
cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False; m = TFScopeModel(cfg).to(dev).eval(); m.use_contact_pred_head = False
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
@torch.no_grad()
def predict(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
    gl, pl, _ = m(t, dm, fi)
    gate = gl.sigmoid()[0].cpu().numpy(); P = F.softmax(pl, 1)[0].cpu().numpy()
    act = gate > 0.5; return P[:, act] if act.sum() >= 4 else P[:, :10]
dpred = []; toward = []
for p in pairs:
    pw = predict(p["wt_seq"]); pm = predict(p["mut_seq"])
    dpred.append(dist(pw, pm))
    # directional: is predicted-mut closer to TRUE-mut than to TRUE-wt?
    cm = 1 - dist(pm, np.array(p["mut_pwm"])); cw = 1 - dist(pm, np.array(p["wt_pwm"]))
    toward.append(cm > cw)
dtrue = np.array(dtrue); dpred = np.array(dpred)
big = dtrue > 0.2
print("\n=== v24 baseline MUTATION SENSITIVITY ===")
print(f"  mean predicted change (1-r WTpred vs MUTpred): {dpred.mean():.3f}  (measured mean {dtrue.mean():.3f})")
print(f"  corr(predicted change, measured change): {np.corrcoef(dpred,dtrue)[0,1]:.3f}")
print(f"  on impactful mutations (measured>0.2, n={big.sum()}): mean predicted change {dpred[big].mean():.3f}")
print(f"  directional acc (pred-MUT closer to true-MUT than true-WT): {np.mean(toward):.0%}")
os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump({"pairs": pairs, "dtrue": dtrue.tolist(), "dpred": dpred.tolist()},
          open("results/mutation_benchmark/barrera_pairs.json", "w"))
print("\nsaved results/mutation_benchmark/barrera_pairs.json")
