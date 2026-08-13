"""HOXD13 Q317K (homeodomain position-50 Q->K) directional specificity-switch score,
for combined (learned-10) and semfam34_fixed. Same difference-in-differences metric as
Fig 4a (MyoD1): score competing motifs under WT vs mutant predicted PWMs.

  S(seq|PWM) = best PWM log-odds (bg 0.25), max over offsets + both strands
  Δ_switch   = [S_mut(MUT) - S_mut(WT)] - [S_WT(MUT) - S_WT(WT)]
  >0 -> Q317K shifts preference toward the mutant motif (switch reproduced)

WT recognizes CCAATAAAA (core AATAAA); mutant -> GGGATTAA / GGAT(T)AA.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

WT_DBD  = "VALNQPDMCVYRRGRKKRVPYTKLQLKELENEYAINKFINKDKRRRISAATNLSERQVTIWFQNRRVKDKKIVSKLKDTVS"
MUT_DBD = "VALNQPDMCVYRRGRKKRVPYTKLQLKELENEYAINKFINKDKRRRISAATNLSERQVTIWFKNRRVKDKKIVSKLKDTVS"
diff = [i for i in range(len(WT_DBD)) if WT_DBD[i] != MUT_DBD[i]]
assert len(diff) == 1 and WT_DBD[diff[0]] == "Q" and MUT_DBD[diff[0]] == "K", (diff,)
p = diff[0]
print(f"Q317K at DBD pos {p} (context ...{WT_DBD[p-3:p+4]} -> {MUT_DBD[p-3:p+4]}...)\n")

# (label, WT motif, mutant motif)
PAIRS = [
 ("core AATAAA/GGATAA",    "AATAAA",    "GGATAA"),
 ("core AATAAA/GGATTAA",   "AATAAA",    "GGATTAA"),
 ("full CCAATAAAA/GGGATTAA","CCAATAAAA","GGGATTAA"),
]
FID = 4  # Homeodomain in both learned-10 and rebin34

MODELS = {
 "combined":       "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
 "semfam34_fixed": "/data1/leihuang/project/TFScope/checkpoints/v19_combined_semfam34_contact_fixed/rag_seed42/ckpt_best.pt",
}

def load(ck):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.dirname(ck) + "/config.json")).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"], strict=False)
    return m

@torch.no_grad()
def predict(m, seq, fid=FID):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()

B = {"A":0,"C":1,"G":2,"T":3}
def rc(s): return s[::-1].translate(str.maketrans("ACGT","TGCA"))
def score(P, seq):
    lo = np.log2(np.clip(P,1e-6,1)/0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B[c] for c in s]
        for off in range(0, W-L+1):
            best = max(best, float(sum(lo[idx[j], off+j] for j in range(L))))
    return best
def cons(P, g):
    c = np.where(g>0.5)[0]; lo,hi=(c.min(),c.max()+1) if len(c)>=4 else (0,P.shape[1])
    return "".join("ACGT"[i] for i in P[:,lo:hi].argmax(0))

out = {}
for name, ck in MODELS.items():
    m = load(ck)
    gw, pw = predict(m, WT_DBD); gm, pm = predict(m, MUT_DBD)
    print(f"\n########## {name} ##########")
    print(f"  WT  consensus: {cons(pw,gw)}")
    print(f"  MUT consensus: {cons(pm,gm)}")
    rec = {"wt_consensus": cons(pw,gw), "mut_consensus": cons(pm,gm), "switch": {}}
    for label, WT_MOTIF, mut in PAIRS:
        Swt_w, Swt_m = score(pw, WT_MOTIF), score(pw, mut)
        Smt_w, Smt_m = score(pm, WT_MOTIF), score(pm, mut)
        dWT  = Swt_m - Swt_w          # mut-motif preference under WT protein
        dMUT = Smt_m - Smt_w          # mut-motif preference under mutant protein
        dsw  = dMUT - dWT
        rec["switch"][label] = {"WT_S_wtmotif": round(Swt_w,2), "WT_S_mutmotif": round(Swt_m,2),
                                "MUT_S_wtmotif": round(Smt_w,2), "MUT_S_mutmotif": round(Smt_m,2),
                                "delta_switch": round(dsw,2)}
        verdict = "switch reproduced (>0)" if dsw > 0 else "NOT reproduced"
        print(f"  [{WT_MOTIF} vs {mut}]  WT: S(wt)={Swt_w:.2f} S(mut)={Swt_m:.2f} (d={dWT:+.2f}) | "
              f"MUT: S(wt)={Smt_w:.2f} S(mut)={Smt_m:.2f} (d={dMUT:+.2f})  ->  Δ_switch={dsw:+.2f}  {verdict}")
    out[name] = rec

os.makedirs("results/hoxd13_mut", exist_ok=True)
json.dump(out, open("results/hoxd13_mut/switch_score.json","w"), indent=1)
print("\nsaved results/hoxd13_mut/switch_score.json")
