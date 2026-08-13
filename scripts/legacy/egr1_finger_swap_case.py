"""Multi-mutation specificity-switch case study: EGR1/Zif268 zinc-finger-2 swap.

EGR1 binds 5'-GCG TGG GCG-3' via 3 C2H2 fingers; finger 2 (recognition helix
RSDHLTT) reads the MIDDLE triplet TGG. We swap F2's helix to finger-3's GCG-
reading helix (RSDHLTT -> RSDERKR, 4 residue changes), so the middle finger
should now read GCG -> predicted site GCG GCG GCG. This is a 3-bp motif change
driven by a 4-residue swap: much larger on both axes (residues changed AND motif
difference) than the single-point MyoD1 L112R (1 residue, 1-bp E-box).

Directional switch score (generalized Fig-4a metric):
  d(P)      = S(GCGGCGGCG | P) - S(GCGTGGGCG | P)     (higher => prefers all-GCG)
  Delta_sw  = d(swap) - d(WT)                          (>0 => finger swap captured)

Usage: python scripts/egr1_finger_swap_case.py <CKPT_DIR> [ckpt] [tag]
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKDIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42"
CKNAME = sys.argv[2] if len(sys.argv) > 2 else "ckpt_best.pt"
TAG = sys.argv[3] if len(sys.argv) > 3 else "residue"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
FID = 0  # EGR1 = C2H2_short

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKDIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKDIR, CKNAME), map_location=dev, weights_only=False)["model"], strict=False)

WT_DBD  = "RPYACPVESCDRRFSRSADLTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHL"
SWAP_DBD = WT_DBD[:43] + "RSDERKR" + WT_DBD[50:]     # F2 helix RSDHLTT -> RSDERKR
assert WT_DBD[43:50] == "RSDHLTT" and SWAP_DBD[43:50] == "RSDERKR"
SITE_WT   = "GCGTGGGCG"     # EGR1 canonical
SITE_SWAP = "GCGGCGGCG"     # middle triplet TGG -> GCG expected after F2 swap
B = {"A":0,"C":1,"G":2,"T":3}

@torch.no_grad()
def predict(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([FID], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()
def rc(s): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
def score(P, seq):
    lo = np.log2(np.clip(P, 1e-6, 1) / 0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B[c] for c in s]
        for off in range(0, W - L + 1):
            best = max(best, float(sum(lo[idx[j], off + j] for j in range(L))))
    return best
def core(gate, P):
    c = np.where(gate > 0.5)[0]
    if len(c) < 4:
        ic = (P*np.log2(P+1e-9)).sum(0)+2; a = ic.argmax(); c = np.arange(max(0,a-4), min(P.shape[1],a+5))
    lo, hi = c.min(), c.max()+1
    return "".join("ACGT"[i] for i in P[:, lo:hi].argmax(0))

gw, pw = predict(WT_DBD); gs, ps = predict(SWAP_DBD)
print(f"=== EGR1 finger-2 swap ({TAG}: {CKDIR.split('/')[-2]}) ===")
print(f"WT   (F2=RSDHLTT) predicted core: {core(gw, pw)}")
print(f"SWAP (F2=RSDERKR) predicted core: {core(gs, ps)}")
print(f"\nsite scores  S(GCGTGGGCG)=WT-site   S(GCGGCGGCG)=swap-site")
for tag, P in [("WT", pw), ("SWAP", ps)]:
    sWT, sSW = score(P, SITE_WT), score(P, SITE_SWAP)
    print(f"  {tag:5} S(WT-site)={sWT:+.2f}  S(swap-site)={sSW:+.2f}  d=S(swap)-S(WT)={sSW-sWT:+.2f}")
dWT = score(pw, SITE_SWAP) - score(pw, SITE_WT)
dSW = score(ps, SITE_SWAP) - score(ps, SITE_WT)
print(f"\nDelta_switch = d(swap) - d(WT) = {dSW - dWT:+.2f}  "
      f"({'CAPTURED (>0): F2 swap shifts middle TGG->GCG' if dSW-dWT>0 else 'NOT captured'})")
out = {"model": TAG, "ckpt": f"{CKDIR}/{CKNAME}", "wt_core": core(gw, pw), "swap_core": core(gs, ps),
       "S_wt": {"WTsite": round(score(pw,SITE_WT),3), "swapsite": round(score(pw,SITE_SWAP),3)},
       "S_swap": {"WTsite": round(score(ps,SITE_WT),3), "swapsite": round(score(ps,SITE_SWAP),3)},
       "delta_switch": round(float(dSW - dWT), 3)}
os.makedirs("results/egr1_finger_swap", exist_ok=True)
json.dump(out, open(f"results/egr1_finger_swap/egr1_swap_{TAG}.json", "w"), indent=1)
print(f"saved results/egr1_finger_swap/egr1_swap_{TAG}.json")
