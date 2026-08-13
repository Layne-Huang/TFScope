"""Phase 9 (cont.) — validate SIGNED-DELTA metrics on held-out NATIVE same-family pairs,
where between-gene ΔPWMs are clean (no cross-platform WT/MUT bias). For held-out pairs
(S1,S2) same family: predicted Δz = center(z2)-center(z1) vs measured Δ = center(logP2)-
center(logP1) over the shared consensus core. Reports centered-delta corr, per-position
corr, directional acc, magnitude ratio, signed-base-change acc. Uses the integrated
Phase-8 decoder; the Phase-8 held-out split is reconstructed by replaying its seed.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F, warnings, random, itertools
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; BASES = "ACGT"; AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; NPOS = 24
FEATS = "/data1/leihuang/TFScope/phase8_feats.npz"; CK = "/data1/leihuang/TFScope/phase8_recog_energy.pt"

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
def ic_core(p, thr=0.3):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= thr)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
def align_to_ref(gt, ref):
    Lg = min(gt.shape[1], NPOS); gt = gt[:, :Lg]; best = (-9, 0, False)
    for rf, g in [(False, gt), (True, rc(gt))]:
        for s in range(0, NPOS - Lg + 1):
            r = np.corrcoef(ref[:, s:s + Lg].ravel(), g.ravel())[0, 1]
            if r == r and r > best[0]: best = (r, s, rf)
    return best[1], best[2], Lg
def cent(x): return x - x.mean(0, keepdims=True)

sd = torch.load(CK, map_location=dev)
m = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=2, use_second_shell=False).to(dev).eval()
m.load_state_dict(sd["model"]); refs = {k: np.array(v, np.float32) for k, v in sd["refs"].items()}
D = np.load(FEATS, allow_pickle=True); rows = list(D["rows"]); FAM = {"bHLH": 0, "HD": 1}

# reconstruct Phase-8 held-out split (same seed/order as phase8_integrate.py)
random.seed(1); ho = set()
for fam in FAM:
    idx = [i for i in range(len(rows)) if rows[i]["fam"] == fam]; random.shuffle(idx)
    ho |= set(idx[:max(1, len(idx) // 7)])

@torch.no_grad()
def zlog(feat, seq, fam): return m(torch.tensor(feat).float().to(dev), onehot(seq).to(dev), fam_id=fam).t().cpu().numpy()
def framed(p, fam):
    s, rf, Lg = align_to_ref(p, refs[fam]); P = np.full((4, NPOS), 0.25, np.float32)
    P[:, s:s + Lg] = (rc(p) if rf else p)[:, :Lg]; return P

out = {}
for fam, fi in FAM.items():
    hidx = [i for i in ho if rows[i]["fam"] == fam]
    core = slice(*ic_core(refs[fam]))
    dcorr, ppos, diracc, magr, sbc = [], [], [], [], []
    random.seed(7); pairs = list(itertools.combinations(hidx, 2)); random.shuffle(pairs); pairs = pairs[:400]
    for i, j in pairs:
        z1 = zlog(rows[i]["feat"], rows[i]["seq"], fi); z2 = zlog(rows[j]["feat"], rows[j]["seq"], fi)
        P1 = framed(rows[i]["meas"], fam); P2 = framed(rows[j]["meas"], fam)
        dp = (cent(z2) - cent(z1))[:, core]
        dt = (cent(np.log(P2 + 1e-6)) - cent(np.log(P1 + 1e-6)))[:, core]
        if dt.std() < 1e-4: continue
        dcorr.append(np.corrcoef(dp.ravel(), dt.ravel())[0, 1])
        diracc.append(float((dp * dt).sum() > 0)); magr.append(np.linalg.norm(dp) / (np.linalg.norm(dt) + 1e-6))
        pp = [np.corrcoef(dp[:, k], dt[:, k])[0, 1] for k in range(dp.shape[1]) if dp[:, k].std() > 1e-6 and dt[:, k].std() > 1e-6]
        ppos.append(np.nanmean(pp) if pp else np.nan)
        # signed base-change acc at columns where measured top base differs
        t1 = np.log(P1 + 1e-6)[:, core].argmax(0); t2 = np.log(P2 + 1e-6)[:, core].argmax(0)
        p1 = z1[:, core].argmax(0); p2 = z2[:, core].argmax(0); ch = np.where(t1 != t2)[0]
        if len(ch): sbc.append(np.mean([p2[k] == t2[k] for k in ch]))
    out[fam] = dict(n_pairs=len(dcorr), centered_delta_corr=float(np.nanmean(dcorr)),
                    per_position_delta_corr=float(np.nanmean(ppos)), directional_acc=float(np.mean(diracc)),
                    signed_base_change_acc=float(np.nanmean(sbc)) if sbc else float("nan"),
                    pred_over_true_magnitude=float(np.nanmean(magr)))
    print(f"[{fam}] held-out native pairs n={out[fam]['n_pairs']}: "
          f"Δcorr={out[fam]['centered_delta_corr']:.3f} per-pos={out[fam]['per_position_delta_corr']:.3f} "
          f"dir-acc={out[fam]['directional_acc']:.3f} signed-base-acc={out[fam]['signed_base_change_acc']:.3f} "
          f"magratio={out[fam]['pred_over_true_magnitude']:.3f}")
json.dump(out, open("results/mutation_benchmark/phase9_native_delta.json", "w"), indent=1)
print("saved results/mutation_benchmark/phase9_native_delta.json")
