#!/usr/bin/env python
"""Phase 3 train + go/no-go: recognition-energy decoder on native bHLH (frozen ESM,
MyoD1 cluster held out), then predict MyoD1 WT & L112R ZERO-SHOT (no mutant labels).

Go: L112R shifts predicted E-box toward CACGTG (Δ_switch>0, correct signed central
base), beats WT-copy and v24, magnitude not collapsed, WT stays a valid E-box.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings, random
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; NPZ = "/data1/leihuang/TFScope/phase3_bhlh.npz"
AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; BASES = "ACGT"
NPOS = 24; torch.manual_seed(0); np.random.seed(0); random.seed(0)

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x

def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
def ic_core(p, thr=0.2):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= thr)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])

def best_align(Pp, gt):
    """place gt (4,Lg) into pred frame (4,NPOS); return (shift, rc_flag)."""
    Lg = gt.shape[1]; best = (-9, 0, False)
    for rcf, g in [(False, gt), (True, rc(gt))]:
        for s in range(0, NPOS - Lg + 1):
            r = np.corrcoef(Pp[:, s:s + Lg].ravel(), g.ravel())[0, 1]
            if r == r and r > best[0]: best = (r, s, rcf)
    return best[1], best[2]

def motif_score(P, motif):
    """max log-odds placement (both orientations) of `motif` in P (4,NPOS)."""
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            sc = sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm))
            best = max(best, sc)
    return best

# ── data ──
d = np.load(NPZ, allow_pickle=True)
embs, pwms, seqs, genes = d["embs"], d["pwms"], d["seqs"], d["genes"]
myo = d["myo"][0]

def align_to_ref(gt, ref):
    """fixed (shift,rc) placing gt (4,Lg) into ref frame (4,NPOS), best corr."""
    Lg = min(gt.shape[1], NPOS); gt = gt[:, :Lg]; best = (-9, 0, False)
    for rcf, g in [(False, gt), (True, rc(gt))]:
        for s in range(0, NPOS - Lg + 1):
            r = np.corrcoef(ref[:, s:s + Lg].ravel(), g.ravel())[0, 1]
            if r == r and r > best[0]: best = (r, s, rcf)
    return best[1], best[2], Lg

# build a shared bHLH consensus reference (2 passes), then fix each example's registration
gts = [pwms[i].astype(np.float32) for i in range(len(pwms))]
ref = np.full((4, NPOS), 0.25, np.float32)
c = gts[np.argmax([g.shape[1] for g in gts])]; ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
for _ in range(2):
    acc = np.full((4, NPOS), 1e-3, np.float32)
    for g in gts:
        s, rcf, Lg = align_to_ref(g, ref); gg = (rc(g) if rcf else g)[:, :Lg]
        acc[:, s:s + Lg] += gg
    ref = acc / acc.sum(0, keepdims=True)

data = []
for i in range(len(embs)):
    g = gts[i]; s, rcf, Lg = align_to_ref(g, ref)
    gg = (rc(g) if rcf else g)[:, :Lg].copy()
    data.append(dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(str(seqs[i])).to(dev),
                     gt=torch.tensor(gg, device=dev), s=s, Lg=Lg))
print(f"train rows {len(data)} | fixed registration to shared bHLH consensus | held-out MyoD1 zero-shot")

def train(second_shell=True, epochs=120, lr=1e-3, tag=""):
    torch.manual_seed(0)
    m = RecognitionEnergyDecoder(n_pos=NPOS, use_second_shell=second_shell).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        random.shuffle(data); tot = 0.0
        for ex in data:
            Zt = m(ex["h"], ex["oh"], second_shell_on=second_shell).t()   # (4,NPOS)
            s, Lg = ex["s"], ex["Lg"]
            logP = F.log_softmax(Zt[:, s:s + Lg], 0)
            loss = -(ex["gt"] * logP).sum(0).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        if (ep + 1) % 50 == 0: print(f"  [{tag}] ep{ep+1} loss {tot/len(data):.3f}")
    return m

@torch.no_grad()
def predict(m, emb, seq, oracle=None):
    h = torch.tensor(emb).float().to(dev); oh = onehot(seq).to(dev)
    ob = None
    if oracle is not None:
        ob = torch.zeros(len(seq), device=dev)
        for i in oracle:
            if i < len(seq): ob[i] = 4.0
    z = m(h, oh, oracle_bias=ob)
    return F.softmax(z.t(), 0).cpu().numpy()                # (4,NPOS)

def gonogo(m, tag, oracle=None):
    Pw = predict(m, myo["wt_emb"], myo["wt_seq"], oracle)
    Pm = predict(m, myo["mut_emb"], myo["mut_seq"], oracle)
    a, b = ic_core(Pw); a2 = a; b2 = min(a + 6, b) if b - a >= 6 else b
    consw = "".join(BASES[i] for i in Pw[:, a:b].argmax(0))
    consm = "".join(BASES[i] for i in Pm[:, a:b].argmax(0))
    dpred = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]
    # directional switch: S(CACGTG) - S(CAGCTG), MUT minus WT
    def dd(P): return motif_score(P, "CACGTG") - motif_score(P, "CAGCTG")
    dswitch = dd(Pm) - dd(Pw)
    print(f"  [{tag}] WT cons={consw}  MUT cons={consm}  Δpred={dpred:.3f}  "
          f"Δ_switch(→CACGTG)={dswitch:+.2f}  [WT dd={dd(Pw):+.2f} MUT dd={dd(Pm):+.2f}]")
    return dict(tag=tag, wt=consw, mut=consm, dpred=float(dpred), dswitch=float(dswitch))

print("\n=== TRAIN: direct-only (fast; core recognition-energy test) ===")
m_dir = train(second_shell=False, tag="direct")
print("\n=== GO/NO-GO (direct-only) ===")
res = []
res.append(gonogo(m_dir,  "recog-energy DIRECT-ONLY (learned)"))
res.append(gonogo(m_dir,  "recog-energy DIRECT-ONLY + ORACLE contact", oracle=list(range(15))))
print("\n=== TRAIN: full (direct+second-shell) ===")
m_full = train(second_shell=True, tag="full")
print("\n=== GO/NO-GO (full) ===")
res.append(gonogo(m_full, "recog-energy full (learned contact)"))
res.append(gonogo(m_full, "recog-energy full + ORACLE contact", oracle=list(range(15))))
print("  [baseline] WT-copy                         Δpred=0.000  Δ_switch=+0.00 (no change by construction)")
print("  [baseline] v24 (from phase2)               Δpred=0.162  Δ_switch: consensus stays CAGCTG (no flip)")
os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(res, open("results/mutation_benchmark/phase3_gonogo.json", "w"), indent=1)
torch.save({"full": m_full.state_dict(), "direct": m_dir.state_dict()},
           "checkpoints/phase3_recog_energy.pt" if os.path.isdir("checkpoints") else "/data1/leihuang/TFScope/phase3_recog_energy.pt")
print("\nsaved results/mutation_benchmark/phase3_gonogo.json + decoder ckpt")
