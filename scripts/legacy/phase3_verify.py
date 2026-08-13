#!/usr/bin/env python
"""Phase 3 rigor: verify the recognition-energy go/no-go with (a) signed central-base
change (GC->CG at the E-box centre), (b) WT absolute covR vs MyoD1's MEASURED PWM
(no regression), (c) a FamilyCode baseline = family-average bHLH motif (aligned).
Uses the saved decoder; MyoD1 stays zero-shot.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; NPZ = "/data1/leihuang/TFScope/phase3_bhlh.npz"
CKP = "checkpoints/phase3_recog_energy.pt" if os.path.exists("checkpoints/phase3_recog_energy.pt") else "/data1/leihuang/TFScope/phase3_recog_energy.pt"
AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; BASES = "ACGT"; NPOS = 24

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
def ic_core(p, thr=0.2):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= thr)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
def align(a, bmat):  # place b(4,Lb) into a(4,NPOS) best corr; return corr
    Lb = min(bmat.shape[1], a.shape[1]); best = -9
    for g in [bmat[:, :Lb], rc(bmat[:, :Lb])]:
        for s in range(0, a.shape[1] - Lb + 1):
            r = np.corrcoef(a[:, s:s + Lb].ravel(), g.ravel())[0, 1]
            if r == r: best = max(best, r)
    return best

d = np.load(NPZ, allow_pickle=True); myo = d["myo"][0]
sd = torch.load(CKP, map_location=dev)
m = RecognitionEnergyDecoder(n_pos=NPOS, use_second_shell=False).to(dev).eval()
m.load_state_dict(sd["direct"]);

@torch.no_grad()
def pred(emb, seq, oracle=None):
    h = torch.tensor(emb).float().to(dev); oh = onehot(seq).to(dev)
    ob = None
    if oracle is not None:
        ob = torch.zeros(len(seq), device=dev)
        for i in oracle:
            if i < len(seq): ob[i] = 4.0
    z = m(h, oh, oracle_bias=ob)
    return z.t().cpu().numpy(), F.softmax(z.t(), 0).cpu().numpy()   # logits(4,NPOS), P

# find the E-box (CANNTG) window in WT pred by best match to consensus C-A-x-x-T-G
def find_ebox(P):
    idx = {b: i for i, b in enumerate(BASES)}; best = (-9, 0, False)
    for rcf, g in [(False, P), (True, rc(P))]:
        for s in range(0, NPOS - 6 + 1):
            w = g[:, s:s + 6]
            sc = w[idx['C'], 0] + w[idx['A'], 1] + w[idx['T'], 4] + w[idx['G'], 5]  # anchor C A .. T G
            if sc > best[0]: best = (sc, s, rcf)
    return best[1], best[2]

zw, Pw = pred(myo["wt_emb"], myo["wt_seq"])
zm, Pm = pred(myo["mut_emb"], myo["mut_seq"])
s, rcf = find_ebox(Pw)
gW = rc(zw) if rcf else zw; gM = rc(zm) if rcf else zm
ebW = gW[:, s:s + 6]; ebM = gM[:, s:s + 6]                       # (4,6) CANNTG frame
print("E-box window (WT):", "".join(BASES[i] for i in ebW.argmax(0)),
      " (MUT):", "".join(BASES[i] for i in ebM.argmax(0)))
# central dinucleotide = positions 2,3.  WT target GC, MUT target CG.
for pos, lbl in [(2, "central-1"), (3, "central-2")]:
    cw = {b: float(ebW[i, pos] - ebW[:, pos].mean()) for i, b in enumerate(BASES)}
    cm = {b: float(ebM[i, pos] - ebM[:, pos].mean()) for i, b in enumerate(BASES)}
    dG = {b: cm[b] - cw[b] for b in BASES}
    print(f"  {lbl}: WT top={max(cw,key=cw.get)} MUT top={max(cm,key=cm.get)}  "
          f"ΔlogOdds C={dG['C']:+.2f} G={dG['G']:+.2f}  (want pos2 G→C: C+/G- ; pos3 C→G: G+/C-)")

# WT absolute covR vs MEASURED MyoD1 WT PWM (str_700 in training table; held out from decoder)
tr = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet")
meas = tr[tr.filename == "str_700"].iloc[0]["pwm"]
meas = np.frombuffer(meas, np.float32).reshape(4, -1)
wt_covr = align(Pw, meas)
print(f"\nWT absolute covR (pred vs measured MyoD1 str_700): {wt_covr:.3f}")

# FamilyCode baseline = family-average bHLH motif (aligned consensus), same for WT & MUT
gts = [d["pwms"][i].astype(np.float32) for i in range(len(d["pwms"]))]
ref = np.full((4, NPOS), 0.25, np.float32)
c = gts[int(np.argmax([g.shape[1] for g in gts]))]; ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
for _ in range(2):
    acc = np.full((4, NPOS), 1e-3, np.float32)
    for g in gts:
        Lb = min(g.shape[1], NPOS); best = (-9, 0, False)
        for rf, gg in [(False, g[:, :Lb]), (True, rc(g[:, :Lb]))]:
            for ss in range(0, NPOS - Lb + 1):
                r = np.corrcoef(ref[:, ss:ss + Lb].ravel(), gg.ravel())[0, 1]
                if r == r and r > best[0]: best = (r, ss, rf)
        _, ss, rf = best; gg = (rc(g) if rf else g)[:, :Lb]; acc[:, ss:ss + Lb] += gg
    ref = acc / acc.sum(0, keepdims=True)
fc_covr = align(ref, meas)
print(f"FamilyCode (bHLH family-average) covR vs measured MyoD1: {fc_covr:.3f}  | FamilyCode Δ_switch=0 (identical WT/MUT by construction)")

json.dump(dict(wt_ebox="".join(BASES[i] for i in ebW.argmax(0)),
               mut_ebox="".join(BASES[i] for i in ebM.argmax(0)),
               wt_covr=float(wt_covr), familycode_covr=float(fc_covr)),
          open("results/mutation_benchmark/phase3_verify.json", "w"), indent=1)
print("\nsaved results/mutation_benchmark/phase3_verify.json")
