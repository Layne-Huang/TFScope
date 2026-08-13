#!/usr/bin/env python
"""Phase 4 go/no-go battery for the bHLH recognition-energy decoder.

(A) GENE-HELD-OUT WT covR: train recog-energy on ~80% of bHLH genes, eval WT covR on
    held-out genes vs FamilyCode (family-average) and nearest-paralog. Directly tests
    the "FamilyCode consistently better -> no-go" gate and WT generalization.
(B) MyoD1 mutation battery (MyoD1 fully zero-shot): specificity switch (L112R), an
    UNSEEN substitution (L112K), and NEUTRAL controls (non-contacting residues) ->
    neutral calibration + specificity (Δ_switch large only for the specificity residue).

Baselines: WT-copy, FamilyCode, nearest-paralog, recog-energy (+ v24/E1 numbers cited
from Phase-2/3). No mutant labels used in training. Repro at bottom.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings, random
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; NPZ = "/data1/leihuang/TFScope/phase3_bhlh.npz"
AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; BASES = "ACGT"; NPOS = 24
torch.manual_seed(0); np.random.seed(0); random.seed(0)

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
def ic_core(p, thr=0.2):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= thr)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
def align_corr(a, bmat):
    Lb = min(bmat.shape[1], a.shape[1]); best = -9
    for g in [bmat[:, :Lb], rc(bmat[:, :Lb])]:
        for s in range(0, a.shape[1] - Lb + 1):
            r = np.corrcoef(a[:, s:s + Lb].ravel(), g.ravel())[0, 1]
            if r == r: best = max(best, r)
    return best
def align_to_ref(gt, ref):
    Lg = min(gt.shape[1], NPOS); gt = gt[:, :Lg]; best = (-9, 0, False)
    for rcf, g in [(False, gt), (True, rc(gt))]:
        for s in range(0, NPOS - Lg + 1):
            r = np.corrcoef(ref[:, s:s + Lg].ravel(), g.ravel())[0, 1]
            if r == r and r > best[0]: best = (r, s, rcf)
    return best[1], best[2], Lg
def seqid(a, b):
    L = min(len(a), len(b)); return sum(a[i] == b[i] for i in range(L)) / max(L, 1)

# ── data ──
d = np.load(NPZ, allow_pickle=True)
genes = [str(g) for g in d["genes"]]; seqs = [str(s) for s in d["seqs"]]
embs = d["embs"]; pwms = [p.astype(np.float32) for p in d["pwms"]]; myo = d["myo"][0]
uniq = sorted(set(genes))
held_genes = set(uniq[::5])                                     # ~20% gene-held-out
tr_idx = [i for i in range(len(genes)) if genes[i] not in held_genes]
ho_idx = [i for i in range(len(genes)) if genes[i] in held_genes]
print(f"bHLH: {len(uniq)} genes -> train {len(uniq)-len(held_genes)} / held-out {len(held_genes)} genes "
      f"({len(tr_idx)}/{len(ho_idx)} rows); MyoD1 separately zero-shot")

def build_ref(idxs):
    gts = [pwms[i] for i in idxs]
    ref = np.full((4, NPOS), 0.25, np.float32)
    c = gts[int(np.argmax([g.shape[1] for g in gts]))]; ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
    for _ in range(2):
        acc = np.full((4, NPOS), 1e-3, np.float32)
        for g in gts:
            s, rcf, Lg = align_to_ref(g, ref); acc[:, s:s + Lg] += (rc(g) if rcf else g)[:, :Lg]
        ref = acc / acc.sum(0, keepdims=True)
    return ref
ref_tr = build_ref(tr_idx)                                      # FamilyCode = train family-avg

# ── train recog-energy on TRAIN genes only ──
train_data = []
for i in tr_idx:
    s, rcf, Lg = align_to_ref(pwms[i], ref_tr)
    gg = (rc(pwms[i]) if rcf else pwms[i])[:, :Lg].copy()
    train_data.append(dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(seqs[i]).to(dev),
                           gt=torch.tensor(gg, device=dev), s=s, Lg=Lg))
m = RecognitionEnergyDecoder(n_pos=NPOS, use_second_shell=False).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
for ep in range(120):
    random.shuffle(train_data); tot = 0.0
    for ex in train_data:
        Zt = m(ex["h"], ex["oh"]).t(); s, Lg = ex["s"], ex["Lg"]
        loss = -(ex["gt"] * F.log_softmax(Zt[:, s:s + Lg], 0)).sum(0).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sch.step()
    if (ep + 1) % 60 == 0: print(f"  train ep{ep+1} loss {tot/len(train_data):.3f}")

@torch.no_grad()
def re_pred(emb, seq, oracle=None):
    h = torch.tensor(emb).float().to(dev); oh = onehot(seq).to(dev)
    ob = None
    if oracle is not None:
        ob = torch.zeros(len(seq), device=dev)
        for i in oracle:
            if i < len(seq): ob[i] = 4.0
    z = m(h, oh, oracle_bias=ob)
    return z.t().cpu().numpy(), F.softmax(z.t(), 0).cpu().numpy()

# ── (A) gene-held-out WT covR ──
tr_seqs = [(genes[i], seqs[i], pwms[i]) for i in tr_idx]
def _pad(p):
    a = np.full((4, NPOS), 0.25, np.float32); a[:, :min(p.shape[1], NPOS)] = p[:, :NPOS]; return a
re_c, fc_c, np_c = [], [], []
for i in ho_idx:
    _, P = re_pred(embs[i], seqs[i]); meas = pwms[i]
    re_c.append(align_corr(P, meas)); fc_c.append(align_corr(ref_tr, meas))
    j = max(tr_seqs, key=lambda t: seqid(seqs[i], t[1]))        # nearest paralog by seq-id
    np_c.append(align_corr(_pad(j[2]), meas))
print(f"\n(A) GENE-HELD-OUT WT covR ({len(ho_idx)} genes):")
print(f"    recog-energy   {np.mean(re_c):.3f} ± {np.std(re_c):.3f}")
print(f"    FamilyCode     {np.mean(fc_c):.3f} ± {np.std(fc_c):.3f}")
print(f"    nearest-paralog{np.mean(np_c):.3f} ± {np.std(np_c):.3f}")
wins = sum(re_c[k] > fc_c[k] for k in range(len(re_c)))
print(f"    recog-energy beats FamilyCode on {wins}/{len(re_c)} held-out genes")

# ── (B) MyoD1 mutation battery (zero-shot): needs raw-ESM embeddings for extra mutants ──
WT = myo["wt_seq"]
def mk(pos, to): return WT[:pos] + to + WT[pos + 1:]
muts = {"L112R (specificity)": (11, "R"), "L112K (unseen subst)": (11, "K"),
        "neutral@40->A": (40, "A"), "neutral@35->G": (35, "G")}
import esm as esm_lib
esm_model, alph = esm_lib.pretrained.esm2_t33_650M_UR50D(); esm_model = esm_model.eval().to(dev)
for p in esm_model.parameters(): p.requires_grad = False
bc = alph.get_batch_converter()
@torch.no_grad()
def embed(seq):
    _, _, t = bc([("x", seq)]); r = esm_model(t.to(dev), repr_layers=[33])["representations"][33][0]
    return r[1:1 + len(seq)].float().cpu().numpy()
def motif_score(P, motif):
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            best = max(best, sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm)))
    return best
_, Pw = re_pred(myo["wt_emb"], WT)
ddw = motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG")
a, b = ic_core(Pw)
print(f"\n(B) MyoD1 mutation battery (zero-shot; WT E-box={''.join(BASES[i] for i in Pw[:,a:b].argmax(0))[:6]}):")
batt = {}
for name, (pos, to) in muts.items():
    em = embed(mk(pos, to)); _, Pm = re_pred(em, mk(pos, to))
    dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - ddw
    dpred = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]
    batt[name] = dict(dswitch=float(dsw), dpred=float(dpred))
    print(f"    {name:<24} Δ_switch={dsw:+.2f}  Δpred={dpred:.3f}")
print("    (want: specificity L112R Δ_switch>0 & Δpred>neutral; neutrals ~0)")

os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(dict(wt_covr=dict(recog=float(np.mean(re_c)), familycode=float(np.mean(fc_c)),
               nearest_paralog=float(np.mean(np_c)), n=len(ho_idx),
               recog_beats_fc=f"{wins}/{len(re_c)}"), battery=batt),
          open("results/mutation_benchmark/phase4_battery.json", "w"), indent=1)
print("\nsaved results/mutation_benchmark/phase4_battery.json")
