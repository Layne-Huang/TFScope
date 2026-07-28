#!/usr/bin/env python
"""AUDIT Section III — consolidated ablation table (bHLH, raw ESM, MyoD1 zero-shot).
Trains each variant under an IDENTICAL protocol/eval to isolate where the gain comes from:
ESM contextual re-embedding vs recognition-energy decoder vs native-pair delta supervision
vs explicit AA channel. Baselines WT-copy / FamilyCode / (v24 cited). No new architecture.

Variants: simple_esm_mean, simple_esm_attn (low-cap MLP heads on ESM), recog_full,
recog_equiv (+native-pair Δ), recog_noAA (φ w/o AA one-hot). Eval: held-out WT covR;
MyoD1 zero-shot switch (WT/MUT E-box, Δ_switch, pos2/pos3 signed-correct, PWM-dist);
native-pair signed-Δ (dir-acc, Δcorr, mag-ratio); neutral Δ_switch; param counts.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, warnings, random, itertools
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; BASES = "ACGT"; AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; NPOS = 24
NPZ = "/data1/leihuang/TFScope/phase3_bhlh.npz"; torch.manual_seed(0); np.random.seed(0); random.seed(0)

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
def find_ebox(P):
    ix = {b: i for i, b in enumerate(BASES)}; best = (-9, 0, False)
    for rf, g in [(False, P), (True, rc(P))]:
        for s in range(0, NPOS - 6 + 1):
            w = g[:, s:s + 6]; sc = w[ix['C'], 0] + w[ix['A'], 1] + w[ix['T'], 4] + w[ix['G'], 5]
            if sc > best[0]: best = (sc, s, rf)
    return best[1], best[2]
def motif_score(P, motif):
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            best = max(best, sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm)))
    return best
def cent(x): return x - x.mean(0, keepdims=True)

# ── data (bHLH raw ESM, MyoD1 cluster already excluded) ──
D = np.load(NPZ, allow_pickle=True)
genes = [str(g) for g in D["genes"]]; seqs = [str(s) for s in D["seqs"]]
embs = D["embs"]; pwms = [p.astype(np.float32) for p in D["pwms"]]; myo = D["myo"][0]
uniq = sorted(set(genes)); held = set(uniq[::5])
tr_idx = [i for i in range(len(genes)) if genes[i] not in held]
ho_idx = [i for i in range(len(genes)) if genes[i] in held]
gts = [pwms[i] for i in tr_idx]
ref = np.full((4, NPOS), 0.25, np.float32); c = gts[int(np.argmax([g.shape[1] for g in gts]))]
ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
for _ in range(2):
    acc = np.full((4, NPOS), 1e-3, np.float32)
    for g in gts:
        s, rf, Lg = align_to_ref(g, ref); acc[:, s:s + Lg] += (rc(g) if rf else g)[:, :Lg]
    ref = acc / acc.sum(0, keepdims=True)
CA, CB = ic_core(ref, 0.3)
def frame(p):
    s, rf, Lg = align_to_ref(p, ref); P = np.full((4, NPOS), 0.25, np.float32)
    P[:, s:s + Lg] = (rc(p) if rf else p)[:, :Lg]; return P, s, Lg
tr = [dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(seqs[i]).to(dev),
           Pfull=torch.tensor(frame(pwms[i])[0], device=dev), s=frame(pwms[i])[1], Lg=frame(pwms[i])[2]) for i in tr_idx]

# extra MyoD1 mutants (raw ESM): neutral + L112K
WT = myo["wt_seq"]
import esm as esm_lib
em, alph = esm_lib.pretrained.esm2_t33_650M_UR50D(); em = em.eval().to(dev)
for p in em.parameters(): p.requires_grad = False
bc = alph.get_batch_converter()
@torch.no_grad()
def embed(seq):
    _, _, t = bc([("x", seq)]); r = em(t.to(dev), repr_layers=[33])["representations"][33][0]
    return r[1:1 + len(seq)].float().cpu().numpy()
myo_emb = {"WT": myo["wt_emb"], "L112R": myo["mut_emb"],
           "L112K": embed(WT[:11] + "K" + WT[12:]), "neutral@40A": embed(WT[:40] + "A" + WT[41:])}
myo_seq = {"WT": WT, "L112R": WT[:11] + "R" + WT[12:], "L112K": WT[:11] + "K" + WT[12:], "neutral@40A": WT[:40] + "A" + WT[41:]}
del em; torch.cuda.empty_cache()

# ── models ──
class SimpleMean(nn.Module):
    def __init__(s, d=1280, h=64): super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 4 * NPOS))
    def forward(s, h, oh=None): return s.net(h.mean(0)).view(NPOS, 4)
class SimpleAttn(nn.Module):
    def __init__(s, d=1280, h=64):
        super().__init__(); s.q = nn.Parameter(torch.randn(NPOS, h) * 0.02); s.k = nn.Linear(d, h); s.v = nn.Linear(d, 4)
    def forward(s, h, oh=None):
        a = F.softmax(s.q @ s.k(h).T, -1); return (a @ s.v(h))                # (NPOS,4)
def make(name):
    if name == "simple_esm_mean": return SimpleMean()
    if name == "simple_esm_attn": return SimpleAttn()
    return RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=1, use_second_shell=False)

def fwd(m, name, h, oh, noAA=False):
    if name.startswith("simple"): return m(h)                                # (NPOS,4)
    if noAA: oh = torch.zeros_like(oh)
    return m(h, oh)                                                          # (NPOS,4)

def train(name, equiv=False, noAA=False, epochs=120):
    torch.manual_seed(0); m = make(name).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); core = slice(CA, CB)
    for ep in range(epochs):
        random.shuffle(tr)
        for ex in tr:
            z = fwd(m, name, ex["h"], ex["oh"], noAA=noAA).t(); s, Lg = ex["s"], ex["Lg"]
            loss = -(ex["Pfull"][:, s:s + Lg] * F.log_softmax(z[:, s:s + Lg], 0)).sum(0).mean()
            if equiv:
                ex2 = tr[random.randrange(len(tr))]; z2 = fwd(m, name, ex2["h"], ex2["oh"], noAA=noAA).t()
                dp = cent(z2)[:, core] - cent(z)[:, core]
                dt = (cent(torch.log(ex2["Pfull"] + 1e-6)) - cent(torch.log(ex["Pfull"] + 1e-6)))[:, core]
                loss = loss + 2.0 * F.smooth_l1_loss(dp, dt)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m

@torch.no_grad()
def pred(m, name, emb, seq, noAA=False):
    z = fwd(m, name, torch.tensor(emb).float().to(dev), onehot(seq).to(dev), noAA).t()
    return z.cpu().numpy(), F.softmax(z, 0).cpu().numpy()

def evaluate(m, name, noAA=False):
    # held-out WT covR
    covr = np.mean([align_corr(pred(m, name, embs[i], seqs[i], noAA)[1], pwms[i]) for i in ho_idx])
    # MyoD1 switch + direction
    zw, Pw = pred(m, name, myo_emb["WT"], myo_seq["WT"], noAA); zm, Pm = pred(m, name, myo_emb["L112R"], myo_seq["L112R"], noAA)
    s, rf = find_ebox(Pw); gw, gm = (rc(Pw) if rf else Pw), (rc(Pm) if rf else Pm)
    ebW = "".join(BASES[i] for i in gw[:, s:s + 6].argmax(0)); ebM = "".join(BASES[i] for i in gm[:, s:s + 6].argmax(0))
    dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    zgw = (rc(zw) if rf else zw); zgm = (rc(zm) if rf else zm)
    d2 = zgm[:, s + 2] - zgm[:, s + 2].mean() - (zgw[:, s + 2] - zgw[:, s + 2].mean())
    d3 = zgm[:, s + 3] - zgm[:, s + 3].mean() - (zgw[:, s + 3] - zgw[:, s + 3].mean())
    pos2_ok = bool(d2[1] > 0 and d2[2] < 0)          # C up, G down
    pos3_ok = bool(d3[2] > 0 and d3[1] < 0)          # G up, C down
    pwmdist = float(np.abs(gm[:, s:s + 6] - gw[:, s:s + 6]).sum())
    # neutral
    _, Pn = pred(m, name, myo_emb["neutral@40A"], myo_seq["neutral@40A"], noAA)
    dnu = (motif_score(Pn, "CACGTG") - motif_score(Pn, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    a, b = ic_core(Pw); dpred_neu = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pn[:, a:b].ravel())[0, 1]
    # native-pair signed-delta (held-out)
    core = slice(CA, CB); random.seed(7); pairs = list(itertools.combinations(ho_idx, 2)); random.shuffle(pairs); pairs = pairs[:300]
    dc, da, mr = [], [], []
    for i, j in pairs:
        z1 = pred(m, name, embs[i], seqs[i], noAA)[0]; z2 = pred(m, name, embs[j], seqs[j], noAA)[0]
        P1, P2 = frame(pwms[i])[0], frame(pwms[j])[0]
        dp = (cent(z2) - cent(z1))[:, core]; dt = (cent(np.log(P2 + 1e-6)) - cent(np.log(P1 + 1e-6)))[:, core]
        if dt.std() < 1e-4: continue
        dc.append(np.corrcoef(dp.ravel(), dt.ravel())[0, 1]); da.append(float((dp * dt).sum() > 0))
        mr.append(np.linalg.norm(dp) / (np.linalg.norm(dt) + 1e-6))
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return dict(name=name, wt_covr=float(covr), myod1_wt=ebW, myod1_mut=ebM, dswitch=float(dsw),
                pos2_correct=pos2_ok, pos3_correct=pos3_ok, pwm_dist=pwmdist,
                neutral_dswitch=float(dnu), neutral_dpred=float(dpred_neu),
                native_delta_corr=float(np.nanmean(dc)), native_dir_acc=float(np.mean(da)),
                native_mag_ratio=float(np.nanmean(mr)), trainable_params=int(ntr))

# baselines (no train): WT-copy, FamilyCode
def baseline_family():
    covr = np.mean([align_corr(ref, pwms[i]) for i in ho_idx])
    return dict(name="FamilyCode", wt_covr=float(covr), myod1_wt="(family-avg)", myod1_mut="(≡WT)",
                dswitch=0.0, pos2_correct=False, pos3_correct=False, pwm_dist=0.0,
                neutral_dswitch=0.0, neutral_dpred=0.0, native_delta_corr=0.0, native_dir_acc=0.5,
                native_mag_ratio=0.0, trainable_params=0)

results = [dict(name="WT-copy", wt_covr=float("nan"), myod1_wt="(=truth)", myod1_mut="(≡WT)", dswitch=0.0,
                pos2_correct=False, pos3_correct=False, pwm_dist=0.0, neutral_dswitch=0.0, neutral_dpred=0.0,
                native_delta_corr=0.0, native_dir_acc=0.5, native_mag_ratio=0.0, trainable_params=0),
           baseline_family()]
plan = [("simple_esm_mean", dict()), ("simple_esm_attn", dict()),
        ("recog_full", dict()), ("recog_equiv", dict(equiv=True)), ("recog_noAA", dict(noAA=True))]
for nm, kw in plan:
    base = "recog" if nm.startswith("recog") else nm
    mdl = train(base, equiv=kw.get("equiv", False), noAA=kw.get("noAA", False))
    r = evaluate(mdl, base, noAA=kw.get("noAA", False))
    r["name"] = nm; results.append(r)
    print(f"[{nm}] covR={r['wt_covr']:.3f} MyoD1 {r['myod1_wt']}->{r['myod1_mut']} Δsw={r['dswitch']:+.2f} "
          f"pos2ok={r['pos2_correct']} pos3ok={r['pos3_correct']} | native dir={r['native_dir_acc']:.2f} "
          f"Δcorr={r['native_delta_corr']:.2f} mag={r['native_mag_ratio']:.2f} | neutralΔsw={r['neutral_dswitch']:+.2f} "
          f"| params={r['trainable_params']:,}", flush=True)

os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(results, open("results/mutation_benchmark/audit_ablation.json", "w"), indent=1)
print("\nsaved results/mutation_benchmark/audit_ablation.json")
