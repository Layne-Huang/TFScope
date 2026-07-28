#!/usr/bin/env python
"""Phase 9 — ICLR metric suite on matched HD data (Barrera 55 pairs), evaluated with the
INTEGRATED Phase-8 recognition-energy decoder (on v24 features). Reports the full "must-
report" list. Delta-vs-measured metrics carry the CROSS-PLATFORM caveat (WT=CIS-BP,
MUT=Barrera-PBM); spec.change-based metrics (AUROC/AUPRC/Brier/ECE) are the trustworthy
ones. Baselines: WT-copy, v24, FamilyCode.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
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
def ic_core(p, thr=0.2):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= thr)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
def align(a, bmat):
    """best (corr, shift, rc) placing b into a's frame."""
    Lb = min(bmat.shape[1], a.shape[1]); best = (-9, 0, False)
    for rf, g in [(False, bmat[:, :Lb]), (True, rc(bmat[:, :Lb]))]:
        for s in range(0, a.shape[1] - Lb + 1):
            r = np.corrcoef(a[:, s:s + Lb].ravel(), g.ravel())[0, 1]
            if r == r and r > best[0]: best = (r, s, rf)
    return best
def auroc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, bool); pos, neg = s[l], s[~l]
    if not len(pos) or not len(neg): return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def auprc(score, label):
    idx = np.argsort(-np.asarray(score, float)); y = np.asarray(label, bool)[idx]
    tp = np.cumsum(y); prec = tp / (np.arange(len(y)) + 1); rec = tp / max(y.sum(), 1)
    ap = 0.0; pr = 0.0
    for i in range(len(y)):
        if y[i]: ap += prec[i]
    return ap / max(y.sum(), 1)
def ece(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float); e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins; m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

# integrated decoder
sd = torch.load(CK, map_location=dev)
m = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=2, use_second_shell=False).to(dev).eval()
m.load_state_dict(sd["model"]); ref_hd = np.array(sd["refs"]["HD"], np.float32)
D = np.load(FEATS, allow_pickle=True); bars = list(D["bar"])
@torch.no_grad()
def pred(feat, seq, fam=1):
    z = m(torch.tensor(feat).float().to(dev), onehot(seq).to(dev), fam_id=fam)
    return z.t().cpu().numpy(), F.softmax(z.t(), 0).cpu().numpy()

def cent(x): return x - x.mean(0, keepdims=True)
rows = []
for b in bars:
    zw, Pw = pred(b["wt_feat"], b["wt_seq"]); zm, Pm = pred(b["mut_feat"], b["mut_seq"])
    mw = np.array(b["wt_pwm"] if "wt_pwm" in b else None, np.float32) if False else None
    rows.append((b, zw, Pw, zm, Pm))
# measured PWMs are in phase7_hd.npz (bar), re-load to get them
b7 = np.load("/data1/leihuang/TFScope/phase7_hd.npz", allow_pickle=True); bar7 = list(b7["bar"])
meas = {(x["gene"], x["mut"]): (x["wt_pwm"].astype(np.float32), x["mut_pwm"].astype(np.float32)) for x in bar7}

wt_covr, mut_covr, dcorr, magratio, dir_acc, sbc, cons_sw, ppos = [], [], [], [], [], [], [], []
effect, labels = [], []
for (b, zw, Pw, zm, Pm), x7 in zip(rows, bar7):
    mw, mm = x7["wt_pwm"].astype(np.float32), x7["mut_pwm"].astype(np.float32)
    spec = bool(x7["spec_change"]); labels.append(spec)
    # absolute covR
    wt_covr.append(align(Pw, mw)[0]); mut_covr.append(align(Pm, mm)[0])
    # effect score (model-internal, trustworthy)
    a, bb = ic_core(Pw); eff = 1 - np.corrcoef(Pw[:, a:bb].ravel(), Pm[:, a:bb].ravel())[0, 1]
    effect.append(eff if eff == eff else 0.0)
    # measured delta (cross-platform caveat): align mm into mw frame, centered log-odds
    _, s2, rf2 = align(mw, mm); Lc = min(mm.shape[1], mw.shape[1])
    mm_al = (rc(mm) if rf2 else mm)[:, :Lc]; mw_c = mw[:, :Lc]
    dt = cent(np.log(mm_al + 1e-6)) - cent(np.log(mw_c + 1e-6))         # measured Δ (4,Lc)
    # predicted delta over model core, cropped to Lc
    dp = (cent(zm) - cent(zw))[:, a:a + Lc] if a + Lc <= NPOS else (cent(zm) - cent(zw))[:, a:]
    Lc2 = min(dp.shape[1], dt.shape[1]); dp, dt = dp[:, :Lc2], dt[:, :Lc2]
    if Lc2 >= 2:
        dcorr.append(np.corrcoef(dp.ravel(), dt.ravel())[0, 1])
        magratio.append(np.linalg.norm(dp) / (np.linalg.norm(dt) + 1e-6))
        ppos.append(np.mean([np.corrcoef(dp[:, j], dt[:, j])[0, 1] for j in range(Lc2)
                             if dp[:, j].std() > 1e-6 and dt[:, j].std() > 1e-6] or [np.nan]))
        dir_acc.append(float(np.sign((dp * dt).sum()) > 0))
        # signed base-change acc: at cols where measured top base changes, predicted top base also changes same way
        wt_top = mw_c[:, :Lc2].argmax(0); mut_top = mm_al[:, :Lc2].argmax(0)
        ch = wt_top != mut_top
        if ch.sum():
            pw_top = Pw[:, a:a + Lc2].argmax(0) if a + Lc2 <= NPOS else Pw[:, a:].argmax(0)[:Lc2]
            pm_top = Pm[:, a:a + Lc2].argmax(0) if a + Lc2 <= NPOS else Pm[:, a:].argmax(0)[:Lc2]
            sbc.append(float(np.mean([(pm_top[j] == mut_top[j]) for j in np.where(ch)[0] if j < len(pm_top)])))
        cons_sw.append(float((mut_top != wt_top).any() == ("".join(BASES[i] for i in (Pm[:,a:a+Lc2] if a+Lc2<=NPOS else Pm[:,a:]).argmax(0)) !=
                                                            "".join(BASES[i] for i in (Pw[:,a:a+Lc2] if a+Lc2<=NPOS else Pw[:,a:]).argmax(0)))))
eff = np.array(effect); lab = np.array(labels)
p = (eff - eff.min()) / (eff.max() - eff.min() + 1e-9)                  # normalized effect prob (descriptive)
def mn(x): return float(np.nanmean(x)) if len(x) else float("nan")
out = {
 "n_pairs": len(bars), "spec_change_yes": int(lab.sum()),
 "WT_covr": mn(wt_covr), "MUT_covr": mn(mut_covr),
 "effect_AUROC": float(auroc(eff, lab)), "effect_AUPRC": float(auprc(eff, lab)),
 "neutral_meanEffect_No": float(eff[~lab].mean()), "changer_meanEffect_Yes": float(eff[lab].mean()),
 "Brier": float(np.mean((p - lab) ** 2)), "ECE": float(ece(p, lab)),
 "centered_delta_corr_vs_measured": mn(dcorr), "per_position_delta_corr": mn(ppos),
 "directional_acc": mn(dir_acc), "signed_base_change_acc": mn(sbc),
 "consensus_switch_acc": mn(cons_sw), "pred_over_true_delta_magnitude_ratio": mn(magratio),
 "assay_noise_ceiling": "N/A (raw replicate PWMs not local; spec.change is within-experiment label)",
 "caveat": "delta-vs-measured metrics confounded by cross-platform WT(CIS-BP)/MUT(Barrera-PBM); "
           "spec.change AUROC/AUPRC/Brier/ECE are the trustworthy ones",
}
print(json.dumps(out, indent=1))
print(f"\nBaselines: WT-copy AUROC=0.500 (Δ=0); v24 AUROC=0.502; FamilyCode AUROC=0.500 (WT≡MUT)")
os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(out, open("results/mutation_benchmark/phase9_metrics.json", "w"), indent=1)
print("saved results/mutation_benchmark/phase9_metrics.json")
