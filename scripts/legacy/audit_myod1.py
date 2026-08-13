#!/usr/bin/env python
"""Audit Section I: DIRECT MyoD1 WT vs L112R output inspection (no summary-only scores).
Model = integrated Phase-8 recognition-energy decoder on v24 features (the one that gave
Δ_switch +2.42). Prints full predicted PWM, E-box window, central-column A/C/G/T probs +
centered logits, per-column signed logit delta, PWM distance, central GC->CG switch, and
normal vs ORACLE contact. Verifies whether +2.42 is correct direction/position/magnitude.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; BASES = "ACGT"; AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; NPOS = 24
FEATS = "/data1/leihuang/TFScope/phase8_feats.npz"; CK = "/data1/leihuang/TFScope/phase8_recog_energy.pt"
np.set_printoptions(precision=2, suppress=True)

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
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

sd = torch.load(CK, map_location=dev)
m = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=2, use_second_shell=False).to(dev).eval()
m.load_state_dict(sd["model"])
D = np.load(FEATS, allow_pickle=True); myo = D["myo"][0]

@torch.no_grad()
def run(feat, seq, oracle=None):
    ob = None
    if oracle is not None:
        ob = torch.zeros(len(seq), device=dev)
        for i in oracle:
            if i < len(seq): ob[i] = 4.0
    z = m(torch.tensor(feat).float().to(dev), onehot(seq).to(dev), fam_id=0, oracle_bias=ob).t()
    return z.cpu().numpy(), F.softmax(z, 0).cpu().numpy()

def inspect(tag, oracle):
    zw, Pw = run(myo["wt_feat"], myo["wt_seq"], oracle)
    zm, Pm = run(myo["mut_feat"], myo["mut_seq"], oracle)
    s, rf = find_ebox(Pw)
    gw, gm = (rc(Pw) if rf else Pw), (rc(Pm) if rf else Pm)
    zgw, zgm = (rc(zw) if rf else zw), (rc(zm) if rf else zm)
    ebW = "".join(BASES[i] for i in gw[:, s:s + 6].argmax(0)); ebM = "".join(BASES[i] for i in gm[:, s:s + 6].argmax(0))
    print(f"\n===== {tag} =====")
    print(f"WT  E-box(window {s}) consensus = {ebW}   (want CAGCTG)")
    print(f"MUT E-box(window {s}) consensus = {ebM}   (want CACGTG)")
    for pos, lbl in [(2, "central-1 (WT G->C)"), (3, "central-2 (WT C->G)")]:
        pw = gw[:, s + pos]; pm = gm[:, s + pos]
        czw = zgw[:, s + pos] - zgw[:, s + pos].mean(); czm = zgm[:, s + pos] - zgm[:, s + pos].mean()
        print(f"  {lbl}:")
        print(f"    WT  prob ACGT={pw}  centered-logit={czw}")
        print(f"    MUT prob ACGT={pm}  centered-logit={czm}")
        print(f"    Δcentered-logit ACGT={czm-czw}")
    # per-column signed logit delta over E-box 6mer
    dz = (zgm - zgw)[:, s:s + 6]
    print(f"  per-column |Δlogit| over E-box: {np.abs(dz).sum(0)}")
    print(f"  PWM L1 distance (E-box) = {np.abs(gm[:,s:s+6]-gw[:,s:s+6]).sum():.3f}")
    dsw = (motif_score(Pm, 'CACGTG') - motif_score(Pm, 'CAGCTG')) - (motif_score(Pw, 'CACGTG') - motif_score(Pw, 'CAGCTG'))
    print(f"  S_WT(CAGCTG)={motif_score(Pw,'CAGCTG'):.2f} S_WT(CACGTG)={motif_score(Pw,'CACGTG'):.2f}")
    print(f"  S_MUT(CAGCTG)={motif_score(Pm,'CAGCTG'):.2f} S_MUT(CACGTG)={motif_score(Pm,'CACGTG'):.2f}")
    print(f"  Δ_switch(->CACGTG) = {dsw:+.2f}")
    return dict(tag=tag, wt_ebox=ebW, mut_ebox=ebM, dswitch=float(dsw))

print("MyoD1 WT vs L112R — integrated Phase-8 decoder on v24 features")
print("NOTE gate/span: recog-energy decoder has NO separate gate; motif = softmax over NPOS frame (span from IC).")
r1 = inspect("normal contact (learned)", None)
r2 = inspect("ORACLE contact (recognition residues 0..14)", list(range(15)))
json.dump([r1, r2], open("results/mutation_benchmark/audit_myod1.json", "w"), indent=1)
print("\nsaved results/mutation_benchmark/audit_myod1.json")
