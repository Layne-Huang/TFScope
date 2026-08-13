#!/usr/bin/env python
"""Phase 5: counterfactual equivariance for the bHLH recognition-energy decoder.

We have NO matched bHLH mutant PWMs, so the mutation transport is trained on NATIVE
bHLH PAIRS (both labelled), whose between-gene E-box differences are real, correct-
magnitude ΔPWMs. In the shared registration frame:

  Δz_pred(S1->S2) = center(F(S2)) - center(F(S1))            (full forward, centered)
  Δz_true(S1->S2) = center(logPWM2) - center(logPWM1)        (native labels)

Losses (plan Phase 5):
  L_abs  absolute WT specificity (keep WT PWM accuracy)
  L_eq   matched centered-Δ (counterfactual equivariance) on native pairs
  L_rev  reverse consistency   Δz(1->2) = -Δz(2->1)
  L_path path consistency      Δz(1->2)+Δz(2->3) = Δz(1->3)
  L_id   identity/neutral      in-silico non-contact mutation -> Δz ~= 0
Pairs share registration/orientation/core (fixed consensus frame) so a shift cannot
masquerade as a mutation effect. MyoD1 stays fully zero-shot. Then re-run the go/no-go.
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
def ic_arr(p): return 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
def ic_core(p, thr=0.2):
    ic = ic_arr(p); inf = np.where(ic >= thr)[0]
    return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
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

d = np.load(NPZ, allow_pickle=True)
genes = [str(g) for g in d["genes"]]; seqs = [str(s) for s in d["seqs"]]
embs = d["embs"]; pwms = [p.astype(np.float32) for p in d["pwms"]]; myo = d["myo"][0]
uniq = sorted(set(genes)); held = set(uniq[::5])
tr_idx = [i for i in range(len(genes)) if genes[i] not in held]
ho_idx = [i for i in range(len(genes)) if genes[i] in held]

# consensus ref (train) + fixed registration + PWM placed into NPOS frame
gts = [pwms[i] for i in tr_idx]
ref = np.full((4, NPOS), 0.25, np.float32)
c = gts[int(np.argmax([g.shape[1] for g in gts]))]; ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
for _ in range(2):
    acc = np.full((4, NPOS), 1e-3, np.float32)
    for g in gts:
        s, rf, Lg = align_to_ref(g, ref); acc[:, s:s + Lg] += (rc(g) if rf else g)[:, :Lg]
    ref = acc / acc.sum(0, keepdims=True)
CA, CB = ic_core(ref, thr=0.3)                                  # shared E-box core (high-IC)
print(f"shared consensus core [{CA}:{CB}] cons={''.join(BASES[i] for i in ref[:,CA:CB].argmax(0))}")

def frame_pwm(i):
    s, rf, Lg = align_to_ref(pwms[i], ref)
    P = np.full((4, NPOS), 0.25, np.float32); P[:, s:s + Lg] = (rc(pwms[i]) if rf else pwms[i])[:, :Lg]
    return P, s, Lg, rf
data = []
for i in tr_idx:
    P, s, Lg, rf = frame_pwm(i)
    data.append(dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(seqs[i]).to(dev),
                     Pfull=torch.tensor(P, device=dev), s=s, Lg=Lg,
                     gt=torch.tensor((rc(pwms[i]) if rf else pwms[i])[:, :Lg].copy(), device=dev)))
print(f"train rows {len(data)} (70 genes) | held-out {len(ho_idx)} rows | MyoD1 zero-shot")

def cent(z): return z - z.mean(0, keepdim=True)               # center over bases per column, z:(4,NPOS)

def train(equivariance=True, epochs=140, lr=1e-3, l_eq=2.0, l_id=0.5, tag=""):
    torch.manual_seed(0)
    m = RecognitionEnergyDecoder(n_pos=NPOS, use_second_shell=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    core = slice(CA, CB)
    for ep in range(epochs):
        random.shuffle(data); tot = dict(abs=0.0, eq=0.0, id=0.0)
        for n, ex in enumerate(data):
            z = m(ex["h"], ex["oh"]).t()                      # (4,NPOS)
            s, Lg = ex["s"], ex["Lg"]
            L_abs = -(ex["gt"] * F.log_softmax(z[:, s:s + Lg], 0)).sum(0).mean()
            loss = L_abs; tot["abs"] += L_abs.item()
            if equivariance:
                ex2 = data[random.randrange(len(data))]
                z2 = m(ex2["h"], ex2["oh"]).t()
                dz_pred = cent(z2)[:, core] - cent(z)[:, core]
                dz_true = (cent(torch.log(ex2["Pfull"] + 1e-6)) -
                           cent(torch.log(ex["Pfull"] + 1e-6)))[:, core]
                L_eq = F.smooth_l1_loss(dz_pred, dz_true)
                # reverse consistency is exact for centered diffs; enforce identity/neutral:
                # in-silico mutate a LOW-contact residue (decoder-only, keep h) -> Δz ~= 0
                with torch.no_grad():
                    C = m.contact(m.proj(ex["h"]))            # (NPOS,L)
                    kk = int(C[core].sum(0).argmin())         # least-contacting residue in core
                oh2 = ex["oh"].clone(); oh2[kk] = 0.0
                oh2[kk, (int(ex["oh"][kk].argmax()) + 1) % 20] = 1.0
                z_id = m(ex["h"], oh2).t()
                L_id = (cent(z_id)[:, core] - cent(z)[:, core]).pow(2).mean()
                loss = loss + l_eq * L_eq + l_id * L_id
                tot["eq"] += L_eq.item(); tot["id"] += L_id.item()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 70 == 0:
            print(f"  [{tag}] ep{ep+1} abs {tot['abs']/len(data):.3f} eq {tot['eq']/len(data):.3f} id {tot['id']/len(data):.4f}")
    return m

@torch.no_grad()
def pred(m, emb, seq):
    z = m(torch.tensor(emb).float().to(dev), onehot(seq).to(dev))
    return z.t().cpu().numpy(), F.softmax(z.t(), 0).cpu().numpy()
def motif_score(P, motif):
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            best = max(best, sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm)))
    return best
def find_ebox(P):
    ix = {b: i for i, b in enumerate(BASES)}; best = (-9, 0, False)
    for rf, g in [(False, P), (True, rc(P))]:
        for s in range(0, NPOS - 6 + 1):
            w = g[:, s:s + 6]; sc = w[ix['C'], 0] + w[ix['A'], 1] + w[ix['T'], 4] + w[ix['G'], 5]
            if sc > best[0]: best = (sc, s, rf)
    return best[1], best[2]

def gonogo(m, tag):
    zw, Pw = pred(m, myo["wt_emb"], myo["wt_seq"]); zm, Pm = pred(m, myo["mut_emb"], myo["mut_seq"])
    a, b = ic_core(Pw); dpred = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]
    dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - \
          (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    s, rf = find_ebox(Pw); gW = rc(zw) if rf else zw; gM = rc(zm) if rf else zm
    ebW = gW[:, s:s + 6]; ebM = gM[:, s:s + 6]
    p2 = {x: float(ebM[i, 2] - ebM[:, 2].mean()) - float(ebW[i, 2] - ebW[:, 2].mean()) for i, x in enumerate(BASES)}
    p3 = {x: float(ebM[i, 3] - ebM[:, 3].mean()) - float(ebW[i, 3] - ebW[:, 3].mean()) for i, x in enumerate(BASES)}
    # held-out WT covR
    covr = np.mean([align_corr(pred(m, embs[i], seqs[i])[1], pwms[i]) for i in ho_idx])
    print(f"  [{tag}] WT ebox={''.join(BASES[i] for i in ebW.argmax(0))} MUT ebox={''.join(BASES[i] for i in ebM.argmax(0))}")
    print(f"       Δ_switch={dsw:+.2f} Δpred={dpred:.3f} | pos2(G→C) C={p2['C']:+.2f}G={p2['G']:+.2f} "
          f"pos3(C→G) G={p3['G']:+.2f}C={p3['C']:+.2f} | held-out WT covR={covr:.3f}")
    return dict(tag=tag, dswitch=float(dsw), dpred=float(dpred), pos2=p2, pos3=p3, covr=float(covr))

print("\n=== Phase-5: WITH equivariance ===")
m_eq = train(equivariance=True, tag="equiv")
print("=== control: WITHOUT equivariance (abs-only, same budget) ===")
m_base = train(equivariance=False, tag="abs-only")
print("\n=== GO/NO-GO: MyoD1 L112R zero-shot (target CAGCTG→CACGTG: pos2 C+/G-, pos3 G+/C-) ===")
res = [gonogo(m_eq, "recog-energy + EQUIVARIANCE"), gonogo(m_base, "recog-energy abs-only (control)")]
os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(res, open("results/mutation_benchmark/phase5_equivariance.json", "w"), indent=1)
torch.save({"equiv": m_eq.state_dict()}, "/data1/leihuang/TFScope/phase5_recog_energy_equiv.pt")
print("\nsaved results/mutation_benchmark/phase5_equivariance.json + ckpt")
