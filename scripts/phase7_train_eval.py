#!/usr/bin/env python
"""Phase 7 (homeodomain) train + go/no-go. Train recog-energy + counterfactual
equivariance on NATIVE HD pairs (Barrera clusters + a native gene-held-out set both
excluded from training). Eval:
  (A) held-out native HD WT covR vs FamilyCode / nearest-paralog / abs-only control
  (B) 55 Barrera WT/MUT pairs ZERO-SHOT: AUROC(Δpred, spec.change) [vs v24=0.502],
      mean Δpred Yes/No, directional corr to measured Δ.
No mutant labels used in training.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings, random
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; NPZ = "/data1/leihuang/TFScope/phase7_hd.npz"
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
def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

d = np.load(NPZ, allow_pickle=True)
genes = [str(g) for g in d["genes"]]; seqs = [str(s) for s in d["seqs"]]
embs = d["embs"]; pwms = [p.astype(np.float32) for p in d["pwms"]]; bar = list(d["bar"])
uniq = sorted(set(genes)); held = set(uniq[::5])
tr_idx = [i for i in range(len(genes)) if genes[i] not in held]
ho_idx = [i for i in range(len(genes)) if genes[i] in held]

gts = [pwms[i] for i in tr_idx]
ref = np.full((4, NPOS), 0.25, np.float32)
c = gts[int(np.argmax([g.shape[1] for g in gts]))]; ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
for _ in range(2):
    acc = np.full((4, NPOS), 1e-3, np.float32)
    for g in gts:
        s, rf, Lg = align_to_ref(g, ref); acc[:, s:s + Lg] += (rc(g) if rf else g)[:, :Lg]
    ref = acc / acc.sum(0, keepdims=True)
CA, CB = ic_core(ref, thr=0.3)
print(f"HD: {len(uniq)} native genes -> train {len(uniq)-len(held)}/held-out {len(held)} "
      f"({len(tr_idx)}/{len(ho_idx)} rows); 55 Barrera zero-shot; core[{CA}:{CB}]="
      f"{''.join(BASES[i] for i in ref[:,CA:CB].argmax(0))}")

def frame_pwm(i):
    s, rf, Lg = align_to_ref(pwms[i], ref)
    P = np.full((4, NPOS), 0.25, np.float32); P[:, s:s + Lg] = (rc(pwms[i]) if rf else pwms[i])[:, :Lg]
    return P, s, Lg, rf
train_data = []
for i in tr_idx:
    P, s, Lg, rf = frame_pwm(i)
    train_data.append(dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(seqs[i]).to(dev),
                           Pfull=torch.tensor(P, device=dev), s=s, Lg=Lg,
                           gt=torch.tensor((rc(pwms[i]) if rf else pwms[i])[:, :Lg].copy(), device=dev)))
def cent(z): return z - z.mean(0, keepdim=True)

def train(equivariance, epochs=120, lr=1e-3, l_eq=2.0, l_id=0.5, tag=""):
    torch.manual_seed(0)
    m = RecognitionEnergyDecoder(n_pos=NPOS, use_second_shell=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); core = slice(CA, CB)
    for ep in range(epochs):
        random.shuffle(train_data); tot = 0.0
        for ex in train_data:
            z = m(ex["h"], ex["oh"]).t(); s, Lg = ex["s"], ex["Lg"]
            loss = -(ex["gt"] * F.log_softmax(z[:, s:s + Lg], 0)).sum(0).mean()
            if equivariance:
                ex2 = train_data[random.randrange(len(train_data))]
                z2 = m(ex2["h"], ex2["oh"]).t()
                dz_pred = cent(z2)[:, core] - cent(z)[:, core]
                dz_true = (cent(torch.log(ex2["Pfull"] + 1e-6)) - cent(torch.log(ex["Pfull"] + 1e-6)))[:, core]
                L_eq = F.smooth_l1_loss(dz_pred, dz_true)
                with torch.no_grad():
                    C = m.contact(m.proj(ex["h"])); kk = int(C[core].sum(0).argmin())
                oh2 = ex["oh"].clone(); oh2[kk] = 0.0; oh2[kk, (int(ex["oh"][kk].argmax()) + 1) % 20] = 1.0
                L_id = (cent(m(ex["h"], oh2).t())[:, core] - cent(z)[:, core]).pow(2).mean()
                loss = loss + l_eq * L_eq + l_id * L_id
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        sch.step()
        if (ep + 1) % 60 == 0: print(f"  [{tag}] ep{ep+1} loss {tot/len(train_data):.3f}")
    return m

@torch.no_grad()
def pred(m, emb, seq):
    z = m(torch.tensor(emb).float().to(dev), onehot(seq).to(dev))
    return F.softmax(z.t(), 0).cpu().numpy()

def partA(m, tag):
    tr_seqs = [(seqs[i], pwms[i]) for i in tr_idx]
    def pad(p):
        a = np.full((4, NPOS), 0.25, np.float32); a[:, :min(p.shape[1], NPOS)] = p[:, :NPOS]; return a
    re_c, fc_c, np_c = [], [], []
    for i in ho_idx:
        re_c.append(align_corr(pred(m, embs[i], seqs[i]), pwms[i])); fc_c.append(align_corr(ref, pwms[i]))
        j = max(tr_seqs, key=lambda t: seqid(seqs[i], t[0])); np_c.append(align_corr(pad(j[1]), pwms[i]))
    print(f"  (A) [{tag}] held-out HD WT covR: recog {np.mean(re_c):.3f} | FamilyCode {np.mean(fc_c):.3f} | paralog {np.mean(np_c):.3f}")
    return float(np.mean(re_c)), float(np.mean(fc_c)), float(np.mean(np_c))

def partB(m, tag):
    dpreds, labels = [], []
    for b in bar:
        Pw = pred(m, b["wt_emb"], b["wt_seq"]); Pm = pred(m, b["mut_emb"], b["mut_seq"])
        a, bb = ic_core(Pw)
        dp = 1 - np.corrcoef(Pw[:, a:bb].ravel(), Pm[:, a:bb].ravel())[0, 1]
        dpreds.append(dp if dp == dp else 0.0); labels.append(b["spec_change"])
    au = auroc(dpreds, labels); dp = np.array(dpreds); lab = np.array(labels)
    print(f"  (B) [{tag}] Barrera zero-shot: AUROC(Δpred,spec.change)={au:.3f} "
          f"(v24=0.502) | meanΔ Yes={dp[lab].mean():.4f} No={dp[~lab].mean():.4f}")
    return dict(auroc=float(au), mean_yes=float(dp[lab].mean()), mean_no=float(dp[~lab].mean()))

print("\n=== TRAIN HD: + equivariance ==="); m_eq = train(True, tag="equiv")
print("=== TRAIN HD: abs-only control ==="); m_base = train(False, tag="abs-only")
print("\n=== GO/NO-GO (HD) ===")
res = {"equiv": {}, "abs_only": {}}
res["equiv"]["A"] = partA(m_eq, "equiv"); res["equiv"]["B"] = partB(m_eq, "equiv")
res["abs_only"]["A"] = partA(m_base, "abs-only"); res["abs_only"]["B"] = partB(m_base, "abs-only")
os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(res, open("results/mutation_benchmark/phase7_hd.json", "w"), indent=1)
torch.save({"equiv": m_eq.state_dict()}, "/data1/leihuang/TFScope/phase7_hd_equiv.pt")
print("\nsaved results/mutation_benchmark/phase7_hd.json + ckpt")
