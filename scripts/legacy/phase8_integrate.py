#!/usr/bin/env python
"""Phase 8 Stage-1 (detached): fold the recognition-energy decoder into full TFScope.

Stage 1 = "train only the new decoder": run RecognitionEnergyDecoder on v24's OWN frozen
encoder features (post-MoE residue reps, 1280-d) instead of raw ESM -> this REPLACES the
free PWMHeadV18 regression path while keeping v24's encoder / N-chain / span gate. Losses:
  L_abs   native PWM (measured)                       -- absolute WT specificity
  L_eq    counterfactual equivariance on native pairs -- mutation transport
  L_dist  KL to v24's predicted WT PWM (distillation) -- no regression vs v24
Combined bHLH + HD (family prior n_fam=2, low capacity). Eval preserves all 3 mechanisms:
held-out WT covR (recog vs v24), MyoD1 L112R switch (bHLH), Barrera AUROC (HD). Zero mutant
labels in training. Encoder-unfreezing (stages 2-4) is left for an attended run.
Idempotent: skips the v24 feature precompute if phase8_feats.npz exists. Writes DONE flag.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings, random
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"
FEATS = "/data1/leihuang/TFScope/phase8_feats.npz"
CKOUT = "/data1/leihuang/TFScope/phase8_recog_energy.pt"
RES = "results/mutation_benchmark/phase8_integrate.json"
DONE = "results/mutation_benchmark/phase8_DONE.flag"
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
def auroc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, bool); pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    o = np.argsort(np.concatenate([pos, neg])); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def crop_ic(p):
    a, b = ic_core(p); b = min(b, a + NPOS); return p[:, a:b]

# ══ PART 1: precompute v24 encoder features + v24 predicted PWM ══
if not os.path.exists(FEATS):
    print("[precompute] loading v24 ...", flush=True)
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    from tfscope.data.dataset import AA_TO_TOKEN
    CK = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
    cfg = TFScopeConfig()
    for k, v in json.load(open(CK + "/config.json")).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    vm = TFScopeModel(cfg).to(dev).eval(); vm.use_contact_pred_head = False
    vm.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
    cap = {}
    vm.residue_moe.register_forward_hook(lambda m, i, o: cap.__setitem__("f", (o[0] if isinstance(o, tuple) else o).detach()))
    @torch.no_grad()
    def v24(seq):
        t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
        dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
        _, pl, _ = vm(t, dm, fi)
        return cap["f"][0].float().cpu().numpy(), F.softmax(pl[0], 0).float().cpu().numpy()
    b3 = np.load("/data1/leihuang/TFScope/phase3_bhlh.npz", allow_pickle=True)
    b7 = np.load("/data1/leihuang/TFScope/phase7_hd.npz", allow_pickle=True)
    rows = []
    src = [("bHLH", b3["seqs"], b3["pwms"]), ("HD", b7["seqs"], b7["pwms"])]
    tot = sum(len(s) for _, s, _ in src); n = 0
    for fam, seqs, pwms in src:
        for i in range(len(seqs)):
            f, vp = v24(str(seqs[i]))
            rows.append(dict(fam=fam, seq=str(seqs[i]), feat=f, meas=pwms[i].astype(np.float32), v24=crop_ic(vp)))
            n += 1
            if n % 150 == 0: print(f"[precompute] {n}/{tot}", flush=True)
    myo = b3["myo"][0]
    wf, wv = v24(myo["wt_seq"]); mf, mv = v24(myo["mut_seq"])
    myo_out = dict(wt_seq=myo["wt_seq"], mut_seq=myo["mut_seq"], wt_feat=wf, mut_feat=mf)
    bars = []
    for b in list(b7["bar"]):
        wf, _ = v24(b["wt_seq"]); mf, _ = v24(b["mut_seq"])
        bars.append(dict(wt_feat=wf, mut_feat=mf, wt_seq=b["wt_seq"], mut_seq=b["mut_seq"], spec=bool(b["spec_change"])))
    def to_obj(l):
        a = np.empty(len(l), dtype=object)
        for i, x in enumerate(l): a[i] = x
        return a
    np.savez_compressed(FEATS, rows=to_obj(rows), myo=to_obj([myo_out]), bar=to_obj(bars))
    print(f"[precompute] saved {FEATS} ({len(rows)} rows)", flush=True)
    del vm; torch.cuda.empty_cache()

# ══ PART 2: train recognition-energy decoder on v24 features ══
D = np.load(FEATS, allow_pickle=True)
rows = list(D["rows"]); myo = D["myo"][0]; bars = list(D["bar"])
FAM = {"bHLH": 0, "HD": 1}
# per-family consensus + registration
refs = {}
for fam in FAM:
    gts = [r["meas"] for r in rows if r["fam"] == fam]
    ref = np.full((4, NPOS), 0.25, np.float32); c = gts[int(np.argmax([g.shape[1] for g in gts]))]
    ref[:, :min(c.shape[1], NPOS)] = c[:, :NPOS]
    for _ in range(2):
        acc = np.full((4, NPOS), 1e-3, np.float32)
        for g in gts:
            s, rf, Lg = align_to_ref(g, ref); acc[:, s:s + Lg] += (rc(g) if rf else g)[:, :Lg]
        ref = acc / acc.sum(0, keepdims=True)
    refs[fam] = ref
# gene-held-out (~15%) by row index deterministically per family
random.seed(1)
ho = set()
for fam in FAM:
    idx = [i for i in range(len(rows)) if rows[i]["fam"] == fam]; random.shuffle(idx)
    ho |= set(idx[:max(1, len(idx) // 7)])
def frame(p, fam):
    s, rf, Lg = align_to_ref(p, refs[fam])
    P = np.full((4, NPOS), 0.25, np.float32); P[:, s:s + Lg] = (rc(p) if rf else p)[:, :Lg]
    return P, s, Lg, rf
tr = []
for i, r in enumerate(rows):
    if i in ho: continue
    P, s, Lg, rf = frame(r["meas"], r["fam"]); Pv, _, _, _ = frame(r["v24"], r["fam"])
    tr.append(dict(fam=FAM[r["fam"]], famname=r["fam"],
                   h=torch.tensor(r["feat"]).float().to(dev), oh=onehot(r["seq"]).to(dev),
                   Pfull=torch.tensor(P, device=dev), Pv24=torch.tensor(Pv, device=dev), s=s, Lg=Lg))
print(f"[train] rows {len(tr)} | held-out {len(ho)} | families {[ (f, sum(1 for t in tr if t['famname']==f)) for f in FAM]}", flush=True)
cores = {f: ic_core(refs[f], 0.3) for f in FAM}
def cent(z): return z - z.mean(0, keepdim=True)

m = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=2, use_second_shell=False).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
EP = 100; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
by_fam = {0: [t for t in tr if t["fam"] == 0], 1: [t for t in tr if t["fam"] == 1]}
for ep in range(EP):
    random.shuffle(tr); agg = dict(abs=0., eq=0., dist=0.)
    for ex in tr:
        z = m(ex["h"], ex["oh"], fam_id=ex["fam"]).t(); s, Lg = ex["s"], ex["Lg"]
        L_abs = -(ex["Pfull"][:, s:s + Lg] * F.log_softmax(z[:, s:s + Lg], 0)).sum(0).mean()
        CA, CB = cores[ex["famname"]]; core = slice(CA, CB)
        # distillation: recog WT core -> v24 WT core
        L_dist = F.kl_div(F.log_softmax(z[:, core], 0), ex["Pv24"][:, core], reduction="batchmean")
        # equivariance: within-family native pair
        ex2 = by_fam[ex["fam"]][random.randrange(len(by_fam[ex["fam"]]))]
        z2 = m(ex2["h"], ex2["oh"], fam_id=ex2["fam"]).t()
        dz_pred = cent(z2)[:, core] - cent(z)[:, core]
        dz_true = (cent(torch.log(ex2["Pfull"] + 1e-6)) - cent(torch.log(ex["Pfull"] + 1e-6)))[:, core]
        L_eq = F.smooth_l1_loss(dz_pred, dz_true)
        loss = L_abs + 2.0 * L_eq + 0.3 * L_dist
        opt.zero_grad(); loss.backward(); opt.step()
        agg["abs"] += float(L_abs); agg["eq"] += float(L_eq); agg["dist"] += float(L_dist)
    sch.step()
    if (ep + 1) % 25 == 0:
        print(f"[train] ep{ep+1} abs {agg['abs']/len(tr):.3f} eq {agg['eq']/len(tr):.3f} dist {agg['dist']/len(tr):.3f}", flush=True)

@torch.no_grad()
def pred(feat, seq, fam):
    z = m(torch.tensor(feat).float().to(dev), onehot(seq).to(dev), fam_id=fam)
    return F.softmax(z.t(), 0).cpu().numpy()
def motif_score(P, motif):
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            best = max(best, sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm)))
    return best

# ── eval ──
out = {}
# (i) held-out WT covR: recog vs v24 (teacher)
re_c, v24_c = [], []
for i in ho:
    r = rows[i]; P = pred(r["feat"], r["seq"], FAM[r["fam"]])
    re_c.append(align_corr(P, r["meas"])); v24_c.append(align_corr(r["v24"], r["meas"]))
out["held_out_wt_covr"] = dict(recog=float(np.mean(re_c)), v24=float(np.mean(v24_c)), n=len(ho))
print(f"[eval] held-out WT covR: recog {np.mean(re_c):.3f} | v24 {np.mean(v24_c):.3f} (n={len(ho)})", flush=True)
# (ii) MyoD1 L112R switch (bHLH mechanism)
Pw = pred(myo["wt_feat"], myo["wt_seq"], 0); Pm = pred(myo["mut_feat"], myo["mut_seq"], 0)
dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
a, b = ic_core(Pw)
out["myod1"] = dict(wt_cons="".join(BASES[i] for i in Pw[:, a:b].argmax(0))[:6],
                    dswitch=float(dsw), dpred=float(1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]))
print(f"[eval] MyoD1: WT {out['myod1']['wt_cons']} Δ_switch={dsw:+.2f} Δpred={out['myod1']['dpred']:.3f}", flush=True)
# (iii) Barrera HD AUROC(Δpred, spec.change)
dp, lab = [], []
for b in bars:
    Pw = pred(b["wt_feat"], b["wt_seq"], 1); Pm = pred(b["mut_feat"], b["mut_seq"], 1)
    aa, bb = ic_core(Pw); v = 1 - np.corrcoef(Pw[:, aa:bb].ravel(), Pm[:, aa:bb].ravel())[0, 1]
    dp.append(v if v == v else 0.0); lab.append(b["spec"])
out["barrera_auroc"] = float(auroc(dp, lab))
print(f"[eval] Barrera AUROC(Δpred,spec.change)={out['barrera_auroc']:.3f} (v24=0.502)", flush=True)

os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(out, open(RES, "w"), indent=1)
torch.save({"model": m.state_dict(), "refs": {k: v.tolist() for k, v in refs.items()}}, CKOUT)
open(DONE, "w").write("done\n")
print(f"[done] saved {RES}, {CKOUT}, {DONE}", flush=True)
