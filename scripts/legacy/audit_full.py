#!/usr/bin/env python
"""RIGOROUS AUDIT (detached). Apples-to-apples on ONE bHLH test set, all models same eval.
Priorities: (1) unified table + leakage audit + bootstrap CI + 3 seeds + gene-balanced;
(2) MyoD1 per-column resolution (dir vs top-base vs consensus); (3) strict held-out (gene
+ seq-cluster); (4) ESM embedding-intervention radius curve; (5) consolidated ablation.
NO new architecture. v24 evaluated on the SAME held-out bHLH genes via cached predictions.
Writes results/mutation_benchmark/audit_full.json incrementally.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, warnings, random, itertools
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; BASES = "ACGT"; AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; NPOS = 24
NPZ = "/data1/leihuang/TFScope/phase3_bhlh.npz"; FEATS = "/data1/leihuang/TFScope/phase8_feats.npz"
OUT = "results/mutation_benchmark/audit_full.json"
RESULT = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(RESULT, open(OUT, "w"), indent=1)

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
def boot_ci(vals, nb=2000):
    vals = np.asarray(vals); idx = np.random.RandomState(0).randint(0, len(vals), (nb, len(vals)))
    bs = vals[idx].mean(1); return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ── data ──
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
TR = [dict(h=torch.tensor(embs[i]).float().to(dev), oh=onehot(seqs[i]).to(dev),
           Pfull=torch.tensor(frame(pwms[i])[0], device=dev), s=frame(pwms[i])[1], Lg=frame(pwms[i])[2]) for i in tr_idx]

# v24 predictions on the SAME bHLH rows (match by seq)
b8 = np.load(FEATS, allow_pickle=True); v24map = {}
for r in b8["rows"]:
    if r["fam"] == "bHLH": v24map[r["seq"]] = np.array(r["v24"], np.float32)

# leakage audit
myo_wt = myo["wt_seq"]
RESULT["leakage_audit"] = {
    "n_train_rows": len(tr_idx), "n_test_rows": len(ho_idx),
    "n_train_genes": len(uniq) - len(held), "n_test_genes": len(held),
    "family": "bHLH only (this table)", "MyoD1_in_train": any(g.upper() == "MYOD1" for g in genes),
    "MyoD1_wt_seq_in_train_seqs": myo_wt in [seqs[i] for i in tr_idx],
    "note": "MyoD1 whole cluster excluded at prep (phase3_prep); test=gene-held-out bHLH; "
            "v24 covR computed on the SAME held-out seqs via cached phase8 predictions.",
    "v24_apples_note": "prior v24=0.47 was on 228 bHLH+HD rows; simple=0.87 on 96 bHLH — NOT same. "
                       "This table recomputes v24 on the identical bHLH held-out set.",
}
save()

# ── models ──
class SimpleMean(nn.Module):
    def __init__(s, d=1280, h=64): super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 4 * NPOS))
    def forward(s, h, oh=None): return s.net(h.mean(0)).view(NPOS, 4)
class SimpleAttn(nn.Module):
    def __init__(s, d=1280, h=64):
        super().__init__(); s.q = nn.Parameter(torch.randn(NPOS, h) * 0.02); s.k = nn.Linear(d, h); s.v = nn.Linear(d, 4)
    def forward(s, h, oh=None): return F.softmax(s.q @ s.k(h).T, -1) @ s.v(h)
def make(kind):
    return SimpleMean() if kind == "mean" else SimpleAttn() if kind == "attn" else \
        RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=1, use_second_shell=False)
def fwd(m, kind, h, oh, noAA=False):
    if kind in ("mean", "attn"): return m(h)
    return m(h, torch.zeros_like(oh) if noAA else oh)

def train(kind, equiv=False, noAA=False, seed=0, epochs=100):
    torch.manual_seed(seed); random.seed(seed); m = make(kind).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); core = slice(CA, CB); tr = list(TR)
    for ep in range(epochs):
        random.shuffle(tr)
        for ex in tr:
            z = fwd(m, kind, ex["h"], ex["oh"], noAA).t(); s, Lg = ex["s"], ex["Lg"]
            loss = -(ex["Pfull"][:, s:s + Lg] * F.log_softmax(z[:, s:s + Lg], 0)).sum(0).mean()
            if equiv:
                e2 = tr[random.randrange(len(tr))]; z2 = fwd(m, kind, e2["h"], e2["oh"], noAA).t()
                dp = cent(z2)[:, core] - cent(z)[:, core]
                dt = (cent(torch.log(e2["Pfull"] + 1e-6)) - cent(torch.log(ex["Pfull"] + 1e-6)))[:, core]
                loss = loss + 2.0 * F.smooth_l1_loss(dp, dt)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m
@torch.no_grad()
def P_of(m, kind, emb, seq, noAA=False):
    z = fwd(m, kind, torch.tensor(emb).float().to(dev), onehot(seq).to(dev), noAA).t()
    return z.cpu().numpy(), F.softmax(z, 0).cpu().numpy()

# MyoD1 mutants (raw ESM): embed extras
import esm as esm_lib
em, alph = esm_lib.pretrained.esm2_t33_650M_UR50D(); em = em.eval().to(dev)
for p in em.parameters(): p.requires_grad = False
bc = alph.get_batch_converter()
@torch.no_grad()
def embed(seq):
    _, _, t = bc([("x", seq)]); r = em(t.to(dev), repr_layers=[33])["representations"][33][0]
    return r[1:1 + len(seq)].float().cpu().numpy()
WT = myo["wt_seq"]
MY = {"WT": (myo["wt_emb"], WT), "L112R": (myo["mut_emb"], WT[:11] + "R" + WT[12:]),
      "neutral@40A": (embed(WT[:40] + "A" + WT[41:]), WT[:40] + "A" + WT[41:])}
del em; torch.cuda.empty_cache()

# ── per-model eval ──
def wt_covr_pergene(m, kind, noAA=False):
    per = {}
    for i in ho_idx:
        per.setdefault(genes[i], []).append(align_corr(P_of(m, kind, embs[i], seqs[i], noAA)[1], pwms[i]))
    return [np.mean(v) for v in per.values()]                    # gene-balanced list
def myod1_detail(m, kind, noAA=False, oracle=None):
    def pr(key):
        emb, sq = MY[key]
        if kind == "recog" and oracle is not None:
            ob = torch.zeros(len(sq), device=dev)
            for i in oracle:
                if i < len(sq): ob[i] = 4.0
            with torch.no_grad():
                z = m(torch.tensor(emb).float().to(dev), onehot(sq).to(dev), oracle_bias=ob).t()
            return z.cpu().numpy(), F.softmax(z, 0).cpu().numpy()
        return P_of(m, kind, emb, sq, noAA)
    zw, Pw = pr("WT"); zm, Pm = pr("L112R")
    s, rf = find_ebox(Pw); gw, gm = (rc(Pw) if rf else Pw), (rc(Pm) if rf else Pm)
    zgw, zgm = (rc(zw) if rf else zw), (rc(zm) if rf else zm)
    def col(g, z, p): return dict(prob={b: round(float(g[k, s + p]), 3) for k, b in enumerate(BASES)},
                                  cent_logit={b: round(float(z[k, s + p] - z[:, s + p].mean()), 2) for k, b in enumerate(BASES)},
                                  top=BASES[int(g[:, s + p].argmax())])
    dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    d2 = cent(zgm[:, s + 2:s + 3])[:, 0] - cent(zgw[:, s + 2:s + 3])[:, 0]
    d3 = cent(zgm[:, s + 3:s + 4])[:, 0] - cent(zgw[:, s + 3:s + 4])[:, 0]
    return dict(WT_ebox="".join(BASES[i] for i in gw[:, s:s + 6].argmax(0)),
                MUT_ebox="".join(BASES[i] for i in gm[:, s:s + 6].argmax(0)),
                pos2_WT=col(gw, zgw, 2), pos2_MUT=col(gm, zgm, 2),
                pos3_WT=col(gw, zgw, 3), pos3_MUT=col(gm, zgm, 3),
                pos2_dir_C_up_G_down=bool(d2[1] > 0 and d2[2] < 0), pos2_topbase_is_C=bool(gm[:, s + 2].argmax() == 1),
                pos3_dir_G_up_C_down=bool(d3[2] > 0 and d3[1] < 0), pos3_topbase_is_G=bool(gm[:, s + 3].argmax() == 2),
                consensus_is_CACGTG=bool("".join(BASES[i] for i in gm[:, s:s + 6].argmax(0)) == "CACGTG"),
                dswitch=round(float(dsw), 2))
def native_delta(m, kind, noAA=False):
    core = slice(CA, CB); random.seed(7); pairs = list(itertools.combinations(ho_idx, 2)); random.shuffle(pairs); pairs = pairs[:300]
    dc, da, mr = [], [], []
    for i, j in pairs:
        z1 = P_of(m, kind, embs[i], seqs[i], noAA)[0]; z2 = P_of(m, kind, embs[j], seqs[j], noAA)[0]
        P1, P2 = frame(pwms[i])[0], frame(pwms[j])[0]
        dp = (cent(z2) - cent(z1))[:, core]; dt = (cent(np.log(P2 + 1e-6)) - cent(np.log(P1 + 1e-6)))[:, core]
        if dt.std() < 1e-4: continue
        dc.append(np.corrcoef(dp.ravel(), dt.ravel())[0, 1]); da.append(float((dp * dt).sum() > 0))
        mr.append(np.linalg.norm(dp) / (np.linalg.norm(dt) + 1e-6))
    return float(np.nanmean(dc)), float(np.mean(da)), float(np.nanmean(mr))
def neutral(m, kind, noAA=False):
    _, Pw = P_of(m, kind, MY["WT"][0], MY["WT"][1], noAA); _, Pn = P_of(m, kind, MY["neutral@40A"][0], MY["neutral@40A"][1], noAA)
    dsw = (motif_score(Pn, "CACGTG") - motif_score(Pn, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    a, b = ic_core(Pw); return float(dsw), float(1 - np.corrcoef(Pw[:, a:b].ravel(), Pn[:, a:b].ravel())[0, 1])

# ── PART A: unified table, 3 seeds ──
VARIANTS = [("simple_esm_mean", "mean", {}), ("simple_esm_attn", "attn", {}),
            ("free_attn+delta", "attn", {"equiv": True}), ("recog_full", "recog", {}),
            ("recog_equiv", "recog", {"equiv": True}), ("recog_noAA", "recog", {"noAA": True})]
RESULT.setdefault("table", {})
if "recog_noAA" in RESULT["table"]:
    print("[A] table already complete — skipping Part A", flush=True); VARIANTS = []
# no-train baselines on SAME test genes
fam_covr = [np.mean([align_corr(ref, pwms[i]) for i in ho_idx if genes[i] == g]) for g in held]
v24_covr = []
for g in held:
    vs = [align_corr(v24map[seqs[i]], pwms[i]) for i in ho_idx if genes[i] == g and seqs[i] in v24map]
    if vs: v24_covr.append(np.mean(vs))
mci = boot_ci(fam_covr); RESULT["table"]["FamilyCode"] = dict(wt_covr_mean=mci[0], ci95=[mci[1], mci[2]], params=0)
mci = boot_ci(v24_covr); RESULT["table"]["v24"] = dict(wt_covr_mean=mci[0], ci95=[mci[1], mci[2]], n_genes=len(v24_covr), params="full")
RESULT["table"]["WT-copy"] = dict(wt_covr_mean=None, note="MUT-copy Δ=0; covR undefined (=truth)")
save(); print("[A] baselines done: FamilyCode", RESULT["table"]["FamilyCode"]["wt_covr_mean"],
              "| v24", RESULT["table"]["v24"]["wt_covr_mean"], flush=True)

for name, kind, kw in VARIANTS:
    covrs, dsw, p2d, p2t, p3d, p3t, cons, ndc, nda, nmr, nud = [], [], [], [], [], [], [], [], [], [], []
    for seed in range(3):
        m = train(kind, equiv=kw.get("equiv", False), noAA=kw.get("noAA", False), seed=seed)
        covrs.append(wt_covr_pergene(m, kind, kw.get("noAA", False)))
        d = myod1_detail(m, kind, kw.get("noAA", False))
        dsw.append(d["dswitch"]); p2d.append(d["pos2_dir_C_up_G_down"]); p2t.append(d["pos2_topbase_is_C"])
        p3d.append(d["pos3_dir_G_up_C_down"]); p3t.append(d["pos3_topbase_is_G"]); cons.append(d["consensus_is_CACGTG"])
        c1, c2, c3 = native_delta(m, kind, kw.get("noAA", False)); ndc.append(c1); nda.append(c2); nmr.append(c3)
        nud.append(neutral(m, kind, kw.get("noAA", False))[0])
    per_gene_mean = np.mean(covrs, 0); mci = boot_ci(per_gene_mean)
    ntr = sum(p.numel() for p in make(kind).parameters())
    RESULT["table"][name] = dict(
        wt_covr_mean=round(mci[0], 3), ci95=[round(mci[1], 3), round(mci[2], 3)],
        wt_covr_seeds=[round(float(np.mean(c)), 3) for c in covrs],
        myod1_dswitch_mean=round(float(np.mean(dsw)), 2),
        pos2_dir=float(np.mean(p2d)), pos2_topbase_C=float(np.mean(p2t)),
        pos3_dir=float(np.mean(p3d)), pos3_topbase_G=float(np.mean(p3t)),
        consensus_CACGTG=float(np.mean(cons)),
        native_delta_corr=round(float(np.mean(ndc)), 3), native_dir_acc=round(float(np.mean(nda)), 3),
        native_mag_ratio=round(float(np.mean(nmr)), 3), neutral_dswitch=round(float(np.mean(nud)), 3),
        trainable_params=int(ntr))
    save(); print(f"[A] {name}: covR {RESULT['table'][name]['wt_covr_mean']} CI{RESULT['table'][name]['ci95']} "
                  f"Δsw {RESULT['table'][name]['myod1_dswitch_mean']} pos3_topG {RESULT['table'][name]['pos3_topbase_G']} "
                  f"consensus {RESULT['table'][name]['consensus_CACGTG']} | nat dir {RESULT['table'][name]['native_dir_acc']}", flush=True)

# ── PART B: MyoD1 per-column dump (seed 0) ──
RESULT["myod1_percolumn"] = {}
for name, kind, kw in [("simple_esm_mean", "mean", {}), ("simple_esm_attn", "attn", {}),
                       ("recog_full", "recog", {}), ("recog_equiv", "recog", {"equiv": True})]:
    m = train(kind, equiv=kw.get("equiv", False), seed=0)
    RESULT["myod1_percolumn"][name] = myod1_detail(m, kind)
    if kind == "recog":
        RESULT["myod1_percolumn"][name + "+oracle"] = myod1_detail(m, kind, oracle=list(range(15)))
    save()
print("[B] MyoD1 per-column done", flush=True)

# ── PART C: ESM embedding-intervention radius curve (MyoD1) ──
def hybrid(radius):
    e = MY["WT"][0].copy()
    if radius == "full": e = MY["L112R"][0].copy()
    else:
        lo, hi = max(0, 11 - radius), min(len(WT), 11 + radius + 1); e[lo:hi] = MY["L112R"][0][lo:hi]
    return e
RESULT["esm_intervention"] = {}
for name, kind, kw in [("simple_esm_mean", "mean", {}), ("recog_equiv", "recog", {"equiv": True})]:
    m = train(kind, equiv=kw.get("equiv", False), seed=0)
    _, Pw = P_of(m, kind, MY["WT"][0], MY["WT"][1])
    curve = {}
    for r in [0, 1, 3, 5, "full"]:
        _, Pm = P_of(m, kind, hybrid(r), MY["WT"][1])
        dsw = (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
        a, b = ic_core(Pw); dpred = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]
        curve[str(r)] = dict(dswitch=round(float(dsw), 2), dpred=round(float(dpred), 3))
    # AA-one-hot-only (WT emb, mutant one-hot) = local transport
    _, Pl = P_of(m, kind, MY["WT"][0], MY["L112R"][1])
    dsw_local = (motif_score(Pl, "CACGTG") - motif_score(Pl, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
    curve["AA_onehot_only(WT_emb)"] = dict(dswitch=round(float(dsw_local), 2))
    RESULT["esm_intervention"][name] = curve; save()
    print(f"[C] {name} intervention: {curve}", flush=True)

RESULT["done"] = True; save()
print("\n[DONE] saved", OUT, flush=True)
