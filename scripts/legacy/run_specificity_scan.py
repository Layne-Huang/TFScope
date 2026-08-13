"""Systematic specificity-design scan across the full TF set (no hand-picked targets).

For every target TF (dedup gene, curated PWM) in the main families, with its top-8 most-similar
same-family off-targets (by predicted PWM), compute:
  m1 target_self_corr        : corr(predicted, experimental) for the target  (self-prediction acc)
  m2 pred_off_corr_mean/max  : target vs off-targets, PREDICTED PWMs
  m3 exp_off_corr_mean/max   : target vs off-targets, EXPERIMENTAL PWMs
  m4 predicted_separability  : margin a GA reaches optimising PREDICTED margin  (predicted-oracle)
  m5 exp_oracle_margin       : margin a GA reaches optimising EXPERIMENTAL margin (upper bound)
  m6 tfscope_transfer_margin : EXPERIMENTAL margin of the predicted-optimised (TFScope) designs (held-out)

Hypothesis: higher predicted separability  ->  higher experimental transfer (m6).
Outputs: results/specificity_design/scan_table.tsv + pwm_cache.npz (predicted/experimental PWMs).
Usage: python scripts/run_specificity_scan.py
"""
import os, sys, json, pickle
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

OUT = "results/specificity_design"; os.makedirs(OUT, exist_ok=True)
CACHE = f"{OUT}/pwm_cache.pkl"
CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_aug_dbd.parquet"
FAMILIES = ["Homeodomain", "bHLH", "bZIP", "ETS", "Forkhead", "Nuclear_Receptor",
            "C2H2_short", "C2H2_medium", "C2H2_long"]
POOL_PER_FAM = 130          # genes cached per family
TARGETS_PER_FAM = 60        # targets scanned per family
ENSURE = ["LHX5", "MYOG", "CREB3L2", "ELK1"]   # original hand-picked hard cases
N_OFF = 8
L = 24; GCMIN, GCMAX, HOMO = 0.35, 0.65, 3
BG = 0.25; EPS = 1e-3; LAM = 1.0; NBG = 20000
COMP = np.array([3, 2, 1, 0])
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# ───────── scoring helpers ─────────
def gc_frac(s): return ((s == 1) | (s == 2)).mean(1)
def maxhomo(s):
    run = np.ones(len(s), int); mx = run.copy()
    for j in range(1, L):
        run = np.where(s[:, j] == s[:, j - 1], run + 1, 1); mx = np.maximum(mx, run)
    return mx
def valid_mask(s): g = gc_frac(s); return (g >= GCMIN) & (g <= GCMAX) & (maxhomo(s) <= HOMO)
def random_valid(n, r):
    out = []
    while len(out) < n:
        s = r.integers(0, 4, (n * 2, L)); out.extend(list(s[valid_mask(s)]))
    return np.array(out[:n])
def llr_of(P): f = np.log((P + EPS) / BG); return (f, f[COMP][:, ::-1])
def scan_max(seqs, pair):
    best = np.full(len(seqs), -1e30)
    for llr in pair:
        Lk = llr.shape[1]
        if Lk > L: continue
        idx = np.arange(Lk)
        for o in range(L - Lk + 1):
            best = np.maximum(best, llr[seqs[:, o:o + Lk], idx].sum(1))
    return best
def pwm_corr(A, B):
    def rc(P): return P[COMP][:, ::-1]
    best = -1
    for Bx in (B, rc(B)):
        for off in range(-(B.shape[1] - 1), A.shape[1]):
            a0, b0 = max(0, off), max(0, -off); ov = min(A.shape[1] - a0, Bx.shape[1] - b0)
            if ov < 4: continue
            a = A[:, a0:a0 + ov].ravel(); b = Bx[:, b0:b0 + ov].ravel()
            if a.std() > 1e-9 and b.std() > 1e-9: best = max(best, np.corrcoef(a, b)[0, 1])
    return best

BGSEQS = random_valid(NBG, np.random.default_rng(7))
def zmodel(P): s = scan_max(BGSEQS, llr_of(P)); return float(s.mean()), float(s.std() + 1e-9)

# ───────── cache predicted + experimental PWMs ─────────
def exp_pwm(b):
    P = np.frombuffer(b, dtype=np.float32).reshape(4, -1).astype(float); P = P / P.sum(0, keepdims=True)
    ic = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(0); k = np.where(ic > 0.25)[0]
    return P[:, k.min():k.max() + 1] if len(k) >= 4 else P

df = pd.read_parquet(PARQ)
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
print(f"loaded cache: {len(cache)} TFs")
# determine genes to ensure per family (head(POOL) + ENSURE list), predict only those missing
want_rows = []
for fam in FAMILIES:
    fdf = df[df.family_name == fam].drop_duplicates("gene_symbol")
    genes = list(fdf.head(POOL_PER_FAM).gene_symbol) + [g for g in ENSURE if g in set(fdf.gene_symbol)]
    for g in dict.fromkeys(genes):
        if g not in cache:
            want_rows.append(fdf[fdf.gene_symbol == g].iloc[0])
if want_rows:
    print(f"predicting {len(want_rows)} new PWMs ...")
    mc = TFScopeConfig()
    for k, v in json.load(open(os.path.dirname(CKPT) + "/config.json")).items():
        if hasattr(mc, k):
            try: setattr(mc, k, type(getattr(mc, k))(v))
            except Exception: pass
    mc.use_retrieval = False
    model = TFScopeModel(mc).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
    @torch.no_grad()
    def predict_pwm(seq, fid):
        t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
        dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
        gl, pl, _ = model(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
        g = gl.sigmoid()[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy()
        cols = np.where(g > 0.5)[0]
        if len(cols) < 4:
            ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); cols = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
        return p[:, cols.min():cols.max() + 1]
    for i, r in enumerate(want_rows):
        dbd = str(r.sequence)[int(r.dbd_start):int(r.dbd_end)]
        if not (15 <= len(dbd) <= 200): continue
        try: cache[r.gene_symbol] = dict(family=r.family_name, pred=predict_pwm(dbd, int(r.family_id)), exp=exp_pwm(r.pwm))
        except Exception: continue
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(want_rows)}")
    pickle.dump(cache, open(CACHE, "wb"))
print(f"cache now: {len(cache)} TFs | by family: " +
      ", ".join(f"{fam}={sum(1 for v in cache.values() if v['family']==fam)}" for fam in FAMILIES))

# precompute zmodels
for g in cache:
    cache[g]["zp"] = zmodel(cache[g]["pred"]); cache[g]["ze"] = zmodel(cache[g]["exp"])

# ───────── GA + metrics ─────────
def Zscore(seqs, P, zmu_sd): return (scan_max(seqs, llr_of(P)) - zmu_sd[0]) / zmu_sd[1]
def ga(target, offs, which, seeds=2, pop=250, gen=25, mut=0.05):
    Pt = cache[target][which]; zt_ms = cache[target]["zp" if which == "pred" else "ze"]
    cons0 = Pt.argmax(0); emb = random_valid(1, np.random.default_rng(0))[0].copy(); emb[:len(cons0)] = cons0
    floor = 0.6 * float(Zscore(emb[None], Pt, zt_ms)[0])
    allbest = []
    for sd in range(seeds):
        r = np.random.default_rng(sd + (99 if which == "exp" else 0)); pop_ = random_valid(pop, r)
        for _ in range(gen):
            zt = Zscore(pop_, Pt, zt_ms)
            zo = np.max([Zscore(pop_, cache[o][which], cache[o]["zp" if which == "pred" else "ze"]) for o in offs], axis=0)
            fit = np.where((zt >= floor) & valid_mask(pop_), zt - LAM * zo, -1e9)
            order = np.argsort(-fit); el = pop_[order[:max(1, pop // 20)]]; par = pop_[order[:max(2, pop // 5)]]
            kids = []
            while len(kids) < pop - len(el):
                i, j = r.integers(0, len(par), 2); a = par[i].copy()
                if r.random() < 0.3: cx = r.integers(1, L); a[cx:] = par[j][cx:]
                mm = r.random(L) < mut; a[mm] = r.integers(0, 4, mm.sum()); kids.append(a)
            pop_ = np.vstack([el, np.array(kids)])
        allbest.append(pop_[valid_mask(pop_)])
    return np.vstack(allbest)

def margin(seqs, target, offs, which):
    zt = Zscore(seqs, cache[target][which], cache[target]["zp" if which == "pred" else "ze"])
    zo = np.max([Zscore(seqs, cache[o][which], cache[o]["zp" if which == "pred" else "ze"]) for o in offs], axis=0)
    return zt - zo

rows = []
for fam in FAMILIES:
    genes = [g for g in cache if cache[g]["family"] == fam]
    for target in genes[:TARGETS_PER_FAM]:
        others = [g for g in genes if g != target]
        if len(others) < N_OFF: continue
        corrs = sorted(((pwm_corr(cache[target]["pred"], cache[o]["pred"]), o) for o in others), reverse=True)
        offs = [o for _, o in corrs[:N_OFF]]
        m1 = pwm_corr(cache[target]["pred"], cache[target]["exp"])
        pc = [pwm_corr(cache[target]["pred"], cache[o]["pred"]) for o in offs]
        ec = [pwm_corr(cache[target]["exp"], cache[o]["exp"]) for o in offs]
        prop = ga(target, offs, "pred")                # TFScope-guided (optimise predicted)
        m4 = float(np.median(margin(prop, target, offs, "pred")))   # predicted separability/oracle
        m6 = float(np.median(margin(prop, target, offs, "exp")))    # held-out experimental transfer
        eo = ga(target, offs, "exp")
        m5 = float(np.median(margin(eo, target, offs, "exp")))      # experimental-oracle upper bound
        rows.append(dict(target=target, family=fam, off_targets=";".join(offs),
                         target_self_corr=round(m1, 3),
                         pred_off_corr_mean=round(float(np.mean(pc)), 3), pred_off_corr_max=round(float(np.max(pc)), 3),
                         exp_off_corr_mean=round(float(np.mean(ec)), 3), exp_off_corr_max=round(float(np.max(ec)), 3),
                         predicted_separability=round(m4, 3), exp_oracle_margin=round(m5, 3),
                         tfscope_transfer_margin=round(m6, 3)))
    print(f"scanned {fam}: {sum(1 for r in rows if r['family']==fam)} targets")

T = pd.DataFrame(rows)
# case bins by hardest predicted off-target similarity
def case(c): return "easy" if c < 0.85 else ("moderate" if c < 0.95 else "hard")
T["case"] = T["pred_off_corr_max"].apply(case)
T.to_csv(f"{OUT}/scan_table.tsv", sep="\t", index=False)
from scipy.stats import spearmanr
ok = T.dropna(subset=["predicted_separability", "tfscope_transfer_margin"])
rho_sep, p_sep = spearmanr(ok.predicted_separability, ok.tfscope_transfer_margin)
rho_corr, p_corr = spearmanr(ok.pred_off_corr_max, ok.tfscope_transfer_margin)
print(f"\nN targets scanned: {len(T)}")
print(f"cases: {T['case'].value_counts().to_dict()}")
print(f"Spearman(predicted_separability, exp_transfer) = {rho_sep:.2f} (p={p_sep:.1e})")
print(f"Spearman(pred_off_corr_max,    exp_transfer) = {rho_corr:.2f} (p={p_corr:.1e})  [expect negative]")
print(f"saved {OUT}/scan_table.tsv")
