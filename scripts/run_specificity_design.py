"""Fig 3d (new) — specificity-aware forward DNA design with TFScope.

Pipeline (per target TF LHX5/MYOG/CREB3L2/ELK1):
  1. predict TFScope PWM for target + same-family candidate pool; pick 5-8 hardest off-targets
     by predicted-PWM similarity (fwd/rev/offset aligned).
  2. score model: PWM log-odds, max over windows+both strands on 24-bp DNA; Z-normalise vs 50k
     random GC-constrained backgrounds. SEPARATE models for predicted (optimisation/oracle) and
     curated experimental PWMs (independent validation — never used in optimisation).
  3. specificity-aware GA maximising J = Z_target - lambda*max_off Z_off - alpha*C(constraints),
     vs baselines: random, consensus-embedding, target-only GA. 20 diverse designs/method.
  4. evaluate all designs on predicted AND experimental PWMs; specificity margin M = Z_t - max_o Z_o.

Outputs under results/specificity_design/: off_target_selection.tsv, random_background_stats.tsv,
final_designs.tsv, oracle_evaluation.tsv, experimental_pwm_evaluation.tsv, summary.json.
Usage: python scripts/run_specificity_design.py [TARGET ...]   (default: all four)
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, yaml, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

cfg = yaml.safe_load(open("configs/specificity_design.yaml"))
OUT = "results/specificity_design"; os.makedirs(OUT, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
L = cfg["design"]["length"]; GCMIN, GCMAX = cfg["design"]["gc_min"], cfg["design"]["gc_max"]
HOMO = cfg["design"]["max_homopolymer"]; BG = cfg["objective"]["bg"]; EPS = cfg["objective"]["pseudocount"]
LAM = cfg["objective"]["lambda_off"]; ALPHA = cfg["objective"]["alpha_constraint"]
NBG = cfg["background"]["n_random"]; rng = np.random.default_rng(cfg["seed"])
TARGETS = sys.argv[1:] or list(cfg["targets"].keys())
COMP = np.array([3, 2, 1, 0])   # A<->T, C<->G

# ───────────────────────── model + PWMs ─────────────────────────
mc = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(cfg["model"]["checkpoint"]) + "/config.json")).items():
    if hasattr(mc, k):
        try: setattr(mc, k, type(getattr(mc, k))(v))
        except Exception: pass
mc.use_retrieval = False
model = TFScopeModel(mc).to(dev).eval()
model.load_state_dict(torch.load(cfg["model"]["checkpoint"], map_location=dev, weights_only=False)["model"], strict=False)

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

def exp_pwm(gt_bytes):
    P = np.frombuffer(gt_bytes, dtype=np.float32).reshape(4, -1).astype(float)
    P = P / P.sum(0, keepdims=True)
    ic = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(0)
    keep = np.where(ic > 0.25)[0]
    if len(keep) >= 4: P = P[:, keep.min():keep.max() + 1]
    return P

def pwm_corr(A, B):                       # max correlation over offset + strand
    def rc(P): return P[COMP][:, ::-1]
    best = -1
    for Bx in (B, rc(B)):
        for off in range(-(B.shape[1] - 1), A.shape[1]):
            a0, b0 = max(0, off), max(0, -off); ov = min(A.shape[1] - a0, Bx.shape[1] - b0)
            if ov < 4: continue
            a = A[:, a0:a0 + ov].ravel(); b = Bx[:, b0:b0 + ov].ravel()
            if a.std() > 1e-9 and b.std() > 1e-9: best = max(best, np.corrcoef(a, b)[0, 1])
    return best

# ───────────────────────── scoring (vectorised) ─────────────────────────
def llr_of(P):
    fwd = np.log((P + EPS) / BG); return fwd, fwd[COMP][:, ::-1]   # (4,Lk) fwd, rc

def scan_max(seqs, llr_pair):
    """seqs (N,L) int; max log-odds over all windows + both strands."""
    best = np.full(len(seqs), -1e30)
    for llr in llr_pair:
        Lk = llr.shape[1]
        if Lk > L: continue
        idx = np.arange(Lk)
        for o in range(L - Lk + 1):
            w = seqs[:, o:o + Lk]
            sc = llr[w, idx].sum(1)
            best = np.maximum(best, sc)
    return best

# constraints
def gc_frac(seqs): return ((seqs == 1) | (seqs == 2)).mean(1)
def max_homopolymer(seqs):
    run = np.ones(len(seqs), int); mx = np.ones(len(seqs), int)
    for j in range(1, L):
        same = seqs[:, j] == seqs[:, j - 1]; run = np.where(same, run + 1, 1); mx = np.maximum(mx, run)
    return mx
def valid_mask(seqs):
    gc = gc_frac(seqs); return (gc >= GCMIN) & (gc <= GCMAX) & (max_homopolymer(seqs) <= HOMO)

def random_valid(n, r):
    out = []
    while len(out) < n:
        s = r.integers(0, 4, (n * 2, L)); s = s[valid_mask(s)]
        out.extend(list(s))
    return np.array(out[:n])

# background Z-model (shared background, GC-constrained)
BGSEQS = random_valid(NBG, np.random.default_rng(7))
def zmodel(P):
    s = scan_max(BGSEQS, llr_of(P)); return float(s.mean()), float(s.std() + 1e-9)

# ───────────────────────── per-target pipeline ─────────────────────────
df = pd.read_parquet(cfg["parquet"])
offsel_rows, design_rows, oracle_rows, exp_rows = [], [], [], []
summary = {}

for target in TARGETS:
    fam = cfg["targets"][target]
    trow = df[(df.gene_symbol == target) & (df.family_name == fam)].iloc[0]
    tdbd = str(trow.sequence)[int(trow.dbd_start):int(trow.dbd_end)]
    P_t_pred = predict_pwm(tdbd, int(trow.family_id))
    P_t_exp = exp_pwm(trow.pwm)

    # candidate pool (same family, distinct genes, exclude target)
    pool = df[(df.family_name == fam) & (df.gene_symbol != target)].drop_duplicates("gene_symbol")
    pool = pool.head(cfg["offtarget"]["pool_per_family"])
    cand = []
    for r in pool.itertuples():
        dbd = str(r.sequence)[int(r.dbd_start):int(r.dbd_end)]
        if not (15 <= len(dbd) <= 200): continue
        Pp = predict_pwm(dbd, int(r.family_id))
        cand.append((pwm_corr(P_t_pred, Pp), r.gene_symbol, Pp, exp_pwm(r.pwm)))
    cand.sort(key=lambda x: -x[0])
    offs = cand[:cfg["offtarget"]["n_per_target"]]
    for rank, (corr, g, Pp, Pe) in enumerate(offs):
        offsel_rows.append(dict(target=target, off_target=g, rank=rank + 1, pred_pwm_corr=round(corr, 3)))
    print(f"[{target}] off-targets: " + ", ".join(f"{g}({c:.2f})" for c, g, _, _ in offs))

    # score models (predicted + experimental)
    pred_pwms = {target: P_t_pred, **{g: Pp for _, g, Pp, _ in offs}}
    exp_pwms = {target: P_t_exp, **{g: Pe for _, g, _, Pe in offs}}
    zpred = {k: zmodel(P) for k, P in pred_pwms.items()}
    zexp = {k: zmodel(P) for k, P in exp_pwms.items()}
    llr_pred = {k: llr_of(P) for k, P in pred_pwms.items()}
    llr_exp = {k: llr_of(P) for k, P in exp_pwms.items()}
    offnames = [g for _, g, _, _ in offs]

    def Z(seqs, k, which):                                  # standardized score
        mu, sd = (zpred if which == "pred" else zexp)[k]
        lp = (llr_pred if which == "pred" else llr_exp)[k]
        return (scan_max(seqs, lp) - mu) / sd
    def margin(seqs, which):
        zt = Z(seqs, target, which)
        zo = np.max([Z(seqs, g, which) for g in offnames], axis=0)
        return zt, zo, zt - zo

    # ── GA ──
    g = cfg["ga"]
    # target-binding floor: a specificity design must KEEP strong target binding (plan crit. #2).
    # floor = 60% of the best achievable target Z (consensus embedding), prevents the degenerate
    # "make everything low" margin solution.
    cons0 = P_t_pred.argmax(0)
    emb0 = random_valid(1, np.random.default_rng(0))[0].copy(); emb0[:len(cons0)] = cons0
    ZT_FLOOR = 0.6 * float(Z(emb0[None], target, "pred")[0])

    def ga(use_off, seed):
        r = np.random.default_rng(1000 + seed)
        pop = random_valid(g["population"], r)
        for gen in range(g["generations"]):
            zt = Z(pop, target, "pred")
            fit = zt.copy()
            if use_off:
                zo = np.max([Z(pop, gg, "pred") for gg in offnames], axis=0)
                fit = np.where(zt >= ZT_FLOOR, zt - LAM * zo, -1e9)   # keep target binding high
            fit = np.where(valid_mask(pop), fit, -1e9)
            order = np.argsort(-fit)
            n_el = max(1, int(g["elite_fraction"] * len(pop)))
            n_par = max(2, int(g["parent_fraction"] * len(pop)))
            elite = pop[order[:n_el]]; parents = pop[order[:n_par]]
            kids = []
            while len(kids) < len(pop) - n_el:
                i, j = r.integers(0, n_par, 2); a = parents[i].copy()
                if r.random() < g["crossover_prob"]:
                    cx = r.integers(1, L); a[cx:] = parents[j][cx:]
                mm = r.random(L) < g["mutation_rate"]; a[mm] = r.integers(0, 4, mm.sum())
                kids.append(a)
            pop = np.vstack([elite, np.array(kids)])
        return pop[valid_mask(pop)]

    def diverse_top(seqs, scores, n, minh):
        order = np.argsort(-scores); chosen = []
        for i in order:
            s = seqs[i]
            if all((s != seqs[c]).sum() >= minh for c in chosen): chosen.append(i)
            if len(chosen) >= n: break
        return [seqs[c] for c in chosen]

    methods = {}
    # proposed + target-only GA (pool over seeds)
    for name, use_off in [("proposed", True), ("target_only", False)]:
        allcand = np.vstack([ga(use_off, s) for s in range(g["seeds"])])
        _, _, m = margin(allcand, "pred"); sc = m if use_off else Z(allcand, target, "pred")
        methods[name] = np.array(diverse_top(allcand, sc, g["final_designs"], g["min_hamming"]))
    # random baseline
    methods["random"] = random_valid(g["final_designs"], np.random.default_rng(99))
    # consensus-embedding baseline: target predicted-PWM argmax, embedded at all positions/strands
    cons = P_t_pred.argmax(0); cons_rc = COMP[cons[::-1]]
    cemb = []
    for core in (cons, cons_rc):
        for o in range(L - len(core) + 1):
            base = random_valid(1, np.random.default_rng(o))[0].copy(); base[o:o + len(core)] = core
            if valid_mask(base[None])[0]: cemb.append(base)
    cemb = np.array(cemb) if cemb else random_valid(g["final_designs"], np.random.default_rng(5))
    zc = Z(cemb, target, "pred"); methods["consensus"] = np.array(diverse_top(cemb, zc, g["final_designs"], g["min_hamming"]))

    # ── evaluate all methods on predicted + experimental ──
    bases = "ACGT"
    for meth, seqs in methods.items():
        if len(seqs) == 0: continue
        ztp, zop, mp = margin(seqs, "pred"); zte, zoe, me = margin(seqs, "exp")
        # worst off-target per design (experimental)
        zo_each = np.array([Z(seqs, gg, "exp") for gg in offnames])  # (n_off, N)
        worst = [offnames[i] for i in zo_each.argmax(0)]
        for i in range(len(seqs)):
            s = "".join(bases[b] for b in seqs[i])
            design_rows.append(dict(target_tf=target, method=meth, sequence=s,
                                    target_z_pred=round(float(ztp[i]), 3), max_offtarget_z_pred=round(float(zop[i]), 3),
                                    margin_pred=round(float(mp[i]), 3), target_z_exp=round(float(zte[i]), 3),
                                    max_offtarget_z_exp=round(float(zoe[i]), 3), margin_exp=round(float(me[i]), 3),
                                    gc=round(float(((seqs[i] == 1) | (seqs[i] == 2)).mean()), 2), worst_offtarget=worst[i]))
    summary[target] = dict(family=fam, off_targets=offnames,
                           median_margin_exp={mth: round(float(np.median([d["margin_exp"] for d in design_rows
                                              if d["target_tf"] == target and d["method"] == mth])), 3)
                                              for mth in methods},
                           median_target_z_exp={mth: round(float(np.median([d["target_z_exp"] for d in design_rows
                                                if d["target_tf"] == target and d["method"] == mth])), 3)
                                                for mth in methods})
    print(f"  median exp margin: " + "  ".join(f"{m}={summary[target]['median_margin_exp'][m]}" for m in ["random","consensus","target_only","proposed"]))

pd.DataFrame(offsel_rows).to_csv(f"{OUT}/off_target_selection.tsv", sep="\t", index=False)
pd.DataFrame(design_rows).to_csv(f"{OUT}/final_designs.tsv", sep="\t", index=False)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
# primary endpoint
wins = sum(1 for t in summary if summary[t]["median_margin_exp"]["proposed"] > summary[t]["median_margin_exp"]["consensus"]
           and summary[t]["median_margin_exp"]["proposed"] > summary[t]["median_margin_exp"]["target_only"])
print(f"\nPRIMARY ENDPOINT: proposed exp-margin > consensus AND target-only in {wins}/{len(summary)} targets")
json.dump(dict(summary=summary, primary_wins=wins, n_targets=len(summary)), open(f"{OUT}/endpoint.json", "w"), indent=1)
print(f"saved {OUT}/{{off_target_selection,final_designs,summary,endpoint}}")
