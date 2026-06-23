"""Fig 3d panel — successful specificity-aware designs (good examples).

For the best-transfer, well-predicted targets across families, re-run the TFScope-guided GA
(optimise predicted target-vs-offtarget margin), take the top design TFScope nominates, and show
its INDEPENDENT experimental-PWM Z-score profile: high for the target, low for the off-targets.
Designs are nominated by predicted margin; the experimental scores are held out (not optimised).

Reads the cached PWMs + scan table from run_specificity_scan.py.
Out: figures/figure3d_good_examples/figure3d_good_examples.{png,pdf}; results/specificity_design/good_designs.tsv
"""
import os, pickle
import numpy as np, pandas as pd
SRC = "results/specificity_design"; OUTD = "figures/figure3d_good_examples"; os.makedirs(OUTD, exist_ok=True)
cache = pickle.load(open(f"{SRC}/pwm_cache.pkl", "rb"))
T = pd.read_csv(f"{SRC}/scan_table.tsv", sep="\t")
L = 24; GCMIN, GCMAX, HOMO = 0.35, 0.65, 3; BG = 0.25; EPS = 1e-3; LAM = 1.0; NBG = 20000
COMP = np.array([3, 2, 1, 0])
FAM_COL = {"Homeodomain": "#7B6BB1", "bHLH": "#55A868", "bZIP": "#D95F4C", "ETS": "#E69F00",
           "Forkhead": "#3B9AB2", "C2H2_short": "#CC6677", "C2H2_medium": "#882255", "Nuclear_Receptor": "#117733"}

# ── scoring helpers (same as scan) ──
def gc(s): return ((s == 1) | (s == 2)).mean(1)
def homo(s):
    run = np.ones(len(s), int); mx = run.copy()
    for j in range(1, L): run = np.where(s[:, j] == s[:, j - 1], run + 1, 1); mx = np.maximum(mx, run)
    return mx
def valid(s): g = gc(s); return (g >= GCMIN) & (g <= GCMAX) & (homo(s) <= HOMO)
def rvalid(n, r):
    o = []
    while len(o) < n:
        s = r.integers(0, 4, (n * 2, L)); o.extend(list(s[valid(s)]))
    return np.array(o[:n])
def llr(P): f = np.log((P + EPS) / BG); return (f, f[COMP][:, ::-1])
def smax(seqs, pair):
    best = np.full(len(seqs), -1e30)
    for m in pair:
        Lk = m.shape[1]
        if Lk > L: continue
        idx = np.arange(Lk)
        for o in range(L - Lk + 1): best = np.maximum(best, m[seqs[:, o:o + Lk], idx].sum(1))
    return best
def pcorr(A, B):
    best = -1
    for Bx in (B, B[COMP][:, ::-1]):
        for off in range(-(B.shape[1] - 1), A.shape[1]):
            a0, b0 = max(0, off), max(0, -off); ov = min(A.shape[1] - a0, Bx.shape[1] - b0)
            if ov < 4: continue
            a = A[:, a0:a0 + ov].ravel(); b = Bx[:, b0:b0 + ov].ravel()
            if a.std() > 1e-9 and b.std() > 1e-9: best = max(best, np.corrcoef(a, b)[0, 1])
    return best
BGS = rvalid(NBG, np.random.default_rng(7))
for g in cache:
    cache[g]["zp"] = (lambda s: (s.mean(), s.std() + 1e-9))(smax(BGS, llr(cache[g]["pred"])))
    cache[g]["ze"] = (lambda s: (s.mean(), s.std() + 1e-9))(smax(BGS, llr(cache[g]["exp"])))
def Z(seqs, g, which): mu, sd = cache[g]["zp" if which == "pred" else "ze"]; return (smax(seqs, llr(cache[g][which])) - mu) / sd

def ga(target, offs, pop=400, gen=35, seeds=3, mut=0.05):
    Pt = cache[target]["pred"]; cons = Pt.argmax(0)
    emb = rvalid(1, np.random.default_rng(0))[0].copy(); emb[:len(cons)] = cons
    floor = 0.6 * float(Z(emb[None], target, "pred")[0]); allp = []
    for sd in range(seeds):
        r = np.random.default_rng(sd); p = rvalid(pop, r)
        for _ in range(gen):
            zt = Z(p, target, "pred"); zo = np.max([Z(p, o, "pred") for o in offs], axis=0)
            fit = np.where((zt >= floor) & valid(p), zt - LAM * zo, -1e9)
            order = np.argsort(-fit); el = p[order[:pop // 20]]; par = p[order[:pop // 5]]
            kids = []
            while len(kids) < pop - len(el):
                i, j = r.integers(0, len(par), 2); a = par[i].copy()
                if r.random() < 0.3: cx = r.integers(1, L); a[cx:] = par[j][cx:]
                mm = r.random(L) < mut; a[mm] = r.integers(0, 4, mm.sum()); kids.append(a)
            p = np.vstack([el, np.array(kids)])
        allp.append(p[valid(p)])
    return np.vstack(allp)

# ── pick best-transfer well-predicted target per family (diverse good examples) ──
good = T[(T.target_self_corr > 0.95) & (T.tfscope_transfer_margin > 0.3)]
picks = (good.sort_values("tfscope_transfer_margin", ascending=False)
         .drop_duplicates("family").head(6).sort_values("tfscope_transfer_margin", ascending=False))
print("good examples:", list(picks.target))

rows = []; panels = []
for _, r in picks.iterrows():
    tgt = r.target; fam = r.family; offs = r.off_targets.split(";")
    pop = ga(tgt, offs)
    zt = Z(pop, tgt, "pred"); zo = np.max([Z(pop, o, "pred") for o in offs], axis=0)
    best = pop[np.argmax(zt - zo)]                                   # design TFScope nominates (predicted margin)
    seq = "".join("ACGT"[b] for b in best)
    zt_e = float(Z(best[None], tgt, "exp")[0]); zo_e = [float(Z(best[None], o, "exp")[0]) for o in offs]
    panels.append(dict(tgt=tgt, fam=fam, seq=seq, zt_e=zt_e, offs=offs, zo_e=zo_e,
                       margin_exp=zt_e - max(zo_e)))
    rows.append(dict(target=tgt, family=fam, design=seq, target_z_exp=round(zt_e, 2),
                     max_offtarget_z_exp=round(max(zo_e), 2), margin_exp=round(zt_e - max(zo_e), 2),
                     off_targets=";".join(offs)))
pd.DataFrame(rows).to_csv(f"{SRC}/good_designs.tsv", sep="\t", index=False)

# ── figure: one cell per example, experimental Z bars (target vs off-targets) ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = len(panels); ncol = 3; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow))
axes = np.atleast_1d(axes).ravel()
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
for k, p in enumerate(panels):
    ax = axes[k]; col = FAM_COL.get(p["fam"], "#4575b4")
    labels = [p["tgt"]] + p["offs"]; vals = [p["zt_e"]] + p["zo_e"]
    colors = [col] + ["#cfd4da"] * len(p["offs"])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.6)
    ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[0].set_color(col)
    ax.set_title(f"{p['tgt']} ({p['fam']})", fontsize=8, color=col, fontweight="bold", loc="left", pad=10)
    ax.text(0.0, 1.02, f"design 5′-{p['seq']}-3′", transform=ax.transAxes, fontsize=4.6,
            family="monospace", color="#333", va="bottom")
    ax.text(0.98, 0.04, f"exp margin +{p['margin_exp']:.1f}", transform=ax.transAxes, fontsize=6.5,
            ha="right", va="bottom", color=col, fontweight="bold")
    ax.set_xlabel("experimental-PWM Z-score", fontsize=6.5); ax.tick_params(labelsize=6, length=2)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
for k in range(n, len(axes)): axes[k].axis("off")
fig.suptitle("TFScope-guided designs are target-selective on independent experimental PWMs (good examples)",
             fontsize=10.5, fontweight="bold", y=1.0)
fig.tight_layout()
out = f"{OUTD}/figure3d_good_examples"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
for p in panels: print(f"  {p['tgt']:<10} {p['fam']:<13} exp: target Z={p['zt_e']:+.2f} maxoff={max(p['zo_e']):+.2f} margin=+{p['margin_exp']:.2f}  {p['seq']}")
print(f"saved {out}.png/.pdf + good_designs.tsv")
