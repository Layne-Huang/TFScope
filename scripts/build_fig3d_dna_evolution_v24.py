"""Fig 3d — model-guided DNA optimization (in-silico SELEX) recovers TF consensus.

TFScope acts as a binding oracle: given a protein it predicts a PWM, which defines a binding
landscape over DNA. Starting from RANDOM DNA, a population is evolved (select high-affinity,
mutate) to maximize the predicted binding score. Across families the evolution converges from
random to each factor's correct, distinct consensus, recovering the curated motif — a
demonstration that the predicted specificity can drive sequence design.

(Protein-side single-residue redesign, e.g. homeodomain Q50K, is NOT captured by the
sequence-only model and is deliberately not shown here; that limitation motivates the
structure-based pipeline in Fig 4.)

Combined no-RAG model. Outputs: results/fig3d_evolution/fig3d_evolution.json;
figures/figure3d_dna_evolution/figure3d_dna_evolution.{png,pdf}
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import trimmed_core, aligned_cols

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_aug_dbd.parquet"
OUTD = "figures_v24/figure3d_dna_evolution"; os.makedirs(OUTD, exist_ok=True)
RES = "results/fig3d_evolution"; os.makedirs(RES, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
# one clean, recognisable TF per family (lead = homeodomain)
PICK = {"Homeodomain": "LHX5", "bHLH": "MYOG", "bZIP": "CREB3L2", "ETS": "ELK1"}
FAM_COL = {"Homeodomain": "#7B6BB1", "bHLH": "#55A868", "bZIP": "#D95F4C", "ETS": "#E69F00"}

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CKPT) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict_pwm(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    g = gl.sigmoid()[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy()
    cols = np.where(g > 0.5)[0]
    if len(cols) < 4:
        ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); cols = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    return p[:, cols.min():cols.max() + 1]                        # (4, L) predicted motif core

def selex(pwm, n_pop=400, n_gen=24, mut=0.07, seed=42):
    """Evolve random DNA to maximise predicted binding (PWM log-likelihood)."""
    L = pwm.shape[1]; rng = np.random.default_rng(seed)
    logP = np.log(np.clip(pwm, 1e-6, 1))                          # (4, L)
    pop = rng.integers(0, 4, (n_pop, L))
    def score(p): return logP[p, np.arange(L)].sum(1)
    mean_aff, snaps = [], {}
    for gtot in range(n_gen):
        sc = score(pop); mean_aff.append(float(sc.mean()))
        if gtot in (0, n_gen // 2, n_gen - 1):
            snaps[gtot] = np.stack([(pop == b).mean(0) for b in range(4)])  # (4, L) population freq
        keep = pop[np.argsort(-sc)[:n_pop // 2]]
        kids = keep[rng.integers(0, len(keep), n_pop - len(keep))].copy()
        msk = rng.random(kids.shape) < mut; kids[msk] = rng.integers(0, 4, int(msk.sum()))
        pop = np.vstack([keep, kids])
    # per-seq max possible affinity (consensus) for normalisation
    best = logP.max(0).sum()
    return np.array(mean_aff), snaps, snaps[n_gen - 1], best, L

def colr(A, B):
    rs = [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])
          if A[:, j].std() > 1e-8 and B[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan

d = pd.read_parquet(PARQ); d["g"] = d.gene_symbol
results = {}
for fam, gene in PICK.items():
    row = d[(d.family_name == fam) & (d.g == gene)].iloc[0]
    s, e = int(row.dbd_start), int(row.dbd_end); dbd = str(row.sequence)[s:e]
    pwm = predict_pwm(dbd, int(row.family_id))
    aff, snaps, final, best, L = selex(pwm, seed=42)
    # recovery of the EVOLVED consensus vs the curated (experimental) motif
    gt = np.frombuffer(row.pwm, dtype=np.float32).reshape(4, -1).astype(float)
    core = trimmed_core(gt, np.ones(gt.shape[1], bool))
    al, cols, _ = aligned_cols(final, core) if core is not None else (final, [], None)
    rec = colr(np.clip(al[:, cols], 1e-8, 1) / np.clip(al[:, cols], 1e-8, 1).sum(0), core[:, cols]) if len(cols) >= 4 else np.nan
    cons = "".join("ACGT"[final[:, j].argmax()] for j in range(L))
    results[fam] = dict(gene=gene, L=L, mean_aff=aff.tolist(), best_aff=float(best),
                        pred_pwm=pwm.tolist(), final_freq=final.tolist(),
                        gen0_freq=snaps[0].tolist(), consensus=cons, recovery_r=round(float(rec), 3))
    print(f"{fam:<13}{gene:<9} evolved consensus={cons:<12} recovery r(vs curated)={results[fam]['recovery_r']}")
json.dump(results, open(f"{RES}/fig3d_evolution.json", "w"), indent=1)

# ───────────────────────── figure ─────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
def logo(ax, freq, title="", col="black"):
    P = np.clip(freq, 1e-9, 1); P = P / P.sum(0, keepdims=True)
    ic = np.maximum(2 + (P * np.log2(P)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((P * ic).T, columns=list("ACGT")), ax=ax,
                   color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    if title: ax.set_title(title, fontsize=8, color=col, pad=2)

fig = plt.figure(figsize=(11, 4.8))
gs = fig.add_gridspec(6, 2, width_ratios=[1.2, 1.0], hspace=1.1, wspace=0.22,
                      left=0.085, right=0.98, top=0.9, bottom=0.12)

# (a) convergence curves: normalized affinity vs generation, all families
axa = fig.add_subplot(gs[:, 0])
for fam in PICK:
    a = np.array(results[fam]["mean_aff"]); best = results[fam]["best_aff"]; rand = a[0]
    norm = (a - rand) / (best - rand + 1e-9)                      # 0=random, 1=consensus-optimal
    axa.plot(norm, "-", lw=1.8, color=FAM_COL[fam], label=f"{fam} · {results[fam]['gene']}")
axa.axhline(1.0, color="#999", ls=":", lw=1)
axa.text(len(a) - 1, 1.005, "optimum", fontsize=7, va="bottom", ha="right", color="#777")
axa.set_xlabel("evolution generation", fontsize=9)
axa.set_ylabel("predicted binding affinity  (0 = random, 1 = optimal)", fontsize=8.5)
axa.set_title("a  In-silico SELEX converges from random DNA", fontsize=9.5, fontweight="bold", loc="left")
axa.legend(fontsize=7, frameon=False, loc="lower right"); axa.set_ylim(-0.03, 1.08)
for sp in ["top", "right"]: axa.spines[sp].set_visible(False)

# (b) homeodomain: random (gen0) -> evolved (genN) population logos
hd = results["Homeodomain"]
fig.text(0.555, 0.945, "b  Random DNA → recovered consensus", fontsize=9.5, fontweight="bold")
axb0 = fig.add_subplot(gs[0, 1]); logo(axb0, np.array(hd["gen0_freq"]))
axb0.text(0.0, 1.05, "start: random DNA (flat, ~0 bits)", transform=axb0.transAxes, fontsize=6.5, color="#777", va="bottom")
logo(fig.add_subplot(gs[1, 1]), np.array(hd["final_freq"]),
     f"evolved {hd['consensus']}  ·  {hd['gene']} (homeodomain, r={hd['recovery_r']})", FAM_COL["Homeodomain"])

# (c) evolved consensus for the other families (each recovers its distinct motif)
fig.text(0.555, 0.545, "c  Each factor recovers its own motif", fontsize=9.5, fontweight="bold")
for i, fam in enumerate(["bHLH", "bZIP", "ETS"]):
    r = results[fam]
    logo(fig.add_subplot(gs[3 + i, 1]), np.array(r["final_freq"]),
         f"evolved {r['consensus']}  ·  {r['gene']} ({fam}, r={r['recovery_r']})", FAM_COL[fam])

out = f"{OUTD}/figure3d_dna_evolution"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"saved {out}.png/.pdf + {RES}/fig3d_evolution.json")
