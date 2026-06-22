"""Fig 2c — TFScope residue importance agrees with an experimentally-derived
recognition code (rCLAMPS; Wetzel, Zhang & Singh, Genome Res 2022).

Independent of TFScope's training prior: rCLAMPS infers, per structural family, which
protein positions read DNA bases. We HMM-align each test homeodomain to the Pfam
Homeobox HMM (the same alignment rCLAMPS uses), map the rCLAMPS recognition columns to
sequence residues, and ask whether TFScope's in-silico alanine-scan importance is
enriched at those base-reading positions vs the rest of the domain (per-TF AUROC,
precision, importance metagene). For C2H2 zinc fingers we score against the canonical
helix code {-1,+2,+3,+6} and contrast with the structural Zn-coordinating Cys/His
ligands — resolving the geometric-contact confound seen in Fig 2b's sub-0.5 tail.

Outputs: results/per_family/fig2c_recognition_code.json
         figures/figure2c_recognition_code/figure2c_recognition_code.{png,pdf}
"""
import os, re, json, subprocess
import numpy as np
import pandas as pd

HMMALIGN = "/data1/leihuang/miniconda3/envs/multiflow/bin/hmmalign"
HMM = "/data1/leihuang/rCLAMPS/pfamHMMs/Homeobox.hmm"
HD_CMAP = "/data1/leihuang/rCLAMPS/precomputedInputs/homeodomain_contactMap.txt"
ALA = "results/per_family/alascan_population.json"
SEQP = "data/processed/tf_pwm_aug_dbd.parquet"
OUTJ = "results/per_family/fig2c_recognition_code.json"
OUTD = "figures/figure2c_recognition_code"
os.makedirs(OUTD, exist_ok=True)

rows = json.load(open(ALA))["rows"]
seqdf = pd.read_parquet(SEQP, columns=["filename", "sequence", "dbd_start", "dbd_end"])
seqdf["fn"] = seqdf["filename"].astype(str).str.replace(".txt", "", regex=False)

def dbd_for(fn):
    m = seqdf[seqdf.fn == fn]
    if len(m) == 0:
        m = seqdf[seqdf.fn.str.startswith("_".join(fn.split("_")[:3]))]
    if len(m) == 0: return None
    r = m.iloc[0]; return str(r.sequence)[int(r.dbd_start):int(r.dbd_end)]

# rCLAMPS homeodomain recognition HMM match states (1-based)
HD_REC = sorted(set(int(l.split()[1]) for l in open(HD_CMAP)
                    if l.strip() and not l.startswith("bpos")))

def auroc(pos_vals, neg_vals):
    pos, neg = np.asarray(pos_vals, float), np.asarray(neg_vals, float)
    if len(pos) == 0 or len(neg) == 0: return np.nan
    wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
               for _ in [0])
    return float(wins) / (len(pos) * len(neg))

# ─────────────────────────── homeodomain (HMM-aligned) ───────────────────────────
hd = [r for r in rows if r["family"] == "Homeodomain"]
recs = []
for r in hd:
    fn = r["filename"].replace(".txt", ""); dbd = dbd_for(fn)
    if dbd is None: continue
    recs.append((fn, dbd, np.nan_to_num(np.array(r["imp"], float)), r["gene"], r.get("auc", np.nan)))
with open("/tmp/fig2c_hd.fa", "w") as f:
    for fn, dbd, *_ in recs: f.write(f">{fn}\n{dbd}\n")
subprocess.run([HMMALIGN, "--trim", "--amino", "-o", "/tmp/fig2c_hd.sto", HMM, "/tmp/fig2c_hd.fa"],
               check=True, capture_output=True)

# parse stockholm
aln = {}; rf = ""
for line in open("/tmp/fig2c_hd.sto"):
    if line.startswith("#=GC RF"): rf += line.split()[2]
    elif line.startswith("#") or line.startswith("//") or not line.strip(): continue
    else:
        p = line.split()
        if len(p) == 2: aln[p[0]] = aln.get(p[0], "") + p[1]

NCOL = 57   # Homeobox match states
metagene = {c: [] for c in range(1, NCOL + 1)}     # HMM col -> list of per-TF z-importance
hd_results = []
for fn, dbd, imp, gene, geom_auc in recs:
    a = aln[fn]
    z = (imp - imp.mean()) / (imp.std() + 1e-9)
    ms2res = {}; ridx = -1; ms = 0
    for i, ch in enumerate(a):
        match = rf[i] != "."
        if ch not in "-.": ridx += 1
        if match:
            ms += 1
            if ch not in "-.": ms2res[ms] = ridx
    rec_idx = sorted(set(ms2res[m] for m in HD_REC if m in ms2res))
    rest_idx = [i for i in range(len(imp)) if i not in rec_idx]
    if not rec_idx or not rest_idx: continue
    a_uroc = auroc(imp[rec_idx], imp[rest_idx])
    k = len(rec_idx)
    topk = set(np.argsort(-imp)[:k].tolist())
    prec = len(topk & set(rec_idx)) / k
    hd_results.append(dict(fn=fn, gene=gene, n_rec=k,
                           auroc=round(a_uroc, 3), precision=round(prec, 3),
                           geom_auroc=(round(float(geom_auc), 3) if geom_auc == geom_auc else None),
                           rec_mean_z=round(float(z[rec_idx].mean()), 3),
                           rest_mean_z=round(float(z[rest_idx].mean()), 3)))
    for m, ri in ms2res.items(): metagene[m].append(float(z[ri]))

meta_mean = {c: (float(np.mean(v)) if v else np.nan) for c, v in metagene.items()}

# ─────────────────────────── C2H2 (helix code + Zn ligands) ───────────────────────
C2H2 = re.compile(r"C.{2,4}C.{8,18}H.{3,6}H")
CODE_OFF = {-1, 2, 3, 6}     # helix positions relative to helix start (+1 = first helical res)
c2_rec, c2_zn, c2_other = [], [], []   # z-importance pools
c2_results = []
for r in rows:
    if "C2H2" not in r["family"]: continue
    fn = r["filename"].replace(".txt", ""); dbd = dbd_for(fn)
    if dbd is None: continue
    imp = np.nan_to_num(np.array(r["imp"], float)); z = (imp - imp.mean()) / (imp.std() + 1e-9)
    rec_idx, zn_idx = set(), set()
    for m in C2H2.finditer(dbd):
        seg = dbd[m.start():m.end()]
        cpos = [i for i, a in enumerate(seg) if a == "C"]
        hpos = [i for i, a in enumerate(seg) if a == "H"]
        if len(cpos) < 2 or len(hpos) < 2: continue
        helix0 = m.start() + cpos[1] + 2            # +1 (first helical residue)
        for off in CODE_OFF:
            p = helix0 + (off - 1)                  # off=+1 -> helix0
            if 0 <= p < len(dbd): rec_idx.add(p)
        for c in cpos[:2] + hpos[-2:]:              # Zn-coordinating Cys/His
            zn_idx.add(m.start() + c)
    other_idx = set(range(len(dbd))) - rec_idx - zn_idx
    if not rec_idx: continue
    c2_rec += [z[i] for i in rec_idx]; c2_zn += [z[i] for i in zn_idx]
    c2_other += [z[i] for i in other_idx]
    if rec_idx and other_idx:
        c2_results.append(dict(fn=fn, gene=r["gene"], n_rec=len(rec_idx),
                               # recognition code vs OTHER non-structural positions (Zn ligands excluded)
                               auroc_vs_other=round(auroc([imp[i] for i in rec_idx],
                                                          [imp[i] for i in other_idx]), 3),
                               rec_mean_z=round(float(np.mean([z[i] for i in rec_idx])), 3),
                               zn_mean_z=round(float(np.mean([z[i] for i in zn_idx])) if zn_idx else np.nan, 3)))

# ─────────────────────────── summary ───────────────────────────
hd_auc = [x["auroc"] for x in hd_results]
summary = dict(
    hd_n=len(hd_results), hd_median_auroc=round(float(np.median(hd_auc)), 3),
    hd_frac_above_chance=round(float(np.mean(np.array(hd_auc) > 0.5)), 3),
    hd_rec_mean_z=round(float(np.mean([x["rec_mean_z"] for x in hd_results])), 3),
    hd_rest_mean_z=round(float(np.mean([x["rest_mean_z"] for x in hd_results])), 3),
    c2_n=len(c2_results),
    c2_rec_mean_z=round(float(np.mean(c2_rec)), 3),
    c2_zn_mean_z=round(float(np.mean(c2_zn)), 3),
    c2_other_mean_z=round(float(np.mean(c2_other)), 3),
    c2_code_vs_other_auroc=round(float(auroc(c2_rec, c2_other)), 3),
    hd_rec_positions=HD_REC, c2_code_offsets=sorted(CODE_OFF),
)
json.dump(dict(summary=summary, homeodomain=hd_results, c2h2=c2_results,
               metagene={str(c): meta_mean[c] for c in meta_mean}),
          open(OUTJ, "w"), indent=1)
print("=== Fig 2c recognition-code summary ===")
for k, v in summary.items():
    if "positions" not in k and "offsets" not in k: print(f"  {k}: {v}")
print(f"  saved {OUTJ}")

# ─────────────────────────── figure ───────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [2.0, 1.05]})

# (a) homeodomain importance metagene over HMM columns; recognition cols highlighted
cols = list(range(1, NCOL + 1))
vals = [meta_mean[c] if not np.isnan(meta_mean[c]) else 0 for c in cols]
isrec = [c in HD_REC for c in cols]
ax[0].bar([c for c, r in zip(cols, isrec) if not r], [v for v, r in zip(vals, isrec) if not r],
          color="#c9ccd1", width=0.85, label="other domain positions")
ax[0].bar([c for c, r in zip(cols, isrec) if r], [v for v, r in zip(vals, isrec) if r],
          color="#d73027", width=0.85, label="rCLAMPS recognition positions")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xlabel("Homeobox HMM position (Pfam PF00046)", fontsize=10)
ax[0].set_ylabel("TFScope importance (mean z-score)", fontsize=10)
ax[0].set_title(f"a  Homeodomain importance peaks at the recognition code (n={len(hd_results)} TFs)",
                fontsize=10.5, fontweight="bold", loc="left")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[0].set_ylim(top=max(vals) * 1.18)
ax[0].annotate("recognition helix-3\n(pos 50, Asn51)", (50.5, meta_mean[50]),
               xytext=(33, max(vals) * 1.02), fontsize=7.8, color="#7a1a12", ha="center",
               arrowprops=dict(arrowstyle="-", color="#7a1a12", lw=0.7))
ax[0].annotate("N-terminal arm", (3.5, meta_mean[3]),
               xytext=(10, max(vals) * 0.82), fontsize=7.8, color="#7a1a12", ha="center",
               arrowprops=dict(arrowstyle="-", color="#7a1a12", lw=0.7))

# (b) per-TF AUROC: geometric contacts (Fig 2b) vs recognition code (Fig 2c), paired
import numpy as np
g = [x["geom_auroc"] for x in hd_results if x["geom_auroc"] is not None]
c = [x["auroc"] for x in hd_results if x["geom_auroc"] is not None]
xj = np.random.RandomState(0).uniform(-0.04, 0.04, len(g))
for gi, ci, j in zip(g, c, xj):
    ax[1].plot([1 + j, 2 + j], [gi, ci], "-", color="#bbb", lw=0.7, zorder=1)
ax[1].scatter(np.ones(len(g)) + xj, g, s=22, color="#7f7f7f", zorder=2)
ax[1].scatter(np.full(len(c), 2) + xj, c, s=22, color="#d73027", zorder=2)
for xpos, vals_ in [(1, g), (2, c)]:
    ax[1].plot([xpos - 0.18, xpos + 0.18], [np.median(vals_)] * 2, "k-", lw=2, zorder=3)
    ax[1].text(xpos, np.median(vals_) + 0.02, f"{np.median(vals_):.2f}", ha="center", fontsize=8, fontweight="bold")
ax[1].axhline(0.5, color="k", ls=":", lw=0.8); ax[1].text(2.42, 0.505, "chance", fontsize=7, va="bottom")
ax[1].set_xlim(0.6, 2.4); ax[1].set_xticks([1, 2])
ax[1].set_xticklabels(["crystal\ncontacts (2b)", "learned code\n(2c, rCLAMPS)"], fontsize=8.5)
ax[1].set_ylabel("per-TF AUROC (importance ranks targets)", fontsize=9.5)
ax[1].set_title("b  Recovered across two\nindependent ground truths", fontsize=10.5,
                fontweight="bold", loc="left")

fig.suptitle("In-silico mutagenesis recovers an experimentally-derived recognition code (rCLAMPS; Wetzel et al. 2022)",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.tight_layout()
out = f"{OUTD}/figure2c_recognition_code"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"  saved {out}.png/.pdf")

# ── supplementary: C2H2 alanine-scan confound (structural Zn ligands) ──
figs, axs = plt.subplots(figsize=(4.6, 4.2))
data = [c2_rec, c2_other, c2_zn]
labels = ["recognition\ncode\n(-1,+2,+3,+6)", "other\npositions", "Zn ligands\n(Cys/His)"]
colors = ["#d73027", "#c9ccd1", "#4575b4"]
bp = axs.boxplot(data, patch_artist=True, widths=0.62, showfliers=False, medianprops=dict(color="k"))
for patch, cc in zip(bp["boxes"], colors): patch.set_facecolor(cc); patch.set_alpha(0.85)
for i, d in enumerate(data, 1):
    axs.scatter(np.random.RandomState(i).uniform(i - 0.16, i + 0.16, len(d)), d, s=7,
                color="k", alpha=0.25, zorder=3)
axs.axhline(0, color="k", lw=0.7, ls=":")
axs.set_xticklabels(labels, fontsize=8.5)
axs.set_ylabel("TFScope importance (z-score)", fontsize=10)
axs.set_title(f"C2H2 zinc finger (n={len(c2_results)} TFs):\nalanine scan flags structural Zn ligands",
              fontsize=10, fontweight="bold")
figs.tight_layout()
outs = f"{OUTD}/figureS_c2h2_zn_confound"
figs.savefig(outs + ".png", dpi=300, bbox_inches="tight")
figs.savefig(outs + ".pdf", bbox_inches="tight")
print(f"  saved {outs}.png/.pdf  (supplementary)")
