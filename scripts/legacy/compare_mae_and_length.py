"""Clean-test comparison of TFScope variants vs DeepPBS:
  (1) PWM MAE  (panel() 'mae', identical protocol for every method)
  (2) motif-length prediction: predicted length vs true length, r + MAE(bp)

TFScope predicted length = #(gate > 0.5) columns (the gate head's job).
DeepPBS has no gate -> predicted length = width of its IC>=0.25 trimmed core.
True length = width of the IC-trimmed GT core (same reference for both).

Out: figures/figure_length_mae/{length_prediction_all.pdf,png}, results/length_mae/summary.json
"""
import os, sys, json
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from eval_full_metrics import trimmed_core, aligned_cols, panel

SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
DATA  = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
NPZ   = "/data1/leihuang/data/DeepPBS/output_cluster40/cluster40_deeppbs_preds.npz"
IC, MAXSH, MINPOS = 0.25, 10, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = "/data1/leihuang/project/TFScope/checkpoints"
MODELS = {
    "combined":     (f"{CK}/v19_combined_fm_deeppbs_contact/rag_seed42", "ckpt_best.pt"),
    "moe_base":     (f"{CK}/v19_residue_moe/residue_moe_seed42",         "ckpt_best.pt"),
    "contact_bias": (f"{CK}/v19_residue_moe_contactbias/contactbias_seed42", "ckpt_best.pt"),
    "deep_tune":    (f"{CK}/v19_residue_moe_deeptune/deeptune_ddp_seed42",   "ckpt_best.pt"),
}

def norm(p):
    p = np.clip(p, 1e-8, 1.0); return p / p.sum(0, keepdims=True)
def trim_ic(p):
    p = norm(p); ic = 2.0 + (p * np.log2(p)).sum(0); inf = np.where(ic >= IC)[0]
    return p[:, inf[0]:inf[-1] + 1] if len(inf) else p

def run_tfscope(ckdir, ckname):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckdir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(ckdir, ckname), map_location=dev,
                                 weights_only=False)["model"], strict=False)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    fns = ds.filenames
    out = []
    i = 0
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                          retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                          retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
            gate = torch.sigmoid(gl).cpu().numpy()
            pwm  = F.softmax(pl, 1).cpu().numpy()
            tgt  = b['target_pwm'].cpu().numpy(); msk = b['pwm_mask'].cpu().numpy()
            for j in range(len(gate)):
                fn = fns[i]; i += 1
                core = trimmed_core(tgt[j], msk[j], IC)
                if core is None or core.shape[1] < MINPOS: continue
                pred_len = int((gate[j] > 0.5).sum())
                if pred_len == 0: pred_len = int((gate[j] > gate[j].max() * 0.5).sum())
                pm = pwm[j][:, msk[j].astype(bool)]
                aligned, cols, _ = aligned_cols(pm, core, MAXSH)
                d = panel(core, aligned, cols)
                if d is None: continue
                out.append(dict(fn=fn, true_len=core.shape[1], pred_len=pred_len, mae=d["mae"], r=d["r"]))
    return out

def run_deeppbs():
    z = np.load(NPZ, allow_pickle=True)
    n = sum(1 for k in z.files if k.startswith("pred_"))
    te = set(json.load(open(SPLIT))["test"])
    df = pd.read_parquet(DATA); df = df[df.filename.isin(te)]
    pref = lambda s: "_".join(s.split("_")[:2]).lower()
    byp = {}
    for r in df.itertuples(): byp.setdefault(pref(r.filename), r)
    out = []
    for i in range(n):
        nm = str(z[f"name_{i}"]); r = byp.get(pref(nm))
        if r is None: continue
        tgt = norm(np.frombuffer(r.pwm, dtype=np.float32).reshape(4, -1))
        core = trimmed_core(tgt, np.ones(tgt.shape[1], bool), IC)
        if core is None or core.shape[1] < MINPOS: continue
        pred = trim_ic(norm(np.asarray(z[f"pred_{i}"], dtype=np.float32).T))
        aligned, cols, _ = aligned_cols(pred, core, MAXSH)
        d = panel(core, aligned, cols)
        if d is None: continue
        out.append(dict(fn=r.filename, true_len=core.shape[1], pred_len=pred.shape[1],
                        mae=d["mae"], r=d["r"]))
    return out

res = {}
for name, (ckdir, ckname) in MODELS.items():
    res[name] = run_tfscope(ckdir, ckname); print(f"{name}: n={len(res[name])}", flush=True)
res["DeepPBS"] = run_deeppbs(); print(f"DeepPBS: n={len(res['DeepPBS'])}", flush=True)

print(f"\n{'model':14}{'n':>5}{'PWM MAE':>10}{'len r':>8}{'len MAE(bp)':>13}{'len bias':>10}")
summary = {}
for name, rows in res.items():
    mae = float(np.mean([x["mae"] for x in rows]))
    tl = np.array([x["true_len"] for x in rows], float)
    pl = np.array([x["pred_len"] for x in rows], float)
    lr = float(np.corrcoef(tl, pl)[0, 1]) if len(tl) > 2 and pl.std() > 0 else np.nan
    lmae = float(np.mean(np.abs(pl - tl))); bias = float(np.mean(pl - tl))
    summary[name] = dict(n=len(rows), pwm_mae=round(mae, 4), len_r=round(lr, 3),
                         len_mae_bp=round(lmae, 2), len_bias_bp=round(bias, 2))
    print(f"{name:14}{len(rows):>5}{mae:>10.4f}{lr:>8.3f}{lmae:>13.2f}{bias:>+10.2f}")

os.makedirs("results/length_mae", exist_ok=True)
json.dump({"summary": summary, "per_record": {k: v for k, v in res.items()}},
          open("results/length_mae/summary.json", "w"), indent=1, default=float)

# NOTE: DeepPBS is EXCLUDED from the length comparison. It does not predict motif
# length — its output PWM width is fixed by the DNA present in the co-crystal
# (pred_i and gt_i have identical widths). Scoring its trimmed width as a "length
# prediction" is circular. Length inference is a TFScope-only capability (gate head).

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
tf_order = ["combined", "moe_base", "contact_bias", "deep_tune"]
all_order = tf_order + ["DeepPBS"]

fig = plt.figure(figsize=(16, 3.9))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.32)
for i, name in enumerate(tf_order):
    ax = fig.add_subplot(gs[0, i])
    rows = res[name]
    tl = np.array([x["true_len"] for x in rows], float)
    pl = np.array([x["pred_len"] for x in rows], float)
    s = summary[name]
    ax.plot([0, 21], [0, 21], ls="--", c="grey", lw=1, zorder=1)
    ax.scatter(tl, pl, s=26, c="#2b7bba", alpha=0.45, edgecolors="none", zorder=2)
    ax.set_title(f"{name}\nr={s['len_r']:.3f}   MAE={s['len_mae_bp']:.2f} bp", fontsize=9)
    ax.set_xlabel("True motif length (bp)")
    if i == 0: ax.set_ylabel("Predicted motif length (bp)")
    ax.set_xlim(0, 21); ax.set_ylim(0, 21); ax.set_aspect("equal")

# MAE bar panel (fair: identical panel() protocol + identical GT cores for all 5)
axb = fig.add_subplot(gs[0, 4])
vals = [summary[n]["pwm_mae"] for n in all_order]
cols = ["#2b7bba"] * 4 + ["#d95f0e"]
bars = axb.bar(range(5), vals, color=cols, width=0.68)
axb.axhline(summary["DeepPBS"]["pwm_mae"], ls="--", c="#d95f0e", lw=1, zorder=0)
for b, v in zip(bars, vals):
    axb.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=7.5)
axb.set_xticks(range(5)); axb.set_xticklabels(all_order, rotation=30, ha="right", fontsize=8)
axb.set_ylabel("PWM MAE  (lower = better)")
axb.set_ylim(0, max(vals) * 1.18)
axb.set_title("PWM MAE — all methods\n(same GT cores, same protocol)", fontsize=9)

fig.suptitle("Motif-length (gate) prediction and PWM MAE — clean deeppbs_cluster40 test set (n=84)\n"
             "DeepPBS omitted from length panels: its PWM width is set by the co-crystal DNA, not predicted",
             fontsize=10.5, fontweight="bold", y=1.10)
os.makedirs("figures/figure_length_mae", exist_ok=True)
for e in ("pdf", "png"):
    fig.savefig(f"figures/figure_length_mae/length_prediction_all.{e}", dpi=200, bbox_inches="tight")
print("\nsaved figures/figure_length_mae/length_prediction_all.{pdf,png}")
