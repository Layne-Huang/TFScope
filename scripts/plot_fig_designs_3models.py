"""Designed TFs (DBP5/6/9/35): predicted PWM logos for combined+contact, semfam34,
and per-protein-text (via homology), over the experimental specificity heatmap.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import logomaker
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from tfscope.data.dataset import AA_TO_TOKEN

DESIGNS = ["DBP005", "DBP006", "DBP009", "DBP035"]
TITLES = ["DBP05", "DBP06", "DBP09", "DBP35"]
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
         "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
BASES = ["A", "C", "G", "T"]; B2I = {b: i for i, b in enumerate(BASES)}
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {}
for r in parq.itertuples():
    g = str(r.gene_symbol).upper()
    if g not in gmeta: gmeta[g] = (str(r.organism), str(r.family_name))

# ── experimental specificity matrix (4 x 14): Median PE/FITC, WT base = WT-row value ──
def exp_matrix(d):
    df = pd.ExcelFile(XLS).parse(SHEET[d]); vc = df.columns[-1]
    df = df.copy(); df["pos_s"] = df["position"].astype(str).str.strip()
    wt_rows = df[df["pos_s"].str.upper() == "WT"]
    wt_val = float(wt_rows[vc].iloc[0]) if len(wt_rows) else float("nan")
    dd = df[df["pos_s"].str.fullmatch(r"\d+")].copy(); dd["posi"] = dd["pos_s"].astype(int)
    L = int(dd["posi"].max())
    if not (wt_val == wt_val):                      # no WT row -> fallback to median signal
        wt_val = float(np.nanmedian(pd.to_numeric(dd[vc], errors="coerce")))
    M = np.full((4, L), np.nan); wtbase = {}
    for p in range(1, L + 1):
        sub = dd[dd["posi"] == p]
        if not len(sub): wtbase[p] = "N"; continue
        ob = str(sub["original_base"].iloc[0]); wtbase[p] = ob
        if ob in B2I: M[B2I[ob], p - 1] = wt_val
        for _, r in sub.iterrows():
            nb = str(r["new_base"])
            if nb in B2I: M[B2I[nb], p - 1] = float(r[vc])
    return M, wtbase, wt_val

# ── models ──
def load(ckpt):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"], strict=False)
    return m
_bert = None
def tvec(s):
    global _bert
    from transformers import AutoTokenizer, AutoModel
    if _bert is None:
        mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD = "/data1/leihuang/.cache"
        _bert = (AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True),
                 AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval())
    tk, bt = _bert
    with torch.no_grad():
        e = {k: v.to(dev) for k, v in tk(s, return_tensors="pt", truncation=True, max_length=128).items()}
        return F.normalize(bt(**e).last_hidden_state[:, 0, :], dim=-1)[0]
@torch.no_grad()
def predict(m, seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

CKPTS = {
 "combined+contact": "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
 "semfam34 (semantic, HTH)": "/data1/leihuang/project/TFScope/checkpoints/v19_combined_semfam34_contact_fixed/rag_seed42/ckpt_best.pt",
 "RAG + contact": "/data1/leihuang/project/TFScope/checkpoints/v19_combined_rag_contact/rag_seed42/ckpt_best.pt",
}
ROWCOL = {"combined+contact": "#1b7837", "semfam34 (semantic, HTH)": "#762a83", "RAG + contact": "#b35806"}
exps = {d: exp_matrix(d) for d in DESIGNS}
# align each predicted PWM to the experimental WT-base consensus frame (for display)
def wt_onehot(d):
    _, wtbase, _ = exps[d]; L = len(wtbase)
    P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1): P[B2I[wtbase[p]], p - 1] = 1.0
    return P / P.sum(0, keepdims=True)

preds = {name: {} for name in CKPTS}
for name, ckpt in CKPTS.items():
    m = load(ckpt)
    for d in DESIGNS:
        seq = e2[d]["prot_seq"]
        if "homology" in name:
            hg = str(e2[d]["top_donor"]); org, fam = gmeta.get(hg.upper(), ("Homo sapiens", "Homeodomain"))
            fe = m.moe.family_embed
            fe.vectors = torch.cat([fe.vectors, tvec(f"Transcription factor {hg} from {org}, {fam} family.")[None].to(fe.vectors.dtype).to(dev)], 0)
            fid = fe.vectors.shape[0] - 1
        else:
            fid = 4
        pv = predict(m, seq, fid)
        # orient (RC if needed) to match the experimental frame, but show the FULL motif
        core = wt_onehot(d)
        _, shift, orient, score = align_pwm(pv, core, max_shift=10, consider_revcomp=True, min_overlap=4)
        preds[name][d] = revcomp_pwm_np(pv) if orient == "rc" else pv   # full predicted PWM
    del m; torch.cuda.empty_cache()

# ── figure: 4 rows (3 logos + heatmap) x 4 designs ──
fig, axes = plt.subplots(4, 4, figsize=(20, 11),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1.3], "hspace": 0.55, "wspace": 0.18})
def logo(ax, pwm, title, color):
    p = np.clip(pwm.T.astype(float), 1e-6, 1); p = p / p.sum(1, keepdims=True)   # (L,4)
    ic = 2.0 + (p * np.log2(p)).sum(1)                                            # (L,) bits
    info = pd.DataFrame(p * ic[:, None], columns=BASES)                           # IC-scaled heights
    logomaker.Logo(info, ax=ax, color_scheme="classic")
    ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.set_ylabel("Bits", fontsize=8)
    ax.set_xticks([]); ax.set_title(title, fontsize=10, color=color)
for j, d in enumerate(DESIGNS):
    for i, name in enumerate(CKPTS):
        logo(axes[i, j], preds[name][d], f"{TITLES[j]}  {name}", ROWCOL[name])
    # heatmap
    M, wtbase, wtv = exps[d]; ax = axes[3, j]
    norm = TwoSlopeNorm(vmin=np.nanmin(M), vcenter=wtv, vmax=np.nanmax(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(4)); ax.set_yticklabels(BASES, fontsize=8)
    ax.set_xticks(range(len(wtbase)))
    ax.set_xticklabels([f"{p}\n{wtbase[p]}" for p in range(1, len(wtbase) + 1)], fontsize=7)
    for p in range(1, len(wtbase) + 1):
        ax.text(p - 1, B2I[wtbase[p]], wtbase[p], ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Positions / WT base", fontsize=8)
    if j == 0: ax.set_ylabel("experimental", fontsize=9)

cbar = fig.colorbar(im, ax=axes[3, :].tolist(), orientation="horizontal", fraction=0.05, pad=0.18, aspect=40)
cbar.set_label("Median PE/FITC (low=stronger binding=blue, high=weaker=red; WT=white)", fontsize=9)
fig.suptitle("Designed TFs (DBP5/6/9/35): combined+contact, semfam34 & RAG+contact predicted PWM (top) vs experimental specificity (bottom)",
             fontsize=13, fontweight="bold", y=0.98)
out = "results/ood_case_study/fig5_rag"
fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png /.pdf")
