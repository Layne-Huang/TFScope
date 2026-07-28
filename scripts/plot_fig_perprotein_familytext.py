"""Per-protein-text under 3 conditionings (homolog / generic-HD-text / HD-centroid)
on the 4 designs: logos over the experimental specificity heatmap, plus a printed
CAC-core recovery check vs the experimental WT-base core.
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

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt"
DESIGNS = ["DBP005", "DBP006", "DBP009", "DBP035"]; TIT = ["DBP05", "DBP06", "DBP09", "DBP35"]
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
         "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
BASES = list("ACGT"); B2I = {b: i for i, b in enumerate(BASES)}
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {str(r.gene_symbol).upper(): (str(r.organism), str(r.family_name)) for r in parq.itertuples()}
fn2fam = {str(r.filename): str(r.family_name) for r in parq.itertuples()}

def exp_matrix(d):
    df = pd.ExcelFile(XLS).parse(SHEET[d]); vc = df.columns[-1]; df = df.copy()
    df["ps"] = df.position.astype(str).str.strip()
    wt = df[df.ps.str.upper() == "WT"]; wtv = float(wt[vc].iloc[0]) if len(wt) else float("nan")
    dd = df[df.ps.str.fullmatch(r"\d+")].copy(); dd["p"] = dd.ps.astype(int); L = int(dd.p.max())
    if wtv != wtv: wtv = float(np.nanmedian(pd.to_numeric(dd[vc], errors="coerce")))
    M = np.full((4, L), np.nan); wtbase = {}
    for p in range(1, L + 1):
        sub = dd[dd.p == p]; ob = str(sub.original_base.iloc[0]); wtbase[p] = ob
        if ob in B2I: M[B2I[ob], p - 1] = wtv
        for _, r in sub.iterrows():
            nb = str(r["new_base"])
            if nb in B2I: M[B2I[nb], p - 1] = float(r[vc])
    return M, wtbase, wtv
exps = {d: exp_matrix(d) for d in DESIGNS}
def wt_onehot(d):
    _, wb, _ = exps[d]; L = len(wb); P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1):
        if wb[p] in B2I: P[B2I[wb[p]], p - 1] = 1.0
    return P / P.sum(0, keepdims=True)

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
fe = m.moe.family_embed; n0 = fe.vectors.shape[0]
from transformers import AutoTokenizer, AutoModel
mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD = "/data1/leihuang/.cache"
tok = AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True)
bert = AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval()
@torch.no_grad()
def tvec(s):
    e = {k: v.to(dev) for k, v in tok(s, return_tensors="pt", truncation=True, max_length=128).items()}
    return F.normalize(bert(**e).last_hidden_state[:, 0, :], dim=-1)[0].cpu()
E = torch.load("data/processed/perprotein_text_embeddings.pt", map_location="cpu", weights_only=False)
hidx = [i for i, f in enumerate(E["filenames"]) if fn2fam.get(f) == "Homeodomain"]
centroid = F.normalize(E["embeddings"][hidx].mean(0), dim=0)
generic = tvec("Transcription factor from Homo sapiens, Homeodomain family.")
@torch.no_grad()
def predict(seq, vec):
    fe.vectors = torch.cat([fe.vectors[:n0], vec[None].to(fe.vectors.dtype).to(dev)], 0)
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([n0], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); return F.softmax(pl, 1)[0].cpu().numpy()[:, :max(4, int((gate > 0.5).sum()))]

ROWS = {"homolog": "#2166ac", "generic HD text": "#1b7837", "HD centroid": "#b35806"}
preds = {r: {} for r in ROWS}; cons = {r: {} for r in ROWS}
for d in DESIGNS:
    hg = str(e2[d]["top_donor"]); org, fam = gmeta.get(hg.upper(), ("Homo sapiens", "Homeodomain"))
    vecs = {"homolog": tvec(f"Transcription factor {hg} from {org}, {fam} family."),
            "generic HD text": generic, "HD centroid": centroid}
    core = wt_onehot(d)
    for r, vec in vecs.items():
        pv = predict(e2[d]["prot_seq"], vec)
        _, sh, ori, _ = align_pwm(pv, core, max_shift=10, consider_revcomp=True, min_overlap=4)
        po = revcomp_pwm_np(pv) if ori == "rc" else pv
        preds[r][d] = po; cons[r][d] = "".join(np.array(BASES)[po.argmax(0)])

# ── CAC/CACA recovery report ──
print("\n=== CAC-core recovery (design WT-core vs predicted consensus) ===")
for d in DESIGNS:
    _, wb, _ = exps[d]; wtcore = "".join(wb[p] for p in range(1, len(wb) + 1))
    print(f"\n{d}  experimental WT-base core: {wtcore}   (contains CAC: {'CAC' in wtcore or 'GTG' in wtcore})")
    for r in ROWS:
        c = cons[r][d]; hit = "CAC" in c or "GTG" in c or "CACA" in c or "TGTG" in c
        print(f"    {r:16s}: {c:18s} {'✓ CAC' if hit else ''}")

# ── figure ──
fig, axes = plt.subplots(4, 4, figsize=(20, 11),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1.3], "hspace": 0.55, "wspace": 0.18})
def logo(ax, pwm, title, color):
    p = np.clip(pwm.T.astype(float), 1e-6, 1); p = p / p.sum(1, keepdims=True)
    ic = 2.0 + (p * np.log2(p)).sum(1)
    logomaker.Logo(pd.DataFrame(p * ic[:, None], columns=BASES), ax=ax, color_scheme="classic")
    ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.set_ylabel("Bits", fontsize=8); ax.set_xticks([])
    ax.set_title(title, fontsize=10, color=color)
for j, d in enumerate(DESIGNS):
    for i, r in enumerate(ROWS):
        logo(axes[i, j], preds[r][d], f"{TIT[j]}  per-protein: {r}", ROWS[r])
    M, wb, wtv = exps[d]; ax = axes[3, j]
    norm = TwoSlopeNorm(vmin=np.nanmin(M), vcenter=wtv, vmax=np.nanmax(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(4)); ax.set_yticklabels(BASES, fontsize=8)
    ax.set_xticks(range(len(wb))); ax.set_xticklabels([f"{p}\n{wb[p]}" for p in range(1, len(wb) + 1)], fontsize=7)
    for p in range(1, len(wb) + 1):
        if wb[p] in B2I: ax.text(p - 1, B2I[wb[p]], wb[p], ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Positions / WT base", fontsize=8)
    if j == 0: ax.set_ylabel("experimental", fontsize=9)
fig.colorbar(im, ax=axes[3, :].tolist(), orientation="horizontal", fraction=0.05, pad=0.18, aspect=40
             ).set_label("Median PE/FITC (low=stronger=blue, high=weaker=red; WT=white)", fontsize=9)
fig.suptitle("Per-protein-text on designs: homolog vs generic-HD-text vs HD-centroid conditioning (top) vs experimental (bottom)",
             fontsize=13, fontweight="bold", y=0.98)
out = "results/design_case_study/fig_perprotein_familytext"
fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("\nsaved", out + ".png")
