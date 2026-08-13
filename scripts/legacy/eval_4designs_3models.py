"""Compare 3 models on the 4 experimentally-characterized de novo designs
(DBP005/006/009/035), scored by 14-bp EXPERIMENTAL-frame core-r against the
single-base-substitution binding preference (Glasscock et al. xls, Median PE/FITC).

Models:
  combined     learned-10, family_id=4 (Homeodomain)
  semfam34     rebin34 semantic, family_id=4 (Homeodomain)
  perprotein   per-protein text via HOMOLOGY transfer (nearest natural homolog's text)
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel

DESIGNS = ["DBP005", "DBP006", "DBP009", "DBP035"]
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
         "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {}
for r in parq.itertuples():
    g = str(r.gene_symbol).upper()
    if g not in gmeta: gmeta[g] = (str(r.organism), str(r.family_name))

def exp_pref(design):           # 4 x 14 experimental preference PWM
    df = pd.ExcelFile(XLS).parse(SHEET[design])
    df = df[df.position.astype(str).str.isdigit()]
    L = int(df.position.astype(int).max())
    P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1):
        sub = df[df.position.astype(int) == p]
        ob = str(sub.original_base.iloc[0]); P[B2I[ob], p - 1] = 1.0     # WT base = reference
        for r in sub.itertuples():
            nb = str(r.new_base)
            if nb in B2I: P[B2I[nb], p - 1] = max(0.0, float(getattr(r, "_4")))  # Median PE/FITC col
    return P / P.sum(0, keepdims=True)

xls_cols = pd.ExcelFile(XLS).parse(SHEET["DBP005"], nrows=1).columns.tolist()
VALCOL = xls_cols[-1]
def exp_pref(design):
    df = pd.ExcelFile(XLS).parse(SHEET[design])
    df = df[df.position.astype(str).str.isdigit()].copy()
    L = int(df.position.astype(int).max())
    P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1):
        sub = df[df.position.astype(int) == p]
        ob = str(sub.original_base.iloc[0])
        if ob in B2I: P[B2I[ob], p - 1] = 1.0
        for _, r in sub.iterrows():
            nb = str(r["new_base"])
            if nb in B2I: P[B2I[nb], p - 1] = max(0.0, float(r[VALCOL]))
    return P / P.sum(0, keepdims=True)

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
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

CKPTS = {
 "combined":    "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
 "semfam34":    "/data1/leihuang/project/TFScope/checkpoints/v19_combined_semfam34_contact_fixed/rag_seed42/ckpt_best.pt",
 "rag_contact": "/data1/leihuang/project/TFScope/checkpoints/v19_combined_rag_contact/rag_seed42/ckpt_best.pt",
}
prefs = {d: exp_pref(d) for d in DESIGNS}
out = {}
for name, ckpt in CKPTS.items():
    m = load(ckpt)
    rs = {}
    for d in DESIGNS:
        seq = e2[d]["prot_seq"]
        if name == "perprotein":
            hg = str(e2[d]["top_donor"]); org, fam = gmeta.get(hg.upper(), ("Homo sapiens", "Homeodomain"))
            fe = m.moe.family_embed
            fe.vectors = torch.cat([fe.vectors, tvec(f"Transcription factor {hg} from {org}, {fam} family.")[None].to(fe.vectors.dtype).to(dev)], 0)
            fid = fe.vectors.shape[0] - 1
        else:
            fid = 4    # Homeodomain
        pv = predict(m, seq, fid)
        aligned, cols, _ = aligned_cols(pv, prefs[d])
        dd = panel(prefs[d], aligned, cols)
        rs[d] = round(float(dd["r"]), 3) if dd else None
    rs["mean"] = round(float(np.nanmean([v for v in rs.values() if v is not None])), 3)
    out[name] = rs
    print(name, rs, flush=True)

json.dump(out, open("results/design_case_study/four_designs_rag.json", "w"), indent=1)
print("\n=== 14-bp experimental-frame core-r ==="); print(json.dumps(out, indent=1))
print("saved results/design_case_study/four_designs_3models.json")
