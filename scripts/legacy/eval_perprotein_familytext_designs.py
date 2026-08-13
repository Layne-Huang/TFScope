"""Can per-protein-text recover the design CAC core if conditioned on a HOMEODOMAIN
family-level text vector (like semfam34's semantic family embedding) instead of the
specific homolog? Test 3 conditionings on the 4 designs:
  homolog     nearest natural homolog's per-protein text (baseline)
  generic     generic homeodomain family text
  centroid    mean per-protein text embedding of all training Homeodomain proteins
Report 14-bp experimental-frame core-r + CAC-core recovery. (perprotein still training; ep-best.)
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt"
DESIGNS = ["DBP005", "DBP006", "DBP009", "DBP035"]
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
         "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}; BASES = np.array(list("ACGT"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {str(r.gene_symbol).upper(): (str(r.organism), str(r.family_name)) for r in parq.itertuples()}
fn2fam = {str(r.filename): str(r.family_name) for r in parq.itertuples()}

def exp_pref(d):
    df = pd.ExcelFile(XLS).parse(SHEET[d]); vc = df.columns[-1]
    df = df[df.position.astype(str).str.fullmatch(r"\d+")].copy(); df["p"] = df.position.astype(int)
    L = int(df.p.max()); P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1):
        sub = df[df.p == p]; ob = str(sub.original_base.iloc[0])
        if ob in B2I: P[B2I[ob], p - 1] = 1.0
        for _, r in sub.iterrows():
            nb = str(r["new_base"])
            if nb in B2I: P[B2I[nb], p - 1] = max(0.0, float(r[vc]))
    return P / P.sum(0, keepdims=True)
prefs = {d: exp_pref(d) for d in DESIGNS}

# ── model + text encoder ──
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

# homeodomain centroid (mean of training Homeodomain per-protein text embeddings)
E = torch.load("data/processed/perprotein_text_embeddings.pt", map_location="cpu", weights_only=False)
hidx = [i for i, f in enumerate(E["filenames"]) if fn2fam.get(f) == "Homeodomain"]
centroid = F.normalize(E["embeddings"][hidx].mean(0), dim=0)
generic = tvec("Transcription factor from Homo sapiens, Homeodomain family.")
print(f"homeodomain centroid from {len(hidx)} training proteins")

@torch.no_grad()
def predict(seq, vec):
    fe.vectors = torch.cat([fe.vectors[:n0], vec[None].to(fe.vectors.dtype).to(dev)], 0)
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([n0], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

def has_cac(pwm, core):                       # CAC core recovery (either strand), after orientation
    _, sh, ori, _ = align_pwm(pwm, core, max_shift=10, consider_revcomp=True, min_overlap=4)
    p = revcomp_pwm_np(pwm) if ori == "rc" else pwm
    con = "".join(BASES[p.argmax(0)])
    return ("CAC" in con) or ("GTG" in con), con

out = {}
for label in ["homolog", "generic", "centroid"]:
    rs = {}; cac = {}
    for d in DESIGNS:
        if label == "homolog":
            hg = str(e2[d]["top_donor"]); org, fam = gmeta.get(hg.upper(), ("Homo sapiens", "Homeodomain"))
            vec = tvec(f"Transcription factor {hg} from {org}, {fam} family.")
        elif label == "generic":
            vec = generic
        else:
            vec = centroid
        pv = predict(e2[d]["prot_seq"], vec)
        aligned, cols, _ = aligned_cols(pv, prefs[d]); dd = panel(prefs[d], aligned, cols)
        rs[d] = round(float(dd["r"]), 3) if dd else None
        ok, con = has_cac(pv, prefs[d]); cac[d] = con + (" ✓CAC" if ok else "")
    rs["mean"] = round(float(np.nanmean([v for v in rs.values() if v is not None])), 3)
    out[label] = {"core_r": rs, "consensus": cac}
    print(label, rs, flush=True)
    for d in DESIGNS: print(f"    {d}: {cac[d]}")
out["reference"] = {"combined": 0.349, "semfam34": 0.359}
json.dump(out, open("results/design_case_study/perprotein_familytext_designs.json", "w"), indent=1)
print("\nsaved results/design_case_study/perprotein_familytext_designs.json")
