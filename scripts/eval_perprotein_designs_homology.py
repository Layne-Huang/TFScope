"""Apply per-protein-text to de novo designs via HOMOLOGY TRANSFER: condition each
design on the per-protein text of its nearest natural homolog (top_donor, ESM2
cosine ~0.95). Score predicted PWM vs the design's bound DNA (one-hot), oracle-aligned.
Compare to the E2 model's stored design panel-r.
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

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}
designs = json.load(open("results/design_case_study/design_e2_predictions.json"))
parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {}
for r in parq.itertuples():
    g = str(r.gene_symbol).upper()
    if g not in gmeta: gmeta[g] = (str(r.organism), str(r.family_name))

# pubmedbert text encoder
from transformers import AutoTokenizer, AutoModel
mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD = "/data1/leihuang/.cache"
tok = AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True)
bert = AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval()
@torch.no_grad()
def tvec(s):
    e = {k: v.to(dev) for k, v in tok(s, return_tensors="pt", truncation=True, max_length=128).items()}
    return F.normalize(bert(**e).last_hidden_state[:, 0, :], dim=-1)[0].cpu()

# model
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
fe = m.moe.family_embed; n0 = fe.vectors.shape[0]

# build homolog text vec per design
def onehot(seq):
    o = np.full((4, len(seq)), 1e-6, np.float32)
    for j, c in enumerate(seq):
        if c in B2I: o[B2I[c], j] = 1.0
    return o / o.sum(0, keepdims=True)

rows, vecs = [], []
for e in designs:
    hg = str(e["top_donor"]).upper()
    org, fam = gmeta.get(hg, ("Homo sapiens", "Homeodomain"))
    text = f"Transcription factor {e['top_donor']} from {org}, {fam} family."
    vecs.append(tvec(text)); rows.append(e)
fe.vectors = torch.cat([fe.vectors, torch.stack(vecs).to(fe.vectors.dtype).to(dev)], 0)

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

pp_r, e2_r = [], []
for i, e in enumerate(rows):
    gt = onehot(e["dna_gt"])
    pv = predict(e["prot_seq"], n0 + i)
    aligned, cols, _ = aligned_cols(pv, gt)
    d = panel(gt, aligned, cols)
    if d: pp_r.append(d["r"])
    e2_r.append(e["panel_r"])
res = dict(n=len(pp_r),
           perprotein_homology_panel_r=round(float(np.nanmean(pp_r)), 3),
           E2_panel_r=round(float(np.nanmean(e2_r)), 3),
           homolog_cos_median=round(float(np.median([e["top_cos"] for e in rows])), 3))
json.dump(res, open("results/design_case_study/perprotein_homology_designs.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("saved results/design_case_study/perprotein_homology_designs.json")
