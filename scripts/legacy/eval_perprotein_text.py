"""Benchmark the per-protein-text model on cluster40 test (84), conditioning each
held-out protein on its OWN pubmedbert text embedding (gene+organism+family) —
leakage-free metadata. The 84 test embeddings are appended to the frozen family
vector buffer and addressed via offset family_ids; predictions are scored with the
same oracle-aligned panel-r as the baseline ladder.

Template reverse-engineered from the training embeddings (cos>=0.99):
   "Transcription factor {gene} from {organism}, {family} family."
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length
from eval_full_metrics import trimmed_core, aligned_cols, panel

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
CD = "/data1/leihuang/.cache"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
TEMPLATE = lambda g, o, f: f"Transcription factor {g} from {o}, {f} family."

# ── model ──
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
fe = m.moe.family_embed                       # SemanticFamilyEmbedding (.vectors, .proj)
n_train = fe.vectors.shape[0]
print(f"family vectors: {tuple(fe.vectors.shape)} (train proteins)")

# ── dataset (test order) ──
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
fns = list(ds.filenames)
df = pd.read_parquet(DATA)
meta = {str(r.filename): (str(r.gene_symbol), str(r.organism), str(r.family_name)) for r in df.itertuples()}

# ── build 84 test text embeddings (pubmedbert CLS, L2-norm) in dataset order ──
from transformers import AutoTokenizer, AutoModel
mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tok = AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True)
bert = AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval()
@torch.no_grad()
def textemb(s):
    e = {k: v.to(dev) for k, v in tok(s, return_tensors="pt", truncation=True, max_length=128).items()}
    return F.normalize(bert(**e).last_hidden_state[:, 0, :], dim=-1)[0].cpu()
test_vecs = torch.stack([textemb(TEMPLATE(*meta[fn])) for fn in fns])    # (84, 768)
print(f"built {test_vecs.shape[0]} test text embeddings")

# ── append to frozen buffer; test protein i -> family_id n_train+i ──
fe.vectors = torch.cat([fe.vectors, test_vecs.to(fe.vectors.dtype).to(dev)], 0)
fn2offset = {fn: n_train + i for i, fn in enumerate(fns)}

# ── inference with overridden per-protein family_id ──
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
preds, tgts = {}, {}
gi = 0
with torch.no_grad():
    for b in ld:
        bs = b["sequence_tokens"].shape[0]
        fid = torch.tensor([fn2offset[fns[gi + j]] for j in range(bs)], dtype=torch.long, device=dev)
        toks = b["sequence_tokens"].to(dev); dm = b["dbd_mask"].to(dev)
        _, pw, _ = m(toks, dm, fid, retrieved_pwms=None, retrieved_masks=None,
                     retrieved_sims=None, recog_prior=None)
        P = F.softmax(pw, 1).cpu().numpy(); T = b["target_pwm"].numpy(); M = b["pwm_mask"].numpy()
        for j in range(bs):
            fn = fns[gi + j]
            preds[fn] = P[j][:, M[j].astype(bool)]; tgts[fn] = (T[j], M[j])
        gi += bs

# ── score (oracle-aligned panel-r, same as ladder) ──
rs, top1, mae = [], [], []
for fn in fns:
    core = trimmed_core(*tgts[fn]); pv = preds[fn]
    if core is None or pv.shape[1] == 0: continue
    aligned, cols, _ = aligned_cols(pv, core)
    d = panel(core, aligned, cols)
    if d: rs.append(d["r"]); top1.append(d["top1"]); mae.append(d["mae"])
res = dict(model="v19_combined_perprotein_text", n=len(rs),
           panel_r=round(float(np.nanmean(rs)), 3),
           top1=round(float(np.nanmean(top1)), 3),
           mae=round(float(np.nanmean(mae)), 3),
           ref={"tfscope_combined": 0.643, "deeppbs": 0.634})
json.dump(res, open("results/v19_deeppbs/perprotein_text_cluster40.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("saved results/v19_deeppbs/perprotein_text_cluster40.json")
