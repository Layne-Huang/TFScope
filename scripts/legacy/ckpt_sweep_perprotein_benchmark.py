"""Sweep per-protein-text CHECKPOINTS on the cluster40 benchmark (84 test TFs).
Test text embeddings (each protein's own metadata) are checkpoint-independent -> built once.
Reports panel-r, top-1, MAE(0-2 scale, =panel-mae x4), IC-r, KL per epoch.
Reference: combined panel-r 0.643 / MAE 0.651 ; DeepPBS 0.634 / 0.657.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length
from eval_full_metrics import trimmed_core, aligned_cols, panel

ROOT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42"
CKPTS = ["ckpt_epoch025.pt","ckpt_epoch050.pt","ckpt_epoch075.pt","ckpt_epoch100.pt",
         "ckpt_epoch125.pt","ckpt_epoch150.pt","ckpt_epoch175.pt","ckpt_epoch200.pt",
         "ckpt_epoch225.pt","ckpt_best.pt"]
DATA = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
TEMPLATE = lambda g, o, f: f"Transcription factor {g} from {o}, {f} family."

cfg0 = TFScopeConfig()
for k, v in json.load(open(os.path.join(ROOT, "config.json"))).items():
    if hasattr(cfg0, k):
        try: setattr(cfg0, k, type(getattr(cfg0, k))(v))
        except: pass
cfg0.use_retrieval = False
ds = TFDataset(cfg0, DATA, SPLIT, split="test", max_seq_len=1024)
fns = list(ds.filenames)
df = pd.read_parquet(DATA)
meta = {str(r.filename): (str(r.gene_symbol), str(r.organism), str(r.family_name)) for r in df.itertuples()}

# build 84 test text embeddings ONCE (checkpoint-independent)
from transformers import AutoTokenizer, AutoModel
mid = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD = "/data1/leihuang/.cache"
tok = AutoTokenizer.from_pretrained(mid, cache_dir=CD, local_files_only=True)
bert = AutoModel.from_pretrained(mid, cache_dir=CD, local_files_only=True).to(dev).eval()
@torch.no_grad()
def tvec(s):
    e = {k: v.to(dev) for k, v in tok(s, return_tensors="pt", truncation=True, max_length=128).items()}
    return F.normalize(bert(**e).last_hidden_state[:, 0, :], dim=-1)[0].cpu()
test_vecs = torch.stack([tvec(TEMPLATE(*meta[fn])) for fn in fns])
del bert; torch.cuda.empty_cache()
print(f"built {len(test_vecs)} test text embeddings")

def benchmark(ck):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ROOT, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    sd = torch.load(os.path.join(ROOT, ck), map_location=dev, weights_only=False)
    m.load_state_dict(sd["model"], strict=False); ep = sd.get("epoch")
    fe = m.moe.family_embed; n0 = fe.vectors.shape[0]
    fe.vectors = torch.cat([fe.vectors, test_vecs.to(fe.vectors.dtype).to(dev)], 0)
    fn2off = {fn: n0 + i for i, fn in enumerate(fns)}
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_variable_length)
    preds, tgts = {}, {}; gi = 0
    with torch.no_grad():
        for b in ld:
            bs = b["sequence_tokens"].shape[0]
            fid = torch.tensor([fn2off[fns[gi + j]] for j in range(bs)], dtype=torch.long, device=dev)
            _, pw, _ = m(b["sequence_tokens"].to(dev), b["dbd_mask"].to(dev), fid,
                         retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
            P = F.softmax(pw, 1).cpu().numpy(); T = b["target_pwm"].numpy(); M = b["pwm_mask"].numpy()
            for j in range(bs):
                fn = fns[gi + j]; preds[fn] = P[j][:, M[j].astype(bool)]; tgts[fn] = (T[j], M[j])
            gi += bs
    acc = {k: [] for k in ["r", "mae", "icr", "kl", "top1"]}
    for fn in fns:
        core = trimmed_core(*tgts[fn]); pv = preds[fn]
        if core is None or pv.shape[1] == 0: continue
        al, cols, _ = aligned_cols(pv, core); d = panel(core, al, cols)
        if d:
            for k in acc: acc[k].append(d[k])
    del m; torch.cuda.empty_cache()
    return ep, {"panel_r": round(np.nanmean(acc["r"]), 3), "top1": round(np.nanmean(acc["top1"]), 3),
                "MAE_0_2": round(np.nanmean(acc["mae"]) * 4, 3), "ic_r": round(np.nanmean(acc["icr"]), 3),
                "KL": round(np.nanmean(acc["kl"]), 3)}

out = {}
print(f"\n{'epoch':>6} {'panel_r':>8} {'top1':>6} {'MAE(0-2)':>9} {'ic_r':>6} {'KL':>6}")
for ck in CKPTS:
    ep, mm = benchmark(ck)
    out[ck] = dict(epoch=ep, **mm)
    print(f"{str(ep):>6} {mm['panel_r']:>8} {mm['top1']:>6} {mm['MAE_0_2']:>9} {mm['ic_r']:>6} {mm['KL']:>6}", flush=True)
out["_reference"] = {"combined": {"panel_r": 0.643, "top1": 0.714, "MAE_0_2": 0.651},
                     "deeppbs": {"panel_r": 0.634, "top1": 0.704, "MAE_0_2": 0.657}}
json.dump(out, open("results/v19_deeppbs/perprotein_ckpt_sweep.json", "w"), indent=1)
print("\nref: combined panel_r 0.643 MAE 0.651 ; DeepPBS 0.634 MAE 0.657")
print("saved results/v19_deeppbs/perprotein_ckpt_sweep.json")
