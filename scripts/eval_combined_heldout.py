"""Run the COMBINED rag_contact model (Fig 1 model, for consistency) on the cluster40_clean
held-out test, emitting a predictions npz in the same format as the model-composition evals.
Excludes the 26 test TFs that are in the combined model's training split (combined_fm_deeppbs)
to remove direct leakage. Retrieval uses the cluster40_clean index (donors <40% identity to
test → no homolog leakage in retrieval).

Out: results/fig3a_heldout/combined_heldout_predictions.npz
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_rag_contact/rag_seed42/ckpt_best.pt"
DATA = "data/processed/tf_pwm_aug_dbd.parquet"
SPLIT = "data/processed/splits/cluster40_clean/split.json"
IDX = "data/processed/tf_nn_index_cluster40_clean.json"
OUT = "results/fig3a_heldout/combined_heldout_predictions.npz"
os.makedirs("results/fig3a_heldout", exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# TFs the combined model trained on (exclude from held-out eval)
combined_train = set(json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))["train"])

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.retrieval_index_path = IDX
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

meta = pd.read_parquet(DATA, columns=["filename", "gene_symbol", "family_name"])
meta["fn"] = meta.filename.astype(str)
g_by = dict(zip(meta.fn, meta.gene_symbol)); f_by = dict(zip(meta.fn, meta.family_name))

ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
rows = []; gi = 0
with torch.no_grad():
    for b in ld:
        bs = b["sequence_tokens"].shape[0]
        bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        gl, pw, _ = m(bt["sequence_tokens"], bt["dbd_mask"], bt["family_id"],
                      retrieved_pwms=bt.get("retrieved_pwms"), retrieved_masks=bt.get("retrieved_masks"),
                      retrieved_sims=bt.get("retrieved_sims"), recog_prior=bt.get("recog_prior"))
        P = F.softmax(pw, 1).cpu().numpy(); Gt = torch.sigmoid(gl).cpu().numpy()
        T = b["target_pwm"].numpy(); M = b["pwm_mask"].numpy()
        for j in range(bs):
            fn = ds.filenames[gi + j]
            if fn in combined_train: continue                    # drop leaked
            rows.append((fn, g_by.get(fn, fn), f_by.get(fn, "?"), P[j], Gt[j], T[j], M[j]))
        gi += bs

W = max(r[3].shape[1] for r in rows)
def pad(a, w):
    if a.ndim == 2: return np.pad(a, ((0, 0), (0, w - a.shape[1])))
    return np.pad(a, (0, w - a.shape[0]))
np.savez(OUT,
         filename=np.array([r[0] for r in rows]), gene=np.array([r[1] for r in rows]),
         family=np.array([r[2] for r in rows]),
         prediction=np.stack([pad(r[3], W) for r in rows]),
         gate=np.stack([pad(r[4], W) for r in rows]),
         target=np.stack([pad(r[5], W) for r in rows]),
         mask=np.stack([pad(r[6], W) for r in rows]))
print(f"saved {OUT}: {len(rows)} held-out TFs (excluded {614 - len(rows)} leaked/train), W={W}")
