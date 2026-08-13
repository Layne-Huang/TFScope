#!/usr/bin/env python
"""Per-TF panel-r on deeppbs_cluster40 test, joined with each TF's homology
(top retrieval cosine to the train donor bank), so we can stratify into
in-distribution (high homology) vs OOD (low homology) and compare two models.

Usage:
  python scripts/eval_ood_by_homology.py <ckpt_dir> <data.parquet> <ckpt_name> <out.json>
Both models share the same split + NN index; only the parquet (family scheme)
and checkpoint differ.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from eval_full_metrics import trimmed_core, aligned_cols, panel
from torch.utils.data import DataLoader

CKPT_DIR, DATA, CKPT_NAME, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
NNIDX = "data/processed/tf_nn_index_deeppbs_cluster40.json"
IC_THRESH, MAX_SHIFT, MIN_POS = 0.25, 10, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: setattr(cfg, k, v)
cfg.use_retrieval = False  # as-trained deployment regime for both models

m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKPT_DIR, CKPT_NAME),
                             map_location=dev, weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

nn = json.load(open(NNIDX))
def homology(fn):
    c = nn.get(fn)
    return float(c[0]["cos_sim"]) if c else float("nan")

out, gi = [], 0
with torch.no_grad():
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
             for k, v in b.items()}
        _, pwm_logits, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                             retrieved_pwms=None, retrieved_masks=None,
                             retrieved_sims=None, recog_prior=b.get('recog_prior'))
        pwm_prob = F.softmax(pwm_logits, dim=1).cpu().numpy()
        target, mask = b['target_pwm'].cpu().numpy(), b['pwm_mask'].cpu().numpy()
        for pred, tgt, msk in zip(pwm_prob, target, mask):
            fn = ds.filenames[gi]; gi += 1
            core = trimmed_core(tgt, msk, IC_THRESH)
            if core is None or core.shape[1] < MIN_POS:
                continue
            pred_mask = pred[:, msk.astype(bool)]
            aligned, cols, _ = aligned_cols(pred_mask, core, MAX_SHIFT)
            d = panel(core, aligned, cols)
            if d is None:
                continue
            out.append({"fn": fn, "gene": ds._fn_to_gene.get(fn),
                        "panel_r": float(d["r"]), "homology": homology(fn)})

json.dump(out, open(OUT, "w"), indent=0)
rs = np.array([o["panel_r"] for o in out]); print(f"{CKPT_DIR.split('/')[-2]}: {len(out)} TFs, mean panel-r {np.nanmean(rs):.4f}")
