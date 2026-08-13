"""Benchmark v19_combined_rag_contact (learned-10 + RAG + contact) on cluster40 test.
Learned-10 family_id comes straight from the parquet; retrieval uses the deeppbs
cluster40 NN index with the deeppbs table as donor pool. Scored with the same
oracle-aligned panel-r as the baseline ladder.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length
from eval_full_metrics import trimmed_core, aligned_cols, panel

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_rag_contact/rag_seed42/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
IDX = "data/processed/tf_nn_index_deeppbs_cluster40.json"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.retrieval_index_path = IDX
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
preds, tgts = {}, {}
gi = 0
with torch.no_grad():
    for b in ld:
        bs = b["sequence_tokens"].shape[0]
        bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        _, pw, _ = m(bt["sequence_tokens"], bt["dbd_mask"], bt["family_id"],
                     retrieved_pwms=bt.get("retrieved_pwms"), retrieved_masks=bt.get("retrieved_masks"),
                     retrieved_sims=bt.get("retrieved_sims"), recog_prior=bt.get("recog_prior"))
        P = F.softmax(pw, 1).cpu().numpy(); T = b["target_pwm"].numpy(); M = b["pwm_mask"].numpy()
        for j in range(bs):
            fn = ds.filenames[gi + j]
            preds[fn] = P[j][:, M[j].astype(bool)]; tgts[fn] = (T[j], M[j])
        gi += bs

rs, top1, mae = [], [], []
for fn in ds.filenames:
    core = trimmed_core(*tgts[fn]); pv = preds[fn]
    if core is None or pv.shape[1] == 0: continue
    aligned, cols, _ = aligned_cols(pv, core)
    d = panel(core, aligned, cols)
    if d: rs.append(d["r"]); top1.append(d["top1"]); mae.append(d["mae"])
res = dict(model="v19_combined_rag_contact", n=len(rs),
           panel_r=round(float(np.nanmean(rs)), 3), top1=round(float(np.nanmean(top1)), 3),
           ref={"tfscope_combined_noRAG": 0.643, "deeppbs": 0.634})
json.dump(res, open("results/v19_deeppbs/rag_contact_cluster40.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print("saved results/v19_deeppbs/rag_contact_cluster40.json")
