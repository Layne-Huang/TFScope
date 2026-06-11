#!/usr/bin/env python
"""Apples-to-apples cluster40 TEST oracle-r at the LAST checkpoint of each run,
in both RAG and noRAG modes. Same protocol as eval_oracle_r_testset.py
(gate>0.5 active cols, target IC>=0.25 core, offset +/-10 + RC, per-col Pearson r).
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

CKROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
SPLIT  = "data/processed/splits/cluster40/split.json"
DATA   = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
dev = "cuda" if torch.cuda.is_available() else "cpu"
IC_THRESH, MAX_SHIFT = 0.25, 10
RUNS = [
    ("baseline (cluster40_v18a_rag)",        "cluster40_v18a_rag",                 "ckpt_epoch150.pt"),
    ("contrastive (aux)",                    "cluster40_v18a_contrast",            "ckpt_epoch175.pt"),
    ("pretrain->finetune DPAC",              "cluster40_v18a_stageB_dpac",         "ckpt_epoch175.pt"),
    ("pretrain->finetune DPAC+HT-SELEX",     "cluster40_v18a_stageB_dpac_htselex", "ckpt_epoch050.pt"),
]

def _ic(p): p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)
def _trim(pwm):
    ic = _ic(pwm); inf = np.where(ic >= IC_THRESH)[0]
    return pwm[:, inf[0]:inf[-1] + 1] if len(inf) else pwm

def eval_ckpt(ckpt_dir, ckpt_file):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=dev,
                                 weights_only=False)["model"], strict=False)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    out = {}
    for mode in ("RAG", "noRAG"):
        rs = []
        with torch.no_grad():
            for b in ld:
                b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
                     for k, v in b.items()}
                kw = dict(retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                          retrieved_sims=b.get('retrieved_sims'))
                if mode == "noRAG":
                    kw = dict(retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None)
                gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                              recog_prior=b.get('recog_prior'), **kw)
                pwm_prob = F.softmax(pl, 1).cpu().numpy(); gate = torch.sigmoid(gl).cpu().numpy()
                tgt = b['target_pwm'].cpu().numpy(); msk = b['pwm_mask'].cpu().numpy()
                for pred, t, mk, g in zip(pwm_prob, tgt, msk, gate):
                    act = g > 0.5
                    if not act.any(): act = g > g.max() * 0.5
                    pc = pred[:, act]
                    tv = t[:, mk.astype(bool)]
                    if pc.shape[1] == 0 or tv.shape[1] == 0: continue
                    tc = _trim(tv)
                    if tc.shape[1] == 0: continue
                    _, _, _, r = align_pwm(pc, tc, max_shift=MAX_SHIFT, consider_revcomp=True)
                    rs.append(r)
        rs = np.array(rs); out[mode] = (rs.mean(), np.median(rs), len(rs))
    return out, cfg.use_retrieval

print(f"{'run':40s} {'mode':6s} {'mean':>7s} {'median':>7s} {'n':>5s}")
print("-" * 70)
rows = []
for name, d, ck in RUNS:
    res, ur = eval_ckpt(os.path.join(CKROOT, d), ck)
    for mode in ("RAG", "noRAG"):
        mean, med, n = res[mode]
        print(f"{name:40s} {mode:6s} {mean:7.4f} {med:7.4f} {n:5d}", flush=True)
        rows.append(dict(run=name, ckpt=ck, mode=mode, oracle_r_mean=round(float(mean), 4),
                         oracle_r_median=round(float(med), 4), n=n))
    print("-" * 70, flush=True)
os.makedirs("results/last_ckpt_eval", exist_ok=True)
json.dump(rows, open("results/last_ckpt_eval/oracle_r_last_ckpts.json", "w"), indent=2)
print("saved -> results/last_ckpt_eval/oracle_r_last_ckpts.json")
