#!/usr/bin/env python
"""Rebuild the Fig 1d baseline ladder under ONE consistent metric, reporting the MEAN
(and median) gate-oracle-r across the 84-TF cluster40 test.

Protocol (identical for every method): each method's predicted informative core is
oracle-aligned (+/-10 shift + RC) to the GT IC-trimmed core `tg`, and the alignment
Pearson r is recorded per TF; we report mean/median across TFs. For TFScope the
predicted core is the gate-active columns (reproducing gate_oracle_r_mean=0.657);
for every other method it is the IC>=0.25 trimmed core of that method's prediction.

Anchors (`tg` per TF) come from a single TFScope run so all methods are scored on the
exact same GT cores. Writes results/baseline_ladder/ladder_mean.json.

ESM2-linear is scored from a predictions file if present (produced by
train_esm2_linear_baseline.py); otherwise it is skipped and flagged.
"""
import os, sys, json, argparse
import numpy as np
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("HF_HOME", "/data1/leihuang/.cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

IC_THRESH, MAX_SHIFT, MIN_POS = 0.25, 10, 4
CKPT_DIR = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42"
SPLIT    = "data/processed/splits/deeppbs_cluster40/split.json"
DATA     = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
TRAIN_PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
NN_INDEX = "data/processed/tf_nn_index_deeppbs_cluster40.json"
DEEPPBS_NPZ = "/data1/leihuang/data/DeepPBS/output_cluster40/cluster40_deeppbs_preds.npz"
ESM2_LINEAR_PREDS = "results/baseline_ladder/esm2_linear_preds.npz"
RNG = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _ic(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)

def trim_ic(pwm, thresh=IC_THRESH):
    """Contiguous IC>=thresh core of a (4,L) PWM."""
    ic = _ic(pwm)
    inf = np.where(ic >= thresh)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]

def score_to_tg(pred_4L, tg):
    """Oracle-aligned Pearson r of a (4,L) prediction against GT core tg (4,L')."""
    core = trim_ic(pred_4L)
    if core.shape[1] == 0:
        return np.nan
    _, _, _, r = align_pwm(core, tg, max_shift=MAX_SHIFT, consider_revcomp=True)
    return float(r)

def decode_pwm(v):
    """Parquet 'pwm' cell (raw float32 bytes) -> (4,L) array, as the dataset does."""
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float32).reshape(4, -1).copy()
    a = np.asarray(v, dtype=np.float32)
    return a if a.shape[0] == 4 else a.T


# ---------------------------------------------------------------- TFScope anchors
def run_tfscope():
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ckpt_best.pt"),
                                 map_location=dev, weights_only=False)["model"], strict=False)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    names = list(ds.filenames)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    anchors, tf_rs, order = {}, [], []
    gi = 0
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                          retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                          retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
            pwm = F.softmax(pl, 1).cpu().numpy(); gate = torch.sigmoid(gl).cpu().numpy()
            tgt = b['target_pwm'].cpu().numpy(); msk = b['pwm_mask'].cpu().numpy()
            for pred, tg_full, ms, ga in zip(pwm, tgt, msk, gate):
                name = names[gi]; gi += 1
                tg = trim_ic(tg_full[:, ms.astype(bool)])
                if tg.shape[1] < MIN_POS:
                    continue
                active = ga > 0.5
                if not active.any():
                    active = ga > ga.max() * 0.5
                pcore = pred[:, active]
                r = np.nan
                if pcore.shape[1] > 0:
                    _, _, _, r = align_pwm(pcore, tg, max_shift=MAX_SHIFT, consider_revcomp=True)
                anchors[name] = tg
                tf_rs.append(r); order.append(name)
    per_tf = dict(zip(order, tf_rs))
    return anchors, order, float(np.nanmean(tf_rs)), float(np.nanmedian(tf_rs)), per_tf


# ---------------------------------------------------------------- baselines
def load_train_pwms():
    df = pd.read_parquet(TRAIN_PARQUET)
    tr = set(json.load(open(SPLIT)).get("train", []))
    return [decode_pwm(row["pwm"]) for _, row in df[df["filename"].isin(tr)].iterrows()]

def eval_random_uniform(anchors, k=50):
    rs = []
    for tg in anchors.values():
        L = tg.shape[1]; per = []
        for _ in range(k):
            rp = RNG.dirichlet(np.ones(4), size=L).T.astype(np.float32)  # (4,L)
            per.append(score_to_tg(rp, tg))
        rs.append(np.nanmean(per))
    return float(np.nanmean(rs)), float(np.nanmedian(rs))

def eval_random_trainpwm(anchors, train_pwms, k=50):
    rs = []
    for tg in anchors.values():
        idx = RNG.integers(0, len(train_pwms), size=k)
        per = [score_to_tg(train_pwms[i], tg) for i in idx]
        rs.append(np.nanmean(per))
    return float(np.nanmean(rs)), float(np.nanmedian(rs))

def eval_nn_pwm(anchors):
    idx = json.load(open(NN_INDEX))
    df = pd.read_parquet(TRAIN_PARQUET)
    by_name = {row["filename"]: decode_pwm(row["pwm"]) for _, row in df.iterrows()}
    fn_gene = {row["filename"]: str(row["gene_symbol"]).upper() for _, row in df.iterrows()}
    rs, miss = [], 0
    for name, tg in anchors.items():
        gene = fn_gene.get(name, name)
        donor = None
        for nb in idx.get(name, []):
            cand = nb["nn_filename"].split("__")[0]
            if fn_gene.get(cand, cand) != gene and cand in by_name:
                donor = by_name[cand]; break
        if donor is None:
            miss += 1; continue
        rs.append(score_to_tg(donor, tg))
    return float(np.nanmean(rs)), float(np.nanmedian(rs)), miss

def eval_deeppbs(anchors):
    z = np.load(DEEPPBS_NPZ, allow_pickle=True)
    n = sum(1 for k in z.keys() if k.startswith("pred_"))
    preds = {}
    for i in range(n):
        nm = str(z[f"name_{i}"]); p = np.asarray(z[f"pred_{i}"], dtype=np.float32)
        if p.shape[1] != 4 and p.shape[0] == 4: p = p.T
        preds[nm] = p.T  # (4,L)
    # DeepPBS names are like '1an4_A_MA0093.3.jaspar.npz'; anchors '1an4_A_USF1.MA0093.3.txt'.
    # Match on the pdb_chain prefix (first two underscore fields), unique per test TF.
    def pdbchain(n): return "_".join(n.split("_")[:2])
    dp_by_pc = {pdbchain(nm): p for nm, p in preds.items()}
    rs, miss, per_tf = [], 0, {}
    for name, tg in anchors.items():
        p = dp_by_pc.get(pdbchain(name))
        if p is None: miss += 1; continue
        r = score_to_tg(p, tg); rs.append(r); per_tf[name] = r
    return float(np.nanmean(rs)), float(np.nanmedian(rs)), miss, per_tf

def eval_esm2_linear(anchors):
    if not os.path.exists(ESM2_LINEAR_PREDS):
        return None
    z = np.load(ESM2_LINEAR_PREDS, allow_pickle=True)
    preds = {str(z["names"][i]): np.asarray(z["preds"][i], dtype=np.float32) for i in range(len(z["names"]))}
    rs = []
    for name, tg in anchors.items():
        p = preds.get(name)
        if p is None: continue
        if p.shape[1] != 4 and p.shape[0] == 4: p = p.T
        rs.append(score_to_tg(p.T if p.shape[1] == 4 else p, tg))
    return float(np.nanmean(rs)), float(np.nanmedian(rs))


if __name__ == "__main__":
    print("[1/6] TFScope anchors ...", flush=True)
    anchors, order, tf_mean, tf_med, tf_per = run_tfscope()
    print(f"      TFScope  mean={tf_mean:.4f}  median={tf_med:.4f}  (n={len(anchors)})", flush=True)

    print("[2/6] DeepPBS ...", flush=True)
    dp_mean, dp_med, dp_miss, dp_per = eval_deeppbs(anchors)
    print(f"      DeepPBS  mean={dp_mean:.4f}  median={dp_med:.4f}  (missing {dp_miss})", flush=True)

    # paired bootstrap TFScope vs DeepPBS on shared TFs (mean-difference)
    shared = [n for n in tf_per if n in dp_per and np.isfinite(tf_per[n]) and np.isfinite(dp_per[n])]
    dt = np.array([tf_per[n] for n in shared]); dd = np.array([dp_per[n] for n in shared])
    diff = dt - dd
    bs = np.array([diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    p_boot = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    sig = {"n_shared": len(shared), "mean_diff": float(diff.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "p_boot": float(p_boot)}
    print(f"      TFScope-DeepPBS mean_diff={sig['mean_diff']:.4f} CI{sig['ci95']} p={sig['p_boot']:.3f}", flush=True)

    print("[3/6] NN-PWM ...", flush=True)
    nn_mean, nn_med, nn_miss = eval_nn_pwm(anchors)
    print(f"      NN-PWM   mean={nn_mean:.4f}  median={nn_med:.4f}  (missing {nn_miss})", flush=True)

    print("[4/6] random-train-PWM ...", flush=True)
    tp = load_train_pwms()
    rt_mean, rt_med = eval_random_trainpwm(anchors, tp)
    print(f"      rand-train  mean={rt_mean:.4f}  median={rt_med:.4f}", flush=True)

    print("[5/6] random-uniform ...", flush=True)
    ru_mean, ru_med = eval_random_uniform(anchors)
    print(f"      rand-unif   mean={ru_mean:.4f}  median={ru_med:.4f}", flush=True)

    print("[6/6] ESM2-linear ...", flush=True)
    el = eval_esm2_linear(anchors)
    if el: print(f"      ESM2-linear mean={el[0]:.4f}  median={el[1]:.4f}", flush=True)
    else:  print("      ESM2-linear preds not found -> skipped", flush=True)

    out = {
        "metric": "mean/median gate-oracle-r; each method's IC core oracle-aligned (+/-10,RC) to GT core; cluster40 n=%d" % len(anchors),
        "ladder_mean": {
            "random_uniform":   {"mean": ru_mean, "median": ru_med},
            "random_train_pwm": {"mean": rt_mean, "median": rt_med},
            "nn_pwm_k1":        {"mean": nn_mean, "median": nn_med},
            "deeppbs_structure":{"mean": dp_mean, "median": dp_med},
            "tfscope_combined": {"mean": tf_mean, "median": tf_med},
        },
    }
    if el:
        out["ladder_mean"]["esm2_linear"] = {"mean": el[0], "median": el[1]}
    out["tfscope_vs_deeppbs"] = sig
    os.makedirs("results/baseline_ladder", exist_ok=True)
    json.dump(out, open("results/baseline_ladder/ladder_mean.json", "w"), indent=1)
    print("\nsaved results/baseline_ladder/ladder_mean.json")
    print(json.dumps(out["ladder_mean"], indent=1))
