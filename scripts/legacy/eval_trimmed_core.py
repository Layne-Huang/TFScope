#!/usr/bin/env python
"""Trimmed-core, alignment-fair benchmark (TFScope vs DeepPBS).

Idea (per user): instead of TFScope adopting DeepPBS's structure-anchored
registration, evaluate EVERY model on the *trimmed informative core that we
predict*, granting each model offset + reverse-complement alignment freedom.
This removes the registration advantage DeepPBS gets from the co-crystal DNA
and the flank-handling differences, so the comparison is apples-to-apples on
the biologically informative columns.

For each test TF:
  target_core = informative core of the target PWM (columns with IC >= THRESH)
  for each model:
      pred_valid = predicted PWM over its valid (gated) columns
      r = align_pwm(pred_valid, target_core).score        # offset x RC, per-col Pearson

Reports mean/median trimmed-core r per model on the DeepPBS-covered shared set.

Usage:
  python scripts/eval_trimmed_core.py --models v14 v17 deeppbs
  python scripts/eval_trimmed_core.py --models v14 v17 v18a v18b deeppbs --ic-thresh 0.25
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
MAX_L = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPTS = {
    "v14":                  f"{CKPT_ROOT}/deeppbs_v14_icpcc/ckpt_best.pt",
    "v17":                  f"{CKPT_ROOT}/deeppbs_v17_200ep/ckpt_best.pt",
    "v18a":                 f"{CKPT_ROOT}/deeppbs_v18a_attnrepair/ckpt_best.pt",
    "v18b":                 f"{CKPT_ROOT}/deeppbs_v18b_contact/ckpt_best.pt",
    "v18a_noRAG":           f"{CKPT_ROOT}/deeppbs_v18a_noRAG/ckpt_best.pt",
    "v14_noRAG":            f"{CKPT_ROOT}/deeppbs_v14_noRAG/ckpt_best.pt",
    "fulldata_clval_best":  f"{CKPT_ROOT}/fulldata_clusterval_v18a/ckpt_best.pt",
    "fulldata_clval_last":  f"{CKPT_ROOT}/fulldata_clusterval_v18a/ckpt_epoch125.pt",
    "trim_best":            f"{CKPT_ROOT}/fulldata_clval_v18a_trim/ckpt_best.pt",
    "trim_ep150":           f"{CKPT_ROOT}/fulldata_clval_v18a_trim/ckpt_epoch150.pt",
    "trim_ep200":           f"{CKPT_ROOT}/fulldata_clval_v18a_trim/ckpt_epoch200.pt",
    "c40_best":             f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_best.pt",
    "c40_ep150":            f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_epoch150.pt",
    "c40_ep200":            f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_epoch200.pt",
}


def infer(ckpt):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    P, T, M = [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            _, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                         retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                         retrieved_sims=b.get("retrieved_sims"), recog_prior=b.get("recog_prior"))
            P.append(F.softmax(pw, 1).cpu().numpy()); T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(M), ds.filenames


def ic_bits(pwm):  # pwm (4, L)
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def trimmed_core(target, mask, thresh):
    """Informative core of the target over its valid columns: (4, Lcore) or None."""
    idx = mask.astype(bool)
    if not idx.any():
        return None
    t = target[:, idx]
    ic = ic_bits(t)
    inf = np.where(ic >= thresh)[0]
    if len(inf) == 0:
        return None
    return t[:, inf[0]:inf[-1] + 1]


def deeppbs_preds(fns):
    df = pd.read_parquet(DATA)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))
    dpbs = np.load("results/deeppbs_blind_benchmark/gene_preds.npz", allow_pickle=True)
    ci = {k.upper(): k for k in dpbs.files}
    out = {}
    for fn in fns:
        gk = ci.get(str(fn2gene.get(fn, "")).upper())
        if gk is not None:
            out[fn] = np.clip(np.array(dpbs[gk]).T, 1e-8, 1.0)   # (4, Ldpbs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v14", "v17", "deeppbs"])
    ap.add_argument("--ic-thresh", type=float, default=0.25, help="bits; informative-core trim")
    ap.add_argument("--max-shift", type=int, default=10)
    args = ap.parse_args()

    # run TFScope models present in --models
    tfs = [m for m in args.models if m != "deeppbs"]
    preds, T, M, fns = {}, None, None, None
    for name in tfs:
        ck = CKPTS[name]
        if not os.path.exists(ck):
            print(f"[skip] {name}: checkpoint not found ({ck})"); continue
        print(f"infer {name} ...", flush=True)
        p, t, m, f = infer(ck)
        preds[name] = p
        if T is None: T, M, fns = t, m, f

    if T is None:  # only deeppbs requested → still need target frame
        # build a minimal pass to get targets/fns
        any_ck = next((CKPTS[k] for k in CKPTS if os.path.exists(CKPTS[k])), None)
        _, T, M, fns = infer(any_ck)

    dpp = deeppbs_preds(fns) if "deeppbs" in args.models else {}

    # trimmed-core targets
    cores = [trimmed_core(T[i], M[i], args.ic_thresh) for i in range(len(fns))]
    shared = [i for i in range(len(fns)) if cores[i] is not None and fns[i] in dpp] \
        if "deeppbs" in args.models else [i for i in range(len(fns)) if cores[i] is not None]
    n = len(shared)

    def score_model(get_pred):
        rs = []
        for i in shared:
            core = cores[i]
            pv = get_pred(i)
            if pv is None or pv.shape[1] == 0:
                continue
            _, _, _, s = align_pwm(pv, core, max_shift=args.max_shift, consider_revcomp=True)
            rs.append(s)
        return np.array(rs)

    results = {}
    for name in tfs:
        if name not in preds: continue
        def gp(i, name=name):
            idx = M[i].astype(bool)
            return preds[name][i][:, idx] if idx.any() else None
        results[name] = score_model(gp)
    if "deeppbs" in args.models:
        results["DeepPBS"] = score_model(lambda i: dpp.get(fns[i]))

    print(f"\n=== Trimmed-core (IC>={args.ic_thresh} bits), offset+RC aligned, "
          f"shared {n} TFs ===")
    print(f"{'Model':<10} {'mean r':>9} {'median r':>10} {'n':>5}")
    print("-" * 38)
    for name, rs in results.items():
        print(f"{name:<10} {np.nanmean(rs):>9.4f} {np.nanmedian(rs):>10.4f} {len(rs):>5}")

    os.makedirs("results/trimmed_core", exist_ok=True)
    json.dump({k: v.tolist() for k, v in results.items()},
              open(f"results/trimmed_core/scores_ic{args.ic_thresh}.json", "w"))
    print(f"\nSaved results/trimmed_core/scores_ic{args.ic_thresh}.json")


if __name__ == "__main__":
    main()
