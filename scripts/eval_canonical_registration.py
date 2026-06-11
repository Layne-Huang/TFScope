#!/usr/bin/env python
"""Canonical-registration fair benchmark (deployable, no oracle alignment).

Unlike eval_trimmed_core.py (which grants every model oracle offset+RC alignment
to the target), this applies the SAME deterministic v16 canonical registration
(trim low-IC flanks + canonical-strand rule) to BOTH the prediction and the
target, left-anchors each at position 0, then scores FIXED per-column Pearson.

This is the honest deployable metric: strand + offset are resolved by a fixed
rule (not by peeking at the target), and DeepPBS is judged in exactly the frame
our model is trained/evaluated in. Applying it to DeepPBS too = fair comparison.

Usage:
  python scripts/eval_canonical_registration.py --models v14 v17 v18a deeppbs
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from canonicalize_pwms import canonicalize   # v16 trim + canonical strand
from torch.utils.data import DataLoader

SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"

CANON      = "data/processed/tf_pwm_deeppbs_only_canon.parquet"
CANON_FULL = "data/processed/tf_pwm_aug_dbd_canon.parquet"
CKPTS = {
    "v14":         f"{CKPT_ROOT}/deeppbs_v14_icpcc/ckpt_best.pt",
    "v16":         f"{CKPT_ROOT}/deeppbs_v16/ckpt_best.pt",
    "v17":         f"{CKPT_ROOT}/deeppbs_v17_200ep/ckpt_best.pt",
    "v18a":        f"{CKPT_ROOT}/deeppbs_v18a_attnrepair/ckpt_best.pt",
    "v18b":        f"{CKPT_ROOT}/deeppbs_v18b_contact/ckpt_best.pt",
    "v18a_noRAG":  f"{CKPT_ROOT}/deeppbs_v18a_noRAG/ckpt_best.pt",
    "fulldata":    f"{CKPT_ROOT}/deeppbs_fulldata_canon/ckpt_best.pt",
    "5fold_f0":    f"{CKPT_ROOT}/deeppbs_5fold_norag/fold_0/ckpt_best.pt",
    "5fold_f1":    f"{CKPT_ROOT}/deeppbs_5fold_norag/fold_1/ckpt_best.pt",
    "5fold_f2":    f"{CKPT_ROOT}/deeppbs_5fold_norag/fold_2/ckpt_best.pt",
    "5fold_f3":    f"{CKPT_ROOT}/deeppbs_5fold_norag/fold_3/ckpt_best.pt",
    "5fold_f4":    f"{CKPT_ROOT}/deeppbs_5fold_norag/fold_4/ckpt_best.pt",
}
# models trained on canonical targets → use the matching canonical parquet for test inference
DATA_OF = {
    "v16":        CANON,
    "fulldata":   CANON_FULL,
    "5fold_f0":   CANON,
    "5fold_f1":   CANON,
    "5fold_f2":   CANON,
    "5fold_f3":   CANON,
    "5fold_f4":   CANON,
}


def infer(ckpt, data=DATA):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    ds = TFDataset(cfg, data, SPLIT, split="test", max_seq_len=1024)
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
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)
    # filename-keyed (robust to parquet row-order differences across models)
    pred = {fn: P[i][:, M[i].astype(bool)] for i, fn in enumerate(ds.filenames)}
    tgt  = {fn: (T[i], M[i]) for i, fn in enumerate(ds.filenames)}
    return pred, tgt


def deeppbs_preds(fns):
    df = pd.read_parquet(DATA)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))
    dpbs = np.load("results/deeppbs_blind_benchmark/gene_preds.npz", allow_pickle=True)
    ci = {k.upper(): k for k in dpbs.files}
    out = {}
    for fn in fns:
        gk = ci.get(str(fn2gene.get(fn, "")).upper())
        if gk is not None:
            out[fn] = np.clip(np.array(dpbs[gk]).T, 1e-8, 1.0)
    return out


def canon_fixed_r(target_core, pred):
    """Canonicalize prediction, left-anchor both, FIXED per-column Pearson over
    the target-core columns (pred padded with uniform beyond its length)."""
    cp = canonicalize(np.clip(pred, 1e-8, 1.0).astype(np.float32))
    cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
    Lt = target_core.shape[1]
    pp = np.full((4, Lt), 0.25, np.float32)
    L = min(cp.shape[1], Lt); pp[:, :L] = cp[:, :L]
    rs = []
    for j in range(Lt):
        a, b = target_core[:, j], pp[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v14", "v17", "deeppbs"])
    args = ap.parse_args()

    tfs = [m for m in args.models if m != "deeppbs"]
    preds, tgt_ref = {}, None
    for name in tfs:
        ck = CKPTS[name]
        if not os.path.exists(ck):
            print(f"[skip] {name}: not found"); continue
        print(f"infer {name} (data={os.path.basename(DATA_OF.get(name, DATA))}) ...", flush=True)
        p, tg = infer(ck, DATA_OF.get(name, DATA))
        preds[name] = p
        if tgt_ref is None or name != "v16":   # prefer a raw-parquet target frame
            tgt_ref = tg
    if tgt_ref is None:
        any_ck = next((k for k in CKPTS if os.path.exists(CKPTS[k])), None)
        _, tgt_ref = infer(CKPTS[any_ck], DATA_OF.get(any_ck, DATA))

    fns = list(tgt_ref.keys())
    dpp = deeppbs_preds(fns) if "deeppbs" in args.models else {}

    # canonical target cores (the common frame everyone is scored against)
    tcore = {}
    for fn in fns:
        T_i, M_i = tgt_ref[fn]; idx = M_i.astype(bool)
        if not idx.any(): continue
        c = canonicalize(np.clip(T_i[:, idx], 1e-8, 1.0).astype(np.float32))
        tcore[fn] = c / c.sum(0, keepdims=True).clip(1e-8)

    shared = [fn for fn in tcore if (fn in dpp)] if "deeppbs" in args.models else list(tcore)
    n = len(shared)

    results = {}
    for name in tfs:
        if name not in preds: continue
        rs = [canon_fixed_r(tcore[fn], preds[name][fn]) for fn in shared if fn in preds[name]]
        results[name] = np.array(rs)
    if "deeppbs" in args.models:
        results["DeepPBS"] = np.array([canon_fixed_r(tcore[fn], dpp[fn]) for fn in shared])

    print(f"\n=== Canonical-registration FIXED scoring (deployable, no oracle "
          f"alignment), shared {n} TFs ===")
    print(f"{'Model':<10} {'mean r':>9} {'median r':>10} {'n':>5}")
    print("-" * 38)
    for name, rs in results.items():
        print(f"{name:<10} {np.nanmean(rs):>9.4f} {np.nanmedian(rs):>10.4f} {len(rs):>5}")

    os.makedirs("results/canonical_reg", exist_ok=True)
    json.dump({k: v.tolist() for k, v in results.items()},
              open("results/canonical_reg/scores.json", "w"))
    print("\nSaved results/canonical_reg/scores.json")


if __name__ == "__main__":
    main()
