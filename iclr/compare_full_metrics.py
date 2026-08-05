#!/usr/bin/env python
"""Full metric-suite comparison of the retrained (PDB-clean) DeepPBS ensemble vs
the v24 5-seed ensemble on the 20 struct-having test genes. Both predictions are
oracle-registered (offset+RC) to the GT motif core, then scored on the aligned
overlap columns with a battery of metrics.

Metrics (per aligned column, gene-balanced unless noted):
  pearson_r   higher  per-column Pearson (== content_r)
  mae         lower   mean |pred-target| over base x column  (DeepPBS's own loss)
  rmse        lower   root mean squared error
  cosine      higher  per-column cosine similarity
  jsd_bits    lower   Jensen-Shannon divergence (bits) per column
  topbase_acc higher  fraction of columns whose argmax base matches (consensus letter)
  macroF1     higher  macro-F1 of consensus-letter (4-class) over pooled columns
  auroc       higher  base-enrichment ROC: label=target>0.25, score=pred (pooled)
  ic_mae      lower   |IC(pred)-IC(target)| bits per column

Run in tfscope env:  PYTHONPATH=src python -m iclr.compare_full_metrics --device cuda:0
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.unified_eval import _aligned_cols, trimmed_core
from iclr.score_v24_ensemble import predict_ensemble
from iclr.baselines import _decode_pwm


def _col_metrics(P, T):
    """P,T = (4,N) aligned & column-normalized. Return dict of scalar metrics
    plus pooled arrays for auroc/F1."""
    P = np.clip(P, 1e-8, 1.0); P = P / P.sum(0, keepdims=True)
    T = np.clip(T, 1e-8, 1.0); T = T / T.sum(0, keepdims=True)
    N = P.shape[1]
    # per-column
    rs, cos, jsd = [], [], []
    for j in range(N):
        p, t = P[:, j], T[:, j]
        rs.append(0.0 if p.std() == 0 or t.std() == 0 else np.corrcoef(p, t)[0, 1])
        cos.append(float(p @ t / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-12)))
        m = 0.5 * (p + t)
        kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
        jsd.append(0.5 * kl(p, m) + 0.5 * kl(t, m))
    mae = float(np.mean(np.abs(P - T)))
    rmse = float(np.sqrt(np.mean((P - T) ** 2)))
    topbase = float(np.mean(P.argmax(0) == T.argmax(0)))
    ic = lambda M: 2.0 + np.sum(M * np.log2(M), axis=0)          # (N,) bits
    ic_mae = float(np.mean(np.abs(ic(P) - ic(T))))
    return dict(pearson_r=float(np.nanmean(rs)), mae=mae, rmse=rmse,
                cosine=float(np.mean(cos)), jsd_bits=float(np.mean(jsd)),
                topbase_acc=topbase, ic_mae=ic_mae,
                _pred_letter=list(P.argmax(0)), _true_letter=list(T.argmax(0)),
                _auc_label=list((T > 0.25).astype(int).ravel()), _auc_score=list(P.ravel()))


def _suite(preds_by_key, gt_by_fn, fn2gene, deeppbs=False, deeppbs_by_gene=None):
    per_gene = {}
    pool_label, pool_score, pool_pl, pool_tl = [], [], [], []
    for fn, gt in gt_by_fn.items():
        g = fn2gene[fn]
        core = trimmed_core(gt)
        if core is None:
            continue
        pred = deeppbs_by_gene[g] if deeppbs else preds_by_key.get(fn)
        if pred is None:
            continue
        # NOTE: DeepPBS is scored UNTRIMMED (its actual raw PWM over the co-crystal
        # DNA). We deliberately do NOT IC-trim it: v24's motif localization is a
        # LEARNED, scored gate, so hand-trimming DeepPBS to its IC core would grant
        # it a free localization v24 had to earn. Untrimmed = the fair comparison.
        # (align_pwm still gives both models the same oracle offset+RC registration.)
        aligned, cols = _aligned_cols(pred, core)
        if len(cols) < 2:
            continue
        m = _col_metrics(aligned[:, cols], core[:, cols])
        per_gene.setdefault(g, []).append(m)
        pool_label += m.pop("_auc_label"); pool_score += m.pop("_auc_score")
        pool_pl += m.pop("_pred_letter"); pool_tl += m.pop("_true_letter")
    # gene-balanced scalar means
    keys = ["pearson_r", "mae", "rmse", "cosine", "jsd_bits", "topbase_acc", "ic_mae"]
    genes = sorted(per_gene)
    gm = {k: float(np.mean([np.mean([r[k] for r in per_gene[g]]) for g in genes])) for k in keys}
    # pooled AUROC + macro-F1
    from sklearn.metrics import roc_auc_score, f1_score
    gm["auroc"] = float(roc_auc_score(pool_label, pool_score)) if len(set(pool_label)) > 1 else float("nan")
    gm["macroF1"] = float(f1_score(pool_tl, pool_pl, average="macro", labels=[0, 1, 2, 3]))
    gm["n_genes"] = len(genes)
    return gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deeppbs-pkl", default="/data1/leihuang/TFScope_store/deeppbs_pdbclean_preds.pkl")
    ap.add_argument("--out", default="results/iclr_phase1_apples_to_apples/full_metric_suite.json")
    a = ap.parse_args()
    tf = Path(".").resolve()

    test20 = set("CLOCK ELF3 ELK4 ERG ETS1 ETS2 ETV1 ETV5 ETV6 FEV FLI1 FOXA2 FOXA3 "
                 "FOXM1 FOXO1 FOXO3 FOXO4 FOXP2 GABPA P53".split())
    split = json.loads((tf / "data/processed/splits/train_v22/split.json").read_text())
    df = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet"); df["filename"] = df.filename.astype(str)
    te = df[df.filename.isin(set(split["test"]))].copy(); te["G"] = te.gene_symbol.astype(str).str.upper()
    te20 = te[te.G.isin(test20)]
    gt_by_fn = {r.filename: _decode_pwm(r.pwm) for r in te20.itertuples()}
    fn2gene = {r.filename: r.G for r in te20.itertuples()}

    # v24 ensemble preds (keyed by filename), restricted later to the 20 genes
    CK = ["/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"]
    for S in (1, 7, 13, 23):
        CK.append(f"checkpoints/iclr_phase1/v24_ens/seed{S}/ckpt_best.pt")
    v24_preds, _ = predict_ensemble(CK, "data/processed/tf_pwm_training_v23.parquet",
                                    "data/processed/splits/train_v22/split.json", a.device)
    deeppbs_by_gene = pickle.load(open(a.deeppbs_pkl, "rb"))

    v24 = _suite(v24_preds, gt_by_fn, fn2gene)
    dpb = _suite(None, gt_by_fn, fn2gene, deeppbs=True, deeppbs_by_gene=deeppbs_by_gene)

    order = ["pearson_r", "cosine", "topbase_acc", "auroc", "macroF1", "mae", "rmse", "jsd_bits", "ic_mae"]
    arrows = {"pearson_r": "↑", "cosine": "↑", "topbase_acc": "↑", "auroc": "↑", "macroF1": "↑",
              "mae": "↓", "rmse": "↓", "jsd_bits": "↓", "ic_mae": "↓"}
    print(f"\n20 struct-having test genes  (v24 n_genes={v24['n_genes']}, DeepPBS n_genes={dpb['n_genes']})")
    print(f"{'metric':14s}{'dir':>4s}{'v24_ens5':>12s}{'DeepPBS':>12s}{'winner':>10s}")
    for k in order:
        better = (v24[k] > dpb[k]) if arrows[k] == "↑" else (v24[k] < dpb[k])
        print(f"{k:14s}{arrows[k]:>4s}{v24[k]:>12.4f}{dpb[k]:>12.4f}{('v24' if better else 'DeepPBS'):>10s}")
    json.dump({"v24_ens5": v24, "DeepPBS_5model": dpb}, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
