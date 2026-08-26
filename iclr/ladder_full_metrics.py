#!/usr/bin/env python
"""Full-metric baseline ladder topped by the 5-seed v24 ENSEMBLE.

Replaces `results/baseline_ladder/ladder_mean.json`, which (a) topped out at the older
`tfscope_combined` model rather than the v24 ensemble, (b) reported a single number
(mean gate-oracle-r), and (c) ran on the cluster40-84 benchmark, where 33 of 41 test
genes are in v23 train. This ladder runs on the CLEAN surfaces of the canonical split
and reports the whole metric panel for every rung.

Rungs, floor to ceiling -- each one adds exactly one capability, so the gap between two
adjacent rungs is the value of that capability:

  random_uniform    a flat 0.25 PWM                      -- the true floor
  random_train_pwm  a real training PWM chosen at random -- "PWMs look like PWMs"
  B0_global         mean of all training PWMs            -- the average motif
  B0_family         mean of the TF's own family          -- family identity alone
  B1_nearest_pwm    PWM of the closest training TF       -- sequence homology lookup
  deeppbs           structure-based competitor           -- co-crystal structure input
  v24_seed42        one trained model                    -- learned sequence -> PWM
  v24_ens5          the 5-seed ensemble                  -- + seed averaging

Metrics. Panel A first registers each prediction to the IC-trimmed GT core with the
same oracle shift + reverse-complement search, then scores the overlapping columns:

  pearson_r    up    per-column Pearson (this is `content_r` elsewhere in the harness)
  cosine       up    per-column cosine similarity
  topbase_acc  up    fraction of columns whose argmax base matches
  auroc        up    base enrichment: label = target > 0.25, score = predicted prob
  macroF1      up    macro-F1 of the 4-class consensus letter
  mae          down  mean |pred - target| per base x column
  rmse         down  root mean squared error
  jsd_bits     down  Jensen-Shannon divergence per column
  kl_bits      down  KL(target || pred) per column -- asymmetric, punishes overconfidence
  ic_mae       down  |IC(pred) - IC(target)| bits per column

Panel B is end-to-end and reports length behaviour separately rather than folding it
into one number: covR (= overlap_r x coverage), coverage, gate_len_mae, gate_len_bias.
Baselines get a PRE-REGISTERED per-family training length (never reads the test target);
trained models use their own predicted gate span.

Aggregation is gene-balanced (mean over per-gene means); CIs are gene-level bootstraps.

Note on the floor: the old ladder's "random r ~ 0.42" was an artefact of scoring with
align_pwm's own alignment score, which rewards a partial overlap. Here a degenerate
(zero-variance) predicted column scores r = 0, so `random_uniform` sits near 0 on
pearson_r while still posting a meaningful MAE -- which is the point of a multi-metric
ladder.

DeepPBS needs a co-crystal structure, so it exists for only 20 of the clean surface's
95 genes. Rather than drop it or quietly show it beside rungs measured on a different
gene set, it gets a MATCHED surface: `deeppbs20` is the co-crystal subset, and every
other rung is rescored there, so all eight methods are compared on identical genes.
Read DeepPBS numbers off `deeppbs20` only -- never against the 95-gene surfaces.

Still not included: esm2_linear, whose cached embeddings were built for cluster40 and
carry a documented train-leakage caveat.

  PYTHONPATH=src python -m iclr.ladder_full_metrics --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm, _nearest_pwm                      # noqa: E402
from iclr.compare_full_metrics import _col_metrics                        # noqa: E402
from iclr.score_v24_ensemble import predict_ensemble                      # noqa: E402
from iclr.unified_eval import (_aligned_cols, baseline_pred_len, panel_B,  # noqa: E402
                               train_length_policy, trimmed_core)
from reclassify_tf_families import classify                               # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
SEED42 = ("/data1/leihuang/project/TFScope/checkpoints/v24_contact/"
          "contact_v24_seed42/ckpt_best.pt")
ENS = [SEED42] + [f"/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/"
                  f"seed{s}/ckpt_best.pt" for s in (1, 7, 13, 23)]
OUT = "results/baseline_ladder/ladder_v24ens_full.json"
# The PDB-disjoint DeepPBS retrain (iclr_retrain_pdb / iclr_folds_pdbdisjoint) -- the
# stricter of the two retrains and the one AUDIT_FINDINGS designates as the fair
# primary comparison. Scored UNTRIMMED at its own length: v24 has to earn motif
# localization with a scored gate, so hand-trimming DeepPBS to its IC core would give
# it a localization step v24 never gets.
DEEPPBS_PKL = "/data1/leihuang/TFScope_store/deeppbs_pdbclean_preds.pkl"

HIGHER_IS_BETTER = {"pearson_r": True, "cosine": True, "topbase_acc": True,
                    "auroc": True, "macroF1": True, "mae": False, "rmse": False,
                    "jsd_bits": False, "kl_bits": False, "ic_mae": False}
ORDER = ["pearson_r", "cosine", "topbase_acc", "auroc", "macroF1",
         "mae", "rmse", "jsd_bits", "kl_bits", "ic_mae"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_taxonomy(df):
    df = df.copy()
    df["g"] = df.gene_symbol.fillna("").astype(str).str.upper()
    seed = {r.g: r.family_name for r in
            df[df.family_name.isin(["C2H2_short", "C2H2_medium", "C2H2_long"])].itertuples()}
    fam = df.g.map(lambda x: classify(x, seed))
    df["fam_lofo"] = fam.map(lambda f: "C2H2" if str(f).startswith("C2H2") else f)
    return df


def _group_mean(pwms):
    """Left-anchored mean PWM over a group, at the group's median motif length."""
    W = max(1, int(np.median([p.shape[1] for p in pwms])))
    acc = np.full((4, W), 0.0); n = np.zeros(W)
    for p in pwms:
        L = min(p.shape[1], W)
        acc[:, :L] += p[:, :L] / p[:, :L].sum(0, keepdims=True).clip(1e-8)
        n[:L] += 1
    avg = np.where(n > 0, acc / np.maximum(n, 1), 0.25)
    return (avg / avg.sum(0, keepdims=True).clip(1e-8)).astype(np.float32)


def _kl_bits(P, T):
    """KL(target || pred) in bits, averaged over columns."""
    P = np.clip(P, 1e-8, 1.0); P = P / P.sum(0, keepdims=True)
    T = np.clip(T, 1e-8, 1.0); T = T / T.sum(0, keepdims=True)
    return float(np.mean(np.sum(T * np.log2(T / P), axis=0)))


def score_method(preds, lens, rows, gt, gene_of, per_gene_out=None):
    """Panel A metric panel + Panel B length behaviour, gene-balanced.

    `per_gene_out` (a dict) receives {gene: {metric: gene-mean}}, so that a paired
    comparison between two rungs can be bootstrapped on the SAME genes downstream.
    Without it only aggregates survive and any rung-vs-rung difference would have to
    be quoted without a paired CI.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    per_gene, pool = {}, {"lab": [], "sc": [], "pl": [], "tl": []}
    for fn in rows:
        pred = preds.get(fn)
        if pred is None:
            continue
        core = trimmed_core(gt[fn])
        if core is None:
            continue
        aligned, cols = _aligned_cols(pred, core)
        if len(cols) < 2:
            continue
        P, T = aligned[:, cols], core[:, cols]
        m = _col_metrics(P, T)
        m["kl_bits"] = _kl_bits(P, T)
        lab = m.pop("_auc_label"); sc = m.pop("_auc_score")
        pl = m.pop("_pred_letter"); tl = m.pop("_true_letter")
        pool["lab"] += lab; pool["sc"] += sc; pool["pl"] += pl; pool["tl"] += tl
        try:
            m["auroc"] = float(roc_auc_score(lab, sc)) if len(set(lab)) > 1 else np.nan
        except ValueError:
            m["auroc"] = np.nan
        m["macroF1"] = float(f1_score(tl, pl, average="macro", labels=[0, 1, 2, 3],
                                      zero_division=0))
        b = panel_B(pred, core, lens.get(fn))
        m.update(covR=b["covR"], coverage=b["coverage"],
                 gate_len_mae=float(b["len_mae"]), gate_len_bias=float(b["len_bias"]))
        per_gene.setdefault(gene_of[fn], []).append(m)

    genes = sorted(per_gene)
    if not genes:
        return None
    keys = ORDER + ["covR", "coverage", "gate_len_mae", "gate_len_bias"]
    out = {}
    for k in keys:
        gm = np.array([np.nanmean([r[k] for r in per_gene[g]]) for g in genes], float)
        if per_gene_out is not None:
            for g, v in zip(genes, gm):
                per_gene_out.setdefault(g, {})[k] = float(v)
        rng = np.random.RandomState(0)
        boot = ([np.nanmean(rng.choice(gm, gm.size, replace=True)) for _ in range(2000)]
                if gm.size > 1 else [gm[0]])
        out[k] = round(float(np.nanmean(gm)), 4)
        out[k + "_ci95"] = [round(float(np.percentile(boot, 2.5)), 4),
                            round(float(np.percentile(boot, 97.5)), 4)]
    out["auroc_pooled"] = (round(float(roc_auc_score(pool["lab"], pool["sc"])), 4)
                           if len(set(pool["lab"])) > 1 else None)
    out["macroF1_pooled"] = round(float(f1_score(pool["tl"], pool["pl"], average="macro",
                                                 labels=[0, 1, 2, 3], zero_division=0)), 4)
    out["n_genes"] = len(genes)
    out["n_rows"] = int(sum(len(v) for v in per_gene.values()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--surfaces", nargs="+",
                    default=["test", "excluded", "clean_combined", "deeppbs20"])
    ap.add_argument("--n-random-draws", type=int, default=20,
                    help="draws averaged for the random_train_pwm rung")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    df = add_taxonomy(pd.read_parquet(DATA)); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm)
    policy = train_length_policy(tr)

    surf_fns = {"test": [f for f in sp["test"] if f in set(df.filename)],
                "excluded": [f for f in sp["excluded"] if f in set(df.filename)]}
    surf_fns["clean_combined"] = surf_fns["test"] + surf_fns["excluded"]

    gt = {r.filename: _decode_pwm(r.pwm) for r in df.itertuples()}
    gene_of = dict(zip(df.filename, df.g))
    fam_of = dict(zip(df.filename, df.fam_lofo))
    fid_of = dict(zip(df.filename, df.family_id))
    seq_of = dict(zip(df.filename, df.sequence))

    # ── DeepPBS: only the co-crystal genes, so it gets its own MATCHED surface ─
    dpbs = {}
    if os.path.exists(DEEPPBS_PKL):
        import pickle
        dpbs = {str(g).upper(): np.asarray(v, np.float32)
                for g, v in pickle.load(open(DEEPPBS_PKL, "rb")).items()}
        cov = [f for f in surf_fns["clean_combined"] if gene_of[f] in dpbs]
        surf_fns["deeppbs20"] = cov
        log(f"DeepPBS: {len(dpbs)} genes with structures -> matched surface "
            f"'deeppbs20' = {len(cov)} rows / {len({gene_of[f] for f in cov})} genes. "
            "Every rung is rescored there so the comparison is on identical genes.")
    else:
        log(f"DeepPBS predictions not found at {DEEPPBS_PKL}; skipping that rung")

    # ── training-free priors, built from TRAIN only ───────────────────────────
    global_prior = _group_mean(list(tr.pwm_arr))
    fam_prior = {f: _group_mean(list(g.pwm_arr)) for f, g in tr.groupby("fam_lofo")}
    log(f"priors: global (n={len(tr)}) + {len(fam_prior)} family priors "
        f"(corrected taxonomy); zero-shot families fall back to global")

    # nearest-neighbour lookup over UNIQUE training sequences (4794 rows -> far fewer)
    tr_uniq = tr.drop_duplicates("sequence")[["sequence", "pwm_arr"]].reset_index(drop=True)
    log(f"nearest-PWM donor pool: {len(tr_uniq)} unique training sequences")

    # ── model predictions (once, over every surface row) ──────────────────────
    all_fns = sorted(set(surf_fns["clean_combined"]))
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"test": all_fns}, tmp); tmp.close()
    model_preds = {}
    try:
        for tag, ck in [("v24_seed42", [SEED42]), ("v24_ens5", ENS)]:
            missing = [c for c in ck if not os.path.exists(c)]
            if missing:
                log(f"SKIP {tag}: missing {missing}"); continue
            t0 = time.time()
            model_preds[tag] = predict_ensemble(ck, DATA, tmp.name, a.device)
            log(f"  {tag}: {len(model_preds[tag][0])} rows predicted ({time.time()-t0:.0f}s)")
    finally:
        os.unlink(tmp.name)

    rng = np.random.RandomState(0)
    tr_pwms = list(tr.pwm_arr)
    nn_cache = {}

    results, pergene_rows = {}, []

    def cap(surf, tag, pg):
        for g, mm in pg.items():
            pergene_rows.append({"surface": surf, "rung": tag, "gene": g, **mm})

    for surf in a.surfaces:
        rows = surf_fns[surf]
        log(f"=== surface {surf}: {len(rows)} rows / "
            f"{len({gene_of[f] for f in rows})} genes ===")
        blen = {fn: baseline_pred_len(fid_of[fn], policy) for fn in rows}
        ladder = {}

        # 1. random uniform
        pg = {}
        ladder["random_uniform"] = score_method(
            {fn: np.full((4, max(1, blen[fn])), 0.25, np.float32) for fn in rows},
            blen, rows, gt, gene_of, per_gene_out=pg)
        cap(surf, "random_uniform", pg)

        # 2. random real training PWM, averaged over draws
        draws = []
        for d in range(a.n_random_draws):
            preds = {fn: tr_pwms[rng.randint(len(tr_pwms))] for fn in rows}
            pg = {} if d == 0 else None
            s = score_method(preds, blen, rows, gt, gene_of, per_gene_out=pg)
            if d == 0 and pg:
                cap(surf, "random_train_pwm", pg)   # draw 0 only; the rung is a mean of draws
            if s:
                draws.append(s)
        fixed = {"n_genes", "n_rows"}
        ladder["random_train_pwm"] = {
            k: (draws[0][k] if k in fixed or k.endswith("_ci95")
                or not isinstance(draws[0][k], (int, float))
                else round(float(np.mean([d[k] for d in draws])), 4))
            for k in draws[0]}
        ladder["random_train_pwm"]["n_draws"] = len(draws)
        ladder["random_train_pwm"]["ci_note"] = ("CIs are from the first draw only; the "
                                                 "point estimates average all draws")

        # 3-4. train-only priors
        pg = {}
        ladder["B0_global"] = score_method({fn: global_prior for fn in rows},
                                           blen, rows, gt, gene_of, per_gene_out=pg)
        cap(surf, "B0_global", pg)
        pg = {}
        ladder["B0_family"] = score_method(
            {fn: fam_prior.get(fam_of[fn], global_prior) for fn in rows},
            blen, rows, gt, gene_of, per_gene_out=pg)
        cap(surf, "B0_family", pg)
        zero_shot = sorted({fam_of[fn] for fn in rows if fam_of[fn] not in fam_prior})
        ladder["B0_family"]["zero_shot_fallback_families"] = zero_shot

        # 5. nearest training PWM by sequence identity
        t0 = time.time()
        preds = {}
        for i, fn in enumerate(rows):
            s = seq_of[fn]
            if s not in nn_cache:
                nn_cache[s] = _nearest_pwm(s, tr_uniq)
            preds[fn] = nn_cache[s]
            if (i + 1) % 100 == 0:
                log(f"    nearest-PWM {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
        pg = {}
        ladder["B1_nearest_pwm"] = score_method(preds, blen, rows, gt, gene_of,
                                                per_gene_out=pg)
        cap(surf, "B1_nearest_pwm", pg)
        log(f"    nearest-PWM done ({time.time()-t0:.0f}s)")

        # 6-7. trained models, using their own gate span for Panel B
        for tag, (p, l) in model_preds.items():
            pg = {}
            ladder[tag] = score_method(p, l, rows, gt, gene_of, per_gene_out=pg)
            cap(surf, tag, pg)

        # 8. DeepPBS -- only where every row's gene has a structure, never partially
        if dpbs and all(gene_of[fn] in dpbs for fn in rows):
            preds = {fn: dpbs[gene_of[fn]] for fn in rows}
            pg = {}
            ladder["deeppbs"] = score_method(
                preds, {fn: preds[fn].shape[1] for fn in rows}, rows, gt, gene_of,
                per_gene_out=pg)
            cap(surf, "deeppbs", pg)

        results[surf] = ladder

    payload = {
        "note": "baseline ladder topped by the v24 5-seed ensemble; full metric panel",
        "surfaces": {s: {"n_rows": len(surf_fns[s]),
                         "n_genes": len({gene_of[f] for f in surf_fns[s]}),
                         "contamination": "clean (never trained, never selected on)",
                         "note": ("co-crystal subset of clean_combined; the only surface "
                                  "where DeepPBS can be scored, so every rung is "
                                  "rescored here on identical genes")
                         if s == "deeppbs20" else "full clean surface"}
                     for s in a.surfaces},
        "protocol": "Panel A = oracle shift+RC registration to IC-trimmed GT core, scored "
                    "on overlap columns; Panel B = predicted length (baselines use a "
                    "pre-registered per-family train length). Gene-balanced; "
                    "CI = gene bootstrap x2000.",
        "metric_direction": HIGHER_IS_BETTER,
        "ladder": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(payload, open(a.out, "w"), indent=1)
    pgcsv = a.out.replace(".json", "_pergene.csv")
    pd.DataFrame(pergene_rows).round(5).to_csv(pgcsv, index=False)
    log(f"per-gene values -> {pgcsv} ({len(pergene_rows)} rows) "
        "[needed for paired rung-vs-rung CIs]")

    for surf, ladder in results.items():
        n_g = len({gene_of[f] for f in surf_fns[surf]})
        print(f"\n=== {surf}  ({len(surf_fns[surf])} rows / {n_g} genes) ===")
        hdr = f"{'rung':<18}" + "".join(f"{k:>12}" for k in ORDER)
        print(hdr); print("-" * len(hdr))
        for tag in ["random_uniform", "random_train_pwm", "B0_global", "B0_family",
                    "B1_nearest_pwm", "deeppbs", "v24_seed42", "v24_ens5"]:
            e = ladder.get(tag)
            if not e:
                continue
            print(f"{tag:<18}" + "".join(f"{e[k]:>12.4f}" for k in ORDER))
        print(f"{'':<18}" + "".join(f"{('up' if HIGHER_IS_BETTER[k] else 'down'):>12}"
                                    for k in ORDER))
        print(f"\n{'rung':<18}{'covR':>10}{'coverage':>10}{'len_mae':>10}{'len_bias':>10}")
        for tag in ["random_uniform", "random_train_pwm", "B0_global", "B0_family",
                    "B1_nearest_pwm", "deeppbs", "v24_seed42", "v24_ens5"]:
            e = ladder.get(tag)
            if not e:
                continue
            print(f"{tag:<18}{e['covR']:>10.4f}{e['coverage']:>10.4f}"
                  f"{e['gate_len_mae']:>10.2f}{e['gate_len_bias']:>+10.2f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
