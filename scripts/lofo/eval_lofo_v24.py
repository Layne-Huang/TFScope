#!/usr/bin/env python
"""Score the finished LOFO v24 models and assemble the paired family-SEEN vs
family-UNSEEN comparison.

Each LOFO run G is scored on two row sets:

  test(G)  every row of the held-out family G          -> family UNSEEN
  ctrl(G)  the global ctrl-gene rows of families != G  -> family SEEN, gene unseen

Because the ctrl gene set is global (see scripts/lofo/build_lofo_splits_v24.py), family
F's own ctrl rows sit in ctrl(G) for every G != F, and in test(F) for the one run that
held F out. That gives, on **identical rows**,

  unseen_F(row) = score of model_F        on that row
  seen_F(row)   = mean over G != F of score of model_G on that row
  delta_F       = seen_F - unseen_F          <- the cost of never seeing family F

so the comparison is paired at the row level and neither side is leaked. delta is
bootstrapped over GENES (the unit of independence), not rows.

Runs with whatever LOFO checkpoints exist; families still training are skipped and
listed, so this is safe to run mid-wave.

  python scripts/lofo/eval_lofo_v24.py --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm                        # noqa: E402
from iclr.unified_eval import panel_A, panel_B, trimmed_core  # noqa: E402
from reclassify_tf_families import classify                   # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLITDIR = "data/processed/splits/lofo_v24"
CKROOT = "checkpoints/lofo_v24"
JOBROOT = "/data1/leihuang/TFScope_store/v26_logs"
OUT = "results/family_lofo/lofo_v24_paired.json"


def training_state(tagname):
    """RUNNING / DONE / FAILED:n / STALE / NO_JOB for a family's detached training job.

    `ckpt_best.pt` is written from the first improving epoch onward, so its presence says
    nothing about convergence. Scoring a still-training family silently mixes half-trained
    models into BOTH sides of the paired comparison (it did, on the first run: bZIP was at
    epoch 60 of 225 and still got a delta reported).
    """
    d = os.path.join(JOBROOT, f"lofo_v24_{tagname}")
    try:
        status = open(os.path.join(d, "STATUS")).read().strip()
    except OSError:
        return "NO_JOB"
    if status == "RUNNING":
        try:
            os.kill(int(open(os.path.join(d, "pid")).read().strip()), 0)
        except (OSError, ValueError):
            return "STALE"
    return status


def ckpt_meta(ck):
    """Epoch the best checkpoint was written at, and its val oracle-r."""
    import torch
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    return {"ckpt_epoch": int(sd.get("epoch", -1)),
            "ckpt_val_oracle_r": round(float(sd.get("best_oracle_r", float("nan"))), 4)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_taxonomy(df):
    df = df.copy()
    df["g"] = df.gene_symbol.fillna("").astype(str).str.upper()
    seed = {r.g: r.family_name for r in
            df[df.family_name.isin(["C2H2_short", "C2H2_medium", "C2H2_long"])].itertuples()}
    fam = df.g.map(lambda x: classify(x, seed))
    df["fam_lofo"] = fam.map(lambda f: "C2H2" if str(f).startswith("C2H2") else f)
    return df


def build_model(ckpt, device):
    import torch
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfgp):
        for k, v in json.load(open(cfgp)).items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model", sd), strict=False)
    return m.eval(), cfg


def predict(model, cfg, split_path, split_key, device, batch=4):
    """Committed motif core per row: (filename -> (4,L)) using the model's gate span."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.data.dataset import TFDataset, collate_variable_length

    ds = TFDataset(cfg, DATA, split_path, split=split_key, max_seq_len=1024)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)
    preds, lens, i0 = {}, {}, 0
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gate_logits, pwm_logits, aux = model(
                b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                retrieved_sims=b.get("retrieved_sims"), esmc_emb=b.get("esmc_emb"))
            P = F.softmax(pwm_logits, dim=1).cpu().numpy()
            W = P.shape[2]
            ss = aux.get("span_start"); sl = aux.get("span_length")
            for j in range(P.shape[0]):
                if ss is not None and sl is not None:
                    s = int(round(float(np.asarray(ss.detach().cpu()).reshape(-1)[j])))
                    l = int(round(float(np.asarray(sl.detach().cpu()).reshape(-1)[j])))
                    s = max(0, min(s, W - 1)); l = max(1, min(l, W - s))
                    core = P[j][:, s:s + l]
                else:
                    g = torch.sigmoid(gate_logits[j]).cpu().numpy() > 0.5
                    core = P[j][:, g] if g.sum() >= 1 else P[j][:, :1]
                fn = ds.filenames[i0 + j]
                preds[fn] = core; lens[fn] = int(core.shape[1])
            i0 += P.shape[0]
    return preds, lens


def score(preds, lens, meta):
    rows = []
    for fn, pred in preds.items():
        m = meta.get(fn)
        if m is None:
            continue
        core = trimmed_core(m["pwm"])
        if core is None:
            continue
        A = panel_A(pred, core)
        B = panel_B(pred, core, lens.get(fn))
        rows.append({"filename": fn, "gene": m["gene"], "family": m["family"],
                     "content_r": A["content_r"], "covR": B["covR"],
                     "coverage": B["coverage"]})
    return rows


def gene_mean(vals_by_gene):
    return float(np.nanmean([np.nanmean(v) for v in vals_by_gene.values()])) if vals_by_gene else float("nan")


def paired_boot(pairs_by_gene, n_boot=5000, seed=0):
    """pairs_by_gene: gene -> (seen_list, unseen_list). Bootstrap the gene-level delta."""
    genes = sorted(pairs_by_gene)
    if not genes:
        return float("nan"), (float("nan"), float("nan")), 0
    d = np.array([np.nanmean(pairs_by_gene[g][0]) - np.nanmean(pairs_by_gene[g][1])
                  for g in genes], float)
    if d.size == 1:
        return float(d[0]), (float("nan"), float("nan")), 1
    rng = np.random.RandomState(seed)
    boot = [np.nanmean(rng.choice(d, d.size, replace=True)) for _ in range(n_boot)]
    return (float(np.nanmean(d)),
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), int(d.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--allow-in-progress", action="store_true",
                    help="also score families whose training job is still RUNNING "
                         "(their ckpt_best may still move; every number becomes provisional)")
    a = ap.parse_args()

    import torch
    assert torch.cuda.is_available() or a.device == "cpu", \
        f"no CUDA for CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"

    df = add_taxonomy(pd.read_parquet(DATA)); df["filename"] = df.filename.astype(str)
    meta = {r.filename: {"gene": r.g, "family": r.fam_lofo, "pwm": _decode_pwm(r.pwm)}
            for r in df.itertuples()}
    manifest = json.load(open(f"{SPLITDIR}/_manifest.json"))

    avail, pending, states = [], [], {}
    for fam in (a.families or list(manifest["families"])):
        tagname = fam.replace("/", "-")
        ck = f"{CKROOT}/{tagname}/ckpt_best.pt"
        st = training_state(tagname)
        states[fam] = st
        if not os.path.exists(ck):
            pending.append((fam, tagname, f"{st}, no ckpt")); continue
        if st == "RUNNING" and not a.allow_in_progress:
            pending.append((fam, tagname, "still training")); continue
        avail.append((fam, tagname, ck, st))
    log(f"scoring ({len(avail)}): " +
        ", ".join(f"{f}[{s}]" for f, _, _, s in avail))
    if pending:
        log(f"skipped ({len(pending)}): " + ", ".join(f"{f}({w})" for f, _, w in pending))
    if a.allow_in_progress:
        log("WARNING: --allow-in-progress — in-flight ckpt_best files are included; "
            "every delta below is provisional on BOTH sides of the pair")
    if not avail:
        sys.exit("no finished LOFO checkpoints yet — nothing to score")
    if len(avail) < 2:
        log("NOTE: only one finished model — 'seen' has no reference model, so the "
            "paired delta cannot be computed yet; the LOFO column is still valid")

    # ── run every model over its own test split and its ctrl split ────────────
    test_rows, ctrl_rows, ck_meta = {}, {}, {}
    for fam, tagname, ck, st in avail:
        t0 = time.time()
        ck_meta[fam] = {**ckpt_meta(ck), "job_status": st}
        model, cfg = build_model(ck, a.device)
        p, l = predict(model, cfg, f"{SPLITDIR}/{tagname}.json", "test", a.device)
        test_rows[fam] = score(p, l, meta)
        p, l = predict(model, cfg, f"{SPLITDIR}/{tagname}__ctrl.json", "test", a.device)
        ctrl_rows[fam] = score(p, l, meta)
        del model
        torch.cuda.empty_cache()
        log(f"  {fam:<18} test={len(test_rows[fam]):>4} rows  ctrl={len(ctrl_rows[fam]):>4} rows "
            f"| ckpt@epoch {ck_meta[fam]['ckpt_epoch']} val_oracle_r "
            f"{ck_meta[fam]['ckpt_val_oracle_r']} ({time.time()-t0:.0f}s)")

    done = [f for f, _, _, _ in avail]

    # ── paired seen-vs-unseen per family, on that family's OWN ctrl rows ──────
    per_family = {}
    for fam in done:
        own = set(manifest["families"][fam]["own_ctrl_rows"])
        unseen = {r["filename"]: r for r in test_rows[fam] if r["filename"] in own}
        pairs = {}
        seen_n_models = 0
        for g in done:
            if g == fam:
                continue
            hits = [r for r in ctrl_rows[g] if r["filename"] in own]
            if hits:
                seen_n_models += 1
            for r in hits:
                if r["filename"] not in unseen:
                    continue
                pairs.setdefault(r["gene"], ([], []))
                pairs[r["gene"]][0].append(r["content_r"])
                pairs[r["gene"]][1].append(unseen[r["filename"]]["content_r"])
        d, ci, ngenes = paired_boot(pairs)
        seen = gene_mean({g: v[0] for g, v in pairs.items()})
        uns = gene_mean({g: v[1] for g, v in pairs.items()})

        # full held-out family (all rows of F), for the headline LOFO number
        full = test_rows[fam]
        by_gene = {}
        for r in full:
            by_gene.setdefault(r["gene"], []).append(r["content_r"])
        per_family[fam] = {
            **ck_meta[fam],
            "n_test_rows": len(full), "n_test_genes": len(by_gene),
            "lofo_content_r_full_family": round(gene_mean(by_gene), 4),
            "paired": {
                "n_genes": ngenes, "n_reference_models": seen_n_models,
                "seen_content_r": round(seen, 4), "unseen_content_r": round(uns, 4),
                "delta": round(d, 4),
                "delta_ci95": [round(c, 4) for c in ci],
            },
        }

    res = {"note": "paired family-SEEN vs family-UNSEEN, identical rows, neither side leaked",
           "metric": "iclr/unified_eval Panel A content_r, gene-balanced; "
                     "delta CI = gene-level bootstrap x5000",
           "allow_in_progress": bool(a.allow_in_progress),
           "families_scored": done, "job_states": states,
           "families_pending": {f: w for f, _, w in pending},
           "per_family": per_family}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    flat = [dict(r, model=f, split="test") for f in done for r in test_rows[f]] + \
           [dict(r, model=f, split="ctrl") for f in done for r in ctrl_rows[f]]
    pd.DataFrame(flat).round(4).to_csv(a.out.replace(".json", "_rows.csv"), index=False)

    print("\n=== LOFO v24: cost of never seeing the family (paired, same rows) ===")
    print(f"{'family':<18} {'ep':>4} {'LOFO r':>8} {'genes':>6} | {'seen':>7} {'unseen':>7} "
          f"{'delta':>8} {'95% CI':>18} {'n_g':>4} {'refs':>5}")
    print("-" * 97)
    for fam, e in sorted(per_family.items(),
                         key=lambda kv: -(kv[1]["paired"]["delta"] if
                                          kv[1]["paired"]["n_genes"] else -9)):
        p = e["paired"]
        ci = p["delta_ci95"]
        if not p["n_genes"]:
            cis, ds, ss, us = "no reference model", "     n/a", "    n/a", "    n/a"
        else:
            cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "n/a (1 gene)"
            ds = f"{p['delta']:>+8.4f}"
            ss = f"{p['seen_content_r']:>7.4f}"; us = f"{p['unseen_content_r']:>7.4f}"
        print(f"{fam:<18} {e['ckpt_epoch']:>4} {e['lofo_content_r_full_family']:>8.4f} "
              f"{e['n_test_genes']:>6} | {ss} {us} {ds} {cis:>18} "
              f"{p['n_genes']:>4} {p['n_reference_models']:>5}")
    print("\n'ep' = epoch the best checkpoint was written at. 'refs' = how many other LOFO "
          "models\ncontributed to the family-SEEN side; the delta gains power as more finish.")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
