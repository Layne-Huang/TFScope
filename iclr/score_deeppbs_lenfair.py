#!/usr/bin/env python
"""Length-FAIR scoring of the PDB-disjoint retrained DeepPBS ensemble vs v24.

Fairness issue this addresses: DeepPBS emits a full-length PWM over the co-crystal
DNA (~12-24 cols). Our unified panel_A aligns pred to the GT core via best
offset+RC, so a LONGER prediction lets align_pwm cherry-pick its best-matching
window -> inflated r (the align_pwm partial-overlap bias). v24, by contrast, is
scored on its gate-committed span (a length it had to PREDICT). So raw content_r
is not apples-to-apples.

We therefore report DeepPBS three ways and print prediction lengths:
  (U) untrimmed  : full DeepPBS PWM aligned to GT core (cherry-pick allowed)
  (T) IC-trimmed : DeepPBS PWM trimmed to its OWN informative core (trimmed_core,
                   same fn used on the GT) BEFORE aligning -> DeepPBS must commit
                   to a length, matching how v24's gate commits. THIS is the fair
                   column.
v24 numbers come from B8_v24 per_sample (already gate-committed).

Run in the deeppbs env:
  /data1/leihuang/miniconda3/envs/deeppbs/bin/python iclr/score_deeppbs_lenfair.py --device cuda:0
"""
from __future__ import annotations
import argparse, importlib.util, json, pickle, re, sys
from functools import reduce
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.data import DataLoader


def _ue(path):
    s = importlib.util.spec_from_file_location("ue", path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


def _decode(raw):
    a = raw.astype(np.float32) if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.float32)
    return a.reshape(4, -1).astype(np.float32)


def _norm(p):
    a = np.asarray(p, np.float32)
    if a.shape[0] != 4 and a.shape[1] == 4: a = a.T
    a = np.clip(a, 0, None); s = a.sum(0, keepdims=True)
    return np.divide(a, s, out=np.full_like(a, 0.25), where=s > 0).astype(np.float32)


def _gene(n):
    m = re.search(r"_([A-Z0-9]+)_(HUMAN|MOUSE|RAT)", n); return m.group(1) if m else None


def _gmean(items, key):
    d = {}
    for it in items: d.setdefault(it["gene"], []).append(float(it[key]))
    gm = {g: float(np.nanmean(v)) for g, v in sorted(d.items())}
    return float(np.nanmean(list(gm.values()))), gm


def run_deeppbs(repo, out_root, models, structs, device):
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "run"))
    from deeppbs.nn import processBatch
    from deeppbs.nn.utils import loadDataset
    from models.model_v2 import Model
    cfg = json.loads((out_root / models[0] / "config.json").read_text())
    cfg["data_dir"] = str(repo / "data/assembly2024")
    batches, nets = [], []
    for mn in models:
        sc = pickle.load(open(out_root / mn / "scaler.pkl", "rb"))
        ds, _, _, _ = loadDataset(structs, cfg["nc"], cfg["labels_key"], cfg["data_dir"],
                                  cache_dataset=False, balance=cfg.get("balance", "unmasked"),
                                  remove_mask=False, scale=True, scaler=sc, pre_transform=None, feature_mask=None)
        batches.append(list(DataLoader(ds, batch_size=1, shuffle=False)))
        net = Model(13, 14, condition=cfg["condition"])
        net.load_state_dict(torch.load(out_root / mn / "Model.best.tar", map_location=device)["model_state_dict"])
        nets.append(net.to(device).eval())
    preds = {}
    for i, st in enumerate(structs):
        outs = []
        for di, net in enumerate(nets):
            b = processBatch(device, batches[di][i])
            with torch.no_grad():
                outs.append(torch.softmax(net(b["batch"]), 1).cpu().numpy())
        avg = reduce(lambda a, b: a + b, outs) / len(outs)
        h = avg.shape[0] // 2
        preds[st] = _norm((avg[:h] + np.flip(avg[h:])) / 2.0)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfscope", type=Path, default=Path("/afs/csail.mit.edu/u/l/leihuang/project/TFScope"))
    ap.add_argument("--repo", type=Path, default=Path("/data1/leihuang/DeepPBS/deeppbsmar24"))
    ap.add_argument("--out-root", type=Path, default=Path("/data1/leihuang/DeepPBS/iclr_retrain_pdb"))
    ap.add_argument("--folds", default="iclr_folds_pdbdisjoint")
    ap.add_argument("--out", type=Path, default=Path("results/iclr_phase1_apples_to_apples/deeppbs_291_lenfair.json"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    tf = a.tfscope.resolve(); ue = _ue(tf / "iclr/unified_eval.py"); dev = torch.device(a.device)
    models = [l.strip() for l in (a.out_root / "model_list.txt").read_text().split() if l.strip()]
    structs = [l.strip() for l in (a.repo / "run" / a.folds / "test20.txt").read_text().split() if l.strip()]
    sg = {s: _gene(s) for s in structs}

    split = json.loads((tf / "data/processed/splits/train_v22/split.json").read_text())
    df = pd.read_parquet(tf / "data/processed/tf_pwm_training_v23.parquet")
    te = df[df.filename.isin(set(split["test"]))].copy(); te["G"] = te.gene_symbol.astype(str).str.upper()

    preds = run_deeppbs(a.repo.resolve(), a.out_root, models, structs, dev)
    pbg = {sg[s]: preds[s] for s in structs if sg[s]}

    rows = []
    for _, r in te[te.G.isin(pbg)].iterrows():
        core = ue.trimmed_core(_decode(r.pwm))
        if core is None: continue
        full = pbg[r.G]
        trimmed = ue.trimmed_core(full)                 # DeepPBS commits to its IC core
        if trimmed is None: trimmed = full
        rows.append({"gene": r.G,
                     "U_content_r": float(ue.panel_A(full, core)["content_r"]),
                     "T_content_r": float(ue.panel_A(trimmed, core)["content_r"]),
                     "gt_len": int(core.shape[1]), "dp_full_len": int(full.shape[1]),
                     "dp_trim_len": int(trimmed.shape[1])})
    U, Ug = _gmean(rows, "U_content_r"); T, Tg = _gmean(rows, "T_content_r")

    um = json.loads((tf / "results/iclr_phase1_apples_to_apples/unified_models.json").read_text())
    v24 = [{"gene": str(s["gene"]).upper(), "A_content_r": s["A_content_r"],
            "B_covR": s["B_covR"], "pred_len": s.get("B_coverage")} for s in um["B8_v24"]["per_sample"]
           if str(s["gene"]).upper() in Ug]
    V, Vg = _gmean(v24, "A_content_r")
    Vcov, _ = _gmean(v24, "B_covR")

    # paired bootstrap on the FAIR (trimmed) axis
    genes = sorted(Tg)
    d = np.array([Tg[g] - Vg[g] for g in genes]); rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    lens = {g: {"gt": next(x["gt_len"] for x in rows if x["gene"] == g),
                "dp_full": next(x["dp_full_len"] for x in rows if x["gene"] == g),
                "dp_trim": next(x["dp_trim_len"] for x in rows if x["gene"] == g)} for g in genes}
    payload = {
        "status": "ok",
        "protocol": "PDB-disjoint retrain (gene+PDB disjoint); length-fair scoring",
        "n_genes": len(genes),
        "DeepPBS_untrimmed_content_r": U,
        "DeepPBS_IC_trimmed_content_r_FAIR": T,
        "v24_content_r": V,
        "v24_covR": Vcov,
        "paired_fair_delta_DeepPBS_minus_v24": float(d.mean()),
        "paired_fair_95CI": ci,
        "frac_DeepPBS_wins_fair": float((d > 0).mean()),
        "mean_lengths": {"gt": float(np.mean([lens[g]["gt"] for g in genes])),
                         "dp_full": float(np.mean([lens[g]["dp_full"] for g in genes])),
                         "dp_trim": float(np.mean([lens[g]["dp_trim"] for g in genes]))},
        "per_gene": [{"gene": g, "DeepPBS_U": Ug[g], "DeepPBS_T_fair": Tg[g], "v24": Vg[g],
                      **lens[g]} for g in genes],
    }
    outp = a.out if a.out.is_absolute() else tf / a.out
    outp.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n_genes", "DeepPBS_untrimmed_content_r",
        "DeepPBS_IC_trimmed_content_r_FAIR", "v24_content_r",
        "paired_fair_delta_DeepPBS_minus_v24", "paired_fair_95CI", "mean_lengths")}, indent=2))
    print("wrote", outp)


if __name__ == "__main__":
    main()
