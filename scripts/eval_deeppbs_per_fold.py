#!/usr/bin/env python
"""For each DeepPBS fold model, report performance on:
  (1) its own held-out validation set (valid{i}.txt)
  (2) the blind benchmark (id.txt)

Step (1) tells us which DeepPBS checkpoint corresponds to which fold (by finding
the valid{i}.txt where the model performs worst — the held-out fold).
Step (2) gives the individual-model performance on the same 130 blind structures
that the 5-model ensemble averages to 0.702 Pearson r.
"""
import os, sys, pickle, json
import numpy as np
import torch
from scipy.stats import pearsonr, entropy as scipy_entropy
from torch_geometric.data import DataLoader

DEEPPBS_DIR = "/n/home13/leihuang/project/DeepPBS"
sys.path.insert(0, DEEPPBS_DIR)
sys.path.insert(0, os.path.join(DEEPPBS_DIR, "run"))
from deeppbs.nn.utils import loadDataset
from deeppbs.nn import processBatch
from models.model_v2 import Model

FOLD_DIR  = os.path.join(DEEPPBS_DIR, "run/folds")
DATA_DIR  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
RUN_DIR   = os.path.join(DEEPPBS_DIR, "run")
NAMES     = [l.strip() for l in open(os.path.join(RUN_DIR, "plot_scripts/txts/DeepPBS.txt"))]
CKPT_PATHS   = [os.path.join(RUN_DIR, "output", c, "Model.best.tar") for c in NAMES]
SCALER_PATHS = [os.path.join(RUN_DIR, "output", c, "scaler.pkl")     for c in NAMES]
BKG = [0.25, 0.25, 0.25, 0.25]


def eval_set(model, scaler, npz_names, device):
    ds, _, _, _ = loadDataset(npz_names, 4, "Y_pwm", DATA_DIR,
                              cache_dataset=False, balance="unmasked",
                              remove_mask=False, scale=True, scaler=scaler)
    dl = list(DataLoader(ds, batch_size=1, shuffle=False))
    maes, rs, ics = [], [], []
    for batch in dl:
        b = batch.to(device); bd = processBatch(device, b)
        with torch.no_grad():
            prob = torch.softmax(model(bd["batch"]), dim=1).cpu().numpy()
        dm0 = b.dna_mask0.cpu().numpy(); dm1 = b.dna_mask1.cpu().numpy()
        pm0 = b.pwm_mask0.cpu().numpy(); pm1 = b.pwm_mask1.cpu().numpy()
        y0  = b.y_pwm0.cpu().numpy();    y1  = b.y_pwm1.cpu().numpy()
        L = dm0.shape[0]
        pred = np.concatenate([prob[:L][dm0], prob[L:][dm1]], axis=0)
        true = np.concatenate([y0[pm0], y1[pm1]], axis=0)
        mae = float(np.mean(np.sum(np.abs(pred - true), axis=1)))
        per_pos_r = [pearsonr(true[j], pred[j])[0] for j in range(len(true))]
        r = float(np.nanmean(per_pos_r))
        ic_t = scipy_entropy(true, BKG, base=2, axis=1)
        ic_p = scipy_entropy(pred, BKG, base=2, axis=1)
        ic = float(pearsonr(ic_t, ic_p)[0]) if len(ic_t) > 1 else np.nan
        maes.append(mae); rs.append(r); ics.append(ic)
    return {
        "n":           len(maes),
        "mae_mean":    float(np.mean(maes)),
        "mae_dp":      float(np.mean(maes)),    # already in DP scale (sum-4-bases)
        "pearson_mean": float(np.nanmean(rs)),
        "pearson_med":  float(np.nanmedian(rs)),
        "ic_mean":      float(np.nanmean([x for x in ics if not np.isnan(x)])),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def load_lines(fp):
        with open(fp) as f:
            return [l.strip() for l in f if l.strip()]

    blind = load_lines(os.path.join(FOLD_DIR, "id.txt"))
    valids = [load_lines(os.path.join(FOLD_DIR, f"valid{i}.txt")) for i in range(5)]

    # Load all 5 checkpoints
    models = []
    scalers = []
    for c in range(len(CKPT_PATHS)):
        m = Model(13, 14, condition="prot_shape", readout="all")
        m.load_state_dict(torch.load(CKPT_PATHS[c], map_location=device,
                                      weights_only=False)["model_state_dict"])
        m.to(device).eval()
        models.append(m)
        scalers.append(pickle.load(open(SCALER_PATHS[c], "rb")))

    # ─── Step 1: identify ckpt → fold by performance pattern ──────────────────
    # The model trained on train{i} (= 4 folds, holding out fold i) should perform
    # WORST on valid{i}.txt (its own held-out fold).
    print("\n" + "="*78)
    print(" DeepPBS fold-model evaluation on each valid{i}.txt  (DP-scale MAE)")
    print("="*78)
    print(f"  {'Ckpt name':<35s}  " + "  ".join([f"valid{i}" for i in range(5)]))
    fold_assignment = {}
    for c_idx, name in enumerate(NAMES):
        row = []
        maes = []
        for f_idx in range(5):
            res = eval_set(models[c_idx], scalers[c_idx], valids[f_idx], device)
            maes.append(res["mae_mean"])
            row.append(f"{res['mae_mean']:.3f}")
        # The fold this model was trained to hold out is the one with the HIGHEST MAE
        held_out = int(np.argmax(maes))
        fold_assignment[name] = held_out
        print(f"  {name:<35s}  " + "  ".join(row) + f"   → held-out fold = {held_out}")

    # ─── Step 2: each fold-model on its own held-out valid set ────────────────
    print("\n" + "="*78)
    print(" Each fold model evaluated on its OWN held-out validation set")
    print("="*78)
    print(f"  {'Model':<5s}  {'n':>4s}  {'MAE(DP)':>8s}  {'MAE/4':>7s}  "
          f"{'Pearson':>8s}  {'med':>7s}  {'IC':>7s}")
    for c_idx, name in enumerate(NAMES):
        fold_i = fold_assignment[name]
        res = eval_set(models[c_idx], scalers[c_idx], valids[fold_i], device)
        print(f"  fold{fold_i}  {res['n']:>4d}  {res['mae_mean']:.3f}    "
              f"{res['mae_mean']/4:.3f}    {res['pearson_mean']:.3f}    "
              f"{res['pearson_med']:.3f}    {res['ic_mean']:.3f}")

    # ─── Step 3: each fold-model on the SAME blind benchmark ─────────────────
    print("\n" + "="*78)
    print(" Each fold model evaluated INDIVIDUALLY on the blind benchmark (id.txt, n=130)")
    print("="*78)
    print(f"  {'Model':<5s}  {'n':>4s}  {'MAE(DP)':>8s}  {'MAE/4':>7s}  "
          f"{'Pearson':>8s}  {'med':>7s}  {'IC':>7s}")
    for c_idx, name in enumerate(NAMES):
        fold_i = fold_assignment[name]
        res = eval_set(models[c_idx], scalers[c_idx], blind, device)
        print(f"  fold{fold_i}  {res['n']:>4d}  {res['mae_mean']:.3f}    "
              f"{res['mae_mean']/4:.3f}    {res['pearson_mean']:.3f}    "
              f"{res['pearson_med']:.3f}    {res['ic_mean']:.3f}")
    print(f"\n  ENSEMBLE (5-model avg) on blind benchmark = 0.553 MAE(DP) / 0.702 Pearson")


if __name__ == "__main__":
    main()
