#!/usr/bin/env python
"""Evaluate TFScope assuming the true PWM length is known.

Two settings:
  (1) FIRST-L  — use first L_true positions of pred (= current evaluate.py behaviour).
  (2) BEST-ALIGN — for each sample, slide a length-L_true window across the 20-position
                    prediction and pick the offset that maximizes mean-per-position Pearson r.
                    Bounds how good the predictions could be if we also knew the right alignment.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import torch.nn.functional as F

sys.path.insert(0, "src")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel


def per_position_pearson(pred, target):
    """pred, target: (4, L). Returns mean per-position Pearson r."""
    rs = []
    for i in range(pred.shape[1]):
        r = pearsonr(target[:, i], pred[:, i])[0]
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def ic_weighted_pearson(pred, target):
    """IC-weighted Pearson — matches TFScope evaluate.py formula."""
    bkg = 0.25
    ic = 2.0 - (-(target * np.log2(np.clip(target, 1e-8, 1))).sum(axis=0))
    w = ic / (ic.sum() + 1e-8)
    w4 = np.tile(w, 4)
    p4 = pred.flatten(); t4 = target.flatten()
    mp = (w4 * p4).sum(); mt = (w4 * t4).sum()
    cov = (w4 * (p4 - mp) * (t4 - mt)).sum()
    sp  = np.sqrt((w4 * (p4 - mp) ** 2).sum())
    st  = np.sqrt((w4 * (t4 - mt) ** 2).sum())
    return cov / (sp * st) if sp > 1e-8 and st > 1e-8 else np.nan


def top1_acc(pred, target):
    return float((pred.argmax(axis=0) == target.argmax(axis=0)).mean())


def kl_div(pred, target):
    p = np.clip(pred, 1e-8, 1.0); t = np.clip(target, 1e-8, 1.0)
    return float((t * (np.log(t) - np.log(p))).sum(axis=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data",  required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config saved alongside checkpoint (same logic as evaluate.py)
    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = TFScopeConfig()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass

    model = TFScopeModel(cfg, use_dummy_backbone=False).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"Loaded {args.ckpt}  (epoch={ck.get('epoch','?')})  missing={len(missing)} unexpected={len(unexpected)}")

    # Patch unseen family embeddings (same as evaluate.py)
    df_all_tmp = pd.read_parquet(args.data)
    with open(args.split) as _f:
        _split_tmp = json.load(_f)
    trained_ids = sorted(df_all_tmp[df_all_tmp["filename"].isin(_split_tmp["train"])]["family_id"].unique())
    all_ids = list(range(10))
    unseen = [i for i in all_ids if i not in trained_ids]
    if unseen and hasattr(model, "family_embedding"):
        try:
            with torch.no_grad():
                w = model.family_embedding.embedding.weight
                mean_w = w[trained_ids].mean(dim=0)
                for u in unseen:
                    w[u] = mean_w
            print(f"Patched {len(unseen)} unseen family embedding(s)")
        except Exception as e:
            print(f"  (family patch skipped: {e})")

    ds = TFDataset(cfg, args.data, args.split, split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                    collate_fn=collate_variable_length)

    all_pwm, all_gate, all_targ, all_mask = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                     for k, v in batch.items()}
            gate_logits, pwm_logits, _ = model(batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"])
            gate = gate_logits.sigmoid()
            pwm = F.softmax(pwm_logits, dim=1)
            all_pwm.append(pwm.cpu().numpy())
            all_gate.append(gate.cpu().numpy())
            all_targ.append(batch["target_pwm"].cpu().numpy())
            all_mask.append(batch["pwm_mask"].cpu().numpy())

    pred = np.concatenate(all_pwm)   # (N, 4, 20)
    gate = np.concatenate(all_gate)  # (N, 20)
    targ = np.concatenate(all_targ)  # (N, 4, 20)
    mask = np.concatenate(all_mask)  # (N, 20)

    n = len(pred)
    print(f"\nEvaluating {n} samples")
    print(f"{'Setting':<20} {'Pearson r mean':>15} {'Pearson r med':>15} {'MAE mean':>10} {'MAE med':>10}")
    print("-" * 75)

    # SETTING 1: FIRST-L (current behaviour — uses true mask)
    r_first, mae_first = [], []
    for i in range(n):
        L = int(mask[i].sum())
        if L < 2: continue
        p = pred[i, :, :L]; t = targ[i, :, :L]
        r_first.append(per_position_pearson(p, t))
        mae_first.append(float(np.abs(p - t).mean()))
    print(f"{'FIRST-L (true L)':<24} "
          f"{np.mean(r_first):>15.4f} {np.median(r_first):>15.4f} "
          f"{np.mean(mae_first):>10.4f} {np.median(mae_first):>10.4f}")

    # SETTING 1b: PREDICTED-L (deployment case — model's gate decides L)
    # If pred_L != true_L, use min(pred_L, true_L) for fair-overlap comparison.
    r_pred, mae_pred = [], []
    length_diffs = []
    for i in range(n):
        true_L = int(mask[i].sum())
        pred_L = int((gate[i] > 0.5).sum())
        if pred_L < 2 or true_L < 2: continue
        length_diffs.append(pred_L - true_L)
        common_L = min(pred_L, true_L)
        p = pred[i, :, :common_L]; t = targ[i, :, :common_L]
        r_pred.append(per_position_pearson(p, t))
        mae_pred.append(float(np.abs(p - t).mean()))
    print(f"{'PRED-L overlap':<24} "
          f"{np.mean(r_pred):>15.4f} {np.median(r_pred):>15.4f} "
          f"{np.mean(mae_pred):>10.4f} {np.median(mae_pred):>10.4f}")
    print(f"  length-diff distribution (pred_L − true_L): "
          f"mean={np.mean(length_diffs):+.2f}, median={np.median(length_diffs):+.0f}, "
          f"perfect={sum(d==0 for d in length_diffs)}/{n}")

    # SETTING 1c: PREDICTED-L strict (pad missing positions of pred with uniform [0.25])
    # — penalises the model for predicting too short.
    r_strict, mae_strict = [], []
    for i in range(n):
        true_L = int(mask[i].sum())
        pred_L = int((gate[i] > 0.5).sum())
        if true_L < 2: continue
        # Build a "deployed prediction" the user would see
        deployed = np.full((4, true_L), 0.25, dtype=np.float32)
        if pred_L > 0:
            covered = min(pred_L, true_L)
            deployed[:, :covered] = pred[i, :, :covered]
        t = targ[i, :, :true_L]
        r_strict.append(per_position_pearson(deployed, t))
        mae_strict.append(float(np.abs(deployed - t).mean()))
    print(f"{'PRED-L strict (pad)':<24} "
          f"{np.nanmean(r_strict):>15.4f} {np.nanmedian(r_strict):>15.4f} "
          f"{np.mean(mae_strict):>10.4f} {np.median(mae_strict):>10.4f}")

    # SETTING 2: GATE-GUIDED with known L — use model's gate to pick the best contiguous
    # L-position window. This is what "knowing the length and using the model's localisation"
    # looks like in practice (no ground-truth cheating).
    r_gg, mae_gg, gg_offsets = [], [], []
    for i in range(n):
        L = int(mask[i].sum())
        if L < 2: continue
        t = targ[i, :, :L]
        best_score, best_off = -np.inf, 0
        for off in range(0, 20 - L + 1):
            score = gate[i, off:off+L].mean()   # mean predicted gate probability over the window
            if score > best_score:
                best_score = score; best_off = off
        p = pred[i, :, best_off:best_off+L]
        r = per_position_pearson(p, t)
        r_gg.append(r); mae_gg.append(float(np.abs(p - t).mean())); gg_offsets.append(best_off)
    print(f"{'GATE-GUIDED (known L)':<20} "
          f"{np.nanmean(r_gg):>15.4f} {np.nanmedian(r_gg):>15.4f} "
          f"{np.mean(mae_gg):>10.4f} {np.median(mae_gg):>10.4f}")

    # SETTING 3: ORACLE-ALIGN — full metrics version, uses ground truth to pick best offset
    r_o, mae_o, icp_o, top1_o, kl_o, offs_o, lengths_o = [], [], [], [], [], [], []
    for i in range(n):
        L = int(mask[i].sum())
        if L < 2: continue
        lengths_o.append(L)
        t = targ[i, :, :L]
        best_r, best_off = -np.inf, 0
        for off in range(0, 20 - L + 1):
            p = pred[i, :, off:off+L]
            r = per_position_pearson(p, t)
            if not np.isnan(r) and r > best_r:
                best_r = r; best_off = off
        p = pred[i, :, best_off:best_off+L]
        r_o.append(per_position_pearson(p, t))
        mae_o.append(float(np.abs(p - t).mean()))
        icp_o.append(ic_weighted_pearson(p, t))
        top1_o.append(top1_acc(p, t))
        kl_o.append(kl_div(p, t))
        offs_o.append(best_off)

    print()
    print("=" * 60)
    print("ORACLE-ALIGN full metrics (best offset chosen per sample)")
    print("=" * 60)
    print(f"  n samples:               {len(r_o)}")
    print(f"  Pearson r mean:          {np.nanmean(r_o):.4f}")
    print(f"  Pearson r median:        {np.nanmedian(r_o):.4f}")
    print(f"  MAE mean:                {np.mean(mae_o):.4f}")
    print(f"  MAE median:              {np.median(mae_o):.4f}")
    print(f"  MAE (DP scale x4) mean:  {np.mean(mae_o)*4:.4f}")
    print(f"  IC-weighted Pearson:     {np.nanmean(icp_o):.4f}")
    print(f"  Top-1 accuracy:          {np.mean(top1_o):.4f}")
    print(f"  KL divergence (T||P):    {np.mean(kl_o):.4f}")
    print(f"  Offsets: 0={offs_o.count(0)}, >0={sum(o>0 for o in offs_o)}, max={max(offs_o)}")
    print(f"  Pearson distribution: <0={(np.array(r_o)<0).sum()}, "
          f"0-0.2={((np.array(r_o)>=0)&(np.array(r_o)<0.2)).sum()}, "
          f"0.2-0.4={((np.array(r_o)>=0.2)&(np.array(r_o)<0.4)).sum()}, "
          f"0.4-0.6={((np.array(r_o)>=0.4)&(np.array(r_o)<0.6)).sum()}, "
          f"0.6-0.8={((np.array(r_o)>=0.6)&(np.array(r_o)<0.8)).sum()}, "
          f">=0.8={(np.array(r_o)>=0.8).sum()}")

    print(f"\nGate-guided offset distribution:")
    print(f"  0 (=first-L): {gg_offsets.count(0)}/{n}")
    print(f"  >0 (shifted): {sum(o>0 for o in gg_offsets)}/{n}")
    print(f"  agrees with oracle: {sum(g==b for g, b in zip(gg_offsets, best_offsets))}/{n}")


if __name__ == "__main__":
    main()
