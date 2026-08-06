#!/usr/bin/env python
"""Pairwise mutation-sensitivity fine-tune of v24. For each within-family variant
pair (A,B) we register both predictions to A's GT motif frame and add a DIRECTIONAL
loss that pushes (pred_A - pred_B) toward (GT_A - GT_B), weighted by the pair's ΔPWM
(switching pairs count more). This teaches "when the sequence changes, the motif
should change THIS way" — the signal the per-example PWM loss never provides.
Only LoRA + head params train; ESM base stays frozen. Warm-started from v24.

  PYTHONPATH=src python scripts/train_pairwise_mut.py --device cuda --epochs 8
"""
import os, sys, json, argparse
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
import pandas as pd

V24 = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
PAIRS = "data/processed/mut_pairs_v23.json"
DATA = "data/processed/tf_pwm_training_v23.parquet"


def decode(raw):
    a = raw.astype(np.float32) if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.float32)
    return a.reshape(4, -1)

def core(pwm, ic=0.2):
    icc = 2.0 + np.sum(np.clip(pwm, 1e-6, 1) * np.log2(np.clip(pwm, 1e-6, 1)), 0)
    k = np.where(icc > ic)[0]
    return pwm[:, k.min():k.max() + 1] if len(k) else pwm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--delta-weight", type=float, default=1.0)
    ap.add_argument("--batch-pairs", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/iclr_phase1/v24ft_pairmut/seed42")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)

    cfg = TFScopeConfig()
    for k, v in json.load(open(f"{V24}/config.json")).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: pass
    cfg.use_retrieval = False
    model = TFScopeModel(cfg).to(dev)
    model.load_state_dict(torch.load(f"{V24}/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)

    # freeze ESM base; train LoRA + everything else (heads/MoE/gate/indicator)
    train_params = []
    for n, p in model.named_parameters():
        trainable = ("lora_" in n) or ("backbone" not in n)
        p.requires_grad_(trainable)
        if trainable: train_params.append(p)
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.01)
    print(f"trainable params: {sum(p.numel() for p in train_params)/1e6:.1f}M")

    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    row = {r.filename: r for r in df.itertuples()}
    pairs = json.load(open(PAIRS))
    print(f"{len(pairs)} pairs")

    def tok(seq):
        return torch.tensor([[AA_TO_TOKEN.get(c, 4) for c in seq]], dtype=torch.long, device=dev)

    def predict(seq, fid):
        t = tok(seq); dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
        fi = torch.tensor([int(fid)], device=dev)
        gl, pl, _ = model(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
        return F.softmax(pl, 1)[0]                      # (4,42) differentiable

    def align_to_ref(pred_soft, ref_np):
        """Place differentiable pred (4,42) into ref frame (4,L) using offset+orient
        computed on the detached pred; returns (4,L) differentiable."""
        pnp = pred_soft.detach().cpu().numpy()
        _, off, orient, _ = align_pwm(pnp, ref_np, max_shift=10, consider_revcomp=True, min_overlap=3)
        p = pred_soft.flip(0).flip(1) if orient == "rc" else pred_soft   # RC = flip base(ACGT->TGCA is reverse rows) + reverse cols
        L = ref_np.shape[1]; out = torch.full((4, L), 0.25, device=dev)
        for i in range(p.shape[1]):
            c = i + off
            if 0 <= c < L: out[:, c] = p[:, i]
        return out

    n_switch = sum(1 for p in pairs if p["dpwm"] > 0.1)
    print(f"switching pairs (dpwm>0.1): {n_switch}")
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        rng.shuffle(pairs)
        tot_std = tot_del = 0.0; nb = 0
        for s in range(0, len(pairs), a.batch_pairs):
            chunk = pairs[s:s + a.batch_pairs]
            opt.zero_grad(); L_std = 0.0; L_del = 0.0; cnt = 0
            for pr in chunk:
                ra, rb = row.get(pr["a"]), row.get(pr["b"])
                if ra is None or rb is None: continue
                gA = core(decode(ra.pwm)); gB = core(decode(rb.pwm))
                pA = predict(str(ra.sequence), ra.family_id)
                pB = predict(str(rb.sequence), rb.family_id)
                gAt = torch.tensor(gA, device=dev, dtype=torch.float32)
                # standard: pred_A aligned to its own GT core
                pA_a = align_to_ref(pA, gA); pB_a = align_to_ref(pB, gB)
                gBt = torch.tensor(gB, device=dev, dtype=torch.float32)
                L_std = L_std + F.l1_loss(pA_a, gAt) + F.l1_loss(pB_a, gBt)
                # directional: both into A's GT frame
                pA_ra = align_to_ref(pA, gA); pB_ra = align_to_ref(pB, gA)
                gB_ra = torch.tensor(align_pwm(gB, gA, max_shift=10, consider_revcomp=True, min_overlap=3)[0],
                                     device=dev, dtype=torch.float32)
                dpred = pA_ra - pB_ra; dgt = gAt - gB_ra
                w = 1.0 + 5.0 * float(pr["dpwm"])       # up-weight switching pairs
                L_del = L_del + w * F.mse_loss(dpred, dgt)
                cnt += 1
            if cnt == 0: continue
            loss = (L_std + a.delta_weight * L_del) / cnt
            loss.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            tot_std += float(L_std) / cnt; tot_del += float(L_del) / cnt; nb += 1
        print(f"epoch {ep+1}/{a.epochs}  L_std={tot_std/max(nb,1):.4f}  L_delta={tot_del/max(nb,1):.4f}", flush=True)
        torch.save({"model": model.state_dict()}, f"{a.out}/ckpt_best.pt")
    json.dump({k: getattr(cfg, k) for k in vars(cfg) if not k.startswith("_")},
              open(f"{a.out}/config.json", "w"), default=str)
    open(f"{a.out}/DONE", "w").write("done")
    print("saved", a.out)


if __name__ == "__main__":
    main()
