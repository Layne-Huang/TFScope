#!/usr/bin/env python
"""Predict a PWM for a single protein sequence with the best v8 (logit-RAG) checkpoint.

Pipeline for a novel sequence:
  1. ESM2 layer-33 embedding, DBD-mean-pooled  → query vector
  2. cosine-sim vs precomputed training embeddings → top-K neighbours (donor pool)
  3. load neighbour PWMs → retrieval prior
  4. v8 forward (logit-residual RAG) → predicted PWM + gate length
"""
import argparse, os, sys, json
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_single/ckpt_epoch200.pt"
DATA  = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
EMB   = "data/processed/tf_dbd_embeddings.npz"
BASES = np.array(list("ACGT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="protein sequence")
    ap.add_argument("--family-id", type=int, default=1, help="0-9 (default 1=C2H2_medium)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="results/single_prediction")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seq = args.seq.strip().upper()

    # ── load model ────────────────────────────────────────────────────────
    cfg = TFScopeConfig()
    cp = os.path.join(os.path.dirname(CKPT), "config.json")
    if os.path.exists(cp):
        s = json.load(open(cp))
        for k, v in s.items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    cfg.use_retrieval = True
    model = TFScopeModel(cfg, use_dummy_backbone=False).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"], strict=False)

    # ── tokenize + DBD mask (whole sequence = DBD for a pure-ZF construct) ──
    tokens = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=device)
    dbd_mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    family_id = torch.tensor([args.family_id], dtype=torch.long, device=device)

    # ── query embedding via ESM2 (layer 33, DBD-mean-pool) ─────────────────
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device).eval()
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([("q", seq)])
    with torch.no_grad():
        rep = esm_model(toks.to(device), repr_layers=[33])["representations"][33]
        qvec = rep[0, 1:1+len(seq)].mean(0).cpu().numpy().astype(np.float32)
    del esm_model; torch.cuda.empty_cache()

    # ── nearest neighbours among TRAINING donors ───────────────────────────
    embs = np.load(EMB); split = json.load(open(SPLIT))
    donors = [fn for fn in embs.files if fn in (set(split["train"]) | set(split["val"]))]
    M = np.stack([embs[fn] for fn in donors]); M = M / (np.linalg.norm(M,axis=1,keepdims=True)+1e-8)
    q = qvec / (np.linalg.norm(qvec)+1e-8)
    sims = M @ q
    top = np.argsort(-sims)[:args.k]

    import pandas as pd
    df = pd.read_parquet(DATA)
    fn2pwm = {r["filename"]: np.frombuffer(r["pwm"],dtype=np.float32).reshape(4,-1) for _,r in df.iterrows()}
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))

    K, ML = args.k, cfg.max_motif_length
    ret_pwms  = torch.full((1,K,4,ML), 0.25, dtype=torch.float32)
    ret_masks = torch.zeros((1,K,ML), dtype=torch.float32)
    ret_sims  = torch.zeros((1,K), dtype=torch.float32)
    print("Nearest training neighbours:")
    for ki, di in enumerate(top):
        fn = donors[di]; pwm = fn2pwm[fn]; L = min(pwm.shape[1], ML)
        ret_pwms[0,ki,:,:L] = torch.from_numpy(pwm[:,:L])
        ret_masks[0,ki,:L] = 1.0
        ret_sims[0,ki] = float(sims[di])
        print(f"  [{ki}] {fn2gene[fn]:12s} cos={sims[di]:.3f}  ({fn})")

    # ── predict ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        gate_logits, pwm_logits, _ = model(tokens, dbd_mask, family_id,
            retrieved_pwms=ret_pwms.to(device), retrieved_masks=ret_masks.to(device),
            retrieved_sims=ret_sims.to(device))
        gate = gate_logits.sigmoid()[0].cpu().numpy()
        pwm  = F.softmax(pwm_logits, dim=1)[0].cpu().numpy()   # (4,20)

    pred_len = max(4, int((gate > 0.5).sum()))
    core = pwm[:, :pred_len]
    consensus = "".join(BASES[core.argmax(0)])
    print(f"\nPredicted motif length: {pred_len}")
    print(f"Consensus: {consensus}")
    print("PWM (rows A,C,G,T; cols = positions):")
    for bi, base in enumerate("ACGT"):
        print(f"  {base}: " + " ".join(f"{x:.2f}" for x in core[bi]))

    np.savez_compressed(os.path.join(args.out,"prediction.npz"),
                        pwm=core, full_pwm=pwm, gate=gate, consensus=consensus, seq=seq)
    # logo
    try:
        sys.path.insert(0, "pwm_rosetta")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from pwm_hybrid.pwm.viz import makeLogo
        fig, ax = plt.subplots(figsize=(max(4,pred_len*0.6),2))
        ppm = core.T; ppm = np.clip(ppm,1e-8,1); ppm = ppm/ppm.sum(1,keepdims=True)
        makeLogo(ppm, ax); ax.set_title(f"v8 prediction (len={pred_len})")
        fig.tight_layout(); fig.savefig(os.path.join(args.out,"logo.pdf"), bbox_inches="tight")
        print(f"\nLogo → {args.out}/logo.pdf")
    except Exception as e:
        print(f"(logo skipped: {e})")
    print(f"Saved → {args.out}/prediction.npz")


if __name__ == "__main__":
    main()
