"""Score a trained checkpoint (or frozen v24 = B8) through the unified evaluator.

Captures BOTH the full predicted PWM content (for Panel A) and the predicted gate
span length (for Panel B), so trained models are compared to baselines on exactly
the same two panels. Sequence-only inference (no contacts/structure fed).

Designed to run on CPU (`--device cpu`) so it never preempts the training GPUs;
thread-limit it (OMP_NUM_THREADS / taskset) to avoid starving training dataloaders.

    OMP_NUM_THREADS=4 taskset -c 0-3 python -m iclr.score_checkpoint_unified \
      --ckpt /data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt \
      --tag B8_v24 --device cpu \
      --out results/iclr_phase1_apples_to_apples/unified_models.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm
from iclr.unified_eval import score_model


def predict_full_and_gate(ckpt, test_data, test_split, device):
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    from tfscope.data.dataset import TFDataset, collate_variable_length

    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfgp):
        for k, v in json.load(open(cfgp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: setattr(cfg, k, v)
    if getattr(cfg, "use_cached_esmc", False):
        cfg.esm_embed_dim = 1152
        cfg.two_chain_input = False
        cfg.chain_id_embedding = False
        cfg.lora_rank = 0
    model = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()
    ds = TFDataset(cfg, test_data, test_split, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    preds, gate_lens = {}, {}
    with torch.no_grad():
        i0 = 0
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gate_logits, pwm_logits, aux = model(
                b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                retrieved_sims=b.get("retrieved_sims"), esmc_emb=b.get("esmc_emb"))
            P = F.softmax(pwm_logits, dim=1).cpu().numpy()               # (B,4,42) full content
            gate = (gate_logits.sigmoid() > 0.5)                          # (B,42)
            has_span = ("span_start" in aux and aux["span_start"] is not None
                        and "span_length" in aux and aux["span_length"] is not None)
            if has_span:
                st = np.asarray(aux["span_start"].detach().cpu()).reshape(-1)
                ln = np.asarray(aux["span_length"].detach().cpu()).reshape(-1)
            L42 = P.shape[2]
            for j in range(P.shape[0]):
                fn = ds.filenames[i0 + j]
                if has_span:
                    s = int(round(float(st[j]))); l = int(round(float(ln[j])))
                    s = max(0, min(s, L42 - 1)); l = max(1, min(l, L42 - s))
                    core = P[j][:, s:s + l]
                else:
                    idx = np.where(gate[j].cpu().numpy())[0]
                    core = P[j][:, idx] if len(idx) else P[j][:, :1]
                    l = core.shape[1]
                # EXTRACT the predicted motif span (not the full 42-col tensor) so
                # align_pwm's +/-shift can reach it and Panel A measures real content.
                preds[fn] = core
                gate_lens[fn] = int(l)
            i0 += P.shape[0]
    return preds, gate_lens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--test-data", default="data/processed/tf_pwm_training_v23.parquet")
    ap.add_argument("--test-split", default="data/processed/splits/train_v22/split.json")
    ap.add_argument("--out", default="results/iclr_phase1_apples_to_apples/unified_models.json")
    args = ap.parse_args()

    import torch
    if args.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    preds, gate_lens = predict_full_and_gate(args.ckpt, args.test_data, args.test_split, args.device)

    df = pd.read_parquet(args.test_data); df["filename"] = df.filename.astype(str)
    te = df[df.filename.isin(set(json.load(open(args.test_split))["test"]))]
    targets = {r.filename: _decode_pwm(r.pwm) for r in te.itertuples()}
    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name, "n_chains": int(getattr(r, "n_chains", 1)),
                         "pred_len": gate_lens.get(r.filename)} for r in te.itertuples()}

    res = score_model(args.tag, preds, targets, meta, length_fn=None)
    prev = json.load(open(args.out)) if os.path.exists(args.out) else {}
    prev[args.tag] = res
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(prev, open(args.out, "w"), indent=2)
    A = res["panelA_oracle_content"]; B = res["panelB_end_to_end"]
    print(f"[{args.tag}] PanelA gene_content_r={A['gene_content_r']:.4f}  "
          f"PanelB gene_covR={B['gene_covR']:.4f}  cov={B['mean_coverage']:.3f}  "
          f"gate_len_mae={B['gate_len_mae']:.2f}  n={res['n_scored']}  -> {args.out}")


if __name__ == "__main__":
    main()
