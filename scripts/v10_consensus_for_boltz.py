#!/usr/bin/env python
"""Regenerate consensus DNA from v10 (logit-RAG with trust gate, the current best
single-model checkpoint at 0.541 mean / 0.638 median Pearson r on blind benchmark).

This is a v10 drop-in for the older v7_consensus_for_af3.py. The downstream
pwm_rosetta pipeline expects:
  results/af3_pipeline_v10/consensus.json        — per-TF consensus_dna + core_dna
  results/af3_pipeline_v10/pwms/{fn}.npz         — pred_pwm + true_pwm + masks
"""
import os, sys, json
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel

CKPT  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v10_single/ckpt_epoch100.pt"
DATA  = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
OUT   = "results/af3_pipeline_v10"
FLANK = 3
BASES = np.array(list("ACGT"))


def main():
    os.makedirs(os.path.join(OUT, "pwms"), exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load config saved with the checkpoint
    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(CKPT), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    # v10 uses retrieval — keep it on for inference
    cfg.use_retrieval = True
    cfg.retrieval_index_path = "data/processed/tf_nn_index.json"
    cfg.retrieval_dropout = 0.0     # no dropout at eval

    model = TFScopeModel(cfg, use_dummy_backbone=False).to(device).eval()
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    print(f"Loaded v10 ckpt @ epoch {state.get('epoch','?')}")

    import pandas as pd
    df = pd.read_parquet(DATA)
    fn_to_seq  = dict(zip(df["filename"], df["sequence"]))
    fn_to_gene = dict(zip(df["filename"], df["gene_symbol"]))

    ds = TFDataset(cfg, DATA, SPLIT, split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                    collate_fn=collate_variable_length)

    records = []
    idx = 0
    with torch.no_grad():
        for batch in dl:
            fns = [ds.filenames[i] for i in range(idx, idx + len(batch["pwm_mask"]))]
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
            gate_logits, pwm_logits, _ = model(
                b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                retrieved_pwms=b.get("retrieved_pwms"),
                retrieved_masks=b.get("retrieved_masks"),
                retrieved_sims=b.get("retrieved_sims"),
            )
            gate = gate_logits.sigmoid().cpu().numpy()
            pwm  = F.softmax(pwm_logits, dim=1).cpu().numpy()
            targ = b["target_pwm"].cpu().numpy()
            mask = b["pwm_mask"].cpu().numpy()

            for j, fn in enumerate(fns):
                true_len = int(mask[j].sum())
                pred_len = int((gate[j] > 0.5).sum())
                pred_len = max(4, min(pred_len, 20))
                p = pwm[j]
                core_idx = p[:, :pred_len].argmax(axis=0)
                core = "".join(BASES[core_idx])
                left  = core[0] * FLANK
                right = core[-1] * FLANK
                consensus = left + core + right

                np.savez_compressed(
                    os.path.join(OUT, "pwms", f"{fn}.npz"),
                    pred_pwm=p.astype(np.float32),
                    true_pwm=targ[j].astype(np.float32),
                    pwm_mask=mask[j].astype(np.float32),
                    pred_len=pred_len, true_len=true_len,
                )
                records.append({
                    "filename":      fn,
                    "gene":          str(fn_to_gene.get(fn, "")),
                    "protein_seq":   fn_to_seq[fn],
                    "consensus_dna": consensus,
                    "core_dna":      core,
                    "pred_len":      pred_len,
                    "true_len":      true_len,
                })
            idx += len(fns)

    with open(os.path.join(OUT, "consensus.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} records → {OUT}/consensus.json")
    print(f"\nSample (compare to v7's poly-G for these — v10 should be sharper):")
    for r in records[:8]:
        print(f"  {r['gene']:12s} core={r['core_dna']}")


if __name__ == "__main__":
    main()
