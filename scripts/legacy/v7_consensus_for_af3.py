#!/usr/bin/env python
"""Stage 1 of the AF3+Rosetta recalibration pipeline.

Run v7_single on the blind test set, extract the predicted PWM per TF, derive a
consensus DNA sequence (argmax per position over the predicted motif length), and
emit everything needed for the downstream AF3 + pwm_rosetta steps:

  results/af3_pipeline/consensus.json   — list of {filename, gene, protein_seq,
                                            consensus_dna, pred_len, true_len}
  results/af3_pipeline/pwms/{fn}.npz    — pred_pwm (4,20), true_pwm (4,20), masks

The consensus DNA uses the gate-predicted length, padded with 3 flanking bp of
the next-most-likely base on each side so AF3 has DNA context to fold against.
"""
import os, sys, json
sys.path.insert(0, "src")
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel

CKPT  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_full/ckpt_epoch200.pt"
DATA  = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
OUT   = "results/af3_pipeline"
FLANK = 3                 # flanking bp of consensus context on each side
BASES = np.array(list("ACGT"))


def main():
    os.makedirs(os.path.join(OUT, "pwms"), exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(CKPT), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    cfg.use_retrieval = False   # v7 has no retrieval

    model = TFScopeModel(cfg, use_dummy_backbone=False).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"], strict=False)

    import pandas as pd
    df = pd.read_parquet(DATA)
    fn_to_seq  = dict(zip(df["filename"], df["sequence"]))
    fn_to_gene = dict(zip(df["filename"], df["gene_symbol"]))

    ds = TFDataset(cfg, DATA, SPLIT, split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_variable_length)

    records = []
    idx = 0
    with torch.no_grad():
        for batch in dl:
            fns = [ds.filenames[i] for i in range(idx, idx + len(batch["pwm_mask"]))]
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
            gate_logits, pwm_logits, _ = model(b["sequence_tokens"], b["dbd_mask"], b["family_id"])
            gate = gate_logits.sigmoid().cpu().numpy()
            pwm  = F.softmax(pwm_logits, dim=1).cpu().numpy()       # (B,4,20)
            targ = b["target_pwm"].cpu().numpy()
            mask = b["pwm_mask"].cpu().numpy()

            for j, fn in enumerate(fns):
                true_len = int(mask[j].sum())
                pred_len = int((gate[j] > 0.5).sum())
                pred_len = max(4, min(pred_len, 20))               # sane bounds
                p = pwm[j]                                          # (4,20)
                core_idx = p[:, :pred_len].argmax(axis=0)          # (pred_len,)
                core = "".join(BASES[core_idx])
                # flanking context: most-likely base just outside the motif (use first/last core base)
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
    print(f"Wrote {len(records)} records to {OUT}/consensus.json")
    print(f"Per-TF PWMs in {OUT}/pwms/")
    # quick stats
    plens = [r["pred_len"] for r in records]
    print(f"Predicted motif length: mean={np.mean(plens):.1f}, range=[{min(plens)},{max(plens)}]")
    print("\nExamples:")
    for r in records[:5]:
        print(f"  {r['gene']:12s} core={r['core_dna']:20s} consensus={r['consensus_dna']}")


if __name__ == "__main__":
    main()
