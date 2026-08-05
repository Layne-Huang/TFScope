"""Score a 5-model v24 ENSEMBLE through the unified evaluator (apples-to-apples
with the 5-model DeepPBS ensemble). Averages softmax(PWM) and gate logits across
the member checkpoints per sample, extracts the committed span from the averaged
gate, and scores exactly like iclr/score_checkpoint_unified.py. Sequence-only.

  OMP_NUM_THREADS=4 python -m iclr.score_v24_ensemble --tag v24_ens5 \
    --ckpts checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt \
            checkpoints/iclr_phase1/v24_ens/seed1/ckpt_best.pt ... --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm
from iclr.unified_eval import score_model


def _build_model(ckpt, device):
    import torch
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfgp):
        for k, v in json.load(open(cfgp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: setattr(cfg, k, v)
    model = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    return model.eval(), cfg


def _member_core(P, aux, j, L42):
    """Extract member j's committed motif core (4,l) via its predicted gate span."""
    if aux.get("span_start") is not None and aux.get("span_length") is not None:
        s = int(round(float(np.asarray(aux["span_start"].detach().cpu()).reshape(-1)[j])))
        l = int(round(float(np.asarray(aux["span_length"].detach().cpu()).reshape(-1)[j])))
        s = max(0, min(s, L42 - 1)); l = max(1, min(l, L42 - s))
        return P[j][:, s:s + l]
    return P[j][:, :1]


def predict_ensemble(ckpts, test_data, test_split, device):
    """Register-aligned ensemble: each member commits to its own motif core, then
    members 1..K are aligned (offset+RC) to member-0's (seed42) core and averaged.
    This avoids blurring that comes from averaging in the padded 42-col frame when
    members place the motif at different registers."""
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.data.dataset import TFDataset, collate_variable_length
    from tfscope.models.alignment import align_pwm

    models = [_build_model(c, device) for c in ckpts]
    _, cfg0 = models[0]
    ds = TFDataset(cfg0, test_data, test_split, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    preds, gate_lens = {}, {}
    with torch.no_grad():
        i0 = 0
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            member_P, member_aux = [], []
            for model, _ in models:
                gate_logits, pwm_logits, aux = model(
                    b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                    retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                    retrieved_sims=b.get("retrieved_sims"), esmc_emb=b.get("esmc_emb"))
                member_P.append(F.softmax(pwm_logits, dim=1).cpu().numpy())   # (B,4,42)
                member_aux.append(aux)
            L42 = member_P[0].shape[2]
            for j in range(member_P[0].shape[0]):
                fn = ds.filenames[i0 + j]
                ref = _member_core(member_P[0], member_aux[0], j, L42)         # seed42 frame
                stack = [ref]
                for m in range(1, len(models)):
                    ci = _member_core(member_P[m], member_aux[m], j, L42)
                    aligned, _, _, _ = align_pwm(ci, ref, max_shift=10,
                                                 consider_revcomp=True, min_overlap=3)
                    stack.append(aligned)                                     # (4, len(ref))
                cons = np.mean(np.stack(stack, 0), 0)
                cons = cons / np.clip(cons.sum(0, keepdims=True), 1e-8, None)
                preds[fn] = cons; gate_lens[fn] = int(cons.shape[1])
            i0 += member_P[0].shape[0]
    return preds, gate_lens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tag", default="v24_ens5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--test-data", default="data/processed/tf_pwm_training_v23.parquet")
    ap.add_argument("--test-split", default="data/processed/splits/train_v22/split.json")
    ap.add_argument("--out", default="results/iclr_phase1_apples_to_apples/unified_models.json")
    args = ap.parse_args()

    preds, gate_lens = predict_ensemble(args.ckpts, args.test_data, args.test_split, args.device)
    df = pd.read_parquet(args.test_data); df["filename"] = df.filename.astype(str)
    te = df[df.filename.isin(set(json.load(open(args.test_split))["test"]))]
    targets = {r.filename: _decode_pwm(r.pwm) for r in te.itertuples()}
    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name, "n_chains": int(getattr(r, "n_chains", 1)),
                         "pred_len": gate_lens.get(r.filename)} for r in te.itertuples()}
    res = score_model(args.tag, preds, targets, meta, length_fn=None)
    res["n_members"] = len(args.ckpts); res["members"] = args.ckpts
    prev = json.load(open(args.out)) if os.path.exists(args.out) else {}
    prev[args.tag] = res
    json.dump(prev, open(args.out, "w"), indent=2)
    A = res["panelA_oracle_content"]; B = res["panelB_end_to_end"]
    print(f"[{args.tag}] n_members={len(args.ckpts)}  PanelA gene_content_r={A['gene_content_r']:.4f}  "
          f"PanelB gene_covR={B['gene_covR']:.4f}  cov={B['mean_coverage']:.3f}  n={res['n_scored']}")


if __name__ == "__main__":
    main()
