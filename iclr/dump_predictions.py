"""Dump v24 predicted motif content + gate length per test row (CPU).

Runs the model twice: with TRUE family_id and with a ROLLED (wrong) family_id, so
we can (a) feed content/length into the 2x2 gate-swap and (b) quantify whether
v24 relies on the family_id metadata at inference (shortcut check). Sequence-only
otherwise. Saves JSON (PWMs as lists) — no further model runs needed downstream.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")


def _extract_span(P_j, gate_j, aux, j, has_span, st, ln):
    L42 = P_j.shape[1]
    if has_span:
        s = int(round(float(st[j]))); l = int(round(float(ln[j])))
        s = max(0, min(s, L42 - 1)); l = max(1, min(l, L42 - s))
        return P_j[:, s:s + l], l
    idx = np.where(gate_j)[0]
    core = P_j[:, idx] if len(idx) else P_j[:, :1]
    return core, core.shape[1]


def run(ckpt, test_data, test_split, device, family_mode):
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    from tfscope.data.dataset import TFDataset, collate_variable_length
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfgp):
        for k, v in json.load(open(cfgp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: setattr(cfg, k, v)
    model = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False); model.eval()
    ds = TFDataset(cfg, test_data, test_split, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    out = {}
    i0 = 0
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            fam = b["family_id"].clone()
            if family_mode == "rolled":
                fam = torch.roll(fam, shifts=1, dims=0)          # each row gets neighbour's (wrong) id
            elif family_mode == "zero":
                fam = torch.zeros_like(fam)
            gl, pw, aux = model(b["sequence_tokens"], b["dbd_mask"], fam,
                                retrieved_pwms=b.get("retrieved_pwms"),
                                retrieved_masks=b.get("retrieved_masks"),
                                retrieved_sims=b.get("retrieved_sims"))
            P = F.softmax(pw, 1).cpu().numpy(); gate = (gl.sigmoid() > 0.5).cpu().numpy()
            has_span = ("span_start" in aux and aux["span_start"] is not None)
            st = np.asarray(aux["span_start"].detach().cpu()).reshape(-1) if has_span else None
            ln = np.asarray(aux["span_length"].detach().cpu()).reshape(-1) if has_span else None
            for j in range(P.shape[0]):
                fn = ds.filenames[i0 + j]
                core, l = _extract_span(P[j], gate[j], aux, j, has_span, st, ln)
                out[fn] = {"content": core.astype(float).tolist(), "len": int(l)}
            i0 += P.shape[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-data", default="data/processed/tf_pwm_training_v23.parquet")
    ap.add_argument("--test-split", default="data/processed/splits/train_v22/split.json")
    ap.add_argument("--out", default="results/iclr_phase1_apples_to_apples/v24_predictions.json")
    args = ap.parse_args()
    res = {}
    for mode in ["true", "rolled"]:
        print(f"[dump] family_mode={mode} ...", flush=True)
        res[mode] = run(args.ckpt, args.test_data, args.test_split, args.device, mode)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print(f"[dump] saved {args.out}  (n={len(res['true'])})")


if __name__ == "__main__":
    main()
