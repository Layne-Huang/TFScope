#!/usr/bin/env python
"""CLEAN per-family-frame head-swap: train the RecognitionEnergyDecoder on the full
train_v22 split using FIXED per-family consensus frames (registration precomputed
ONCE, offline) instead of per-example alignment in the loss. No align_pwm in the
training loop -> fast + stable target. Reuses cached ESM feats. Scores PanelA on 291.

  PYTHONPATH=src python scripts/train_recog_frames.py --epochs 150 --device cuda
"""
import os, sys, json, argparse, hashlib
from collections import defaultdict
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from iclr.unified_eval import panel_A, trimmed_core

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
FEATCACHE = "/data1/leihuang/TFScope_store/recog_esm_feats.pt"
OUT = "checkpoints/iclr_phase1/recog_frames/seed42"
NPOS = 40
AA = "ACDEFGHIKLMNPQRSTVWY"; AA_IDX = {a: i for i, a in enumerate(AA)}


def md5(s): return hashlib.md5(s.encode()).hexdigest()
def decode(raw):
    a = raw.astype(np.float32) if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.float32)
    return a.reshape(4, -1)
def onehot(seq):
    x = np.zeros((len(seq), 20), np.float32)
    for i, c in enumerate(seq):
        if c in AA_IDX: x[i, AA_IDX[c]] = 1.0
    return x
def total_ic(c):
    p = np.clip(c, 1e-6, 1); return float((p * np.log2(p / 0.25)).sum())


def build_frames(rows):
    """rows: [(fn, seq, fam, core)] -> {fn: (target 4xNPOS, mask NPOS)} in a fixed
    per-family frame; registration (offset+RC to family reference) computed once."""
    by_fam = defaultdict(list)
    for r in rows: by_fam[r[2]].append(r)
    frames = {}
    for fam, fr in by_fam.items():
        # reference = longest core (widest frame); tie-break by IC
        ref = max((r[3] for r in fr), key=lambda c: (c.shape[1], total_ic(c)))
        Lr = ref.shape[1]; base = max(0, (NPOS - Lr) // 2)
        for fn, seq, _, core in fr:
            aligned, off, orient, _ = align_pwm(core, ref, max_shift=NPOS,
                                                consider_revcomp=True, min_overlap=3)
            cc = revcomp_pwm_np(core) if orient == "rc" else core         # (4, Lc)
            tgt = np.full((4, NPOS), 0.25, np.float32); mask = np.zeros(NPOS, np.float32)
            for i in range(cc.shape[1]):
                col = base + i + off
                if 0 <= col < NPOS:
                    tgt[:, col] = cc[:, i]; mask[col] = 1.0
            if mask.sum() == 0:                                            # fell outside frame -> left-anchor
                L = min(cc.shape[1], NPOS)
                tgt[:, :L] = cc[:, :L]; mask[:L] = 1.0
            frames[fn] = (tgt, mask)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; os.makedirs(OUT, exist_ok=True)

    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT)); tr = df[df.filename.isin(set(sp["train"]))]; te = df[df.filename.isin(set(sp["test"]))]
    feats = torch.load(FEATCACHE)
    print(f"feats cached: {len(feats)}", flush=True)

    rows = [(r.filename, str(r.sequence), int(r.family_id), trimmed_core(decode(r.pwm))) for r in tr.itertuples()]
    rows = [r for r in rows if r[3] is not None and r[3].shape[1] >= 4 and md5(r[1]) in feats]
    print(f"building per-family frames for {len(rows)} rows ...", flush=True)
    frames = build_frames(rows)
    # pre-stage tensors
    staged = []
    for fn, seq, fam, core in rows:
        tgt, mask = frames[fn]
        staged.append((feats[md5(seq)].float(), onehot(seq), fam,
                       torch.tensor(tgt), torch.tensor(mask)))
    print("frames built. training ...", flush=True)

    dec = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=10, use_second_shell=False).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    idx = np.arange(len(staged))
    for ep in range(a.epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for s in range(0, len(idx), 32):
            opt.zero_grad(); loss = 0.0; cnt = 0
            for j in idx[s:s + 32]:
                h, oh, fam, tgt, mask = staged[j]
                h = h.to(dev); oh = torch.tensor(oh, device=dev); tgt = tgt.to(dev); mask = mask.to(dev)
                m = mask.bool()
                if int(m.sum()) == 0: continue                        # guard: skip empty mask
                z = dec(h, oh, fam_id=int(fam))                       # (NPOS,4)
                pred = F.softmax(z, 1).t()                            # (4,NPOS)
                loss = loss + F.l1_loss(pred[:, m], tgt[:, m]); cnt += 1
            if cnt == 0: continue
            (loss / cnt).backward(); opt.step(); tot += float(loss) / cnt; nb += 1
        if (ep + 1) % 15 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep+1}/{a.epochs}  L1={tot/max(nb,1):.4f}", flush=True)
            torch.save({"model": dec.state_dict()}, f"{OUT}/ckpt_best.pt")

    # ---- eval PanelA on 291 (oracle-align decoder output to each test GT) ----
    dec.eval(); pg = {}
    with torch.no_grad():
        for r in te.itertuples():
            core = trimmed_core(decode(r.pwm))
            if core is None or md5(str(r.sequence)) not in feats: continue
            h = feats[md5(str(r.sequence))].float().to(dev)
            z = dec(h, torch.tensor(onehot(str(r.sequence)), device=dev), fam_id=int(r.family_id))
            pred = F.softmax(z, 1).t().cpu().numpy()
            pg.setdefault(str(r.gene_symbol).upper(), []).append(panel_A(pred, core)["content_r"])
    gene_r = float(np.mean([np.mean(v) for v in pg.values()]))
    print(f"\n=== recog_frames PanelA gene_content_r on 291 = {gene_r:.4f}  (v24=0.629, recog_full-dirty=0.519) ===")
    json.dump({"panelA_gene_content_r": gene_r, "n_genes": len(pg)}, open(f"{OUT}/eval291.json", "w"), indent=2)
    open(f"{OUT}/DONE", "w").write("done")
    print("saved", OUT)


if __name__ == "__main__":
    main()
