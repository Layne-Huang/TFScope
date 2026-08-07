#!/usr/bin/env python
"""Head-swap experiment: train the RecognitionEnergyDecoder on the FULL train_v22
split (all families), on RAW frozen ESM-2 residue features, then score on the 291
unified benchmark (PanelA content_r) vs v24's 0.629, plus Barrera spec.change AUROC.
Tests whether v24's PWMHeadV18 is the bottleneck (Phase-8 Stages 2-4, full-family).

Stage 1: cache raw ESM-2 650M layer-avg residue embeddings for all unique seqs.
Stage 2: train recog decoder (n_fam=10), per-example oracle-aligned L1 PWM loss.
Stage 3: eval PanelA on 291 + spec.change AUROC.

  PYTHONPATH=src python scripts/train_recog_full.py --epochs 120 --device cuda
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.backbone import Backbone
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm
from iclr.unified_eval import panel_A, trimmed_core

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
FEATCACHE = "/data1/leihuang/TFScope_store/recog_esm_feats.pt"
OUT = "checkpoints/iclr_phase1/recog_full/seed42"
NPOS = 24
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


@torch.no_grad()
def extract_feats(seqs, device):
    """Raw frozen ESM-2 650M layer-avg residue embeddings, cached by md5(seq)."""
    cache = torch.load(FEATCACHE) if os.path.exists(FEATCACHE) else {}
    todo = [s for s in seqs if md5(s) not in cache]
    print(f"feat cache: {len(cache)} have, {len(todo)} to extract", flush=True)
    if todo:
        cfg = TFScopeConfig(); cfg.lora_rank = 0; cfg.freeze_encoder = True
        bb = Backbone(cfg); bb.build(torch.device(device))
        for i, s in enumerate(todo):
            tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in s]], dtype=torch.long, device=device)
            h = bb(tok)[0].float().cpu()                    # (L, 1280)
            cache[md5(s)] = h.half()
            if (i + 1) % 200 == 0:
                print(f"  feats {i+1}/{len(todo)}", flush=True); torch.save(cache, FEATCACHE)
        torch.save(cache, FEATCACHE)
    return cache


def recog_pwm(dec, h, seq, fam):
    """-> (4, NPOS) probability matrix."""
    z = dec(h, torch.tensor(onehot(seq), device=h.device), fam_id=int(fam))   # (NPOS,4) logits
    return F.softmax(z, dim=1).t()                                            # (4,NPOS)


def align_diff(pred, ref_np):
    """Place differentiable pred (4,NPOS) into ref_np frame (4,L) via detached offset+RC."""
    pnp = pred.detach().cpu().numpy()
    _, off, orient, _ = align_pwm(pnp, ref_np, max_shift=NPOS, consider_revcomp=True, min_overlap=3)
    p = pred.flip(0).flip(1) if orient == "rc" else pred
    L = ref_np.shape[1]; out = torch.full((4, L), 0.25, device=pred.device)
    for i in range(p.shape[1]):
        c = i + off
        if 0 <= c < L: out[:, c] = p[:, i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device; os.makedirs(OUT, exist_ok=True)

    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT)); tr = df[df.filename.isin(set(sp["train"]))]; te = df[df.filename.isin(set(sp["test"]))]
    feats = extract_feats(sorted(set(df.sequence.astype(str))), dev)

    dec = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=10, use_second_shell=True).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    rows = [(r.filename, str(r.sequence), int(r.family_id), trimmed_core(decode(r.pwm))) for r in tr.itertuples()]
    rows = [r for r in rows if r[3] is not None and r[3].shape[1] >= 4]
    print(f"train rows {len(rows)} | test {len(te)}", flush=True)
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        rng.shuffle(rows); tot = 0.0; nb = 0
        for s in range(0, len(rows), 16):
            opt.zero_grad(); loss = 0.0; cnt = 0
            for fn, seq, fam, core in rows[s:s + 16]:
                h = feats[md5(seq)].float().to(dev)
                pred = recog_pwm(dec, h, seq, fam)
                loss = loss + F.l1_loss(align_diff(pred, core), torch.tensor(core, device=dev)); cnt += 1
            if cnt == 0: continue
            (loss / cnt).backward(); opt.step(); tot += float(loss) / cnt; nb += 1
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep+1}/{a.epochs}  L1={tot/max(nb,1):.4f}", flush=True)
            torch.save({"model": dec.state_dict()}, f"{OUT}/ckpt_best.pt")

    # ---- eval: PanelA content_r on 291 ----
    dec.eval()
    per_gene = {}
    with torch.no_grad():
        for r in te.itertuples():
            core = trimmed_core(decode(r.pwm))
            if core is None: continue
            h = feats[md5(str(r.sequence))].float().to(dev)
            pred = recog_pwm(dec, h, str(r.sequence), int(r.family_id)).cpu().numpy()
            cr = panel_A(pred, core)["content_r"]
            per_gene.setdefault(str(r.gene_symbol).upper(), []).append(cr)
    gene_r = float(np.mean([np.mean(v) for v in per_gene.values()]))
    print(f"\n=== recog_full PanelA gene_content_r on 291 = {gene_r:.4f}  (v24=0.629) ===")
    json.dump({"panelA_gene_content_r": gene_r, "n_genes": len(per_gene)},
              open(f"{OUT}/eval291.json", "w"), indent=2)
    open(f"{OUT}/DONE", "w").write("done")
    print("saved", OUT)


if __name__ == "__main__":
    main()
