#!/usr/bin/env python
"""Stage-A contrastive pretraining: align the TFScope protein tower with DNA.

Trains ContrastivePretrainModel on (protein, bound-DNA) pairs so the ESM2(+LoRA)
encoder + pooling + projection learn DNA-binding-aware representations BEFORE
Stage-B PWM finetuning. Saves an encoder checkpoint that train.py can load via
`--init-from-pretrain`.

Sources (combine with --sources):
  dpac    : data/raw/dpac/DNA_train.csv  (Protein,Motif ; BioLip2, ~11.4k, broad)
  htselex : data/processed/htselex/kmer_enrichment.parquet  (per-TF k-mer landscapes)

Usage:
  python scripts/pretrain_contrastive.py --sources dpac --epochs 30 \
      --out /n/holylabs/.../checkpoints/pretrain_dpac
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import AA_TO_TOKEN, ESM2_PAD_TOKEN
from tfscope.models.pretrain import ContrastivePretrainModel, dna_to_onehot

VALID_DNA = set("ACGT")


def load_pairs(sources, htselex_min_enrich=2.0):
    """Return DataFrame[protein, dna, weight]."""
    frames = []
    if "dpac" in sources:
        d = pd.read_csv("data/raw/dpac/DNA_train.csv").rename(
            columns={"Protein": "protein", "Motif": "dna"})
        d["weight"] = 1.0
        frames.append(d[["protein", "dna", "weight"]])
        print(f"  dpac: {len(d)} pairs")
    if "htselex" in sources:
        h = pd.read_parquet("data/processed/htselex/kmer_enrichment.parquet")
        h = h[h["enrichment"] >= htselex_min_enrich]
        h = h.rename(columns={"sequence": "protein", "kmer": "dna"})
        h["weight"] = 1.0
        frames.append(h[["protein", "dna", "weight"]])
        print(f"  htselex: {len(h)} pairs (enrich>={htselex_min_enrich})")
    df = pd.concat(frames, ignore_index=True)
    # clean: valid protein chars + valid DNA
    df = df[df["dna"].astype(str).apply(lambda s: len(s) >= 3 and set(s.upper()) <= VALID_DNA)]
    df = df[df["protein"].astype(str).str.len().between(20, 500)]
    return df.reset_index(drop=True)


class PairDataset(Dataset):
    def __init__(self, df, max_seq_len=512, dna_max_len=50):
        self.df = df; self.max_seq_len = max_seq_len; self.dna_max_len = dna_max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        toks = [AA_TO_TOKEN.get(a, 3) for a in r["protein"].upper()][:self.max_seq_len]
        return {"tokens": torch.tensor(toks, dtype=torch.long), "dna": r["dna"]}


def collate(batch, dna_max_len):
    maxL = max(len(b["tokens"]) for b in batch)
    B = len(batch)
    tok = torch.full((B, maxL), ESM2_PAD_TOKEN, dtype=torch.long)
    mask = torch.zeros(B, maxL, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["tokens"]); tok[i, :L] = b["tokens"]; mask[i, :L] = True
    dna = dna_to_onehot([b["dna"] for b in batch], dna_max_len)
    return tok, mask, dna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["dpac"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-n-layers", type=int, default=6)
    ap.add_argument("--contrastive-dim", type=int, default=256)
    ap.add_argument("--dna-max-len", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--htselex-min-enrich", type=float, default=2.0)
    ap.add_argument("--dummy-backbone", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print(f"Loading pairs from {args.sources} ...")
    df = load_pairs(args.sources, args.htselex_min_enrich)
    print(f"Total clean pairs: {len(df)} | unique proteins: {df['protein'].nunique()}")

    cfg = TFScopeConfig()
    cfg.lora_rank = args.lora_rank; cfg.lora_alpha = args.lora_alpha
    cfg.lora_n_layers = args.lora_n_layers
    json.dump({k: getattr(cfg, k) for k in vars(cfg) if not k.startswith("_")},
              open(os.path.join(args.out, "config.json"), "w"),
              default=str, indent=2)

    model = ContrastivePretrainModel(cfg, contrastive_dim=args.contrastive_dim,
                                     dna_max_len=args.dna_max_len,
                                     use_dummy_backbone=args.dummy_backbone).to(device)
    if not args.dummy_backbone:
        model.backbone.build(device)

    ds = PairDataset(df, dna_max_len=args.dna_max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    collate_fn=lambda b: collate(b, args.dna_max_len))

    # param groups: LoRA gets its own small LR
    lora_p, other_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (lora_p if "lora_" in n else other_p).append(p)
    opt = torch.optim.AdamW(
        [{"params": other_p, "lr": args.lr},
         {"params": lora_p, "lr": args.lora_lr}], weight_decay=0.01)
    n_train = sum(p.numel() for p in lora_p + other_p)
    print(f"Trainable params: {n_train/1e6:.2f}M  (lora {sum(p.numel() for p in lora_p)/1e6:.2f}M)")

    best = float("inf")
    print(f"\n{'epoch':>6} {'loss':>9} {'acc':>7} {'temp':>7} {'sec':>6}")
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl = ta = n = 0
        for tok, mask, dna in dl:
            tok = tok.to(device); mask = mask.to(device); dna = dna.to(device)
            loss, m = model(tok, dna, dbd_mask=mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += m["loss"]; ta += m["acc"]; n += 1
        tl /= n; ta /= n
        print(f"{ep+1:>6} {tl:>9.4f} {ta:>7.3f} {m['temp']:>7.4f} {time.time()-t0:>6.0f}",
              flush=True)
        ckpt = {"encoder": model.encoder_state_dict(),
                "full": model.state_dict(), "epoch": ep + 1,
                "loss": tl, "acc": ta, "config": cfg.__dict__}
        torch.save(ckpt, os.path.join(args.out, "ckpt_last.pt"))
        if tl < best:
            best = tl
            torch.save(ckpt, os.path.join(args.out, "ckpt_best.pt"))
    print(f"\nDone. Best loss {best:.4f}. Encoder → {args.out}/ckpt_best.pt")


if __name__ == "__main__":
    main()
