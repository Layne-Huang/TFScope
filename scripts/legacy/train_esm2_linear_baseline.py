#!/usr/bin/env python
"""ESM2-linear baseline: frozen ESM2-650M mean-pooled DBD embedding -> MLP -> PWM.

The "is the architecture worth it?" control for Fig 1d. Trains on the cluster40 train
split (aug parquet, DBD-cropped sequences), predicts the 84-TF deeppbs_cluster40 test,
and saves predicted PWMs to results/baseline_ladder/esm2_linear_preds.npz for scoring
by rebuild_baseline_ladder_mean.py under the identical unified protocol.
"""
import os, sys, json
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("HF_HOME", "/data1/leihuang/.cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, esm
import warnings; warnings.filterwarnings("ignore")

DATA  = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
EMB_CACHE = "results/baseline_ladder/esm2_meanemb.npz"
OUT   = "results/baseline_ladder/esm2_linear_preds.npz"
MAXL, DIM = 20, 1280
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def decode_pwm(v):
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float32).reshape(4, -1).copy()
    a = np.asarray(v, dtype=np.float32)
    return a if a.shape[0] == 4 else a.T

def pad_target(pwm4L):
    L = min(pwm4L.shape[1], MAXL)
    t = np.full((4, MAXL), 0.25, np.float32); t[:, :L] = pwm4L[:, :L]
    m = np.zeros(MAXL, np.float32); m[:L] = 1.0
    return t, m


def compute_embeddings(df):
    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE, allow_pickle=True)
        return {str(n): e for n, e in zip(z["names"], z["emb"])}
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    bc = alphabet.get_batch_converter()
    embs, names = {}, []
    rows = list(df.iterrows())
    B = 8
    with torch.no_grad():
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            data = [(str(r["filename"]), str(r["sequence"])[:1022]) for _, r in chunk]
            _, _, toks = bc(data); toks = toks.to(dev)
            out = model(toks, repr_layers=[33])["representations"][33]
            for j, (_, r) in enumerate(chunk):
                seq = str(r["sequence"])[:1022]; Lr = len(seq)
                v = out[j, 1:Lr + 1].mean(0).float().cpu().numpy()  # mean over residues
                embs[str(r["filename"])] = v
            if i % 400 == 0:
                print(f"  emb {i}/{len(rows)}", flush=True)
    np.savez(EMB_CACHE, names=np.array(list(embs.keys())),
             emb=np.stack(list(embs.values())))
    return embs


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, 512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, MAXL * 4))
    def forward(self, x):
        return self.net(x).view(-1, 4, MAXL)


if __name__ == "__main__":
    df = pd.read_parquet(DATA)
    sp = json.load(open(SPLIT))
    tr, te = set(sp["train"]), set(sp["test"])   # cluster40 CLEAN train (467) — leakage-free vs the 84 test.
    # (Full aug corpus leaks: it contains family-matched near-duplicates of test TFs, inflating ESM2-linear
    #  to median 0.74 > TFScope. The clean 467-record split is the fair architecture control, matching
    #  DeepPBS's own training records.)
    df = df[df["filename"].isin(tr | te)].reset_index(drop=True)
    print(f"clean cluster40: train {len(tr & set(df['filename']))}  test {len(te)}", flush=True)

    print("[1/3] ESM2 embeddings ...", flush=True)
    embs = compute_embeddings(df)

    Xtr, Ytr, Mtr = [], [], []
    for _, r in df[df["filename"].isin(tr)].iterrows():
        if r["filename"] not in embs: continue
        t, m = pad_target(decode_pwm(r["pwm"]))
        Xtr.append(embs[r["filename"]]); Ytr.append(t); Mtr.append(m)
    Xtr = torch.tensor(np.stack(Xtr), device=dev)
    Ytr = torch.tensor(np.stack(Ytr), device=dev)
    Mtr = torch.tensor(np.stack(Mtr), device=dev)
    print(f"[2/3] train MLP on {len(Xtr)} TFs ...", flush=True)

    net = MLP().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(300):
        net.train(); opt.zero_grad()
        logits = net(Xtr)                       # (N,4,20)
        logp = F.log_softmax(logits, dim=1)
        ce = -(Ytr * logp).sum(1)               # (N,20) per-position CE
        loss = (ce * Mtr).sum() / Mtr.sum()
        loss.backward(); opt.step()
        if ep % 50 == 0: print(f"    ep{ep} loss {loss.item():.4f}", flush=True)

    print("[3/3] predict test ...", flush=True)
    net.eval(); names, preds = [], []
    with torch.no_grad():
        for _, r in df[df["filename"].isin(te)].iterrows():
            if r["filename"] not in embs: continue
            x = torch.tensor(embs[r["filename"]][None], device=dev)
            p = F.softmax(net(x), dim=1)[0].cpu().numpy()   # (4,20)
            names.append(r["filename"]); preds.append(p)
    os.makedirs("results/baseline_ladder", exist_ok=True)
    np.savez(OUT, names=np.array(names), preds=np.stack(preds))
    print(f"saved {OUT}  ({len(names)} test TFs)")
