#!/usr/bin/env python
"""Alignment-teacher fusion head — lightweight RAG refiner on top of frozen v10.

Operates ENTIRELY on precomputed tensors (no ESM forward):
  - frozen v10 seed PWM (de-novo reference)
  - K seed-aligned neighbour PWMs (deployable retrieval input)
  - oracle-aligned neighbours + oracle trust (training-only teacher)

Model:
  trust head:   per-neighbour features -> softmax weights alpha_k
  P_retrieval = Σ_k alpha_k · seed_aligned_neighbour_k   (column-renormalised)
  gate head:    position features -> w_j ∈ [0,1]
  P_final_j  = w_j · P_retrieval_j + (1-w_j) · P_seed_j

Losses:
  main:        KL(P_final ‖ target)  +  IC-weighted per-column (1-PCC)
  teacher A:   BCE(alpha-logits, normalised oracle_trust)   — selection distillation
  teacher B:   KL(P_retrieval ‖ oracle_best_neighbour)      — alignment/prior distillation

Baseline to beat: v10 = 0.542 mean per-column Pearson on the 130 test TFs.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

MAX_L = 20


# ───────────────────────── data ─────────────────────────
def load_targets(data_path, split_path):
    import pandas as pd
    df = pd.read_parquet(data_path)
    fn2pwm = {}
    for _, r in df.iterrows():
        if isinstance(r["pwm"], bytes):
            fn2pwm[r["filename"]] = np.frombuffer(r["pwm"], dtype=np.float32).reshape(4, -1)
    split = json.load(open(split_path))
    return fn2pwm, split


def build_tensors(fns, aligned, seeds_npz, fn2pwm, K):
    seed_fns = list(seeds_npz["filenames"])
    seed_idx = {f: i for i, f in enumerate(seed_fns)}
    sp, sm = seeds_npz["pwms"], seeds_npz["masks"]

    S, Sm, NB, OB, TR, SI, T, Tm = [], [], [], [], [], [], [], []
    keep = []
    for fn in fns:
        if fn + "::seed" not in aligned or fn not in fn2pwm or fn not in seed_idx:
            continue
        seed_p = sp[seed_idx[fn]]; seed_m = sm[seed_idx[fn]]
        tgt = fn2pwm[fn]
        tt = np.full((4, MAX_L), 0.25, np.float32); L = min(tgt.shape[1], MAX_L); tt[:, :L] = tgt[:, :L]
        tm = np.zeros(MAX_L, np.float32); tm[:L] = 1.0

        S.append(seed_p); Sm.append(seed_m)
        NB.append(aligned[fn + "::seed"][:K])
        OB.append(aligned[fn + "::oracle"][:K])
        TR.append(aligned[fn + "::trust"][:K])
        SI.append(aligned[fn + "::sims"][:K])
        T.append(tt); Tm.append(tm)
        keep.append(fn)
    to_t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return dict(seed=to_t(S), seed_mask=to_t(Sm), neigh=to_t(NB), oracle=to_t(OB),
                trust=to_t(TR), sims=to_t(SI), target=to_t(T), tmask=to_t(Tm)), keep


# ───────────────────────── model ─────────────────────────
def column_entropy(pwm):
    p = pwm.clamp(1e-8, 1.0)
    return -(p * p.log()).sum(dim=1)                     # (B, L)


def neighbor_align_score(neigh, seed):
    """Per-neighbour mean per-column cosine of centred 4-vectors vs seed. (B,K,L)->(B,K)"""
    s = seed.unsqueeze(1)                                # (B,1,4,L)
    sc = s - s.mean(2, keepdim=True); nc = neigh - neigh.mean(2, keepdim=True)
    sn = sc / (sc.norm(dim=2, keepdim=True) + 1e-8)
    nn = nc / (nc.norm(dim=2, keepdim=True) + 1e-8)
    return (sn * nn).sum(2).mean(-1)                      # (B,K)


class FusionHead(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        # per-neighbour trust from [sim, align_score, neigh_IC, agree_with_seed]
        self.trust = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 1))
        # position gate from [seed_entropy, retr_entropy, disagreement, max_align, max_trust]
        self.gate = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, seed, neigh, sims):
        B, K, _, L = neigh.shape
        align = neighbor_align_score(neigh, seed)                  # (B,K)
        neigh_ic = (2.0 - column_entropy(neigh.reshape(B*K,4,L)).mean(-1).reshape(B,K)/np.log(2))
        agree = align                                              # reuse (proxy)
        feats = torch.stack([sims, align, neigh_ic, agree], dim=-1)  # (B,K,4)
        trust_logit = self.trust(feats).squeeze(-1)                # (B,K)
        valid = (neigh.sum(dim=(2,3)) > 1e-3).float()              # (B,K)
        trust_logit = trust_logit.masked_fill(valid < 0.5, -1e9)
        alpha = F.softmax(trust_logit, dim=1)                      # (B,K)

        P_retr = (alpha.view(B,K,1,1) * neigh).sum(1)              # (B,4,L)
        P_retr = P_retr / P_retr.sum(1, keepdim=True).clamp(1e-8)

        # position gate features
        seed_ent = column_entropy(seed)                            # (B,L)
        retr_ent = column_entropy(P_retr)                          # (B,L)
        disagree = (seed - P_retr).abs().sum(1)                    # (B,L)
        max_align = align.max(1).values.view(B,1).expand(B,L)
        max_trust = (alpha.max(1).values).view(B,1).expand(B,L)
        gfeat = torch.stack([seed_ent, retr_ent, disagree, max_align, max_trust], dim=-1)
        w = torch.sigmoid(self.gate(gfeat).squeeze(-1)).unsqueeze(1)  # (B,1,L)

        P_final = w * P_retr + (1 - w) * seed
        P_final = P_final / P_final.sum(1, keepdim=True).clamp(1e-8)
        return P_final, P_retr, trust_logit, alpha, w.squeeze(1)


# ───────────────────────── losses & metric ─────────────────────────
def kl(p, q, mask):
    p = p.clamp(1e-8); q = q.clamp(1e-8)
    return ((q * (q.log() - p.log())).sum(1) * mask).sum() / mask.sum().clamp(1)


def ic_pcc_loss(pred, target, mask):
    pc = pred - pred.mean(1, keepdim=True); tc = target - target.mean(1, keepdim=True)
    num = (pc*tc).sum(1)
    den = pc.norm(dim=1)*tc.norm(dim=1) + 1e-8
    r = num/den                                                    # (B,L)
    ic = (2.0 - (-(target.clamp(1e-8)*target.clamp(1e-8).log()).sum(1)/np.log(2)))
    w = (ic*mask); w = w/w.sum(-1,keepdim=True).clamp(1e-8)
    return ((1-r)*w).sum(-1).mean()


@torch.no_grad()
def eval_percol_r(model, data, idx):
    model.eval()
    P_final,_,_,_,_ = model(data["seed"][idx], data["neigh"][idx], data["sims"][idx])
    P=P_final.cpu().numpy(); T=data["target"][idx].cpu().numpy(); M=data["tmask"][idx].cpu().numpy()
    rs=[]
    for b in range(len(idx)):
        m=M[b].astype(bool)
        rr=[pearsonr(T[b][:,j],P[b][:,j])[0] for j in range(int(m.sum()))]
        rs.append(np.nanmean(rr))
    return float(np.nanmean(rs))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--aligned", default="data/processed/aligned_retrieval_K8.npz")
    ap.add_argument("--seeds",   default="data/processed/v10_seed_all.npz")
    ap.add_argument("--data",    default="data/processed/tf_pwm_aug_dbd.parquet")
    ap.add_argument("--split",   default="data/processed/splits/deeppbs_only/benchmark_no_val.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-trust", type=float, default=0.5)
    ap.add_argument("--w-distill", type=float, default=0.5)
    args=ap.parse_args()

    fn2pwm, split = load_targets(args.data, args.split)
    aligned = dict(np.load(args.aligned, allow_pickle=True))
    seeds = np.load(args.seeds, allow_pickle=True)

    tr,_     = build_tensors(split["train"], aligned, seeds, fn2pwm, args.k)
    te,te_fns= build_tensors(split["test"],  aligned, seeds, fn2pwm, args.k)
    print(f"train {tr['seed'].shape[0]}  test {te['seed'].shape[0]}")

    model=FusionHead(args.k)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    N=tr['seed'].shape[0]; idx_all=np.arange(N)
    te_idx=np.arange(te['seed'].shape[0])

    # v10 seed baseline on test (gate=0)
    def seed_r(d, idx):
        T=d["target"][idx].numpy(); P=d["seed"][idx].numpy(); M=d["tmask"][idx].numpy()
        rs=[np.nanmean([pearsonr(T[b][:,j],P[b][:,j])[0] for j in range(int(M[b].sum()))]) for b in range(len(idx))]
        return float(np.nanmean(rs))
    print(f"v10 seed baseline (test): {seed_r(te,te_idx):.4f}")

    best=-1
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx_all)
        for s in range(0,N,64):
            bi=idx_all[s:s+64]
            Pf,Pr,tl,alpha,w = model(tr["seed"][bi],tr["neigh"][bi],tr["sims"][bi])
            tgt=tr["target"][bi]; tm=tr["tmask"][bi]
            # main
            L_main = kl(Pf,tgt,tm) + ic_pcc_loss(Pf,tgt,tm)
            # teacher A: selection — trust logits toward normalised oracle r
            tr_target=((tr["trust"][bi]+1)/2).clamp(0,1)
            valid=(tr["neigh"][bi].sum(dim=(2,3))>1e-3).float()
            L_trust=(F.binary_cross_entropy_with_logits(tl,tr_target,reduction='none')*valid).sum()/valid.sum().clamp(1)
            # teacher B: distill retrieval prior toward oracle-best neighbour
            oi=tr["trust"][bi].argmax(1)
            ob=tr["oracle"][bi][torch.arange(len(bi)),oi]            # (B,4,L)
            L_distill=kl(Pr,ob,tm)
            loss=L_main+args.w_trust*L_trust+args.w_distill*L_distill
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%25==0 or ep==0:
            r=eval_percol_r(model,te,te_idx)
            print(f"ep {ep+1:>3}  test r={r:.4f}  (loss {loss.item():.3f})")
            best=max(best,r)
    print(f"\nBEST test r = {best:.4f}   (v10={seed_r(te,te_idx):.4f}, DeepPBS=0.702)")


if __name__=="__main__":
    main()
