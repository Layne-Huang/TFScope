#!/usr/bin/env python
"""E1: give v24 a PAIRED mutation objective (freeze all but the PWM head).

Signal reaches the decoder (E0) but is routed to flanks, not the specificity core.
So we supervise the WT->MUT DIFFERENCE directly, in a SHARED registration frame
(so a pure flank shift is not counted as specificity change):

  Delta_pred = centered(z_MUT) - centered(z_WT)   (centered over bases per column)
Losses: absolute (keep) + delta L1 + magnitude match (anti-collapse) + directional
cosine (impactful pairs). Only pwm_head trains; ESM/MoE/projection/gate frozen.

Eval: held-out genes -> Delta_pred (1-corr) vs measured; + WT covR (no regression).
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm
dev = "cuda"; CK = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
HELD = set(json.load(open("results/mutation_benchmark/heldout_genes.json")))
pairs = json.load(open("results/mutation_benchmark/barrera_pairs.json"))["pairs"]

def ic_core(p):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= 0.2)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]

# pre-align each pair: measured MUT->WT frame, crop to WT core -> mwt,mmut (4,Lc)
data = []
for p in pairs:
    wt = np.array(p["wt_pwm"], np.float32); mut = np.array(p["mut_pwm"], np.float32)
    al, sh, ori, _ = align_pwm(mut, wt, max_shift=6, consider_revcomp=True)  # mut in wt frame
    a, b = ic_core(wt); a = max(a, 0); b = min(b, wt.shape[1])
    mwt = wt[:, a:b]; mmut = al[:, a:b]
    if mwt.shape[1] < 4: continue
    data.append(dict(gene=p["gene"], wt_seq=p["wt_seq"], mut_seq=p["mut_seq"],
                     mwt=mwt, mmut=mmut))
train_all = [d for d in data if d["gene"] not in HELD]
test  = [d for d in data if d["gene"] in HELD]
# carve a gene-disjoint VAL from train for honest early-stopping (no peeking at test)
tr_genes = sorted({d["gene"] for d in train_all})
val_genes = set(tr_genes[::4])                      # ~1/4 of train genes
train = [d for d in train_all if d["gene"] not in val_genes]
val   = [d for d in train_all if d["gene"] in val_genes]
print(f"pairs: {len(data)} | fit {len(train)} | val {len(val)} | held-out {len(test)}")

cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev); m.use_contact_pred_head = False
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
# freeze all but pwm_head
for n, pr in m.named_parameters(): pr.requires_grad = n.startswith("pwm_head.")
opt = torch.optim.Adam([pr for pr in m.parameters() if pr.requires_grad], lr=3e-4)
ntr = sum(pr.numel() for pr in m.parameters() if pr.requires_grad)
print(f"trainable (pwm_head): {ntr:,}")

def logits(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
    _, pl, _ = m(t, dm, fi); return pl[0]                      # (4, 42)

def align_model(zwt_sm, mwt):
    """brute-force place measured core (4,Lc) onto model 42-frame via WT; return (start, oriented mwt/mmut fn)."""
    Lc = mwt.shape[1]; best = (-9, 0, False)
    z = zwt_sm.detach().cpu().numpy()
    for o, ref in [(False, mwt), (True, rc(mwt))]:
        for s in range(0, z.shape[1] - Lc + 1):
            r = np.corrcoef(z[:, s:s + Lc].ravel(), ref.ravel())[0, 1]
            if r > best[0]: best = (r, s, o)
    return best[1], best[2]

def cent(z): return z - z.mean(0, keepdim=True)   # center over bases per column

def pair_loss(d):
    zwt = logits(d["wt_seq"]); zmut = logits(d["mut_seq"])
    mwt = torch.tensor(d["mwt"], device=dev); mmut = torch.tensor(d["mmut"], device=dev)
    with torch.no_grad():
        s, o = align_model(F.softmax(zwt, 0), d["mwt"])
    Lc = mwt.shape[1]
    zw = zwt[:, s:s + Lc]; zm = zmut[:, s:s + Lc]
    tw = mwt if not o else torch.flip(mwt[[3, 2, 1, 0]], dims=[1])
    tm = mmut if not o else torch.flip(mmut[[3, 2, 1, 0]], dims=[1])
    # absolute (keep model calibrated)
    L_abs = F.kl_div(F.log_softmax(zw, 0), tw, reduction="batchmean") + \
            F.kl_div(F.log_softmax(zm, 0), tm, reduction="batchmean")
    # delta on centered logits
    pd = cent(zm) - cent(zw)
    td = (torch.log(tm + 1e-6) - torch.log(tm + 1e-6).mean(0)) - \
         (torch.log(tw + 1e-6) - torch.log(tw + 1e-6).mean(0))
    L_delta = F.smooth_l1_loss(pd, td)
    L_mag = (pd.norm() - td.norm()).abs() / (td.norm() + 1e-6)          # anti-collapse
    if td.norm() > 0.5:                                                 # directional on impactful
        L_dir = 1 - F.cosine_similarity(pd.flatten(), td.flatten(), dim=0)
    else:
        L_dir = torch.zeros((), device=dev)
    return L_abs + 3.0 * L_delta + 0.5 * L_mag + 1.0 * L_dir

@torch.no_grad()
def evaluate(rows, tag):
    m.eval(); dp = []; dt = []; covr = []
    for d in rows:
        zwt = logits(d["wt_seq"]); zmut = logits(d["mut_seq"])
        s, o = align_model(F.softmax(zwt, 0), d["mwt"]); Lc = d["mwt"].shape[1]
        pw = F.softmax(zwt[:, s:s + Lc], 0).cpu().numpy(); pm = F.softmax(zmut[:, s:s + Lc], 0).cpu().numpy()
        tw = d["mwt"] if not o else rc(d["mwt"])
        dp.append(1 - np.corrcoef(pw.ravel(), pm.ravel())[0, 1])
        dt.append(1 - np.corrcoef(d["mwt"].ravel(), d["mmut"].ravel())[0, 1])
        covr.append(np.corrcoef(pw.ravel(), tw.ravel())[0, 1])
    dp, dt = np.array(dp), np.array(dt)
    c = float(np.corrcoef(dp, dt)[0, 1]) if len(dp) > 2 else float("nan")
    r = float(np.nanmean(covr))
    if tag: print(f"  [{tag}] mean Δpred={dp.mean():.3f} (measured {dt.mean():.3f}) "
                  f"corr={c:.3f}  WT-core r={r:.3f}")
    m.train(); return dp, dt, c, r

import copy
print("\n=== held-out BEFORE ==="); evaluate(test, "before")
import random; rng = random.Random(0)
best = {"score": -9, "step": 0, "state": None}
for step in range(1, 601):
    d = rng.choice(train); opt.zero_grad(); L = pair_loss(d); L.backward(); opt.step()
    if step % 50 == 0:
        _, _, vc, vr = evaluate(val, "")                # val: pick best without touching test
        score = vc + 0.5 * vr                           # direction + keep WT
        print(f"step {step:3d} loss {L.item():5.2f}  val corr={vc:.3f} WTr={vr:.3f}  score={score:.3f}")
        if score > best["score"]:
            best.update(score=score, step=step,
                        state=copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()
                                             if k.startswith("pwm_head.")}))
print(f"\nbest val step={best['step']} score={best['score']:.3f}")
# restore best pwm_head, report held-out test at that step
if best["state"] is not None:
    m.load_state_dict(best["state"], strict=False)
print("=== held-out TEST @ best-val ==="); dp, dt, c, r = evaluate(test, "best")
os.makedirs("checkpoints/v24_e1_paired", exist_ok=True)
torch.save({"model": m.state_dict(), "best_step": best["step"],
            "test_corr": c, "test_wtr": r}, "checkpoints/v24_e1_paired/pwmhead_ft.pt")
print("saved checkpoints/v24_e1_paired/pwmhead_ft.pt")
