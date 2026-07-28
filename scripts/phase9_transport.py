#!/usr/bin/env python
"""Phase 9 — structured mutation transport vs ordinary full-forward paired-delta.

The recognition-energy φ(a_i,h_i) is a POTENTIAL, so the LOCAL transport (flip only the AA
one-hot at the mutated residue k, keep the WT ESM context h) is a potential DIFFERENCE:
   T_local(a->a') = C[:,k]·(φ(a',h_k) - φ(a,h_k))
which is reverse/path/identity consistent BY CONSTRUCTION. The ORDINARY full-forward
transport (let ESM re-embed the mutant) is richer but NOT structurally consistent.

We measure, at MyoD1 pos 11 across several substitutions, for BOTH transports:
  identity  ||T(a->a)||                       (want 0)
  reverse   ||T(a->b)+T(b->a)||               (want 0)
  path      ||T(a->c)-(T(a->b)+T(b->c))||     (want 0)  <- the discriminating test
  L112R Δ_switch (->CACGTG)                   (want >0, correct direction)
This is the "ordinary paired-delta vs structured transport" ablation the plan names.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F, warnings, itertools
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.models.recognition_energy import RecognitionEnergyDecoder
dev = "cuda"; BASES = "ACGT"; AA = "ACDEFGHIKLMNPQRSTVWY"; AAi = {a: i for i, a in enumerate(AA)}; NPOS = 24
CK = "/data1/leihuang/TFScope/phase8_recog_energy.pt"
WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"; POS = 11
SUBS = ["L", "R", "K", "Q", "E", "A"]                                # L = WT

def onehot(seq):
    x = torch.zeros(len(seq), 20)
    for i, a in enumerate(seq):
        if a in AAi: x[i, AAi[a]] = 1.0
    return x
def rc(p): return p[[3, 2, 1, 0]][:, ::-1]
def mkseq(a): return WT[:POS] + a + WT[POS + 1:]

# ── v24 encoder features for each substitution ──
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
V = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
cfg = TFScopeConfig()
for k, v in json.load(open(V + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
vm = TFScopeModel(cfg).to(dev).eval(); vm.use_contact_pred_head = False
vm.load_state_dict(torch.load(V + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
cap = {}
vm.residue_moe.register_forward_hook(lambda m, i, o: cap.__setitem__("f", (o[0] if isinstance(o, tuple) else o).detach()))
@torch.no_grad()
def v24feat(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
    vm(t, dm, fi); return cap["f"][0].float()
feats = {a: v24feat(mkseq(a)) for a in SUBS}; wt_feat = feats["L"]

# ── decoder ──
sd = torch.load(CK, map_location=dev)
m = RecognitionEnergyDecoder(esm_dim=1280, n_pos=NPOS, n_fam=2, use_second_shell=False).to(dev).eval()
m.load_state_dict(sd["model"])
@torch.no_grad()
def z_of(feat, seq): return m(feat, onehot(seq).to(dev), fam_id=0).t().cpu().numpy()   # (4,NPOS)
def cent(x): return x - x.mean(0, keepdims=True)

# transports a->b (centered logit deltas over full frame)
def T_full(a, b):   return cent(z_of(feats[b], mkseq(b))) - cent(z_of(feats[a], mkseq(a)))
def T_local(a, b):  return cent(z_of(wt_feat, mkseq(b))) - cent(z_of(wt_feat, mkseq(a)))  # keep WT h

def norm(M): return float(np.linalg.norm(M))
def consistency(Tfn):
    ident = np.mean([norm(Tfn(a, a)) for a in SUBS])
    revs = [norm(Tfn(a, b) + Tfn(b, a)) for a, b in itertools.combinations(SUBS, 2)]
    paths = [norm(Tfn(a, c) - (Tfn(a, b) + Tfn(b, c))) for a, b, c in itertools.permutations(SUBS, 3)]
    return dict(identity=float(ident), reverse=float(np.mean(revs)), path=float(np.mean(paths)))

def motif_score(P, motif):
    idx = [BASES.index(c) for c in motif]; Lm = len(motif); best = -1e9
    for g in [P, rc(P)]:
        for s in range(0, NPOS - Lm + 1):
            best = max(best, sum(np.log(max(g[idx[j], s + j], 1e-6) / 0.25) for j in range(Lm)))
    return best
def dswitch(zwt, zmut):
    Pw, Pm = softmax(zwt), softmax(zmut)
    return (motif_score(Pm, "CACGTG") - motif_score(Pm, "CAGCTG")) - (motif_score(Pw, "CACGTG") - motif_score(Pw, "CAGCTG"))
def softmax(z): return np.exp(z - z.max(0)) / np.exp(z - z.max(0)).sum(0)

zwt = z_of(wt_feat, WT)
zmut_full = z_of(feats["R"], mkseq("R"))                      # full-forward L112R
zmut_local = z_of(wt_feat, mkseq("R"))                        # local (WT context) L112R
out = {
 "full_forward (ordinary paired-delta)": {**consistency(T_full), "L112R_dswitch": float(dswitch(zwt, zmut_full))},
 "local_transport (structured)":         {**consistency(T_local), "L112R_dswitch": float(dswitch(zwt, zmut_local))},
}
print(json.dumps(out, indent=1))
print("\ninterpretation: local(structured) identity/reverse/path ≈0 by construction; "
      "full-forward path >0 (not composable). Both should keep L112R Δ_switch>0.")
import os; os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump(out, open("results/mutation_benchmark/phase9_transport.json", "w"), indent=1)
print("saved results/mutation_benchmark/phase9_transport.json")
