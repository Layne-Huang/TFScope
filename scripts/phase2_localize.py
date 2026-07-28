#!/usr/bin/env python
"""Phase 2: localize where the MyoD1 WT->L112R mutation signal disappears.

Instruments the 5 stages (ESM -> ResidueMoE -> projection -> contact-attention ->
PWM logits) and runs 4 conditions:
  (1) normal inference
  (2) shared gate/span (Δpred scored on the shared WT core -- registration control)
  (3) shared ORACLE 1D contact (recognition_residues -> contact_override, both WT&MUT)
  (4) forced-augment: big 2D attention bias L112-residue -> central E-box positions

Outputs per condition: per-stage cosine/L2 WT-vs-MUT (pooled + at mutation site +
neighborhood), contact-attention diff, centered PWM-logit diff, signed central-base
log-odds (the CAGCTG<->CACGTG switch), and final Δpred = 1-corr over the WT core.

Decision rule (user): oracle recovers switch -> fix contact PERCEPTION; oracle still
fails -> replace DECODER; signal dies after MoE -> bypass family-conditioned MoE.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.pwm_head_v18 import ContactCrossAttention
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda"; CK = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"     # MyoD1 bHLH DBD
MUTPOS = 11                                                     # L112 (DBD-relative), L->R
MUT = WT[:MUTPOS] + "R" + WT[MUTPOS + 1:]
assert WT[MUTPOS] == "L" and MUT[MUTPOS] == "R"
BASES = "ACGT"

cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval(); m.use_contact_pred_head = False
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)

# ── stage-capture hooks ──
cap = {}
def hk(name):
    def f(mod, inp, out):
        cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return f
m.backbone.register_forward_hook(hk("esm"))
if getattr(m, "residue_moe", None) is not None: m.residue_moe.register_forward_hook(hk("moe"))
m.projection.register_forward_hook(hk("proj"))

# ── condition-4 monkeypatch: 2D additive attention bias (Lq central x residue MUTPOS) ──
FORCE = {"on": False}
_orig = ContactCrossAttention.forward
def patched(self, q, esm, tokens, key_valid, contact_bias=None, fam_ctx=None):
    if not FORCE["on"]:
        return _orig(self, q, esm, tokens, key_valid, contact_bias, fam_ctx)
    # add a strong 2D bias: central motif positions attend to the mutated residue
    B, Lk = key_valid.shape
    b2 = torch.zeros(B, 1, Lk, device=q.device)
    b2[:, 0, MUTPOS] = 8.0                                     # boost residue MUTPOS
    if contact_bias is None: contact_bias = torch.zeros(B, Lk, device=q.device)
    # fold the per-residue part in, then stash 2D for the central-position rows
    self._force2d = b2
    return _orig(self, q, esm, tokens, key_valid, contact_bias + b2[:, 0], fam_ctx)
ContactCrossAttention.forward = patched

def toks(seq): return torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)

@torch.no_grad()
def run(seq, contact_override=None):
    cap.clear()
    t = toks(seq); dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
    gl, pl, aux = m(t, dm, fi, contact_override=contact_override)
    return dict(esm=cap["esm"][0], moe=cap.get("moe", cap["esm"])[0],
                proj=cap["proj"][0], logits=pl[0].float(),
                attn=(aux["attn"][0].float() if "attn" in aux else None))

def cos(a, b): return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
def l2(a, b): return float((a - b).norm())

def ic_core(p):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= 0.2)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])

def consensus(P, a, b): return "".join(BASES[i] for i in P[:, a:b].argmax(0))

def central_logodds(z, a, b):
    """signed log-odds at the E-box central dinucleotide (positions a+2,a+3 of a 6bp core)."""
    if b - a < 6: return None
    zc = z[:, a:a + 6]                                          # (4,6) CANNTG frame
    out = {}
    for pos, lbl in [(2, "N1"), (3, "N2")]:
        lo = {base: float(zc[i, pos] - zc[:, pos].mean()) for i, base in enumerate(BASES)}
        out[lbl] = lo
    return out

def report(cond, wtc=None):
    fw = run(WT, wtc); fm = run(MUT, wtc)
    # per-stage WT/MUT representation diffs
    print(f"\n### condition: {cond}")
    for st in ["esm", "moe", "proj"]:
        A, Bp = fw[st], fm[st]
        if A.dim() == 2:                                        # (L,D) per-residue
            pooled_c = cos(A.mean(0), Bp.mean(0)); pooled_l = l2(A.mean(0), Bp.mean(0))
            site_c = cos(A[MUTPOS], Bp[MUTPOS]); site_l = l2(A[MUTPOS], Bp[MUTPOS])
            nb = slice(max(0, MUTPOS - 2), MUTPOS + 3)
            nb_c = cos(A[nb].flatten(), Bp[nb].flatten())
            print(f"  {st:5} pooled cos={pooled_c:.3f} L2={pooled_l:.2f} | site cos={site_c:.3f} L2={site_l:.2f} | nbhd cos={nb_c:.3f}")
        else:                                                  # (D,) pooled vector
            print(f"  {st:5} vector cos={cos(A, Bp):.3f} L2={l2(A, Bp):.2f}")
    if fw["attn"] is not None:
        print(f"  attn  cos={cos(fw['attn'], fm['attn']):.3f} L2={l2(fw['attn'], fm['attn']):.3f}")
    zw, zm = fw["logits"].cpu().numpy(), fm["logits"].cpu().numpy()
    Pw, Pm = F.softmax(fw["logits"], 0).cpu().numpy(), F.softmax(fm["logits"], 0).cpu().numpy()
    a, b = ic_core(Pw)
    zc = (zm - zm.mean(0)) - (zw - zw.mean(0))                  # centered logit delta (4,42)
    dpred = 1 - np.corrcoef(Pw[:, a:b].ravel(), Pm[:, a:b].ravel())[0, 1]
    print(f"  WT core [{a}:{b}] cons={consensus(Pw,a,b)}  MUT cons={consensus(Pm,a,b)}")
    print(f"  centered |Δlogit| core mean={np.abs(zc[:,a:b]).mean():.3f} max={np.abs(zc[:,a:b]).max():.2f}  |  Δpred(1-corr)={dpred:.4f}")
    cl = central_logodds(zw, a, b); cm = central_logodds(zm, a, b)
    if cl:
        for k in cl:
            dG = {base: cm[k][base] - cl[k][base] for base in BASES}
            top = max(dG, key=dG.get)
            print(f"  central {k}: WT top={max(cl[k],key=cl[k].get)} MUT top={max(cm[k],key=cm[k].get)}  ΔlogOdds(C/G swap): "
                  f"C={dG['C']:+.2f} G={dG['G']:+.2f}")
    return dpred

print(f"MyoD1 WT vs L112R (DBD pos {MUTPOS})  |  central-base switch target: WT CAGCTG(GC) -> MUT CACGTG(CG)")
# oracle 1D contact: recognition_residues [0..14] -> per-residue prior
rr = json.load(open("data/contact_maps/recognition_residues_v23.json")).get("str_700", list(range(15)))
oracle = torch.zeros(1, len(WT), device=dev)
for i in rr:
    if i < len(WT): oracle[0, i] = 1.0

report("1_normal")
report("2_shared_gate (Δpred on shared WT core)")            # logits are span-independent; registration control
report("3_oracle_1D_contact", wtc=oracle)
FORCE["on"] = True; report("4_force_L112_to_attention"); FORCE["on"] = False
print("\n(see docs/MUTATION_EXPERIMENTS.md Phase-2 for the decision reading)")
