#!/usr/bin/env python
"""Does the v18 contact-aware cross-attention detect specificity-switching mutations?

Runs KLF4 WT vs K409Q and MyoD WT vs L122R through a model (v17 degenerate baseline
or v18a/v18b), de-novo (retrieved=None so only the protein pathway can create a
WT/mutant difference), and reports the v18a success criteria:
  - attention WT-vs-mut Pearson r  (<1 = mutation-sensitive)
  - output PWM WT-vs-mut Pearson r (<1 = prediction changes)
  - attention mass on the mutated residue (>0 = the head reads the causal residue)
  - row-constancy / entropy (rank-1 collapse diagnostics)
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
CKPTS = {"v17": f"{CKPT_ROOT}/deeppbs_v17_200ep/ckpt_best.pt",
         "v18a": f"{CKPT_ROOT}/deeppbs_v18a_attnrepair/ckpt_best.pt",
         "v18b": f"{CKPT_ROOT}/deeppbs_v18b_contact/ckpt_best.pt",
         "v18a_noRAG": f"{CKPT_ROOT}/deeppbs_v18a_noRAG/ckpt_best.pt"}
dev = "cuda" if torch.cuda.is_available() else "cpu"

CASES = {
    "KLF4": dict(fam=0, mutsite=18,
        wt="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH",
        mut_aa="Q"),
    "MyoD": dict(fam=3, mutsite=11,
        wt="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA",
        mut_aa="R"),
}


def load(ck):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ck), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"], strict=False)
    is_v18 = getattr(cfg, "pwm_head_v18", False)
    cap = {}
    if not is_v18:   # v17: hook the legacy cross-attn
        m.pwm_head.cross_attn.register_forward_hook(
            lambda mo, i, o: cap.__setitem__("a", o[1].detach().cpu().numpy()))
    return m, is_v18, cap


def run(m, is_v18, cap, seq, fam):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fam], device=dev)
    with torch.no_grad():
        _, pl, _ = m(tok, dm, fi, retrieved_pwms=None, retrieved_masks=None,
                     retrieved_sims=None, recog_prior=None)
    attn = (m.pwm_head._last_attn[0].cpu().numpy() if is_v18 else cap["a"][0])  # (Lq, Lk)
    pwm = F.softmax(pl, 1)[0].cpu().numpy()                                     # (4, Lq)
    return attn, pwm


def metrics(aW, aM, pW, pM, mutsite):
    L = aW.shape[1]
    massW, massM = aW[:, mutsite].sum(), aM[:, mutsite].sum()
    # row-constancy: mean pairwise corr between PWM-position attention rows
    C = np.corrcoef(aW); iu = np.triu_indices(C.shape[0], 1)
    rowconst = float(np.nanmean(C[iu]))
    p = aW / (aW.sum(1, keepdims=True) + 1e-9)
    ent = float(-(p * np.log(p + 1e-12)).sum(1).mean())
    return dict(attn_r=pearsonr(aW.flatten(), aM.flatten())[0],
                pwm_r=pearsonr(pW.flatten(), pM.flatten())[0],
                mass_wt=float(massW), mass_mut=float(massM),
                rowconst=rowconst, ent=ent, maxent=float(np.log(L)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v17", "v18a"])
    args = ap.parse_args()
    for name in args.models:
        ck = CKPTS[name]
        if not os.path.exists(ck):
            print(f"[skip] {name}: not found"); continue
        m, is_v18, cap = load(ck)
        print(f"\n========== {name} ({'v18 contact head' if is_v18 else 'legacy cross-attn'}) ==========")
        for tf, c in CASES.items():
            mut = c["wt"][:c["mutsite"]] + c["mut_aa"] + c["wt"][c["mutsite"] + 1:]
            aW, pW = run(m, is_v18, cap, c["wt"], c["fam"])
            aM, pM = run(m, is_v18, cap, mut, c["fam"])
            d = metrics(aW, aM, pW, pM, c["mutsite"])
            print(f"  {tf} (mut res{c['mutsite']+1} {c['wt'][c['mutsite']]}->{c['mut_aa']}):")
            print(f"    output PWM  WT-vs-mut r = {d['pwm_r']:.4f}   (<1 ⇒ prediction changes)")
            print(f"    attention   WT-vs-mut r = {d['attn_r']:.4f}   (<1 ⇒ attn mutation-sensitive)")
            print(f"    attn mass on mutated residue: WT={d['mass_wt']:.3f}  MUT={d['mass_mut']:.3f}  (>0 ⇒ reads it)")
            print(f"    row-constancy r = {d['rowconst']:.3f}   entropy = {d['ent']:.2f}/{d['maxent']:.2f}")


if __name__ == "__main__":
    main()
