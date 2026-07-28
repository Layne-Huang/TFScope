"""MyoD1 WT vs L12R specificity-switch test with the `combined` model (no-RAG, learned-10, contact).

WT MyoD1 bHLH binds the muscle E-box  CASSTG  (CACCTG / CAGCTG).
The basic-region L->R substitution (local resnum 12) switches it to the
canonical E-box  CACGTG.  Question: does TFScope's predicted PWM shift the
central E-box dinucleotide CC/GC -> CG from sequence alone?
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
WT  = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MUT = "RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
FID = 3                       # bHLH in the 10-family scheme
B = np.array(list("ACGT"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
model = TFScopeModel(cfg).to(dev).eval()
model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

def predict(seq):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([FID], dtype=torch.long, device=dev)
    with torch.no_grad():
        gl, pl, _ = model(tok, dm, fi, retrieved_pwms=None, retrieved_masks=None,
                          retrieved_sims=None, recog_prior=None)
        gate = gl.sigmoid()[0].cpu().numpy()
        pwm = F.softmax(pl, 1)[0].cpu().numpy()          # (4, 20)
    L = max(4, int((gate > 0.5).sum()))
    return pwm, pwm[:, :L], L

def find_ebox(consensus):
    """Locate a CANNTG E-box; return (start, central_dinucleotide) or (None,None)."""
    for i in range(len(consensus) - 5):
        w = consensus[i:i+6]
        if w[0] == 'C' and w[1] == 'A' and w[4] == 'T' and w[5] == 'G':
            return i, w[2:4]
    return None, None

pw_wt, c_wt, L_wt = predict(WT)
pw_mt, c_mt, L_mt = predict(MUT)
con_wt = ''.join(B[c_wt.argmax(0)]); con_mt = ''.join(B[c_mt.argmax(0)])

print(f"WT  consensus (len {L_wt}): {con_wt}")
print(f"MUT consensus (len {L_mt}): {con_mt}")
s_wt, nn_wt = find_ebox(con_wt); s_mt, nn_mt = find_ebox(con_mt)
print(f"\nE-box (CANNTG) central dinucleotide:")
print(f"  WT : {'CA['+nn_wt+']TG' if nn_wt else 'no clean E-box in consensus ('+con_wt+')'}")
print(f"  MUT: {'CA['+nn_mt+']TG' if nn_mt else 'no clean E-box in consensus ('+con_mt+')'}")
print(f"  Target biology:  WT -> CC/GC (CACCTG/CAGCTG)   MUT -> CG (CACGTG)")

r_full = pearsonr(pw_wt.flatten(), pw_mt.flatten())[0]
diff = np.abs(pw_wt - pw_mt).sum(0)
print(f"\nWT vs MUT full-PWM Pearson r = {r_full:.4f}  (1.0 = no change at all)")
print(f"per-position L1 diff: max={diff.max():.3f} mean={diff.mean():.3f}  argmax pos={int(diff.argmax())}")
# where (if anywhere) the central E-box base shifted toward G
if s_wt is not None and s_mt is not None:
    cpos_wt = s_wt + 2  # first 'N' of CANNTG (0-based within core)
    cpos_mt = s_mt + 2
    print(f"\nCentral-position base probabilities (the position that should gain G):")
    print(f"  WT  core pos {cpos_wt}: " + " ".join(f"{b}={c_wt[j,cpos_wt]:.2f}" for j,b in enumerate('ACGT')))
    print(f"  MUT core pos {cpos_mt}: " + " ".join(f"{b}={c_mt[j,cpos_mt]:.2f}" for j,b in enumerate('ACGT')))

os.makedirs("results/myod1_mut", exist_ok=True)
np.savez("results/myod1_mut/combined_wt_mut_pwms.npz",
         pwm_wt=pw_wt, pwm_mut=pw_mt, core_wt=c_wt, core_mut=c_mt,
         con_wt=con_wt, con_mut=con_mt)

# ── overlay logos ──────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import logomaker, pandas as pd
    fig, axes = plt.subplots(2, 1, figsize=(max(5, L_wt*0.5), 3.4))
    for ax, (core, title) in zip(axes, [(c_wt, f"WT MyoD1   {con_wt}"),
                                         (c_mt, f"L12R mutant   {con_mt}")]):
        ppm = np.clip(core.T, 1e-8, 1); ppm = ppm / ppm.sum(1, keepdims=True)
        ic = (ppm * (np.log2(ppm) - np.log2(0.25))).sum(1, keepdims=True).clip(0)  # bits
        info = pd.DataFrame(ppm * ic, columns=list("ACGT"))
        logomaker.Logo(info, ax=ax, color_scheme="classic")
        ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([0, 1, 2])
        ax.set_ylabel("bits"); ax.set_title(title, fontsize=9, fontweight="bold")
    fig.suptitle(f"TFScope (combined) — MyoD1 WT vs L12R    WT/MUT PWM r={r_full:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/myod1_mut/combined_wt_vs_mut_logo.png", dpi=160, bbox_inches="tight")
    fig.savefig("results/myod1_mut/combined_wt_vs_mut_logo.pdf", bbox_inches="tight")
    print("\nSaved results/myod1_mut/combined_wt_vs_mut_logo.png/.pdf")
except Exception as e:
    print(f"\n[logo skipped: {e}] PWMs saved to results/myod1_mut/combined_wt_mut_pwms.npz")
