"""Does the family label matter for semfam34? Hold the protein fixed, sweep
family_id across all 34 semantic families, and measure how much the predicted
PWM/consensus changes. Compare with the canonical combined model (learned-10).
If output is ~invariant to family_id -> family conditioning is vestigial.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
B = np.array(list("ACGT")); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CK = "/data1/leihuang/project/TFScope/checkpoints"

PROTEINS = {  # name: (seq, true_family_name)
 "MyoD1(bHLH)": ("RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA", "bHLH"),
 "KLF4(C2H2)":  ("HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH", "C2H2"),
}
FAMS34 = ['C2H2_short','C2H2_medium','C2H2_long','bHLH','Homeodomain','bZIP','Nuclear_Receptor',
 'Forkhead','ETS','AP2/ERF','HMG/SOX','T-box','RHD/NFkB','E2F/DP','MADS/SRF','GATA','RFX','STAT',
 'p53','MYB/SANT','PAX','Runt','Grainyhead/CP2','DMRT','TEA/TEAD','NF-Y/CBF','IRF','NDT80','MBD',
 'GCM','THAP','HTH','ARID/SAND','Other']

def load(ckpt_dir):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, "ckpt_best.pt"), map_location=dev, weights_only=False)["model"], strict=False)
    return m

@torch.no_grad()
def predict_full(m, seq, fid, ML=20):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    L = max(4, int((gate > 0.5).sum()))
    return pwm[:, :ML], pwm[:, :L]   # fixed-window PWM (for divergence), gated core (for consensus)

def sweep(model_name, ckpt_dir, nfam, fam_names):
    m = load(ckpt_dir)
    print(f"\n########## {model_name}  (num_families={nfam}) ##########")
    for pname, (seq, truefam) in PROTEINS.items():
        cons = {}; pwms = {}
        for fid in range(nfam):
            pwm_fix, core = predict_full(m, seq, fid)
            cons[fid] = "".join(B[core.argmax(0)]); pwms[fid] = pwm_fix
        uniq = sorted(set(cons.values()))
        # mean L1 distance of each family's PWM vs the mean PWM across families
        stack = np.stack([pwms[f] for f in range(nfam)])
        meanp = stack.mean(0)
        spread = float(np.mean([np.abs(pwms[f] - meanp).sum(0).mean() for f in range(nfam)]))  # avg per-pos L1
        tf = fam_names.index(truefam) if truefam in fam_names else None
        print(f"\n  {pname}: {len(uniq)} distinct consensus across {nfam} families | mean per-pos L1 spread = {spread:.3f}")
        if tf is not None:
            print(f"    true family [{truefam} id={tf}] -> {cons[tf]}")
        # show a few representative families
        for fid in sorted(set([0, 3, 4, 5, 6, nfam-1] + ([tf] if tf is not None else []))):
            if fid < nfam:
                tag = " <-- TRUE" if fid == tf else ""
                print(f"    fid {fid:2d} {fam_names[fid] if fid<len(fam_names) else '':<16} -> {cons[fid]}{tag}")

sweep("semfam34_contact", f"{CK}/v19_combined_semfam34_contact/rag_seed42", 34, FAMS34)
# contrast: canonical combined (learned-10) — does family matter there?
LEARNED10 = ['C2H2','?1','?2','bHLH','Homeodomain','bZIP','NR','?7','?8','Other']
sweep("combined(learned10)", f"{CK}/v19_combined_fm_deeppbs_contact/rag_seed42", 10, LEARNED10)
