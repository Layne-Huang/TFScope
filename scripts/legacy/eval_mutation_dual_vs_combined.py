"""MyoD1 L112R specificity switch: combined (learned-10) vs dual_family (rebin34).
WT muscle E-box (CC/GC central) -> L112R canonical CACGTG (CG central).
Recovering G at the central dinucleotide on the mutant = correct biology.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

WT  = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MUT = "RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
B = np.array(list("ACGT")); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPTS = {
 "combined (learned-10)":   "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
 "dual_family (rebin34)":   "/data1/leihuang/project/TFScope/checkpoints/v19_combined_dual_family_rebin34/rag_seed42/ckpt_best.pt",
}
FID = 3  # bHLH in both schemes

def load(ckpt):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"], strict=False)
    return m

def predict(m, seq, fid):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    with torch.no_grad():
        gl, pl, _ = m(tok, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
        gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    L = max(4, int((gate > 0.5).sum())); return pwm[:, :L]

def ebox(core):
    con = "".join(B[core.argmax(0)])
    for i in range(len(con) - 5):
        w = con[i:i+6]
        if w[0]=="C" and w[1]=="A" and w[4]=="T" and w[5]=="G": return con, w[2:4]
    return con, None

print(f"{'model':<26} {'WT consensus':<16} {'WT Ebox':<9} {'MUT consensus':<16} {'MUT Ebox':<9} verdict")
print("-"*100)
for name, ckpt in CKPTS.items():
    m = load(ckpt)
    cw, cm = predict(m, WT, FID), predict(m, MUT, FID)
    conw, ebw = ebox(cw); conm, ebm = ebox(cm)
    sw = f"CA[{ebw}]TG" if ebw else "no Ebox"; sm = f"CA[{ebm}]TG" if ebm else "no Ebox"
    verdict = "G recovered ✓" if (ebm and "G" in ebm and ebm != ebw) else "no switch"
    print(f"{name:<26} {conw[:15]:<16} {sw:<9} {conm[:15]:<16} {sm:<9} {verdict}")
print("\nbiology: WT -> CACCTG/CAGCTG (CC/GC central); L112R -> CACGTG (CG central). Recovering G = correct.")
