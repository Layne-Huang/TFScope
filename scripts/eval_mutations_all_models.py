"""Run ALL trained TFScope versions on two mutation cases:
  MyoD1 L112R : WT muscle E-box (CC/GC central) -> canonical CACGTG (CG central).
  KLF4  K19Q  : C2H2 GC-box / CACCC recognition; report WT vs MUT consensus shift.
Reports predicted consensus for WT and MUT, plus the MyoD1 central E-box dinucleotide.
family_id per model taken from its own parquet (bHLH for MyoD1, C2H2 for KLF4).
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
B = np.array(list("ACGT")); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CK = "/data1/leihuang/project/TFScope/checkpoints"

MYOD1_WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MYOD1_MUT = "RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
KLF4_WT = "HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
KLF4_MUT = KLF4_WT[:18] + "Q" + KLF4_WT[19:]

MODELS = [  # name, ckpt_dir, parquet
 ("combined",        f"{CK}/v19_combined_fm_deeppbs_contact/rag_seed42",  "tf_pwm_deeppbs_only_canon_trim"),
 ("nocontact",       f"{CK}/v19_combined_fm_deeppbs_nocontact/rag_seed42", "tf_pwm_deeppbs_only_canon_trim"),
 ("dimerdup",        f"{CK}/v19_combined_dimerdup/rag_seed42",             "tf_pwm_deeppbs_only_canon_trim"),
 ("coarse12_contact",f"{CK}/v19_combined_coarse12_contact/rag_seed42",     "tf_pwm_deeppbs_coarse"),
 ("coarse12_matched",f"{CK}/v19_combined_coarse12_matched/rag_seed42",     "tf_pwm_deeppbs_coarse"),
 ("semfam34",        f"{CK}/v19_combined_semfam34_contact/rag_seed42",     "tf_pwm_deeppbs_rebin34"),
 ("semfam34_fixed",  f"{CK}/v19_combined_semfam34_contact_fixed/rag_seed42","tf_pwm_deeppbs_rebin34"),
 ("semfam46",        f"{CK}/v19_combined_semfam46_contact/rag_seed42",     "tf_pwm_deeppbs_famv2"),
 ("dual_family",     f"{CK}/v19_combined_dual_family_rebin34/rag_seed42",  "tf_pwm_deeppbs_rebin34"),
 ("v23_nchain",      f"{CK}/v23_nchain/nchain_v23_seed42",                 "tf_pwm_training_v23"),
 ("v23_fulldata",     f"{CK}/v23_fulldata/nchain_v23_full_seed42",         "tf_pwm_training_v23"),
 ("v24_contact",      f"{CK}/v24_contact/contact_v24_seed42",               "tf_pwm_training_v23"),
]

def fid_for(parquet, gene):
    df = pd.read_parquet(f"data/processed/{parquet}.parquet")
    for col in ("filename", "gene_symbol"):   # v23 uses seq_/str_ filenames -> gene_symbol
        if col in df.columns:
            sub = df[df[col].astype(str).str.upper().str.contains(gene.upper())]
            if len(sub):
                return int(sub["family_id"].mode().iloc[0])
    return 0

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
def predict(m, seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()
    return "".join(B[pwm[:, :max(4, int((gate > 0.5).sum()))].argmax(0)])

def ebox(con):
    for i in range(len(con) - 5):
        w = con[i:i+6]
        if w[0]=="C" and w[1]=="A" and w[4]=="T" and w[5]=="G": return w[2:4]
    return None

out = {}
for name, ckpt_dir, parquet in MODELS:
    m = load(ckpt_dir)
    fb, fc = fid_for(parquet, "MYOD1"), fid_for(parquet, "KLF4")
    cw, cm = predict(m, MYOD1_WT, fb), predict(m, MYOD1_MUT, fb)
    ew, em = ebox(cw), ebox(cm)
    sw = "G recovered" if (em and "G" in em and em != ew) else "no switch"
    kw, km = predict(m, KLF4_WT, fc), predict(m, KLF4_MUT, fc)
    out[name] = {"myod1": {"wt": cw, "mut": cm, "wt_center": ew, "mut_center": em, "switch": sw},
                 "klf4": {"wt": kw, "mut": km, "changed": kw != km}}
    print(f"done {name}", flush=True)
    del m; torch.cuda.empty_cache()

json.dump(out, open("results/design_case_study/mutations_all_models.json", "w"), indent=1)
print("\n=== MyoD1 L112R (bHLH): WT->CACCTG/CAGCTG (CC/GC) ; L112R->CACGTG (CG) ===")
print(f"{'model':<18}{'WT consensus':<16}{'WT-c':<5}{'MUT consensus':<16}{'MUT-c':<5}{'switch'}")
print("-"*70)
for n, r in out.items():
    d = r["myod1"]; print(f"{n:<18}{d['wt'][:15]:<16}{str(d['wt_center']):<5}{d['mut'][:15]:<16}{str(d['mut_center']):<5}{d['switch']}")
print("\n=== KLF4 K19Q (C2H2): GC-box/CACCC; WT vs MUT consensus ===")
print(f"{'model':<18}{'WT consensus':<18}{'MUT consensus':<18}{'changed'}")
print("-"*64)
for n, r in out.items():
    d = r["klf4"]; print(f"{n:<18}{d['wt'][:17]:<18}{d['mut'][:17]:<18}{'yes' if d['changed'] else 'no'}")
print("\nsaved results/design_case_study/mutations_all_models.json")
