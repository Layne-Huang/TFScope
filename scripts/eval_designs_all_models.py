"""Run ALL trained TFScope versions on the 4 experimentally-characterized de novo
designs (DBP005/006/009/035; target core CACAT -> 'CAC core'). For each model report:
  - predicted consensus (gated core)
  - CAC recovered? (consensus contains 'CAC' or revcomp 'GTG')
  - core-r vs the 14-bp experimental single-substitution preference
family_id per model = its own parquet's family_id for the design's top natural homolog.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel

DESIGNS = ["DBP005", "DBP006", "DBP009", "DBP035"]
SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
         "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}; B = np.array(list("ACGT"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
CK = "/data1/leihuang/project/TFScope/checkpoints"

MODELS = [  # name, ckpt_dir, parquet (for family_id lookup of the design's homolog)
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

xls_cols = pd.ExcelFile(XLS).parse(SHEET["DBP005"], nrows=1).columns.tolist(); VALCOL = xls_cols[-1]
WTOV = {"DBP006": 0.1202}   # WT-row override where the sheet's WT value is missing/wrong
def exp_pref(design):
    # IMPORTANT: in this assay a LOWER Median PE/FITC value = STRONGER binding.
    # Preference strength is therefore INVERSELY proportional to the value
    # (1/value), NOT the value itself. Using the value directly (the old code)
    # inverted every design's experimental target.
    df_all = pd.ExcelFile(XLS).parse(SHEET[design])
    wt = df_all[df_all.position.astype(str) == "WT"]
    wtv = WTOV.get(design) or (float(wt[VALCOL].iloc[0]) if len(wt) else 1.0)
    df = df_all[df_all.position.astype(str).str.isdigit()].copy()
    L = int(df.position.astype(int).max()); P = np.full((4, L), 1e-6, np.float32)
    for p in range(1, L + 1):
        sub = df[df.position.astype(int) == p]; ob = str(sub.original_base.iloc[0])
        if ob in B2I: P[B2I[ob], p - 1] = 1.0 / max(wtv, 1e-3)              # WT base strength
        for _, r in sub.iterrows():
            nb = str(r["new_base"])
            if nb in B2I: P[B2I[nb], p - 1] = 1.0 / max(float(r[VALCOL]), 1e-3)  # low value -> strong
    return P / P.sum(0, keepdims=True)
prefs = {d: exp_pref(d) for d in DESIGNS}

def fam_lookup(parquet):
    df = pd.read_parquet(f"data/processed/{parquet}.parquet")
    cols = [c for c in ("filename", "gene_symbol") if c in df.columns]
    def fid_for(gene):
        for c in cols:      # v23 uses seq_/str_ filenames -> fall back to gene_symbol
            sub = df[df[c].astype(str).str.upper().str.contains(gene.upper())]
            if len(sub):
                return int(sub["family_id"].mode().iloc[0])
        return None
    return fid_for

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
    return pwm[:, :max(4, int((gate > 0.5).sum()))]

def has_cac(con): return ("CAC" in con) or ("GTG" in con)  # CAC core or its revcomp

out = {}
print(f"{'model':<18}{'design':<8}{'consensus':<18}{'CAC?':<6}{'core-r':>7}")
print("-"*60)
for name, ckpt_dir, parquet in MODELS:
    m = load(ckpt_dir); fid_for = fam_lookup(parquet)
    rec = {}
    for d in DESIGNS:
        donor = str(e2[d].get("top_donor", "POU2F1"))
        fid = fid_for(donor) or fid_for("POU2F1") or fid_for("PHA") or 4
        pv = predict(m, e2[d]["prot_seq"], fid)
        con = "".join(B[pv.argmax(0)])
        aligned, cols, _ = aligned_cols(pv, prefs[d]); dd = panel(prefs[d], aligned, cols)
        r = round(float(dd["r"]), 3) if dd else None
        rec[d] = {"consensus": con, "cac": has_cac(con), "core_r": r, "fid": fid}
        print(f"{name:<18}{d:<8}{con[:17]:<18}{'YES' if has_cac(con) else 'no':<6}{(r if r is not None else float('nan')):>7.3f}")
    rec["mean_core_r"] = round(float(np.nanmean([rec[d]["core_r"] for d in DESIGNS if rec[d]["core_r"] is not None])), 3)
    rec["cac_count"] = int(sum(rec[d]["cac"] for d in DESIGNS))
    out[name] = rec
    print(f"{'  -> '+name:<18}{'':8}{'':18}{str(rec['cac_count'])+'/4':<6}{rec['mean_core_r']:>7.3f}\n")

json.dump(out, open("results/design_case_study/designs_all_models.json", "w"), indent=1)
print("saved results/design_case_study/designs_all_models.json")
print("\nCAC-recovery + mean core-r leaderboard:")
for n, r in sorted(out.items(), key=lambda kv: -kv[1]["mean_core_r"]):
    print(f"  {n:<18} CAC {r['cac_count']}/4   mean core-r {r['mean_core_r']}")
