"""Run the per-protein-text model on (1) the 4 de novo designs and (2) the
MyoD1/KLF4 mutation cases, using its text-conditioning mechanism:
each query is conditioned on a PubMedBERT text embedding appended to the frozen
family-vector buffer and addressed by an offset family_id.
  - designs : text = nearest natural homolog (top_donor) "TF {gene} from {org}, {fam} family."
  - MyoD1   : "Transcription factor MYOD1 from Homo sapiens, bHLH family."
  - KLF4    : "Transcription factor KLF4 from Homo sapiens, C2H2 family."
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel
B = np.array(list("ACGT")); B2I = {"A":0,"C":1,"G":2,"T":3}; dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt"

DESIGNS = ["DBP005","DBP006","DBP009","DBP035"]
SHEET = {"DBP005":"Extended_Data_Figure_1_C_DBP005","DBP006":"Extended_Data_Figure_1_D_DBP006",
         "DBP009":"Extended_Data_Figure_1_E_DBP009","DBP035":"Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
e2 = {e["name"]:e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
MYOD1_WT="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MYOD1_MUT="RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
KLF4_WT="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
KLF4_MUT=KLF4_WT[:18]+"Q"+KLF4_WT[19:]

parq = pd.read_parquet("data/processed/tf_pwm_combined_perprot.parquet")
gmeta = {}
for r in parq.itertuples():
    g = str(r.gene_symbol).upper()
    if g not in gmeta: gmeta[g] = (str(r.organism), str(r.family_name))

xls_cols = pd.ExcelFile(XLS).parse(SHEET["DBP005"], nrows=1).columns.tolist(); VALCOL = xls_cols[-1]
def exp_pref(design):
    df = pd.ExcelFile(XLS).parse(SHEET[design]); df = df[df.position.astype(str).str.isdigit()].copy()
    L = int(df.position.astype(int).max()); P = np.full((4,L),1e-6,np.float32)
    for p in range(1,L+1):
        sub=df[df.position.astype(int)==p]; ob=str(sub.original_base.iloc[0])
        if ob in B2I: P[B2I[ob],p-1]=1.0
        for _,r in sub.iterrows():
            nb=str(r["new_base"])
            if nb in B2I: P[B2I[nb],p-1]=max(0.0,float(r[VALCOL]))
    return P/P.sum(0,keepdims=True)
prefs = {d:exp_pref(d) for d in DESIGNS}

cfg = TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CKPT),"config.json"))).items():
    if hasattr(cfg,k):
        try: setattr(cfg,k,type(getattr(cfg,k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT,map_location=dev,weights_only=False)["model"],strict=False)
N0 = m.moe.family_embed.vectors.shape[0]

_bert=None
def tvec(s):
    global _bert
    from transformers import AutoTokenizer, AutoModel
    if _bert is None:
        mid="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"; CD="/data1/leihuang/.cache"
        _bert=(AutoTokenizer.from_pretrained(mid,cache_dir=CD,local_files_only=True),
               AutoModel.from_pretrained(mid,cache_dir=CD,local_files_only=True).to(dev).eval())
    tk,bt=_bert
    with torch.no_grad():
        e={k:v.to(dev) for k,v in tk(s,return_tensors="pt",truncation=True,max_length=128).items()}
        return F.normalize(bt(**e).last_hidden_state[:,0,:],dim=-1)[0]

@torch.no_grad()
def predict(seq, text, ML=20):
    fe = m.moe.family_embed
    fe.vectors = torch.cat([fe.vectors[:N0], tvec(text)[None].to(fe.vectors.dtype).to(dev)], 0)
    fid = fe.vectors.shape[0]-1
    t=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([fid],device=dev)
    gl,pl,_=m(t,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None,recog_prior=None)
    gate=gl.sigmoid()[0].cpu().numpy(); pwm=F.softmax(pl,1)[0].cpu().numpy()
    L=max(4,int((gate>0.5).sum()))
    return pwm[:,:L], "".join(B[pwm[:,:L].argmax(0)])
def ebox(con):
    for i in range(len(con)-5):
        w=con[i:i+6]
        if w[0]=="C" and w[1]=="A" and w[4]=="T" and w[5]=="G": return w[2:4]
    return None

print("=== per-protein-text on 4 designs (CAC core = CACAT target) ===")
print(f"{'design':<8}{'homolog':<10}{'consensus':<18}{'CAC?':<6}{'core-r':>7}")
rs=[]; cacn=0
for d in DESIGNS:
    hg=str(e2[d]["top_donor"]); org,fam=gmeta.get(hg.upper(),("Homo sapiens","Homeodomain"))
    pv,con=predict(e2[d]["prot_seq"], f"Transcription factor {hg} from {org}, {fam} family.")
    al,cols,_=aligned_cols(pv,prefs[d]); dd=panel(prefs[d],al,cols); r=float(dd["r"]) if dd else float("nan")
    cac = ("CAC" in con) or ("GTG" in con); rs.append(r); cacn+=int(cac)
    print(f"{d:<8}{hg:<10}{con[:17]:<18}{'YES' if cac else 'no':<6}{r:>7.3f}")
print(f"  -> per-protein-text: CAC {cacn}/4 | mean core-r {np.nanmean(rs):.3f}")

print("\n=== per-protein-text on mutations ===")
_,mw=predict(MYOD1_WT,"Transcription factor MYOD1 from Homo sapiens, bHLH family.")
_,mm=predict(MYOD1_MUT,"Transcription factor MYOD1 from Homo sapiens, bHLH family.")
print(f"MyoD1  WT {mw[:12]:<14} center {ebox(mw)}   MUT {mm[:12]:<14} center {ebox(mm)}   switch={'G recovered' if (ebox(mm) and 'G' in (ebox(mm) or '') and ebox(mm)!=ebox(mw)) else 'no switch'}")
_,kw=predict(KLF4_WT,"Transcription factor KLF4 from Homo sapiens, C2H2 family.")
_,km=predict(KLF4_MUT,"Transcription factor KLF4 from Homo sapiens, C2H2 family.")
print(f"KLF4   WT {kw[:12]:<14}          MUT {km[:12]:<14}          changed={'yes' if kw!=km else 'no'}")
