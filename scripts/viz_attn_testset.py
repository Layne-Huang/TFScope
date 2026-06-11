import os,sys,json,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"src")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
MODELS=[("v17 (robust-RAG)",f"{ROOT}/deeppbs_v17_robustRAG/ckpt_best.pt"),
        ("v14_noRAG",f"{ROOT}/deeppbs_v14_noRAG/ckpt_best.pt")]
dev="cuda" if torch.cuda.is_available() else "cpu"

df=pd.read_parquet("data/processed/tf_pwm_deeppbs_only.parquet").set_index("filename")
test=json.load(open("data/processed/splits/deeppbs_only/benchmark_no_val.json"))["test"]
# pick diverse test TFs (different families), one per family, distinct genes
seen=set(); picks=[]
for fn in test:
    if fn not in df.index: continue
    r=df.loc[fn]
    if isinstance(r,pd.DataFrame): r=r.iloc[0]
    key=(r["family_name"],r["gene_symbol"])
    if r["family_name"] in seen: continue
    seen.add(r["family_name"]); picks.append(fn)
    if len(picks)>=5: break
print("Selected test TFs:")
for fn in picks:
    r=df.loc[fn]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
    print(f"  {r['gene_symbol']:8s} family={r['family_name']:10s} DBD={r['dbd_end']-r['dbd_start']:3d}aa  {fn}")

def load(ck):
    cfg=TFScopeConfig()
    for k,v in json.load(open(os.path.join(os.path.dirname(ck),"config.json"))).items():
        if hasattr(cfg,k):
            try:setattr(cfg,k,type(getattr(cfg,k))(v))
            except:pass
    m=TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(ck,map_location=dev,weights_only=False)["model"],strict=False)
    cap={}
    m.pwm_head.cross_attn.register_forward_hook(lambda mo,i,o:cap.__setitem__("a",o[1].detach().cpu().numpy()))
    def run(seq):
        tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
        dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([0],device=dev)
        with torch.no_grad(): m(tok,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None)
        return cap["a"][0]
    return run

def collapse_metrics(A):
    # A: (Lq, Lk) attention rows sum to 1
    L=A.shape[1]
    p=A/(A.sum(1,keepdims=True)+1e-9)
    ent=-(p*np.log(p+1e-12)).sum(1).mean()            # mean per-position entropy (nats)
    # row-constancy: mean pairwise corr between PWM-position attention vectors (rank-1 -> ~1)
    C=np.corrcoef(A); iu=np.triu_indices(C.shape[0],1)
    rowconst=np.nanmean(C[iu])
    return ent, np.log(L), rowconst

for name,ck in MODELS:
    run=load(ck)
    print(f"\n=== {name} ===")
    maps={}
    for fn in picks:
        r=df.loc[fn]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
        seq=r["sequence"][r["dbd_start"]:r["dbd_end"]] if r["dbd_end"]>r["dbd_start"] else r["sequence"]
        A=run(seq); maps[r["gene_symbol"]]=A
        ent,maxent,rc=collapse_metrics(A)
        top=np.argsort(-A.sum(0))[:4]
        print(f"  {r['gene_symbol']:8s} L={A.shape[1]:3d}  entropy={ent:.2f}/{maxent:.2f}  "
              f"row-constancy r={rc:.2f}  top-residues={list(top+1)}")
    # cross-TF similarity: do DIFFERENT proteins attend to the same residues?
    genes=list(maps); print("  cross-TF attention-profile correlation (per-residue mass):")
    profs={g:maps[g].sum(0)/maps[g].shape[0] for g in genes}
    for i in range(len(genes)):
        for j in range(i+1,len(genes)):
            a,b=profs[genes[i]],profs[genes[j]]; L=min(len(a),len(b))
            print(f"    {genes[i]:8s} vs {genes[j]:8s}: r={pearsonr(a[:L],b[:L])[0]:+.2f}")

# figure for v17: heatmaps of the 5 test TFs
run=load(MODELS[0][1])
fig,axes=plt.subplots(1,len(picks),figsize=(3.2*len(picks),3.4))
for ax,fn in zip(axes,picks):
    r=df.loc[fn]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
    seq=r["sequence"][r["dbd_start"]:r["dbd_end"]] if r["dbd_end"]>r["dbd_start"] else r["sequence"]
    A=run(seq)
    im=ax.imshow(A,aspect="auto",cmap="magma")
    ax.set_title(f"{r['gene_symbol']} ({r['family_name']})",fontsize=9)
    ax.set_xlabel("residue"); ax.set_ylabel("PWM pos")
fig.suptitle("v17 PWM-head cross-attention on real test-set TFs: every TF collapses to a few fixed-residue stripes",fontsize=11)
fig.tight_layout(); out="results/klf4_attn/cross_attn_testset.png"
fig.savefig(out,dpi=140,bbox_inches="tight"); print("\nSaved",out)
