import os,sys,json,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"src"); sys.path.insert(0,"pwm_rosetta")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
MODELS=[("v17 (robust-RAG)",f"{ROOT}/deeppbs_v17_robustRAG/ckpt_best.pt"),
        ("v14_noRAG (trained w/o retrieval)",f"{ROOT}/deeppbs_v14_noRAG/ckpt_best.pt")]
WT ="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
MUT=WT[:18]+"Q"+WT[19:]; MUTSITE=18
dev="cuda" if torch.cuda.is_available() else "cpu"

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
        with torch.no_grad(): _,pl,_=m(tok,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None)
        return cap["a"][0],F.softmax(pl,1)[0].cpu().numpy()
    return run

data={}
for name,ck in MODELS:
    run=load(ck); aW,pW=run(WT); aM,pM=run(MUT)
    data[name]=dict(aW=aW,aM=aM,pW=pW,pM=pM)
    np.savez("results/klf4_attn_noRAG/raw.npz" if "noRAG" in name else "results/klf4_attn/raw.npz",aW=aW,aM=aM)
    print(f"{name}: attn r(WT,MUT)={pearsonr(aW.flatten(),aM.flatten())[0]:.4f}  "
          f"attn->mutres WT={aW[:,MUTSITE].sum():.3f} MUT={aM[:,MUTSITE].sum():.3f}")

# ---- figure: 2 model rows x (WT | MUT | per-residue total + entropy) ----
fig=plt.figure(figsize=(16,7))
gs=fig.add_gridspec(2,3,width_ratios=[1,1,1.1],hspace=0.45,wspace=0.3)
for r,(name,d) in enumerate(data.items()):
    aW,aM=d["aW"],d["aM"]; L=aW.shape[1]
    vmax=max(aW.max(),aM.max())
    for c,(M,t) in enumerate([(aW,"WT"),(aM,"K409Q (mut)")]):
        ax=fig.add_subplot(gs[r,c])
        im=ax.imshow(M,aspect="auto",cmap="magma",vmin=0,vmax=vmax)
        ax.set_title(f"{name}\n{t} cross-attention",fontsize=9)
        ax.set_xlabel("protein residue (DBD)"); ax.set_ylabel("PWM position")
        ax.axvline(MUTSITE,color="cyan",ls="--",lw=1.2)
        ax.text(MUTSITE+1,0.5,"K409Q\nsite",color="cyan",fontsize=7,va="top")
        plt.colorbar(im,ax=ax,fraction=0.046,label="attn weight")
    # right: per-residue attention mass + collapse metric
    ax=fig.add_subplot(gs[r,2])
    tot_w=aW.sum(0)/aW.shape[0]; tot_m=aM.sum(0)/aM.shape[0]
    ax.bar(np.arange(L),tot_w,color="steelblue",alpha=.8,label="WT")
    ax.plot(np.arange(L),tot_m,color="crimson",lw=.9,label="K409Q")
    ax.axvline(MUTSITE,color="cyan",ls="--",lw=1.2)
    # collapse: per-PWM-position entropy of attention (low=collapsed)
    p=aW/ (aW.sum(1,keepdims=True)+1e-9); ent=-(p*np.log(p+1e-12)).sum(1).mean()
    maxent=np.log(L)
    ax.set_title(f"mean attn / residue  (collapse: H={ent:.2f} / {maxent:.2f} nats)",fontsize=9)
    ax.set_xlabel("protein residue (DBD)"); ax.set_ylabel("mean attn weight")
    ax.legend(fontsize=7,loc="upper left")
    # annotate top residues
    for i in np.argsort(-tot_w)[:3]:
        ax.annotate(f"{WT[i]}{i+1}",(i,tot_w[i]),fontsize=7,ha="center",va="bottom")
fig.suptitle("KLF4 PWM-head cross-attention: same degenerate collapse with and without RAG\n"
             "(stripes constant down every PWM row = no position-specific recognition; zero mass at K409Q site)",
             fontsize=11)
os.makedirs("results/klf4_attn",exist_ok=True)
out="results/klf4_attn/cross_attn_compare.png"
fig.savefig(out,dpi=140,bbox_inches="tight"); print("\nSaved",out)
