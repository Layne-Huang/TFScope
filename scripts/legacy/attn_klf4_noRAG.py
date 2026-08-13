import os,sys,json,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"src"); sys.path.insert(0,"pwm_rosetta")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CK="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG/ckpt_best.pt"
WT ="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
MUT=WT[:18]+"Q"+WT[19:]; MUTSITE=18  # K19 (0-based 18)
dev="cuda" if torch.cuda.is_available() else "cpu"
cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CK),"config.json"))).items():
    if hasattr(cfg,k):
        try:setattr(cfg,k,type(getattr(cfg,k))(v))
        except:pass
m=TFScopeModel(cfg).to(dev).eval(); m.load_state_dict(torch.load(CK,map_location=dev,weights_only=False)["model"],strict=False)
captured={}
def hook(mod,inp,out): captured["attn"]=out[1].detach().cpu().numpy()  # (B,Lq,Lk)
m.pwm_head.cross_attn.register_forward_hook(hook)
def run(seq):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([0],device=dev)
    with torch.no_grad():
        _,pl,_=m(tok,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None)
    return captured["attn"][0], F.softmax(pl,1)[0].cpu().numpy()  # (20,L),(4,20)
aW,pW=run(WT); aM,pM=run(MUT)
B="ACGT"
print(f"KLF4 cross-attn (PWM pos x protein residue). L={aW.shape[1]}  mut site=res19(idx18)")
print(f"output PWM WT vs MUT Pearson r = {pearsonr(pW.flatten(),pM.flatten())[0]:.4f}")
print(f"attention map WT vs MUT Pearson r = {pearsonr(aW.flatten(),aM.flatten())[0]:.4f}")
print(f"max|attn diff| = {np.abs(aM-aW).max():.4f}   mean|attn diff| = {np.abs(aM-aW).mean():.4f}")
print(f"\nAttention TO the mutated residue (res19) summed over PWM positions:")
print(f"  WT: {aW[:,MUTSITE].sum():.3f}   MUT: {aM[:,MUTSITE].sum():.3f}")
# which residues attended most (WT), and how mut changes
tot_w=aW.sum(0); tot_m=aM.sum(0)
print("\nTop-5 most-attended residues (WT) and change in MUT:")
for i in np.argsort(-tot_w)[:5]:
    print(f"  res{i+1:2d} {WT[i]}: WT={tot_w[i]:.2f} MUT={tot_m[i]:.2f} d={tot_m[i]-tot_w[i]:+.2f}")
# heatmaps
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,(M,t) in zip(ax,[(aW,"WT"),(aM,"K409Q"),(aM-aW,"MUT - WT")]):
    im=a.imshow(M,aspect="auto",cmap="RdBu_r" if "MUT -" in t else "viridis")
    a.set_title(f"KLF4 {t} cross-attn"); a.set_xlabel("protein residue"); a.set_ylabel("PWM position")
    a.axvline(MUTSITE,color="r",ls="--",lw=0.8); plt.colorbar(im,ax=a,fraction=0.04)
fig.tight_layout(); os.makedirs("results/klf4_attn_noRAG",exist_ok=True)
fig.savefig("results/klf4_attn_noRAG/cross_attn.png",dpi=130,bbox_inches="tight")
print("\nSaved results/klf4_attn_noRAG/cross_attn.png")
