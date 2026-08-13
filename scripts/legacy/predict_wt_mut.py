import os,sys,json,numpy as np,torch,torch.nn.functional as F,pandas as pd
sys.path.insert(0,"src"); sys.path.insert(0,"pwm_rosetta")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pwm_hybrid.pwm.viz import makeLogo
from scipy.stats import pearsonr
CKPT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt"
WT ="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
MUT="RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
EXCLUDE={"MYOD1"}; FID=3; K=3; ML=20; dev="cuda" if torch.cuda.is_available() else "cpu"; B=np.array(list("ACGT"))
cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CKPT),"config.json"))).items():
    if hasattr(cfg,k):
        try:setattr(cfg,k,type(getattr(cfg,k))(v))
        except:pass
cfg.use_retrieval=True
model=TFScopeModel(cfg).to(dev).eval()
model.load_state_dict(torch.load(CKPT,map_location=dev,weights_only=False)["model"],strict=False)
import esm
em,al=esm.pretrained.esm2_t33_650M_UR50D(); em=em.to(dev).eval(); bc=al.get_batch_converter()
embs=np.load("data/processed/tf_dbd_embeddings.npz"); split=json.load(open("data/processed/splits/deeppbs_only/benchmark_no_val.json"))
df=pd.read_parquet("data/processed/tf_pwm_deeppbs_only.parquet")
g=dict(zip(df["filename"],df["gene_symbol"].astype(str).str.upper()))
fn2pwm={r["filename"]:np.frombuffer(r["pwm"],dtype=np.float32).reshape(4,-1).copy() for _,r in df.iterrows()}
donors=[fn for fn in embs.files if fn in (set(split["train"])|set(split["val"])) and g.get(fn) not in EXCLUDE]
Mtr=np.stack([embs[fn] for fn in donors]); Mtr/= (np.linalg.norm(Mtr,axis=1,keepdims=True)+1e-8)
def predict(seq,label):
    _,_,toks=bc([("q",seq)])
    with torch.no_grad():
        rep=em(toks.to(dev),repr_layers=[33])["representations"][33]
        qv=rep[0,1:1+len(seq)].mean(0).cpu().numpy().astype(np.float32)
    q=qv/(np.linalg.norm(qv)+1e-8); sims=Mtr@q; top=np.argsort(-sims)[:K]
    ret=torch.full((1,K,4,ML),0.25); rm=torch.zeros((1,K,ML)); rs=torch.zeros((1,K))
    nbrs=[]
    for ki,di in enumerate(top):
        fn=donors[di]; p=fn2pwm[fn]; L=min(p.shape[1],ML)
        ret[0,ki,:,:L]=torch.from_numpy(p[:,:L]); rm[0,ki,:L]=1; rs[0,ki]=float(sims[di])
        nbrs.append((df[df.filename==fn].iloc[0]['gene_symbol'],float(sims[di])))
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([FID],device=dev)
    with torch.no_grad():
        gl,pl,_=model(tok,dm,fi,retrieved_pwms=ret.to(dev),retrieved_masks=rm.to(dev),retrieved_sims=rs.to(dev))
        gate=gl.sigmoid()[0].cpu().numpy(); pwm=F.softmax(pl,1)[0].cpu().numpy()
    pl_=max(4,int((gate>0.5).sum())); core=pwm[:,:pl_]
    print(f"[{label}] neighbours={nbrs}  len={pl_} consensus={''.join(B[core.argmax(0)])}")
    return pwm,core
pw_wt,c_wt=predict(WT,"WT")
pw_mut,c_mut=predict(MUT,"MUT L12R")
# full-PWM difference over 20 positions
diff=np.abs(pw_wt-pw_mut).sum(0)  # per-position L1
r_full=pearsonr(pw_wt.flatten(),pw_mut.flatten())[0]
print(f"\nWT vs MUT predicted PWM: full Pearson r={r_full:.4f}  max per-pos L1 diff={diff.max():.4f}  mean={diff.mean():.4f}")
# side-by-side logo
fig,axes=plt.subplots(2,1,figsize=(6,3.2))
for ax,(core,t) in zip(axes,[(c_wt,"WT  MyoD1  "+''.join(B[c_wt.argmax(0)])),(c_mut,"MUT L12R  "+''.join(B[c_mut.argmax(0)]))]):
    ppm=np.clip(core.T,1e-8,1); ppm/=ppm.sum(1,keepdims=True); makeLogo(ppm,ax)
    ax.set_ylim(0,2); ax.set_xticks([]); ax.set_title(t,fontsize=8,fontweight="bold")
fig.suptitle(f"v14(LGO): WT vs L12R mutant — PWM Pearson r={r_full:.3f}",fontsize=9)
fig.tight_layout(); os.makedirs("results/myod1_mut",exist_ok=True)
fig.savefig("results/myod1_mut/wt_vs_mut_logo.png",dpi=150,bbox_inches="tight")
fig.savefig("results/myod1_mut/wt_vs_mut_logo.pdf",bbox_inches="tight")
print("Saved results/myod1_mut/wt_vs_mut_logo.png")
