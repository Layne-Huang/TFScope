import os,sys,json,numpy as np,torch,torch.nn.functional as F,pandas as pd
sys.path.insert(0,"src"); sys.path.insert(0,"pwm_rosetta")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pwm_hybrid.pwm.viz import makeLogo

CKPT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt"
seq="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
EXCLUDE_GENE={"MYOD1"}; FAMILY_ID=3; K=3; ML=20; device="cuda" if torch.cuda.is_available() else "cpu"
B=np.array(list("ACGT"))

cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CKPT),"config.json"))).items():
    if hasattr(cfg,k):
        try:setattr(cfg,k,type(getattr(cfg,k))(v))
        except:pass
cfg.use_retrieval=True
model=TFScopeModel(cfg).to(device).eval()
model.load_state_dict(torch.load(CKPT,map_location=device,weights_only=False)["model"],strict=False)

tokens=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=device)
dbd_mask=torch.ones(1,len(seq),dtype=torch.bool,device=device)
family_id=torch.tensor([FAMILY_ID],dtype=torch.long,device=device)

import esm
em,alpha=esm.pretrained.esm2_t33_650M_UR50D(); em=em.to(device).eval(); bc=alpha.get_batch_converter()
_,_,toks=bc([("q",seq)])
with torch.no_grad():
    rep=em(toks.to(device),repr_layers=[33])["representations"][33]
    qvec=rep[0,1:1+len(seq)].mean(0).cpu().numpy().astype(np.float32)
del em; torch.cuda.empty_cache()

embs=np.load("data/processed/tf_dbd_embeddings.npz"); split=json.load(open("data/processed/splits/deeppbs_only/benchmark_no_val.json"))
df=pd.read_parquet("data/processed/tf_pwm_deeppbs_only.parquet")
g=dict(zip(df["filename"],df["gene_symbol"].astype(str).str.upper()))
fn2pwm={r["filename"]:np.frombuffer(r["pwm"],dtype=np.float32).reshape(4,-1) for _,r in df.iterrows()}
donors=[fn for fn in embs.files if fn in (set(split["train"])|set(split["val"])) and g.get(fn) not in EXCLUDE_GENE]
M=np.stack([embs[fn] for fn in donors]); M=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-8)
q=qvec/(np.linalg.norm(qvec)+1e-8); sims=M@q; top=np.argsort(-sims)[:K]

ret=torch.full((1,K,4,ML),0.25); rm=torch.zeros((1,K,ML)); rs=torch.zeros((1,K))
print("LGO neighbours (gene MYOD1 excluded):")
for ki,di in enumerate(top):
    fn=donors[di]; p=fn2pwm[fn]; L=min(p.shape[1],ML)
    ret[0,ki,:,:L]=torch.from_numpy(p[:,:L]); rm[0,ki,:L]=1; rs[0,ki]=float(sims[di])
    cons="".join(B[p[:,j].argmax()] for j in range(p.shape[1]))
    print(f"  [{ki}] {df[df.filename==fn].iloc[0]['gene_symbol']:<10} cos={sims[di]:.3f} consensus={cons}")

with torch.no_grad():
    gl,pl,_=model(tokens,dbd_mask,family_id,retrieved_pwms=ret.to(device),retrieved_masks=rm.to(device),retrieved_sims=rs.to(device))
    gate=gl.sigmoid()[0].cpu().numpy(); pwm=F.softmax(pl,1)[0].cpu().numpy()
pred_len=max(4,int((gate>0.5).sum())); core=pwm[:,:pred_len]
print(f"\nPredicted len={pred_len}  consensus={''.join(B[core.argmax(0)])}")
print("True MyoD1 motif: CAGGTG (E-box)")
fig,ax=plt.subplots(figsize=(max(4,pred_len*0.6),2))
ppm=np.clip(core.T,1e-8,1); ppm/=ppm.sum(1,keepdims=True); makeLogo(ppm,ax)
ax.set_title(f"v14 (LGO) MyoD1 prediction  len={pred_len}  consensus={''.join(B[core.argmax(0)])}",fontsize=8)
fig.tight_layout(); os.makedirs("results/myod1_lgo",exist_ok=True)
fig.savefig("results/myod1_lgo/logo.pdf",bbox_inches="tight"); fig.savefig("results/myod1_lgo/logo.png",dpi=150,bbox_inches="tight")
print("Saved results/myod1_lgo/logo.png")
