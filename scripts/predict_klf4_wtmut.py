import os,sys,json,numpy as np,torch,torch.nn.functional as F,pandas as pd
sys.path.insert(0,"src")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from scipy.stats import pearsonr
CK="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_robustRAG/ckpt_best.pt"
WT ="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
MUT=WT[:18]+"Q"+WT[19:]   # K19Q (best mapping of K409)
EXCLUDE={"KLF4"}; FID=0; K=3; ML=20; dev="cuda" if torch.cuda.is_available() else "cpu"; B=np.array(list("ACGT"))
cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CK),"config.json"))).items():
    if hasattr(cfg,k):
        try:setattr(cfg,k,type(getattr(cfg,k))(v))
        except:pass
cfg.use_retrieval=True
m=TFScopeModel(cfg).to(dev).eval(); m.load_state_dict(torch.load(CK,map_location=dev,weights_only=False)["model"],strict=False)
import esm
em,al=esm.pretrained.esm2_t33_650M_UR50D(); em=em.to(dev).eval(); bc=al.get_batch_converter()
embs=np.load("data/processed/tf_dbd_embeddings.npz"); split=json.load(open("data/processed/splits/deeppbs_only/benchmark_no_val.json"))
df=pd.read_parquet("data/processed/tf_pwm_deeppbs_only.parquet")
g=dict(zip(df["filename"],df["gene_symbol"].astype(str).str.upper()))
fn2pwm={r["filename"]:np.frombuffer(r["pwm"],dtype=np.float32).reshape(4,-1).copy() for _,r in df.iterrows()}
donors=[fn for fn in embs.files if fn in (set(split["train"])|set(split["val"])) and g.get(fn) not in EXCLUDE]
Mtr=np.stack([embs[fn] for fn in donors]); Mtr/=(np.linalg.norm(Mtr,axis=1,keepdims=True)+1e-8)
def emb(seq):
    _,_,toks=bc([("q",seq)])
    with torch.no_grad():
        return em(toks.to(dev),repr_layers=[33])["representations"][33][0,1:1+len(seq)].mean(0).cpu().numpy().astype(np.float32)
def predict(seq,use_ret):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([FID],device=dev)
    rp=rm=rs=None; nbrs=[]
    if use_ret:
        q=emb(seq); q/=np.linalg.norm(q)+1e-8; sims=Mtr@q; top=np.argsort(-sims)[:K]
        rp=torch.full((1,K,4,ML),0.25); rm=torch.zeros((1,K,ML)); rs=torch.zeros((1,K))
        for ki,di in enumerate(top):
            fn=donors[di]; p=fn2pwm[fn]; L=min(p.shape[1],ML)
            rp[0,ki,:,:L]=torch.from_numpy(p[:,:L]); rm[0,ki,:L]=1; rs[0,ki]=float(sims[di]); nbrs.append(g[fn])
        rp,rm,rs=rp.to(dev),rm.to(dev),rs.to(dev)
    with torch.no_grad():
        gl,pl,_=m(tok,dm,fi,retrieved_pwms=rp,retrieved_masks=rm,retrieved_sims=rs)
        return F.softmax(pl,1)[0].cpu().numpy(),nbrs
for mode,use in [("LGO-retrieval",True),("de-novo",False)]:
    pw,nw=predict(WT,use); pm,nm=predict(MUT,use)
    r=pearsonr(pw.flatten(),pm.flatten())[0]
    cw="".join(B[pw[:,j].argmax()] for j in range(9)); cm="".join(B[pm[:,j].argmax()] for j in range(9))
    print(f"\n[{mode}]"); 
    if use: print(f"  WT nbrs={nw} MUT nbrs={nm}")
    print(f"  WT consensus={cw}  K409Q consensus={cm}")
    print(f"  WT vs K409Q PWM Pearson r = {r:.4f}")
print("\n(truth: WT=GGGCGGGGC, K409Q=GGGTGGGTG)")
