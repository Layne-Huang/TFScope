"""For the 4 designs, feed TRUE 4.5A contacts (from the complexes) through the
contact-bias model's trained bias pathway (learned scale), vs the model's own
PREDICTED contacts (the head). Reports predicted core + CAC + r(exp) for both.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda:0"; FID = 4
CK = "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_contactbias/contactbias_seed42"
cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
scale = float(m.pwm_head.bias_scale_param) if m.pwm_head.bias_learnable else m.pwm_head.bias_scale
print(f"learned bias scale = {scale:.3f}")
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
DNA = {'DA','DC','DG','DT','A','C','G','T'}
def true_prior(pdb, prot_seq, cutoff=4.5):
    prot = {}; dna = []
    for ln in open(pdb):
        if not ln.startswith(("ATOM","HETATM")): continue
        rn = ln[17:20].strip(); x,y,z = float(ln[30:38]),float(ln[38:46]),float(ln[46:54])
        if rn in DNA: dna.append((x,y,z))
        elif rn in AA3:
            ri = int(ln[22:26]); prot.setdefault(ri,[AA3[rn],[]]); prot[ri][1].append((x,y,z))
    D = np.array(dna); resn = sorted(prot); chainA = "".join(prot[r][0] for r in resn)
    contact = {r for r in resn if ((np.array(prot[r][1])[:,None,:]-D[None,:,:])**2).sum(-1).min() < cutoff**2}
    prior = np.zeros(len(prot_seq)); off = prot_seq.find(chainA)
    if off >= 0:
        for j, r in enumerate(resn):
            if r in contact and 0 <= off+j < len(prot_seq): prior[off+j] = 1.0
    return prior, int(sum(1 for r in resn if r in contact))

B = {"A":0,"C":1,"G":2,"T":3}; WT="GCAGATCTGCACAT"; L=len(WT); Tt=np.eye(4)[[B[c] for c in WT]].T; VMAX=2.5
def rc(s):
    if isinstance(s,str): return s[::-1].translate(str.maketrans("ACGT","TGCA"))
    return s[[3,2,1,0]][:,::-1]
@torch.no_grad()
def predict(seq, prior=None):
    # prior given -> feed as recog_prior with head disabled (uses TRUE contacts).
    # prior None -> head active (PREDICTED contacts).
    m.use_contact_pred_head = (prior is None)
    t = torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]], device=dev)
    dm = torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi = torch.tensor([FID],device=dev)
    rp = None if prior is None else torch.tensor(prior[None], dtype=torch.float32, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=rp)
    m.use_contact_pred_head = True
    return gl.sigmoid()[0].cpu().numpy(), F.softmax(pl,1)[0].cpu().numpy()
def core_cons(gate,P):
    W=P.shape[1]; c=np.where(gate>0.5)[0]
    if len(c)<4:
        ic=(P*np.log2(P+1e-9)).sum(0)+2; a=ic.argmax(); c=np.arange(max(0,a-4),min(W,a+5))
    return "".join("ACGT"[i] for i in P[:,c.min():c.max()+1].argmax(0))
# experimental relative-binding for r
XLS="case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
SH={"DBP005":"Extended_Data_Figure_1_C_DBP005","DBP009":"Extended_Data_Figure_1_E_DBP009",
    "DBP006":"Extended_Data_Figure_1_D_DBP006","DBP035":"Extended_Data_Figure_1_G_DBP035"}
VALC="Median PE/FITC (Normalized)"; WTOV={"DBP006":0.1202}
def exp_rel(d):
    df=pd.ExcelFile(XLS).parse(SH[d]); wt=df[df.position.astype(str)=="WT"]
    wtv=WTOV.get(d) or float(wt[VALC].iloc[0]); dd=df[df.position.astype(str).str.isdigit()].copy(); dd["position"]=dd.position.astype(int)
    R=np.zeros((4,L))
    for p in range(1,L+1):
        # LOW value = STRONG binding, so relative preference = log2(wtv/value) (inverted)
        for _,r in dd[dd.position==p].iterrows(): R[B[str(r.new_base)],p-1]=np.log2(wtv/max(float(r[VALC]),1e-3))
    return np.clip(R,-VMAX,VMAX)
def tf_rel(gate,P):
    W=P.shape[1]; c=np.where(gate>0.5)[0]
    if len(c)<4:
        ic=(P*np.log2(P+1e-9)).sum(0)+2; a=ic.argmax(); c=np.arange(max(0,a-4),min(W,a+5))
    lo,hi=c.min(),c.max()+1; klen=hi-lo; best=(-1e9,None,None,None,None)
    for st in("+","-"):
        Q=P if st=="+" else rc(P); clo=lo if st=="+" else (W-hi); cc=Q[:,clo:clo+klen]
        for coff in range(-(klen-1),L):
            sc=sum(float(cc[:,j]@Tt[:,coff+j]) for j in range(klen) if 0<=coff+j<L)
            if sc>best[0]: best=(sc,st,coff,Q,clo)
    _,st,coff,Q,clo=best; full=np.full((4,L),0.25)
    for p in range(L):
        k=clo+(p-coff)
        if 0<=k<W: full[:,p]=Q[:,k]
    R=np.zeros((4,L))
    for p in range(L):
        wb=B[WT[p]]; R[:,p]=np.log2((full[wb,p]+1e-6)/(full[:,p]+1e-6))
    return np.clip(R,-VMAX,VMAX)
mask=np.ones((4,L),bool)
for p in range(L): mask[B[WT[p]],p]=False

print(f"\n{'design':8} {'PREDICTED-contacts':28} {'TRUE-contacts':28}")
print(f"{'':8} {'core':16}{'CAC':5}{'r':>6}   {'core':16}{'CAC':5}{'r':>6}  n_contacts")
for d in ["DBP005","DBP006","DBP009","DBP035"]:
    seq=e2[d]["prot_seq"]; E=exp_rel(d)
    gp,Pp=predict(seq,None)                                  # predicted (head)
    prior,nc=true_prior(f"case_study/pdb/design_pdbs/{d}.pdb", seq)
    gt,Pt=predict(seq,prior)                                 # true contacts
    cp=core_cons(gp,Pp); ct=core_cons(gt,Pt)
    def cac(s): return "CAC" if ("CAC" in s or "CAC" in rc(s)) else "---"
    rp=float(np.corrcoef(E[mask],tf_rel(gp,Pp)[mask])[0,1]); rt=float(np.corrcoef(E[mask],tf_rel(gt,Pt)[mask])[0,1])
    print(f"{d:8} {cp:16}{cac(cp):5}{rp:+6.2f}   {ct:16}{cac(ct):5}{rt:+6.2f}  {nc}")
