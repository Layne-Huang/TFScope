"""Inject structure-derived DNA-contact residues at inference via the v18
contact-bias pathway, sweep the bias scale, and check whether CAC recovery /
correlation with the yeast-display data improves for the 4 designs.

recog_prior[i] = 1 for DBD residues that contact DNA (<4.5A) in the design's
complex, 0 otherwise. contact_bias = bias_scale * recog_prior is added to the
cross-attention logits (pwm_head_v18). bias_scale=0 == current behaviour.
NOTE: model was trained with bias_scale=0, so scale>0 is OFF-DISTRIBUTION.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda:0"; FID_HD = 4
CK = "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42"
cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
DNA = {'DA','DC','DG','DT','A','C','G','T'}

def contacts_from_pdb(pdb, prot_seq, cutoff=4.5):
    prot = {}   # resnum -> (aa, list of atom xyz)
    dna = []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")): continue
        rn = ln[17:20].strip(); x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        if rn in DNA:
            dna.append((x, y, z))
        elif rn in AA3:
            ri = int(ln[22:26]); prot.setdefault(ri, [AA3[rn], []]); prot[ri][1].append((x, y, z))
    if not dna: return None, None
    D = np.array(dna)
    resnums = sorted(prot)
    chainA = "".join(prot[r][0] for r in resnums)
    contact_res = set()
    for r in resnums:
        P = np.array(prot[r][1])
        if ((P[:, None, :] - D[None, :, :]) ** 2).sum(-1).min() < cutoff ** 2:
            contact_res.add(r)
    # map chainA residue order -> prot_seq positions (chainA seq should match prot_seq)
    # find offset by locating chainA seq (or its longest run) in prot_seq
    prior = np.zeros(len(prot_seq))
    # direct: assume resnums are contiguous and chainA == prot_seq substring
    off = prot_seq.find(chainA)
    if off >= 0:
        for j, r in enumerate(resnums):
            if r in contact_res and 0 <= off + j < len(prot_seq): prior[off + j] = 1.0
        cov = f"exact map (offset {off})"
    else:
        # fall back: align index-by-index where residues equal
        n = 0
        for j, r in enumerate(resnums):
            if j < len(prot_seq) and prot_seq[j] == prot[r][0]:
                if r in contact_res: prior[j] = 1.0; n += 1
        cov = f"positional map ({n} contacts placed)"
    return prior, f"{len(contact_res)} contacts, {cov}, chainA_len={len(chainA)}"

# build recog_prior per design
PRIOR = {}
for d in ["DBP005", "DBP006", "DBP009", "DBP035"]:
    pr, info = contacts_from_pdb(f"case_study/pdb/design_pdbs/{d}.pdb", e2[d]["prot_seq"])
    PRIOR[d] = pr
    print(f"{d}: {info}  ->  {int(pr.sum())} contact positions marked")

B = {"A":0,"C":1,"G":2,"T":3}; BA = np.array(list("ACGT")); WT = "GCAGATCTGCACAT"; L = len(WT)
Tt = np.eye(4)[[B[c] for c in WT]].T; VMAX = 2.5
def rc(s):
    if isinstance(s, str): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
    return s[[3,2,1,0]][:, ::-1]

@torch.no_grad()
def predict(seq, fid, prior, scale):
    m.pwm_head.bias_scale = float(scale)                 # override at inference
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    rp = None if (prior is None or scale == 0) else torch.tensor(prior[None], dtype=torch.float32, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=rp)
    return gl.sigmoid()[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()

def core_cons(gate, P):
    W = P.shape[1]; c = np.where(gate > 0.5)[0]
    if len(c) < 4:
        ic = (P*np.log2(P+1e-9)).sum(0)+2; a = ic.argmax(); c = np.arange(max(0,a-4), min(W,a+5))
    lo, hi = c.min(), c.max()+1; return "".join("ACGT"[i] for i in P[:, lo:hi].argmax(0))

# experimental relative-binding matrices for correlation
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
SH = {"DBP005":"Extended_Data_Figure_1_C_DBP005","DBP009":"Extended_Data_Figure_1_E_DBP009",
      "DBP006":"Extended_Data_Figure_1_D_DBP006","DBP035":"Extended_Data_Figure_1_G_DBP035"}
VALC = "Median PE/FITC (Normalized)"; WTOV = {"DBP006":0.1202}
def exp_rel(d):
    df = pd.ExcelFile(XLS).parse(SH[d]); wt = df[df.position.astype(str)=="WT"]
    wtv = WTOV.get(d) or float(wt[VALC].iloc[0])
    dd = df[df.position.astype(str).str.isdigit()].copy(); dd["position"] = dd.position.astype(int)
    R = np.zeros((4, L))
    for p in range(1, L+1):
        sub = dd[dd.position==p]
        for _, r in sub.iterrows(): R[B[str(r.new_base)], p-1] = np.log2(float(r[VALC])/wtv)
    return np.clip(R, -VMAX, VMAX)
def tf_rel(gate, P):
    W = P.shape[1]; c = np.where(gate>0.5)[0]
    if len(c)<4:
        ic=(P*np.log2(P+1e-9)).sum(0)+2; a=ic.argmax(); c=np.arange(max(0,a-4),min(W,a+5))
    lo,hi=c.min(),c.max()+1; klen=hi-lo; best=(-1e9,None,None,None,None)
    for st in("+","-"):
        Q=P if st=="+" else rc(P); clo=lo if st=="+" else (W-hi); core=Q[:,clo:clo+klen]
        for coff in range(-(klen-1),L):
            sc=sum(float(core[:,j]@Tt[:,coff+j]) for j in range(klen) if 0<=coff+j<L)
            if sc>best[0]: best=(sc,st,coff,Q,clo)
    _,st,coff,Q,clo=best; full=np.full((4,L),0.25)
    for p in range(L):
        cc=clo+(p-coff)
        if 0<=cc<W: full[:,p]=Q[:,cc]
    R=np.zeros((4,L))
    for p in range(L):
        wb=B[WT[p]]; R[:,p]=np.log2((full[wb,p]+1e-6)/(full[:,p]+1e-6))
    return np.clip(R,-VMAX,VMAX)
EXP = {d: exp_rel(d) for d in SH}
def rmask():
    mk = np.ones((4, L), bool)
    for p in range(L): mk[B[WT[p]], p] = False
    return mk
MK = rmask()

print("\n=== bias-scale sweep (Homeodomain fid=4) — core consensus | CAC? | r vs experiment ===")
for scale in [0, 1, 2, 4, 8]:
    rs = []
    line = []
    for d in ["DBP005", "DBP006", "DBP009", "DBP035"]:
        gate, P = predict(e2[d]["prot_seq"], FID_HD, PRIOR[d], scale)
        con = core_cons(gate, P); cac = "CAC" if ("CAC" in con or "CAC" in rc(con)) else "---"
        r = float(np.corrcoef(EXP[d][MK], tf_rel(gate, P)[MK])[0, 1]); rs.append(r)
        line.append(f"{d[-3:]}:{con[:12]:<12}[{cac}] r={r:+.2f}")
    print(f" scale={scale}:  meanR={np.mean(rs):+.3f}  | " + " | ".join(line))
