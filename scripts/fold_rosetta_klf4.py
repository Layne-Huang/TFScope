#!/usr/bin/env python
"""Boltz(MSA)+Rosetta on KLF4 (C2H2 monomer) + GGGCGGGGC GC-box, fixed site.
Runs WT and K409Q (K19Q in construct); reports interface ddG, Rosetta PWM."""
import os, sys, glob, subprocess, time
import numpy as np
sys.path.insert(0, "pwm_rosetta"); sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
BCACHE="/n/holylabs/lpinello_lab/Lab/leihuang/.boltz"
COMP={'A':'T','T':'A','G':'C','C':'G'}; B=np.array(list("ACGT"))
def rc(s): return ''.join(COMP[c] for c in reversed(s))

def write_fasta(d,name,prot,dna):
    os.makedirs(d,exist_ok=True)
    # C2H2 monomer: one protein chain + dsDNA
    open(f"{d}/{name}.fasta","w").write(f">A|protein\n{prot}\n>B|dna\n{dna}\n>C|dna\n{rc(dna)}\n")

def run_boltz(indir,outdir):
    cmd=["boltz","predict",indir,"--out_dir",outdir,"--cache",BCACHE,"--model","boltz2",
         "--output_format","mmcif","--use_msa_server","--no_kernels","--accelerator","gpu","--devices","1","--override"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: print("  boltz err:",r.stderr[-300:])
    return r.returncode==0

def find_cif(outdir,name):
    for pat in [f"{outdir}/**/{name}_model_0.cif",f"{outdir}/**/*model_0.cif"]:
        h=glob.glob(pat,recursive=True)
        if h: return sorted(h)[0]
    return None

def get_iptm(outdir,name):
    import json
    for pat in [f"{outdir}/**/confidence_{name}_model_0.json",f"{outdir}/**/confidence_*model_0.json"]:
        h=glob.glob(pat,recursive=True)
        if h: return json.load(open(h[0])).get("iptm",0.0)
    return 0.0

def rosetta_pwm(cif,work,tau=1.5):
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv
    os.makedirs(work,exist_ok=True)
    generate_pwm_hybrid(protein_seq=None,dna_seq=None,output_dir=work,wt_pdb=cif,minimize_local=True,use_relax=False)
    seq,E,PPM,PWM=pwm_from_hybrid_csv(f"{work}/pwm_results_hybrid.csv",tau=tau)
    return seq,np.asarray(PPM)

if __name__=="__main__":
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags()); init_pyrosetta(psipred_exe="")
    WT ="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH"
    MUT=WT[:18]+"Q"+WT[19:]
    DNA="GGGCGGGGC"; root="results/klf4_gcbox"; os.makedirs(root,exist_ok=True)
    for label,seq in [("WT",WT),("K409Q",MUT)]:
        rd=f"{root}/{label}"; fdir=f"{rd}/fasta"; bout=f"{rd}/boltz"
        write_fasta(fdir,label,seq,DNA)
        print(f"\n=== KLF4 {label} + {DNA} (rc {rc(DNA)}) ===",flush=True)
        t=time.time()
        if not run_boltz(fdir,bout): print("  boltz failed"); continue
        cif=find_cif(bout,label); iptm=get_iptm(bout,label)
        print(f"  boltz {time.time()-t:.0f}s iPTM={iptm:.3f}",flush=True)
        if not cif: continue
        t=time.time(); seq_dna,PPM=rosetta_pwm(cif,f"{rd}/rosetta")
        cons="".join(B[PPM[i].argmax()] for i in range(PPM.shape[0]))
        print(f"  rosetta {time.time()-t:.0f}s  DNA={seq_dna}  Rosetta consensus={cons}",flush=True)
        np.save(f"{rd}/ppm.npy",PPM)
        for r4 in range(4): print(f"    {B[r4]}: "+" ".join(f"{PPM[j,r4]:.2f}" for j in range(PPM.shape[0])))
    print("\n(truth: KLF4 WT=GGGCGGGGC, K409Q=GGGTGGGTG)")
