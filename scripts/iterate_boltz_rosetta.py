#!/usr/bin/env python
"""Local Boltz + Rosetta iterative calibration, starting from the v14-predicted
consensus DNA. Evolves the DNA (fold -> Rosetta DDG scan -> new consensus ->
refold) until the consensus converges. Run for WT and L12R MyoD to test whether
the structure+physics path detects the single-residue mutation the seq-only
model is blind to.
"""
import os, sys, json, glob, subprocess, time
import numpy as np
sys.path.insert(0, "pwm_rosetta"); sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

BCACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.boltz"
COMP = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
B = np.array(list("ACGT"))
def rc(s): return ''.join(COMP[c] for c in reversed(s))

def write_fasta(d, name, prot, dna):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{name}.fasta","w") as f:
        f.write(f">A|protein\n{prot}\n>B|dna\n{dna}\n>C|dna\n{rc(dna)}\n")

def run_boltz(indir, outdir):
    cmd=["boltz","predict",indir,"--out_dir",outdir,"--cache",BCACHE,"--model","boltz2",
         "--output_format","mmcif","--use_msa_server","--no_kernels",
         "--accelerator","gpu","--devices","1","--override"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: print("  boltz stderr:",r.stderr[-400:])
    return r.returncode==0

def find_cif(outdir,name):
    for pat in [f"{outdir}/**/{name}_model_0.cif",f"{outdir}/**/*model_0.cif"]:
        h=glob.glob(pat,recursive=True)
        if h: return sorted(h)[0]
    return None

def rosetta_pwm(cif, work, tau=1.5):
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv
    os.makedirs(work, exist_ok=True)
    generate_pwm_hybrid(protein_seq=None, dna_seq=None, output_dir=work,
                        wt_pdb=cif, minimize_local=True, use_relax=False)
    seq,E,PPM,PWM = pwm_from_hybrid_csv(f"{work}/pwm_results_hybrid.csv", tau=tau)
    return seq, np.asarray(PPM)   # (L,4) ACGT

def iterate(label, prot, start_dna, root, max_rounds=4):
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags()); init_pyrosetta(psipred_exe="")
    dna=start_dna; prev=None; hist=[]
    for rnd in range(max_rounds):
        print(f"\n--- {label} round {rnd}: DNA={dna} ---", flush=True)
        rd=f"{root}/{label}/round{rnd}"; fdir=f"{rd}/fasta"; bout=f"{rd}/boltz"
        write_fasta(fdir, label, prot, dna)
        t=time.time()
        if not run_boltz(fdir, bout): print("  boltz failed"); break
        cif=find_cif(bout,label)
        print(f"  boltz {time.time()-t:.0f}s cif={cif}", flush=True)
        if not cif: break
        try:
            t=time.time()
            seq,PPM=rosetta_pwm(cif, f"{rd}/rosetta")
            print(f"  rosetta {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"  rosetta failed: {e}"); break
        cons="".join(B[PPM[i].argmax()] for i in range(PPM.shape[0]))
        hist.append({"round":rnd,"dna_in":dna,"consensus":cons})
        print(f"  Rosetta consensus={cons}", flush=True)
        np.save(f"{rd}/ppm.npy", PPM)
        if cons==prev:
            print(f"  CONVERGED at round {rnd}"); break
        prev=cons; dna=cons
    json.dump(hist, open(f"{root}/{label}_history.json","w"), indent=2)
    return hist

if __name__=="__main__":
    WT ="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    MUT="RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    START="ACCATCT"   # v14-predicted consensus (identical for WT and mutant)
    root="results/myod1_evolve"; os.makedirs(root, exist_ok=True)
    hw=iterate("WT", WT, START, root)
    hm=iterate("L12R", MUT, START, root)
    print("\n=== SUMMARY ===")
    print("WT  :", [h["consensus"] for h in hw])
    print("L12R:", [h["consensus"] for h in hm])
