#!/usr/bin/env python
"""Fold-scan-consensus-converge with: homodimer + flanked E-box + 5-structure
ensemble. Each round folds the protein DIMER on (flank + core + flank) DNA,
generates 5 Boltz structures, Rosetta-scans each, averages the PWM over the 5,
updates the CORE consensus (flanks fixed), and iterates until the core converges.

Run for WT and L122R MyoD to infer each one's preferred E-box core de novo.
"""
import os, sys, glob, subprocess, time, json
import numpy as np
sys.path.insert(0, "pwm_rosetta"); sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
BCACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.boltz"
COMP = {'A':'T','T':'A','G':'C','C':'G'}; B = np.array(list("ACGT"))
def rc(s): return ''.join(COMP[c] for c in reversed(s))

FLANK_L, FLANK_R = "GGG", "GGG"     # fixed flanks
N_SAMPLES = 5

def write_dimer_fasta(d, name, prot, dna_full):
    os.makedirs(d, exist_ok=True)
    # two protein chains (homodimer) + two DNA strands
    open(f"{d}/{name}.fasta","w").write(
        f">A|protein\n{prot}\n>B|protein\n{prot}\n>C|dna\n{dna_full}\n>D|dna\n{rc(dna_full)}\n")

def run_boltz(indir, outdir):
    cmd=["boltz","predict",indir,"--out_dir",outdir,"--cache",BCACHE,"--model","boltz2",
         "--output_format","mmcif","--use_msa_server","--no_kernels",
         "--diffusion_samples",str(N_SAMPLES),
         "--accelerator","gpu","--devices","1","--override"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: print("  boltz err:",r.stderr[-400:])
    return r.returncode==0

def find_cifs(outdir, name):
    h=glob.glob(f"{outdir}/**/{name}_model_*.cif",recursive=True)
    if not h: h=glob.glob(f"{outdir}/**/*model_*.cif",recursive=True)
    return sorted(set(h))

def iptm_of(cif):
    d=os.path.dirname(cif); base=os.path.basename(cif).replace(".cif","")
    cand=glob.glob(f"{d}/confidence_{base}.json")
    if cand:
        try: return json.load(open(cand[0])).get("iptm",0.0)
        except: return 0.0
    return 0.0

def rosetta_ppm(cif, work, tau=1.5):
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv
    os.makedirs(work, exist_ok=True)
    generate_pwm_hybrid(protein_seq=None, dna_seq=None, output_dir=work,
                        wt_pdb=cif, minimize_local=True, use_relax=False)
    seq,E,PPM,PWM = pwm_from_hybrid_csv(f"{work}/pwm_results_hybrid.csv", tau=tau)
    return seq, np.asarray(PPM)   # (Lfull,4)

def iterate(label, prot, start_core, root, max_rounds=5):
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags()); init_pyrosetta(psipred_exe="")
    core = start_core; prev = None; hist=[]; off=len(FLANK_L)
    for rnd in range(max_rounds):
        dna_full = FLANK_L + core + FLANK_R
        print(f"\n--- {label} round {rnd}: core={core} full={dna_full} ---", flush=True)
        rd=f"{root}/{label}/round{rnd}"; fdir=f"{rd}/fasta"; bout=f"{rd}/boltz"
        write_dimer_fasta(fdir, label, prot, dna_full)
        t=time.time()
        if not run_boltz(fdir,bout): print("  boltz failed"); break
        cifs=find_cifs(bout,label)
        print(f"  boltz {time.time()-t:.0f}s  {len(cifs)} structures  iPTM={[round(iptm_of(c),2) for c in cifs]}", flush=True)
        if not cifs: break
        # Rosetta scan each structure, average PPM over the ensemble
        ppms=[]
        t=time.time()
        for si,cif in enumerate(cifs):
            try:
                _,ppm=rosetta_ppm(cif, f"{rd}/rosetta_{si}")
                ppms.append(ppm)
            except Exception as e:
                print(f"   [warn] rosetta sample {si}: {e}")
        if not ppms: print("  all rosetta failed"); break
        # align lengths and average
        Lmin=min(p.shape[0] for p in ppms)
        ens=np.mean([p[:Lmin] for p in ppms],axis=0)            # (Lmin,4)
        print(f"  rosetta x{len(ppms)} {time.time()-t:.0f}s", flush=True)
        core_ppm=ens[off:off+len(core)]
        new_core="".join(B[core_ppm[i].argmax()] for i in range(core_ppm.shape[0]))
        full_cons="".join(B[ens[i].argmax()] for i in range(ens.shape[0]))
        hist.append({"round":rnd,"core_in":core,"new_core":new_core,"full":full_cons,
                     "n_struct":len(ppms),"iptm":[float(iptm_of(c)) for c in cifs]})
        print(f"  ensemble full consensus={full_cons}  CORE={new_core}", flush=True)
        np.save(f"{rd}/ensemble_ppm.npy", ens)
        if new_core==prev:
            print(f"  CONVERGED at round {rnd}"); break
        prev=new_core; core=new_core
    json.dump(hist, open(f"{root}/{label}_history.json","w"), indent=2)
    return hist

if __name__=="__main__":
    WT ="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    MUT="RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    START="CAGCTG"   # canonical E-box scaffold (neutral start for both)
    root="results/myod1_dimer_ensemble"; os.makedirs(root,exist_ok=True)
    hw=iterate("WT",   WT,  START, root)
    hm=iterate("L122R",MUT, START, root)
    print("\n=== SUMMARY (core trajectories) ===")
    print("WT   :", [h["new_core"] for h in hw], " ground truth CAGCTG/CACCTG")
    print("L122R:", [h["new_core"] for h in hm], " ground truth CACGTG")
