#!/usr/bin/env python
"""De-novo specificity inference by constrained physics search.

For each protein (WT, L122R MyoD), fold the homodimer on every CA-NN-TG E-box
core (16 variants) with Boltz+MSA, compute the Rosetta protein-DNA interface
energy, and RANK cores. The lowest-energy core is the inferred preferred motif.
We do NOT assume CACGTG; we score all 16 and let physics pick.

Success = WT argmin in {CAGCTG(GC), CACCTG(CC)}, L122R argmin = CACGTG(CG).
"""
import os, sys, glob, subprocess, time, itertools
import numpy as np
sys.path.insert(0, "pwm_rosetta"); sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
BCACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.boltz"
COMP = {'A':'T','T':'A','G':'C','C':'G'}
def rc(s): return ''.join(COMP[c] for c in reversed(s))

LFLANK, RFLANK = "GGAT", "ATCC"     # fixed flanks, identical for all cores
def site(core): return LFLANK + core + RFLANK   # 14 bp

def write_fasta(d, name, prot, dna):
    os.makedirs(d, exist_ok=True)
    # bHLH HOMODIMER: two protein chains + dsDNA
    open(f"{d}/{name}.fasta","w").write(
        f">A|protein\n{prot}\n>B|protein\n{prot}\n>C|dna\n{dna}\n>D|dna\n{rc(dna)}\n")

def run_boltz(indir, outdir):
    cmd=["boltz","predict",indir,"--out_dir",outdir,"--cache",BCACHE,"--model","boltz2",
         "--output_format","mmcif","--use_msa_server","--no_kernels",
         "--accelerator","gpu","--devices","1","--override"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: print("   boltz err:",r.stderr[-200:])
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

def interface_ddg(cif):
    """Minimized protein-DNA interface ddG (binding energy proxy; lower = stronger)."""
    import pyrosetta
    from pwm_hybrid.rosetta import init as _init
    from pwm_hybrid.rosetta.scoring import calculate_interface_ddg
    pose = pyrosetta.pose_from_file(cif)
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    mm=MoveMap(); mm.set_bb(True); mm.set_chi(True)
    mover=MinMover(); mover.score_function(_init.sfxn); mover.movemap(mm); mover.apply(pose)
    return float(calculate_interface_ddg(pose))

if __name__=="__main__":
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags()); init_pyrosetta(psipred_exe="")
    WT ="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    MUT="RKAATMRERRRRSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    cores=[a+b for a,b in itertools.product("ACGT","ACGT")]   # 16 NN
    root="results/myod1_denovo"; os.makedirs(root,exist_ok=True)
    import json
    results={}
    for label,seq in [("WT",WT),("L122R",MUT)]:
        results[label]={}
        for nn in cores:
            core="CA"+nn+"TG"; dna=site(core); tag=f"{label}_{nn}"
            rd=f"{root}/{tag}"; fdir=f"{rd}/fasta"; bout=f"{rd}/boltz"
            write_fasta(fdir,tag,seq,dna)
            ok=run_boltz(fdir,bout); cif=find_cif(bout,tag) if ok else None
            if not cif:
                print(f"{tag}: fold FAILED",flush=True); results[label][nn]=None; continue
            iptm=get_iptm(bout,tag)
            try: ddg=interface_ddg(cif)
            except Exception as e: print(f"{tag}: ddg err {e}"); ddg=None
            results[label][nn]={"core":core,"ddg":ddg,"iptm":iptm}
            print(f"{tag}: core=CA{nn}TG iPTM={iptm:.3f} interfaceddG={ddg}",flush=True)
        json.dump(results,open(f"{root}/results.json","w"),indent=2)
    # ── rank ──
    print("\n=== INFERRED PREFERENCE (rank cores by interface ddG, lower=stronger) ===")
    for label in results:
        valid={nn:v["ddg"] for nn,v in results[label].items() if v and v["ddg"] is not None}
        ranked=sorted(valid.items(),key=lambda x:x[1])
        print(f"\n{label}: top-5 cores")
        for nn,d in ranked[:5]:
            print(f"   CA{nn}TG  ddG={d:.2f}")
        if ranked:
            best=ranked[0][0]
            print(f"   => INFERRED motif: CA{best}TG")
