#!/usr/bin/env python
"""MM-GBSA scan WT vs mutant MyoD-DNA complexes; compare PWMs + interface energy."""
import os, sys, glob, json, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src"); sys.path.insert(0,"pwm_rosetta")
from run_mmpbsa_scan import scan_dna_positions, prepare_complex_from_cif, build_system_from_modeller, minimize_and_energy
from scipy.stats import pearsonr

BOLTZ_OUT = sys.argv[1]      # boltz out dir
CORE_LEN  = int(sys.argv[2]) # DNA length
B = np.array(list("ACGT"))

def find_cif(name):
    for pat in [f"{BOLTZ_OUT}/**/{name}_model_0.cif", f"{BOLTZ_OUT}/**/*{name}*model_0.cif"]:
        h=glob.glob(pat,recursive=True)
        if h: return sorted(h)[0]
    return None

res={}
for name in ["MYOD_WT","MYOD_L12R"]:
    cif=find_cif(name)
    if cif is None:
        print(f"[FAIL] no CIF for {name}"); continue
    print(f"\n=== {name}  {cif} ===", flush=True)
    ppm,ddg,wt=scan_dna_positions(cif, 0, CORE_LEN, tau=1.5, min_iter=300)
    if ppm is None: print(f"[FAIL] scan {name}"); continue
    res[name]={"ppm":ppm,"ddg":ddg,"wt":wt}
    print(f"  DNA={wt}  consensus={''.join(B[ppm[i].argmax()] for i in range(ppm.shape[0]))}")

if len(res)==2:
    a,b=res["MYOD_WT"]["ppm"],res["MYOD_L12R"]["ppm"]
    L=min(a.shape[0],b.shape[0])
    r=pearsonr(a[:L].flatten(),b[:L].flatten())[0]
    dd=np.abs(res["MYOD_WT"]["ddg"][:L]-res["MYOD_L12R"]["ddg"][:L])
    print(f"\n=== WT vs L12R (MM-GBSA structure-derived) ===")
    print(f"  PWM Pearson r = {r:.4f}   (seq-only model gave 0.990)")
    print(f"  mean |ΔΔG_WT - ΔΔG_MUT| per base = {dd.mean():.2f} kcal/mol  max={dd.max():.2f}")
    np.savez_compressed("results/myod1_struct/wt_mut_mmpbsa.npz",
        wt_ppm=a, mut_ppm=b, wt_ddg=res["MYOD_WT"]["ddg"], mut_ddg=res["MYOD_L12R"]["ddg"])
    print("  saved results/myod1_struct/wt_mut_mmpbsa.npz")
EOF
