#!/usr/bin/env python
"""Inject structural Zn2+ ions into the AF3 inputs for the C2H2 ZF folds.

Each construct is a 3-finger C2H2 array -> needs 3 Zn2+. The original zf_rag folds
were run WITHOUT zinc, so the fingers can't chelate and the interface energies are
unphysical (zf93 +634 kcal/mol). We add 3 ZN CCD ligands (ids D,E,F) to each existing
*_data.json (protein MSA preserved) so re-folding is pure inference (no re-MSA), and
pwm_rosetta will then detect the ions (metal_free=0).

Modes:
  wt       : 4 WT complexes, zinc added            -> results/af3_zinc/wt/inputs/<zf>_znwt.json
  mutscan  : every single-base DNA mutant, zinc    -> results/af3_zinc/mutscan/<zf>/inputs/
  deeppbs  : DeepPBS-consensus seed, zinc           -> results/af3_zinc/deeppbs/inputs/
"""
import argparse, copy, json, os
import numpy as np

AF3 = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag"
COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}
ZFS = ["zf21", "zf93", "zf92", "zf129"]
N_ZN = 3
BASES = ["A", "C", "G", "T"]


def rc(s):
    return "".join(COMP[b] for b in reversed(s))


def load_tmpl(zf):
    return json.load(open(f"{AF3}/{zf}_rag/{zf}_rag_data.json"))


def add_zinc(d):
    """Append N_ZN ZN ligand entries with fresh chain ids after existing ones."""
    used = []
    for s in d["sequences"]:
        k = list(s.keys())[0]
        ids = s[k]["id"]
        used += ids if isinstance(ids, list) else [ids]
    # next free single-letter chain ids
    nxt = [c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used]
    for i in range(N_ZN):
        d["sequences"].append({"ligand": {"id": [nxt[i]], "ccdCodes": ["ZN"]}})
    return d


def set_dna(d, fwd):
    seen = 0
    for s in d["sequences"]:
        if "dna" in s:
            s["dna"]["sequence"] = fwd if seen == 0 else rc(fwd)
            seen += 1
    return d


def build_wt(out):
    for zf in ZFS:
        d = add_zinc(copy.deepcopy(load_tmpl(zf)))
        d["name"] = f"{zf}_znwt"
        in_dir = os.path.join(out, "wt", "inputs"); os.makedirs(in_dir, exist_ok=True)
        json.dump(d, open(os.path.join(in_dir, f"{zf}_znwt.json"), "w"))
        fwd = [s["dna"]["sequence"] for s in d["sequences"] if "dna" in s][0]
        print(f"wt {zf}: zinc x{N_ZN}, DNA {fwd}")


def build_mutscan(out):
    total = 0
    for zf in ZFS:
        tmpl = add_zinc(copy.deepcopy(load_tmpl(zf)))
        fwd = [s["dna"]["sequence"] for s in tmpl["sequences"] if "dna" in s][0]
        in_dir = os.path.join(out, "mutscan", zf, "inputs"); os.makedirs(in_dir, exist_ok=True)
        man = {"zf": zf, "wt_fwd": fwd, "length": len(fwd), "mutants": []}
        wt = copy.deepcopy(tmpl); wt["name"] = f"{zf}_WT"
        json.dump(wt, open(os.path.join(in_dir, f"{zf}_WT.json"), "w"))
        man["wt_name"] = f"{zf}_WT"
        for p, wb in enumerate(fwd):
            for mb in "ACGT":
                if mb == wb:
                    continue
                nf = fwd[:p] + mb + fwd[p + 1:]
                nm = f"{zf}_{wb}{p+1}{mb}"
                d = set_dna(copy.deepcopy(tmpl), nf); d["name"] = nm
                json.dump(d, open(os.path.join(in_dir, nm + ".json"), "w"))
                man["mutants"].append({"name": nm, "pos": p + 1, "wt_base": wb,
                                       "mut_base": mb, "fwd": nf})
        json.dump(man, open(os.path.join(out, "mutscan", zf, "manifest.json"), "w"), indent=2)
        total += 1 + len(man["mutants"])
        print(f"mutscan {zf}: WT + {len(man['mutants'])} mutants (zinc)")
    print("mutscan total folds:", total)


def build_deeppbs(out):
    in_dir = os.path.join(out, "deeppbs", "inputs"); os.makedirs(in_dir, exist_ok=True)
    man = {}
    for zf in ZFS:
        P = np.load(f"results/zf_struct/{zf}_model.npz_predict.npz")["P"]
        cons = "".join(BASES[i] for i in P.argmax(1))
        d = set_dna(add_zinc(copy.deepcopy(load_tmpl(zf))), cons)
        d["name"] = f"{zf}_deeppbs_zn"
        json.dump(d, open(os.path.join(in_dir, f"{zf}_deeppbs_zn.json"), "w"))
        man[zf] = {"deeppbs_consensus": cons}
        print(f"deeppbs {zf}: zinc, DNA {cons}")
    json.dump(man, open(os.path.join(out, "deeppbs", "manifest.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wt", "mutscan", "deeppbs", "all"], default="wt")
    ap.add_argument("--out", default="results/af3_zinc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode in ("wt", "all"): build_wt(a.out)
    if a.mode in ("mutscan", "all"): build_mutscan(a.out)
    if a.mode in ("deeppbs", "all"): build_deeppbs(a.out)


if __name__ == "__main__":
    main()
