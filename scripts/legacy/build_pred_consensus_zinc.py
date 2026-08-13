#!/usr/bin/env python
"""Build AF3 inputs (with 3 Zn2+) for the TFScope- and DeepPBS-predicted consensus
folds (Analysis 1). Clones each protein's MSA data.json, injects zinc, sets DNA.

Templates (MSA source):
  zf21/zf93/zf92 -> af3_runs/zf_rag/<zf>_rag/<zf>_rag_data.json  (protein unchanged)
  zf129          -> af3_runs/zf129_new/zf129new/zf129new_data.json (corrected protein)
"""
import argparse, copy, json, os, csv
import numpy as np

ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs"
COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}
BASES = ["A", "C", "G", "T"]
N_ZN = 3
ZFS = ["zf21", "zf93", "zf92", "zf129"]


def rc(s):
    return "".join(COMP[b] for b in reversed(s))


def template_path(zf):
    if zf == "zf129":
        return f"{ROOT}/zf129_new/zf129new/zf129new_data.json"
    return f"{ROOT}/zf_rag/{zf}_rag/{zf}_rag_data.json"


def tfscope_consensus(zf):
    rows = []
    for r in csv.reader(open(f"results/zf_struct/tfscope_rag_best/{zf}.pwm.tsv"), delimiter="\t"):
        if r and r[0] != "pos":
            rows.append([float(x) for x in r[1:5]])
    return "".join(BASES[i] for i in np.array(rows).argmax(1))


def deeppbs_consensus(zf):
    P = np.load(f"results/zf_struct/{zf}_model.npz_predict.npz")["P"]
    return "".join(BASES[i] for i in P.argmax(1))


def add_zinc(d):
    used = []
    for s in d["sequences"]:
        k = list(s.keys())[0]
        ids = s[k]["id"]
        used += ids if isinstance(ids, list) else [ids]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfs", nargs="*", default=ZFS)
    ap.add_argument("--out", default="results/af3_analysis1")
    a = ap.parse_args()
    in_dir = os.path.join(a.out, "inputs")
    os.makedirs(in_dir, exist_ok=True)
    manifest = {}
    for zf in a.zfs:
        tp = template_path(zf)
        if not os.path.exists(tp):
            print(f"[skip] {zf}: template not ready ({tp})")
            continue
        tmpl = json.load(open(tp))
        for method, cons in [("tfscope", tfscope_consensus(zf)),
                             ("deeppbs", deeppbs_consensus(zf))]:
            d = set_dna(add_zinc(copy.deepcopy(tmpl)), cons)
            name = f"{zf}_{method}_zn"
            d["name"] = name
            json.dump(d, open(os.path.join(in_dir, name + ".json"), "w"))
            manifest[name] = {"zf": zf, "method": method, "consensus": cons, "length": len(cons)}
            print(f"{name}: DNA {cons} ({len(cons)}bp) + 3 ZN")
    mf = os.path.join(a.out, "manifest.json")
    old = json.load(open(mf)) if os.path.exists(mf) else {}
    old.update(manifest)
    json.dump(old, open(mf, "w"), indent=2)
    print(f"manifest -> {mf} ({len(old)} entries)")


if __name__ == "__main__":
    main()
