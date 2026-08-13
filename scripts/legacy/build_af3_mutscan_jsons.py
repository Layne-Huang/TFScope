#!/usr/bin/env python
"""Build AF3-per-mutation input JSONs for the canonical (expensive) PWM workflow.

For each designed ZF we already have the wild-type AF3 fold + its *_data.json
(protein MSA embedded). Here we clone that data.json and, for every DNA position
x each of the 3 alternative bases, swap chain B (forward) and chain C
(reverse-complement) — keeping the protein + MSA identical so each mutant is a
pure-inference fold (--norun_data_pipeline, no MSA recompute).

This is the AF3-folds-every-mutation method (NOT pwm_rosetta's fold-once + Rosetta
mutate). Rosetta is later used only to score interface ΔΔG on each AF3 fold.

Output:
  results/af3_mutscan/<zf>/inputs/<name>.json     (one per WT + mutant)
  results/af3_mutscan/<zf>/manifest.json
"""
import argparse, copy, json, os

AF3_RUN = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag"
COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}
ZFS = ["zf21", "zf93", "zf92", "zf129"]


def revcomp(s):
    return "".join(COMP[b] for b in reversed(s))


def set_dna(tmpl, fwd):
    """Return a deep copy of the data.json with chain B=fwd, chain C=revcomp(fwd)."""
    d = copy.deepcopy(tmpl)
    rev = revcomp(fwd)
    seen = 0
    for s in d["sequences"]:
        if "dna" in s:
            s["dna"]["sequence"] = fwd if seen == 0 else rev
            seen += 1
    return d


def build_zf(zf, out_root):
    tmpl = json.load(open(f"{AF3_RUN}/{zf}_rag/{zf}_rag_data.json"))
    fwd = [s["dna"]["sequence"] for s in tmpl["sequences"] if "dna" in s][0]
    in_dir = os.path.join(out_root, zf, "inputs")
    os.makedirs(in_dir, exist_ok=True)

    manifest = {"zf": zf, "wt_fwd": fwd, "length": len(fwd), "mutants": []}

    # WT entry (re-fold for an apples-to-apples WT score in the same inference run)
    wt_name = f"{zf}_WT"
    d = set_dna(tmpl, fwd); d["name"] = wt_name
    json.dump(d, open(os.path.join(in_dir, wt_name + ".json"), "w"))
    manifest["wt_name"] = wt_name

    for pos0, wt_base in enumerate(fwd):
        for mut in "ACGT":
            if mut == wt_base:
                continue
            new_fwd = fwd[:pos0] + mut + fwd[pos0 + 1:]
            name = f"{zf}_{wt_base}{pos0 + 1}{mut}"
            d = set_dna(tmpl, new_fwd); d["name"] = name
            json.dump(d, open(os.path.join(in_dir, name + ".json"), "w"))
            manifest["mutants"].append(
                {"name": name, "pos": pos0 + 1, "wt_base": wt_base,
                 "mut_base": mut, "fwd": new_fwd})

    json.dump(manifest, open(os.path.join(out_root, zf, "manifest.json"), "w"), indent=2)
    n = len(manifest["mutants"])
    print(f"{zf}: WT + {n} mutants ({len(fwd)} bp x 3)  -> {in_dir}")
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfs", nargs="*", default=ZFS)
    ap.add_argument("--out", default="results/af3_mutscan")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = sum(build_zf(z, a.out) for z in a.zfs)
    print(f"TOTAL folds to run: {total}")


if __name__ == "__main__":
    main()
