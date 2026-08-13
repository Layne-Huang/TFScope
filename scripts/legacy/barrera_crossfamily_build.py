#!/usr/bin/env python
"""Build cross-family Barrera WT/MUT DBD-crop pairs for genes NOT in the HD set,
using CIS-BP full proteins (assert full[pos-1]==from) + the trusted training DBD crop.

Only genes present in cis_bp/prot_seq.txt with a locatable DBD crop are built here
(PAX/POU locally; C2H2-ZF/Forkhead/NR need a UniProt fetch -> reported as skipped).
Output merges with the 20-gene HD set -> results/mutation_benchmark/crossfamily_pairs.json
(fields: gene, mut, wt_seq, mut_seq, spec_change, family, in_dbd).
"""
import os, sys, json, csv
sys.path.insert(0, "src")
import pandas as pd
PROT = "/data1/leihuang/rCLAMPS/cis_bp/prot_seq.txt"
S6 = "/data1/leihuang/rCLAMPS/barrera2016_SuppTable_S6_combined.csv"
UNIPROT = os.environ.get("UNIPROT_FASTA", "")   # optional extra full-protein source

# CIS-BP full proteins: gene -> list of full seqs (col2=name, col9=Protein_seq)
prot = {}
with open(PROT) as f:
    rd = csv.reader(f, delimiter="\t"); next(rd)
    for l in rd:
        if len(l) > 9 and l[9]:
            prot.setdefault(l[2], []).append(l[9])

# augment with UniProt canonical fasta (header ...GN=<gene>...)
if UNIPROT and os.path.exists(UNIPROT):
    import re
    g, s = None, []
    def flush():
        if g and s: prot.setdefault(g, []).append("".join(s))
    for line in open(UNIPROT):
        if line.startswith(">"):
            flush(); s = []
            mm = re.search(r"GN=(\S+)", line); g = mm.group(1) if mm else None
        else:
            s.append(line.strip())
    flush()

def full_for(gene, fr, pos):
    """longest CIS-BP isoform with residue `fr` at 1-based `pos`."""
    best = ""
    for s in prot.get(gene, []):
        if len(s) >= pos and s[pos - 1] == fr and len(s) > len(best):
            best = s
    return best

tr = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet")
crop_of = {g: (grp.iloc[0]["sequence"], grp.iloc[0]["family_name"])
           for g, grp in tr.groupby("gene_symbol")}

hd_pairs = json.load(open("results/mutation_benchmark/barrera_pairs.json"))["pairs"]
hd_genes = {p["gene"] for p in hd_pairs}
s6 = pd.read_csv(S6)
spec = {}
for _, r in s6.iterrows():
    k = (r["prot"], r["sub"])
    spec[k] = spec.get(k, False) or (str(r["spec.change"]).strip() == "Yes")

out, skipped = [], {}
# 1) HD set: already have crops (wt_seq/mut_seq); attach spec.change + family
for p in hd_pairs:
    k = (p["gene"], p["mut"])
    if k not in spec: continue
    out.append(dict(gene=p["gene"], mut=p["mut"], wt_seq=p["wt_seq"], mut_seq=p["mut_seq"],
                    spec_change=bool(spec[k]),
                    family=crop_of.get(p["gene"], ("", "Homeodomain"))[1], in_dbd=True))
# 2) non-HD set: build crops from CIS-BP full + training DBD crop
for (gene, sub), sc in spec.items():
    if gene in hd_genes: continue
    if gene not in crop_of: skipped.setdefault("no_crop", []).append(gene); continue
    fr, to = sub[0], sub[-1]
    try: pos = int(sub[1:-1])
    except ValueError: continue
    full = full_for(gene, fr, pos)
    if not full: skipped.setdefault("no_fullprot", []).append(gene); continue
    crop, fam = crop_of[gene]
    off = full.find(crop)
    if off < 0: skipped.setdefault("crop_not_substr", []).append(gene); continue
    rel = (pos - 1) - off
    in_dbd = 0 <= rel < len(crop) and crop[rel] == fr
    mut_seq = crop[:rel] + to + crop[rel + 1:] if in_dbd else crop   # out-of-DBD -> identical crop
    out.append(dict(gene=gene, mut=sub, wt_seq=crop, mut_seq=mut_seq,
                    spec_change=bool(sc), family=fam, in_dbd=bool(in_dbd)))

os.makedirs("results/mutation_benchmark", exist_ok=True)
json.dump({"pairs": out}, open("results/mutation_benchmark/crossfamily_pairs.json", "w"))
fam = {}
for r in out: fam.setdefault(r["family"], [0, 0]); fam[r["family"]][0] += 1; fam[r["family"]][1] += r["spec_change"]
print(f"built {len(out)} pairs | in-DBD {sum(r['in_dbd'] for r in out)} | spec.change Yes {sum(r['spec_change'] for r in out)}")
print("by family (n, nYes):", {k: tuple(v) for k, v in sorted(fam.items())})
print("skipped genes:", {k: sorted(set(v)) for k, v in skipped.items()})
print("saved results/mutation_benchmark/crossfamily_pairs.json")
