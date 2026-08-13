#!/usr/bin/env python3
"""Generate 36 Boltz-2 FASTA inputs for RELA DNA-position scanning.

For each (position, non-WT base) pair on the forward strand GGGAAATTCCCG,
produce a FASTA with:
  - Chain A: RELA protein, pointing to pre-computed MSA (no server call)
  - Chain B: mutant forward strand
  - Chain C: corresponding reverse complement

Output: results/boltz_rela_dna_scan/inputs/{mut_name}.fasta
        results/boltz_rela_dna_scan/mutation_manifest.json
"""
import os, json

FWD_SEQ  = "GGGAAATTCCCG"
PROTEIN  = ("PYVEIIEQPKQRGMRFRYKCEGRSAGSIPGERSTDTTKTHPTIKINGYTGPGTVRISLVTKDPPHR"
            "PHPHELVGKDCRDGYYEADLCPDRSIHSFQNLGIQCVKKRDLEQAISQRIQTNNNPFHVPIEEQRG"
            "DYDLNAVRLCFQVTVRDPAGRPLLLTPVLSHPIFDNRAPNTAELKICRVNRNSGSCLGGDEIFLLCD"
            "KVQKEDIEVYFTGPGWEARGSFSQADVHRQVAIVFRTPPYADPSLQAPVRVSMQLRRPSDRELSEPM"
            "EFQYLPD")
MSA_CSV  = ("/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/"
            "rela_af3_msa/boltz_results_rela_msa/msa/rela_msa_0.csv")

COMP = {"A":"T","T":"A","G":"C","C":"G"}

def rev_comp(seq):
    return "".join(COMP[b] for b in reversed(seq))

OUT_DIR = "results/boltz_rela_dna_scan/inputs"
os.makedirs(OUT_DIR, exist_ok=True)

mutations = []
for pos0, wt_base in enumerate(FWD_SEQ):
    for mut_base in "ACGT":
        if mut_base == wt_base:
            continue
        mut_fwd = FWD_SEQ[:pos0] + mut_base + FWD_SEQ[pos0+1:]
        mut_rev = rev_comp(mut_fwd)
        name    = f"rela_{wt_base}{pos0+1}{mut_base}"
        fasta   = os.path.join(OUT_DIR, f"{name}.fasta")
        with open(fasta, "w") as f:
            f.write(f">A|protein|{MSA_CSV}\n{PROTEIN}\n")
            f.write(f">B|dna\n{mut_fwd}\n")
            f.write(f">C|dna\n{mut_rev}\n")
        mutations.append({
            "name":    name,
            "pos":     pos0 + 1,
            "wt_base": wt_base,
            "mut_base":mut_base,
            "fwd_dna": mut_fwd,
            "rev_dna": mut_rev,
            "fasta":   fasta,
        })

# Also write WT entry (needed for ΔΔG reference)
wt_rev = rev_comp(FWD_SEQ)
wt_fasta = os.path.join(OUT_DIR, "rela_WT.fasta")
with open(wt_fasta, "w") as f:
    f.write(f">A|protein|{MSA_CSV}\n{PROTEIN}\n")
    f.write(f">B|dna\n{FWD_SEQ}\n")
    f.write(f">C|dna\n{wt_rev}\n")
mutations.insert(0, {
    "name": "rela_WT", "pos": 0, "wt_base": "", "mut_base": "",
    "fwd_dna": FWD_SEQ, "rev_dna": wt_rev, "fasta": wt_fasta,
})

manifest_path = "results/boltz_rela_dna_scan/mutation_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(mutations, f, indent=2)

print(f"Generated {len(mutations)-1} mutant + 1 WT FASTAs → {OUT_DIR}/")
print(f"Manifest → {manifest_path}")
