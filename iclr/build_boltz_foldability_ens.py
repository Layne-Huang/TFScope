"""Build Boltz-2 inputs for the v24-ENSEMBLE-vs-DeepPBS foldability head-to-head.

Same design as build_boltz_foldability.py, but the TFScope DNA is the 5-seed v24
ENSEMBLE consensus (register-aligned average) instead of the single-seed v24 core.
Only the genes whose ensemble consensus DIFFERS from single-seed are folded here
(as {gene}_ens); for the identical genes the collector reuses the cached
{gene}_v24 fold, and the DeepPBS side is reused verbatim (it never depends on us).
"""
import json, os

V24 = "results/af3_v24_foldability/v24_consensus.json"
ENS = "results/af3_v24_foldability/ens_consensus.json"
OUT = "/data1/leihuang/TFScope_store/boltz_v24_ens"
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def rc(s): return ''.join(COMP.get(b, 'N') for b in reversed(s))


def yaml_for(protein, dna_fwd):
    return (f"version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {protein}\n"
            f"  - dna:\n      id: B\n      sequence: {dna_fwd}\n"
            f"  - dna:\n      id: C\n      sequence: {rc(dna_fwd)}\n")


def main():
    os.makedirs(f"{OUT}/inputs", exist_ok=True)
    prot = {r["gene"]: r["protein_seq"] for r in json.load(open(V24))}
    ens = {r["gene"]: r for r in json.load(open(ENS))}
    jobs = []
    for gene, r in ens.items():
        if r["ens"] == r["v24"]:
            continue                                   # identical -> reuse v24 fold
        name = f"{gene}_ens"
        open(f"{OUT}/inputs/{name}.yaml", "w").write(yaml_for(prot[gene], r["ens"]))
        jobs.append({"name": name, "gene": gene, "family": r["family"],
                     "dna": r["ens"], "dna_len": len(r["ens"])})
    json.dump({"jobs": jobs, "n_jobs": len(jobs)}, open(f"{OUT}/plan.json", "w"), indent=2)
    print(f"wrote {len(jobs)} ensemble-consensus folds to {OUT}/inputs (differing genes only)")


if __name__ == "__main__":
    main()
