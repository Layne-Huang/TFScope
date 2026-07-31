"""Build Boltz-2 inputs for the v24-vs-DeepPBS foldability head-to-head.

For each of the 41 TFs, two complexes are folded: protein + v24-consensus dsDNA,
and protein + DeepPBS-consensus dsDNA. The protein is IDENTICAL across the two
conditions (and some proteins repeat across TFs), so the protein MSA is computed
ONCE per unique sequence and reused via the YAML `msa:` field — the strategy the
run script implements (this builder just lays out the jobs grouped by protein).

DeepPBS consensus is reused verbatim from the existing AF3 manifest (it does not
depend on our model). v24 consensus is read from results/af3_v24_foldability/.
"""
import json, os, hashlib
import pandas as pd

MAN = "/data1/leihuang/project/TFScope/AF3_consensus_folding/jobs_manifest.csv"
V24 = "results/af3_v24_foldability/v24_consensus.json"
OUT = "/data1/leihuang/TFScope_store/boltz_v24"
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def rc(s): return ''.join(COMP.get(b, 'N') for b in reversed(s))
def phash(seq): return hashlib.md5(seq.encode()).hexdigest()[:10]


def yaml_for(protein, dna_fwd, msa_path=None):
    msa_line = f"\n      msa: {msa_path}" if msa_path else ""
    return (f"version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {protein}{msa_line}\n"
            f"  - dna:\n      id: B\n      sequence: {dna_fwd}\n"
            f"  - dna:\n      id: C\n      sequence: {rc(dna_fwd)}\n")


def main():
    os.makedirs(f"{OUT}/inputs", exist_ok=True)
    os.makedirs(f"{OUT}/msa", exist_ok=True)
    man = pd.read_csv(MAN)
    v24 = {r["gene"]: r for r in json.load(open(V24))}
    # DeepPBS consensus from manifest (source==DeepPBS, strand1 = fwd dna)
    dpp = {r.gene: r for r in man[man.source == "DeepPBS"].drop_duplicates("gene").itertuples()}

    jobs, proteins = [], {}
    for gene, vr in v24.items():
        prot = vr["protein_seq"]; ph = phash(prot)
        proteins[ph] = prot
        for src, dna in [("v24", vr["core_dna"]),
                         ("deeppbs", getattr(dpp.get(gene), "strand1", None))]:
            if not dna:
                continue
            name = f"{gene}_{src}"
            jobs.append({"name": name, "gene": gene, "family": vr["family"],
                         "source": src, "protein_hash": ph, "protein": prot,
                         "dna": dna, "dna_len": len(dna)})
    # write YAMLs (msa path per protein — filled by the run script after MSA precompute)
    for j in jobs:
        msa_path = f"{OUT}/msa/{j['protein_hash']}.a3m"
        open(f"{OUT}/inputs/{j['name']}.yaml", "w").write(
            yaml_for(j["protein"], j["dna"], msa_path=msa_path))
    json.dump({"jobs": jobs, "proteins": proteins,
               "n_jobs": len(jobs), "n_unique_proteins": len(proteins)},
              open(f"{OUT}/plan.json", "w"), indent=2)
    print(f"jobs: {len(jobs)}  unique proteins (MSA computed once each): {len(proteins)}")
    print(f"  v24 folds: {sum(j['source']=='v24' for j in jobs)}  "
          f"deeppbs folds: {sum(j['source']=='deeppbs' for j in jobs)}")
    print(f"wrote YAMLs -> {OUT}/inputs/, plan -> {OUT}/plan.json")


if __name__ == "__main__":
    main()
