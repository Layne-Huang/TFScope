#!/bin/bash
#SBATCH -J af3_zf_rag
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 06:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag_%j.err

# Fold the 4 designed zinc-finger arrays with TFScope's RAG-predicted consensus DNA, via the
# local AF3 pipeline (AF3_single.py -> MMseqs2 MSA -> Singularity SIF; no databases needed).
# Protein = chain A of each design_pdbs/zf_samples/<zf>_model.pdb; DNA = RAG-best consensus.

set -eo pipefail
source /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3tools_venv/bin/activate
cd /n/home13/leihuang/project/TFScope
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag
mkdir -p $OUT
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true

declare -A PROT DNA
PROT[zf21]=ERPFQCRICMRNFSRNFILQRHIRTHTGEKPFQCRICMRNFSQGGNLVRHIRTHTGEKPFQCRICMRNFSQQTGLNVHIRTHTGLR
PROT[zf93]=ERPFQCRICMRNFSAPSKLKRHIRTHTGEKPFQCRICMRNFSHKSSLTRHIRTHTGEKPFQCRICMRNFSQRNALSGHIRTHTGLR
PROT[zf92]=EAPFQCRICMRNFSDSPTLRRHTRTHTGEKPFQCRICMANFSQRSSLVRHLRTHTGEKPFQCRICMANFSERGNLTRHLKTHTGEKG
PROT[zf129]=EAPFQCRICMRNFSDSPTLRRHTRTHTGEKPFQCRICMANFSQRSSLVRHLRTHTGEKPFQCRICMANFSERGNLTRHLKTHTGEKG
DNA[zf21]=AATAAAAAAAATT
DNA[zf93]=AAAAAAAAGTGATT
DNA[zf92]=CCCAGTCGGCCCTT
DNA[zf129]=CCCAGTCGGCCCTT

for s in zf21 zf93 zf92 zf129; do
  echo "=== AF3 $s | DNA ${DNA[$s]} ==="
  python AF3_single.py \
      --protein-seq "${PROT[$s]}" \
      --dna-seq "${DNA[$s]}" \
      --job-name "${s}_rag" \
      --output-dir "$OUT" || echo "FAILED $s"
done
echo "Done at $(date)"
