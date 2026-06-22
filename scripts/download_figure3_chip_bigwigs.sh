#!/usr/bin/env bash
# Download the ChIP-seq bigWig signal tracks + blacklist used by Figure 3 panel b.
# Idempotent: skips files that already exist and are non-empty. Stable accession-based URLs.
#
# Usage:  bash scripts/download_figure3_chip_bigwigs.sh [DEST_DIR]
# Default DEST_DIR matches configs/figure3_orphan_tf_validation.yaml (/data1/leihuang/chip_bigwigs).
set -euo pipefail
DEST="${1:-/data1/leihuang/chip_bigwigs}"
mkdir -p "$DEST"
echo "Downloading Figure 3 bigWigs -> $DEST"

# name  url
declare -a FILES=(
  # ENCODE fold-change-over-control bigWigs (GRCh38)
  "ADNP.fc.bigWig|https://www.encodeproject.org/files/ENCFF254FCX/@@download/ENCFF254FCX.bigWig"   # ENCSR440VKE K562
  "ZHX2.fc.bigWig|https://www.encodeproject.org/files/ENCFF037HXV/@@download/ENCFF037HXV.bigWig"   # ENCSR407BEZ HepG2
  "ZHX3.fc.bigWig|https://www.encodeproject.org/files/ENCFF091KOP/@@download/ENCFF091KOP.bigWig"   # ENCSR367KYL HepG2
  # GEO GSE280248 SOHLH1 HEK293 raw-coverage bigWig
  "SOHLH1.bw|https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8589nnn/GSM8589942/suppl/GSM8589942_THC0345GPZD_SOHLH1_ChIP1_hg38_PE.bw"
)

for entry in "${FILES[@]}"; do
  name="${entry%%|*}"; url="${entry##*|}"
  out="$DEST/$name"
  if [[ -s "$out" ]]; then echo "  [skip] $name (exists)"; continue; fi
  echo "  [get ] $name"
  curl -fsSL "$url" -o "$out"
  echo "        $(du -h "$out" | cut -f1)  $name"
done

# ENCODE hg38 blacklist v2
BL="$DEST/hg38-blacklist.bed"
if [[ ! -s "$BL" ]]; then
  echo "  [get ] hg38-blacklist.v2"
  curl -fsSL "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz" -o "$BL.gz"
  gunzip -f "$BL.gz"
fi
echo "Done. Update bigwig/blacklist paths in configs/figure3_orphan_tf_validation.yaml if DEST != default."
