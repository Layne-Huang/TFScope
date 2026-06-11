#!/usr/bin/env python
"""Stage 2 (Boltz variant): build Boltz-2 FASTA inputs for protein + dsDNA consensus.

Boltz FASTA format (one record per chain):
    >A|protein
    <protein sequence>
    >B|dna
    <forward consensus DNA>
    >C|dna
    <reverse-complement DNA>

Reads results/af3_pipeline/consensus.json, writes one .fasta per TF into
results/boltz_pipeline/inputs/{safe_name}.fasta, plus a manifest mirroring the
AF3 one (so the same pwm_rosetta calibration step can consume Boltz outputs).
"""
import argparse, json, os, re

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
MIN_DNA = 12
FLANK_UNIT = "GC"


def revcomp(s): return "".join(COMP[b] for b in reversed(s))


def pad_dna(core):
    if len(core) >= MIN_DNA:
        return core, 0
    need = MIN_DNA - len(core); left_n = need // 2; right_n = need - left_n
    left  = (FLANK_UNIT * ((left_n // 2) + 1))[:left_n]
    right = (FLANK_UNIT * ((right_n // 2) + 1))[:right_n]
    return left + core + right, left_n


def safe_name(fn): return re.sub(r"[^A-Za-z0-9_.-]", "_", fn.replace(".txt", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default="results/af3_pipeline/consensus.json")
    ap.add_argument("--out-dir",  default="results/boltz_pipeline/inputs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = json.load(open(args.consensus))
    if args.only:
        records = [r for r in records if r["filename"] in set(args.only)]
    if args.limit:
        records = records[: args.limit]

    manifest = []
    for r in records:
        core = r["core_dna"]
        fwd, core_offset = pad_dna(core)
        rev = revcomp(fwd)
        name = safe_name(r["filename"])
        fasta = (
            f">A|protein\n{r['protein_seq']}\n"
            f">B|dna\n{fwd}\n"
            f">C|dna\n{rev}\n"
        )
        path = os.path.join(args.out_dir, f"{name}.fasta")
        with open(path, "w") as f:
            f.write(fasta)
        manifest.append({
            "filename": r["filename"], "af3_name": name, "fasta": path,
            "fwd_dna": fwd, "core": core, "core_offset": core_offset,
            "pred_len": r["pred_len"], "true_len": r["true_len"],
        })

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} Boltz FASTA inputs to {args.out_dir}/")
    for m in manifest[:5]:
        print(f"  {m['af3_name']:30s} fwd_dna={m['fwd_dna']}")


if __name__ == "__main__":
    main()
