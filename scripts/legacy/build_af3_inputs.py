#!/usr/bin/env python
"""Stage 2: build AlphaFold3 input JSONs for protein + double-stranded consensus DNA.

Reads results/af3_pipeline/consensus.json and writes one AF3 JSON per TF into
results/af3_pipeline/af3_inputs/{safe_name}.json

Each JSON contains the TF protein chain (A) plus the consensus DNA as two
complementary strands (B = forward, C = reverse complement) so AF3 builds dsDNA.
The forward DNA is the v7 core consensus, padded with a neutral GC flank to a
minimum length so AF3 has stable duplex context.
"""
import argparse, json, os, re

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
MIN_DNA = 12   # minimum dsDNA length for stable AF3 duplex
FLANK_UNIT = "GC"   # neutral, balanced flank


def revcomp(s):
    return "".join(COMP[b] for b in reversed(s))


def pad_dna(core):
    """Pad the core consensus symmetrically with a neutral GC flank to >= MIN_DNA."""
    if len(core) >= MIN_DNA:
        return core, 0
    need = MIN_DNA - len(core)
    left_n = need // 2
    right_n = need - left_n
    left  = (FLANK_UNIT * ((left_n // 2) + 1))[:left_n]
    right = (FLANK_UNIT * ((right_n // 2) + 1))[:right_n]
    return left + core + right, left_n


def safe_name(fn):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", fn.replace(".txt", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default="results/af3_pipeline/consensus.json")
    ap.add_argument("--out-dir",  default="results/af3_pipeline/af3_inputs")
    ap.add_argument("--use-core", action="store_true",
                    help="Use the bare core consensus (padded) instead of the flanked consensus")
    ap.add_argument("--limit", type=int, default=0, help="Only build first N (for pilot)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only build these filenames (pilot subset)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.consensus) as f:
        records = json.load(f)

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

        af3 = {
            "name": name,
            "sequences": [
                {"protein": {"id": "A", "sequence": r["protein_seq"]}},
                {"dna":     {"id": "B", "sequence": fwd}},
                {"dna":     {"id": "C", "sequence": rev}},
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w") as fout:
            json.dump(af3, fout, indent=2)
        manifest.append({
            "filename": r["filename"], "af3_name": name, "json": path,
            "fwd_dna": fwd, "core": core, "core_offset": core_offset,
            "pred_len": r["pred_len"], "true_len": r["true_len"],
        })

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} AF3 input JSONs to {args.out_dir}/")
    for m in manifest[:5]:
        print(f"  {m['af3_name']:30s} fwd_dna={m['fwd_dna']}  (core_offset={m['core_offset']})")


if __name__ == "__main__":
    main()
