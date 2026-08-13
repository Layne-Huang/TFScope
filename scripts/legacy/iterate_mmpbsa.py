#!/usr/bin/env python
"""Iterative Boltz2+MM-GBSA calibration for a single TF.

Pipeline per round:
  1. Generate consensus DNA from current PWM (argmax per position)
  2. Write FASTA with new DNA sequence (both strands)
  3. Run Boltz2 with --use_msa_server
  4. Run MM-GBSA scan on new structure
  5. Check convergence: stop if consensus unchanged

Usage:
    python scripts/iterate_mmpbsa.py \
        --af3-name 2e42_A_CEBPB.CEBPB_HUMAN.H11MO.0.A \
        --start-dna TATTTAAAAATA \
        --max-rounds 5 \
        --out results/mmpbsa_iterate_cebpb_wrong \
        --label wrong_start

    python scripts/iterate_mmpbsa.py \
        --af3-name 2e42_A_CEBPB.CEBPB_HUMAN.H11MO.0.A \
        --start-dna TATTGCGCAATA \
        --max-rounds 5 \
        --out results/mmpbsa_iterate_cebpb_correct \
        --label correct_start
"""

import argparse, json, os, subprocess, sys, time
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, "src")
sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
os.environ["PYTHONUNBUFFERED"] = "1"

BOLTZ_CACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.boltz"
BASES       = list("ACGT")
COMP        = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
MAX_L       = 20


def revcomp(seq):
    return ''.join(COMP.get(b,'N') for b in reversed(seq))


def consensus_from_pwm(pwm, mask):
    """Argmax per valid position → consensus string."""
    L = int(mask.sum())
    return ''.join(BASES[pwm[:, i].argmax()] for i in range(L))


def write_fasta(af3_name, protein_seq, dna_sense, out_dir):
    """Write Boltz-format FASTA with protein + sense + antisense strands."""
    fa = os.path.join(out_dir, f"{af3_name}.fasta")
    anti = revcomp(dna_sense)
    with open(fa, "w") as f:
        f.write(f">A|protein\n{protein_seq}\n")
        f.write(f">B|dna\n{dna_sense}\n")
        f.write(f">C|dna\n{anti}\n")
    return fa


def run_boltz(fasta_dir, out_dir):
    """Run Boltz2 with MSA server. Returns path to output CIF."""
    cmd = [
        "boltz", "predict", fasta_dir,
        "--out_dir",        out_dir,
        "--cache",          BOLTZ_CACHE,
        "--model",          "boltz2",
        "--output_format",  "mmcif",
        "--use_msa_server",
        "--no_kernels",
        "--accelerator",    "gpu",
        "--devices",        "1",
        "--override",
    ]
    print(f"  Running Boltz: {' '.join(cmd[-4:])}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Boltz stderr: {result.stderr[-500:]}", flush=True)
        raise RuntimeError(f"Boltz failed: exit {result.returncode}")
    return result


def find_cif(boltz_out, af3_name):
    import glob
    patterns = [
        os.path.join(boltz_out, "**", f"{af3_name}_model_0.cif"),
        os.path.join(boltz_out, "**", f"*model_0.cif"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def get_iptm(boltz_out, af3_name):
    import glob
    for pat in [
        os.path.join(boltz_out, "**", f"confidence_{af3_name}_model_0.json"),
        os.path.join(boltz_out, "**", "confidence_*_model_0.json"),
    ]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            d = json.load(open(hits[0]))
            return d.get("iptm", 0.0)
    return 0.0


def run_mmpbsa(cif_path, af3_name, out_dir, core_offset, core_len, tau=1.5, min_iter=300):
    import importlib.util, os as _os
    spec = importlib.util.spec_from_file_location(
        "run_mmpbsa_scan",
        _os.path.join(_os.path.dirname(__file__), "run_mmpbsa_scan.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    scan_dna_positions = mod.scan_dna_positions
    ppm, ddg, wt_seq = scan_dna_positions(cif_path, core_offset, core_len, tau=tau, min_iter=min_iter)
    if ppm is None:
        return None, None, wt_seq
    L = min(ppm.shape[0], MAX_L)
    core_ppm = np.full((4, MAX_L), 0.25, dtype=np.float32)
    core_ppm[:, :L] = ppm[:L].T
    mask = np.zeros(MAX_L); mask[:L] = 1.0
    out_path = os.path.join(out_dir, f"{af3_name}.npz")
    np.savez_compressed(out_path, core_ppm=core_ppm, ddg=ddg,
                        core_offset=core_offset, core_len=core_len,
                        wt_seq=wt_seq, tau=tau)
    return core_ppm, mask, wt_seq


def load_target(af3_name, split_path, data_path):
    fn = af3_name + ".txt"
    with open(split_path) as f:
        test_fns = json.load(f)["test"]
    import pandas as pd
    df = pd.read_parquet(data_path)
    row = df[df["filename"] == fn].iloc[0]
    pwm_bytes = row["pwm"]
    if isinstance(pwm_bytes, bytes):
        targ = np.frombuffer(pwm_bytes, dtype=np.float32).reshape(4, -1)
    else:
        return None, None
    L = min(targ.shape[1], MAX_L)
    target = np.full((4, MAX_L), 0.25); target[:, :L] = targ[:, :L]
    mask = np.zeros(MAX_L); mask[:L] = 1.0
    return target, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af3-name",   required=True)
    ap.add_argument("--start-dna",  required=True, help="Initial DNA sense sequence")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--label",      default="run")
    ap.add_argument("--tau",        type=float, default=1.5)
    ap.add_argument("--min-iter",   type=int,   default=300)
    ap.add_argument("--split",   default="data/processed/splits/deeppbs_only/benchmark_no_val.json")
    ap.add_argument("--data",    default="data/processed/tf_pwm_aug_dbd.parquet")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load protein sequence from existing FASTA
    fasta_src = (f"results/boltz_pipeline_v10/inputs_all_msa/{args.af3_name}.fasta")
    protein_seq = None
    if os.path.exists(fasta_src):
        for line in open(fasta_src):
            if line.startswith(">A"):
                protein_seq = open(fasta_src).readlines()[1].strip()
                break

    # Get core offset/len from existing Rosetta run
    cal_path = (f"/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/"
                f"pwm_rosetta_runs/aug_v10/{args.af3_name}/calibrated_pwm.npz")
    cal_d       = np.load(cal_path)
    core_offset = int(cal_d["core_offset"])
    core_len    = int(cal_d["core_len"])

    # Load target PWM for metrics
    target, t_mask = load_target(args.af3_name, args.split, args.data)

    def metrics(pwm, mask):
        idx = mask.astype(bool)
        r, _ = pearsonr(target[:,idx].flatten(), pwm[:,idx].flatten())
        return r

    # ── Iteration loop ────────────────────────────────────────────────────────
    current_dna = args.start_dna
    prev_consensus = None
    history = []

    print(f"\n{'='*60}", flush=True)
    print(f"Iterative MM-GBSA: {args.af3_name}", flush=True)
    print(f"Start DNA: {current_dna}  label={args.label}", flush=True)
    print(f"True motif: TTGCGCAA (JASPAR CEBPB)", flush=True)
    print(f"{'='*60}", flush=True)

    for rnd in range(args.max_rounds):
        print(f"\n─── Round {rnd} ─────────────────────────────────────", flush=True)
        print(f"  DNA input: {current_dna}", flush=True)

        rnd_dir = os.path.join(args.out, args.label, f"round{rnd:02d}")
        os.makedirs(rnd_dir, exist_ok=True)

        # 1. Write FASTA
        fasta_dir = os.path.join(rnd_dir, "fasta")
        os.makedirs(fasta_dir, exist_ok=True)
        write_fasta(args.af3_name, protein_seq, current_dna, fasta_dir)

        # 2. Run Boltz2 + MSA
        boltz_out = os.path.join(rnd_dir, "boltz_out")
        t0 = time.time()
        run_boltz(fasta_dir, boltz_out)
        cif = find_cif(boltz_out, args.af3_name)
        iptm = get_iptm(boltz_out, args.af3_name)
        print(f"  Boltz done in {time.time()-t0:.0f}s  iPTM={iptm:.3f}  CIF={cif}", flush=True)

        if cif is None:
            print("  [FAIL] No CIF produced", flush=True)
            break

        # 3. MM-GBSA scan
        mmpbsa_dir = os.path.join(rnd_dir, "mmpbsa")
        os.makedirs(mmpbsa_dir, exist_ok=True)
        t0 = time.time()
        ppm, mask, wt_seq = run_mmpbsa(cif, args.af3_name, mmpbsa_dir,
                                        core_offset, core_len, args.tau, args.min_iter)
        print(f"  MM-GBSA done in {time.time()-t0:.0f}s", flush=True)

        if ppm is None:
            print("  [FAIL] MM-GBSA returned None", flush=True)
            break

        # 4. Evaluate
        new_consensus = consensus_from_pwm(ppm, mask)
        r_val = metrics(ppm, mask) if target is not None else float('nan')
        print(f"  MM-GBSA consensus: {new_consensus}", flush=True)
        print(f"  Pearson r vs true: {r_val:.4f}", flush=True)

        history.append({
            "round":     rnd,
            "dna_input": current_dna,
            "consensus": new_consensus,
            "iptm":      iptm,
            "pearson_r": r_val,
        })

        # 5. Convergence check
        if new_consensus == prev_consensus:
            print(f"\n  ✓ CONVERGED at round {rnd} (consensus unchanged)", flush=True)
            break

        prev_consensus  = new_consensus
        current_dna     = new_consensus  # use as DNA for next round

    # Save history
    hist_path = os.path.join(args.out, f"history_{args.label}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Summary — {args.label}:", flush=True)
    for h in history:
        print(f"  Round {h['round']}: DNA={h['dna_input']}  "
              f"→ consensus={h['consensus']}  r={h['pearson_r']:.4f}  iPTM={h['iptm']:.3f}", flush=True)
    print(f"History saved: {hist_path}", flush=True)


if __name__ == "__main__":
    main()
