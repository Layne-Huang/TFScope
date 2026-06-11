#!/usr/bin/env python
"""Run pwm_rosetta mutation scan directly on crystal structure PDB files.

For each entry in the 130 test set:
  1. rename_chains on the crystal PDB (protein→A, DNA→B)
  2. Load with PyRosetta
  3. Compute WT interface ddG
  4. Mutate each DNA position to 3 alternatives, minimize locally, compute ddG
  5. Convert energies → PPM/PWM via Boltzmann
  6. Save CSV + calibrated_pwm.npz aligned to ground truth length

No structure generation (no AF3/Boltz, no global relax). Crystal PDBs are already downloaded.

Output: /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test/

Usage:
    python scripts/run_pwm_rosetta_crystal.py
    python scripts/run_pwm_rosetta_crystal.py --only 1a1g_A_Egr1.MA0162.1
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pwm_rosetta")

CRYSTAL_PDB_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/crystal_pdbs"
OUT_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
PARQUET = "data/processed/tf_pwm_deeppbs_only.parquet"
DEEPPBS_NPZ_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
TAU = 1.5
BASES = ["A", "C", "G", "T"]
DNA_RES = {"DA", "DT", "DG", "DC"}


def align_pwm_to_ground_truth(arr, true_pwm):
    """Slide arr over true_pwm and return the best-aligned slice of length len(true_pwm)."""
    from scipy.stats import pearsonr

    L_pred = len(arr)
    L_true = len(true_pwm)
    if L_pred == L_true:
        return arr
    if L_pred < L_true:
        return arr  # can't trim; caller handles length mismatch

    best_r = -2.0
    best_start = 0
    for start in range(L_pred - L_true + 1):
        flat_pred = arr[start : start + L_true].flatten()
        flat_true = true_pwm.flatten()
        if flat_pred.std() < 1e-8 or flat_true.std() < 1e-8:
            continue
        r = pearsonr(flat_pred, flat_true)[0]
        if r > best_r:
            best_r = r
            best_start = start

    return arr[best_start : best_start + L_true]


def energies_to_ppm_pwm(energies, tau):
    """Convert (L, 4) energy matrix → (PPM, PWM) via Boltzmann with pseudocount."""
    deltaE = energies - energies.min(axis=1, keepdims=True)
    PPM = np.exp(-deltaE / tau)
    PPM /= PPM.sum(axis=1, keepdims=True)
    PPM_safe = np.clip(PPM, 1e-6, 1.0)
    PPM_safe /= PPM_safe.sum(axis=1, keepdims=True)
    PWM = np.log2(PPM_safe / 0.25)
    return PPM, PWM


def find_binding_dna_chains(pdb_path, pdb_id, chain_id, npz_dir=DEEPPBS_NPZ_DIR, s1_thresh=1.5):
    """Return (strand1_chain_id, strand2_chain_id) using DeepPBS NPZ V_dna coordinates.

    Identifies the two DNA chains in the crystal PDB that actually bind to the
    specified protein chain by matching P-atom coordinates to the NPZ V_dna array.
    Falls back to all DNA chains if no NPZ match found.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    model = struct[0]

    # Collect P-atom coords per DNA chain
    chain_p = {}
    for chain in model:
        p_coords = []
        for res in chain:
            if res.resname.strip() in DNA_RES and "P" in res:
                p_coords.append(res["P"].get_vector().get_array())
        if p_coords:
            chain_p[chain.id] = np.array(p_coords)

    all_dna_chains = list(chain_p.keys())
    if len(all_dna_chains) <= 2:
        return all_dna_chains  # nothing to filter

    # Find NPZ file for this entry
    prefix = pdb_id + "_" + chain_id + "_"
    candidates = [f for f in os.listdir(npz_dir) if f.startswith(prefix)]
    if not candidates:
        print(f"  [WARN] no DeepPBS NPZ found for {pdb_id}_{chain_id} — using all DNA chains")
        return all_dna_chains

    # Try each NPZ candidate; pick the one where min_dist to s1 is smallest
    best_s1, best_s2, best_d1 = None, None, float("inf")
    for cand in candidates:
        try:
            npz = np.load(os.path.join(npz_dir, cand), allow_pickle=True)
            ref = npz["V_dna"][:, 0, :]  # (N, 3) P-like atom coords of strand1
        except Exception:
            continue

        min_dists = {}
        for cid, p_arr in chain_p.items():
            d = np.linalg.norm(p_arr[:, None, :] - ref[None, :, :], axis=2).min()
            min_dists[cid] = d

        sorted_chains = sorted(min_dists, key=min_dists.get)
        d1 = min_dists[sorted_chains[0]]
        if d1 < best_d1:
            best_d1 = d1
            best_s1 = sorted_chains[0]
            best_s2 = sorted_chains[1] if len(sorted_chains) > 1 else None

    if best_s1 is None or best_d1 >= s1_thresh:
        print(f"  [WARN] NPZ coords don't match crystal PDB (min_dist={best_d1:.1f}Å) — using all DNA chains")
        return all_dna_chains

    result = [best_s1]
    if best_s2:
        result.append(best_s2)
    print(f"  Binding DNA chains from NPZ: {result} (s1_dist={best_d1:.2f}Å, discarding {[c for c in all_dna_chains if c not in result]})")
    return result


def filter_pdb_to_binding_chains(pdb_path, out_path, protein_chain_id, dna_chain_ids):
    """Write a PDB keeping only protein_chain + specified dna_chain_ids."""
    from Bio.PDB import PDBParser, PDBIO, Select

    class ChainSelect(Select):
        def __init__(self, keep):
            self.keep = set(keep)
        def accept_chain(self, chain):
            return chain.id in self.keep

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    io = PDBIO()
    io.set_structure(struct)
    keep = {protein_chain_id} | set(dna_chain_ids)
    io.save(out_path, ChainSelect(keep))


def extract_first_model(pdb_path, out_path):
    """Write a single-model PDB from an NMR multi-model PDB (keeps MODEL 1 only)."""
    from Bio.PDB import PDBParser, PDBIO, Select

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    models = list(structure.get_models())
    if len(models) <= 1:
        return pdb_path  # already single-model, no-op

    # Keep only first model
    for m in models[1:]:
        structure.detach_child(m.id)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    print(f"  NMR structure: kept model 1 of {len(models)} → {out_path}")
    return out_path


def get_binding_positions(pdb_id, chain_id, renamed_pdb, npz_dir=DEEPPBS_NPZ_DIR, coord_thresh=5.0):
    """Return 1-based strand1 positions to scan, restricted to the NPZ binding site.

    Uses dna_mask + V_dna coordinates from the DeepPBS NPZ to identify which
    strand1 residues (in the renamed PDB's chain B) are in the actual binding site.
    Returns None if NPZ not found or all positions match (no restriction needed).
    """
    from Bio.PDB import PDBParser

    prefix = pdb_id + "_" + chain_id + "_"
    candidates = [f for f in os.listdir(npz_dir) if f.startswith(prefix)]
    if not candidates:
        return None

    # Use the NPZ with most masked positions (most informative)
    best_npz, best_mask_count = None, 0
    for cand in candidates:
        try:
            npz = np.load(os.path.join(npz_dir, cand), allow_pickle=True)
            n = int(npz["dna_mask"][0].sum())
            if n > best_mask_count:
                best_mask_count = n
                best_npz = npz
        except Exception:
            continue

    if best_npz is None:
        return None

    dna_mask = best_npz["dna_mask"][0]      # (N,) bool for strand1
    ref_coords = best_npz["V_dna"][:, 0, :] # (N, 3) P-like atom coords

    masked_idx = np.where(dna_mask)[0]       # NPZ indices of binding positions
    if len(masked_idx) == len(dna_mask):
        return None  # all positions masked — no restriction needed

    masked_ref = ref_coords[masked_idx]      # (K, 3) coords of binding positions

    # Get strand1 P-atom coords from the renamed PDB (chain B, first half)
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", renamed_pdb)
    dna_residues = [res for res in struct[0]["B"] if res.resname.strip() in DNA_RES]
    strand1 = dna_residues[: len(dna_residues) // 2]

    strand1_p = []
    for res in strand1:
        if "P" in res:
            strand1_p.append((len(strand1_p), res["P"].get_vector().get_array()))

    if not strand1_p:
        return None

    p_indices, p_coords = zip(*strand1_p)
    p_coords = np.array(p_coords)  # (M, 3)

    # For each binding position in NPZ, find closest strand1 residue
    binding_pos = set()
    for ref_c in masked_ref:
        dists = np.linalg.norm(p_coords - ref_c, axis=1)
        best = int(np.argmin(dists))
        if dists[best] < coord_thresh:
            binding_pos.add(p_indices[best] + 1)  # 1-based position in strand1

    if not binding_pos or len(binding_pos) == len(strand1):
        return None

    result = sorted(binding_pos)
    print(f"  Binding site positions (1-based strand1): {result} out of {len(strand1)} total")
    return result


def run_mutation_scan(wt_pose, tau=TAU, scan_positions=None):
    """Run mutation scan on wt_pose; return (seq, energies, PPM, PWM).

    scan_positions: list of 1-based strand1 positions to scan (None = all).
    """
    from pwm_hybrid.rosetta.structure import extract_dna_sequence, get_dna_strand_positions
    from pwm_hybrid.rosetta.mutations import mutate_dna_base, minimize_around_mutation
    from pwm_hybrid.rosetta.scoring import calculate_interface_ddg

    seq = extract_dna_sequence(wt_pose)
    L = len(seq)
    if L == 0:
        raise ValueError("No DNA sequence found in pose")

    wt_ddg = calculate_interface_ddg(wt_pose)
    print(f"  WT ddG = {wt_ddg:.3f}  seq = {seq}  len={L}")

    energies = np.full((L, 4), wt_ddg)
    for i, base in enumerate(seq):
        energies[i][BASES.index(base)] = wt_ddg

    positions_to_scan = scan_positions if scan_positions else list(range(1, L + 1))
    if scan_positions:
        print(f"  Scanning {len(positions_to_scan)}/{L} positions (binding site only)")

    rows = []
    for pos in positions_to_scan:
        orig_base = seq[pos - 1]
        for new_base in BASES:
            if new_base == orig_base:
                continue
            try:
                mut_pose = mutate_dna_base(wt_pose, pos, new_base)
                minimize_around_mutation(mut_pose, pos)
                mut_ddg = calculate_interface_ddg(mut_pose)
                b_idx = BASES.index(new_base)
                energies[pos - 1][b_idx] = mut_ddg
                rows.append({
                    "position": pos,
                    "original": orig_base,
                    "mutant": new_base,
                    "wt_ddg": wt_ddg,
                    "mut_ddg": mut_ddg,
                    "ddg_diff": mut_ddg - wt_ddg,
                })
                print(f"    pos={pos} {orig_base}->{new_base}  ddg={mut_ddg:.3f}")
            except Exception as e:
                print(f"    [WARN] pos={pos} {orig_base}->{new_base}: {e}")

    PPM, PWM = energies_to_ppm_pwm(energies, tau)
    return seq, energies, PPM, PWM, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only process these entry names (no .txt suffix)")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--crystal-dir", default=CRYSTAL_PDB_DIR)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run even if CSV/NPZ already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import pyrosetta
    from pyrosetta import pose_from_pdb
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    from pwm_hybrid.rosetta.structure import rename_chains

    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")

    # Load ground truth PWMs
    df_pwm = pd.read_parquet(args.parquet)
    pwm_lookup = {}
    for _, row in df_pwm.iterrows():
        raw = row["pwm"]
        if isinstance(raw, bytes):
            arr = np.frombuffer(raw, dtype=np.float32).reshape(4, -1).T  # (L, 4)
        else:
            arr = np.array(raw, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 4:
                arr = arr.T
        pwm_lookup[row["filename"]] = arr

    with open(args.split) as f:
        test_ids = json.load(f)["test"]

    if args.only:
        only_set = set(args.only)
        test_ids = [t for t in test_ids if t.replace(".txt", "") in only_set]

    ok, failed = 0, []

    for tid in test_ids:
        entry_name = tid.replace(".txt", "")
        pdb_path = os.path.join(args.crystal_dir, entry_name + ".pdb")
        work_dir = os.path.join(args.out_dir, entry_name)
        csv_path = os.path.join(work_dir, "pwm_results_hybrid.csv")
        npz_path = os.path.join(work_dir, "calibrated_pwm.npz")

        if not os.path.exists(pdb_path):
            print(f"[SKIP] {entry_name}: crystal PDB not found")
            failed.append((tid, "no_pdb"))
            continue

        if os.path.exists(npz_path) and not args.overwrite:
            print(f"[skip] {entry_name}: NPZ exists")
            ok += 1
            continue

        os.makedirs(work_dir, exist_ok=True)
        print(f"\n=== {entry_name} ===")

        # Step 1a: strip NMR extra models (keep model 1 only)
        single_model_pdb = os.path.join(work_dir, "model1.pdb")
        try:
            source_pdb = extract_first_model(pdb_path, single_model_pdb)
        except Exception as e:
            print(f"[ERROR] extract_first_model: {e}")
            failed.append((tid, f"extract_first_model: {e}"))
            continue

        # Step 1b: filter to only the 2 DNA chains that bind this protein chain
        parts = entry_name.split("_")
        pdb_id, chain_id = parts[0], parts[1]
        binding_dna = find_binding_dna_chains(source_pdb, pdb_id, chain_id)
        filtered_pdb = os.path.join(work_dir, "filtered.pdb")
        try:
            filter_pdb_to_binding_chains(source_pdb, filtered_pdb, chain_id, binding_dna)
        except Exception as e:
            print(f"[ERROR] filter_pdb_to_binding_chains: {e}")
            failed.append((tid, f"filter_pdb: {e}"))
            continue

        # Step 1c: rename chains (protein→A, DNA→B)
        renamed_pdb = os.path.join(work_dir, "renamed.pdb")
        try:
            rename_chains(filtered_pdb, renamed_pdb)
        except Exception as e:
            print(f"[ERROR] rename_chains: {e}")
            failed.append((tid, f"rename_chains: {e}"))
            continue

        # Step 2: load pose
        try:
            wt_pose = pose_from_pdb(renamed_pdb)
        except Exception as e:
            print(f"[ERROR] pose_from_pdb: {e}")
            failed.append((tid, f"pose_from_pdb: {e}"))
            continue

        # Step 3: identify binding site positions, then run mutation scan
        scan_positions = get_binding_positions(pdb_id, chain_id, renamed_pdb)

        if os.path.exists(csv_path) and not args.overwrite:
            print(f"  CSV exists — loading")
            from pwm_hybrid import pwm_from_hybrid_csv
            seq, energies, PPM, PWM = pwm_from_hybrid_csv(csv_path, tau=args.tau)[0:4]
            rows = None
        else:
            try:
                seq, energies, PPM, PWM, rows = run_mutation_scan(
                    wt_pose, tau=args.tau, scan_positions=scan_positions)
            except Exception as e:
                print(f"[ERROR] mutation scan: {e}")
                failed.append((tid, f"scan: {e}"))
                continue

            # Save CSV
            if rows:
                pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Step 4: align to ground truth length and save NPZ
        true_pwm = pwm_lookup.get(tid)
        if true_pwm is not None:
            PPM_aligned = align_pwm_to_ground_truth(PPM, true_pwm)
            PWM_aligned = align_pwm_to_ground_truth(PWM, true_pwm)
        else:
            PPM_aligned = PPM
            PWM_aligned = PWM

        np.savez(npz_path,
                 pwm=PWM_aligned,
                 ppm=PPM_aligned,
                 true_pwm=true_pwm if true_pwm is not None else np.array([]),
                 filename=tid)
        print(f"[ok] {entry_name}  pred_len={len(PWM_aligned)}"
              + (f"  true_len={len(true_pwm)}" if true_pwm is not None else ""))
        ok += 1

    print(f"\nDone: {ok}/{len(test_ids)} completed → {args.out_dir}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, reason in failed:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
