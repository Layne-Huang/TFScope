"""
Orchestration: hybrid AF3 + PyRosetta PWM generation.
"""

import os
import glob

import pandas as pd

from pwm_hybrid.af3 import HAS_AF3, AF3_IMPORT_ERROR, AF3_folding
from pwm_hybrid.rosetta.structure import (
    convert_cif_to_pdb, rename_chains, extract_dna_sequence
)
from pwm_hybrid.rosetta.mutations import mutate_dna_base, minimize_around_mutation
from pwm_hybrid.rosetta.scoring import calculate_interface_ddg
from pwm_hybrid.rosetta import init as _rosetta_init

try:
    from pyrosetta import pose_from_pdb
except ImportError:
    pose_from_pdb = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


def generate_pwm_hybrid(protein_seq, dna_seq, output_dir,
                        af3_output_dir=None, wt_pdb=None,
                        template_json=None, zf_count=0,
                        minimize_local=True, use_relax=False):
    """
    Generate a PWM using the hybrid AF3 + Rosetta approach.

    Parameters
    ----------
    protein_seq : str or list[str]
        Protein sequence(s). Used only when *wt_pdb* is None.
    dna_seq : str or None
        DNA sequence. If None, extracted from the PDB structure.
    output_dir : str
        Directory for output files.
    af3_output_dir : str, optional
        Directory for AF3 wild-type output (created if absent).
    wt_pdb : str, optional
        Pre-existing wild-type PDB.  If provided, AF3 is skipped.
    template_json : str, optional
        AF3 template JSON file.
    zf_count : int
        Number of zinc fingers (passed to AF3).
    minimize_local : bool
        Perform local minimization around each mutation.
    use_relax : bool
        Use full FastRelax (slower).

    Returns
    -------
    pd.DataFrame
        Mutation results with ddG values.
    """
    os.makedirs(output_dir, exist_ok=True)

    if wt_pdb is None and not HAS_AF3:
        raise RuntimeError(
            "AF3 module is required but not available.\n"
            f"Import error: {AF3_IMPORT_ERROR}\n\n"
            "Solutions:\n"
            "1. Provide a wild-type PDB with wt_pdb= (recommended)\n"
            "2. Use a Python 3.11+ environment with af3cli installed\n"
            "3. Run AF3 separately and pass the resulting PDB"
        )

    # ── STEP 1: Wild-type structure ────────────────────────────────────────
    if wt_pdb is None:
        print("=" * 60)
        print("STEP 1: Running AF3 for wild-type complex...")
        print("=" * 60)

        if af3_output_dir is None:
            af3_output_dir = os.path.join(output_dir, "wildtype_af3")
        os.makedirs(af3_output_dir, exist_ok=True)

        AF3_folding(
            protein_seq=protein_seq if isinstance(protein_seq, list) else [protein_seq],
            dna_seq=dna_seq,
            job_name="wildtype",
            output_dir=af3_output_dir,
            template=template_json,
            zf_count=zf_count
        )

        af3_files = glob.glob(os.path.join(af3_output_dir, "wildtype", "wildtype_model.*"))
        if not af3_files:
            raise FileNotFoundError(f"AF3 output not found in {af3_output_dir}")
        wt_structure = af3_files[0]
    else:
        print(f"Using existing wild-type: {wt_pdb}")
        wt_structure = wt_pdb

    print(f"Wild-type structure: {wt_structure}")

    # ── STEP 2: Prepare wild-type pose ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preparing wild-type for mutation analysis...")
    print("=" * 60)

    if wt_structure.endswith('.cif'):
        wt_pdb_path = wt_structure.replace('.cif', '_tmp.pdb')
        convert_cif_to_pdb(wt_structure, wt_pdb_path)
    else:
        wt_pdb_path = wt_structure

    wt_renamed_path = wt_pdb_path.replace('.pdb', '_renamed_pwm_hybrid.pdb')
    metal_free = rename_chains(wt_pdb_path, wt_renamed_path)
    wt_pdb_path = wt_renamed_path

    wt_pose = pose_from_pdb(wt_pdb_path)

    wt_ddg_pre = calculate_interface_ddg(wt_pose)
    print(f"Before relax: Wild-type interface ddG: {wt_ddg_pre:.3f}")

    if minimize_local or use_relax:
        print("Minimizing wild-type structure...")
        scorefxn = _rosetta_init.sfxn
        if use_relax:
            pre = scorefxn(wt_pose)
            print(f"Score before relaxation: {pre:.3f}")
            _rosetta_init.fast_relax.apply(wt_pose)
            post = scorefxn(wt_pose)
            print(f"Score after relaxation: {post:.3f}")
        else:
            from pyrosetta.rosetta.protocols.minimization_packing import MinMover
            from pyrosetta.rosetta.core.kinematics import MoveMap
            movemap = MoveMap()
            movemap.set_bb(True)
            movemap.set_chi(True)
            mm = MinMover()
            mm.score_function(scorefxn)
            mm.movemap(movemap)
            mm.apply(wt_pose)

    wt_ddg = calculate_interface_ddg(wt_pose)
    print(f"Wild-type interface ddG: {wt_ddg:.3f}")

    extracted_seq = extract_dna_sequence(wt_pose)
    print(f"Extracted DNA sequence: {extracted_seq}")

    if not extracted_seq:
        print("\nERROR: No DNA sequence found in the structure!")
        print(f"  Total residues: {wt_pose.total_residue()}")
        for i in range(1, min(wt_pose.total_residue() + 1, 6)):
            print(f"  Residue {i}: {wt_pose.residue(i).name()}")
        return pd.DataFrame()

    if dna_seq is None:
        dna_seq = extracted_seq
    elif extracted_seq.upper() != dna_seq.upper():
        print(f"WARNING: Sequence mismatch! Expected: {dna_seq}, Got: {extracted_seq}")

    # ── STEP 3: Generate all mutations ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating mutations and calculating ddG...")
    print("=" * 60)

    results = []
    na = ['A', 'C', 'G', 'T']
    total_mutations = len(dna_seq) * 3
    pbar = tqdm(total=total_mutations, desc="Processing mutations")

    for i in range(len(dna_seq)):
        original_base = dna_seq[i].upper()
        for j in range(3):
            new_base = na[(na.index(original_base) + j + 1) % 4]
            mut_name = f"{i + 1}_{original_base}to{new_base}".lower()
            pbar.set_description(f"Processing {mut_name}")

            try:
                mut_pose = mutate_dna_base(wt_pose, i + 1, new_base)

                if minimize_local and not use_relax:
                    mut_pose = minimize_around_mutation(mut_pose, i + 1)

                if use_relax:
                    _rosetta_init.fast_relax.apply(mut_pose)

                mut_ddg = calculate_interface_ddg(mut_pose)
                delta_ddg = mut_ddg - wt_ddg

                results.append({
                    'position': i + 1,
                    'original': original_base,
                    'mutant': new_base,
                    'mutation_name': mut_name,
                    'wt_ddg': wt_ddg,
                    'mut_ddg': mut_ddg,
                    'ddG': delta_ddg,
                    'metal_free': metal_free,
                })
            except Exception as e:
                print(f"\nError processing {mut_name}: {e}")
                results.append({
                    'position': i + 1,
                    'original': original_base,
                    'mutant': new_base,
                    'mutation_name': mut_name,
                    'wt_ddg': wt_ddg,
                    'mut_ddg': None,
                    'ddG': None,
                    'error': str(e),
                    'metal_free': metal_free,
                })
            pbar.update(1)

    pbar.close()

    # ── STEP 4: Save results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Saving results...")
    print("=" * 60)

    df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "pwm_results_hybrid.csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    # Clean up temp files
    for f in glob.glob(wt_pdb_path.replace('.pdb', '*_tmp.pdb')):
        try:
            os.remove(f)
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total mutations processed: {len(results)}")
    if len(results) == 0:
        print("WARNING: No mutations were processed!")
        return df

    if 'ddG' in df.columns and df['ddG'].notna().any():
        print(f"Successful: {df['ddG'].notna().sum()}")
        print(f"Failed:     {df['ddG'].isna().sum()}")
        print(f"ddG mean: {df['ddG'].mean():.3f}  std: {df['ddG'].std():.3f}")
        print(f"    min:  {df['ddG'].min():.3f}  max: {df['ddG'].max():.3f}")

    return df
