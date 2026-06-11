#!/usr/bin/env python
"""MM-GBSA DNA base scanning for PWM calibration.

Replaces Rosetta ΔΔG with OpenMM AMBER14/DNA.OL15 + GBn2 implicit solvent.
For each DNA position in the binding core, mutates all 4 bases, minimizes,
and computes ΔΔG = E_complex(mut) - E_complex(wt).

Usage (single sample):
    python scripts/run_mmpbsa_scan.py \
        --cif results/boltz_pipeline_v10/... \
        --af3-name 1a1g_A_Egr1.MA0162.1 \
        --out results/mmpbsa_v10

Usage (batch over all 130 test TFs):
    python scripts/run_mmpbsa_scan.py --batch --out results/mmpbsa_v10
"""

import argparse, json, os, sys, tempfile, time
import numpy as np

sys.path.insert(0, "pwm_rosetta")
sys.path.insert(0, "src")

os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

BOLTZ_DIR   = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/out_all/boltz_results_inputs_all/predictions"
ROSETTA_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v10"
SPLIT       = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
MAX_L       = 20
BASES       = ['A', 'C', 'G', 'T']
BASE_IDX    = {b: i for i, b in enumerate(BASES)}


DNA_BASES   = {'DA', 'DC', 'DG', 'DT'}
PROTEIN_RES = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS "
                  "MET PHE PRO SER THR TRP TYR VAL".split())

_FF = None   # module-level cache

def _get_ff():
    from openmm.app import ForceField
    global _FF
    if _FF is None:
        _FF = ForceField('amber14/protein.ff14SB.xml',
                         'amber14/DNA.OL15.xml',
                         'implicit/gbn2.xml')
    return _FF


def _patch_gbforce(system, solute_dielectric=2.0, solvent_dielectric=78.5):
    """Patch the CustomGBForce dielectric constants after system creation.

    The gbn2.xml has soluteDielectric=1.0 and solventDielectric=78.5 hardcoded
    in the energy expression strings. We replace them via string substitution.
    """
    import openmm as _mm
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if not isinstance(f, _mm.CustomGBForce):
            continue
        for j in range(f.getNumEnergyTerms()):
            expr, ctype = f.getEnergyTermParameters(j)
            expr = expr.replace('soluteDielectric=1.0',
                                f'soluteDielectric={solute_dielectric}')
            expr = expr.replace(f'solventDielectric=78.5',
                                f'solventDielectric={solvent_dielectric}')
            f.setEnergyTermParameters(j, expr, ctype)
    return system


def prepare_complex_from_cif(cif_path):
    """Load Boltz CIF, fix terminals, add H → return Modeller ready for OpenMM.

    Fixes two issues in Boltz output:
      1. 5'-terminal DNA residue has a spurious 5'-phosphate (P/OP1/OP2);
         AMBER DT5/DA5/… templates expect none → delete those atoms.
      2. C-terminal protein residue is missing OXT;
         AMBER CLYS/… templates require it → add at reflected position.
    """
    from openmm.app import PDBxFile, Modeller, element
    from openmm import Vec3
    from openmm.unit import nanometers, Quantity

    pdb    = PDBxFile(cif_path)
    top    = pdb.topology
    pos_nm = [Vec3(*p.value_in_unit(nanometers)) for p in pdb.positions]
    ff     = _get_ff()

    # Find 5'-terminal DNA phosphate atoms
    del_idx = set()
    for chain in top.chains():
        dna_res = [r for r in chain.residues() if r.name in DNA_BASES]
        if dna_res:
            for a in dna_res[0].atoms():
                if a.name in ('P', 'OP1', 'OP2'):
                    del_idx.add(a.index)

    # Find C-terminal protein residue
    c_term = None
    for chain in top.chains():
        prot = [r for r in chain.residues() if r.name in PROTEIN_RES]
        if prot:
            c_term = prot[-1]

    # Rebuild topology
    new_top    = type(top)()
    new_pos_nm = []
    idx_map    = {}
    oxt_atom   = None
    c_backbone = None

    for chain in top.chains():
        nc = new_top.addChain(chain.id)
        for res in chain.residues():
            nr = new_top.addResidue(res.name, nc)
            for atom in res.atoms():
                if atom.index in del_idx:
                    continue
                na = new_top.addAtom(atom.name, atom.element, nr)
                idx_map[atom.index] = na.index
                new_pos_nm.append(pos_nm[atom.index])
                if res is c_term and atom.name == 'C':
                    c_backbone = na

            if res is c_term:
                by_name = {a.name: pos_nm[a.index] for a in res.atoms()}
                if 'OXT' not in by_name and 'C' in by_name and 'O' in by_name:
                    c = by_name['C']; o = by_name['O']
                    oxt_atom = new_top.addAtom('OXT', element.oxygen, nr)
                    new_pos_nm.append(Vec3(2*c.x-o.x, 2*c.y-o.y, 2*c.z-o.z))

    all_new = list(new_top.atoms())
    for b in top.bonds():
        if b[0].index in del_idx or b[1].index in del_idx:
            continue
        new_top.addBond(all_new[idx_map[b[0].index]], all_new[idx_map[b[1].index]])
    if oxt_atom is not None and c_backbone is not None:
        new_top.addBond(c_backbone, oxt_atom)

    modeller = Modeller(new_top, Quantity([Vec3(*v) for v in new_pos_nm], nanometers))
    modeller.addHydrogens(ff, pH=7.0)
    return modeller


def prepare_mutant_from_pdb(pdb_path, ff):
    """Load PyRosetta PDB, strip H + fix terminals, add H via OpenMM.

    PyRosetta adds H when loading structures. Stripping them first
    lets OpenMM's addHydrogens produce the correct template-matched topology.
    Also applies the same 5'-phosphate and OXT fixes as prepare_complex_from_cif.
    """
    from openmm.app import PDBFile, Modeller, element
    from openmm import Vec3
    from openmm.unit import nanometers, Quantity

    pdb    = PDBFile(pdb_path)
    top    = pdb.topology
    pos_nm = [Vec3(*p.value_in_unit(nanometers)) for p in pdb.positions]

    # Atoms to delete: H atoms + 5'-terminal DNA phosphate
    del_idx = set()
    for chain in top.chains():
        dna_res = [r for r in chain.residues() if r.name in DNA_BASES]
        if dna_res:
            for a in dna_res[0].atoms():
                if a.name in ('P', 'OP1', 'OP2'):
                    del_idx.add(a.index)
    for atom in top.atoms():
        if atom.element is not None and atom.element.symbol == 'H':
            del_idx.add(atom.index)

    c_term = None
    for chain in top.chains():
        prot = [r for r in chain.residues() if r.name in PROTEIN_RES]
        if prot:
            c_term = prot[-1]

    new_top    = type(top)()
    new_pos_nm = []
    idx_map    = {}
    oxt_atom   = None
    c_backbone = None

    for chain in top.chains():
        nc = new_top.addChain(chain.id)
        for res in chain.residues():
            nr = new_top.addResidue(res.name, nc)
            for atom in res.atoms():
                if atom.index in del_idx:
                    continue
                na = new_top.addAtom(atom.name, atom.element, nr)
                idx_map[atom.index] = na.index
                new_pos_nm.append(pos_nm[atom.index])
                if res is c_term and atom.name == 'C':
                    c_backbone = na
            if res is c_term:
                by_name = {a.name: pos_nm[a.index] for a in res.atoms()
                           if a.index not in del_idx}
                if 'OXT' not in by_name and 'C' in by_name and 'O' in by_name:
                    c = by_name['C']; o = by_name['O']
                    oxt_atom = new_top.addAtom('OXT', element.oxygen, nr)
                    new_pos_nm.append(Vec3(2*c.x-o.x, 2*c.y-o.y, 2*c.z-o.z))

    all_new = list(new_top.atoms())
    for b in top.bonds():
        if b[0].index in del_idx or b[1].index in del_idx:
            continue
        new_top.addBond(all_new[idx_map[b[0].index]], all_new[idx_map[b[1].index]])
    if oxt_atom is not None and c_backbone is not None:
        new_top.addBond(c_backbone, oxt_atom)

    modeller = Modeller(new_top, Quantity([Vec3(*v) for v in new_pos_nm], nanometers))
    modeller.addHydrogens(ff, pH=7.0)
    return modeller


def build_system_from_modeller(modeller, dna_only=False):
    """Build OpenMM simulation. If dna_only=True, keep only DNA chains (for reference state)."""
    from openmm.app import NoCutoff, HBonds, Simulation, Modeller
    from openmm import LangevinIntegrator
    from openmm.unit import kelvin, picosecond
    ff = _get_ff()

    if dna_only:
        # Keep only non-protein chains (DNA)
        top = modeller.topology
        keep = set()
        for chain in top.chains():
            if not any(r.name in PROTEIN_RES for r in chain.residues()):
                keep.add(chain.id)
        atoms_keep = [a for a in top.atoms() if a.residue.chain.id in keep]
        idx_set    = {a.index for a in atoms_keep}
        new_top    = type(top)()
        new_pos    = []
        idx_map    = {}
        pos        = list(modeller.positions)
        for chain in top.chains():
            if chain.id not in keep:
                continue
            nc = new_top.addChain(chain.id)
            for res in chain.residues():
                nr = new_top.addResidue(res.name, nc)
                for atom in res.atoms():
                    na = new_top.addAtom(atom.name, atom.element, nr)
                    idx_map[atom.index] = na.index
                    new_pos.append(pos[atom.index])
        all_new = list(new_top.atoms())
        for b in top.bonds():
            if b[0].index in idx_set and b[1].index in idx_set:
                new_top.addBond(all_new[idx_map[b[0].index]], all_new[idx_map[b[1].index]])
        from openmm.unit import Quantity, nanometers
        from openmm import Vec3
        new_pos_q = Quantity([Vec3(*p.value_in_unit(nanometers)) for p in new_pos], nanometers)
        use_mod = Modeller(new_top, new_pos_q)
    else:
        use_mod = modeller

    system = ff.createSystem(use_mod.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    # Patch GBn2 dielectric: soluteDielectric=2.0 better for protein-DNA interface
    _patch_gbforce(system, solute_dielectric=2.0, solvent_dielectric=78.5)
    intg   = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
    # Use fastest available platform: CUDA > OpenCL > CPU > Reference
    import openmm as _mm
    platform = None
    for name in ('CUDA', 'OpenCL', 'CPU'):
        try:
            platform = _mm.Platform.getPlatformByName(name)
            break
        except Exception:
            pass
    sim = Simulation(use_mod.topology, system, intg, platform=platform)
    sim.context.setPositions(use_mod.positions)
    return sim


def minimize_and_energy(sim, max_iter=300):
    """Minimize and return potential energy in kcal/mol."""
    from openmm.unit import kilocalories_per_mole
    sim.minimizeEnergy(maxIterations=max_iter)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)


def scan_dna_positions(cif_path, core_offset, core_len, tau=1.5, min_iter=300):
    """Scan all 4 bases at each core position using MM-GBSA (AMBER14+GBn2).

    Uses PyRosetta for DNA mutation and OpenMM for energy evaluation.

    Returns
    -------
    ppm    : (core_len, 4)  Boltzmann position probability matrix
    ddg    : (core_len, 4)  raw ΔΔG matrix (kcal/mol)
    wt_seq : str            wild-type DNA sequence
    """
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags
    from pwm_hybrid.rosetta.mutations import mutate_dna_base
    from pwm_hybrid.rosetta.structure import get_dna_strand_positions

    pyrosetta.init(get_pyrosetta_init_flags() + " -out:level 0", silent=True)
    pose = pyrosetta.pose_from_file(cif_path)

    strand1, _ = get_dna_strand_positions(pose)
    wt_seq = ''.join(b for _, b in strand1)

    ddg = np.zeros((core_len, 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        # WT: complex energy + free-DNA energy (reference state)
        try:
            wt_mod      = prepare_complex_from_cif(cif_path)
            wt_sim_cplx = build_system_from_modeller(wt_mod, dna_only=False)
            wt_sim_dna  = build_system_from_modeller(wt_mod, dna_only=True)
            e_wt_cplx   = minimize_and_energy(wt_sim_cplx, min_iter)
            e_wt_dna    = minimize_and_energy(wt_sim_dna,  min_iter)
            print(f"  WT complex: {e_wt_cplx:.1f}  DNA-only: {e_wt_dna:.1f} kcal/mol", flush=True)
        except Exception as ex:
            print(f"  [FAIL] WT energy: {ex}")
            return None, None, wt_seq

        for pos_i in range(core_len):
            strand_pos = core_offset + pos_i + 1
            wt_base = wt_seq[strand_pos - 1] if strand_pos - 1 < len(wt_seq) else 'N'

            for b_i, base in enumerate(BASES):
                if base == wt_base:
                    ddg[pos_i, b_i] = 0.0
                    continue
                try:
                    mut_pose = mutate_dna_base(pose, strand_pos, base)
                    mut_pdb  = os.path.join(tmpdir, f"mut_p{pos_i}_{base}.pdb")
                    mut_pose.dump_pdb(mut_pdb)
                    ff   = _get_ff()
                    mmod = prepare_mutant_from_pdb(mut_pdb, ff)
                    # ΔΔG_bind = ΔE_complex − ΔE_DNA (cancels intrinsic base energy)
                    e_mut_cplx = minimize_and_energy(build_system_from_modeller(mmod, dna_only=False), min_iter)
                    e_mut_dna  = minimize_and_energy(build_system_from_modeller(mmod, dna_only=True),  min_iter)
                    ddg[pos_i, b_i] = (e_mut_cplx - e_wt_cplx) - (e_mut_dna - e_wt_dna)
                except Exception as ex:
                    print(f"  [WARN] pos={strand_pos} base={base}: {ex}")
                    ddg[pos_i, b_i] = 0.0

        # Boltzmann PPM
        logits = -ddg / tau
        logits -= logits.max(axis=1, keepdims=True)
        ppm     = np.exp(logits)
        ppm    /= ppm.sum(axis=1, keepdims=True)

    return ppm, ddg, wt_seq


def process_one(af3_name, out_dir, tau=1.5):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{af3_name}.npz")
    if os.path.exists(out_path):
        print(f"  [SKIP] {af3_name} — already done")
        return True

    # Find CIF
    cif = os.path.join(BOLTZ_DIR, af3_name, f"{af3_name}_model_0.cif")
    if not os.path.exists(cif):
        print(f"  [SKIP] {af3_name}: no CIF at {cif}")
        return False

    # Get core offset/length from existing Rosetta run
    cal_path = os.path.join(ROSETTA_DIR, af3_name, "calibrated_pwm.npz")
    if not os.path.exists(cal_path):
        print(f"  [SKIP] {af3_name}: no calibrated_pwm.npz")
        return False
    cal_d       = np.load(cal_path)
    core_offset = int(cal_d["core_offset"])
    core_len    = int(cal_d["core_len"])

    print(f"\n=== {af3_name}  core_offset={core_offset} core_len={core_len} ===", flush=True)
    t0 = time.time()
    ppm, ddg, wt_seq = scan_dna_positions(cif, core_offset, core_len, tau=tau)
    elapsed = time.time() - t0

    if ppm is None:
        print(f"  [FAIL] {af3_name}: scan returned None")
        return False

    # Pad to MAX_L
    L = min(ppm.shape[0], MAX_L)
    core_ppm_padded = np.full((4, MAX_L), 0.25, dtype=np.float32)
    core_ppm_padded[:, :L] = ppm[:L].T

    np.savez_compressed(out_path,
                        core_ppm=core_ppm_padded,
                        ddg=ddg,
                        core_offset=core_offset,
                        core_len=core_len,
                        wt_seq=wt_seq,
                        tau=tau)
    print(f"  Done in {elapsed:.0f}s → {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif",       default=None,  help="Single CIF path (single-sample mode)")
    ap.add_argument("--af3-name",  default=None,  help="af3_name for single-sample mode")
    ap.add_argument("--batch",     action="store_true", help="Run all 130 test TFs")
    ap.add_argument("--pilot",     action="store_true",
                    help="Only run 31 cases where Rosetta currently improves")
    ap.add_argument("--out",       default="results/mmpbsa_v10")
    ap.add_argument("--tau",       type=float, default=1.5)
    ap.add_argument("--min-iter",  type=int,   default=300)
    args = ap.parse_args()

    if args.batch or args.pilot:
        import pandas as pd
        df_cal = pd.read_csv("results/v10_seed_calibration_comparison.csv")
        if args.pilot:
            # Only cases where Rosetta currently improves (31 samples)
            names = df_cal[df_cal["improvement"] > 0]["af3_name"].tolist()
            print(f"Pilot mode: {len(names)} samples where Rosetta improves")
        else:
            names = df_cal["af3_name"].tolist()
            print(f"Batch mode: {len(names)} samples")

        ok = fail = skip = 0
        for name in names:
            result = process_one(name, args.out, tau=args.tau)
            if result: ok += 1
            else: fail += 1
        print(f"\nDone: {ok} ok, {fail} failed")
    else:
        if args.af3_name is None:
            ap.error("Provide --af3-name in single-sample mode")
        process_one(args.af3_name, args.out, tau=args.tau)


if __name__ == "__main__":
    main()
