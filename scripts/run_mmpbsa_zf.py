#!/usr/bin/env python
"""MM-GBSA (OpenMM AMBER14/DNA.OL15 + GBn2) DNA base scan for the ZF design structures.

Mirrors the pwm_rosetta design scan, but replaces the Rosetta RM8B ΔΔG with proper
implicit-solvent energetics:  ΔΔG_bind(base) = ΔE_complex − ΔE_DNA, Boltzmann → PWM.

Each ZF design PDB = protein chain A + 11-nt DNA duplex (chains B,C) whose chain-B
sequence IS the designed/bound site (the ground truth).  We score all 4 bases at every
DNA position and ask whether MM-GBSA recovers the bound base.

Usage:
  python scripts/run_mmpbsa_zf.py --pdb design_pdbs/zf_samples/zf21_model.pdb --name zf21 \
      --out results/zf_struct/mmpbsa
  # all four:
  python scripts/run_mmpbsa_zf.py --all --out results/zf_struct/mmpbsa
"""
import argparse, json, os, sys, tempfile, time
import numpy as np

sys.path.insert(0, "pwm_rosetta")
sys.path.insert(0, "scripts")

# reuse the validated MM-GBSA building blocks
from run_mmpbsa_scan import (_get_ff, prepare_mutant_from_pdb,
                             build_system_from_modeller, minimize_and_energy, BASES)

ZF_PDBS = {
    "zf21":  "design_pdbs/zf_samples/zf21_model.pdb",
    "zf93":  "design_pdbs/zf_samples/zf93_model.pdb",
    "zf92":  "design_pdbs/zf_samples/zf92_model.pdb",
    "zf129": "design_pdbs/zf_samples/zf129_model.pdb",
}


def scan_pdb(pdb_path, tau=1.5, min_iter=300):
    """Scan all 4 bases at every DNA position of a protein-DNA PDB via MM-GBSA."""
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags
    from pwm_hybrid.rosetta.mutations import mutate_dna_base
    from pwm_hybrid.rosetta.structure import get_dna_strand_positions

    pyrosetta.init(get_pyrosetta_init_flags() + " -out:level 0", silent=True)
    pose = pyrosetta.pose_from_file(pdb_path)
    strand1, _ = get_dna_strand_positions(pose)
    wt_seq   = ''.join(b for _, b in strand1)
    core_len = len(strand1)
    ff  = _get_ff()
    ddg = np.zeros((core_len, 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        wt_pdb = os.path.join(tmp, "wt.pdb"); pose.dump_pdb(wt_pdb)
        wt_mod    = prepare_mutant_from_pdb(wt_pdb, ff)
        e_wt_cplx = minimize_and_energy(build_system_from_modeller(wt_mod, dna_only=False), min_iter)
        e_wt_dna  = minimize_and_energy(build_system_from_modeller(wt_mod, dna_only=True),  min_iter)
        print(f"  WT complex {e_wt_cplx:.1f}  DNA-only {e_wt_dna:.1f}  seq={wt_seq}", flush=True)

        for pos_i in range(core_len):
            wt_base = wt_seq[pos_i]
            for b_i, base in enumerate(BASES):
                if base == wt_base:
                    ddg[pos_i, b_i] = 0.0
                    continue
                try:
                    mp   = mutate_dna_base(pose, pos_i + 1, base)
                    mpdb = os.path.join(tmp, f"m{pos_i}_{base}.pdb"); mp.dump_pdb(mpdb)
                    mmod = prepare_mutant_from_pdb(mpdb, ff)
                    e_mc = minimize_and_energy(build_system_from_modeller(mmod, dna_only=False), min_iter)
                    e_md = minimize_and_energy(build_system_from_modeller(mmod, dna_only=True),  min_iter)
                    ddg[pos_i, b_i] = (e_mc - e_wt_cplx) - (e_md - e_wt_dna)
                except Exception as ex:
                    print(f"  [WARN] pos={pos_i+1} base={base}: {ex}", flush=True)
                    ddg[pos_i, b_i] = 0.0
            row = ddg[pos_i]
            print(f"  pos {pos_i+1:2d} wt={wt_base} ddG={np.round(row,2)}", flush=True)

    logits  = -ddg / tau
    logits -= logits.max(axis=1, keepdims=True)
    ppm     = np.exp(logits); ppm /= ppm.sum(axis=1, keepdims=True)
    return ppm, ddg, wt_seq


def onehot(seq):
    idx = {b: i for i, b in enumerate(BASES)}
    M = np.full((len(seq), 4), 0.0, dtype=np.float32)
    for i, b in enumerate(seq):
        if b in idx:
            M[i, idx[b]] = 1.0
    return M


def col_r(pred, gt):
    """mean per-column Pearson r (pred, gt same length, same register)."""
    rs = []
    for p, g in zip(pred, gt):
        if p.std() < 1e-8 or g.std() < 1e-8:
            continue
        rs.append(np.corrcoef(p, g)[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def run_one(name, pdb, out_dir, tau, min_iter):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== MM-GBSA {name}  ({pdb}) ===", flush=True)
    t0 = time.time()
    ppm, ddg, wt_seq = scan_pdb(pdb, tau=tau, min_iter=min_iter)
    consensus = ''.join(BASES[i] for i in ppm.argmax(1))
    gt = onehot(wt_seq)
    r  = col_r(ppm, gt)
    n_match = sum(c == w for c, w in zip(consensus, wt_seq))
    res = dict(name=name, pdb=pdb, wt_seq=wt_seq, consensus=consensus,
               n_match=n_match, n_pos=len(wt_seq), oracle_r_vs_design=round(r, 3),
               tau=tau, min_iter=min_iter, seconds=round(time.time() - t0, 0))
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                        ppm=ppm.T, ddg=ddg, wt_seq=wt_seq)
    json.dump(res, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=2)
    print(f"  consensus={consensus}  recovers {n_match}/{len(wt_seq)}  "
          f"col-r={r:.3f}  ({res['seconds']:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb"); ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="results/zf_struct/mmpbsa")
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--min-iter", type=int, default=300)
    a = ap.parse_args()

    targets = list(ZF_PDBS.items()) if a.all else [(a.name, a.pdb)]
    summary = [run_one(n, p, a.out, a.tau, a.min_iter) for n, p in targets]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("\n=== MM-GBSA ZF summary ===")
    for s in summary:
        print(f"  {s['name']:6s} {s['consensus']:16s} recover {s['n_match']}/{s['n_pos']}  col-r {s['oracle_r_vs_design']}")
    print(f"saved -> {a.out}/summary.json")


if __name__ == "__main__":
    main()
