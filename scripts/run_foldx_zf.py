#!/usr/bin/env python
"""FoldX AnalyseComplex DNA base scan for the ZF design structures.

FoldX cannot mutate DNA bases natively, so we use the same PyRosetta machinery as
pwm_rosetta / MM-GBSA to generate each base mutant (mutate + local minimise), then score
the protein-DNA interaction with FoldX `AnalyseComplex` (chains A vs BC).
  ΔΔG_bind(base) = Interaction(mutant) − Interaction(WT)  →  Boltzmann → PWM.

This gives a FoldX-energy-function counterpart to the Rosetta and MM-GBSA PWMs on the
exact same minimised geometries.

Usage:
  python scripts/run_foldx_zf.py --all --out results/zf_struct/foldx_pwm
  python scripts/run_foldx_zf.py --pdb design_pdbs/zf_samples/zf21_model.pdb --name zf21 \
      --out results/zf_struct/foldx_pwm
"""
import argparse, json, os, sys, shutil, subprocess, tempfile, time
import numpy as np

sys.path.insert(0, "pwm_rosetta")

FOLDX = "/n/home13/leihuang/project/TFScope/bin/foldx_20270131"
MOLEC = "/n/home13/leihuang/project/TFScope/bin/molecules"
BASES = ["A", "C", "G", "T"]

ZF_PDBS = {
    "zf21":  "design_pdbs/zf_samples/zf21_model.pdb",
    "zf93":  "design_pdbs/zf_samples/zf93_model.pdb",
    "zf92":  "design_pdbs/zf_samples/zf92_model.pdb",
    "zf129": "design_pdbs/zf_samples/zf129_model.pdb",
}


def foldx_interaction(pose, workdir, tag):
    """Dump pose, run FoldX AnalyseComplex (A vs BC), return Interaction Energy (kcal/mol)."""
    pdb = os.path.join(workdir, f"{tag}.pdb")
    pose.dump_pdb(pdb)
    cmd = [FOLDX, "--command=AnalyseComplex", f"--pdb={tag}.pdb",
           "--analyseComplexChains=A,BC"]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    summ = os.path.join(workdir, f"Summary_{tag}_AC.fxout")
    if not os.path.exists(summ):
        raise RuntimeError(f"FoldX produced no summary for {tag}: {r.stdout[-300:]}")
    with open(summ) as fh:
        last = [ln for ln in fh if ln.strip()][-1].split("\t")
    # columns: Pdb Group1 Group2 Intra1 Intra2 InteractionEnergy Stab1 Stab2
    return float(last[5])


def scan_pdb(pdb_path, tau=1.5):
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    from pwm_hybrid.rosetta.mutations import mutate_dna_base, minimize_around_mutation
    from pwm_hybrid.rosetta.structure import get_dna_strand_positions

    pyrosetta.init(get_pyrosetta_init_flags() + " -out:level 0", silent=True)
    init_pyrosetta(psipred_exe="")          # sets _init.sfxn used by minimize_around_mutation
    pose = pyrosetta.pose_from_file(pdb_path)
    strand1, _ = get_dna_strand_positions(pose)
    wt_seq   = ''.join(b for _, b in strand1)
    core_len = len(strand1)
    ddg = np.zeros((core_len, 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as work:
        os.symlink(MOLEC, os.path.join(work, "molecules"))
        e_wt = foldx_interaction(pose, work, "wt")
        print(f"  WT interaction={e_wt:.2f}  seq={wt_seq}", flush=True)
        for pos_i in range(core_len):
            wt_base = wt_seq[pos_i]
            for b_i, base in enumerate(BASES):
                if base == wt_base:
                    ddg[pos_i, b_i] = 0.0
                    continue
                try:
                    mp = mutate_dna_base(pose, pos_i + 1, base)
                    mp = minimize_around_mutation(mp, pos_i + 1)
                    e_mut = foldx_interaction(mp, work, f"m{pos_i}_{base}")
                    ddg[pos_i, b_i] = e_mut - e_wt
                except Exception as ex:
                    print(f"  [WARN] pos={pos_i+1} base={base}: {ex}", flush=True)
                    ddg[pos_i, b_i] = 0.0
            print(f"  pos {pos_i+1:2d} wt={wt_base} ddG={np.round(ddg[pos_i],2)}", flush=True)

    logits  = -ddg / tau
    logits -= logits.max(axis=1, keepdims=True)
    ppm     = np.exp(logits); ppm /= ppm.sum(axis=1, keepdims=True)
    return ppm, ddg, wt_seq


def onehot(seq):
    idx = {b: i for i, b in enumerate(BASES)}
    M = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in idx:
            M[i, idx[b]] = 1.0
    return M


def col_r(pred, gt):
    rs = []
    for p, g in zip(pred, gt):
        if p.std() < 1e-8 or g.std() < 1e-8:
            continue
        rs.append(np.corrcoef(p, g)[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def run_one(name, pdb, out_dir, tau):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== FoldX {name}  ({pdb}) ===", flush=True)
    t0 = time.time()
    ppm, ddg, wt_seq = scan_pdb(pdb, tau=tau)
    consensus = ''.join(BASES[i] for i in ppm.argmax(1))
    r = col_r(ppm, onehot(wt_seq))
    n_match = sum(c == w for c, w in zip(consensus, wt_seq))
    res = dict(name=name, pdb=pdb, wt_seq=wt_seq, consensus=consensus,
               n_match=n_match, n_pos=len(wt_seq), oracle_r_vs_design=round(r, 3),
               tau=tau, seconds=round(time.time() - t0, 0))
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
    ap.add_argument("--out", default="results/zf_struct/foldx_pwm")
    ap.add_argument("--tau", type=float, default=1.5)
    a = ap.parse_args()
    targets = list(ZF_PDBS.items()) if a.all else [(a.name, a.pdb)]
    summary = [run_one(n, p, a.out, a.tau) for n, p in targets]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("\n=== FoldX ZF summary ===")
    for s in summary:
        print(f"  {s['name']:6s} {s['consensus']:16s} recover {s['n_match']}/{s['n_pos']}  col-r {s['oracle_r_vs_design']}")
    print(f"saved -> {a.out}/summary.json")


if __name__ == "__main__":
    main()
