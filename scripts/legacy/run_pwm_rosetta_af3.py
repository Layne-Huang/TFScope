#!/usr/bin/env python
"""Canonical pwm_rosetta README workflow for TFScope's predicted zinc-finger PWMs.

Pipeline (README "How it works"):
  AF3 wild-type fold (protein + TFScope RAG-consensus DNA, run ONCE)
      -> PyRosetta single-base mutation scan (mutate every base x3, local minimise)
      -> interface ΔΔG per mutation  -> Boltzmann PPM -> PWM / logo

This is the fully DEPLOYABLE structure-based predictor: it never sees the true binding
site — AF3 folds TFScope's predicted consensus, and Rosetta single-mutation ΔΔG turns that
fold into a PWM.  We then oracle-align the resulting PWM to the true designed motif.

Usage:
  python scripts/run_pwm_rosetta_af3.py --out results/zf_struct/pwm_rosetta_af3
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, "pwm_rosetta")

AF3_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf_rag"
BASES   = ["A", "C", "G", "T"]

# true designed binding sites (chain-B of the design structures = ground truth)
TRUE_MOTIF = {
    "zf21":  "ATTAGAAGTGA",
    "zf93":  "CTTTGTTGGCA",
    "zf92":  "AGATGTAGCCT",
    "zf129": "CGGGGACGTCA",
}
IPTM = {"zf21": 0.82, "zf93": 0.70, "zf92": 0.44, "zf129": 0.44}


def onehot(seq):
    idx = {b: i for i, b in enumerate(BASES)}
    M = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in idx:
            M[i, idx[b]] = 1.0
    return M


def revcomp_ppm(ppm):
    # reverse positions and swap A<->T, C<->G  (ACGT index 0,1,2,3 -> T,G,C,A = 3,2,1,0)
    return ppm[::-1, ::-1].copy()


def oracle_align_r(pred, gt):
    """max mean per-column Pearson r over all shifts x {fwd, revcomp}."""
    def best(p):
        Lp, Lg = p.shape[0], gt.shape[0]
        best_r, best_off = -2.0, 0
        for off in range(-(Lp - 1), Lg):
            rs = []
            for j in range(Lg):
                i = j - off
                if 0 <= i < Lp:
                    a, b = p[i], gt[j]
                    if a.std() > 1e-8 and b.std() > 1e-8:
                        rs.append(np.corrcoef(a, b)[0, 1])
            if len(rs) >= 4:
                m = float(np.mean(rs))
                if m > best_r:
                    best_r, best_off = m, off
        return best_r, best_off
    r_f, off_f = best(pred)
    r_r, off_r = best(revcomp_ppm(pred))
    return (r_f, "fwd", off_f) if r_f >= r_r else (r_r, "rev", off_r)


def run_one(name, out_dir):
    from pwm_hybrid.rosetta.structure import convert_cif_to_pdb
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv, plot

    rdir = os.path.join(out_dir, name)
    os.makedirs(rdir, exist_ok=True)
    cif = os.path.join(AF3_DIR, f"{name}_rag", f"{name}_rag_model.cif")
    pdb = os.path.join(rdir, f"{name}_af3.pdb")
    convert_cif_to_pdb(cif, pdb)

    print(f"\n=== Rosetta-on-AF3 {name}  (ipTM {IPTM[name]}) ===", flush=True)
    t0 = time.time()
    generate_pwm_hybrid(protein_seq=None, dna_seq=None, output_dir=rdir,
                        wt_pdb=pdb, minimize_local=True, use_relax=False)
    csv = os.path.join(rdir, "pwm_results_hybrid.csv")
    seq, energies, PPM, PWM = pwm_from_hybrid_csv(csv, tau=1.5)
    PPM = np.asarray(PPM)
    consensus = ''.join(BASES[i] for i in PPM.argmax(1))
    gt = onehot(TRUE_MOTIF[name])
    r, orient, off = oracle_align_r(PPM, gt)
    np.save(os.path.join(rdir, "pwm.npy"), PPM.T)
    try:
        plot(PPM, seq=seq, outpath=rdir, filename_prefix=f"{name}_af3_rosetta")
    except Exception as ex:
        print(f"  [plot warn] {ex}")
    res = dict(name=name, iptm=IPTM[name], af3_dna=seq, consensus=consensus,
               true_motif=TRUE_MOTIF[name], oracle_r_vs_true=round(r, 3),
               align=orient, offset=off, seconds=round(time.time() - t0, 0))
    json.dump(res, open(os.path.join(rdir, f"{name}.json"), "w"), indent=2)
    print(f"  AF3-DNA={seq}  consensus={consensus}  true={TRUE_MOTIF[name]}  "
          f"oracle-r={r:.3f} ({orient}+{off})  [{res['seconds']:.0f}s]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/zf_struct/pwm_rosetta_af3")
    ap.add_argument("--only", nargs="*", default=list(TRUE_MOTIF))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")

    summary = [run_one(n, a.out) for n in a.only]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("\n=== Rosetta-on-AF3 PWM summary (predicted ZF) ===")
    for s in summary:
        print(f"  {s['name']:6s} ipTM {s['iptm']}  consensus {s['consensus']:14s} "
              f"true {s['true_motif']:11s}  oracle-r {s['oracle_r_vs_true']}")
    print(f"saved -> {a.out}/summary.json")


if __name__ == "__main__":
    main()
