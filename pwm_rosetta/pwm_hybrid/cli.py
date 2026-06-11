"""Command-line entry point: ``pwm-hybrid``."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid PWM generation: AF3 wild-type + Rosetta mutations"
    )
    parser.add_argument('-protein_seq', type=str, help='Protein sequence')
    parser.add_argument('-dna_seq', type=str, help='DNA sequence')
    parser.add_argument('-pdb', type=str, help='Input PDB file (skip AF3)')
    parser.add_argument('-output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('-template', type=str, help='AF3 template JSON')
    parser.add_argument('-zf_count', type=int, default=0,
                        help='Number of zinc fingers')
    parser.add_argument('-no_minimize', action='store_true',
                        help='Skip local minimization')
    parser.add_argument('-relax', action='store_true',
                        help='Use full relaxation (slower)')
    parser.add_argument(
        '-psipred_exe', type=str, default=None,
        help=(
            'Path to runpsipred_single executable. '
            'Defaults to PSIPRED_EXE env var, then '
            '/software/psipred4/runpsipred_single. '
            'Pass empty string to disable SSPrediction filters.'
        )
    )

    args = parser.parse_args()

    if not args.pdb and not (args.protein_seq and args.dna_seq):
        parser.error("Must provide either -pdb OR both -protein_seq and -dna_seq")

    # Deferred PyRosetta init — happens here, not at import time
    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta

    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe=args.psipred_exe)

    from pwm_hybrid.pipeline import generate_pwm_hybrid

    if args.pdb:
        generate_pwm_hybrid(
            protein_seq=None,
            dna_seq=None,
            output_dir=args.output_dir,
            wt_pdb=args.pdb,
            minimize_local=not args.no_minimize,
            use_relax=args.relax,
        )
    else:
        generate_pwm_hybrid(
            protein_seq=args.protein_seq,
            dna_seq=args.dna_seq,
            output_dir=args.output_dir,
            template_json=args.template,
            zf_count=args.zf_count,
            minimize_local=not args.no_minimize,
            use_relax=args.relax,
        )


if __name__ == "__main__":
    main()
