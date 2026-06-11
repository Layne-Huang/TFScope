import os
import subprocess
import argparse
import glob
import shutil

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.MMCIFParser import MMCIFParser
from af3cli import InputBuilder, ProteinSequence, DNASequence, InputFile, CCDLigand  # type: ignore

def _is_dna_res(res):
    name = res.get_resname().upper()
    return name.startswith("D") and len(name) == 2

def _is_zn_res(res):
    return res.get_resname().upper() == "ZN"


def clean_protein_seq(seq):
    return "".join(aa for aa in seq if aa in VALID_AA)


def clean_dna_seq(seq):
    return "".join(b for b in seq if b in VALID_DNA)

# Cluster paths (override via environment variables if running elsewhere).
_AF3_ROOT     = "/n/holylabs/lpinello_lab/Lab/leihuang"
AF3_INPUT_DIR = os.environ.get("AF3_INPUT_DIR", f"{_AF3_ROOT}/TFScope/alphafold3/af_input")
AF3_SCRIPT    = os.environ.get("AF3_SCRIPT",    f"{_AF3_ROOT}/af3_mmseqs_scripts/af3_mmseqs2/alphafold3.py")
MODEL_PARAMS  = os.environ.get("AF3_MODEL_PARAMS", f"{_AF3_ROOT}/AF3_parameter")
# DATABASE is unused with --mmseqs2 (remote MSA) but the wrapper still mounts it; a placeholder dir is fine.
DATABASE      = os.environ.get("AF3_DATABASE",  f"{_AF3_ROOT}/TFScope/alphafold3/public_databases")
AF3_WORKDIR   = os.environ.get("AF3_WORKDIR",   f"{_AF3_ROOT}/af3_mmseqs_scripts/af3_mmseqs2")


def extract_sequences_from_pdb(pdb_path):
    """Return (protein_chains, dna_chains, zn_count) from a PDB or CIF file."""
    parser = MMCIFParser(QUIET=True) if pdb_path.endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    protein_chains, dna_chains = {}, {}
    zn_count = 0

    for model in structure:
        for chain in model:
            resnames = []
            is_dna_chain = True
            is_protein_chain = True

            for res in chain:
                resname = res.get_resname().strip()
                if _is_dna_res(res):
                    resnames.append(resname[-1])
                    is_protein_chain = False
                elif is_aa(res, standard=True):
                    resnames.append(seq1(resname))
                    is_dna_chain = False
                elif _is_zn_res(res):
                    zn_count += 1

            if not resnames:
                continue

            seq_str = "".join(resnames)
            if is_dna_chain:
                cleaned = clean_dna_seq(seq_str)
                if cleaned:
                    dna_chains[chain.id] = cleaned
            elif is_protein_chain:
                cleaned = clean_protein_seq(seq_str)
                if cleaned:
                    protein_chains[chain.id] = cleaned

    return protein_chains, dna_chains, zn_count


def make_af3_json(proteins, dna, job_name, outfile="complex.json", zf_count=0):
    builder = InputBuilder()
    builder.set_name(job_name)
    builder.set_version(2)

    for seq in proteins:
        builder.add_sequence(ProteinSequence(seq))

    if dna is not None:
        dna_obj = DNASequence(dna)
        builder.add_sequence(dna_obj)
        builder.add_sequence(dna_obj.reverse_complement())

    for _ in range(zf_count):
        builder.add_ligand(CCDLigand("ZN"))

    builder.build().write(outfile)


def rebuild_with_new_dna(template_path, new_dna_seq, output_path, name):
    data = InputFile.read(template_path)
    builder = InputBuilder()
    builder.set_name(name)
    builder.set_version(data.version)

    for seq in data.sequences:
        if seq._seq_type == "protein":
            builder.add_sequence(seq)

    if new_dna_seq is not None:
        dna = DNASequence(new_dna_seq)
        builder.add_sequence(dna)
        builder.add_sequence(dna.reverse_complement())

    builder.build().write(output_path)
    print(f"Rebuilt from template: {output_path}")


def AF3_folding(pdb_file=None, protein_seq=None, dna_seq=None,
                job_name=None, output_dir=None, template=None, zf_count=0):
    assert pdb_file is not None or protein_seq is not None or template is not None, \
        "Provide --pdb, --protein-seq, or --template."

    if pdb_file is not None:
        protein, dna, zn_from_pdb = extract_sequences_from_pdb(pdb_file)
        # Use zinc count from PDB if found; otherwise keep the caller-supplied zf_count
        if zn_from_pdb > 0:
            zf_count = zn_from_pdb
        if protein_seq is None:
            protein_seq = list(protein.values())
        if dna_seq is None and dna:
            dna_seq = list(dna.values())[0]
            print(f"DNA from PDB: {dna_seq}")
        print(f"Zinc ions: {zf_count} (from {'PDB' if zn_from_pdb > 0 else 'caller'})")

    if job_name is None:
        job_name = os.path.basename(pdb_file).replace(".pdb", "_AF3") if pdb_file else "af3_job"

    if output_dir is None:
        base = os.path.dirname(pdb_file) if pdb_file else "."
        output_dir = os.path.join(base, "af_output")

    os.makedirs(AF3_INPUT_DIR, exist_ok=True)
    out_json = os.path.join(AF3_INPUT_DIR, job_name + ".json")

    if template is not None:
        rebuild_with_new_dna(template, dna_seq, out_json, job_name)
    else:
        make_af3_json(protein_seq, dna_seq, job_name, outfile=out_json, zf_count=zf_count)

    print(f"AF3 JSON: {out_json}")

    cmd = (
        f"cd {AF3_WORKDIR} && "
        f"python {AF3_SCRIPT} "
        f"{out_json} {output_dir} "
        f"--model_params {MODEL_PARAMS} "
        f"--database {DATABASE} "
        f"--mmseqs2"
    )
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    print(stderr.decode())

    out_path = os.path.join(output_dir, job_name)
    for path in glob.glob(os.path.join(out_path, "seed-1*")):
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def build_parser():
    p = argparse.ArgumentParser(
        description="Run a single AF3 fold for a protein or protein-DNA complex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Protein only (no DNA)
  python AF3_single.py \\
      --protein-seq MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL \\
      --job-name myprotein \\
      --output-dir ./af3_out

  # Protein + DNA complex
  python AF3_single.py \\
      --protein-seq GPKPPYSYAALIAMAIHGAPDKR \\
      --dna-seq AGATTGTTTATTGAGA \\
      --job-name complex1 \\
      --output-dir ./af3_out

  # Multiple protein chains (comma-separated)
  python AF3_single.py \\
      --protein-seq "SEQCHAIN1,SEQCHAIN2" \\
      --dna-seq AGATTGTTTATTGAGA \\
      --job-name homodimer \\
      --output-dir ./af3_out

  # From PDB (extracts protein + DNA automatically)
  python AF3_single.py \\
      --pdb complex.pdb \\
      --job-name pdb_fold \\
      --output-dir ./af3_out

  # From PDB but override DNA
  python AF3_single.py \\
      --pdb complex.pdb \\
      --dna-seq AGATTGTTTATTGAGA \\
      --job-name custom_dna \\
      --output-dir ./af3_out

  # Use existing AF3 JSON as protein template (with or without new DNA)
  python AF3_single.py \\
      --template /path/to/complex.json \\
      --dna-seq AGATTGTTTATTGAGA \\
      --job-name new_dna \\
      --output-dir ./af3_out
""",
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--pdb", metavar="FILE",
                     help="Input PDB or CIF file (extracts protein + DNA sequences)")
    src.add_argument("--protein-seq", metavar="SEQ[,SEQ...]",
                     help="Protein sequence(s), comma-separated for multiple chains")
    src.add_argument("--template", metavar="JSON",
                     help="Existing AF3 JSON to use as protein template")

    p.add_argument("--dna-seq", metavar="SEQ",
                   help="DNA sequence (optional — omit for protein-only folding)")
    p.add_argument("--job-name", metavar="NAME", default=None,
                   help="Job name (used for output filenames; defaults to PDB basename or 'af3_job')")
    p.add_argument("--output-dir", metavar="DIR", required=True,
                   help="Directory to write AF3 results")
    p.add_argument("--zf-count", type=int, default=0,
                   help="Number of zinc ions to add (ignored when --pdb is used; count extracted from structure)")
    return p


def main():
    args = build_parser().parse_args()

    pdb_file    = None
    protein_seq = None
    dna_seq     = args.dna_seq.upper() if args.dna_seq else None
    zf_count    = args.zf_count
    template    = args.template

    if args.pdb:
        pdb_file = args.pdb
        protein_chains, dna_chains, zn_from_pdb = extract_sequences_from_pdb(pdb_file)
        # Use zinc count from PDB if present; otherwise fall back to --zf-count
        zf_count = zn_from_pdb if zn_from_pdb > 0 else args.zf_count
        protein_seq = list(protein_chains.values())
        if dna_seq is None and dna_chains:
            dna_seq = list(dna_chains.values())[0]
            print(f"DNA from PDB: {dna_seq}")
        print(f"Zinc ions: {zf_count} (from {'PDB' if zn_from_pdb > 0 else '--zf-count'})")

    elif args.protein_seq:
        protein_seq = [s.strip() for s in args.protein_seq.split(",")]

    elif args.template:
        pass  # protein comes from the template JSON

    else:
        raise ValueError("Supply one of --pdb, --protein-seq, or --template.")

    os.makedirs(args.output_dir, exist_ok=True)
    AF3_folding(
        pdb_file=pdb_file,
        protein_seq=protein_seq,
        dna_seq=dna_seq,
        job_name=args.job_name,
        output_dir=args.output_dir,
        template=template,
        zf_count=zf_count,
    )


if __name__ == "__main__":
    main()
