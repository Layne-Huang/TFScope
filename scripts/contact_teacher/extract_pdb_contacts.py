#!/usr/bin/env python
"""Extract residue->base contact maps from protein-DNA co-crystal PDB structures.

These contacts are the TEACHER targets for the contact-distillation loss (step 3 of the
contact-sparse-attention plan): for each DNA base, which protein residues read it. We compute
min heavy-atom distance between every protein residue and every DNA base, threshold to define
contacts, and emit a per-base distribution over residues.

Self-contained PDB parsing (no Biopython dependency): reads ATOM/HETATM records, groups by
chain + residue, separates protein (20 aa) from DNA (DA/DC/DG/DT), computes pairwise
min-heavy-atom distances.

CLI:
  python scripts/contact_teacher/extract_pdb_contacts.py --pdb 1HDD.pdb --sanity-homeodomain
  python scripts/contact_teacher/extract_pdb_contacts.py --pdb-id 1HDD --out results/contact_teacher
"""
import os, sys, json, argparse, urllib.request
import numpy as np

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
DNA = {"DA","DC","DG","DT"}
AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
          "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
          "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
CONTACT_THRESH = 4.5   # Angstrom, min heavy-atom distance

def download_pdb(pdb_id, dest):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, dest)
    return dest

def parse_pdb(path):
    """Return dict: {'prot': [(chain,resnum,resname,one,coords(N,3))], 'dna': [...]}.
    Only heavy atoms (skip H). Takes first altloc."""
    prot, dna = {}, {}
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        atom = ln[12:16].strip()
        if atom.startswith("H") or (len(atom) > 1 and atom[0].isdigit() and atom[1] == "H"):
            continue
        alt = ln[16]
        if alt not in (" ", "A"):
            continue
        resn = ln[17:20].strip()
        chain = ln[21]
        resseq = ln[22:27].strip()      # incl insertion code
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        key = (chain, resseq)
        if resn in AA3:
            prot.setdefault(key, {"resn": resn, "xyz": []})["xyz"].append(xyz)
        elif resn in DNA:
            dna.setdefault(key, {"resn": resn, "xyz": []})["xyz"].append(xyz)
    def pack(d):
        out = []
        for (ch, rs), v in d.items():
            out.append({"chain": ch, "resseq": rs, "resn": v["resn"],
                        "xyz": np.asarray(v["xyz"], dtype=float)})
        return out
    return pack(prot), pack(dna)

def min_dist(a, b):
    # a:(Na,3) b:(Nb,3) -> min pairwise euclidean
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(d.min())

def contact_map(prot, dna):
    """Return (D matrix [Lprot,Ldna], prot_labels, dna_labels)."""
    D = np.zeros((len(prot), len(dna)))
    for i, p in enumerate(prot):
        for j, b in enumerate(dna):
            D[i, j] = min_dist(p["xyz"], b["xyz"])
    pl = [f"{p['chain']}/{AA3to1[p['resn']]}{p['resseq']}" for p in prot]
    dl = [f"{b['chain']}/{b['resn'][-1]}{b['resseq']}" for b in dna]
    return D, pl, dna_labels_clean(dl)

def dna_labels_clean(dl):
    return [d.replace("D", "") for d in dl]

def base_residue_targets(D, thresh=CONTACT_THRESH, tau=2.0):
    """For each base (column), soft target over residues: softmax(-d/tau) restricted to
    residues within `thresh`. Returns targets (Ldna,Lprot) and the hard contact set."""
    Lp, Ld = D.shape
    T = np.zeros((Ld, Lp)); contacts = []
    for j in range(Ld):
        near = np.where(D[:, j] <= thresh)[0]
        if len(near) == 0:
            continue
        w = np.exp(-D[near, j] / tau); w /= w.sum()
        T[j, near] = w
        contacts.append((j, near.tolist()))
    return T, contacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb"); ap.add_argument("--pdb-id")
    ap.add_argument("--out", default="results/contact_teacher")
    ap.add_argument("--thresh", type=float, default=CONTACT_THRESH)
    ap.add_argument("--sanity-homeodomain", action="store_true",
                    help="check contacts concentrate on homeodomain recognition residues")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    path = args.pdb
    if args.pdb_id:
        path = os.path.join(args.out, f"{args.pdb_id.upper()}.pdb")
        if not os.path.exists(path):
            download_pdb(args.pdb_id, path)
    prot, dna = parse_pdb(path)
    print(f"[contacts] {os.path.basename(path)}: {len(prot)} protein residues, {len(dna)} DNA bases")
    if not prot or not dna:
        sys.exit("no protein or no DNA chain found")
    D, pl, dl = contact_map(prot, dna)
    T, contacts = base_residue_targets(D, args.thresh)
    n_contacting_bases = len(contacts)
    n_pairs = int((D <= args.thresh).sum())
    print(f"[contacts] {n_contacting_bases}/{len(dna)} bases contacted; "
          f"{n_pairs} residue-base pairs within {args.thresh} A")

    # contacting residues (those reading >=1 base)
    contacting_res = sorted({r for _, rs in contacts for r in rs})
    print("[contacts] reading residues:",
          ", ".join(pl[r] for r in contacting_res[:40]))

    if args.sanity_homeodomain:
        sanity_homeodomain(prot, pl, contacting_res)

    np.savez(os.path.join(args.out, os.path.basename(path).replace(".pdb", "_contacts.npz")),
             D=D, T=T, prot_labels=np.array(pl), dna_labels=np.array(dl),
             thresh=args.thresh)
    print(f"saved -> {args.out}/{os.path.basename(path).replace('.pdb','_contacts.npz')}")

def sanity_homeodomain(prot, pl, contacting_res):
    """Engrailed/Antennapedia-type homeodomain DNA recognition is dominated by
    Ile47, Gln50, Asn51 (recognition helix) + Arg5/Arg3 (N-term arm) contacting the
    TAAT core. We check that the reading residues include these identities."""
    print("\n=== HOMEODOMAIN SANITY TEST ===")
    reading = [pl[r] for r in contacting_res]
    # the canonical recognition residues by amino-acid identity in the reading set
    want = {"N": "Asn51-like", "Q": "Gln50-like", "I": "Ile47-like", "R": "Arg arm"}
    hits = {}
    for lab in reading:
        aa = lab.split("/")[1][0]
        if aa in want:
            hits.setdefault(aa, []).append(lab)
    print("reading residues:", ", ".join(reading))
    print("\ncanonical recognition identities present in the contact set:")
    for aa, desc in want.items():
        present = hits.get(aa, [])
        flag = "PASS" if present else "----"
        print(f"  [{flag}] {desc:12s} ({aa}): {', '.join(present) if present else 'absent'}")
    crit = sum(1 for aa in ("N", "Q", "R") if aa in hits)
    print(f"\n=> {crit}/3 critical recognition identities (Asn/Gln/Arg) found in contacts "
          f"{'-- SANITY PASS' if crit >= 2 else '-- CHECK ALIGNMENT'}")

if __name__ == "__main__":
    main()
