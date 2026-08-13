#!/usr/bin/env python
"""Targeted re-check: only structures with is_dimer=True in the current file
need re-verification against the consolidated (DNA-duplex AND real
protein-protein contact) criterion -- a False row can never flip to True
under a strictly more restrictive check. Prints progress every 25 structures.
"""
import sys, time
import pandas as pd

sys.path.insert(0, "scripts")
from build_deeppbs_structural_v2 import fetch_cif, load_chains, find_dimer_partners

df = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2.parquet")
print(f"total rows: {len(df)}", flush=True)

gene_lookup = df.set_index(["pdb_id", "chain_id"])["gene"].to_dict()

to_check_pdbs = sorted(df[df["is_dimer"] == True]["pdb_id"].unique())
print(f"structures needing re-check (currently is_dimer=True): {len(to_check_pdbs)}", flush=True)

new_partner_chains = {}
new_partner_genes = {}
new_is_dimer = {}

t0 = time.time()
for i, pdb_id in enumerate(to_check_pdbs):
    group = df[df["pdb_id"] == pdb_id]
    try:
        cif = fetch_cif(pdb_id)
        prot_chains, dna_coords, dna_chain_atoms = load_chains(cif)
        contacts, chain_dna_contacts, protein_partners = find_dimer_partners(prot_chains, dna_coords, dna_chain_atoms)
    except Exception as e:
        print(f"  [{i+1}] {pdb_id}: ERROR {e}", flush=True)
        continue
    for cid in group["chain_id"]:
        partners = protein_partners.get(cid, [])
        new_partner_chains[(pdb_id, cid)] = partners
        new_partner_genes[(pdb_id, cid)] = [gene_lookup.get((pdb_id, c)) for c in partners]
        new_is_dimer[(pdb_id, cid)] = len(partners) > 0

    if (i + 1) % 25 == 0 or (i + 1) == len(to_check_pdbs):
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        remaining = (len(to_check_pdbs) - i - 1) * rate
        print(f"  [{i+1}/{len(to_check_pdbs)}] elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s", flush=True)

def get_col(r, d, default):
    return d.get((r["pdb_id"], r["chain_id"]), default)

df["partner_chains"] = df.apply(lambda r: get_col(r, new_partner_chains, r["partner_chains"]), axis=1)
df["partner_genes"] = df.apply(lambda r: get_col(r, new_partner_genes, r["partner_genes"]), axis=1)
df["is_dimer"] = df.apply(lambda r: get_col(r, new_is_dimer, r["is_dimer"]), axis=1)

print(f"final is_dimer counts:\n{df['is_dimer'].value_counts()}", flush=True)
df.to_parquet("/tmp/tf_pwm_deeppbs_v2_targeted.parquet")
print("saved to /tmp/tf_pwm_deeppbs_v2_targeted.parquet", flush=True)
