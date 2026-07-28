#!/usr/bin/env python
"""Batch driver for build_deeppbs_structural_v2.py.

Iterates over the full candidate PDB pool (existing DeepPBS cache + newly
found structures, 1278 total per the audit session), extracts DNA-contacting
chains with a contiguous crop, resolves gene + PWM per chain, and writes
incremental JSONL (resilient to interruption) plus a final parquet.

Output schema, one row per (pdb_id, chain) that has a resolvable gene + PWM:
  pdb_id, chain_id, gene, uniprot_id, sequence, seq_length, gap_flag,
  n_contact_residues, pwm (bytes, float32 (4,L)), pwm_source, pwm_source_id,
  motif_length, partner_chains (list of other DNA-contacting chain ids in the
  same structure), partner_genes, is_dimer
"""
import json, os, sys, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_deeppbs_structural_v2 import (
    fetch_cif, load_chains, find_dimer_partners,
    fetch_chain_uniprot_map, resolve_gene, resolve_pwm,
)

ALL_IDS_FILE = "/tmp/all_pdb_ids_to_process.json"
OUT_JSONL = "data/processed/tf_pwm_deeppbs_v2.jsonl"
FAIL_LOG = "data/processed/tf_pwm_deeppbs_v2_failures.jsonl"
OUT_PARQUET = "data/processed/tf_pwm_deeppbs_v2.parquet"


def already_done_ids():
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["_pdb_id"])
                except Exception:
                    pass
    if os.path.exists(FAIL_LOG):
        with open(FAIL_LOG) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pdb_id"])
                except Exception:
                    pass
    return done


def process_one(pdb_id, out_f, fail_f):
    try:
        cif_path = fetch_cif(pdb_id)
    except Exception as e:
        fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "fetch_cif", "error": str(e)}) + "\n")
        return 0

    try:
        prot_chains, dna_coords, dna_chain_atoms = load_chains(cif_path)
    except Exception as e:
        fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "load_chains", "error": str(e)}) + "\n")
        return 0

    if len(dna_coords) == 0 or not prot_chains:
        fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "no_dna_or_protein", "error": ""}) + "\n")
        return 0

    contacts, chain_dna_contacts, protein_partners = find_dimer_partners(prot_chains, dna_coords, dna_chain_atoms)
    if not contacts:
        fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "no_contact", "error": ""}) + "\n")
        return 0

    chain_uniprot = fetch_chain_uniprot_map(pdb_id)

    chain_gene = {}
    for cid in contacts:
        uids = chain_uniprot.get(cid, [])
        gene = resolve_gene(uids) if uids else None
        if gene:
            chain_gene[cid] = gene

    n_written = 0
    for cid, (seq, gap_flag, n_contact) in contacts.items():
        gene = chain_gene.get(cid)
        if not gene:
            fail_f.write(json.dumps({"pdb_id": pdb_id, "chain": cid, "stage": "no_gene", "error": ""}) + "\n")
            continue
        try:
            pwm_result = resolve_pwm(gene)
        except Exception as e:
            fail_f.write(json.dumps({"pdb_id": pdb_id, "chain": cid, "gene": gene, "stage": "resolve_pwm", "error": str(e)}) + "\n")
            continue
        if pwm_result is None:
            fail_f.write(json.dumps({"pdb_id": pdb_id, "chain": cid, "gene": gene, "stage": "no_pwm", "error": ""}) + "\n")
            continue
        pwm, pwm_source, pwm_source_id = pwm_result

        # true partners come straight from find_dimer_partners's authoritative
        # protein_partners dict (primary-DNA-duplex overlap AND real
        # protein-protein contact, both required -- see its docstring)
        partners = protein_partners.get(cid, [])
        partner_genes = [chain_gene.get(c) for c in partners]

        row = {
            "_pdb_id": pdb_id,
            "pdb_id": pdb_id,
            "chain_id": cid,
            "gene": gene,
            "sequence": seq,
            "seq_length": len(seq),
            "gap_flag": bool(gap_flag),
            "n_contact_residues": int(n_contact),
            "pwm_flat": pwm.astype(np.float32).flatten().tolist(),
            "pwm_shape": list(pwm.shape),
            "pwm_source": pwm_source,
            "pwm_source_id": str(pwm_source_id),
            "motif_length": int(pwm.shape[1]),
            "partner_chains": partners,
            "partner_genes": partner_genes,
            "is_dimer": len(partners) > 0,
        }
        out_f.write(json.dumps(row) + "\n")
        n_written += 1

    if n_written == 0 and contacts:
        fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "contacts_found_but_no_output", "error": ""}) + "\n")

    return n_written


def main():
    all_ids = json.load(open(ALL_IDS_FILE))
    done = already_done_ids()
    todo = [i for i in all_ids if i not in done]
    print(f"[{time.strftime('%H:%M:%S')}] total={len(all_ids)} already_done={len(done)} todo={len(todo)}", flush=True)

    total_rows = 0
    with open(OUT_JSONL, "a") as out_f, open(FAIL_LOG, "a") as fail_f:
        for i, pdb_id in enumerate(todo):
            try:
                n = process_one(pdb_id, out_f, fail_f)
                total_rows += n
            except Exception as e:
                fail_f.write(json.dumps({"pdb_id": pdb_id, "stage": "unhandled", "error": str(e), "tb": traceback.format_exc()}) + "\n")
            out_f.flush(); fail_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(todo)} structures processed, {total_rows} rows written so far", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE. {total_rows} new rows written this run.", flush=True)

    # compile final parquet from the full JSONL
    rows = [json.loads(l) for l in open(OUT_JSONL)]
    for r in rows:
        r["pwm"] = np.array(r.pop("pwm_flat"), dtype=np.float32).reshape(r.pop("pwm_shape")).tobytes()
        r.pop("_pdb_id", None)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {len(df)} total rows to {OUT_PARQUET}", flush=True)


if __name__ == "__main__":
    main()
