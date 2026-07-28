#!/usr/bin/env python
"""Resumable version: writes one JSON line per structure IMMEDIATELY as it's
computed (not batched at the end), so an AFS-token interruption (this
environment has repeatedly dropped tokens mid-run, shorter than their stated
klist validity would suggest) only costs the time since the last checkpoint,
never the whole run. Re-running this script skips whatever's already in the
checkpoint file and only processes what's missing.
"""
import json, os, sys, time
import pandas as pd

sys.path.insert(0, "scripts")
from build_deeppbs_structural_v2 import fetch_cif, load_chains, find_dimer_partners

CHECKPOINT = "/tmp/dimer_resumable_checkpoint.jsonl"

df = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2.parquet")
print(f"total rows: {len(df)}", flush=True)

to_check_pdbs = sorted(df[df["is_dimer"] == True]["pdb_id"].unique())
print(f"structures needing re-check: {len(to_check_pdbs)}", flush=True)

done_pdbs = set()
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        for line in f:
            try:
                done_pdbs.add(json.loads(line)["pdb_id"])
            except Exception:
                pass
print(f"already checkpointed: {len(done_pdbs)}", flush=True)

todo = [p for p in to_check_pdbs if p not in done_pdbs]
print(f"remaining to process: {len(todo)}", flush=True)

t0 = time.time()
n_ok, n_err = 0, 0
with open(CHECKPOINT, "a") as ckpt_f:
    for i, pdb_id in enumerate(todo):
        group = df[df["pdb_id"] == pdb_id]
        try:
            cif = fetch_cif(pdb_id)
            prot_chains, dna_coords, dna_chain_atoms = load_chains(cif)
            contacts, chain_dna_contacts, protein_partners = find_dimer_partners(prot_chains, dna_coords, dna_chain_atoms)
            record = {
                "pdb_id": pdb_id,
                "chains": {
                    cid: {"partners": protein_partners.get(cid, [])}
                    for cid in group["chain_id"]
                },
            }
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()
            n_ok += 1
        except Exception as e:
            ckpt_f.write(json.dumps({"pdb_id": pdb_id, "error": str(e)}) + "\n")
            ckpt_f.flush()
            n_err += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = (len(todo) - i - 1) * rate
            print(f"  [{i+1}/{len(todo)}] ok={n_ok} err={n_err} elapsed={elapsed:.0f}s est_remaining={remaining:.0f}s", flush=True)

print(f"DONE this pass. ok={n_ok} err={n_err}", flush=True)
