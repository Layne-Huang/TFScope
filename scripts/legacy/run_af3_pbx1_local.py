"""PBX1 AF3+Rosetta calibration case study — step 3: fetch MSA once, run all 9 local
AF3 folds via the af3_mmseqs_scripts Docker wrapper (this machine, not the Harvard
cluster). Run with the `base` conda env (has af3cli/requests/Bio/tqdm installed).

All 9 jobs share the identical PBX1 DBD protein chain, so the remote MMseqs2 MSA
(https://a3m.mmseqs.com) is fetched ONCE and reused across all 9 job JSONs instead
of 9 redundant remote calls.
"""
import sys, os, json, glob, copy

sys.path.insert(0, "/data1/leihuang/af3_mmseqs_scripts/af3_mmseqs2")
from add_mmseqs_msa import add_msa_to_json
from alphafold3 import run_alphafold3

IN_DIR = "/data1/leihuang/project/TFScope/case_study/pbx1/af3_inputs"  # must be local disk, not AFS: Docker bind-mounts can't authenticate to AFS
MMSEQS_DIR = os.path.join(IN_DIR, "mmseqs")  # MSA-augmented copies live here, never globbed as a source job
OUT_DIR = "/data1/leihuang/project/TFScope/case_study/pbx1/af3_output"
MODEL_PARAMS = "/data1/leihuang/AF3_parameter"
DATABASE = "/data1/leihuang/af3_placeholder_db"  # unused: MSA is pre-embedded, no templates


def main():
    os.makedirs(MMSEQS_DIR, exist_ok=True)
    jsons = sorted(f for f in glob.glob(os.path.join(IN_DIR, "*.json")) if not f.endswith("manifest.json"))
    print(f"{len(jsons)} jobs found in {IN_DIR}")

    # 1. fetch MSA once, from the first job (all share the same protein chain)
    base = json.load(open(jsons[0]))
    print(f"Fetching MMseqs2 MSA for protein chain (len={len(base['sequences'][0]['protein']['sequence'])}) ...")
    augmented = add_msa_to_json(
        input_json=jsons[0], templates=False, num_templates=0,
        custom_template=None, custom_template_chain=None, target_id=None,
        af3_json=copy.deepcopy(base), to_file=False,
    )
    protein_entry = next(s for s in augmented["sequences"] if "protein" in s)["protein"]
    msa_fields = {
        "unpairedMsa": protein_entry["unpairedMsa"],
        "pairedMsa": protein_entry["pairedMsa"],
        "templates": protein_entry["templates"],
    }
    print(f"MSA fetched ({len(msa_fields['unpairedMsa'])} chars unpairedMsa). Reusing across all {len(jsons)} jobs.")

    # 2. patch every job with the same MSA fields, write *_mmseqs.json
    mmseqs_jsons = []
    for jp in jsons:
        d = json.load(open(jp))
        for s in d["sequences"]:
            if "protein" in s:
                s["protein"].update(msa_fields)
        out_path = os.path.join(MMSEQS_DIR, os.path.basename(jp))
        json.dump(d, open(out_path, "w"))
        mmseqs_jsons.append(out_path)
    print(f"Wrote {len(mmseqs_jsons)} MSA-augmented job JSONs to {MMSEQS_DIR}/.")

    # 3. run each fold via the Docker wrapper
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, jp in enumerate(mmseqs_jsons, 1):
        name = os.path.basename(jp).replace(".json", "")
        done_marker = os.path.join(OUT_DIR, name, f"{name}_model.cif")
        if os.path.exists(done_marker):
            print(f"[{i}/{len(mmseqs_jsons)}] {name} already done, skipping")
            continue
        print(f"[{i}/{len(mmseqs_jsons)}] folding {name} ...")
        try:
            run_alphafold3(jp, OUT_DIR, MODEL_PARAMS, DATABASE, skip_data_pipeline=True)
        except Exception as e:
            print(f"  FAILED {name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
