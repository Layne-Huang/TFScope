"""AF3+Rosetta calibration case study — step 3 (generic): fetch MSA once per gene,
run all local AF3 folds via the af3_mmseqs_scripts Docker wrapper on this machine.
Run with the `base` conda env (has af3cli/requests/Bio/tqdm installed).

Usage: python scripts/run_af3_local_generic.py --gene E2F4
"""
import sys, os, json, glob, copy, argparse

sys.path.insert(0, "/data1/leihuang/af3_mmseqs_scripts/af3_mmseqs2")
from add_mmseqs_msa import add_msa_to_json
from alphafold3 import run_alphafold3

MODEL_PARAMS = "/data1/leihuang/AF3_parameter"
DATABASE = "/data1/leihuang/af3_placeholder_db"  # unused: MSA is pre-embedded, no templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    args = ap.parse_args()
    gene = args.gene

    # Docker bind-mounts can't authenticate to AFS -- inputs/outputs must live on local disk.
    IN_DIR = f"/data1/leihuang/project/TFScope/case_study/{gene.lower()}/af3_inputs"
    MMSEQS_DIR = os.path.join(IN_DIR, "mmseqs")
    OUT_DIR = f"/data1/leihuang/project/TFScope/case_study/{gene.lower()}/af3_output"
    os.makedirs(MMSEQS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    src_dir = f"results/calibration_case_study/{gene}/af3_inputs"
    for f in glob.glob(os.path.join(src_dir, "*.json")):
        if f.endswith("manifest.json"):
            continue
        dst = os.path.join(IN_DIR, os.path.basename(f))
        json.dump(json.load(open(f)), open(dst, "w"))

    jsons = sorted(f for f in glob.glob(os.path.join(IN_DIR, "*.json")) if not f.endswith("manifest.json"))
    print(f"{len(jsons)} jobs found for {gene}")

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
    print(f"MSA fetched ({len(msa_fields['unpairedMsa'])} chars). Reusing across all {len(jsons)} jobs.")

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

    print(f"Done with {gene}.")


if __name__ == "__main__":
    main()
