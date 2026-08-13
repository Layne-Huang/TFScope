#!/usr/bin/env python
"""Download ProTrek_650M checkpoint and extract the text encoder.

ProTrek checkpoint structure (westlake-repl/ProTrek_650M_UniRef50):
  ckpt['model'] is the state dict with numeric top-level prefixes:
    "1.*"  — ESM-2 sequence encoder (573 keys)
    "2.*"  — Text encoder: PubMedBERT backbone + projection head (200 keys)
    "3.*"  — Structure encoder (522 keys)

  Text encoder keys:
    "2.model.*"  → BERT weights (strip to "model.*")
    "2.out.*"    → projection head weight [1024, 768] + bias [1024]

Saves the text encoder to:
  {CACHE_DIR}/protrek/ProTrek_650M_UniRef50/text_model/
    pytorch_model.bin     — BERT weights (loadable by AutoModel)
    text_projection.pt    — {"weight": ..., "bias": ...}
    protrek_meta.json     — {"projection_dim": 1024, "has_projection": true}
    tokenizer files       — copied from bundled PubMedBERT in the HF repo

The precompute_family_embeddings.py script picks this up automatically.

Usage:
    mamba run -n tfscope python scripts/download_protrek.py
"""

import json
import os
import sys

CACHE_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache"
os.environ["HF_HOME"]            = CACHE_DIR
os.environ["TORCH_HOME"]         = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig

REPO_ID   = "westlake-repl/ProTrek_650M_UniRef50"
CKPT_FILE = "ProTrek_650M.pt"

# Text backbone: PubMedBERT (used as ProTrek's text encoder backbone)
TEXT_BACKBONE = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Numeric prefix for the text encoder in the ProTrek checkpoint
TEXT_PREFIX = "2."

OUT_DIR = os.path.join(CACHE_DIR, "protrek", "ProTrek_650M_UniRef50", "text_model")


def main():
    # ── 1. Download full checkpoint ───────────────────────────────────────────
    print(f"Downloading {REPO_ID}/{CKPT_FILE} ...")
    ckpt_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CKPT_FILE,
        cache_dir=os.path.join(CACHE_DIR, "hub"),
    )
    print(f"Checkpoint at: {ckpt_path}")

    # ── 2. Load checkpoint and get state dict ─────────────────────────────────
    print("Loading checkpoint (weights_only=False) ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print(f"  Using ckpt['model'] — epoch={ckpt.get('epoch', '?')}, "
              f"step={ckpt.get('global_step', '?')}")
    else:
        state_dict = ckpt

    all_keys = list(state_dict.keys())
    print(f"Total keys in state dict: {len(all_keys)}")

    # Verify text encoder prefix exists
    text_keys = [k for k in all_keys if k.startswith(TEXT_PREFIX)]
    print(f"Text encoder keys (prefix '{TEXT_PREFIX}'): {len(text_keys)}")
    if not text_keys:
        print("ERROR: No keys found with prefix '2.' — checkpoint may have different structure.")
        prefixes = sorted({k.split(".")[0] for k in all_keys})
        print(f"Top-level prefixes found: {prefixes}")
        sys.exit(1)

    # ── 3. Extract BERT backbone weights ──────────────────────────────────────
    # Keys: "2.model.*" → strip "2." to get "model.*"
    bert_state = {
        k[len(TEXT_PREFIX):]: v
        for k, v in state_dict.items()
        if k.startswith(TEXT_PREFIX + "model.")
    }
    print(f"BERT backbone keys extracted: {len(bert_state)}")
    if not bert_state:
        print("ERROR: No BERT backbone keys found (expected '2.model.*')")
        print("Sample text encoder keys:", text_keys[:10])
        sys.exit(1)

    # ── 4. Extract projection head weights ───────────────────────────────────
    # Keys: "2.out.weight" [1024, 768], "2.out.bias" [1024]
    proj_weight_key = TEXT_PREFIX + "out.weight"
    proj_bias_key   = TEXT_PREFIX + "out.bias"

    proj_state = {}
    if proj_weight_key in state_dict:
        proj_state["weight"] = state_dict[proj_weight_key]
        print(f"Projection weight: {state_dict[proj_weight_key].shape}")
    if proj_bias_key in state_dict:
        proj_state["bias"] = state_dict[proj_bias_key]
        print(f"Projection bias:   {state_dict[proj_bias_key].shape}")

    if not proj_state:
        print("WARNING: No projection head found (keys '2.out.weight', '2.out.bias')")
        print("Continuing without projection — embeddings will be raw BERT CLS vectors.")

    # ── 5. Save BERT weights ──────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    bert_out = os.path.join(OUT_DIR, "pytorch_model.bin")
    torch.save(bert_state, bert_out)
    print(f"\nSaved BERT weights → {bert_out}")

    # ── 6. Save projection weights ────────────────────────────────────────────
    if proj_state:
        proj_out = os.path.join(OUT_DIR, "text_projection.pt")
        torch.save(proj_state, proj_out)
        print(f"Saved projection  → {proj_out}")

        proj_dim = proj_state["weight"].shape[0]
        meta = {"projection_dim": proj_dim, "has_projection": True}
        with open(os.path.join(OUT_DIR, "protrek_meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"Saved meta        → protrek_meta.json  (proj_dim={proj_dim})")

    # ── 7. Save tokenizer ─────────────────────────────────────────────────────
    # Try bundled tokenizer in repo first (avoids HF rate limits)
    tokenizer_loaded = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ID,
            subfolder="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            cache_dir=os.path.join(CACHE_DIR, "hub"),
        )
        tokenizer.save_pretrained(OUT_DIR)
        print(f"Saved tokenizer (from bundled repo subfolder) → {OUT_DIR}")
        tokenizer_loaded = True
    except Exception as e:
        print(f"Bundled tokenizer not found ({e}), falling back to HF PubMedBERT ...")

    if not tokenizer_loaded:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_BACKBONE,
            cache_dir=os.path.join(CACHE_DIR, "hub"),
        )
        tokenizer.save_pretrained(OUT_DIR)
        print(f"Saved tokenizer (from {TEXT_BACKBONE}) → {OUT_DIR}")

    # ── 8. Save BERT config ───────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(
        TEXT_BACKBONE,
        cache_dir=os.path.join(CACHE_DIR, "hub"),
    )
    config.save_pretrained(OUT_DIR)
    print(f"Saved BERT config → {OUT_DIR}")

    print(f"\nDone. Text encoder saved to:\n  {OUT_DIR}")
    print("You can now run:")
    print("  mamba run -n tfscope python scripts/precompute_family_embeddings.py "
          "--text-model protrek")


if __name__ == "__main__":
    main()
