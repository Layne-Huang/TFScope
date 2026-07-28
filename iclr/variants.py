"""ICLR 2026 Phase-I necessity audit — variant registry (plan §3).

Each variant is defined as a delta from the immutable v24 recipe (the exact
flags in ``scripts/run_v24_contact_ddp.sh``). This module is the single source
of truth for *what* B0–B8 are; it emits ready-to-run ``scripts/train.py`` /
baseline command lines so every variant uses identical preprocessing, PWM
registration, max output length, optimisation budget, early-stopping rule, and
evaluation code (plan §2, rules 2–3).

Nothing here trains anything; it prints commands. Run:

    python -m iclr.variants                 # list variants + purposes
    python -m iclr.variants B5 --seed 1     # print the exact command for B5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

# ── canonical benchmark artifacts (plan §2 rule 2) ─────────────────────────────
# The canonical Phase-I benchmark lives entirely in one parquet + one split:
#   * split['train'] (4794 rows)  -> training
#   * split['test']  (291 str_* rows) -> the immutable 291-row structure test set
#   * monomer vs multimer is read from the parquet's n_chains / is_dimer columns
# All variants use the identical split and are evaluated on the identical 291-row
# test set (plan §2 rules 2–3). Override on the CLI if your node's paths differ.
DATA_TRAIN   = "data/processed/tf_pwm_training_v23.parquet"
SPLIT_TRAIN  = "data/processed/splits/train_v22/split.json"          # cluster40-derived
TEST_DATA    = DATA_TRAIN                                            # test rows = split['test']
TEST_SPLIT   = SPLIT_TRAIN
# Frozen v24 reference (plan §2 rule 1: immutable). The canonical base checkpoint
# is contact_v24_seed42/ckpt_best.pt; set --v24-ckpt to the path on your node.
V24_CKPT     = "checkpoints/v24_e1_paired/pwmhead_ft.pt"

# Optimisation / registration / gate / loss flags shared by every *trained*
# variant so the only thing that differs is the architectural component under
# test (plan §2 rule 3). Data/split/out/seed/DDP are added per-run.
COMMON_TRAIN = [
    "--warmup-steps", "150", "--workers", "2", "--save-every", "25",
    "--gate-mode", "span", "--max-motif-length", "42", "--motif-overflow-policy", "error",
    "--latent-registration",
    "--pwm-cov-r-weight", "0.25", "--pwm-core-ic-thresh", "0.25",
    "--gate-length-weight", "0.05",
    "--ic-pcc-weight", "0.5", "--topbase-weight", "0.1", "--topbase-margin", "2.0",
    "--group-balanced-sampling",
    "--eval-oracle-r", "--oracle-r-every", "5", "--oracle-aggregation", "gene",
    "--early-stop-patience", "30",
    "--precision", "bf16", "--tf32", "--no-wandb",
]

# v24's frozen-vs-tuned encoder recipe. Simple baselines (B2–B4) keep ESM fully
# frozen (lora-rank 0); v24-derived ablations (B5–B7) keep v24's LoRA recipe.
FROZEN_ESM = ["--lora-rank", "0"]
V24_LORA   = ["--lora-rank", "16", "--lora-alpha", "32", "--lora-n-layers", "6",
              "--lr", "4.5e-4", "--lora-lr", "7.5e-6"]

# The full v24 head/MoE/contact/chain block (everything that makes v24 "complete").
V24_MOE     = ["--moe-granularity", "residue", "--num-experts", "8",
               "--n-shared-experts", "2", "--top-k", "2", "--expert-hidden-dim", "512",
               "--balance-loss-weight", "0.01", "--diversity-loss-weight", "0.0",
               "--family-embedding-path", "none"]
V24_V18     = ["--pwm-head-v18"]
V24_CONTACT = ["--v18-contact-supervision", "--v18-contact-weight", "0.3",
               "--v18-contact-bias-scale", "1.0", "--v18-contact-bias-learnable", "1",
               "--contact-distill-weight", "0.2",
               "--contact-targets-path", "data/contact_maps/contact_targets_v23.json",
               "--recognition-prior-path", "data/contact_maps/recognition_residues_v23.json"]
V24_CHAIN   = ["--two-chain-input", "--chain-id-embedding", "--max-chains", "4"]


@dataclass
class Variant:
    vid: str
    name: str
    purpose: str
    kind: str                    # "baseline" (training-free) | "train" | "frozen_ckpt"
    extra: list = field(default_factory=list)   # train.py flags beyond COMMON_TRAIN
    note: str = ""

    def command(self, seed: int = 42, out_root: str = "checkpoints/iclr_phase1") -> str:
        if self.kind == "frozen_ckpt":
            return (f"# B8 is the frozen v24 checkpoint — do NOT retrain.\n"
                    f"python -m iclr.evaluate --ckpt {V24_CKPT} "
                    f"--test-data {TEST_DATA} --test-split {TEST_SPLIT} "
                    f"--tag {self.vid}_v24 --out {out_root}/{self.vid}")
        if self.kind == "baseline":
            return (f"python -m iclr.baselines --variant {self.vid} "
                    f"--train-data {DATA_TRAIN} --split {SPLIT_TRAIN} "
                    f"--test-data {TEST_DATA} --test-split {TEST_SPLIT} "
                    f"--out {out_root}/{self.vid}")
        flags = COMMON_TRAIN + self.extra
        return (f"python scripts/train.py --data {DATA_TRAIN} --split {SPLIT_TRAIN} "
                f"--out {out_root}/{self.vid}/seed{seed} --seed {seed} --epochs 225 "
                f"--batch-size 12 --grad-accum-steps 3 " + " ".join(flags))


# ── the registry (plan §3 table) ───────────────────────────────────────────────
VARIANTS: dict[str, Variant] = {v.vid: v for v in [
    Variant("B0", "family-average PWM", "family prior floor", "baseline"),
    Variant("B1", "nearest training sequence/PWM", "memorisation & retrieval control", "baseline"),
    Variant("B2", "frozen ESM + mean pool + MLP", "strongest simple baseline", "train",
            extra=FROZEN_ESM + ["--mean-pool", "--no-moe"],
            note="no v18 head, no contacts, no chain; masked mean pooling."),
    Variant("B3", "frozen ESM + attention pool + MLP", "pooling control", "train",
            extra=FROZEN_ESM + ["--no-moe"],
            note="B2 with gated-attention pooling instead of mean pooling."),
    Variant("B4", "ESM + span gate", "isolates variable-length prediction", "train",
            extra=FROZEN_ESM + ["--no-moe", "--gate-length-weight", "0.1"],
            note="B3 emphasising the continuous span gate (variable-length core)."),
    Variant("B5", "v24 without MoE", "tests MoE necessity", "train",
            extra=V24_LORA + V24_V18 + V24_CONTACT + V24_CHAIN + ["--no-moe"]),
    Variant("B6", "v24 without contact losses/bias", "tests contact contribution", "train",
            extra=V24_LORA + V24_MOE + V24_V18 + V24_CHAIN,
            note="drops --v18-contact-* and --contact-distill-weight entirely."),
    Variant("B7", "v24 N-chain, minimal head", "tests chain-input contribution", "train",
            extra=V24_LORA + V24_CHAIN + ["--no-moe"],
            note="keeps N-chain input but strips MoE, v18 head, and contacts."),
    Variant("B8", "complete v24", "frozen reference", "frozen_ckpt"),
]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vid", nargs="?", help="variant id (B0..B8); omit to list all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default="checkpoints/iclr_phase1")
    args = ap.parse_args()

    if args.vid is None:
        print(f"{'ID':<4}{'kind':<12}{'name':<34}purpose")
        print("-" * 92)
        for v in VARIANTS.values():
            print(f"{v.vid:<4}{v.kind:<12}{v.name:<34}{v.purpose}")
        print("\nPer plan §3: run B0–B8 on the identical cluster40 split + 291-row test.")
        print("Use ≥3 seeds for trained variants (§2 rule 4). Example:")
        print("  python -m iclr.variants B5 --seed 1")
        return

    v = VARIANTS[args.vid.upper()]
    print(f"# {v.vid}: {v.name} — {v.purpose}")
    if v.note:
        print(f"# note: {v.note}")
    print(v.command(seed=args.seed, out_root=args.out_root))


if __name__ == "__main__":
    main()
