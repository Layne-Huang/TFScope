#!/usr/bin/env python
"""TFScope training entry point.

Basic usage (quick sanity check with dummy backbone):
    python scripts/train.py --dummy --epochs 1 --batch-size 8 --no-wandb

Train on real data with LOFO split (held-out family = Homeodomain):
    python scripts/train.py \
        --data data/processed/tf_pwm.parquet \
        --split data/processed/splits/lofo/Homeodomain.json \
        --out /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_homeodomain \
        --epochs 50 --batch-size 32

Resume from checkpoint:
    python scripts/train.py ... --resume .../ckpt_best.pt
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

# Must be set before torch.hub makes any network calls
_DEFAULT_CACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
os.environ.setdefault("TORCH_HOME", _DEFAULT_CACHE)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import (
    GeneBalancedSampler,
    SyntheticTFDataset,
    TFDataset,
    collate_variable_length,
)
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from tfscope.losses.tfscope_loss import TFScopeLoss


def get_cosine_lr(step, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to zero."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args():
    p = argparse.ArgumentParser(description="Train TFScope seed model")

    # Data
    p.add_argument("--data", default="data/processed/tf_pwm.parquet")
    p.add_argument("--split", default=None,
                   help="Path to split JSON. If omitted, uses all data for training.")
    # MoE / family taxonomy (override TFScopeConfig defaults; needed for the rebin run)
    p.add_argument("--num-families", type=int, default=None,
                   help="family-embedding size; match the parquet's family_id range (e.g. 34 for rebin)")
    p.add_argument("--num-experts", type=int, default=None, help="MoE expert count")
    p.add_argument("--diversity-loss-weight", type=float, default=None,
                   help="weight on family_diversity_loss (entropy-maximizing; set 0 to disable "
                        "and let experts specialize)")
    p.add_argument("--balance-loss-weight", type=float, default=None,
                   help="weight on load_balance_loss (set 0 to stop forcing uniform expert usage)")
    p.add_argument("--n-shared-experts", type=int, default=None,
                   help="DeepSeek-style always-active shared experts (set 0 to force routing)")
    p.add_argument("--moe-residual", type=int, default=None,
                   help="1=keep input skip x+MoE (default); 0=non-residual (output=shared+routed+proto)")
    p.add_argument("--route-supervision-weight", type=float, default=None,
                   help="weight on CE(gate_logits, family_id) routing supervision; use with a "
                        "mode-relabeled parquet so family_id==mode and num_experts==num_modes")
    p.add_argument("--top-k", type=int, default=None, help="MoE top-k routing")
    p.add_argument("--moe-granularity", default=None, choices=["protein", "residue"],
                   help="'protein'=pooled MOEBlock (1 routing decision/protein); "
                        "'residue'=per-DBD-token ResidueMoE (DeepSeekMoE-style, emergent)")
    p.add_argument("--no-moe", dest="use_moe", action="store_false", default=True,
                   help="Bypass MoE entirely (ICLR necessity audit B5: 'v24 without MoE'). "
                        "No routing / no MoE aux losses; everything else unchanged.")
    p.add_argument("--mean-pool", dest="mean_pool", action="store_true",
                   help="Use masked mean pooling instead of gated-attention pooling "
                        "(ICLR baseline B2: 'frozen ESM + mean pool + MLP').")
    p.add_argument("--contact-pred-head", action="store_true",
                   help="add frozen ESM→contact probe head; its P(contact) feeds the v18 contact bias")
    p.add_argument("--contact-probe-path", default=None,
                   help="joblib LogisticRegression contact probe to warm-start the head")
    p.add_argument("--v18-contact-bias-learnable", type=int, default=None,
                   help="1=learn the contact-bias scale (init=--v18-contact-bias-scale); 0=fixed")
    p.add_argument("--expert-hidden-dim", type=int, default=None,
                   help="per-expert hidden width (shrink when adding experts to hold total params)")
    p.add_argument("--family-embedding-path", default=None,
                   help="semantic family-embedding file; pass 'none' to force the learned embedding "
                        "(required when num-families != the precomputed file's family count)")
    p.add_argument("--v18-attn-sparse", default=None,
                   choices=["softmax", "entmax15", "sparsemax", "entmax_learn"],
                   help="hard-sparse recognition attention normalizer (default softmax = current model)")
    p.add_argument("--v18-attn-alpha-init", type=float, default=None,
                   help="initial alpha for --v18-attn-sparse entmax_learn (1<alpha<=2)")
    p.add_argument("--contact-distill-weight", type=float, default=None,
                   help="weight for structural contact-distillation KL (0 = off); needs contact_targets.json")
    p.add_argument("--contact-targets-path", default=None,
                   help="2D contact targets json (default data/contact_maps/contact_targets.json)")
    p.add_argument("--dummy", action="store_true",
                   help="Use synthetic random data + DummyBackbone (no ESM weights needed)")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-lr",       type=float, default=1e-5)
    p.add_argument("--lora-rank",     type=int,   default=0,
                   help="LoRA rank for ESM-2 fine-tuning (0=disabled)")
    p.add_argument("--lora-alpha",    type=float, default=16.0)
    p.add_argument("--lora-n-layers", type=int,   default=6,
                   help="Number of ESM-2 tail layers to inject LoRA into")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--gene-balanced-sampling",
        action="store_true",
        help="Sample genes uniformly, then choose one motif record per sampled gene.",
    )
    p.add_argument(
        "--group-balanced-sampling",
        action="store_true",
        help="Sample group_id values uniformly (gene+sequence+motif source).",
    )
    p.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Forward/backward compute precision. BF16 uses CUDA autocast.",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 tensor-core math for remaining FP32 matmuls on CUDA.",
    )

    # Model
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")

    # Output
    p.add_argument("--out",
                   default="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/run",
                   help="Directory for checkpoints and logs")
    p.add_argument("--save-every", type=int, default=20,
                   help="Save checkpoint every N epochs")
    p.add_argument("--save-epochs", default=None,
                   help="Comma-separated exact epochs to save (e.g. '150,175,200,225,250'); "
                        "when set, overrides --save-every for milestone saves")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--init-from", default=None,
                   help="Load ONLY model weights from this checkpoint (fresh optimizer/"
                        "schedule/epoch) — for stage-2 finetuning after pretraining")
    p.add_argument(
        "--init-model",
        default=None,
        help="Load model weights exactly from a compatible checkpoint.",
    )
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop training if the monitored metric doesn't improve for N epochs (0=disabled)")
    p.add_argument("--no-save-best", action="store_true",
                   help="Skip saving ckpt_best.pt (useful when val is a placeholder)")

    # Oracle-r early stopping
    p.add_argument("--eval-oracle-r", action="store_true",
                   help="Use oracle-aligned val Pearson r (instead of val loss) for "
                        "best-checkpoint selection and early stopping")
    p.add_argument("--oracle-r-every", type=int, default=5,
                   help="Compute oracle r every N epochs (default 5)")
    p.add_argument("--oracle-r-n-tfs", type=int, default=0,
                   help="Number of val TFs for oracle-r (0 = complete validation set)")
    p.add_argument("--oracle-aggregation", choices=["row", "gene"], default="gene",
                   help="Aggregate checkpoint-selection covR by row or equally by gene")
    p.add_argument("--two-chain-input", action="store_true",
                   help="Feed heterodimer partner DBD as a second chain "
                        "(chain1 + <eos> + partner) with dbd_mask over both; "
                        "requires a partner_sequence column in --data.")
    p.add_argument("--allow-unannotated-multichain", action="store_true",
                   help="Use any available partner sequence, bypassing multichain_eligible")
    p.add_argument("--chain-id-embedding", action="store_true",
                   help="Add per-protomer chain-identity embeddings (order-aware)")
    p.add_argument("--max-chains", type=int, default=2,
                   help="Max protomers fed (self + max_chains-1 partners). "
                        "2=dimer, 4=tetramer (p53/HSF/NF-Y/IRF); needs partner_seqs in --data")
    p.add_argument("--legacy-oracle-r", action="store_true",
                   help="Select on the LEGACY length-blind gate-oracle-r (overlap "
                        "only). Default is coverage-aware (r x coverage), matching "
                        "eval_full_metrics.panel_full so a collapsed gate cannot win.")
    # Benchmark eval: periodic oracle-r on a held-out, contamination-controlled set
    # (e.g. cluster40 test). val-oracle-r is lenient + on easy paralogs, so it's a poor
    # selector; this selects ckpt_best_bench on the hard test-like metric instead.
    p.add_argument("--benchmark-eval", action="store_true",
                   help="periodic oracle-r on a held-out benchmark; save ckpt_best_bench on it")
    p.add_argument("--benchmark-data", default=None, help="parquet for the benchmark eval")
    p.add_argument("--benchmark-split", default=None, help="split json for the benchmark eval (uses test split)")
    p.add_argument("--benchmark-every", type=int, default=None,
                   help="epochs between benchmark evals (default = --oracle-r-every)")

    # v14 de-novo loss terms (target base composition)
    p.add_argument("--ic-pcc-weight",  type=float, default=0.0,
                   help="IC-weighted per-column (1-Pearson) loss weight")
    p.add_argument("--topbase-weight", type=float, default=0.0,
                   help="Top-base margin loss weight (high-IC positions)")
    p.add_argument("--topbase-margin", type=float, default=2.0)
    # DPAC-style in-batch contrastive (anti family-collapse)
    p.add_argument("--contrastive-weight", type=float, default=0.0,
                   help="Weight on in-batch PWM contrastive (InfoNCE) loss")
    p.add_argument("--contrastive-tau", type=float, default=0.1,
                   help="Temperature for the contrastive similarity logits")
    p.add_argument("--init-from-pretrain", default=None,
                   help="Stage-A contrastive checkpoint to warm-start the protein encoder")
    p.add_argument(
        "--latent-registration",
        action="store_true",
        help="Marginalize PWM supervision over offset and reverse-complement states.",
    )
    p.add_argument("--registration-max-shift", type=int, default=10)
    p.add_argument("--registration-min-overlap", type=int, default=4)
    p.add_argument("--registration-temperature", type=float, default=0.1)
    p.add_argument("--registration-coverage-penalty", type=float, default=0.5)
    p.add_argument("--gate-length-weight", type=float, default=0.0,
                   help="penalty on |soft_len - gt_len|; couples the gate to the "
                        "eval protocol, where a shorter gate is scored on fewer "
                        "columns and gets an inflated r. Try 0.05-0.1.")
    p.add_argument("--gate-mode", choices=["independent", "span"], default="independent")
    p.add_argument("--max-motif-length", type=int, default=20)
    p.add_argument("--motif-overflow-policy",
                   choices=["error", "warn", "truncate"], default="warn")
    p.add_argument("--pwm-cov-r-weight", type=float, default=0.0,
                   help="weight on differentiable full-core r x soft-coverage loss")
    p.add_argument("--pwm-core-ic-thresh", type=float, default=0.25)
    p.add_argument(
        "--registration-anchor-path",
        default="",
        help="Optional train-only E3 consensus-relative anchor TSV.",
    )
    p.add_argument(
        "--register-head",
        action="store_true",
        help="Predict the 42-state register and export internal PWM coordinates.",
    )
    p.add_argument("--register-loss-weight", type=float, default=0.5)
    p.add_argument(
        "--register-head-only",
        action="store_true",
        help="Freeze the base model and fine-tune only the register head.",
    )

    # Retrieval augmentation (v8 RAG-TFScope)
    p.add_argument("--use-retrieval", action="store_true",
                   help="Enable retrieval-augmented PWM cross-attention")
    p.add_argument("--residual-prior", action="store_true",
                   help="v12: output = log(prior) + alpha*delta instead of de-novo+log-prior")
    p.add_argument("--retrieval-k", type=int, default=3,
                   help="Number of nearest neighbours to retrieve per sample")
    p.add_argument("--retrieval-dropout", type=float, default=0.15,
                   help="Classifier-free guidance: prob of zeroing retrieval at train")
    p.add_argument("--aligned-trust-target", action="store_true")
    p.add_argument("--trust-rank-weight", type=float, default=0.0)
    p.add_argument("--trust-rank-margin", type=float, default=0.1)
    p.add_argument(
        "--positionwise-retrieval-gate",
        action="store_true",
        help="Gate retrieval independently at each PWM position using local donor support.",
    )
    p.add_argument(
        "--align-retrieved-pwms",
        action="store_true",
        help="Align retrieved PWMs to the de-novo prediction before fusion.",
    )
    p.add_argument("--retrieval-alignment-max-shift", type=int, default=10)
    p.add_argument("--retrieval-alignment-min-overlap", type=int, default=4)
    p.add_argument(
        "--retrieval-reranker-only",
        action="store_true",
        help="Fine-tune only donor trust and motif-wide retrieval gate parameters.",
    )
    p.add_argument("--retrieval-index-path", default="data/processed/tf_nn_index.json",
                   help="Path to JSON NN index produced by build_nn_index.py")
    # Robust-RAG training augmentations (v17)
    p.add_argument("--full-retrieval-dropout", type=float, default=0.0)
    p.add_argument("--neighbor-dropout",       type=float, default=0.0)
    p.add_argument("--hard-negative-rate",     type=float, default=0.0)
    p.add_argument("--hard-negative-per-sample", type=int, default=1)
    p.add_argument("--all-bad-case-rate",      type=float, default=0.0)

    # v18 contact-aware head
    p.add_argument("--pwm-head-v18", action="store_true",
                   help="Use the v18 contact-aware residual PWM head")
    p.add_argument("--v18-freeze-prior", action="store_true",
                   help="Train ONLY the v18 contact branch; freeze the prior head + encoder")
    p.add_argument("--v18-row-div-weight", type=float, default=0.05)
    p.add_argument("--v18-hub-weight",     type=float, default=0.05)
    p.add_argument("--v18-delta-scale-init", type=float, default=0.1)
    p.add_argument("--v18-contact-supervision", action="store_true",
                   help="v18b: supervise attention onto family-canonical recognition residues")
    p.add_argument("--v18-contact-weight", type=float, default=0.3)
    p.add_argument("--v18-contact-bias-scale", type=float, default=0.0)
    p.add_argument("--v18-contact-code", action="store_true",
                   help="v18b: family/aa contact-code MLP for Δz values")
    p.add_argument("--dual-family", action="store_true",
                   help="fuse learned-id + semantic family (gated by homology), deep-injected into v18 head")
    p.add_argument("--dual-family-dim", type=int, default=None,
                   help="dim of the fused family conditioning vector (default 64)")
    p.add_argument("--dual-family-semantic-path", default=None,
                   help="semantic family vectors for the dual head (e.g. family_embeddings_10.pt)")
    p.add_argument("--recognition-prior-path",
                   default="data/contact_maps/recognition_residues.json")

    # W&B
    p.add_argument("--wandb-project", default="TFScope",
                   help="W&B project name")
    p.add_argument("--wandb-name", default=None,
                   help="W&B run name (defaults to basename of --out)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging")

    return p.parse_args()


def init_wandb(args, config):
    import wandb
    run_name = args.wandb_name or os.path.basename(args.out.rstrip("/"))
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            # model
            "esm_model":          config.esm_model,
            "esm_embed_dim":      config.esm_embed_dim,
            "freeze_encoder":     config.freeze_encoder,
            "num_experts":        config.num_experts,
            "top_k":              config.top_k,
            "proj_hidden_dim":    config.proj_hidden_dim,
            "max_motif_length":   config.max_motif_length,
            "gate_mode":          config.gate_mode,
            "pwm_cov_r_weight":   config.pwm_cov_r_weight,
            # training
            "epochs":             args.epochs,
            "batch_size":         args.batch_size,
            "grad_accum_steps":    args.grad_accum_steps,
            "effective_batch_size": config.effective_batch_size,
            "world_size":          getattr(config, "world_size", 1),
            "lr":                 args.lr,
            "lora_lr":            args.lora_lr,
            "warmup_steps":       args.warmup_steps,
            # split
            "split":              args.split,
            "dummy":              args.dummy,
            "gene_balanced_sampling": args.gene_balanced_sampling,
            "group_balanced_sampling": args.group_balanced_sampling,
            "oracle_aggregation": args.oracle_aggregation,
            "latent_registration": args.latent_registration,
            "registration_anchor_path": args.registration_anchor_path,
        },
        resume="allow",
        dir=args.out,
    )
    return run


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def checkpoint_model_state(model):
    """Save trainable adapters while omitting the frozen 650M ESM weights."""
    state = {}
    for key, value in model.state_dict().items():
        if not key.startswith("backbone._esm_model") or ".lora_" in key:
            state[key] = value
    return state


def assert_lora_loaded(model, missing_keys, checkpoint_path):
    expected = {
        key
        for key in model.state_dict()
        if key.startswith("backbone._esm_model") and ".lora_" in key
    }
    missing_lora = sorted(expected.intersection(missing_keys))
    if missing_lora:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing "
            f"{len(missing_lora)} trained LoRA tensors"
        )


def reduce_epoch_metrics(metrics, device, distributed):
    if not distributed:
        return metrics
    values = torch.tensor(
        [metrics["loss"], metrics["gate_loss"], metrics["pwm_loss"]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return dict(zip(("loss", "gate_loss", "pwm_loss"), values.tolist()))


def make_loaders(args, config, rank=0, world_size=1):
    if args.dummy:
        train_ds = SyntheticTFDataset(config, n_samples=512, seed=args.seed)
        val_ds   = SyntheticTFDataset(config, n_samples=64,  seed=args.seed + 1)
        train_sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if world_size > 1
            else None
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=train_sampler is None,
                                  sampler=train_sampler,
                                  num_workers=0, collate_fn=collate_variable_length)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=collate_variable_length)
        return train_loader, val_loader

    train_ds = TFDataset(config, args.data, args.split, split="train",
                         max_seq_len=args.max_seq_len)
    val_ds   = TFDataset(config, args.data, args.split, split="val",
                         max_seq_len=args.max_seq_len)
    train_sampler = None
    if args.gene_balanced_sampling and args.group_balanced_sampling:
        raise ValueError(
            "--gene-balanced-sampling and --group-balanced-sampling are mutually exclusive"
        )
    if args.gene_balanced_sampling:
        train_sampler = GeneBalancedSampler(
            train_ds.df["gene_symbol"].tolist(),
            num_samples=len(train_ds),
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
    elif args.group_balanced_sampling:
        train_sampler = GeneBalancedSampler(
            train_ds.group_ids,
            num_samples=len(train_ds),
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
    elif world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_variable_length)
    return train_loader, val_loader


def autocast_context(device, precision):
    device_type = device.type if isinstance(device, torch.device) else device
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device_type == "cuda" and precision == "bf16",
    )


def run_train_epoch(model, loss_fn, loader, optimizer, scheduler,
                    device, global_step, wandb_run, grad_accum_steps=1,
                    precision="fp32", distributed=False):
    """One training epoch. Logs per-step loss to W&B."""
    model.train()
    total_loss = gate_loss = pwm_loss = 0.0
    n_batches = 0
    accumulated_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}

        is_last_batch = batch_idx + 1 == len(loader)
        should_step = accumulated_batches + 1 == grad_accum_steps or is_last_batch
        sync_context = (
            model.no_sync()
            if distributed and not should_step
            else contextlib.nullcontext()
        )
        with sync_context, autocast_context(device, precision):
            gate_logits, pwm_logits, aux = model(
                batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
                retrieved_pwms=batch.get('retrieved_pwms'),
                retrieved_masks=batch.get('retrieved_masks'),
                retrieved_sims=batch.get('retrieved_sims'),
                recog_prior=batch.get('recog_prior'),
            )
            loss, metrics = loss_fn(
                aux.get("internal_gate_logits", gate_logits),
                aux.get("internal_pwm_logits", pwm_logits),
                batch['target_pwm'].float(), batch['pwm_mask'].float(),
                aux.get('gate_logits'), aux.get('top_indices'), aux.get('family_id'),
                trust_logits=aux.get('trust_logits'),
                retrieved_pwms=aux.get('retrieved_pwms'),
                retrieved_masks=aux.get('retrieved_masks'),
                attn=aux.get('attn'),
                attn_key_mask=aux.get('attn_key_mask'),
                recog_prior=aux.get('recog_prior'),
                contact_target=batch.get('contact_target'),
                contact_base_mask=batch.get('contact_base_mask'),
                registration_anchor_mask=batch.get('registration_anchor_mask'),
                registration_anchor_mode=batch.get('registration_anchor_mode'),
                registration_orientation=batch.get('registration_orientation'),
                registration_offset=batch.get('registration_offset'),
                register_logits=aux.get('register_logits'),
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                (loss / grad_accum_steps).backward()

        # ── Robust-RAG diagnostics (Feature 6) ─────────────────────────────
        rag_diag = {}
        if aux.get('trust_logits') is not None:
            with torch.no_grad():
                ts = torch.sigmoid(aux['trust_logits'])                 # (B,K)
                rag_diag['trust_mean']     = ts.mean().item()
                rag_diag['max_trust_mean'] = ts.max(dim=-1).values.mean().item()
            if aux.get('beta_gated') is not None:
                bg = aux['beta_gated']
                rag_diag['beta_gated_mean'] = bg.mean().item()
                rag_diag['beta_lt_0.05']    = (bg < 0.05).float().mean().item()
            if aux.get('retrieved_masks') is not None:
                rm = aux['retrieved_masks']                             # (B,K,L)
                nb_valid = (rm.sum(dim=-1) > 0).float()                 # (B,K)
                rag_diag['frac_neighbors_dropped'] = 1.0 - nb_valid.mean().item()
                rag_diag['frac_full_dropout']      = (nb_valid.sum(dim=-1) == 0).float().mean().item()

        if not (torch.isnan(loss) or torch.isinf(loss)):
            accumulated_batches += 1
            if should_step:
                if accumulated_batches < grad_accum_steps:
                    correction = grad_accum_steps / accumulated_batches
                    for parameter in list(model.parameters()) + list(loss_fn.parameters()):
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_fn.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_batches = 0
                global_step += 1

            if wandb_run is not None and should_step:
                wandb_run.log({
                    "train/loss":             loss.item(),
                    "train/gate_loss":        metrics['gate_loss'],
                    "train/pwm_loss":         metrics['pwm_loss'],
                    "train/pwm_kl":           metrics.get('pwm_kl', 0),
                    "train/pwm_l1":           metrics.get('pwm_l1', 0),
                    "train/pwm_ic":           metrics.get('pwm_ic', 0),
                    "train/pwm_entropy":      metrics.get('pwm_entropy', 0),
                    "train/pwm_cov_r":        metrics.get('pwm_cov_r', 0),
                    "train/ordinal_violation":metrics.get('ordinal_violation', 0),
                    "train/length_mae":       metrics.get('length_mae', 0),
                    "train/length_bias":      metrics.get('length_bias', 0),
                    "train/length_loss":      metrics.get('length_loss', 0),
                    "train/registration_loss": metrics.get('registration_loss', 0),
                    "train/registration_entropy": metrics.get('registration_entropy', 0),
                    "train/registration_coverage": metrics.get('registration_coverage', 0),
                    "train/registration_anchor_fraction": metrics.get(
                        'registration_anchor_fraction', 0
                    ),
                    "train/register_supervision": metrics.get(
                        'register_supervision', 0
                    ),
                    "train/register_accuracy": metrics.get(
                        'register_accuracy', 0
                    ),
                    "train/balance_loss":     metrics.get('balance_loss', 0),
                    "train/diversity_loss":   metrics.get('diversity_loss', 0),
                    "train/sigma_gate":       metrics.get('sigma_gate', 1),
                    "train/sigma_pwm":        metrics.get('sigma_pwm', 1),
                    "lr":  scheduler.get_last_lr()[0],
                    "step": global_step,
                    **{f"rag/{k}": v for k, v in rag_diag.items()},
                }, step=global_step)

        total_loss += loss.item()
        gate_loss  += metrics['gate_loss']
        pwm_loss   += metrics['pwm_loss']
        n_batches  += 1

    n = max(n_batches, 1)
    epoch_metrics = {'loss': total_loss / n, 'gate_loss': gate_loss / n,
                     'pwm_loss': pwm_loss / n}
    return reduce_epoch_metrics(epoch_metrics, device, distributed), global_step


@torch.no_grad()
def run_val_epoch(model, loss_fn, loader, device, precision="fp32"):
    model.eval()
    total_loss = gate_loss = pwm_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        with autocast_context(device, precision):
            gate_logits, pwm_logits, aux = model(
                batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
                retrieved_pwms=batch.get('retrieved_pwms'),
                retrieved_masks=batch.get('retrieved_masks'),
                retrieved_sims=batch.get('retrieved_sims'),
                recog_prior=batch.get('recog_prior'),
            )
            loss, metrics = loss_fn(
                aux.get("internal_gate_logits", gate_logits),
                aux.get("internal_pwm_logits", pwm_logits),
                batch['target_pwm'].float(), batch['pwm_mask'].float(),
                aux.get('gate_logits'), aux.get('top_indices'), aux.get('family_id'),
                trust_logits=aux.get('trust_logits'),
                retrieved_pwms=aux.get('retrieved_pwms'),
                retrieved_masks=aux.get('retrieved_masks'),
                attn=aux.get('attn'),
                attn_key_mask=aux.get('attn_key_mask'),
                recog_prior=aux.get('recog_prior'),
                contact_target=batch.get('contact_target'),
                contact_base_mask=batch.get('contact_base_mask'),
                registration_anchor_mask=batch.get('registration_anchor_mask'),
                registration_anchor_mode=batch.get('registration_anchor_mode'),
                registration_orientation=batch.get('registration_orientation'),
                registration_offset=batch.get('registration_offset'),
                register_logits=aux.get('register_logits'),
            )
        total_loss += loss.item()
        gate_loss  += metrics['gate_loss']
        pwm_loss   += metrics['pwm_loss']
        n_batches  += 1

    n = max(n_batches, 1)
    return {'loss': total_loss / n, 'gate_loss': gate_loss / n,
            'pwm_loss': pwm_loss / n}


@torch.no_grad()
def run_oracle_r_eval(model, val_loader, device, n_tfs=0,
                      ic_thresh=0.25, max_shift=10, precision="fp32",
                      coverage_aware=True, aggregation="gene"):
    """Oracle-aligned Pearson r on up to n_tfs val TFs.

    For each TF: extract gate-predicted active columns, align vs the trimmed
    informative core of the target (offset + RC freedom), record per-col r.

    coverage_aware (default True): scale the overlap r by coverage
    (n_overlap / L_target_core), matching eval_full_metrics.panel_full's r_cov.
    Without it the gate-based r has a length blind spot -- a short gate is
    scored on fewer, easier columns and gets an inflated r, so a plain-r
    selector rewards gate collapse. The coverage-scaled r removes that, so
    ckpt_best is chosen on the same honest quantity we report at eval.
    """
    def _ic(pwm):
        p = np.clip(pwm, 1e-8, 1.0)
        return 2.0 + (p * np.log2(p)).sum(0)

    def _trim_core(pwm, thresh):
        ic = _ic(pwm)
        inf = np.where(ic >= thresh)[0]
        if len(inf) == 0:
            return pwm
        return pwm[:, inf[0]:inf[-1] + 1]

    model.eval()
    dataset_size = len(val_loader.dataset)
    limit = dataset_size if n_tfs is None or n_tfs <= 0 else min(n_tfs, dataset_size)
    dataset_df = getattr(val_loader.dataset, "df", None)
    if dataset_df is not None and "gene_symbol" in dataset_df:
        all_genes = dataset_df["gene_symbol"].fillna("").astype(str).str.upper().tolist()
    else:
        all_genes = [f"ROW_{i}" for i in range(dataset_size)]
    preds, targets, masks, gates_list, genes = [], [], [], [], []
    collected = 0
    for batch in val_loader:
        if collected >= limit:
            break
        batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        with autocast_context(device, precision):
            gate_logits, pwm_logits, _ = model(
                batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
                retrieved_pwms=batch.get('retrieved_pwms'),
                retrieved_masks=batch.get('retrieved_masks'),
                retrieved_sims=batch.get('retrieved_sims'),
                recog_prior=batch.get('recog_prior'),
            )
        pwm_prob = (
            F.softmax(pwm_logits, dim=1).float().cpu().numpy()
        )                                                          # (B, 4, L)
        gate_prob = torch.sigmoid(gate_logits).float().cpu().numpy()  # (B, L)
        target    = batch['target_pwm'].cpu().numpy()             # (B, 4, L)
        mask      = batch['pwm_mask'].cpu().numpy()               # (B, L)
        take = min(pwm_prob.shape[0], limit - collected)
        preds.extend(pwm_prob[:take])
        targets.extend(target[:take])
        masks.extend(mask[:take])
        gates_list.extend(gate_prob[:take])
        genes.extend(all_genes[collected:collected + take])
        collected += take

    scored = []
    for pred, tgt, msk, gate, gene in zip(
        preds, targets, masks, gates_list, genes
    ):
        active = gate > 0.5
        if not active.any():
            active = gate > gate.max() * 0.5
        pred_core = pred[:, active]
        if pred_core.shape[1] == 0:
            continue
        tgt_valid = tgt[:, msk.astype(bool)]
        if tgt_valid.shape[1] == 0:
            continue
        tgt_core = _trim_core(tgt_valid, ic_thresh)
        if tgt_core.shape[1] == 0:
            continue
        _, shift, _, r = align_pwm(pred_core, tgt_core, max_shift=max_shift,
                                    consider_revcomp=True)
        if coverage_aware:
            # coverage = fraction of the target core covered by the aligned
            # prediction (neighbour col i -> reference col i+shift). Scaling r
            # by this is exactly panel_full's r_cov: a short/collapsed gate
            # that only overlaps a few easy columns is penalised for the GT
            # columns it leaves uncovered, so it can no longer win selection.
            wp, lr = pred_core.shape[1], tgt_core.shape[1]
            n_overlap = sum(1 for i in range(wp) if 0 <= i + shift < lr)
            cov = n_overlap / lr if lr > 0 else 0.0
            scored.append((gene, r * cov))
        else:
            scored.append((gene, r))

    if not scored:
        return 0.0
    if aggregation == "row":
        return float(np.mean([score for _, score in scored]))
    by_gene = {}
    for gene, score in scored:
        by_gene.setdefault(gene, []).append(score)
    return float(np.mean([np.mean(values) for values in by_gene.values()]))


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1")
    if args.registration_anchor_path and not args.latent_registration:
        raise ValueError(
            "--registration-anchor-path requires --latent-registration"
        )
    if args.registration_max_shift < 0:
        raise ValueError("--registration-max-shift must be non-negative")
    if args.registration_min_overlap < 1:
        raise ValueError("--registration-min-overlap must be positive")
    if args.registration_temperature <= 0:
        raise ValueError("--registration-temperature must be positive")
    if args.registration_coverage_penalty < 0:
        raise ValueError("--registration-coverage-penalty must be non-negative")
    if args.gate_length_weight < 0:
        raise ValueError("--gate-length-weight must be non-negative")
    if args.pwm_cov_r_weight < 0:
        raise ValueError("--pwm-cov-r-weight must be non-negative")
    if args.max_motif_length < 4:
        raise ValueError("--max-motif-length must be at least 4")
    if args.register_head and not args.latent_registration:
        raise ValueError("--register-head requires --latent-registration")
    if args.register_loss_weight < 0:
        raise ValueError("--register-loss-weight must be non-negative")
    rank, local_rank, world_size = distributed_context()
    distributed = world_size > 1
    is_main = rank == 0
    if distributed:
        if not torch.cuda.is_available():
            raise ValueError("DDP training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if is_main:
        print(
            f"Device: {device}  |  rank: {rank}/{world_size}  |  "
            f"distributed: {distributed}"
        )
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("--precision bf16 requires CUDA")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    if is_main:
        print(f"Precision: {args.precision}  |  TF32: {args.tf32}")

    if is_main:
        os.makedirs(args.out, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    # Config
    config = TFScopeConfig(
        freeze_encoder=args.freeze_encoder,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_n_layers=args.lora_n_layers,
        learning_rate=args.lr,
        lora_learning_rate=args.lora_lr,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        use_retrieval=args.use_retrieval,
        residual_prior=args.residual_prior,
        retrieval_k=args.retrieval_k,
        retrieval_dropout=args.retrieval_dropout,
        aligned_trust_target=args.aligned_trust_target,
        trust_rank_loss_weight=args.trust_rank_weight,
        trust_rank_margin=args.trust_rank_margin,
        positionwise_retrieval_gate=args.positionwise_retrieval_gate,
        align_retrieved_pwms=args.align_retrieved_pwms,
        retrieval_alignment_max_shift=args.retrieval_alignment_max_shift,
        retrieval_alignment_min_overlap=args.retrieval_alignment_min_overlap,
        retrieval_index_path=args.retrieval_index_path,
        pwm_ic_pcc_weight=args.ic_pcc_weight,
        pwm_topbase_weight=args.topbase_weight,
        pwm_topbase_margin=args.topbase_margin,
        pwm_contrastive_weight=args.contrastive_weight,
        pwm_contrastive_tau=args.contrastive_tau,
        latent_registration=args.latent_registration,
        registration_max_shift=args.registration_max_shift,
        registration_min_overlap=args.registration_min_overlap,
        registration_temperature=args.registration_temperature,
        registration_coverage_penalty=args.registration_coverage_penalty,
        gate_length_weight=args.gate_length_weight,
        gate_mode=args.gate_mode,
        max_motif_length=args.max_motif_length,
        motif_overflow_policy=args.motif_overflow_policy,
        pwm_cov_r_weight=args.pwm_cov_r_weight,
        pwm_core_ic_thresh=args.pwm_core_ic_thresh,
        two_chain_input=args.two_chain_input,
        require_multichain_eligible=not args.allow_unannotated_multichain,
        chain_id_embedding=args.chain_id_embedding,
        max_chains=args.max_chains,
        registration_anchor_path=args.registration_anchor_path,
        register_head=args.register_head,
        register_loss_weight=args.register_loss_weight,
        full_retrieval_dropout=args.full_retrieval_dropout,
        neighbor_dropout=args.neighbor_dropout,
        hard_negative_rate=args.hard_negative_rate,
        hard_negative_per_sample=args.hard_negative_per_sample,
        all_bad_case_rate=args.all_bad_case_rate,
        pwm_head_v18=args.pwm_head_v18,
        v18_freeze_prior=args.v18_freeze_prior,
        v18_row_div_weight=args.v18_row_div_weight,
        v18_hub_weight=args.v18_hub_weight,
        v18_delta_scale_init=args.v18_delta_scale_init,
        v18_contact_supervision=args.v18_contact_supervision,
        v18_contact_weight=args.v18_contact_weight,
        v18_contact_bias_scale=args.v18_contact_bias_scale,
        v18_contact_code=args.v18_contact_code,
        recognition_prior_path=args.recognition_prior_path,
    )
    if args.contact_pred_head:
        config.contact_pred_head = True
    if args.contact_probe_path is not None:
        config.contact_probe_path = args.contact_probe_path
    if args.v18_contact_bias_learnable is not None:
        config.v18_contact_bias_learnable = bool(args.v18_contact_bias_learnable)
    config.grad_accum_steps = args.grad_accum_steps
    config.world_size = world_size
    config.effective_batch_size = (
        args.batch_size * args.grad_accum_steps * world_size
    )
    config.precision = args.precision
    config.tf32 = args.tf32

    # MoE / family taxonomy overrides (rebin run): only applied when explicitly given.
    config.use_moe = bool(getattr(args, "use_moe", True))  # --no-moe -> False (B5)
    if getattr(args, "mean_pool", False):
        config.pool_type = "mean"                          # B2 baseline pooling
    if args.num_families is not None:      config.num_families = args.num_families
    if args.num_experts is not None:       config.num_experts = args.num_experts
    if args.diversity_loss_weight is not None: config.diversity_loss_weight = args.diversity_loss_weight
    if args.balance_loss_weight is not None:   config.balance_loss_weight = args.balance_loss_weight
    if args.n_shared_experts is not None:      config.n_shared_experts = args.n_shared_experts
    if args.moe_residual is not None:          config.moe_residual = bool(args.moe_residual)
    if args.route_supervision_weight is not None: config.route_supervision_weight = args.route_supervision_weight
    if args.top_k is not None:             config.top_k = args.top_k
    if args.moe_granularity is not None:   config.moe_granularity = args.moe_granularity
    if args.expert_hidden_dim is not None: config.expert_hidden_dim = args.expert_hidden_dim
    if args.family_embedding_path is not None:
        # 'none' -> force LearnedFamilyEmbedding (precomputed semantic file is fixed at 10 families)
        config.family_embedding_path = (
            "" if args.family_embedding_path.lower() == "none" else args.family_embedding_path)
    if args.v18_attn_sparse is not None:       config.v18_attn_sparse = args.v18_attn_sparse
    if args.v18_attn_alpha_init is not None:   config.v18_attn_alpha_init = args.v18_attn_alpha_init
    config.use_dual_family = bool(args.dual_family)
    if args.dual_family_dim is not None:           config.dual_family_dim = args.dual_family_dim
    if args.dual_family_semantic_path is not None: config.dual_family_semantic_path = args.dual_family_semantic_path
    if config.use_dual_family and is_main:
        print(f"dual-family fusion ON (dim={config.dual_family_dim}, "
              f"semantic_path={config.dual_family_semantic_path or '(MoE path)'})")
    if args.contact_distill_weight is not None: config.contact_distill_weight = args.contact_distill_weight
    if args.contact_targets_path is not None:   config.contact_targets_path = args.contact_targets_path
    if getattr(config, "contact_distill_weight", 0.0) > 0:
        print(f"contact distillation: weight={config.contact_distill_weight} "
              f"targets={getattr(config,'contact_targets_path','data/contact_maps/contact_targets.json')}")
    if is_main:
        print(f"v18 attention normalizer: {config.v18_attn_sparse}"
              + (f" (alpha_init={config.v18_attn_alpha_init})" if config.v18_attn_sparse == "entmax_learn" else ""))
        print(f"MoE: num_families={config.num_families} num_experts={config.num_experts} "
              f"top_k={config.top_k} expert_hidden={config.expert_hidden_dim} "
              f"family_embed={'learned' if not config.family_embedding_path else config.family_embedding_path}")
        if config.latent_registration:
            anchor_description = (
                config.registration_anchor_path
                if config.registration_anchor_path
                else "none (fully latent)"
            )
            print(
                "Latent registration: "
                f"shift=±{config.registration_max_shift} "
                f"min_overlap={config.registration_min_overlap} "
                f"temperature={config.registration_temperature} "
                f"coverage_penalty={config.registration_coverage_penalty} "
                f"anchors={anchor_description}"
            )
        if config.register_head:
            print(
                f"Register head: {2 * (2 * config.registration_max_shift + 1)} "
                f"states | supervised weight={config.register_loss_weight}"
            )

    # Data
    train_loader, val_loader = make_loaders(args, config, rank, world_size)
    # Benchmark loader (contamination-controlled held-out test, e.g. cluster40) for
    # checkpoint selection on the hard test-like metric — built on the main rank only.
    bench_loader = None
    if getattr(args, "benchmark_eval", False) and args.benchmark_data and is_main and not args.dummy:
        bench_ds = TFDataset(config, args.benchmark_data,
                             args.benchmark_split or args.split, split="test",
                             max_seq_len=args.max_seq_len)
        bench_loader = DataLoader(bench_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=collate_variable_length)
        print(f"benchmark eval: {len(bench_ds)} TFs from "
              f"{os.path.basename(args.benchmark_data)} (test split)")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    config.total_steps = total_steps
    effective_batch_size = args.batch_size * args.grad_accum_steps * world_size
    if is_main:
        print(
            f"Train micro-batches/rank/epoch: {len(train_loader)}  |  "
            f"Optimizer steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}"
        )
        print(
            f"Micro-batch/rank: {args.batch_size}  |  "
            f"Gradient accumulation: {args.grad_accum_steps}  |  "
            f"World size: {world_size}  |  Global effective batch: {effective_batch_size}"
        )

    # Model & loss
    model   = TFScopeModel(config, use_dummy_backbone=args.dummy).to(device)
    loss_fn = TFScopeLoss(config).to(device)
    if config.lora_rank > 0 and not args.dummy:
        model.backbone.build(device)   # eagerly load ESM so LoRA params exist before optimizer

    if args.init_model:
        checkpoint = torch.load(
            args.init_model, map_location=device, weights_only=False
        )
        missing, unexpected = model.load_state_dict(
            checkpoint["model"], strict=False
        )
        assert_lora_loaded(model, missing, args.init_model)
        if is_main:
            print(
                f"Initialised exact model weights from {args.init_model} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    # Stage-A → Stage-B: warm-start the protein encoder from contrastive pretraining.
    if getattr(args, "init_from_pretrain", None):
        pre = torch.load(args.init_from_pretrain, map_location=device, weights_only=False)
        enc = pre.get("encoder", pre)
        res = model.load_state_dict(enc, strict=False)
        loaded = len(enc) - len([k for k in enc if k in res.unexpected_keys])
        print(f"Warm-started {loaded}/{len(enc)} encoder tensors from "
              f"{args.init_from_pretrain} (epoch {pre.get('epoch','?')}, "
              f"loss {pre.get('loss','?')})")

    # v18a: optionally freeze everything except the new contact-aware branch.
    if args.pwm_head_v18 and args.v18_freeze_prior:
        trainable_prefixes = (
            "pwm_head.q_pos", "pwm_head.q_build", "pwm_head.contact_attn",
            "pwm_head.contact_out", "pwm_head.code_mlp", "pwm_head.fam_embed",
            "pwm_head.log_lambda",
        )
        n_tr = 0
        for name, p in model.named_parameters():
            keep = any(name.startswith(pre) for pre in trainable_prefixes)
            p.requires_grad = keep
            n_tr += p.numel() if keep else 0
        print(f"v18 freeze-prior: training contact branch only ({n_tr/1e6:.2f}M params)")

    if args.register_head_only:
        if not config.register_head:
            raise ValueError("--register-head-only requires --register-head")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("register_head.")
        if is_main:
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            print(f"Register-head-only fine-tuning: {trainable:,} parameters")

    if args.retrieval_reranker_only:
        if not config.use_retrieval:
            raise ValueError(
                "--retrieval-reranker-only requires --use-retrieval"
            )
        gate_suffixes = (
            "retrieval_beta",
            "conf_scale",
            "conf_thresh",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith("trust_predictor.")
                or name.endswith(gate_suffixes)
            )
        if is_main:
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            print(f"Retrieval-reranker-only fine-tuning: {trainable:,} parameters")

    # ── model size summary ────────────────────────────────────────────────────
    def _count(module):
        total     = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _fmt(n):
        return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K"

    rows = [
        ("backbone (ESM-2)",      model.backbone),
        ("global_pool",           model.global_pool),
        ("dbd_pool",              model.dbd_pool),
        ("projection",            model.projection),
    ]
    if not getattr(model, "use_moe", True):
        pass  # MoE bypassed (B5): no MoE submodules to report
    elif getattr(model, "moe_granularity", "protein") == "residue":
        rows += [
            ("residue_moe.experts",   model.residue_moe.experts),
            ("residue_moe.shared",    model.residue_moe.shared_experts),
            ("residue_moe.router",    model.residue_moe.router),
        ]
    else:
        rows += [
            ("moe.family_embed",      model.moe.family_embed),
            ("moe.gating",            model.moe.gating),
            ("moe.experts (×12)",     model.moe.experts),
            ("moe.film",              model.moe.film),
        ]
    rows += [
        ("gate_head",             model.gate_head),
        ("pwm_head",              model.pwm_head),
    ]
    if getattr(model, "use_register_head", False):
        rows.append(("register_head", model.register_head))
    if is_main:
        print("\n── Model size ───────────────────────────────────────────")
        print(f"  {'Module':<26} {'Total':>8}  {'Trainable':>10}")
        print(f"  {'-'*26} {'-'*8}  {'-'*10}")
        for name, mod in rows:
            t, tr = _count(mod)
            print(f"  {name:<26} {_fmt(t):>8}  {_fmt(tr):>10}")
        total_t, total_tr = _count(model)
        loss_t, loss_tr   = _count(loss_fn)
        print(f"  {'─'*48}")
        print(f"  {'TOTAL (model)':<26} {_fmt(total_t):>8}  {_fmt(total_tr):>10}")
        print(f"  {'loss_fn (σ params)':<26} {_fmt(loss_t):>8}  {_fmt(loss_tr):>10}")
        print(f"──────────────────────────────────────────────────────────\n")

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # LoRA params live inside backbone but need a higher lr than frozen base weights
    raw_model = unwrap(model)
    raw_loss_fn = unwrap(loss_fn)
    # DDP has already broadcast rank-0 parameters; use rank-specific RNG streams
    # for dropout and other stochastic training operations.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    lora_params  = [p for p in raw_model.backbone.parameters() if p.requires_grad]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and not any(p is lp for lp in lora_params)]
    param_groups = [{'params': other_params, 'lr': args.lr}]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr})
        if is_main:
            print(f"LoRA params: {sum(p.numel() for p in lora_params):,}  (lr={args.lora_lr})")
    optimizer = torch.optim.AdamW(
        param_groups + [{'params': loss_fn.parameters(), 'lr': args.lr}],
        weight_decay=0.01, betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: get_cosine_lr(s, config.warmup_steps, total_steps))

    start_epoch  = 0
    global_step  = 0
    best_val_loss  = float('inf')
    best_oracle_r  = -float('inf')
    best_bench     = -float('inf')
    patience_counter = 0

    # Stage-2 finetune init: load ONLY model weights, keep fresh optimizer/schedule
    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt['model']
        if args.pwm_head_v18:
            # v14's entire pwm_head IS the v18 prior branch → remap keys.
            sd = {(k.replace("pwm_head.", "pwm_head.prior_head.", 1)
                   if k.startswith("pwm_head.") else k): v
                  for k, v in sd.items()}
        missing, unexpected = raw_model.load_state_dict(sd, strict=False)
        n_prior = sum(1 for m in missing if m.startswith("pwm_head.prior_head."))
        if is_main:
            print(f"Initialised model weights from {args.init_from} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)}, "
                  f"prior_head missing={n_prior}); fresh optimizer/schedule/epoch")

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, _ = raw_model.load_state_dict(ckpt['model'], strict=False)
        assert_lora_loaded(raw_model, missing, args.resume)
        raw_loss_fn.load_state_dict(ckpt['loss_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch    = ckpt['epoch'] + 1
        global_step    = ckpt.get('global_step', start_epoch * steps_per_epoch)
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        best_oracle_r  = ckpt.get('best_oracle_r', -float('inf'))
        best_bench     = ckpt.get('best_bench', -float('inf'))
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}")

    # Save config
    if is_main:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    # W&B
    wandb_run = None
    if is_main and not args.no_wandb:
        try:
            wandb_run = init_wandb(args, config)
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed ({e}) — continuing without logging")

    use_oracle = args.eval_oracle_r and not args.dummy
    oracle_label = "Oracle-r" if args.legacy_oracle_r else "CovR"
    oracle_hdr = f"  {oracle_label:>9}" if use_oracle else ""
    if is_main:
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
              f"{'L_gate':>8}  {'L_pwm':>8}{oracle_hdr}  {'Time':>6}")
        print("-" * (60 + (11 if use_oracle else 0)))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_m, global_step = run_train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, global_step, wandb_run, args.grad_accum_steps,
            args.precision, distributed)

        if distributed:
            dist.barrier(device_ids=[local_rank])
        if is_main:
            val_m = run_val_epoch(
                raw_model, raw_loss_fn, val_loader, device, args.precision
            )
        else:
            val_m = {"loss": 0.0, "gate_loss": 0.0, "pwm_loss": 0.0}

        # Oracle-r eval (every --oracle-r-every epochs or final epoch)
        oracle_r = None
        is_oracle_epoch = (use_oracle and (
            (epoch + 1) % args.oracle_r_every == 0 or epoch + 1 == args.epochs))
        if is_oracle_epoch and is_main:
            oracle_r = run_oracle_r_eval(
                raw_model, val_loader, device,
                n_tfs=args.oracle_r_n_tfs,
                precision=args.precision,
                coverage_aware=not args.legacy_oracle_r,
                aggregation=args.oracle_aggregation,
                ic_thresh=args.pwm_core_ic_thresh)

        # Benchmark eval on the contamination-controlled held-out set (test-like
        # metric; the proper selector vs the lenient val-oracle-r on easy paralogs)
        bench_r = None
        bench_every = args.benchmark_every or args.oracle_r_every
        if bench_loader is not None and is_main and (
                (epoch + 1) % bench_every == 0 or epoch + 1 == args.epochs):
            bench_r = run_oracle_r_eval(raw_model, bench_loader, device,
                                        n_tfs=100000, precision=args.precision,
                                        coverage_aware=not args.legacy_oracle_r,
                                        aggregation=args.oracle_aggregation,
                                        ic_thresh=args.pwm_core_ic_thresh)
        if distributed:
            payload = torch.tensor(
                [
                    val_m["loss"],
                    val_m["gate_loss"],
                    val_m["pwm_loss"],
                    oracle_r if oracle_r is not None else float("nan"),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(payload, src=0)
            val_m = dict(zip(
                ("loss", "gate_loss", "pwm_loss"), payload[:3].tolist()
            ))
            oracle_r = (
                None if torch.isnan(payload[3]) else float(payload[3].item())
            )

        elapsed = time.time() - t0
        oracle_str = f"  {oracle_r:>9.4f}" if oracle_r is not None else (
                     f"  {'':>9}"           if use_oracle else "")
        if is_main:
            print(f"{epoch+1:>6}  {train_m['loss']:>10.4f}  {val_m['loss']:>10.4f}  "
                  f"{val_m['gate_loss']:>8.4f}  {val_m['pwm_loss']:>8.4f}"
                  f"{oracle_str}  {elapsed:>5.0f}s")
            if bench_r is not None:
                print(f"        benchmark oracle-r: {bench_r:.4f}"
                      f"{'  *BEST*' if bench_r > best_bench else ''}")

        # ── determine is_best ─────────────────────────────────────────────
        if use_oracle:
            if oracle_r is not None:
                is_best = oracle_r > best_oracle_r
                if is_best:
                    best_oracle_r = oracle_r
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                is_best = False   # skip patience on non-oracle epochs
        else:
            is_best = val_m['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_m['loss']
                patience_counter = 0
            else:
                patience_counter += 1

        # Log epoch-level validation metrics to W&B
        if is_main and wandb_run is not None:
            log_d = {
                "val/loss":      val_m['loss'],
                "val/gate_loss": val_m['gate_loss'],
                "val/pwm_loss":  val_m['pwm_loss'],
                "epoch":         epoch + 1,
            }
            if oracle_r is not None:
                log_d["val/oracle_r"] = oracle_r
            wandb_run.log(log_d, step=global_step)
            if is_best:
                if use_oracle:
                    wandb_run.summary["best_oracle_r"]    = best_oracle_r
                    wandb_run.summary["best_oracle_epoch"] = epoch + 1
                else:
                    wandb_run.summary["best_val_loss"]  = best_val_loss
                    wandb_run.summary["best_val_epoch"] = epoch + 1

        # Checkpoint — exclude frozen backbone (~50 MB vs ~2.8 GB).
        # Two independent triggers:
        #   (a) milestone save at exact multiples of --save-every  →  ckpt_epoch{N}.pt
        #   (b) new best metric  →  overwrites ckpt_best.pt only
        if getattr(args, "save_epochs", None):
            _save_set = {int(x) for x in str(args.save_epochs).split(",") if x.strip()}
            save_milestone = (epoch + 1) in _save_set
        else:
            save_milestone = (epoch + 1) % args.save_every == 0
        save_best      = is_best and not args.no_save_best
        save_bench     = (bench_r is not None and bench_r > best_bench)
        if save_bench:
            best_bench = bench_r
        if is_main and (save_milestone or save_best or save_bench):
            trainable_state = checkpoint_model_state(raw_model)
            ckpt = {
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         trainable_state,
                'loss_fn':       raw_loss_fn.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_oracle_r': best_oracle_r,
                'best_bench':    best_bench,
                'train_metrics': train_m,
                'val_metrics':   val_m,
            }
            if oracle_r is not None:
                ckpt['oracle_r'] = oracle_r
            if bench_r is not None:
                ckpt['bench_r'] = bench_r
            if save_milestone:
                path = os.path.join(args.out, f"ckpt_epoch{epoch+1:03d}.pt")
                torch.save(ckpt, path)
            if save_best:
                torch.save(ckpt, os.path.join(args.out, "ckpt_best.pt"))
                if use_oracle:
                    print(f"  *** New best oracle r: {best_oracle_r:.4f} ***")
                else:
                    print(f"  *** New best val loss: {best_val_loss:.4f} ***")
            if save_bench:
                torch.save(ckpt, os.path.join(args.out, "ckpt_best_bench.pt"))
                print(f"  *** New best BENCHMARK oracle-r: {best_bench:.4f} (ckpt_best_bench.pt) ***")

        # Early stopping (only checked on oracle-r epochs when --eval-oracle-r is set)
        check_patience = (not use_oracle) or is_oracle_epoch
        should_stop = (
            args.early_stop_patience > 0
            and check_patience
            and patience_counter >= args.early_stop_patience
        )
        if distributed:
            stop_tensor = torch.tensor(
                int(should_stop), dtype=torch.int32, device=device
            )
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            metric_str = (f"oracle r has not improved for {patience_counter} oracle-r epochs"
                          if use_oracle else
                          f"val loss has not improved for {patience_counter} epochs")
            if is_main:
                print(f"\nEarly stopping at epoch {epoch+1}: {metric_str} "
                      f"(patience={args.early_stop_patience}).")
                if use_oracle:
                    print(f"Best oracle r was {best_oracle_r:.4f}.")
                else:
                    print(f"Best val loss was {best_val_loss:.4f}.")
            break

    if is_main:
        if use_oracle:
            print(f"\nDone. Best oracle r: {best_oracle_r:.4f}")
        else:
            print(f"\nDone. Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {args.out}/")

    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
