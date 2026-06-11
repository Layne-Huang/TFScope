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
import json
import os
import sys
import time

# Must be set before torch.hub makes any network calls
_DEFAULT_CACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
os.environ.setdefault("TORCH_HOME", _DEFAULT_CACHE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from tfscope.train.trainer import get_cosine_lr, _expert_utilization


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
    p.add_argument("--top-k", type=int, default=None, help="MoE top-k routing")
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

    # Model
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")

    # Output
    p.add_argument("--out",
                   default="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/run",
                   help="Directory for checkpoints and logs")
    p.add_argument("--save-every", type=int, default=20,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--init-from", default=None,
                   help="Load ONLY model weights from this checkpoint (fresh optimizer/"
                        "schedule/epoch) — for stage-2 finetuning after pretraining")
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
    p.add_argument("--oracle-r-n-tfs", type=int, default=100,
                   help="Number of val TFs to use for oracle-r eval (default 100)")

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

    # Retrieval augmentation (v8 RAG-TFScope)
    p.add_argument("--use-retrieval", action="store_true",
                   help="Enable retrieval-augmented PWM cross-attention")
    p.add_argument("--residual-prior", action="store_true",
                   help="v12: output = log(prior) + alpha*delta instead of de-novo+log-prior")
    p.add_argument("--retrieval-k", type=int, default=3,
                   help="Number of nearest neighbours to retrieve per sample")
    p.add_argument("--retrieval-dropout", type=float, default=0.15,
                   help="Classifier-free guidance: prob of zeroing retrieval at train")
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
            # training
            "epochs":             args.epochs,
            "batch_size":         args.batch_size,
            "lr":                 args.lr,
            "lora_lr":            args.lora_lr,
            "warmup_steps":       args.warmup_steps,
            # split
            "split":              args.split,
            "dummy":              args.dummy,
            "gene_balanced_sampling": args.gene_balanced_sampling,
        },
        resume="allow",
        dir=args.out,
    )
    return run


def make_loaders(args, config):
    if args.dummy:
        train_ds = SyntheticTFDataset(config, n_samples=512, seed=args.seed)
        val_ds   = SyntheticTFDataset(config, n_samples=64,  seed=args.seed + 1)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_variable_length)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, collate_fn=collate_variable_length)
        return train_loader, val_loader

    train_ds = TFDataset(config, args.data, args.split, split="train",
                         max_seq_len=args.max_seq_len)
    val_ds   = TFDataset(config, args.data, args.split, split="val",
                         max_seq_len=args.max_seq_len)
    train_sampler = None
    if args.gene_balanced_sampling:
        train_sampler = GeneBalancedSampler(
            train_ds.df["gene_symbol"].tolist(),
            num_samples=len(train_ds),
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


def run_train_epoch(model, loss_fn, loader, optimizer, scheduler,
                    device, global_step, wandb_run):
    """One training epoch. Logs per-step loss to W&B."""
    model.train()
    total_loss = gate_loss = pwm_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}

        gate_logits, pwm_logits, aux = model(
            batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
            retrieved_pwms=batch.get('retrieved_pwms'),
            retrieved_masks=batch.get('retrieved_masks'),
            retrieved_sims=batch.get('retrieved_sims'),
            recog_prior=batch.get('recog_prior'),
        )
        loss, metrics = loss_fn(
            gate_logits, pwm_logits,
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
        )

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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss":             loss.item(),
                    "train/gate_loss":        metrics['gate_loss'],
                    "train/pwm_loss":         metrics['pwm_loss'],
                    "train/pwm_kl":           metrics.get('pwm_kl', 0),
                    "train/pwm_l1":           metrics.get('pwm_l1', 0),
                    "train/pwm_ic":           metrics.get('pwm_ic', 0),
                    "train/pwm_entropy":      metrics.get('pwm_entropy', 0),
                    "train/ordinal_violation":metrics.get('ordinal_violation', 0),
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
        global_step += 1

    n = max(n_batches, 1)
    return ({'loss': total_loss / n, 'gate_loss': gate_loss / n,
             'pwm_loss': pwm_loss / n},
            global_step)


@torch.no_grad()
def run_val_epoch(model, loss_fn, loader, device):
    model.eval()
    total_loss = gate_loss = pwm_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        gate_logits, pwm_logits, aux = model(
            batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
            retrieved_pwms=batch.get('retrieved_pwms'),
            retrieved_masks=batch.get('retrieved_masks'),
            retrieved_sims=batch.get('retrieved_sims'),
            recog_prior=batch.get('recog_prior'),
        )
        loss, metrics = loss_fn(
            gate_logits, pwm_logits,
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
        )
        total_loss += loss.item()
        gate_loss  += metrics['gate_loss']
        pwm_loss   += metrics['pwm_loss']
        n_batches  += 1

    n = max(n_batches, 1)
    return {'loss': total_loss / n, 'gate_loss': gate_loss / n,
            'pwm_loss': pwm_loss / n}


@torch.no_grad()
def run_oracle_r_eval(model, val_loader, device, n_tfs=100,
                      ic_thresh=0.25, max_shift=10):
    """Oracle-aligned Pearson r on up to n_tfs val TFs.

    For each TF: extract gate-predicted active columns, align vs the trimmed
    informative core of the target (offset + RC freedom), record per-col r.
    Returns mean r (higher = better motif content).
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
    preds, targets, masks, gates_list = [], [], [], []
    collected = 0
    for batch in val_loader:
        if collected >= n_tfs:
            break
        batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        gate_logits, pwm_logits, _ = model(
            batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
            retrieved_pwms=batch.get('retrieved_pwms'),
            retrieved_masks=batch.get('retrieved_masks'),
            retrieved_sims=batch.get('retrieved_sims'),
            recog_prior=batch.get('recog_prior'),
        )
        pwm_prob  = F.softmax(pwm_logits, dim=1).cpu().numpy()   # (B, 4, L)
        gate_prob = torch.sigmoid(gate_logits).cpu().numpy()      # (B, L)
        target    = batch['target_pwm'].cpu().numpy()             # (B, 4, L)
        mask      = batch['pwm_mask'].cpu().numpy()               # (B, L)
        take = min(pwm_prob.shape[0], n_tfs - collected)
        preds.extend(pwm_prob[:take])
        targets.extend(target[:take])
        masks.extend(mask[:take])
        gates_list.extend(gate_prob[:take])
        collected += take

    rs = []
    for pred, tgt, msk, gate in zip(preds, targets, masks, gates_list):
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
        _, _, _, r = align_pwm(pred_core, tgt_core, max_shift=max_shift,
                                consider_revcomp=True)
        rs.append(r)

    return float(np.mean(rs)) if rs else 0.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out, exist_ok=True)

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
        retrieval_index_path=args.retrieval_index_path,
        pwm_ic_pcc_weight=args.ic_pcc_weight,
        pwm_topbase_weight=args.topbase_weight,
        pwm_topbase_margin=args.topbase_margin,
        pwm_contrastive_weight=args.contrastive_weight,
        pwm_contrastive_tau=args.contrastive_tau,
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

    # MoE / family taxonomy overrides (rebin run): only applied when explicitly given.
    if args.num_families is not None:      config.num_families = args.num_families
    if args.num_experts is not None:       config.num_experts = args.num_experts
    if args.top_k is not None:             config.top_k = args.top_k
    if args.expert_hidden_dim is not None: config.expert_hidden_dim = args.expert_hidden_dim
    if args.family_embedding_path is not None:
        # 'none' -> force LearnedFamilyEmbedding (precomputed semantic file is fixed at 10 families)
        config.family_embedding_path = (
            "" if args.family_embedding_path.lower() == "none" else args.family_embedding_path)
    if args.v18_attn_sparse is not None:       config.v18_attn_sparse = args.v18_attn_sparse
    if args.v18_attn_alpha_init is not None:   config.v18_attn_alpha_init = args.v18_attn_alpha_init
    if args.contact_distill_weight is not None: config.contact_distill_weight = args.contact_distill_weight
    if args.contact_targets_path is not None:   config.contact_targets_path = args.contact_targets_path
    if getattr(config, "contact_distill_weight", 0.0) > 0:
        print(f"contact distillation: weight={config.contact_distill_weight} "
              f"targets={getattr(config,'contact_targets_path','data/contact_maps/contact_targets.json')}")
    print(f"v18 attention normalizer: {config.v18_attn_sparse}"
          + (f" (alpha_init={config.v18_attn_alpha_init})" if config.v18_attn_sparse == "entmax_learn" else ""))
    print(f"MoE: num_families={config.num_families} num_experts={config.num_experts} "
          f"top_k={config.top_k} expert_hidden={config.expert_hidden_dim} "
          f"family_embed={'learned' if not config.family_embedding_path else config.family_embedding_path}")

    # Data
    train_loader, val_loader = make_loaders(args, config)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    config.total_steps = total_steps
    print(f"Train batches/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    # Model & loss
    model   = TFScopeModel(config, use_dummy_backbone=args.dummy).to(device)
    loss_fn = TFScopeLoss(config).to(device)
    if config.lora_rank > 0 and not args.dummy:
        model.backbone.build(device)   # eagerly load ESM so LoRA params exist before optimizer

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
        ("moe.family_embed",      model.moe.family_embed),
        ("moe.gating",            model.moe.gating),
        ("moe.experts (×12)",     model.moe.experts),
        ("moe.film",              model.moe.film),
        ("gate_head",             model.gate_head),
        ("pwm_head",              model.pwm_head),
    ]
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

    # LoRA params live inside backbone but need a higher lr than frozen base weights
    lora_params  = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and not any(p is lp for lp in lora_params)]
    param_groups = [{'params': other_params, 'lr': args.lr}]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr})
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
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_prior = sum(1 for m in missing if m.startswith("pwm_head.prior_head."))
        print(f"Initialised model weights from {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)}, "
              f"prior_head missing={n_prior}); fresh optimizer/schedule/epoch")

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        loss_fn.load_state_dict(ckpt['loss_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch    = ckpt['epoch'] + 1
        global_step    = ckpt.get('global_step', start_epoch * steps_per_epoch)
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        best_oracle_r  = ckpt.get('best_oracle_r', -float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Save config
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # W&B
    wandb_run = None
    if not args.no_wandb:
        try:
            wandb_run = init_wandb(args, config)
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed ({e}) — continuing without logging")

    use_oracle = args.eval_oracle_r and not args.dummy
    oracle_hdr = f"  {'Oracle-r':>9}" if use_oracle else ""
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'L_gate':>8}  {'L_pwm':>8}{oracle_hdr}  {'Time':>6}")
    print("-" * (60 + (11 if use_oracle else 0)))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_m, global_step = run_train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, global_step, wandb_run)

        val_m = run_val_epoch(model, loss_fn, val_loader, device)

        # Oracle-r eval (every --oracle-r-every epochs or final epoch)
        oracle_r = None
        is_oracle_epoch = (use_oracle and (
            (epoch + 1) % args.oracle_r_every == 0 or epoch + 1 == args.epochs))
        if is_oracle_epoch:
            oracle_r = run_oracle_r_eval(
                model, val_loader, device,
                n_tfs=args.oracle_r_n_tfs)

        elapsed = time.time() - t0
        oracle_str = f"  {oracle_r:>9.4f}" if oracle_r is not None else (
                     f"  {'':>9}"           if use_oracle else "")
        print(f"{epoch+1:>6}  {train_m['loss']:>10.4f}  {val_m['loss']:>10.4f}  "
              f"{val_m['gate_loss']:>8.4f}  {val_m['pwm_loss']:>8.4f}"
              f"{oracle_str}  {elapsed:>5.0f}s")

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
        if wandb_run is not None:
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
        save_milestone = (epoch + 1) % args.save_every == 0
        save_best      = is_best and not args.no_save_best
        if save_milestone or save_best:
            trainable_state = {k: v for k, v in model.state_dict().items()
                               if not k.startswith('backbone._esm_model')}
            ckpt = {
                'epoch':         epoch,
                'global_step':   global_step,
                'model':         trainable_state,
                'loss_fn':       loss_fn.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_oracle_r': best_oracle_r,
                'train_metrics': train_m,
                'val_metrics':   val_m,
            }
            if oracle_r is not None:
                ckpt['oracle_r'] = oracle_r
            if save_milestone:
                path = os.path.join(args.out, f"ckpt_epoch{epoch+1:03d}.pt")
                torch.save(ckpt, path)
            if save_best:
                torch.save(ckpt, os.path.join(args.out, "ckpt_best.pt"))
                if use_oracle:
                    print(f"  *** New best oracle r: {best_oracle_r:.4f} ***")
                else:
                    print(f"  *** New best val loss: {best_val_loss:.4f} ***")

        # Early stopping (only checked on oracle-r epochs when --eval-oracle-r is set)
        check_patience = (not use_oracle) or is_oracle_epoch
        if args.early_stop_patience > 0 and check_patience and \
                patience_counter >= args.early_stop_patience:
            metric_str = (f"oracle r has not improved for {patience_counter} oracle-r epochs"
                          if use_oracle else
                          f"val loss has not improved for {patience_counter} epochs")
            print(f"\nEarly stopping at epoch {epoch+1}: {metric_str} "
                  f"(patience={args.early_stop_patience}).")
            if use_oracle:
                print(f"Best oracle r was {best_oracle_r:.4f}.")
            else:
                print(f"Best val loss was {best_val_loss:.4f}.")
            break

    if use_oracle:
        print(f"\nDone. Best oracle r: {best_oracle_r:.4f}")
    else:
        print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.out}/")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
