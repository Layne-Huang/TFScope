import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.losses.balance import load_balance_loss, family_diversity_loss
from tfscope.losses.registration import latent_registration_loss


class TFScopeLoss(nn.Module):
    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.config = config

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _masked_mean(per_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average per-position values over valid (masked) positions."""
        valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return ((per_pos * mask).sum(dim=1) / valid.squeeze(1)).mean()

    # ── individual PWM terms ──────────────────────────────────────────────────

    @staticmethod
    def _pwm_kl(pred_log_probs: torch.Tensor, target_pwm: torch.Tensor,
                mask: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
        """KL(target ∥ pred) averaged over valid positions.
        label_smoothing mixes target toward uniform: t' = (1-α)t + α/4."""
        if label_smoothing > 0.0:
            target_pwm = (1.0 - label_smoothing) * target_pwm + label_smoothing * 0.25
        kl_per_pos = F.kl_div(
            pred_log_probs,
            (target_pwm + 1e-8),
            reduction='none',
            log_target=False,
        ).sum(dim=1)                                       # (B, L)
        return TFScopeLoss._masked_mean(kl_per_pos, mask)

    @staticmethod
    def _pwm_entropy(pred_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean entropy H(pred) over valid positions.
        Minimising this sharpens the predicted distribution toward one nucleotide."""
        pred_probs = pred_log_probs.exp()
        entropy_per_pos = -(pred_probs * pred_log_probs).sum(dim=1)   # (B, L) ≥ 0
        return TFScopeLoss._masked_mean(entropy_per_pos, mask)

    @staticmethod
    def _pwm_l1(pred_log_probs: torch.Tensor, target_pwm: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """L1 / MAE between softmax(pred) and target over valid positions.

        Inspired by DeepPBS (trainer.py:364).  More robust than KL when the
        predicted probability approaches zero at positions the target assigns
        non-zero mass.
        """
        pred_probs = pred_log_probs.exp()                  # (B, 4, L)
        l1_per_pos = (pred_probs - target_pwm).abs().sum(dim=1)   # (B, L)
        return TFScopeLoss._masked_mean(l1_per_pos, mask)

    @staticmethod
    def _pwm_ic(pred_log_probs: torch.Tensor, target_pwm: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """IC-matching loss: |IC(target) - IC(pred)| per position.

        IC(p) = KL(p ∥ uniform) = -H(p) + log(4)  (in nats).
        Penalises wrong specificity level independently of nucleotide identity.
        Inspired by DeepPBS icLoss (trainer.py:351-358).
        """
        log4 = math.log(4.0)

        t = target_pwm.clamp(1e-8, 1.0)
        ic_target = (t * t.log()).sum(dim=1) + log4        # (B, L)

        pred_probs = pred_log_probs.exp()
        ic_pred = (pred_probs * pred_log_probs).sum(dim=1) + log4  # (B, L)

        ic_diff = (ic_target - ic_pred).abs()              # (B, L)
        return TFScopeLoss._masked_mean(ic_diff, mask)

    @staticmethod
    def _pwm_ic_pcc(pred_log_probs: torch.Tensor, target_pwm: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """IC-weighted per-column (1 - Pearson r) loss.

        For each motif position, correlate the predicted 4-vector (A,C,G,T)
        against the target 4-vector, weight by target information content, and
        penalise (1 - r). Directly optimises the per-column Pearson metric the
        benchmark reports — addresses v10's base-composition gap (IC-r high but
        column-r only ~0.54).
        """
        pred = pred_log_probs.exp()                            # (B,4,L)
        pc = pred   - pred.mean(dim=1, keepdim=True)
        tc = target_pwm - target_pwm.mean(dim=1, keepdim=True)
        num = (pc * tc).sum(dim=1)                             # (B,L)
        den = pc.norm(dim=1) * tc.norm(dim=1) + 1e-8
        r = num / den                                          # (B,L) ∈ [-1,1]

        log4 = math.log(4.0)
        t = target_pwm.clamp(1e-8, 1.0)
        ic = (t * t.log()).sum(dim=1) + log4                  # (B,L) target IC (nats)
        w = ic * mask
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)    # IC-normalised weights
        return ((1.0 - r) * w).sum(dim=1).mean()

    @staticmethod
    def _pwm_coverage_r(
        pred_log_probs: torch.Tensor,
        pred_gate: torch.Tensor,
        target_pwm: torch.Tensor,
        mask: torch.Tensor,
        ic_thresh_bits: float,
    ) -> torch.Tensor:
        """Differentiable analogue of full-core coverage-aware column r.

        Correlation is computed for every informative target column, multiplied
        by the soft gate occupancy, and divided by the number of target-core
        columns. A missed column therefore contributes zero, matching ``r_cov``
        instead of disappearing from the denominator.
        """
        pred = pred_log_probs.exp()
        pc = pred - pred.mean(dim=1, keepdim=True)
        tc = target_pwm - target_pwm.mean(dim=1, keepdim=True)
        r = (pc * tc).sum(dim=1) / (
            pc.norm(dim=1) * tc.norm(dim=1)
        ).clamp(min=1e-8)
        target = target_pwm.clamp(1e-8, 1.0)
        ic_bits = 2.0 + (target * torch.log2(target)).sum(dim=1)
        core = mask * (ic_bits >= ic_thresh_bits).to(mask.dtype)
        # If no column crosses the IC threshold, retain the valid target span.
        has_core = core.sum(dim=1, keepdim=True) > 0
        core = torch.where(has_core, core, mask)
        score = (
            r * pred_gate.sigmoid() * core
        ).sum(dim=1) / core.sum(dim=1).clamp(min=1.0)
        return (1.0 - score).mean()

    @staticmethod
    def _pwm_topbase_margin(pred_pwm: torch.Tensor, target_pwm: torch.Tensor,
                            mask: torch.Tensor, margin: float,
                            ic_thresh_nats: float) -> torch.Tensor:
        """Hinge margin pushing the true top base's logit above the runner-up,
        applied only at high-IC (confident) target positions."""
        B, _, L = pred_pwm.shape
        true_top = target_pwm.argmax(dim=1)                    # (B,L)
        z_true = pred_pwm.gather(1, true_top.unsqueeze(1)).squeeze(1)   # (B,L)
        neg = pred_pwm.clone()
        neg.scatter_(1, true_top.unsqueeze(1), float('-inf'))
        z_second = neg.max(dim=1).values                       # (B,L)
        hinge = F.relu(margin - (z_true - z_second))           # (B,L)

        log4 = math.log(4.0)
        t = target_pwm.clamp(1e-8, 1.0)
        ic = (t * t.log()).sum(dim=1) + log4                  # (B,L) nats
        hi = (ic > ic_thresh_nats).float() * mask
        return (hinge * hi).sum() / hi.sum().clamp(min=1.0)

    @staticmethod
    def _pwm_contrastive(pred_log_probs: torch.Tensor, target_pwm: torch.Tensor,
                         mask: torch.Tensor, tau: float) -> torch.Tensor:
        """DPAC-style in-batch contrastive loss over PWMs (anti family-collapse).

        Targets are in the canonical left-anchored trimmed frame, so positions
        are comparable across samples without a learned aligner. Build a pairwise
        similarity s[i,j] = IC-weighted mean per-column cosine between predicted
        PWM i and target PWM j over their overlapping valid columns, then apply a
        symmetric InfoNCE so each prediction matches its OWN target better than
        any other protein's target in the batch. Parameter-free; punishes
        collapse to a generic family-average motif (the LFO Homeodomain failure).
        """
        B = pred_log_probs.shape[0]
        if B < 2:
            return pred_log_probs.new_zeros(())
        P = pred_log_probs.exp()                              # (B,4,L)
        T = target_pwm                                        # (B,4,L)
        # center over the 4 bases, then unit-normalise each column → cosine ready
        Pc = P - P.mean(dim=1, keepdim=True)
        Tc = T - T.mean(dim=1, keepdim=True)
        Pn = Pc / (Pc.norm(dim=1, keepdim=True) + 1e-8)      # (B,4,L)
        Tn = Tc / (Tc.norm(dim=1, keepdim=True) + 1e-8)      # (B,4,L)
        # per-column cosine for every (pred i, target j): (B,B,L)
        cos = torch.einsum('ibl,jbl->ijl', Pn, Tn)
        # target-IC weight per (target j, column l), gated by both masks
        log4 = math.log(4.0)
        t = target_pwm.clamp(1e-8, 1.0)
        ic = ((t * t.log()).sum(dim=1) + log4).clamp(min=0.0)  # (B,L) target IC nats
        wj = (ic * mask)                                       # (B,L)
        # pairwise weight w[i,j,l] = wj[j,l] * mask_i[l]
        w = mask.unsqueeze(1) * wj.unsqueeze(0)               # (B,B,L)
        denom = w.sum(dim=2).clamp(min=1e-6)                  # (B,B)
        s = (cos * w).sum(dim=2) / denom                      # (B,B) similarity
        logits = s / tau
        labels = torch.arange(B, device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))

    # ── v18 attention-repair / contact-supervision terms ──────────────────────

    @staticmethod
    def _v18_attn_terms(attn, key_mask, pwm_mask, recog_prior, hub_frac):
        """Compute (row_diversity, hub_penalty, contact_xent) from attention.

        attn:        (B, Lq, Lk) attention weights (rows sum to 1)
        key_mask:    (B, Lk) bool  True = valid DBD residue
        pwm_mask:    (B, Lq)       1 = valid motif column
        recog_prior: (B, Lk) or None  soft recognition-residue target (>=0)
        Returns three scalars (contact_xent may be None).
        """
        B, Lq, Lk = attn.shape
        col = pwm_mask.unsqueeze(-1)                                  # (B, Lq, 1)
        n_col = pwm_mask.sum(dim=1).clamp(min=1)                      # (B,)

        # row-diversity: mean pairwise cosine between valid motif columns' attn rows.
        an = F.normalize(attn + 1e-9, dim=-1)                        # (B, Lq, Lk)
        S = torch.matmul(an, an.transpose(1, 2))                     # (B, Lq, Lq)
        pair = pwm_mask.unsqueeze(2) * pwm_mask.unsqueeze(1)         # (B, Lq, Lq)
        eye = torch.eye(Lq, device=attn.device).view(1, Lq, Lq)
        pair = pair * (1 - eye)
        denom = pair.sum(dim=(1, 2)).clamp(min=1)
        row_div = ((S * pair).sum(dim=(1, 2)) / denom).mean()        # want SMALL

        # hub penalty: residue usage summed over valid columns shouldn't exceed u_max.
        usage = (attn * col).sum(dim=1)                              # (B, Lk)
        u_max = (hub_frac * n_col).view(B, 1)
        hub = (F.relu(usage - u_max) ** 2)
        hub = (hub * key_mask.float()).sum(dim=1) / key_mask.float().sum(dim=1).clamp(min=1)
        hub = hub.mean()

        # contact supervision: pull attention marginal onto recognition residues.
        contact = None
        if recog_prior is not None:
            r = (recog_prior * key_mask.float())                     # (B, Lk)
            has = (r.sum(dim=1) > 0)
            if has.any():
                rn = r / r.sum(dim=1, keepdim=True).clamp(min=1e-8)  # target dist
                marg = (attn * col).sum(dim=1) / n_col.view(B, 1)    # (B, Lk) attn marginal
                xent = -(rn * torch.log(marg + 1e-8)).sum(dim=1)     # (B,)
                contact = xent[has].mean()
        return row_div, hub, contact

    @staticmethod
    def _contact_distill(attn, target, base_mask):
        """Distill REAL structural contacts into the (sparse) attention.

        attn:      (B, Lq, Lk)   predicted attention rows (one distribution per base)
        target:    (B, Lq, Lk)   structural target rows from a co-crystal (softmax(-d/tau));
                                  rows for non-contacting / non-structured bases are all-zero
        base_mask: (B, Lq)       1 where this base has a structural target (structured subset)
        Returns mean KL(target || attn) over masked base positions, or None if none present.
        """
        m = base_mask.bool()
        if m.sum() == 0:
            return None
        a = attn[m].clamp_min(1e-8)                                  # (N, Lk)
        t = target[m]                                                # (N, Lk), each row sums to 1
        kl = (t * (t.clamp_min(1e-8).log() - a.log())).sum(dim=-1)   # (N,)
        return kl.mean()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, pred_gate, pred_pwm, target_pwm, pwm_mask,
                gate_logits=None, top_indices=None, family_id=None,
                trust_logits=None, retrieved_pwms=None, retrieved_masks=None,
                attn=None, attn_key_mask=None, recog_prior=None,
                contact_target=None, contact_base_mask=None,
                registration_anchor_mask=None,
                registration_anchor_mode=None,
                registration_orientation=None,
                registration_offset=None,
                register_logits=None):
        """
        Args:
            pred_gate:  (B, max_length) gate logits (pre-sigmoid)
            pred_pwm:   (B, 4, max_length) nucleotide logits
            target_pwm: (B, 4, max_length) target probability distributions
            pwm_mask:   (B, max_length) ground-truth position mask (0/1)
            gate_logits, top_indices, family_id: for MOE auxiliary losses

        Returns:
            total_loss: scalar
            metrics: dict of individual losses for logging
        """
        pred_log_probs = F.log_softmax(pred_pwm, dim=1)    # (B, 4, L)
        registration_metrics = {}
        if getattr(self.config, "latent_registration", False):
            total, registration_metrics, loss_target_pwm, loss_pwm_mask = (
                latent_registration_loss(
                    pred_gate,
                    pred_pwm,
                    target_pwm,
                    pwm_mask,
                    max_shift=self.config.registration_max_shift,
                    min_overlap=self.config.registration_min_overlap,
                    temperature=self.config.registration_temperature,
                    coverage_penalty=self.config.registration_coverage_penalty,
                    gate_weight=self.config.gate_loss_weight,
                    l1_weight=self.config.pwm_l1_weight,
                    ic_weight=self.config.pwm_ic_weight,
                    ic_pcc_weight=self.config.pwm_ic_pcc_weight,
                    topbase_weight=self.config.pwm_topbase_weight,
                    topbase_margin=self.config.pwm_topbase_margin,
                    topbase_ic_thresh_bits=self.config.pwm_topbase_ic_thresh,
                    anchor_mask=registration_anchor_mask,
                    anchor_mode=registration_anchor_mode,
                    anchor_orientation=registration_orientation,
                    anchor_offset=registration_offset,
                    register_logits=register_logits,
                    register_loss_weight=getattr(
                        self.config, "register_loss_weight", 0.0
                    ),
                )
            )
            L_gate = registration_metrics["registration_gate"]
            L_l1 = registration_metrics["registration_l1"]
            L_ic = registration_metrics["registration_ic"]
            L_ic_pcc = registration_metrics["registration_ic_pcc"]
            L_top = registration_metrics["registration_topbase"]
            ordinal_violation = pred_pwm.new_zeros(())
            L_entropy = self._pwm_entropy(
                pred_log_probs, loss_pwm_mask
            )
            L_pwm = (
                registration_metrics["registration_pwm"]
                + self.config.pwm_entropy_weight * L_entropy
            )
            total = total + self.config.pwm_entropy_weight * L_entropy
            metrics_ic_pcc = L_ic_pcc.item()
            metrics_top = L_top.item()
        else:
            loss_target_pwm = target_pwm
            loss_pwm_mask = pwm_mask
            L_gate = F.binary_cross_entropy_with_logits(pred_gate, pwm_mask)
            gate_probs = pred_gate.sigmoid()
            ordinal_violation = F.relu(
                gate_probs[:, 1:] - gate_probs[:, :-1]
            ).mean()
            L_gate = L_gate + self.config.gate_ordinal_weight * ordinal_violation

            L_l1 = self._pwm_l1(pred_log_probs, target_pwm, pwm_mask)
            L_ic = self._pwm_ic(pred_log_probs, target_pwm, pwm_mask)
            L_entropy = self._pwm_entropy(pred_log_probs, pwm_mask)
            L_pwm = (
                self.config.pwm_l1_weight * L_l1
                + self.config.pwm_ic_weight * L_ic
                + self.config.pwm_entropy_weight * L_entropy
            )
            if getattr(self.config, "pwm_ic_pcc_weight", 0.0) > 0:
                L_ic_pcc = self._pwm_ic_pcc(
                    pred_log_probs, target_pwm, pwm_mask
                )
                L_pwm = L_pwm + self.config.pwm_ic_pcc_weight * L_ic_pcc
                metrics_ic_pcc = L_ic_pcc.item()
            else:
                metrics_ic_pcc = 0.0
            if getattr(self.config, "pwm_topbase_weight", 0.0) > 0:
                L_top = self._pwm_topbase_margin(
                    pred_pwm,
                    target_pwm,
                    pwm_mask,
                    margin=self.config.pwm_topbase_margin,
                    ic_thresh_nats=(
                        self.config.pwm_topbase_ic_thresh * math.log(2.0)
                    ),
                )
                L_pwm = L_pwm + self.config.pwm_topbase_weight * L_top
                metrics_top = L_top.item()
            else:
                metrics_top = 0.0
            total = self.config.gate_loss_weight * L_gate + L_pwm

        # ── B2: length coupling (applies to BOTH branches) ───────────────────
        # `pwm_mask` is always the GT mask, so gt_len is well defined in either
        # branch. soft_len uses the sigmoid sum (differentiable) rather than a
        # hard >0.5 count, so gradients reach the gate head.
        soft_len = pred_gate.sigmoid().sum(dim=1)          # (B,)
        gt_len = pwm_mask.sum(dim=1)                       # (B,)
        L_length = F.smooth_l1_loss(soft_len, gt_len)
        if getattr(self.config, "gate_length_weight", 0.0) > 0:
            total = total + self.config.gate_length_weight * L_length

        cov_r_weight = getattr(self.config, "pwm_cov_r_weight", 0.0)
        if cov_r_weight > 0:
            L_cov_r = self._pwm_coverage_r(
                pred_log_probs,
                pred_gate,
                loss_target_pwm,
                loss_pwm_mask,
                getattr(self.config, "pwm_core_ic_thresh", 0.25),
            )
            total = total + cov_r_weight * L_cov_r
            L_pwm = L_pwm + cov_r_weight * L_cov_r
        else:
            L_cov_r = pred_pwm.new_zeros(())

        # B1: length observability -- reported regardless of whether the
        # penalty is enabled, so the baseline length error is always visible.
        with torch.no_grad():
            pred_len_hard = (pred_gate.sigmoid() > 0.5).float().sum(dim=1)
            length_mae = (pred_len_hard - gt_len).abs().mean()
            length_bias = (pred_len_hard - gt_len).mean()   # signed: <0 = too short

        # DPAC-style in-batch contrastive (anti family-collapse)
        if getattr(self.config, "pwm_contrastive_weight", 0.0) > 0:
            L_contrast = self._pwm_contrastive(
                pred_log_probs, loss_target_pwm, loss_pwm_mask,
                tau=getattr(self.config, "pwm_contrastive_tau", 0.1))
            L_pwm = L_pwm + self.config.pwm_contrastive_weight * L_contrast
            total = total + self.config.pwm_contrastive_weight * L_contrast
            metrics_contrast = L_contrast.item()
        else:
            metrics_contrast = 0.0

        metrics = {
            'gate_loss':         L_gate.item(),
            'pwm_loss':          L_pwm.item(),
            'pwm_l1':            L_l1.item(),
            'pwm_ic':            L_ic.item(),
            'pwm_entropy':       L_entropy.item(),
            'pwm_ic_pcc':        metrics_ic_pcc,
            'pwm_topbase':       metrics_top,
            'pwm_contrastive':   metrics_contrast,
            'ordinal_violation': ordinal_violation.item(),
            'length_loss':       L_length.item(),
            'length_mae':        length_mae.item(),
            'length_bias':       length_bias.item(),
            'pwm_cov_r':         L_cov_r.item(),
        }
        metrics.update(
            {key: value.item() for key, value in registration_metrics.items()}
        )

        # ── Trust predictor auxiliary loss (v10) ──────────────────────────────
        # Supervise the learned PWM-transfer-quality scorer using actual
        # per-position Pearson r between each retrieved PWM and the target.
        if (trust_logits is not None and retrieved_pwms is not None
            and retrieved_masks is not None):
            from tfscope.models.retrieval import (
                compute_aligned_true_trust,
                compute_true_trust,
                pairwise_trust_rank_loss,
            )
            with torch.no_grad():
                if getattr(self.config, "aligned_trust_target", False):
                    trust_target = compute_aligned_true_trust(
                        retrieved_pwms,
                        retrieved_masks,
                        loss_target_pwm,
                        loss_pwm_mask,
                        max_shift=getattr(
                            self.config, "registration_max_shift", 10
                        ),
                        min_overlap=getattr(
                            self.config, "registration_min_overlap", 4
                        ),
                    )
                else:
                    trust_target = compute_true_trust(
                        retrieved_pwms, retrieved_masks,
                        loss_target_pwm, loss_pwm_mask)                          # (B, K) ∈ [0,1]
            # Only supervise on actually-present neighbours
            valid_neighbour = (retrieved_masks.sum(dim=-1) > 0).float()           # (B, K)
            L_trust_per = F.binary_cross_entropy_with_logits(
                trust_logits, trust_target, reduction='none')                     # (B, K)
            L_trust = (L_trust_per * valid_neighbour).sum() / valid_neighbour.sum().clamp(min=1.0)
            total = total + self.config.trust_loss_weight * L_trust
            metrics['trust_loss'] = L_trust.item()
            rank_weight = getattr(
                self.config, "trust_rank_loss_weight", 0.0
            )
            if rank_weight > 0:
                L_trust_rank = pairwise_trust_rank_loss(
                    trust_logits,
                    trust_target,
                    valid_neighbour,
                    margin=getattr(self.config, "trust_rank_margin", 0.1),
                )
                total = total + rank_weight * L_trust_rank
                metrics["trust_rank_loss"] = L_trust_rank.item()
            with torch.no_grad():
                metrics['trust_mean']   = trust_logits.sigmoid().mean().item()
                metrics['trust_target'] = (trust_target * valid_neighbour).sum().item() \
                                          / valid_neighbour.sum().clamp(min=1.0).item()

        # ── v18 attention-repair + contact supervision ────────────────────────
        if attn is not None and attn_key_mask is not None:
            row_div, hub, contact = self._v18_attn_terms(
                attn, attn_key_mask, loss_pwm_mask,
                recog_prior if getattr(self.config, "v18_contact_supervision", False) else None,
                getattr(self.config, "v18_hub_frac", 0.34),
            )
            w_div = getattr(self.config, "v18_row_div_weight", 0.0)
            w_hub = getattr(self.config, "v18_hub_weight", 0.0)
            total = total + w_div * row_div + w_hub * hub
            metrics['v18_row_div'] = row_div.item()
            metrics['v18_hub']     = hub.item()
            if contact is not None:
                w_c = getattr(self.config, "v18_contact_weight", 0.0)
                total = total + w_c * contact
                metrics['v18_contact'] = contact.item()

            # contact-DISTILLATION from real co-crystal structures (structured subset only)
            w_cd = getattr(self.config, "contact_distill_weight", 0.0)
            if w_cd > 0 and contact_target is not None and contact_base_mask is not None:
                cd = self._contact_distill(attn, contact_target, contact_base_mask)
                if cd is not None:
                    total = total + w_cd * cd
                    metrics['contact_distill'] = cd.item()

        # ── MoE auxiliary losses ───────────────────────────────────────────────
        if gate_logits is not None and top_indices is not None and family_id is not None:
            L_balance = load_balance_loss(
                gate_logits, top_indices,
                num_experts=self.config.num_experts,
                alpha=self.config.balance_loss_weight,
            )
            L_diversity = family_diversity_loss(
                gate_logits, family_id,
                num_families=self.config.num_families,
                num_experts=self.config.num_experts,
                alpha=self.config.diversity_loss_weight,
            )
            total = total + L_balance + L_diversity
            metrics['balance_loss']   = L_balance.item()
            metrics['diversity_loss'] = L_diversity.item()

            # ── routing supervision: push expert i to own recognition-mode i ──
            # (use with a mode-relabeled parquet where family_id == mode_id and
            #  num_experts == num_families == num_modes)
            w_route = getattr(self.config, "route_supervision_weight", 0.0)
            if w_route > 0 and gate_logits.shape[-1] == self.config.num_families:
                L_route = F.cross_entropy(gate_logits, family_id.long())
                total = total + w_route * L_route
                metrics['route_loss'] = L_route.item()

        return total, metrics
