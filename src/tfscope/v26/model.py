"""v26 model assembly.

Forward signature accepts ONLY sequence-derived tensors. There is no parameter through which
family_id, motif source, gene, accession, PDB id, seq/str provenance or a structure-availability
flag could be supplied -- audit Finding G is closed by the interface, not by convention.
assert_no_metadata_inputs() re-checks this at runtime.

Chain handling: chains arrive already flattened into the batch dimension, one row per chain, with
chain_index mapping rows back to examples. ESM therefore encodes every chain independently
(Finding H), and partners are aggregated permutation-invariantly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tfscope.v26.config import V26Config
from tfscope.v26.context import AttentionPool, GatedFlankContext, PartnerSetAggregator
from tfscope.v26.encoder import PerChainESMEncoder
from tfscope.v26.heads import ContactAwarePWMHead, MotifLengthHead
from tfscope.v26.moe import DenseFFN, SequenceConditionedMoE

BANNED_KEYS = {
    "family_id", "family", "family_name", "motif_source", "source_id", "pfam", "pfam_id",
    "interpro", "interpro_id", "gene", "gene_id", "gene_symbol", "uniprot", "uniprot_id",
    "accession", "primary_accession", "pdb_id", "structure_id", "provenance", "set_label",
    "seq_or_str", "has_structure", "structure_available", "assay_type", "dbd_families",
    "dbd_families_for_analysis_only", "legacy_filename",
}


def assert_no_metadata_inputs(batch: dict):
    """Raise if a batch carries any identity/provenance tensor. Called every forward."""
    bad = sorted(k for k in batch if k.lower() in BANNED_KEYS)
    if bad:
        raise ValueError(
            f"v26 forbids metadata inputs; batch contains {bad}. "
            "Family/source/provenance are for stratification, error analysis and reporting only.")


class TFScopeV26(nn.Module):
    def __init__(self, cfg: V26Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or V26Config()
        d = cfg.d_model
        self.encoder = PerChainESMEncoder(cfg)
        self.mixer = (SequenceConditionedMoE(d, cfg.expert_hidden, cfg.n_routed_experts,
                                            cfg.n_shared_experts, cfg.top_k, cfg.dropout)
                      if cfg.use_moe else
                      DenseFFN(d, cfg.expert_hidden, cfg.n_routed_experts + cfg.n_shared_experts,
                               cfg.dropout))
        self.core_pool = AttentionPool(d)
        self.flank = GatedFlankContext(d, gate_bias_init=cfg.flank_gate_bias_init,
                                       flank_dropout=cfg.flank_dropout)
        self.partners = PartnerSetAggregator(d, gate_bias_init=cfg.partner_gate_bias_init,
                                             partner_dropout=cfg.partner_dropout)
        self.length_head = MotifLengthHead(d, cfg.min_motif_length, cfg.max_motif_length)
        self.pwm_head = ContactAwarePWMHead(d, cfg.max_motif_length, cfg.pwm_attn_heads,
                                           cfg.lambda_contact_init)
        self.contact1d = nn.Linear(d, 1)                      # per-residue DNA-contact logit

    def build(self, device):
        self.encoder.build(device)

    def forward(self, sequence_tokens, dbd_mask, chain_index, is_primary,
                use_flank=None, use_partners=None, _batch_for_check=None):
        """
        sequence_tokens (C, L) one row per CHAIN, across the whole batch
        dbd_mask        (C, L) True inside the DBD core of that chain
        chain_index     (C,)   example index each chain row belongs to
        is_primary      (C,)   True for the example's primary chain
        """
        if _batch_for_check is not None:
            assert_no_metadata_inputs(_batch_for_check)
        cfg = self.cfg
        use_flank = cfg.use_flank if use_flank is None else use_flank
        use_partners = cfg.use_partners if use_partners is None else use_partners

        h, pad = self.encoder(sequence_tokens, dbd_mask)          # (C,L,d)
        core_ignore = pad | (~dbd_mask)
        z_chain_core = self.core_pool(h, core_ignore)             # (C,d)
        h, mix_aux = self.mixer(h, z_chain_core, pad)

        # per-chain context (flank residual), then split primary vs partner rows
        z_chain, fl_aux = self.flank(h, pad, dbd_mask, use_flank=use_flank)

        B = int(chain_index.max().item()) + 1 if chain_index.numel() else 0
        prim_rows = torch.nonzero(is_primary, as_tuple=True)[0]
        order = torch.argsort(chain_index[prim_rows])
        prim_rows = prim_rows[order]
        z_primary = z_chain[prim_rows]                            # (B,d)

        # pack partners as a padded SET per example (no order feature anywhere)
        P = max(1, cfg.max_partners)
        z_part = z_chain.new_zeros(B, P, z_chain.shape[-1])
        p_mask = torch.zeros(B, P, dtype=torch.bool, device=z_chain.device)
        fill = torch.zeros(B, dtype=torch.long, device=z_chain.device)
        for r in torch.nonzero(~is_primary, as_tuple=True)[0].tolist():
            b = int(chain_index[r].item())
            k = int(fill[b].item())
            if k < P:
                z_part[b, k] = z_chain[r]
                p_mask[b, k] = True
                fill[b] = k + 1
        z, pa_aux = self.partners(z_primary, z_part, p_mask, use_partners=use_partners)

        gate, len_aux = self.length_head(z)
        h_prim = h[prim_rows]
        core_ignore_prim = core_ignore[prim_rows]
        pwm, pwm_aux = self.pwm_head(z, h_prim, core_ignore_prim)
        contact_logit = self.contact1d(h_prim).squeeze(-1)        # (B,L) 1-D contact prediction

        aux = {"gate": gate, "z": z, "balance_loss": mix_aux["balance_loss"],
               "alpha_flank": fl_aux["alpha"], "beta_partner": pa_aux["beta"],
               "contact1d_logit": contact_logit,
               "core_ignore_primary": core_ignore_prim,
               "primary_rows": prim_rows}
        aux.update(len_aux)
        aux.update(pwm_aux)
        return pwm, gate, aux

    def param_counts(self):
        tot = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        esm = sum(p.numel() for n, p in self.named_parameters() if "_esm" in n)
        return {"total_excl_esm_trunk": tot - esm, "trainable": train, "esm_in_module": esm}
