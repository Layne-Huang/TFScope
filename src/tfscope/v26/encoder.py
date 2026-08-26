"""v26 encoder: per-chain ESM-2 + amino-acid identity skip + residue refiner.

Fixes three audit findings at once:

  Finding H -- v24 concatenated every chain into ONE ESM-2 forward separated by <eos>, plus a
    per-token chain-ID embedding (dataset.py:547-566, backbone.py:147-151). That makes the
    representation ORDER-AWARE: swapping two protomers of a homodimer changes the prediction.
    Here every chain is a separate row of the ESM batch, so chains never attend to each other and
    no chain index is ever supplied.

  Finding G -- no family_id, motif source, gene, accession, PDB id or provenance enters the model.
    The forward signature physically cannot accept them.

  mutation blindness -- a frozen PLM is nearly invariant to a single substitution, and pooling
    dilutes what little changes. An explicit amino-acid identity skip plus a relative-position
    embedding gives a substitution a direct, non-diluted path into the residue state:

        h_i = LayerNorm(h_i^ESM + W_aa AA(a_i) + W_pos relpos(i))

LoRA and the ESM loader are reused from tfscope.models.backbone so the encoder is byte-identical
to v24's where it should be; only the chain handling and the skip connections are new.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.models.backbone import ESM2_BOS_TOKEN, ESM2_PAD_TOKEN, LoRALinear

N_AA_TOKENS = 33            # ESM-2 alphabet size (upper bound; unused rows stay unlearned)


class ResidueRefiner(nn.Module):
    """Small post-PLM transformer so residue states can specialise for PWM reading.

    Deliberately shallow: the point is to re-mix ESM features and the AA skip, not to re-learn
    protein language from 1.7k proteins (bigger capacity overfit in the v24 exploration).
    """

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # pad_mask: True where PADDING (torch convention for src_key_padding_mask)
        return self.enc(h, src_key_padding_mask=pad_mask)


class PerChainESMEncoder(nn.Module):
    """Encode each protein chain independently with a shared ESM-2.

    forward() takes chains already flattened into the batch dimension. The caller is responsible
    for the (example -> chain rows) mapping; see complex.ChainPacker. Because chains occupy
    separate batch rows, ESM attention cannot cross a chain boundary, and no chain-identity
    feature exists anywhere in the model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.esm_embed_dim
        self.n_layers = cfg.esm_layers_to_average
        self.layer_weights = nn.Parameter(torch.zeros(self.n_layers))
        self._esm = None

        d = cfg.d_model
        self.esm_proj = nn.Linear(self.embed_dim, d)
        self.aa_embed = nn.Embedding(N_AA_TOKENS, d)              # identity skip
        self.pos_embed = nn.Linear(3, d)                          # relative-DBD position
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.refiner = ResidueRefiner(d, cfg.refiner_layers, cfg.refiner_heads, cfg.dropout)
        nn.init.zeros_(self.pos_embed.bias)

    # ---- ESM plumbing (reused from v24 so the frozen trunk is identical)
    def _load_esm(self, device):
        if self._esm is not None:
            if next(self._esm.parameters()).device != device:
                self._esm = self._esm.to(device)
            return self._esm
        import esm as esm_lib
        model, _ = getattr(esm_lib.pretrained, self.cfg.esm_model)()
        model = model.to(device)
        if self.cfg.freeze_encoder:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        if self.cfg.lora_rank > 0:
            n = model.num_layers
            start = max(0, n - self.cfg.lora_n_layers)
            for i in range(start, n):
                a = model.layers[i].self_attn
                a.q_proj = LoRALinear(a.q_proj, self.cfg.lora_rank, self.cfg.lora_alpha).to(device)
                a.v_proj = LoRALinear(a.v_proj, self.cfg.lora_rank, self.cfg.lora_alpha).to(device)
        self._esm = model
        return model

    def build(self, device):
        self._load_esm(device)

    def _esm_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self._load_esm(tokens.device)
        B = tokens.shape[0]
        bos = torch.full((B, 1), ESM2_BOS_TOKEN, dtype=torch.long, device=tokens.device)
        t = torch.cat([bos, tokens], dim=1)
        n = model.num_layers
        repr_layers = list(range(n - self.n_layers + 1, n + 1))
        if self.cfg.lora_rank > 0:
            out = model(t, repr_layers=repr_layers, return_contacts=False)
        else:
            with torch.no_grad():
                out = model(t, repr_layers=repr_layers, return_contacts=False)
        w = F.softmax(self.layer_weights, dim=0)
        return sum(wi * out["representations"][l][:, 1:, :]
                   for wi, l in zip(w, repr_layers))

    def forward(self, tokens: torch.Tensor, dbd_mask: torch.Tensor):
        """tokens (C, L) one row per CHAIN; dbd_mask (C, L) True inside the DBD core.

        Returns h (C, L, d), pad_mask (C, L) True where padding.
        """
        pad = tokens.eq(ESM2_PAD_TOKEN)
        h_esm = self._esm_forward(tokens)                         # (C, L, esm_dim)
        h = self.esm_proj(h_esm) + self.aa_embed(tokens)          # identity skip

        # relative position features: position within chain, within DBD, and DBD membership.
        C, L = tokens.shape
        idx = torch.arange(L, device=tokens.device).unsqueeze(0).expand(C, L).float()
        valid = (~pad).float()
        n_valid = valid.sum(1, keepdim=True).clamp(min=1)
        rel_chain = idx / n_valid                                  # 0..1 along the chain
        d = dbd_mask.float()
        n_dbd = d.sum(1, keepdim=True).clamp(min=1)
        dbd_start = torch.argmax(d, dim=1, keepdim=True).float()
        rel_dbd = ((idx - dbd_start) / n_dbd).clamp(-2.0, 3.0)     # 0..1 inside the DBD
        feats = torch.stack([rel_chain, rel_dbd, d], dim=-1)
        h = h + self.pos_embed(feats)

        h = self.drop(self.norm(h))
        h = self.refiner(h, pad_mask=pad)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return h, pad
