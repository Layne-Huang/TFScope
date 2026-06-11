"""Stage-A contrastive pretraining model (DPAC-style protein↔DNA alignment).

Reuses the *transferable* TFScope protein-encoder submodules under identical
attribute names (`backbone`, `dbd_indicator`, `global_pool`, `dbd_pool`,
`projection`) so the learned weights load straight into `TFScopeModel` for
Stage-B PWM finetuning. Adds a light DNA tower (the bound motif is short, ≤~50
bp, so a small 1-D CNN beats a genomic LM here) and CLIP-style projection heads.

Loss: symmetric InfoNCE (in-batch negatives) over L2-normalised protein/DNA
embeddings with a learnable temperature — teaches the protein tower "what DNA
this protein binds" before it ever has to emit a PWM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.models.backbone import Backbone
from tfscope.models.pooling import GatedAttentionPooling, ProjectionHead

# DNA one-hot channel order must match the PWM convention used elsewhere.
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}


def dna_to_onehot(seqs, max_len):
    """List[str] → (B, 4, max_len) one-hot float, right-padded with zeros."""
    B = len(seqs)
    x = torch.zeros(B, 4, max_len)
    for i, s in enumerate(seqs):
        s = s.upper()[:max_len]
        for j, ch in enumerate(s):
            k = DNA_VOCAB.get(ch)
            if k is not None:
                x[i, k, j] = 1.0
    return x


class DNAEncoder(nn.Module):
    """Small multi-scale 1-D CNN over one-hot DNA → fixed embedding."""

    def __init__(self, dim=512, channels=256):
        super().__init__()
        # parallel kernels capture motifs of different widths
        self.convs = nn.ModuleList([
            nn.Conv1d(4, channels, k, padding=k // 2) for k in (5, 9, 15)
        ])
        self.bn = nn.BatchNorm1d(channels * 3)
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, dim), nn.GELU(), nn.LayerNorm(dim),
        )

    def forward(self, onehot):                       # (B, 4, L)
        feats = [F.gelu(c(onehot)) for c in self.convs]
        h = torch.cat(feats, dim=1)                  # (B, 3C, L)
        h = self.bn(h)
        h = h.max(dim=-1).values                     # global max-pool over positions
        return self.proj(h)                          # (B, dim)


class ContrastivePretrainModel(nn.Module):
    """Protein tower (transferable TFScope encoder) + DNA tower + CLIP heads."""

    def __init__(self, config: TFScopeConfig, contrastive_dim: int = 256,
                 dna_max_len: int = 50, use_dummy_backbone: bool = False):
        super().__init__()
        self.config = config
        self.dna_max_len = dna_max_len

        # ── transferable protein encoder (names MUST match TFScopeModel) ──
        if use_dummy_backbone:
            from tfscope.models.backbone import DummyBackbone
            self.backbone = DummyBackbone(config)
        else:
            self.backbone = Backbone(config)
        self.dbd_indicator = nn.Parameter(torch.zeros(1, 1, config.esm_embed_dim))
        self.global_pool = GatedAttentionPooling(config)
        self.dbd_pool    = GatedAttentionPooling(config)
        self.projection  = ProjectionHead(config)

        # ── DNA tower + projection heads (pretrain-only) ──
        self.dna_encoder = DNAEncoder(dim=contrastive_dim)
        self.prot_head = nn.Sequential(
            nn.Linear(config.proj_hidden_dim, contrastive_dim), nn.GELU(),
            nn.LayerNorm(contrastive_dim),
        )
        self.dna_head = nn.Sequential(
            nn.Linear(contrastive_dim, contrastive_dim), nn.GELU(),
            nn.LayerNorm(contrastive_dim),
        )
        # learnable temperature (CLIP); clamp at use
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_protein(self, sequence_tokens, dbd_mask=None):
        emb = self.backbone(sequence_tokens)                          # (B,L,D)
        if dbd_mask is None:
            dbd_mask = torch.ones(sequence_tokens.shape, dtype=torch.bool,
                                  device=sequence_tokens.device)
        dbd_emb = emb + dbd_mask.unsqueeze(-1).float() * self.dbd_indicator
        g = self.global_pool(dbd_emb)
        d = self.dbd_pool(dbd_emb, dbd_mask)
        combined = self.projection(g, d)                              # (B, proj_hidden)
        return F.normalize(self.prot_head(combined), dim=-1)

    def encode_dna(self, dna_onehot):
        return F.normalize(self.dna_head(self.dna_encoder(dna_onehot)), dim=-1)

    def forward(self, sequence_tokens, dna_onehot, dbd_mask=None):
        zp = self.encode_protein(sequence_tokens, dbd_mask)           # (B, d)
        zd = self.encode_dna(dna_onehot)                              # (B, d)
        scale = self.logit_scale.clamp(max=4.6052).exp()             # cap temp≈0.01
        logits = scale * zp @ zd.t()                                  # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item(),
                      "temp": (1 / scale).item()}

    def encoder_state_dict(self):
        """Only the keys that transfer into TFScopeModel (encoder portion)."""
        keep = ("backbone.", "dbd_indicator", "global_pool.", "dbd_pool.",
                "projection.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}
