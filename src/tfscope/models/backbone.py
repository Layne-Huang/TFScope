import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tfscope.config import TFScopeConfig

ESM2_PAD_TOKEN = 1   # <pad>
ESM2_BOS_TOKEN = 0   # <cls>


class LoRALinear(nn.Module):
    """Frozen linear + trainable low-rank delta: y = W x + (alpha/r) * B A x."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        d_in  = linear.weight.shape[1]
        d_out = linear.weight.shape[0]
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora


class Backbone(nn.Module):
    """ESM-2 encoder.  Prepends <cls> before forwarding, strips it from output.

    Sequence tokens from the dataset are raw amino acid IDs (no special tokens).
    Padding is represented by ESM2_PAD_TOKEN=1.  Learned layer weights over the
    last `esm_layers_to_average` transformer layers are kept for potential
    fine-tuning; during frozen operation only the final layer is used.
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.esm_embed_dim
        self.n_layers = config.esm_layers_to_average
        self.layer_weights = nn.Parameter(torch.zeros(self.n_layers))
        self._esm_model = None

    def _load_esm(self, device: torch.device):
        if self._esm_model is not None:
            # Move to device if it was loaded on a different one (e.g. CPU → CUDA)
            if next(self._esm_model.parameters()).device != device:
                self._esm_model = self._esm_model.to(device)
            return self._esm_model
        import esm as esm_lib
        loader = getattr(esm_lib.pretrained, self.config.esm_model)
        model, _ = loader()
        model = model.to(device)
        if self.config.freeze_encoder:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        if self.config.lora_rank > 0:
            self._inject_lora(model, device)
        self._esm_model = model
        return model

    def _inject_lora(self, model: nn.Module, device: torch.device):
        """Replace q_proj and v_proj in the last lora_n_layers with LoRALinear."""
        n = model.num_layers
        rank  = self.config.lora_rank
        alpha = self.config.lora_alpha
        start = max(0, n - self.config.lora_n_layers)
        for i in range(start, n):
            attn = model.layers[i].self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank, alpha).to(device)
            attn.v_proj = LoRALinear(attn.v_proj, rank, alpha).to(device)

    def build(self, device: torch.device):
        """Eagerly load ESM so LoRA params exist before optimizer setup."""
        self._load_esm(device)

    def _forward_esm_split(self, model: nn.Module, tokens: torch.Tensor,
                            repr_layers: list) -> dict:
        """Layer-by-layer ESM-2 forward with gradient split at the LoRA boundary.

        Layers 0..(n-lora_n_layers-1) run under no_grad; the last lora_n_layers
        run with gradients so LoRA adapters receive signal.  Activation memory
        is only retained for the tail layers.
        """
        n_total   = model.num_layers
        lora_start = max(0, n_total - self.config.lora_n_layers)
        repr_set  = set(repr_layers)

        padding_mask = tokens.eq(model.padding_idx)

        with torch.no_grad():
            x = model.embed_scale * model.embed_tokens(tokens)
            if model.token_dropout:
                x.masked_fill_((tokens == model.mask_idx).unsqueeze(-1), 0.0)
                mask_ratio_train = 0.15 * 0.8
                src_lengths = (~padding_mask).sum(-1)
                mask_ratio_obs = (tokens == model.mask_idx).sum(-1).to(x.dtype) / src_lengths
                x = x * (1 - mask_ratio_train) / (1 - mask_ratio_obs)[:, None, None]
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            pm = padding_mask if padding_mask.any() else None
            x = x.transpose(0, 1)   # (T, B, E)

            # Frozen layers
            for i in range(lora_start):
                x, _ = model.layers[i](x, self_attn_padding_mask=pm)

        # Detach to avoid storing activations for frozen layers
        x = x.detach()

        hidden = {}
        # LoRA layers — gradients flow here, activation checkpointing saves memory
        for i in range(lora_start, n_total):
            layer = model.layers[i]
            # closure captures pm; use_reentrant=False avoids deprecated codepath
            def _layer_fn(x, layer=layer, pm=pm):
                out, _ = layer(x, self_attn_padding_mask=pm)
                return out
            x = checkpoint(_layer_fn, x, use_reentrant=False)
            if (i + 1) in repr_set:
                hidden[i + 1] = x.transpose(0, 1)

        x = model.emb_layer_norm_after(x).transpose(0, 1)  # (B, T, E)
        if n_total in repr_set:
            hidden[n_total] = x

        return {'representations': hidden}

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_tokens: (B, L) amino acid token IDs, pad=ESM2_PAD_TOKEN

        Returns:
            embeddings: (B, L, embed_dim)
        """
        B, L = sequence_tokens.shape
        device = sequence_tokens.device
        model = self._load_esm(device)

        # Prepend <cls>; ESM-2 expects it but we skip <eos> for simplicity
        bos = torch.full((B, 1), ESM2_BOS_TOKEN, dtype=torch.long, device=device)
        tokens = torch.cat([bos, sequence_tokens], dim=1)   # (B, L+1)

        n = model.num_layers
        repr_layers = list(range(n - self.n_layers + 1, n + 1))

        if self.config.lora_rank > 0:
            results = self._forward_esm_split(model, tokens, repr_layers)
        else:
            with torch.no_grad():
                results = model(tokens, repr_layers=repr_layers, return_contacts=False)

        # Strip <cls> from all layer outputs
        layer_reprs = [results['representations'][l][:, 1:, :] for l in repr_layers]

        # Weighted average across layers (softmax-normalized)
        weights = F.softmax(self.layer_weights, dim=0)
        embeddings = sum(w * r for w, r in zip(weights, layer_reprs))  # (B, L, D)

        return embeddings


class DummyBackbone(nn.Module):
    """Random-output backbone for fast architecture testing (no ESM weights needed)."""

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.embed_dim = config.esm_embed_dim

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        B, L = sequence_tokens.shape
        return torch.randn(B, L, self.embed_dim, device=sequence_tokens.device)
