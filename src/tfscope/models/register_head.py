"""V19 E5 offset/reverse-complement register head."""

import torch
import torch.nn as nn

from tfscope.config import TFScopeConfig


class RegisterHead(nn.Module):
    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.max_shift = config.registration_max_shift
        self.num_states = 2 * (2 * self.max_shift + 1)
        self.net = nn.Sequential(
            nn.Linear(config.proj_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_states),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
