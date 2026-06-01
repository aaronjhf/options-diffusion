"""Shared neural building blocks for diffusion models."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CondBlock(nn.Module):
    """FiLM-conditioned residual block: scale and shift from time + conditioning."""

    def __init__(self, in_dim: int, out_dim: int, t_dim: int, c_dim: int, dropout: float = 0.15):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.t_proj = nn.Linear(t_dim, out_dim * 2)
        self.c_proj = nn.Linear(c_dim, out_dim * 2)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = F.silu(self.ln1(self.lin1(x)))
        gamma_t, beta_t = self.t_proj(t_emb).chunk(2, dim=-1)
        gamma_c, beta_c = self.c_proj(c_emb).chunk(2, dim=-1)
        h = (1 + gamma_t + gamma_c) * h + (beta_t + beta_c)
        h = F.silu(self.drop(self.ln2(self.lin2(h))))
        return h + self.skip(x)
