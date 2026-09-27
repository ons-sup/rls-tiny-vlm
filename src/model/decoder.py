"""
Decoder block: self-attention + MLP + residuals + LayerNorm, stacked N times.
Pre-norm design (norm before each sub-layer) - trains more stably than
post-norm for small models like this one.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 ffn_dim: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)   # broadcast over batch
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.final_norm(x)
