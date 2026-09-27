"""
Multi-head self-attention, written by hand with tensor ops.
No nn.MultiheadAttention, no F.scaled_dot_product_attention here
(that function is only allowed in tests/benchmarks - see tests/test_attention.py).
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, T, d_model)
        attn_mask: (T, T) bool, True = allowed to attend, False = blocked
        returns:   (B, T, d_model)
        """
        B, T, D = x.shape
        H, Dh = self.num_heads, self.d_head

        q = self.q_proj(x)   # (B, T, D)
        k = self.k_proj(x)   # (B, T, D)
        v = self.v_proj(x)   # (B, T, D)

        # split heads: (B, T, H, Dh) -> (B, H, T, Dh)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        # scaled dot-product scores: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)

        # apply mask: set blocked positions to -inf before softmax
        # attn_mask is (T, T) -> broadcast over (B, H, T, T)
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)   # (B, H, T, T)
        weights = self.dropout(weights)

        out = weights @ v                          # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)   # merge heads
        return self.out_proj(out)


def build_prefix_mask(num_visual: int, num_letters: int) -> torch.Tensor:
    """
    Prefix-LM mask over a sequence of length T = num_visual + num_letters.
    Rule: position i may attend to position j if
        j < num_visual         (visual tokens are always fully visible)
     OR j <= i                 (causal among the rest)
    This makes the visual block bidirectional among itself, and the
    letter block causal but always able to see the whole image.
    Returns a (T, T) bool tensor, True = allowed.
    """
    T = num_visual + num_letters
    idx = torch.arange(T)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)          # (T,T) j <= i
    visible_visual = idx.unsqueeze(0) < num_visual         # (T,T) j < num_visual
    mask = causal | visible_visual
    return mask


if __name__ == "__main__":
    mha = MultiHeadSelfAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 10, 128)
    mask = build_prefix_mask(num_visual=6, num_letters=4)
    out = mha(x, mask)
    print("output shape:", out.shape)   # expect (2, 10, 128)
