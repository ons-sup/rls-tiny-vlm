"""
Required correctness test: your hand-written attention must match
F.scaled_dot_product_attention on random inputs with the same mask,
to about 1e-5 in float32.

Run: python -m tests.test_attention
"""

import torch
import torch.nn.functional as F

from src.model.attention import MultiHeadSelfAttention, build_prefix_mask


def test_attention_matches_sdpa():
    torch.manual_seed(0)
    B, T, D, H = 2, 12, 32, 4
    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    mha.eval()

    x = torch.randn(B, T, D)
    mask = build_prefix_mask(num_visual=7, num_letters=5)   # (T, T) bool

    # --- your implementation ---
    with torch.no_grad():
        your_out = mha(x, mask)

    # --- reference: reuse the SAME learned projections, just recompute the
    # attention math itself with the built-in fused kernel ---
    with torch.no_grad():
        q = mha.q_proj(x).view(B, T, H, D // H).transpose(1, 2)
        k = mha.k_proj(x).view(B, T, H, D // H).transpose(1, 2)
        v = mha.v_proj(x).view(B, T, H, D // H).transpose(1, 2)
        # SDPA expects an additive mask or bool mask where True = keep
        ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        ref_out = ref_out.transpose(1, 2).contiguous().view(B, T, D)
        ref_out = mha.out_proj(ref_out)

    max_diff = (your_out - ref_out).abs().max().item()
    print("max abs diff:", max_diff)
    assert torch.allclose(your_out, ref_out, atol=1e-5), f"mismatch: {max_diff}"
    print("PASSED: hand-written attention matches F.scaled_dot_product_attention")


if __name__ == "__main__":
    test_attention_matches_sdpa()
