"""
CNN vision encoder. No pretrained backbones - built from nn.Conv2d only.

Input:  (B, 3, 64, 64)
Output: (B, num_visual_tokens, d_model)

Design (option A - minimal):
  64x64 -> conv stride2 -> 32x32 (32 ch)
  32x32 -> conv stride2 -> 16x16 (64 ch)
  16x16 -> conv stride2 ->  8x8  (128 ch)
  flatten spatial dims -> 64 visual tokens of size 128
  linear "adapter" projects 128 -> d_model
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CNNEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(3, 32),     # (B,3,64,64)  -> (B,32,32,32)
            ConvBlock(32, 64),    # (B,32,32,32) -> (B,64,16,16)
            ConvBlock(64, 128),   # (B,64,16,16) -> (B,128,8,8)
        )
        self.adapter = nn.Linear(128, d_model)   # per-token projection

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, 64, 64)
        feat = self.blocks(images)               # (B, C=128, H'=8, W'=8)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)   # (B, H'*W'=64, C=128)
        visual_tokens = self.adapter(feat)       # (B, 64, d_model)
        return visual_tokens


if __name__ == "__main__":
    enc = CNNEncoder(d_model=128)
    dummy = torch.randn(4, 3, 64, 64)
    out = enc(dummy)
    print("visual tokens shape:", out.shape)   # expect (4, 64, 128)
