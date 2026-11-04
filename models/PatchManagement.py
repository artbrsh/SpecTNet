# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# Source: TinyViT/models/tiny_vit.py (PatchEmbed and PatchMerging).
# The source also credits LeViT:
#   (https://github.com/facebookresearch/levit)
# --------------------------------------------------------

import torch

from .low_level import Conv2d_BN
from .Attention import ECA

class PatchEmbed(torch.nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.act = torch.nn.SiLU(inplace=True)
        self.conv1 = Conv2d_BN(in_chans, self.embed_dim // 2, ks=3, stride=2, pad=1)
        self.conv2 = Conv2d_BN(self.embed_dim // 2, self.embed_dim // 2, ks=3, stride=1, pad=1, groups=self.embed_dim // 2)
        self.conv3 = Conv2d_BN(self.embed_dim // 2, self.embed_dim, ks=3, stride=2, pad=1)
        self.conv4 = Conv2d_BN(self.embed_dim, self.embed_dim, ks=3, stride=1, pad=1, groups=self.embed_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        x = x + self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)

        x = x + self.conv4(x)

        return x


class PatchMerging(torch.nn.Module):
    def __init__(self, input_resolution, dim, out_dim, is_flatten=True):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution

        self.proj = torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                Conv2d_BN(dim, out_dim, ks=1, stride=1)
            )

        self.act = torch.nn.SiLU(inplace=True)
        self.conv1 = Conv2d_BN(dim, out_dim * 2, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim * 2, out_dim * 2, 3, 1, (3-1)//2, groups=out_dim * 2)
        self.conv3 = Conv2d_BN(out_dim * 2, out_dim, 1, 1, 0)
        self.conv4 = Conv2d_BN(out_dim, out_dim, 3, 2, (3-1)//2, groups=out_dim)

        self.eca = ECA()

        self.is_flatten = is_flatten

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B, _, _ = x.shape
            # (B, C, H, W)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.act(out)
        out = self.eca(out)
        out = self.conv4(out)

        shortcut = self.proj(x)
        out.add_(shortcut)
        x = out

        if self.is_flatten:
            x = x.flatten(2).transpose(1, 2)
        return x
