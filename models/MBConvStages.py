# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# Source: TinyViT/models/tiny_vit.py (MBConv and ConvLayer).
# The source also credits LeViT:
#   (https://github.com/facebookresearch/levit)
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .low_level import Conv2d_BN, DropPath


def _make_activation(act_ctor):
    try:
        return act_ctor(inplace=True)
    except TypeError:
        return act_ctor()


class SpatialStage1Block(torch.nn.Module):
    def __init__(self, channels, activation, expand_ratio, drop_path=0.):
        super().__init__()

        self.dw3 = Conv2d_BN(
            channels, channels, ks=3, stride=1, pad=(3 - 1) // 2, groups=channels
        )
        self.dw5 = Conv2d_BN(
            channels, channels, ks=5, stride=1, pad=(5 - 1) // 2, groups=channels
        )
        self.dw7 = Conv2d_BN(
            channels, channels, ks=7, stride=1, pad=(7 - 1) // 2, groups=channels
        )
        self.dw_merged = None

        hidden_channels = int(channels * expand_ratio)
        self.ffn = torch.nn.Sequential(
            Conv2d_BN(channels, hidden_channels, ks=1),
            _make_activation(activation),
            Conv2d_BN(hidden_channels, channels, ks=1, bn_weight_init=0.0),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else torch.nn.Identity()

    @torch.no_grad()
    def merge_spatial_branches(self):
        """Replace the three evaluation branches with an equivalent depthwise 7x7."""
        if self.training:
            raise ValueError('Stage 1 merging requires evaluation mode')
        if self.dw_merged is not None:
            return self

        branches = (self.dw3, self.dw5, self.dw7)
        if any(isinstance(branch, Conv2d_BN) and branch.bn.training for branch in branches):
            raise ValueError('Stage 1 BatchNorm must be in evaluation mode before merging')
        # Also accept branches already folded by fuse_for_inference().
        conv3, conv5, conv7 = [
            branch.fuse() if isinstance(branch, Conv2d_BN) else branch
            for branch in branches
        ]
        merged = torch.nn.Conv2d(
            conv7.in_channels, conv7.out_channels, kernel_size=7, padding=3,
            groups=conv7.groups, bias=True,
            device=conv7.weight.device, dtype=conv7.weight.dtype,
        )
        merged.weight.copy_(F.pad(conv3.weight, (2, 2, 2, 2))
                            + F.pad(conv5.weight, (1, 1, 1, 1)) + conv7.weight)
        merged.bias.copy_(conv3.bias + conv5.bias + conv7.bias)
        self.dw_merged = merged.eval().requires_grad_(False)
        del self.dw3, self.dw5, self.dw7
        return self

    def forward(self, x):
        if self.dw_merged is None:
            spatial = self.dw3(x) + self.dw5(x) + self.dw7(x)
        else:
            if self.training:
                raise RuntimeError('Merged stage 1 blocks cannot be used for training')
            spatial = self.dw_merged(x)

        out = x + self.drop_path(self.ffn(spatial))
        return out


class SpatialStage1Stage(torch.nn.Module):
    def __init__(self, dim, input_resolution, depth, expand_ratio,
                 activation=torch.nn.GELU,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 is_flatten=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = torch.nn.ModuleList([
            SpatialStage1Block(dim, activation, expand_ratio,
                               drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, is_flatten=is_flatten)
        else:
            self.downsample = None

    def forward(self, x):
        if x.ndim == 3:
            B, _, _ = x.shape
            H, W = self.input_resolution
            x = x.transpose(1, 2).reshape(B, self.dim, H, W)

        if self.use_checkpoint and self.training:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MBConv(torch.nn.Module):
    def __init__(self, in_chans, out_chans, k_size, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = _make_activation(activation)

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=k_size, stride=1, pad=(k_size-1)//2, groups=self.hidden_chans)
        self.act2 = _make_activation(activation)

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = _make_activation(activation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else torch.nn.Identity()

    def forward(self, x):
        # print('MBConv x', x.shape)
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class MBConvStage(torch.nn.Module):
    def __init__(self, dim, k_size, input_resolution, depth,
                 activation= torch.nn.GELU,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4., is_flatten=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = torch.nn.ModuleList([
            MBConv(dim, dim, k_size, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, is_flatten=is_flatten)
        else:
            self.downsample = None

    def forward(self, x):
        if x.ndim == 3:
            B, _, _ = x.shape
            H, W = self.input_resolution
            x = x.transpose(1, 2).reshape(B, self.dim, H, W)

        if self.use_checkpoint and self.training:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x
