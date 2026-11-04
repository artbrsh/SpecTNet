# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# Source: TinyViT/models/tiny_vit.py (Conv2d_BN, DropPath and Mlp).
# Conv2d_BN and fuse(): adapted from LeViT via TinyViT.
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# Apache-2.0; see licenses/LeViT-Apache-2.0.txt.
# Modified for SpecTNet; see THIRD_PARTY_NOTICES.md.
# --------------------------------------------------------

import torch

from timm.models.layers import DropPath as TimmDropPath


def _align_conv_weight_singleton_strides_(conv: torch.nn.Conv2d):
    """Align strides on size-1 dims with common backend grad layouts for DDP."""
    if not isinstance(conv, torch.nn.Conv2d):
        return conv
    w = conv.weight.data
    if w.ndim != 4:
        return conv
    stride = list(w.stride())
    changed = False
    if w.shape[1] == 1 and stride[1] != 1:
        stride[1] = 1
        changed = True
    if w.shape[2] == 1 and stride[2] != stride[0]:
        stride[2] = stride[0]
        changed = True
    if w.shape[3] == 1 and stride[3] != stride[0]:
        stride[3] = stride[0]
        changed = True
    if changed:
        conv.weight.data = w.as_strided(w.shape, tuple(stride))
    return conv


@torch.no_grad()
def fuse_bn_before_conv2d(bn: torch.nn.BatchNorm2d, conv: torch.nn.Conv2d) -> torch.nn.Conv2d:
    """Fold an evaluation BatchNorm2d into a following unpadded Conv2d."""
    if not isinstance(bn, torch.nn.BatchNorm2d):
        raise TypeError(f"Expected BatchNorm2d, got {type(bn).__name__}")
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError(f"Expected Conv2d, got {type(conv).__name__}")
    if bn.num_features != conv.in_channels:
        raise ValueError(
            f"BN num_features ({bn.num_features}) must match Conv2d in_channels ({conv.in_channels})"
        )
    if bn.training:
        raise ValueError('BatchNorm must be in evaluation mode before fusion')
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError('BatchNorm fusion requires stored running statistics')
    if bn.weight is None or bn.bias is None:
        raise ValueError('BatchNorm fusion requires affine weight and bias')
    if conv.padding not in ((0, 0), 'valid'):
        raise ValueError('BN-before-convolution fusion requires zero spatial padding')

    device = conv.weight.device
    dtype = conv.weight.dtype

    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).to(device=device, dtype=dtype)
    shift = (bn.bias - bn.running_mean * bn.weight /
             torch.sqrt(bn.running_var + bn.eps)).to(device=device, dtype=dtype)

    w = conv.weight
    b0 = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=device, dtype=dtype)

    w_fused = w.clone()
    b_add = torch.empty(conv.out_channels, device=device, dtype=dtype)

    if conv.groups == 1:
        w_fused = w * scale.view(1, -1, 1, 1)
        b_add = (w * shift.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    else:
        in_per_group = conv.in_channels // conv.groups
        out_per_group = conv.out_channels // conv.groups
        for g in range(conv.groups):
            in_slice = slice(g * in_per_group, (g + 1) * in_per_group)
            out_slice = slice(g * out_per_group, (g + 1) * out_per_group)
            w_group = w[out_slice]
            scale_group = scale[in_slice].view(1, in_per_group, 1, 1)
            shift_group = shift[in_slice].view(1, in_per_group, 1, 1)
            w_fused[out_slice] = w_group * scale_group
            b_add[out_slice] = (w_group * shift_group).sum(dim=(1, 2, 3))

    b_fused = b0 + b_add

    fused = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=device, dtype=dtype)
    fused.weight.data.copy_(w_fused)
    fused.bias.data.copy_(b_fused)
    return fused


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        conv = torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        _align_conv_weight_singleton_strides_(conv)
        self.add_module('c', conv)
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m = m.to(device=c.weight.device, dtype=c.weight.dtype)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


def _make_activation(act_ctor):
    try:
        return act_ctor(inplace=True)
    except TypeError:
        return act_ctor()


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = torch.nn.LayerNorm(in_features)
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.act = _make_activation(act_layer)
        self.drop = torch.nn.Dropout(drop, inplace=True) if drop > 0 else torch.nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def fuse_conv2d_bn_inplace(module: torch.nn.Module):
    """In-place fusion of Conv2d_BN blocks for inference latency/throughput."""
    for name, child in list(module.named_children()):
        if isinstance(child, Conv2d_BN):
            module._modules[name] = child.fuse()
        else:
            fuse_conv2d_bn_inplace(child)
    return module
