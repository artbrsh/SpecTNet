# Copyright (c) 2022 Microsoft
# Licensed under the MIT License (see LICENSE).

import torch
import torch.utils.checkpoint as checkpoint

from .low_level import (
    DropPath,
    fuse_bn_before_conv2d,
    _align_conv_weight_singleton_strides_,
)


def _make_activation(act_ctor):
    try:
        return act_ctor(inplace=True)
    except TypeError:
        return act_ctor()



class DWSpatialMLPMixerBlock(torch.nn.Module):
    """
    Mixer block where spatial mixing is a depthwise conv.
    Input/output: (B, C, H, W)
    """

    def __init__(
        self,
        channels,
        mlp_ratio=2.0,
        dw_kernel_size=3,
        dw_stride=1,
        dw_padding=None,
        use_pw_after_dw=False,
        dropout=0.0,
        drop_path=0.0,
        act_layer=torch.nn.GELU,
        norm_layer=torch.nn.BatchNorm2d,
    ):
        super().__init__()
        self._bn_pre_fused = False

        if dw_padding is None:
            dw_padding = dw_kernel_size // 2

        # Spatial mixing (DWConv)
        self.norm1 = norm_layer(channels)
        self.dwconv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=dw_kernel_size,
            stride=dw_stride,
            padding=dw_padding,
            groups=channels,
            bias=False,
        )
        _align_conv_weight_singleton_strides_(self.dwconv)

        self.pw_after_dw = None
        if use_pw_after_dw:
            self.pw_after_dw = torch.nn.Conv2d(
                channels, channels, kernel_size=1, bias=False
            )
            _align_conv_weight_singleton_strides_(self.pw_after_dw)

        self.spatial_act = _make_activation(act_layer)
        self.spatial_drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else torch.nn.Identity()

        # Channel mixing (MLP/FFN)
        self.norm2 = norm_layer(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, kernel_size=1),
            _make_activation(act_layer),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Conv2d(hidden, channels, kernel_size=1),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
        )
        _align_conv_weight_singleton_strides_(self.mlp[0])
        _align_conv_weight_singleton_strides_(self.mlp[3])

    @torch.no_grad()
    def fuse_bn_preceding_conv_inplace(self):
        if self._bn_pre_fused:
            return self

        # Keep norm1: zero padding after BN cannot be folded into a constant
        # convolution bias without changing the outputs at the spatial borders.

        if isinstance(self.norm2, torch.nn.BatchNorm2d) and isinstance(self.mlp[0], torch.nn.Conv2d):
            self.mlp[0] = fuse_bn_before_conv2d(self.norm2, self.mlp[0])
            self.norm2 = torch.nn.Identity()

        self._bn_pre_fused = True
        return self

    def forward(self, x):
        # Spatial mixing branch
        y = self.norm1(x)
        y = self.dwconv(y)
        if self.pw_after_dw is not None:
            y = self.pw_after_dw(y)
        y = self.spatial_act(y)
        y = self.spatial_drop(y)
        x = x + self.drop_path(y)

        # Channel mixing branch
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)
        return x


class ConvMixer(torch.nn.Module):
    """
    Stage without self-attention, using DWSpatialMLPMixerBlock.
    Accepts tokens (B, L, C) and returns feature maps (B, C, H, W).
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth=2,
        mlp_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        use_pw_after_dw=False,
        norm_layer=torch.nn.BatchNorm2d,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.downsample = None

        self.blocks = torch.nn.ModuleList([
            DWSpatialMLPMixerBlock(
                channels=dim,
                mlp_ratio=mlp_ratio,
                dropout=drop,
                use_pw_after_dw=use_pw_after_dw,
                norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        if (self.training or torch.is_grad_enabled()) and (L != H * W):
            raise ValueError("input feature has wrong size")

        # Keep standard contiguous NCHW to avoid DDP grad/bucket stride mismatch warnings.
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        if self.use_checkpoint and self.training:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        return x
