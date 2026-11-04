# --------------------------------------------------------
# SpecTNet Model Architecture
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# The source also credits LeViT:
#   (https://github.com/facebookresearch/levit)
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


from .PatchManagement import PatchEmbed, PatchMerging
from .MBConvStages import MBConvStage, SpatialStage1Stage
from .ViTStages import ViTStage
from .ConvMixer import ConvMixer
from .low_level import fuse_conv2d_bn_inplace


class SpecTNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 expand_ratios=(3.0, 4.0, 4.0, 4.0),
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        if len(expand_ratios) != self.num_layers or len(embed_dims) != self.num_layers:
            raise ValueError('EXPAND_RATIOS and EMBED_DIMS must have one value per stage in DEPTHS')
        for stage, (dim, ratio) in enumerate(zip(embed_dims, expand_ratios), start=1):
            if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                    or not math.isfinite(ratio) or ratio <= 0 or int(dim * ratio) < 1):
                raise ValueError(f'Invalid expansion ratio for S{stage}: {ratio}; '
                                 'expected a finite positive ratio producing at least one hidden channel')

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.patches_resolution = (img_size // 4, img_size // 4)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample_cls = PatchMerging
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                            self.patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=downsample_cls if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          )
            if i_layer == 0:
                layer = SpatialStage1Stage(
                    **kwargs, expand_ratio=expand_ratios[i_layer], is_flatten=False
                )
            elif i_layer == 1:
                layer = MBConvStage(k_size=5, conv_expand_ratio=expand_ratios[i_layer],
                    **kwargs, is_flatten=True
                )
            elif i_layer == self.num_layers - 1:
                layer = ConvMixer(
                    dim=kwargs["dim"],
                    input_resolution=kwargs["input_resolution"],
                    depth=kwargs["depth"],
                    mlp_ratio=expand_ratios[i_layer],
                    drop=drop_rate,
                    drop_path=kwargs["drop_path"],
                    use_checkpoint=use_checkpoint,
                    use_pw_after_dw=False,
                )
            else:
                layer = ViTStage(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=expand_ratios[i_layer],
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self._is_fused_for_inference = False

    @torch.no_grad()
    def merge_stage1_for_inference(self):
        """Merge S1 spatial branches in place after loading an unfused checkpoint."""
        if self.training:
            raise ValueError('Stage 1 merging requires evaluation mode')
        for block in self.layers[0].blocks:
            block.merge_spatial_branches()
        return self

    @torch.no_grad()
    def fuse_for_inference(self):
        if self._is_fused_for_inference:
            return self
        fuse_conv2d_bn_inplace(self)
        for module in self.modules():
            fuse_fn = getattr(module, "fuse_bn_preceding_conv_inplace", None)
            if callable(fuse_fn):
                fuse_fn()
        self._is_fused_for_inference = True
        return self

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            layer_scale = lr_scales[min(i, depth - 1)]
            layer.apply(lambda x, s=layer_scale: _set_lr_scale(x, s))
            for block in layer.blocks:
                block_scale = lr_scales[i]
                block.apply(lambda x, s=block_scale: _set_lr_scale(x, s))
                i += 1
            if layer.downsample is not None:
                downsample_scale = lr_scales[i - 1]
                layer.downsample.apply(
                    lambda x, s=downsample_scale: _set_lr_scale(x, s))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        if x.ndim == 4:
            x = x.mean(dim=(2, 3))
        else:
            x = x.mean(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm_head(x)
        x = self.head(x)
        return x
