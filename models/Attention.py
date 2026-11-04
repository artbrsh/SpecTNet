# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted from TinyViT
# Source: TinyViT/models/tiny_vit.py (Attention).
# Relative-position biases and their evaluation cache: LeViT via TinyViT.
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# Apache-2.0; see licenses/LeViT-Apache-2.0.txt.
# Modified for SpecTNet; see THIRD_PARTY_NOTICES.md.
# --------------------------------------------------------

import torch
import itertools
import torch.nn.functional as F


class ECA(torch.nn.Module):
    def __init__(self, k_size=3):
        super(ECA, self).__init__()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).contiguous()
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y.sigmoid_()

        return x.mul(y.expand_as(x))


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, h)
        self.attn_output_gate = torch.nn.Linear(dim, self.dh)
        self.proj = torch.nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))

        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))

        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)
        self.ab_fp16 = None
        self.ab_bf16 = None
        self.attn_mask = None
        self.attn_mask_fp16 = None
        self.attn_mask_bf16 = None

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
            self.ab_fp16 = None
            self.ab_bf16 = None
            self.attn_mask = None
            self.attn_mask_fp16 = None
            self.attn_mask_bf16 = None
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            self.ab_fp16 = None
            self.ab_bf16 = None
            self.attn_mask = None
            self.attn_mask_fp16 = None
            self.attn_mask_bf16 = None
        return self

    def _get_eval_bias(self, q: torch.Tensor) -> torch.Tensor:
        rel_bias = self.ab
        if rel_bias.device != q.device:
            rel_bias = rel_bias.to(device=q.device)
            self.ab = rel_bias
            self.ab_fp16 = None
            self.ab_bf16 = None
            self.attn_mask = None
            self.attn_mask_fp16 = None
            self.attn_mask_bf16 = None

        if rel_bias.dtype == q.dtype:
            return rel_bias
        if q.dtype == torch.float16:
            if self.ab_fp16 is None or self.ab_fp16.device != q.device:
                self.ab_fp16 = rel_bias.to(dtype=torch.float16)
            return self.ab_fp16
        if q.dtype == torch.bfloat16:
            if self.ab_bf16 is None or self.ab_bf16.device != q.device:
                self.ab_bf16 = rel_bias.to(dtype=torch.bfloat16)
            return self.ab_bf16
        return rel_bias.to(dtype=q.dtype)

    def _get_eval_attn_mask(self, rel_bias: torch.Tensor) -> torch.Tensor:
        if rel_bias.dtype == torch.float16:
            if self.attn_mask_fp16 is None or self.attn_mask_fp16.device != rel_bias.device:
                self.attn_mask_fp16 = rel_bias.unsqueeze(0)
            return self.attn_mask_fp16
        if rel_bias.dtype == torch.bfloat16:
            if self.attn_mask_bf16 is None or self.attn_mask_bf16.device != rel_bias.device:
                self.attn_mask_bf16 = rel_bias.unsqueeze(0)
            return self.attn_mask_bf16
        if self.attn_mask is None or self.attn_mask.device != rel_bias.device:
            self.attn_mask = rel_bias.unsqueeze(0)
        return self.attn_mask

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)
        gate_score = self.attn_output_gate(x).view(B, N, self.num_heads, self.d)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)

        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.training:
            rel_bias = self.attention_biases[:, self.attention_bias_idxs]
            if rel_bias.device != q.device or rel_bias.dtype != q.dtype:
                rel_bias = rel_bias.to(dtype=q.dtype, device=q.device)
        else:
            rel_bias = self._get_eval_bias(q)

        use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if use_sdpa:
            if self.training:
                attn_mask = rel_bias.unsqueeze(0)
            else:
                attn_mask = self._get_eval_attn_mask(rel_bias)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            ).transpose(1, 2).contiguous()
            x = x * torch.sigmoid(gate_score)
            x = x.reshape(B, N, self.dh)
            x = self.proj(x)
            return x

        attn = q @ k.transpose(-2, -1)
        attn.mul_(self.scale)
        attn.add_(rel_bias)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).contiguous()
        x = x * torch.sigmoid(gate_score)
        x = x.reshape(B, N, self.dh)

        x = self.proj(x)

        return x
