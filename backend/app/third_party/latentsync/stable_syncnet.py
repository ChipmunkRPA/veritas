# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""StableSyncNet from LatentSync (github.com/bytedance/LatentSync, commit a229c3948406bc2cf6eaf4873e662e70c6a04746):
`latentsync/models/stable_syncnet.py`, the `Attention` class of `latentsync/models/attention.py`, and the architecture of
`configs/syncnet/syncnet_16_pixel_attn.yaml`. Apache-2.0; the licence text is in LICENSE next to this file.

Changes from upstream (Veritas, 2026-09-13): diffusers' `FeedForward` with the GEGLU activation is re-implemented here
with the same parameter names (`net.0.proj`, `net.2`), so the checkpoint loads strictly without diffusers; gradient
checkpointing and the unused attention-mask path are removed; einops is replaced by tensor reshapes; the model config is
a constant instead of an OmegaConf file.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SYNCNET_16_PIXEL_ATTN = {
    "audio_encoder": {"in_channels": 1, "block_out_channels": [32, 64, 128, 256, 512, 1024, 2048],
                      "downsample_factors": [[2, 1], 2, 2, 1, 2, 2, [2, 3]], "attn_blocks": [0, 0, 0, 1, 1, 0, 0], "dropout": 0.0},
    "visual_encoder": {"in_channels": 48, "block_out_channels": [64, 128, 256, 256, 512, 1024, 2048, 2048],
                       "downsample_factors": [[1, 2], 2, 2, 2, 2, 2, 2, 2], "attn_blocks": [0, 0, 0, 0, 1, 1, 0, 0], "dropout": 0.0},
}


class StableSyncNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = DownEncoder2D(**{k: config["audio_encoder"][k] for k in
                                              ("in_channels", "block_out_channels", "downsample_factors", "dropout", "attn_blocks")})
        self.visual_encoder = DownEncoder2D(**{k: config["visual_encoder"][k] for k in
                                               ("in_channels", "block_out_channels", "downsample_factors", "dropout", "attn_blocks")})
        self.eval()

    def forward(self, image_sequences, audio_sequences):
        vision_embeds = self.visual_encoder(image_sequences)  # (b, c, 1, 1)
        audio_embeds = self.audio_encoder(audio_sequences)  # (b, c, 1, 1)
        vision_embeds = vision_embeds.reshape(vision_embeds.shape[0], -1)
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)
        return F.normalize(vision_embeds, p=2, dim=1), F.normalize(audio_embeds, p=2, dim=1)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, norm_num_groups=32, eps=1e-6, act_fn="silu", downsample_factor=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act_fn = nn.ReLU() if act_fn == "relu" else nn.SiLU()
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
        if isinstance(downsample_factor, list):
            downsample_factor = tuple(downsample_factor)
        if downsample_factor == 1:
            self.downsample_conv = None
        else:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=0)
            self.pad = (0, 1, 0, 1)
            if isinstance(downsample_factor, tuple):
                if downsample_factor[0] == 1:
                    self.pad = (0, 1, 1, 1)  # the padding order is from back to front
                elif downsample_factor[1] == 1:
                    self.pad = (1, 1, 0, 1)

    def forward(self, input_tensor):
        hidden_states = self.conv1(self.act_fn(self.norm1(input_tensor)))
        hidden_states = self.conv2(self.dropout(self.act_fn(self.norm2(hidden_states))))
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        hidden_states = hidden_states + input_tensor
        if self.downsample_conv is not None:
            hidden_states = self.downsample_conv(F.pad(hidden_states, self.pad, mode="constant", value=0))
        return hidden_states


class GEGLU(nn.Module):
    """diffusers `GEGLU`: a linear projection to twice the width, split into value and GELU gate."""

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    """diffusers `FeedForward(dim, dropout=dropout, activation_fn="geglu")`: mult 4, output width = dim."""

    def __init__(self, dim, dropout=0.0, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def split_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        return tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query = self.split_heads(self.to_q(hidden_states))
        key = self.split_heads(self.to_k(hidden_states))
        value = self.split_heads(self.to_v(hidden_states))
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        batch_size, heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, heads * head_dim)
        return self.to_out[1](self.to_out[0](hidden_states))


class AttentionBlock2D(nn.Module):
    def __init__(self, query_dim, norm_num_groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, dropout=dropout)
        self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.attn = Attention(query_dim=query_dim, heads=8, dim_head=query_dim // 8, dropout=dropout, bias=True)

    def forward(self, hidden_states):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.conv_in(self.norm1(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channel)  # b c h w -> b (h w) c
        hidden_states = self.attn(self.norm2(hidden_states)) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        hidden_states = hidden_states.reshape(batch, height, width, channel).permute(0, 3, 1, 2).contiguous()
        return self.conv_out(hidden_states) + residual


class DownEncoder2D(nn.Module):
    def __init__(self, in_channels=4 * 16, block_out_channels=(64, 128, 256, 256), downsample_factors=(2, 2, 2, 2),
                 norm_num_groups=32, attn_blocks=(1, 1, 1, 1), dropout=0.0, act_fn="silu"):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])
        output_channels = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channels, output_channels = output_channels, block_out_channel
            self.down_blocks.append(ResnetBlock2D(in_channels=input_channels, out_channels=output_channels,
                                                  downsample_factor=downsample_factors[i], norm_num_groups=norm_num_groups,
                                                  dropout=dropout, act_fn=act_fn))
            if attn_blocks[i] == 1:
                self.down_blocks.append(AttentionBlock2D(query_dim=output_channels, dropout=dropout))
        self.norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.act_fn_out = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        return self.act_fn_out(self.norm_out(hidden_states))
