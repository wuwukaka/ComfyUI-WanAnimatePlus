# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================

from typing import  Optional
import torch
import torch.nn.functional as F
from torch import nn

import comfy.ops
ops = comfy.ops.disable_weight_init

def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float(
        "-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out,  kernel_size, stride = 1, dilation = 1, pad_mode='replicate', **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding
        self.conv = ops.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class DownsampleCausal3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv", kernel_size=3, bias=True, stride=2):
        super().__init__()
        self.channels, self.out_channels, self.use_conv, self.padding, self.name = channels, out_channels or channels, use_conv, padding, name
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x, scale=1.0):
        return self.conv(x)


class ResnetBlockCausal3D(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, groups: int = 32, eps: float = 1e-6, conv_3d_out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1)

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        hidden_states = input_tensor
        hidden_states = self.conv1(nn.SiLU()(self.norm1(hidden_states)))
        if temb is not None:
            hidden_states = hidden_states + temb
        hidden_states = self.conv2(nn.SiLU()(self.norm2(hidden_states)))
        return input_tensor + hidden_states


def get_down_block3d(down_block_type: str, num_layers: int, in_channels: int, out_channels: int,
    add_downsample: bool, downsample_stride: int, resnet_eps: float, resnet_act_fn: str, resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None, **kwargs):

    down_block_type = down_block_type[7:] if down_block_type.startswith(
        "UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            downsample_stride=downsample_stride,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )
    raise ValueError(f"{down_block_type} does not exist.")


class DownEncoderBlockCausal3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 1, resnet_eps: float = 1e-6,
        resnet_groups: int = 32, add_downsample: bool = True, downsample_stride: int = 2, downsample_padding: int = 1, **kwargs):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([DownsampleCausal3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        stride=downsample_stride,
                )])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states
