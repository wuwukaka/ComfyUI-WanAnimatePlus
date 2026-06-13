# Copyright (c) 2025 kijai
# Modified from fp8_optimization.py in ComfyUI-WanVideoWrapper.
# Original project: https://github.com/kijai/ComfyUI-WanVideoWrapper
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Added MXFP8 block-scaled quantization support with comfy-kitchen integration,
#     including block-scaled matmul forward, load-time dequantization fallback,
#     and auto-detection of float8_e8m0fnu block scales.
# Licensed under the Apache License, Version 2.0
import torch
import torch.nn as nn
from .utils import log

#based on ComfyUI's and MinusZoneAI's fp8_linear optimization
def fp8_linear_forward(cls, base_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            input_shape = input.shape

            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device).squeeze()

            scale_input = torch.ones((), device=input.device, dtype=torch.float32)

            input = torch.clamp(input, min=-448, max=448, out=input)
            inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous() #always e4m3fn because e5m2 * e5m2 is not supported

            bias = cls.bias.to(base_dtype) if cls.bias is not None else None

            o = torch._scaled_mm(inn, cls.weight.t(), out_dtype=base_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)

            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(base_dtype))
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, base_dtype, params_to_keep={}, scale_weight_keys=None):
    log.info("FP8 matmul enabled")
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                if scale_weight_keys is not None:
                    scale_key = f"{name}.scale_weight"
                    if scale_key in scale_weight_keys:
                        setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))


# ==============================================================================
# MXFP8 block-scaled quantization support (requires comfy-kitchen)
# ==============================================================================

try:
    import comfy_kitchen as ck
    _CK_MXFP8 = hasattr(ck, 'scaled_mm_mxfp8')
except ImportError:
    _CK_MXFP8 = False
    ck = None


def dequantize_mxfp8_weight(weight_fp8, block_scale_e8m0):
    """Dequantize MXFP8 block-scaled weight to bf16.

    weight_fp8:     (out, in) float8_e4m3fn
    block_scale_e8m0: (out, in//32) float8_e8m0fnu in cuBLAS swizzled layout

    Returns: (out, in) torch.bfloat16
    """
    scale_f32 = block_scale_e8m0.float()
    scale_expanded = scale_f32.repeat_interleave(32, dim=-1)[:, :weight_fp8.shape[-1]]
    return (weight_fp8.float() * scale_expanded).to(torch.bfloat16)


def mxfp8_linear_forward(cls, base_dtype, input):
    """MXFP8 block-scaled linear forward, counterpart to fp8_linear_forward.

    Equivalent to Bernini's QuantizedTensor.__torch_dispatch__ path for MXFP8:
    both activation and weight are block-quantized via ck.quantize_mxfp8,
    then ck.scaled_mm_mxfp8 performs the block-scaled matmul.
    """
    if cls.weight.dtype == torch.float8_e4m3fn and _CK_MXFP8 and hasattr(cls, 'block_scale_weight'):
        if len(input.shape) == 3:
            input_shape = input.shape
            inp_2d = input.reshape(-1, input_shape[2]).to(base_dtype)
            # Block-quantize activation (same layout as weight), matching Bernini
            inp_fp8, inp_block_scale = ck.quantize_mxfp8(inp_2d, pad_32x=True)
            block_scale = cls.block_scale_weight.to(input.device)
            bias = cls.bias.to(base_dtype) if cls.bias is not None else None
            o = ck.scaled_mm_mxfp8(
                inp_fp8, cls.weight, inp_block_scale, block_scale,
                bias=bias, out_dtype=base_dtype,
            )
            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(base_dtype))

    # fallback: dequantize weight and do regular matmul
    weight = cls.weight
    if cls.weight.dtype == torch.float8_e4m3fn and hasattr(cls, 'block_scale_weight'):
        weight = dequantize_mxfp8_weight(cls.weight, cls.block_scale_weight)
    weight = weight.to(device=input.device, dtype=base_dtype)
    bias = cls.bias.to(base_dtype) if cls.bias is not None else None
    return torch.nn.functional.linear(input.to(base_dtype), weight, bias)


def convert_mxfp8_linear(module, base_dtype, params_to_keep, block_scale_keys):
    """Replace nn.Linear forward with MXFP8 matmul, counterpart to convert_fp8_linear."""
    log.info("MXFP8 matmul enabled")
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                scale_key = f"{name}.scale_weight"
                if scale_key in block_scale_keys:
                    setattr(submodule, "block_scale_weight", block_scale_keys[scale_key])
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: mxfp8_linear_forward(m, base_dtype, input))
