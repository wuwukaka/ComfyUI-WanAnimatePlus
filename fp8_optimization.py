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
    # Refuse MXFP8 modules — block-scaled weights need ck.scaled_mm_mxfp8,
    # not per-tensor torch._scaled_mm.  Running _scaled_mm with scale_b=1.0
    # on block-quantized weights would produce silently wrong output.
    if weight_dtype == torch.float8_e4m3fn and hasattr(cls, 'block_scale_weight'):
        raise RuntimeError(
            "fp8_linear_forward called on MXFP8 block-scaled module. "
            "The model was incorrectly patched — use mxfp8 mode instead."
        )
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
                if hasattr(submodule, 'original_forward'):
                    continue  # already patched
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
    _CK_MXFP8 = (
        hasattr(ck, 'scaled_mm_mxfp8') and
        hasattr(ck, 'quantize_mxfp8') and
        hasattr(ck, 'dequantize_mxfp8') and
        torch.cuda.is_available() and
        torch.cuda.get_device_capability(0)[0] >= 10
    )
except ImportError:
    _CK_MXFP8 = False
    ck = None


def dequantize_mxfp8_weight(weight_fp8, block_scale_e8m0):
    """Dequantize MXFP8 block-scaled weight to bf16."""
    # Normalize: safetensors stores E8M0 as uint8
    if block_scale_e8m0.dtype == torch.uint8:
        block_scale_e8m0 = block_scale_e8m0.view(torch.float8_e8m0fnu)
    if _CK_MXFP8:
        return ck.dequantize_mxfp8(weight_fp8, block_scale_e8m0, torch.bfloat16)

    # Manual dequant when comfy-kitchen is not available.
    assert weight_fp8.shape[-1] % 32 == 0, \
        f"MXFP8 weight must be 32-aligned, got {weight_fp8.shape[-1]}"
    num_rows = weight_fp8.shape[0]
    num_cols = weight_fp8.shape[-1] // 32
    if ck is None:
        raise RuntimeError(
            "Cannot dequantize MXFP8 weights without comfy-kitchen. "
            "Please install it: pip install comfy-kitchen==0.2.10"
        )
    # Canonical E8M0 conversion: view as uint8, unswizzle, then e8m0_to_f32
    scale_u8 = block_scale_e8m0.view(torch.uint8)
    logical_u8 = ck.from_blocked(scale_u8, num_rows, num_cols)
    if hasattr(ck, 'float_utils'):
        scale_f32 = ck.float_utils.e8m0_to_f32(logical_u8)
    else:
        # Manual E8M0→float32: biased exponent in bits [0:8], shift to float32
        # exponent bits, reinterpret as float32.  Zero maps to 0.0.
        biased = logical_u8.to(torch.int32)
        scale_f32 = (biased << 23).view(torch.float32)
        scale_f32 = torch.where(logical_u8 == 0, torch.zeros_like(scale_f32), scale_f32)
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
            orig_rows, orig_cols = inp_2d.shape[0], cls.weight.shape[0]
            try:
                # Block-quantize activation (same layout as weight), matching Bernini
                inp_fp8, inp_block_scale = ck.quantize_mxfp8(inp_2d, pad_32x=True)
                block_scale = cls.block_scale_weight.to(input.device)
                weight = cls.weight.to(input.device)
                bias = cls.bias.to(device=input.device, dtype=base_dtype) if cls.bias is not None else None
                o = ck.scaled_mm_mxfp8(
                    inp_fp8, weight, inp_block_scale, block_scale,
                    bias=bias, out_dtype=base_dtype,
                )
                # ck.scaled_mm_mxfp8 may pad output to 32-alignment; slice back
                if o.shape[0] != orig_rows or o.shape[1] != orig_cols:
                    o = o[:orig_rows, :orig_cols]
                return o.reshape((-1, input_shape[1], orig_cols))
            except Exception as e:
                # Re-raise OOM immediately — the fallback allocates 2x more
                # memory (bf16 dequant) and would cause a cascading OOM.
                if isinstance(e, RuntimeError) and 'out of memory' in str(e).lower():
                    raise
                # comfy-kitchen kernel failed for non-OOM reasons — cache a
                # dequantized bf16 copy so subsequent calls skip re-dequant.
                if not getattr(cls, '_mxfp8_fallback_warned', False):
                    log.warning(
                        f"ck.scaled_mm_mxfp8 failed, falling back to dequant+F.linear: {e}"
                    )
                    cls._mxfp8_fallback_warned = True
                if getattr(cls, '_mxfp8_dequant_cache', None) is None:
                    cls._mxfp8_dequant_cache = dequantize_mxfp8_weight(
                        cls.weight, cls.block_scale_weight,
                    )
        else:
            return cls.original_forward(input.to(base_dtype))

    # safety: fp8 weight must have a block scale; refuse to guess
    if cls.weight.dtype == torch.float8_e4m3fn and not hasattr(cls, 'block_scale_weight'):
        raise RuntimeError(
            f"Module has float8_e4m3fn weight but no MXFP8 block_scale_weight. "
            f"The model is partially quantized or the MXFP8 setup is incomplete."
        )

    # fallback: use cached dequantized weight or compute on first call
    weight = getattr(cls, '_mxfp8_dequant_cache', None)
    if weight is None and cls.weight.dtype == torch.float8_e4m3fn and hasattr(cls, 'block_scale_weight'):
        weight = dequantize_mxfp8_weight(cls.weight, cls.block_scale_weight)
        cls._mxfp8_dequant_cache = weight
    if weight is None:
        weight = cls.weight
    weight = weight.to(device=input.device, dtype=base_dtype)
    bias = cls.bias.to(device=input.device, dtype=base_dtype) if cls.bias is not None else None
    return torch.nn.functional.linear(input.to(base_dtype), weight, bias)


def convert_mxfp8_linear(module, base_dtype, params_to_keep, block_scale_keys):
    """Replace nn.Linear forward with MXFP8 matmul, counterpart to convert_fp8_linear."""
    patched_count = 0
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                scale_key = f"{name}.scale_weight"
                if scale_key in block_scale_keys:
                    # Weights may be on meta device (init_empty_weights); dtype
                    # check is deferred to forward time where weights are real.
                    if submodule.weight.device.type != 'meta' and submodule.weight.dtype != torch.float8_e4m3fn:
                        raise RuntimeError(
                            f"Module '{name}' has weight dtype {submodule.weight.dtype} "
                            f"but an E8M0 block scale exists. "
                            f"MXFP8 requires float8_e4m3fn weights."
                        )
                    if hasattr(submodule, 'original_forward'):
                        continue  # already patched, prevent double-patching
                    # safetensors stores E8M0 as uint8; ck.scaled_mm_mxfp8 needs e8m0fnu
                    _bs = block_scale_keys[scale_key].clone()
                    if _bs.dtype == torch.uint8:
                        _bs = _bs.view(torch.float8_e8m0fnu)
                    submodule.register_buffer(
                        "block_scale_weight", _bs, persistent=True,
                    )
                    original_forward = submodule.forward
                    setattr(submodule, "original_forward", original_forward)
                    setattr(submodule, "forward", lambda input, m=submodule: mxfp8_linear_forward(m, base_dtype, input))
                    patched_count += 1
    log.info(f"MXFP8 matmul enabled ({patched_count} layers)")
