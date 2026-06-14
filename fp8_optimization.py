# Copyright (c) 2025 kijai
# Modified from fp8_optimization.py in ComfyUI-WanVideoWrapper.
# Original project: https://github.com/kijai/ComfyUI-WanVideoWrapper
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Added MXFP8 block-scaled quantization support with comfy-kitchen integration,
#     including block-scaled matmul forward, runtime effective-weight
#     requantization helpers for unmerged LoRA, pure-PyTorch per-layer
#     dequantization fallback, and auto-detection of float8_e8m0fnu block scales.
# Licensed under the Apache License, Version 2.0
from contextlib import nullcontext

import torch
import torch.nn as nn
from .utils import log


def _ceil_div(a, b):
    return (a + b - 1) // b


def _from_blocked(blocked_matrix, num_rows, num_cols):
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def _e8m0_to_f32(x):
    biased_exp = x.to(torch.int32)
    result = biased_exp << 23
    result = torch.where(biased_exp == 0, torch.zeros_like(result), result)
    return result.view(torch.float32)

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
    from comfy_kitchen.registry import registry as ck_registry
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
    ck_registry = None

_MXFP8_BACKEND_LOGGED = False
_MXFP8_FASTPATH_DISABLED = False
_MXFP8_FASTPATH_DISABLE_REASON = None


def _get_mxfp8_backend_context():
    global _MXFP8_BACKEND_LOGGED
    if (
        _CK_MXFP8 and
        ck_registry is not None and
        ck_registry.is_available("eager") and
        hasattr(torch.nn.functional, "scaled_mm")
    ):
        if not _MXFP8_BACKEND_LOGGED:
            log.info("MXFP8 using comfy-kitchen eager backend")
            _MXFP8_BACKEND_LOGGED = True
        return ck_registry.use_backend("eager")
    return nullcontext()


def mxfp8_fastpath_enabled():
    return _CK_MXFP8 and not _MXFP8_FASTPATH_DISABLED


def disable_mxfp8_fastpath(reason):
    global _MXFP8_FASTPATH_DISABLED, _MXFP8_FASTPATH_DISABLE_REASON
    if _MXFP8_FASTPATH_DISABLED:
        return
    _MXFP8_FASTPATH_DISABLED = True
    _MXFP8_FASTPATH_DISABLE_REASON = str(reason)
    log.warning(f"Disabling MXFP8 fast path for this session, using dequantized fallback: {reason}")


def dequantize_mxfp8_weight(weight_fp8, block_scale_e8m0):
    """Dequantize MXFP8 block-scaled weight to bf16."""
    if block_scale_e8m0.device != weight_fp8.device:
        block_scale_e8m0 = block_scale_e8m0.to(weight_fp8.device)
    # Normalize: safetensors stores E8M0 as uint8
    if block_scale_e8m0.dtype == torch.uint8:
        block_scale_e8m0 = block_scale_e8m0.view(torch.float8_e8m0fnu)
    if _CK_MXFP8:
        with _get_mxfp8_backend_context():
            return ck.dequantize_mxfp8(weight_fp8, block_scale_e8m0, torch.bfloat16)

    # Manual dequant when comfy-kitchen is not available.
    assert weight_fp8.shape[-1] % 32 == 0, \
        f"MXFP8 weight must be 32-aligned, got {weight_fp8.shape[-1]}"
    num_rows = weight_fp8.shape[0]
    num_cols = weight_fp8.shape[-1] // 32
    scale_u8 = block_scale_e8m0.view(torch.uint8)
    logical_u8 = _from_blocked(scale_u8, num_rows, num_cols)
    scale_f32 = _e8m0_to_f32(logical_u8)
    scale_expanded = scale_f32.repeat_interleave(32, dim=-1)[:, :weight_fp8.shape[-1]]
    return (weight_fp8.float() * scale_expanded).to(torch.bfloat16)


def quantize_mxfp8_weight_like(weight, reference_block_scale):
    if not mxfp8_fastpath_enabled():
        raise RuntimeError("comfy-kitchen MXFP8 quantization is unavailable")
    with _get_mxfp8_backend_context():
        q_weight, q_scale = ck.quantize_mxfp8(weight.to(torch.bfloat16), pad_32x=True)
    if q_scale.dtype == torch.uint8:
        q_scale = q_scale.view(torch.float8_e8m0fnu)
    if reference_block_scale is not None and q_scale.shape != reference_block_scale.shape:
        raise RuntimeError(
            f"Quantized MXFP8 scale shape {q_scale.shape} does not match reference scale shape {reference_block_scale.shape}"
        )
    return q_weight, q_scale


def run_mxfp8_linear_kernel(cls, input, bias, weight=None, block_scale=None, out_dtype=None):
    if not mxfp8_fastpath_enabled():
        raise RuntimeError("comfy-kitchen MXFP8 kernel is unavailable")
    effective_weight = weight if weight is not None else cls.weight
    effective_block_scale = block_scale if block_scale is not None else cls.block_scale_weight
    if len(input.shape) != 3:
        effective_weight = dequantize_mxfp8_weight(effective_weight, effective_block_scale).to(
            device=input.device, dtype=out_dtype or input.dtype
        )
        effective_bias = bias.to(device=input.device, dtype=out_dtype or input.dtype) if bias is not None else None
        return torch.nn.functional.linear(input.to(out_dtype or input.dtype), effective_weight, effective_bias)

    input_shape = input.shape
    inp_2d = input.reshape(-1, input_shape[2]).to(out_dtype or bias.dtype if bias is not None else torch.bfloat16)
    orig_rows = inp_2d.shape[0]
    orig_cols = effective_weight.shape[0]
    q_weight = effective_weight.to(input.device)
    q_scale = effective_block_scale.to(input.device)
    bias = bias.to(device=input.device, dtype=out_dtype or inp_2d.dtype) if bias is not None else None
    try:
        with _get_mxfp8_backend_context():
            inp_fp8, inp_block_scale = ck.quantize_mxfp8(inp_2d, pad_32x=True)
            o = ck.scaled_mm_mxfp8(
                inp_fp8, q_weight, inp_block_scale, q_scale,
                bias=bias, out_dtype=out_dtype or inp_2d.dtype,
            )
    except Exception as e:
        if isinstance(e, RuntimeError) and 'out of memory' in str(e).lower():
            raise
        disable_mxfp8_fastpath(e)
        raise
    if o.shape[0] != orig_rows or o.shape[1] != orig_cols:
        o = o[:orig_rows, :orig_cols]
    return o.reshape((-1, input_shape[1], orig_cols))


def mxfp8_linear_forward(cls, base_dtype, input):
    """MXFP8 block-scaled linear forward, counterpart to fp8_linear_forward.

    Equivalent to Bernini's QuantizedTensor.__torch_dispatch__ path for MXFP8:
    both activation and weight are block-quantized via ck.quantize_mxfp8,
    then ck.scaled_mm_mxfp8 performs the block-scaled matmul.
    """
    if cls.weight.dtype == torch.float8_e4m3fn and hasattr(cls, 'block_scale_weight'):
        if len(input.shape) == 3:
            if mxfp8_fastpath_enabled():
                try:
                    return run_mxfp8_linear_kernel(cls, input, cls.bias, out_dtype=base_dtype)
                except Exception as e:
                    if isinstance(e, RuntimeError) and 'out of memory' in str(e).lower():
                        raise
                    if not getattr(cls, '_mxfp8_fallback_warned', False):
                        log.warning(
                            f"MXFP8 fast path failed, falling back to dequant+F.linear: {e}"
                        )
                        cls._mxfp8_fallback_warned = True
        elif hasattr(cls, 'original_forward'):
            weight = dequantize_mxfp8_weight(cls.weight, cls.block_scale_weight).to(device=input.device, dtype=base_dtype)
            bias = cls.bias.to(device=input.device, dtype=base_dtype) if cls.bias is not None else None
            return torch.nn.functional.linear(input.to(base_dtype), weight, bias)

    # safety: fp8 weight must have a block scale; refuse to guess
    if cls.weight.dtype == torch.float8_e4m3fn and not hasattr(cls, 'block_scale_weight'):
        raise RuntimeError(
            f"Module has float8_e4m3fn weight but no MXFP8 block_scale_weight. "
            f"The model is partially quantized or the MXFP8 setup is incomplete."
        )

    weight = cls.weight
    if cls.weight.dtype == torch.float8_e4m3fn and hasattr(cls, 'block_scale_weight'):
        weight = dequantize_mxfp8_weight(cls.weight, cls.block_scale_weight)
    weight = weight.to(device=input.device, dtype=base_dtype)
    bias = cls.bias.to(device=input.device, dtype=base_dtype) if cls.bias is not None else None
    out = torch.nn.functional.linear(input.to(base_dtype), weight, bias)
    del weight, bias
    return out


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
