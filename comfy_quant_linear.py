# Derived from PR #2029 in this fork's upstream/original project,
# kijai/ComfyUI-WanVideoWrapper:
# https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/2029
#
# The upstream PR author's copyright remains with that author and the
# ComfyUI-WanVideoWrapper contributors. Modified portions for WanAnimatePlus
# integration are Copyright (c) 2026 wuwukasi/wuwukaka.
#
# Modifications in this fork:
#   - Integrated the loader with the WanAnimatePlus package/module layout.
#   - Routed WanAnimatePlus CustomLinear quantized layers through direct
#     forward/LoRA helpers so ComfyUI QuantizedTensor dispatch remains intact.
#   - Added loader-side guards for LoRA merging and legacy fp8-scaled paths.
#
# Licensed under the Apache License, Version 2.0.

"""Load ComfyUI-native quantized checkpoints in WanAnimatePlus.

ComfyUI native NVFP4/FP8 checkpoints store packed linear weights together with
``*.comfy_quant`` JSON metadata and scale tensors. This module reconstructs those
weights as ComfyUI ``QuantizedTensor`` instances so regular ``F.linear`` dispatches
to comfy_kitchen kernels through the tensor subclass.
"""

import json
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_COMFY_QUANT_IMPORT_ERRORS = []

try:
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class, QuantizedTensor
    _COMFY_QUANT_BACKEND = "comfy.quant_ops"
except Exception as e:
    _COMFY_QUANT_IMPORT_ERRORS.append(f"comfy.quant_ops: {e}")
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import get_layout_class, QuantizedTensor

        QUANT_ALGOS = {
            "float8_e4m3fn": {
                "storage_t": torch.float8_e4m3fn,
                "comfy_tensor_layout": "TensorCoreFP8Layout",
            },
            "float8_e5m2": {
                "storage_t": torch.float8_e5m2,
                "comfy_tensor_layout": "TensorCoreFP8Layout",
            },
            "nvfp4": {
                "storage_t": torch.uint8,
                "comfy_tensor_layout": "TensorCoreNVFP4Layout",
            },
            "mxfp8": {
                "storage_t": torch.float8_e4m3fn,
                "comfy_tensor_layout": "TensorCoreMXFP8Layout",
            },
        }
        _COMFY_QUANT_BACKEND = "comfy_kitchen.tensor"
    except Exception as kitchen_e:
        _COMFY_QUANT_IMPORT_ERRORS.append(f"comfy_kitchen.tensor: {kitchen_e}")
        QUANT_ALGOS = {}
        QuantizedTensor = None
        get_layout_class = None
        ck = None
        _COMFY_QUANT_BACKEND = None


def is_comfy_quant_state_dict(sd) -> bool:
    """Return True when a state dict uses ComfyUI native quant metadata."""
    if sd is None:
        return False
    return any(k.endswith(".comfy_quant") for k in sd)


def _ensure_comfy_quant_backend():
    if _COMFY_QUANT_BACKEND is not None:
        return
    details = "; ".join(_COMFY_QUANT_IMPORT_ERRORS) or "no backend import attempted"
    raise RuntimeError(
        "ComfyUI-native quantized checkpoint detected, but no native quant backend "
        f"is available ({details}). Update ComfyUI/comfy_kitchen or use a non-native "
        "quantized checkpoint."
    )


def _decode_comfy_quant(t: torch.Tensor) -> dict:
    raw = t.to(torch.uint8).cpu().numpy().tobytes()
    return json.loads(bytes(raw).decode("utf-8"))


def _logical_weight_shape(fmt, packed_weight):
    out_features = packed_weight.shape[0]
    in_features = packed_weight.shape[1] * (2 if fmt == "nvfp4" else 1)
    return out_features, in_features


def get_state_dict_weight_shape(sd, weight_key):
    """Return the logical Linear weight shape for regular or Comfy quant weights."""
    if sd is None:
        raise ValueError("state dict is required")
    if not weight_key.endswith("weight"):
        return tuple(sd[weight_key].shape)

    shape = tuple(sd[weight_key].shape)

    quant_key = weight_key[:-len("weight")] + "comfy_quant"
    if quant_key not in sd:
        return shape

    fmt = _decode_comfy_quant(sd[quant_key]).get("format")
    if fmt == "nvfp4":
        return _logical_weight_shape(fmt, sd[weight_key])
    return shape


def _build_quantized_tensor(sd, prefix, device, compute_dtype):
    _ensure_comfy_quant_backend()
    fmt = _decode_comfy_quant(sd[prefix + "comfy_quant"])["format"]
    if fmt not in QUANT_ALGOS:
        raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")
    qcfg = QUANT_ALGOS[fmt]
    layout_name = qcfg["comfy_tensor_layout"]
    layout = get_layout_class(layout_name)

    weight = sd[prefix + "weight"].to(device=device, dtype=qcfg["storage_t"])
    out_features, in_features = get_state_dict_weight_shape(sd, prefix + "weight")

    if fmt == "nvfp4":
        tensor_scale = sd[prefix + "weight_scale_2"].to(device=device)
        block_scale = sd[prefix + "weight_scale"].to(device=device).view(dtype=torch.float8_e4m3fn)
        params = layout.Params(
            scale=tensor_scale,
            block_scale=block_scale,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )
    elif fmt == "mxfp8":
        scale = sd[prefix + "weight_scale"].to(device=device)
        if hasattr(torch, "float8_e8m0fnu"):
            scale = scale.view(dtype=torch.float8_e8m0fnu)
        params = layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )
        tensor_scale = scale
        block_scale = None
    else:
        scale = sd[prefix + "weight_scale"].to(device=device)
        params = layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )
        tensor_scale = scale
        block_scale = None

    return QuantizedTensor(weight, layout_name, params), fmt, layout_name, tensor_scale, block_scale


def get_comfy_quant_metadata(sd, prefix, device, compute_dtype):
    """Return native quant metadata needed by CustomLinear fallback paths."""
    _ensure_comfy_quant_backend()
    fmt = _decode_comfy_quant(sd[prefix + "comfy_quant"])["format"]
    if fmt not in QUANT_ALGOS:
        raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")
    layout_name = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
    weight_shape = tuple(get_state_dict_weight_shape(sd, prefix + "weight"))

    if fmt == "nvfp4":
        tensor_scale = sd[prefix + "weight_scale_2"].to(device=device)
        block_scale = sd[prefix + "weight_scale"].to(device=device).view(dtype=torch.float8_e4m3fn)
    else:
        tensor_scale = sd[prefix + "weight_scale"].to(device=device)
        if fmt == "mxfp8" and hasattr(torch, "float8_e8m0fnu"):
            tensor_scale = tensor_scale.view(dtype=torch.float8_e8m0fnu)
        block_scale = None

    return {
        "format": fmt,
        "layout": layout_name,
        "logical_shape": weight_shape,
        "backend": _COMFY_QUANT_BACKEND,
        "weight_scale": tensor_scale,
        "block_scale": block_scale,
    }


def bind_comfy_quant_metadata(module, metadata):
    """Attach native quant metadata to a Linear/CustomLinear module."""
    module._comfy_quant_format = metadata["format"]
    module._comfy_quant_layout = metadata["layout"]
    module._comfy_quant_logical_shape = tuple(metadata["logical_shape"])
    module._comfy_quant_backend = metadata["backend"]
    tensor_scale = metadata["weight_scale"]
    block_scale = metadata["block_scale"]
    if hasattr(module, "_comfy_quant_weight_scale"):
        module._buffers["_comfy_quant_weight_scale"] = tensor_scale
    else:
        module.register_buffer("_comfy_quant_weight_scale", tensor_scale, persistent=False)
    if block_scale is not None:
        if hasattr(module, "_comfy_quant_block_scale"):
            module._buffers["_comfy_quant_block_scale"] = block_scale
        else:
            module.register_buffer("_comfy_quant_block_scale", block_scale, persistent=False)
    if hasattr(module, "_linear_forward_direct"):
        module._linear_forward_impl = module._linear_forward_direct
        module._apply_lora_impl = module._apply_lora_direct
        module._apply_single_lora_impl = module._apply_single_lora_direct
        module.scale_weight = None
        module.is_gguf = False


def _normalize_metadata_prefix(prefix):
    if prefix.endswith("weight"):
        prefix = prefix[:-len("weight")]
    if prefix and not prefix.endswith("."):
        prefix += "."
    return prefix


def find_comfy_quant_prefix(sd, prefix):
    """Find a state_dict prefix for native quant metadata across known wrappers."""
    if sd is None:
        return None
    prefix = _normalize_metadata_prefix(prefix)
    candidates = []

    def add(candidate):
        candidate = _normalize_metadata_prefix(candidate)
        if candidate not in candidates:
            candidates.append(candidate)

    add(prefix)
    for wrapper in ("diffusion_model.", "model.diffusion_model.", "model."):
        if prefix.startswith(wrapper):
            add(prefix[len(wrapper):])
        else:
            add(wrapper + prefix)

    for candidate in candidates:
        if candidate + "comfy_quant" in sd:
            return candidate
    return None


def find_comfy_quant_prefix_by_shape(sd, weight_shape, logical_shape):
    """Find native quant metadata when module and state_dict names disagree."""
    if sd is None or weight_shape is None or logical_shape is None:
        return None
    weight_shape = tuple(weight_shape)
    logical_shape = tuple(logical_shape)
    for quant_key in sd:
        if not quant_key.endswith(".comfy_quant"):
            continue
        prefix = quant_key[:-len("comfy_quant")]
        weight = sd.get(prefix + "weight")
        if not isinstance(weight, torch.Tensor):
            continue
        if tuple(weight.shape) != weight_shape:
            continue
        try:
            if tuple(get_state_dict_weight_shape(sd, prefix + "weight")) == logical_shape:
                return prefix
        except Exception:
            continue
    return None


def bind_comfy_quant_metadata_for_prefix(module, sd, prefix, compute_dtype, device=torch.device("cpu")):
    """Attach native quant metadata for one module if the state_dict has it."""
    matched_prefix = find_comfy_quant_prefix(sd, prefix)
    if matched_prefix is None:
        return False

    weight = getattr(module, "weight", None)
    metadata_device = device
    if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
        metadata_device = weight.device
    metadata = get_comfy_quant_metadata(sd, matched_prefix, metadata_device, compute_dtype)
    bind_comfy_quant_metadata(module, metadata)

    logical_shape = tuple(metadata["logical_shape"])
    if len(logical_shape) == 2:
        module.out_features, module.in_features = logical_shape
    return True


def bind_comfy_quant_metadata_for_lora(module, sd, prefix, compute_dtype, device, lora_shape=None):
    """Attach native quant metadata for a LoRA target, falling back to shape match."""
    if bind_comfy_quant_metadata_for_prefix(module, sd, prefix, compute_dtype, device):
        return True

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    matched_prefix = find_comfy_quant_prefix_by_shape(sd, tuple(weight.shape), lora_shape)
    if matched_prefix is None:
        return False
    return bind_comfy_quant_metadata_for_prefix(module, sd, matched_prefix, compute_dtype, device)


def rebind_comfy_quant_metadata(model, sd, compute_dtype, device=torch.device("cpu"), prefix=""):
    """Reattach native quant metadata to existing Linear/CustomLinear modules."""
    if sd is None:
        return 0
    if prefix == "":
        _ensure_comfy_quant_backend()

    rebound = 0
    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        rebound += rebind_comfy_quant_metadata(module, sd, compute_dtype, device, child_prefix)

        if not isinstance(module, nn.Linear):
            continue
        if "loras" in child_prefix:
            continue
        if bind_comfy_quant_metadata_for_prefix(module, sd, child_prefix, compute_dtype, device):
            rebound += 1

    return rebound


def _slice_logical_shape(weight, logical_shape):
    if logical_shape is None or len(weight.shape) < 2:
        return weight
    slices = tuple(slice(0, s) for s in logical_shape)
    if tuple(weight.shape[:2]) != tuple(logical_shape):
        weight = weight[slices]
    return weight


def dequantize_comfy_quant_weight(weight, fmt, compute_dtype, scale=None, block_scale=None, logical_shape=None):
    """Dequantize one Comfy native quantized weight tensor for LoRA/fallback paths."""
    if hasattr(weight, "_qdata") and hasattr(weight, "_params") and hasattr(weight, "dequantize"):
        weight = weight.dequantize()
    else:
        try:
            import comfy_kitchen as ck_local
        except Exception as e:
            raise RuntimeError(
                "ComfyUI-native quant weight lost QuantizedTensor dispatch, and "
                f"comfy_kitchen is unavailable for fallback dequantization: {e}"
            ) from e

        if fmt == "nvfp4":
            if scale is None or block_scale is None:
                raise RuntimeError("NVFP4 fallback dequantization requires weight_scale and weight_scale_2")
            weight = ck_local.dequantize_nvfp4(
                weight.to(device=scale.device, dtype=torch.uint8),
                scale,
                block_scale.view(dtype=torch.float8_e4m3fn),
                compute_dtype,
            )
        elif fmt == "mxfp8":
            if scale is None:
                raise RuntimeError("MXFP8 fallback dequantization requires weight_scale")
            weight = ck_local.dequantize_mxfp8(weight.to(device=scale.device), scale, compute_dtype)
        elif fmt in ("float8_e4m3fn", "float8_e5m2"):
            if scale is None:
                raise RuntimeError(f"{fmt} fallback dequantization requires weight_scale")
            weight = ck_local.dequantize_per_tensor_fp8(weight.to(device=scale.device), scale, compute_dtype)
        else:
            raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")

    return _slice_logical_shape(weight, logical_shape)


def replace_with_comfy_quant_linear(model, sd, compute_dtype, load_device, prefix=""):
    """Assign QuantizedTensor weights to matching Linear/CustomLinear modules."""
    if sd is None:
        return model
    if prefix == "":
        _ensure_comfy_quant_backend()

    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        replace_with_comfy_quant_linear(module, sd, compute_dtype, load_device, child_prefix)

        if not isinstance(module, nn.Linear):
            continue
        if "loras" in child_prefix or (child_prefix + "comfy_quant") not in sd:
            continue

        weight_shape = get_state_dict_weight_shape(sd, child_prefix + "weight")
        qt, fmt, layout_name, tensor_scale, block_scale = _build_quantized_tensor(sd, child_prefix, load_device, compute_dtype)
        module.weight = nn.Parameter(qt, requires_grad=False)
        module.out_features, module.in_features = weight_shape

        bias_key = child_prefix + "bias"
        if module.bias is not None and bias_key in sd:
            module.bias = nn.Parameter(sd[bias_key].to(device=load_device, dtype=compute_dtype), requires_grad=False)

        bind_comfy_quant_metadata(module, {
            "format": fmt,
            "layout": layout_name,
            "logical_shape": tuple(weight_shape),
            "backend": _COMFY_QUANT_BACKEND,
            "weight_scale": tensor_scale,
            "block_scale": block_scale,
        })

    return model
