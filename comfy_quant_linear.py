# Includes native quant loading code adapted in part from PR #2029 in this
# fork's upstream/original project, kijai/ComfyUI-WanVideoWrapper:
# https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/2029
# The repository's MXFP8/block-wise scale_weight support predates the PR #2029
# integration and is a WanAnimatePlus modification.
#
# The upstream PR author's copyright remains with that author and the
# ComfyUI-WanVideoWrapper contributors. Modified portions for WanAnimatePlus
# integration are Copyright (c) 2026 wuwukasi/wuwukaka.
# This file does not claim ownership of ComfyUI QuantizedTensor APIs or the
# upstream PR implementation itself. WanAnimatePlus substantially modified and
# completed the imported PR code for this fork; MXFP8/block-wise scale_weight,
# fallback, materialization, and LoRA integration paths are WanAnimatePlus
# modifications.
#
# Modifications in this fork:
#   - Integrated the loader with the WanAnimatePlus package/module layout.
#   - Routed WanAnimatePlus CustomLinear quantized layers through direct
#     forward/LoRA helpers so ComfyUI QuantizedTensor dispatch remains intact.
#   - Added loader-side guards for LoRA merging and legacy fp8-scaled paths.
#   - Added MXFP8/block-wise scale_weight handling for the WanAnimatePlus
#     CustomLinear path.
#   - Completed/hardened the imported native-quant loading path for this fork's
#     materialization, fallback, and block-swap behavior.
#
# Licensed under the Apache License, Version 2.0.

"""Load ComfyUI-native quantized checkpoints in WanAnimatePlus.

ComfyUI native NVFP4, FP8, MXFP8, and ConvRot checkpoints store packed linear weights
together with ``*.comfy_quant`` JSON metadata and scale tensors. Some NVFP4
checkpoints only store packed uint8 weights plus scale tensors; this module
treats those as native quant weights too. It reconstructs weights as ComfyUI
``QuantizedTensor`` instances so regular ``F.linear`` dispatches to
comfy_kitchen kernels through the tensor subclass.
"""

import json
import logging
import sys

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
_WINDOWS = sys.platform == "win32"


def supports_fp8_compute(device=None) -> bool:
    """Return True if the device supports hardware-accelerated fp8 matmul (CC>=8.9)."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    if props.major >= 9:
        return True
    if props.major < 8 or props.minor < 9:
        return False
    if _torch_version < (2, 3):
        return False
    if _WINDOWS and _torch_version < (2, 4):
        return False
    return True


def supports_nvfp4_compute(device=None) -> bool:
    """Return True if the device supports hardware-accelerated nvfp4 matmul (Blackwell CC>=10)."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return False
    return props.major >= 10

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
            "int8_tensorwise": {
                "storage_t": torch.int8,
                "comfy_tensor_layout": "TensorWiseINT8Layout",
                "quantize_input": False,
            },
            "convrot_w4a4": {
                "storage_t": torch.int8,
                "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
                "quantize_input": False,
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
    if any(k.endswith(".comfy_quant") for k in sd):
        return True
    return any(is_scale_only_nvfp4_weight_key(sd, k) for k in sd)


def _ensure_comfy_quant_backend():
    if _COMFY_QUANT_BACKEND is not None:
        return
    details = "; ".join(_COMFY_QUANT_IMPORT_ERRORS) or "no backend import attempted"
    raise RuntimeError(
        "ComfyUI-native quantized checkpoint detected, but no native quant backend "
        f"is available ({details}). Update ComfyUI/comfy_kitchen or use a non-native "
        "quantized checkpoint."
    )


def _get_layout_class_checked(layout_name, fmt):
    layout = get_layout_class(layout_name) if get_layout_class is not None else None
    if layout is None or not hasattr(layout, "Params"):
        raise RuntimeError(
            f"ComfyUI-native quantization format {fmt} requires layout {layout_name}, "
            "but the running ComfyUI/comfy_kitchen backend does not provide it. "
            "Update ComfyUI/comfy_kitchen or use a checkpoint with a supported quantization format."
        )
    return layout


def _decode_comfy_quant(t: torch.Tensor) -> dict:
    raw = t.to(torch.uint8).cpu().numpy().tobytes()
    return json.loads(bytes(raw).decode("utf-8"))


def _logical_weight_shape(fmt, packed_weight):
    out_features = packed_weight.shape[0]
    in_features = packed_weight.shape[1] * (2 if fmt in ("nvfp4", "convrot_w4a4") else 1)
    return out_features, in_features


def _layer_quant_conf(sd, prefix):
    quant_key = prefix + "comfy_quant"
    if sd is None or quant_key not in sd:
        return {}
    conf = _decode_comfy_quant(sd[quant_key])
    return conf if isinstance(conf, dict) else {}


def _layer_params_conf(layer_conf):
    params_conf = layer_conf.get("params", {})
    return params_conf if isinstance(params_conf, dict) else {}


def _quant_params_for_format(fmt, layer_conf):
    params_conf = _layer_params_conf(layer_conf)
    if fmt == "int8_tensorwise":
        if layer_conf.get("convrot", params_conf.get("convrot", False)):
            return {
                "convrot": True,
                "convrot_groupsize": int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                ),
            }
        return {}
    if fmt == "convrot_w4a4":
        return {
            "convrot_groupsize": int(
                layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
            ),
            "quant_group_size": 64,
            "linear_dtype": str(layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4"))),
        }
    return {}


def _is_weight_key(weight_key):
    return weight_key == "weight" or weight_key.endswith(".weight")


def _prefix_from_weight_key(weight_key):
    if not _is_weight_key(weight_key):
        return None
    return weight_key[:-len("weight")]


def _first_existing_prefixed_tensor(sd, prefix, suffixes):
    for suffix in suffixes:
        key = prefix + suffix
        value = sd.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _get_nvfp4_scale_tensors(sd, prefix):
    if sd is None:
        return None, None
    tensor_scale = _first_existing_prefixed_tensor(sd, prefix, ("weight_scale_2", "scale_weight_2"))
    block_scale = _first_existing_prefixed_tensor(sd, prefix, ("weight_scale", "scale_weight"))
    return tensor_scale, block_scale


def _has_nvfp4_scale_tensors(sd, prefix):
    tensor_scale, block_scale = _get_nvfp4_scale_tensors(sd, prefix)
    return tensor_scale is not None and block_scale is not None


def is_scale_only_nvfp4_prefix(sd, prefix):
    """Return True for NVFP4 packed weights that lack a comfy_quant key."""
    if sd is None or prefix + "comfy_quant" in sd:
        return False
    weight = sd.get(prefix + "weight")
    if not isinstance(weight, torch.Tensor):
        return False
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        return False
    return _has_nvfp4_scale_tensors(sd, prefix)


def is_scale_only_nvfp4_weight_key(sd, weight_key):
    prefix = _prefix_from_weight_key(weight_key)
    return prefix is not None and is_scale_only_nvfp4_prefix(sd, prefix)


def get_comfy_quant_format(sd, prefix):
    if sd is None:
        return None
    quant_key = prefix + "comfy_quant"
    if quant_key in sd:
        return _decode_comfy_quant(sd[quant_key]).get("format")
    if is_scale_only_nvfp4_prefix(sd, prefix):
        return "nvfp4"
    return None


def is_nvfp4_comfy_quant_prefix(sd, prefix):
    return get_comfy_quant_format(sd, prefix) == "nvfp4"


def is_nvfp4_comfy_quant_weight_key(sd, weight_key):
    prefix = _prefix_from_weight_key(weight_key)
    return prefix is not None and is_nvfp4_comfy_quant_prefix(sd, prefix)


def is_nvfp4_comfy_quant_state_dict(sd):
    if sd is None:
        return False
    for key in sd:
        if key.endswith(".comfy_quant"):
            prefix = key[:-len("comfy_quant")]
            if get_comfy_quant_format(sd, prefix) == "nvfp4":
                return True
    return any(is_scale_only_nvfp4_weight_key(sd, k) for k in sd)


def is_native_quant_prefix(sd, prefix):
    if sd is None:
        return False
    return prefix + "comfy_quant" in sd or is_scale_only_nvfp4_prefix(sd, prefix)


def is_native_quant_weight_key(sd, weight_key):
    prefix = _prefix_from_weight_key(weight_key)
    return prefix is not None and is_native_quant_prefix(sd, prefix)


def get_state_dict_weight_shape(sd, weight_key):
    """Return the logical Linear weight shape for regular or Comfy quant weights."""
    if sd is None:
        raise ValueError("state dict is required")
    if not _is_weight_key(weight_key):
        return tuple(sd[weight_key].shape)

    shape = tuple(sd[weight_key].shape)
    prefix = _prefix_from_weight_key(weight_key)

    quant_key = prefix + "comfy_quant"
    if quant_key not in sd:
        if is_scale_only_nvfp4_prefix(sd, prefix):
            return _logical_weight_shape("nvfp4", sd[weight_key])
        return shape

    fmt = _decode_comfy_quant(sd[quant_key]).get("format")
    if fmt in ("nvfp4", "convrot_w4a4"):
        return _logical_weight_shape(fmt, sd[weight_key])
    return shape


def _build_quantized_tensor(sd, prefix, device, compute_dtype):
    _ensure_comfy_quant_backend()
    quant_key = prefix + "comfy_quant"
    if quant_key in sd:
        fmt = _decode_comfy_quant(sd[quant_key])["format"]
    elif is_scale_only_nvfp4_prefix(sd, prefix):
        fmt = "nvfp4"
    else:
        raise RuntimeError(f"No ComfyUI-native quant metadata found for {prefix}")
    if fmt not in QUANT_ALGOS:
        raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")
    qcfg = QUANT_ALGOS[fmt]
    layout_name = qcfg["comfy_tensor_layout"]
    layout = _get_layout_class_checked(layout_name, fmt)
    layer_conf = _layer_quant_conf(sd, prefix)
    quant_params = _quant_params_for_format(fmt, layer_conf)

    weight = sd[prefix + "weight"].to(device=device, dtype=qcfg["storage_t"])
    out_features, in_features = get_state_dict_weight_shape(sd, prefix + "weight")

    if fmt == "nvfp4":
        tensor_scale, block_scale = _get_nvfp4_scale_tensors(sd, prefix)
        if tensor_scale is None or block_scale is None:
            raise RuntimeError(f"NVFP4 native quant metadata is missing scale tensors for {prefix}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.to(device=device).view(dtype=torch.float8_e4m3fn)
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
    elif fmt == "int8_tensorwise":
        scale = sd[prefix + "weight_scale"].to(device=device)
        params = layout.Params(
            scale=scale,
            **quant_params,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )
        tensor_scale = scale
        block_scale = None
    elif fmt == "convrot_w4a4":
        scale = sd[prefix + "weight_scale"].to(device=device)
        params = layout.Params(
            scale=scale,
            **quant_params,
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

    return QuantizedTensor(weight, layout_name, params), fmt, layout_name, tensor_scale, block_scale, quant_params


def get_comfy_quant_metadata(sd, prefix, device, compute_dtype):
    """Return native quant metadata needed by CustomLinear fallback paths."""
    _ensure_comfy_quant_backend()
    quant_key = prefix + "comfy_quant"
    if quant_key in sd:
        fmt = _decode_comfy_quant(sd[quant_key])["format"]
    elif is_scale_only_nvfp4_prefix(sd, prefix):
        metadata = _get_nvfp4_metadata_from_prefix(sd, prefix, device, compute_dtype)
        if metadata is None:
            raise RuntimeError(f"NVFP4 native quant metadata is missing scale tensors for {prefix}")
        return metadata
    else:
        raise RuntimeError(f"No ComfyUI-native quant metadata found for {prefix}")
    if fmt not in QUANT_ALGOS:
        raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")
    layout_name = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
    _get_layout_class_checked(layout_name, fmt)
    layer_conf = _layer_quant_conf(sd, prefix)
    quant_params = _quant_params_for_format(fmt, layer_conf)
    weight_shape = tuple(get_state_dict_weight_shape(sd, prefix + "weight"))

    if fmt == "nvfp4":
        tensor_scale, block_scale = _get_nvfp4_scale_tensors(sd, prefix)
        if tensor_scale is None or block_scale is None:
            raise RuntimeError(f"NVFP4 native quant metadata is missing scale tensors for {prefix}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.to(device=device).view(dtype=torch.float8_e4m3fn)
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
        "quant_params": quant_params,
    }


def bind_comfy_quant_metadata(module, metadata):
    """Attach native quant metadata to a Linear/CustomLinear module."""
    module._comfy_quant_format = metadata["format"]
    module._comfy_quant_layout = metadata["layout"]
    module._comfy_quant_logical_shape = tuple(metadata["logical_shape"])
    module._comfy_quant_backend = metadata["backend"]
    module._comfy_quant_params = dict(metadata.get("quant_params") or {})
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

    for candidate in _metadata_prefix_candidates(prefix):
        if candidate + "comfy_quant" in sd:
            return candidate
    return None


def _metadata_prefix_candidates(prefix):
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
    return candidates


def _get_nvfp4_metadata_from_prefix(sd, prefix, device, compute_dtype, logical_shape=None):
    tensor_scale, block_scale = _get_nvfp4_scale_tensors(sd, prefix)
    if tensor_scale is None or block_scale is None:
        return None
    _ensure_comfy_quant_backend()
    if "nvfp4" not in QUANT_ALGOS:
        return None

    weight = sd.get(prefix + "weight")
    if logical_shape is None:
        if not isinstance(weight, torch.Tensor):
            return None
        logical_shape = _logical_weight_shape("nvfp4", weight)

    return {
        "format": "nvfp4",
        "layout": QUANT_ALGOS["nvfp4"]["comfy_tensor_layout"],
        "logical_shape": tuple(logical_shape),
        "backend": _COMFY_QUANT_BACKEND,
        "weight_scale": tensor_scale.to(device=device),
        "block_scale": block_scale.to(device=device).view(dtype=torch.float8_e4m3fn),
        "quant_params": {},
    }


def find_nvfp4_scale_prefix(sd, prefix):
    """Find an NVFP4 scale prefix when comfy_quant metadata is unavailable."""
    if sd is None:
        return None
    for candidate in _metadata_prefix_candidates(prefix):
        if _has_nvfp4_scale_tensors(sd, candidate):
            return candidate
    return None


def find_nvfp4_scale_prefix_by_shape(sd, weight_shape, logical_shape):
    if sd is None or weight_shape is None or logical_shape is None:
        return None
    weight_shape = tuple(weight_shape)
    logical_shape = tuple(logical_shape)
    matches = []
    for weight_key, weight in sd.items():
        if not weight_key.endswith(".weight") or not isinstance(weight, torch.Tensor):
            continue
        prefix = weight_key[:-len("weight")]
        if not _has_nvfp4_scale_tensors(sd, prefix):
            continue
        if tuple(weight.shape) != weight_shape:
            continue
        if tuple(_logical_weight_shape("nvfp4", weight)) != logical_shape:
            continue
        matches.append(prefix)
    return matches[0] if len(matches) == 1 else None


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
    weight = getattr(module, "weight", None)
    metadata_device = device
    if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
        metadata_device = weight.device

    if matched_prefix is not None:
        metadata = get_comfy_quant_metadata(sd, matched_prefix, metadata_device, compute_dtype)
    else:
        matched_prefix = find_nvfp4_scale_prefix(sd, prefix)
        if matched_prefix is None:
            return False
        logical_shape = None
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.uint8:
            logical_shape = _logical_weight_shape("nvfp4", weight)
        metadata = _get_nvfp4_metadata_from_prefix(sd, matched_prefix, metadata_device, compute_dtype, logical_shape)
        if metadata is None:
            return False

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
    if matched_prefix is not None:
        return bind_comfy_quant_metadata_for_prefix(module, sd, matched_prefix, compute_dtype, device)

    matched_prefix = find_nvfp4_scale_prefix_by_shape(sd, tuple(weight.shape), lora_shape)
    if matched_prefix is None:
        return False
    metadata_device = weight.device if weight.device.type != "meta" else device
    metadata = _get_nvfp4_metadata_from_prefix(sd, matched_prefix, metadata_device, compute_dtype, lora_shape)
    if metadata is None:
        return False
    bind_comfy_quant_metadata(module, metadata)
    logical_shape = tuple(metadata["logical_shape"])
    if len(logical_shape) == 2:
        module.out_features, module.in_features = logical_shape
    return True


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
        elif fmt == "int8_tensorwise":
            if scale is None:
                raise RuntimeError("int8_tensorwise fallback dequantization requires weight_scale")
            if hasattr(ck_local, "dequantize_per_tensor_int8"):
                weight = ck_local.dequantize_per_tensor_int8(
                    weight.to(device=scale.device, dtype=torch.int8), scale, compute_dtype
                )
            else:
                # Manual dequant: int8 * float32 scale → compute_dtype
                weight = weight.to(device=scale.device, dtype=compute_dtype) * scale.to(dtype=compute_dtype)
        elif fmt == "convrot_w4a4":
            raise RuntimeError(
                "convrot_w4a4 fallback dequantization requires QuantizedTensor.dequantize(); "
                "raw storage fallback is not supported by WanAnimatePlus."
            )
        else:
            raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")

    return _slice_logical_shape(weight, logical_shape)


def quantize_raw_comfy_quant_weight(
    weight,
    fmt,
    compute_dtype,
    scale=None,
    block_scale=None,
    logical_shape=None,
    layout_name=None,
    quant_params=None,
):
    """Wrap a raw native quantized storage tensor as a QuantizedTensor."""
    _ensure_comfy_quant_backend()
    if fmt not in QUANT_ALGOS:
        raise RuntimeError(f"Unsupported ComfyUI-native quantization format: {fmt}")
    if logical_shape is None:
        logical_shape = _logical_weight_shape(fmt, weight)
    layout_name = layout_name or QUANT_ALGOS[fmt]["comfy_tensor_layout"]
    layout = _get_layout_class_checked(layout_name, fmt)
    quant_params = dict(quant_params or {})

    qcfg = QUANT_ALGOS[fmt]
    weight = weight.to(dtype=qcfg["storage_t"])

    if fmt == "nvfp4":
        if scale is None or block_scale is None:
            raise RuntimeError("NVFP4 QuantizedTensor rebuild requires weight_scale and weight_scale_2")
        params = layout.Params(
            scale=scale,
            block_scale=block_scale.view(dtype=torch.float8_e4m3fn),
            orig_dtype=compute_dtype,
            orig_shape=tuple(logical_shape),
        )
    elif fmt == "mxfp8":
        if scale is None:
            raise RuntimeError("MXFP8 QuantizedTensor rebuild requires weight_scale")
        if hasattr(torch, "float8_e8m0fnu"):
            scale = scale.view(dtype=torch.float8_e8m0fnu)
        params = layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(logical_shape),
        )
    elif fmt == "int8_tensorwise":
        if scale is None:
            raise RuntimeError("int8_tensorwise QuantizedTensor rebuild requires weight_scale")
        params = layout.Params(
            scale=scale,
            **quant_params,
            orig_dtype=compute_dtype,
            orig_shape=tuple(logical_shape),
        )
    elif fmt == "convrot_w4a4":
        if scale is None:
            raise RuntimeError("convrot_w4a4 QuantizedTensor rebuild requires weight_scale")
        params = layout.Params(
            scale=scale,
            **quant_params,
            orig_dtype=compute_dtype,
            orig_shape=tuple(logical_shape),
        )
    else:
        if scale is None:
            raise RuntimeError(f"{fmt} QuantizedTensor rebuild requires weight_scale")
        params = layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(logical_shape),
        )

    return QuantizedTensor(weight, layout_name, params)


def contains_meta_tensor(value, _seen=None):
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return False
    _seen.add(value_id)

    if isinstance(value, torch.Tensor):
        return value.device.type == "meta"
    if isinstance(value, (tuple, list)):
        return any(contains_meta_tensor(item, _seen) for item in value)
    if isinstance(value, dict):
        return any(contains_meta_tensor(item, _seen) for item in value.values())
    if hasattr(value, "_asdict"):
        return any(contains_meta_tensor(item, _seen) for item in value._asdict().values())
    if hasattr(value, "__dict__") and not isinstance(value, nn.Module):
        return any(contains_meta_tensor(item, _seen) for item in vars(value).values())
    return False


def is_comfy_quant_weight_materialized(module):
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if weight.device.type == "meta":
        return False
    if not (hasattr(weight, "_qdata") and hasattr(weight, "_params")):
        return False
    if contains_meta_tensor(getattr(weight, "_qdata", None)):
        return False
    if contains_meta_tensor(getattr(weight, "_params", None)):
        return False
    if contains_meta_tensor(getattr(module, "_comfy_quant_weight_scale", None)):
        return False
    if contains_meta_tensor(getattr(module, "_comfy_quant_block_scale", None)):
        return False

    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor) and bias.device.type == "meta":
        return False
    return True


def has_comfy_quant_meta_weights(model, sd, prefix=""):
    return ensure_comfy_quant_linear_materialized(
        model,
        sd,
        compute_dtype=None,
        load_device=None,
        prefix=prefix,
        dry_run=True,
    ) > 0


def ensure_comfy_quant_linear_materialized(model, sd, compute_dtype, load_device, prefix="", dry_run=False, device_resolver=None):
    """Rebuild missing/meta native quant Linear weights from a state_dict."""
    if sd is None:
        return 0
    if prefix == "":
        _ensure_comfy_quant_backend()

    rebuilt = 0
    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        rebuilt += ensure_comfy_quant_linear_materialized(
            module,
            sd,
            compute_dtype,
            load_device,
            child_prefix,
            dry_run=dry_run,
            device_resolver=device_resolver,
        )
        if dry_run and rebuilt:
            return rebuilt

        if not isinstance(module, nn.Linear):
            continue
        if "loras" in child_prefix or not is_native_quant_prefix(sd, child_prefix):
            continue
        if dry_run:
            if is_comfy_quant_weight_materialized(module):
                continue
            return rebuilt + 1

        resolved_device = device_resolver(child_prefix) if device_resolver is not None else None
        materialize_device = resolved_device if resolved_device is not None else load_device
        if is_comfy_quant_weight_materialized(module):
            if resolved_device is not None:
                needs_move = getattr(module.weight, "device", None) != materialize_device
                for attr in ("_comfy_quant_weight_scale", "_comfy_quant_block_scale"):
                    tensor = getattr(module, attr, None)
                    if isinstance(tensor, torch.Tensor) and tensor.device != materialize_device:
                        needs_move = True
                        break
                if needs_move:
                    module.to(materialize_device)
            continue

        if compute_dtype is None or materialize_device is None:
            raise RuntimeError("ComfyUI-native quantized Linear materialization requires dtype and device")

        weight_shape = get_state_dict_weight_shape(sd, child_prefix + "weight")
        qt, fmt, layout_name, tensor_scale, block_scale, quant_params = _build_quantized_tensor(
            sd, child_prefix, materialize_device, compute_dtype
        )
        module.weight = nn.Parameter(qt, requires_grad=False)
        module.out_features, module.in_features = weight_shape

        bias_key = child_prefix + "bias"
        if module.bias is not None and bias_key in sd:
            module.bias = nn.Parameter(sd[bias_key].to(device=materialize_device, dtype=compute_dtype), requires_grad=False)

        bind_comfy_quant_metadata(module, {
            "format": fmt,
            "layout": layout_name,
            "logical_shape": tuple(weight_shape),
            "backend": _COMFY_QUANT_BACKEND,
            "weight_scale": tensor_scale,
            "block_scale": block_scale,
            "quant_params": quant_params,
        })
        rebuilt += 1

    return rebuilt


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
        if "loras" in child_prefix or not is_native_quant_prefix(sd, child_prefix):
            continue

        weight_shape = get_state_dict_weight_shape(sd, child_prefix + "weight")
        qt, fmt, layout_name, tensor_scale, block_scale, quant_params = _build_quantized_tensor(
            sd, child_prefix, load_device, compute_dtype
        )
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
            "quant_params": quant_params,
        })

    return model
