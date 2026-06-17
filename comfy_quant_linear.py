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

try:
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class, QuantizedTensor
    _COMFY_QUANT_AVAILABLE = True
except Exception as e:
    _COMFY_QUANT_AVAILABLE = False
    QuantizedTensor = None
    logging.getLogger(__name__).warning(f"comfy.quant_ops unavailable, native NVFP4/FP8 support off: {e}")


def is_comfy_quant_state_dict(sd) -> bool:
    """Return True when a state dict uses ComfyUI native quant metadata."""
    if not _COMFY_QUANT_AVAILABLE or sd is None:
        return False
    return any(k.endswith(".comfy_quant") for k in sd)


def _decode_comfy_quant(t: torch.Tensor) -> dict:
    raw = t.to(torch.uint8).cpu().numpy().tobytes()
    return json.loads(bytes(raw).decode("utf-8"))


def _logical_weight_shape(fmt, packed_weight):
    out_features = packed_weight.shape[0]
    in_features = packed_weight.shape[1] * (2 if fmt == "nvfp4" else 1)
    return out_features, in_features


def _build_quantized_tensor(sd, prefix, device, compute_dtype):
    fmt = _decode_comfy_quant(sd[prefix + "comfy_quant"])["format"]
    qcfg = QUANT_ALGOS[fmt]
    layout_name = qcfg["comfy_tensor_layout"]
    layout = get_layout_class(layout_name)

    weight = sd[prefix + "weight"].to(device=device, dtype=qcfg["storage_t"])
    out_features, in_features = _logical_weight_shape(fmt, weight)

    if fmt == "nvfp4":
        tensor_scale = sd[prefix + "weight_scale_2"].to(device=device)
        block_scale = sd[prefix + "weight_scale"].to(device=device).view(dtype=torch.float8_e4m3fn)
        params = layout.Params(
            scale=tensor_scale,
            block_scale=block_scale,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )
    else:
        scale = sd[prefix + "weight_scale"].to(device=device)
        params = layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=(out_features, in_features),
        )

    return QuantizedTensor(weight, layout_name, params), fmt


def replace_with_comfy_quant_linear(model, sd, compute_dtype, load_device, prefix=""):
    """Assign QuantizedTensor weights to matching Linear/CustomLinear modules."""
    if not _COMFY_QUANT_AVAILABLE or sd is None:
        return model

    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        replace_with_comfy_quant_linear(module, sd, compute_dtype, load_device, child_prefix)

        if not isinstance(module, nn.Linear):
            continue
        if "loras" in child_prefix or (child_prefix + "comfy_quant") not in sd:
            continue

        qt, fmt = _build_quantized_tensor(sd, child_prefix, load_device, compute_dtype)
        module.weight = nn.Parameter(qt, requires_grad=False)

        bias_key = child_prefix + "bias"
        if module.bias is not None and bias_key in sd:
            module.bias = nn.Parameter(sd[bias_key].to(device=load_device, dtype=compute_dtype), requires_grad=False)

        module._comfy_quant_format = fmt
        if hasattr(module, "_linear_forward_direct"):
            module._linear_forward_impl = module._linear_forward_direct
            module._apply_lora_impl = module._apply_lora_direct
            module._apply_single_lora_impl = module._apply_single_lora_direct
            module.scale_weight = None
            module.is_gguf = False

    return model
