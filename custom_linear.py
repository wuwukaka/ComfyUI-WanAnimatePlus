# Modified from custom_linear.py in ComfyUI-WanVideoWrapper.
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Renamed custom torch ops from wanvideo::* to wananimateplus::* to avoid collisions.
#   - Added guarded custom op registration for duplicate imports/stale bytecode.
#   - Added explicit CUDA implementations for the WanAnimatePlus custom ops.
#   - Added MXFP8/block-wise scale_weight expansion before linear forward,
#     before this fork integrated PR #2029.
#   - Later integrated ComfyUI native quantized weight passthrough adapted in
#     part from PR #2029 in this fork's upstream/original project,
#     kijai/ComfyUI-WanVideoWrapper:
#     https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/2029
#     Upstream PR/native-quant logic remains with its original author(s).
#     WanAnimatePlus changes include the earlier MXFP8/block-wise scale_weight
#     path plus later completion, fallback, materialization, and LoRA integration
#     work for this fork.
# Licensed under the Apache License, Version 2.0
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from .comfy_quant_linear import (
    bind_comfy_quant_metadata_for_lora,
    bind_comfy_quant_metadata_for_prefix,
    contains_meta_tensor,
    dequantize_comfy_quant_weight,
    get_state_dict_weight_shape,
    quantize_raw_comfy_quant_weight,
    supports_fp8_compute,
    supports_nvfp4_compute,
)
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor

if not hasattr(torch.ops.wananimateplus, 'apply_lora'):
    @torch.library.custom_op("wananimateplus::apply_lora", mutates_args=())
    def apply_lora(weight: torch.Tensor, lora_diff_0: torch.Tensor, lora_diff_1: torch.Tensor, lora_diff_2: float, lora_strength: torch.Tensor) -> torch.Tensor:
        patch_diff = torch.mm(
            lora_diff_0.flatten(start_dim=1),
            lora_diff_1.flatten(start_dim=1)
        ).reshape(weight.shape)

        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        scale = lora_strength * alpha

        return weight + patch_diff * scale

    @apply_lora.register_fake
    def _(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        return weight.clone()

if not hasattr(torch.ops.wananimateplus, 'apply_single_lora'):
    @torch.library.custom_op("wananimateplus::apply_single_lora", mutates_args=())
    def apply_single_lora(weight: torch.Tensor, lora_diff: torch.Tensor, lora_strength: torch.Tensor) -> torch.Tensor:
        return weight + lora_diff * lora_strength

    @apply_single_lora.register_fake
    def _(weight, lora_diff, lora_strength):
        return weight.clone()

if not hasattr(torch.ops.wananimateplus, 'linear_forward'):
    @torch.library.custom_op("wananimateplus::linear_forward", mutates_args=())
    def linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return torch.nn.functional.linear(input, weight, bias)

    @linear_forward.register_fake
    def _(input, weight, bias):
        output_shape = list(input.shape[:-1]) + [weight.shape[0]]
        return input.new_empty(output_shape)

# Register CUDA backends for all three ops.  These are intentionally placed
# *outside* the hasattr guards above so that they will be registered even
# when the op already existed from a prior import (e.g. a stale .pyc).
try:
    @torch.library.impl("wananimateplus::apply_lora", "CUDA")
    def _apply_lora_cuda(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        patch_diff = torch.mm(lora_diff_0.flatten(start_dim=1), lora_diff_1.flatten(start_dim=1)).reshape(weight.shape)
        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        return weight + patch_diff * lora_strength * alpha
except RuntimeError:
    pass

try:
    @torch.library.impl("wananimateplus::apply_single_lora", "CUDA")
    def _apply_single_lora_cuda(weight, lora_diff, lora_strength):
        return weight + lora_diff * lora_strength
except RuntimeError:
    pass

try:
    @torch.library.impl("wananimateplus::linear_forward", "CUDA")
    def _linear_forward_cuda(input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)
except RuntimeError:
    pass

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None, modules_to_not_convert=[]):

    has_children = list(model.children())
    if not has_children:
        return

    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args, modules_to_not_convert)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix and "dual_controller" not in module_prefix and name not in modules_to_not_convert:
            weight_key = module_prefix + "weight"
            if weight_key not in state_dict:
                continue

            out_features, in_features = get_state_dict_weight_shape(state_dict, weight_key)

            is_gguf = isinstance(state_dict[weight_key], GGUFParameter)

            scale_weight = None
            if not is_gguf and scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"
                scale_weight = scale_weights.get(scale_key)

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weight,
                    allow_compile=allow_compile,
                    is_gguf=is_gguf
                )
            model._modules[name].source_cls = type(module)
            model._modules[name]._wan_layer_prefix = module_prefix
            model._modules[name].requires_grad_(False)
            if not is_gguf:
                bind_comfy_quant_metadata_for_prefix(
                    model._modules[name],
                    state_dict,
                    module_prefix,
                    compute_dtype,
                    torch.device("cpu"),
                )

    return model

def _lora_diff_shape(diff):
    if not isinstance(diff, (tuple, list)) or len(diff) <= 1:
        return None
    return (
        diff[0].flatten(start_dim=1).shape[0],
        diff[1].flatten(start_dim=1).shape[1],
    )


def _force_direct_lora_if_needed(module, state_dict=None, compute_dtype=None, device=torch.device("cpu"), prefix=None):
    if getattr(module, "_comfy_quant_format", None) is not None:
        module._apply_lora_impl = module._apply_lora_direct
        module._apply_single_lora_impl = module._apply_single_lora_direct
        module._linear_forward_impl = module._linear_forward_direct
        return
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
        return
    for lora_diff_names in getattr(module, "lora_diffs", []):
        if not isinstance(lora_diff_names, tuple):
            continue
        lora_diff_0 = getattr(module, lora_diff_names[0])
        lora_diff_1 = getattr(module, lora_diff_names[1])
        lora_shape = (
            lora_diff_0.flatten(start_dim=1).shape[0],
            lora_diff_1.flatten(start_dim=1).shape[1],
        )
        if len(weight.shape) == 2 and lora_shape[0] == weight.shape[0] and lora_shape[1] == weight.shape[1] * 2:
            if state_dict is not None:
                if bind_comfy_quant_metadata_for_lora(
                    module,
                    state_dict,
                    prefix or "",
                    compute_dtype if compute_dtype is not None else module.compute_dtype,
                    device,
                    lora_shape=lora_shape,
                ):
                    module._apply_lora_impl = module._apply_lora_direct
                    module._apply_single_lora_impl = module._apply_single_lora_direct
                    module._linear_forward_impl = module._linear_forward_direct
                    return
            raise RuntimeError(
                "Native quantized LoRA target is missing Comfy quant metadata: "
                f"weight={tuple(weight.shape)}, lora={lora_shape}. Reload the model so "
                "WanAnimatePlus can bind the native quant metadata before sampling."
            )

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu"), state_dict=None, compute_dtype=None):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device, state_dict, compute_dtype)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            if state_dict is not None:
                lora_shape = None
                for diff in lora_diffs:
                    lora_shape = _lora_diff_shape(diff)
                    if lora_shape is not None:
                        break
                bind_comfy_quant_metadata_for_lora(
                    module,
                    state_dict,
                    key,
                    compute_dtype if compute_dtype is not None else module.compute_dtype,
                    device,
                    lora_shape=lora_shape,
                )
            module.set_lora_diffs(lora_diffs, device=device)
            module.set_lora_strengths(lora_strengths, device=device)
            _force_direct_lora_if_needed(module, state_dict, compute_dtype, device, key)
            module._step.fill_(0)   # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False,
        is_gguf=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile
        self.is_gguf = is_gguf

        if not allow_compile:
            self._apply_lora_impl = self._apply_lora_custom_op
            self._apply_single_lora_impl = self._apply_single_lora_custom_op
            self._linear_forward_impl = self._linear_forward_custom_op
        else:
            self._apply_lora_impl = self._apply_lora_direct
            self._apply_single_lora_impl = self._apply_single_lora_direct
            self._linear_forward_impl = self._linear_forward_direct

    def _apply(self, fn, recurse=True):
        if getattr(self, "_comfy_quant_format", None) is None:
            return super()._apply(fn, recurse=recurse)

        if recurse:
            for module in self.children():
                module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue
            param_grad = param.grad
            with torch.no_grad():
                param_applied = fn(param)
            if (not torch.is_inference_mode_enabled()) and param_applied.is_inference():
                param_applied = param_applied.clone()
            out_param = nn.Parameter(param_applied, requires_grad=param.requires_grad)
            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                if (not torch.is_inference_mode_enabled()) and grad_applied.is_inference():
                    grad_applied = grad_applied.clone()
                out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)
            self._parameters[key] = out_param

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self


    # Direct implementations (no custom ops)
    def _apply_lora_direct(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        patch_diff = torch.mm(
            lora_diff_0.flatten(start_dim=1),
            lora_diff_1.flatten(start_dim=1)
        ).reshape(weight.shape) + 0
        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        scale = lora_strength * alpha
        return weight + patch_diff * scale

    def _apply_single_lora_direct(self, weight, lora_diff, lora_strength):
        return weight + lora_diff * lora_strength

    def _linear_forward_direct(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    # Custom op implementations
    def _apply_lora_custom_op(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        return torch.ops.wananimateplus.apply_lora(weight, lora_diff_0, lora_diff_1,
            float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
        )

    def _apply_single_lora_custom_op(self, weight, lora_diff, lora_strength):
        return torch.ops.wananimateplus.apply_single_lora(weight, lora_diff, lora_strength)

    def _linear_forward_custom_op(self, input, weight, bias):
        return torch.ops.wananimateplus.linear_forward(input, weight, bias)

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self.lora_diffs = []
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

    def set_lora_strengths(self, lora_strengths, device=torch.device("cpu")):
        self._lora_strength_tensors = []
        self._lora_strength_is_scheduled = []
        self._step = self._step.to(device)
        for i, strength in enumerate(lora_strengths):
            if isinstance(strength, list):
                tensor = torch.tensor(strength, dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(True)
            else:
                tensor = torch.tensor([strength], dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(False)

    def _get_lora_strength(self, idx):
        strength_tensor = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return strength_tensor.index_select(0, self._step).squeeze(0)
        return strength_tensor[0]

    def _get_weight_with_lora(self, weight):
        """Apply LoRA using custom ops to avoid graph breaks"""
        if not self._has_lora_diffs():
            return weight

        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)

            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])

                weight = self._apply_lora_impl(
                    weight, lora_diff_0, lora_diff_1,
                    float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
                )
            else:
                lora_diff = getattr(self, lora_diff_names)
                weight = self._apply_single_lora_impl(weight, lora_diff, lora_strength)
        return weight

    def _lora_tuple_matrices(self, lora_diff_0, lora_diff_1):
        return lora_diff_0.flatten(start_dim=1), lora_diff_1.flatten(start_dim=1)

    def _lora_alpha(self, lora_diff_1, lora_diff_2):
        alpha = float(lora_diff_2) if lora_diff_2 is not None else 0.0
        return alpha / lora_diff_1.shape[0] if alpha != 0.0 else 1.0

    def _can_apply_lora_to_output(self, weight, input):
        return self._native_quant_lora_output_reject_reason(weight, input) is None

    def _native_quant_lora_output_reject_reason(self, weight, input):
        if not self._has_lora_diffs():
            return "no LoRA diffs are registered"
        if getattr(self, "_comfy_quant_format", None) is None:
            return "module is not a ComfyUI-native quantized Linear"
        if not self._is_comfy_quant_tensor(weight):
            return f"weight is not a QuantizedTensor-like object: type={type(weight).__name__}"
        if self.scale_weight is not None:
            return "scale_weight is present"
        if weight.ndim != 2:
            return f"weight ndim is {weight.ndim}, expected 2"
        if weight.shape[-1] != input.shape[-1]:
            return f"weight/input feature mismatch: weight={tuple(weight.shape)}, input={tuple(input.shape)}"

        out_features, in_features = weight.shape
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_a, lora_b = self._lora_tuple_matrices(lora_diff_0, lora_diff_1)
                if lora_a.ndim != 2 or lora_b.ndim != 2:
                    return f"LoRA {idx} matrices are not 2D: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
                if lora_a.shape[0] != out_features or lora_b.shape[1] != in_features:
                    return (
                        f"LoRA {idx} shape does not match weight/input: "
                        f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}, weight={tuple(weight.shape)}"
                    )
                if lora_a.shape[1] != lora_b.shape[0]:
                    return f"LoRA {idx} rank mismatch: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            else:
                lora_diff = getattr(self, lora_diff_names)
                if tuple(lora_diff.shape) != tuple(weight.shape):
                    return f"LoRA {idx} full diff shape mismatch: diff={tuple(lora_diff.shape)}, weight={tuple(weight.shape)}"
        return None

    def _lora_debug_shapes(self):
        shapes = []
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_a, lora_b = self._lora_tuple_matrices(lora_diff_0, lora_diff_1)
                shapes.append(f"{idx}:tuple A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}")
            else:
                lora_diff = getattr(self, lora_diff_names)
                shapes.append(f"{idx}:diff {tuple(lora_diff.shape)}")
        return "[" + ", ".join(shapes) + "]"

    def _raise_native_quant_lora_fallback(self, weight, input, reason):
        layer = getattr(self, "_wan_layer_prefix", "<unknown>")
        raise RuntimeError(
            "Native quant LoRA would require full-weight dequantization; refusing to avoid VRAM spike. "
            f"layer={layer}, "
            f"format={getattr(self, '_comfy_quant_format', None)}, "
            f"reason={reason}, "
            f"weight_type={type(weight).__name__}, "
            f"weight_shape={tuple(weight.shape) if hasattr(weight, 'shape') else None}, "
            f"weight_device={getattr(weight, 'device', None)}, "
            f"input_shape={tuple(input.shape)}, input_dtype={input.dtype}, input_device={input.device}, "
            f"scale_weight_present={self.scale_weight is not None}, "
            f"lora_shapes={self._lora_debug_shapes()}"
        )

    def _apply_lora_to_output(self, input, out):
        # Equivalent to F.linear(input, weight + A @ B), without materializing A @ B
        # or a full-size output delta.
        if out.device != input.device:
            raise RuntimeError(
                f"LoRA output path device mismatch: input={input.device}, out={out.device}"
            )
        if not out.is_contiguous():
            out = out.contiguous()
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)

            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                lora_a, lora_b = self._lora_tuple_matrices(lora_diff_0, lora_diff_1)
                lora_a = lora_a.to(device=input.device, dtype=out.dtype)
                lora_b = lora_b.to(device=input.device, dtype=input.dtype)
                lora_strength = lora_strength.to(device=input.device, dtype=out.dtype)
                alpha = self._lora_alpha(lora_diff_1, lora_diff_2)
                rank_states = torch.nn.functional.linear(input, lora_b)
                if rank_states.dtype != out.dtype:
                    rank_states = rank_states.to(dtype=out.dtype)
                rank_states = rank_states.reshape(-1, rank_states.shape[-1])
                out_2d = out.view(-1, out.shape[-1])
                out_2d.addmm_(rank_states, (lora_a * (lora_strength * alpha)).t())
            else:
                lora_diff = getattr(self, lora_diff_names)
                lora_diff = lora_diff.to(device=input.device, dtype=input.dtype)
                lora_strength = lora_strength.to(device=input.device, dtype=input.dtype)
                out = out + torch.nn.functional.linear(input, lora_diff) * lora_strength
        return out

    def _raise_comfy_quant_device_mismatch(self, input, mismatches):
        layer = getattr(self, "_wan_layer_prefix", "<unknown>")
        details = ", ".join(f"{name}={dev}" for name, dev in mismatches)
        raise RuntimeError(
            "ComfyUI-native quantized Linear device mismatch before forward. "
            "Block swap must move the whole module to the input device before computation; "
            f"layer={layer}, input_device={input.device}, {details}"
        )

    def _check_comfy_quant_forward_devices(self, weight, input):
        if getattr(self, "_comfy_quant_format", None) != "nvfp4":
            return

        mismatches = []

        def check_tensor(name, tensor):
            if isinstance(tensor, torch.Tensor) and tensor.device != input.device:
                mismatches.append((name, tensor.device))

        check_tensor("weight", weight)
        check_tensor("weight._qdata", getattr(weight, "_qdata", None))

        params = getattr(weight, "_params", None)
        if params is not None:
            check_tensor("weight._params.scale", getattr(params, "scale", None))
            check_tensor("weight._params.block_scale", getattr(params, "block_scale", None))

        check_tensor("_comfy_quant_weight_scale", getattr(self, "_comfy_quant_weight_scale", None))
        check_tensor("_comfy_quant_block_scale", getattr(self, "_comfy_quant_block_scale", None))

        if mismatches:
            self._raise_comfy_quant_device_mismatch(input, mismatches)

    def _prepare_weight(self, input):
        """Prepare weight tensor - handles regular, GGUF, and Comfy native quant weights"""
        if getattr(self, "_comfy_quant_format", None) is not None:
            weight = self.weight
            if self._is_materialized_regular_weight(weight, input):
                return weight.to(input)
            if weight.device.type == "meta" or contains_meta_tensor(getattr(weight, "_qdata", None)):
                raise RuntimeError(
                    "ComfyUI-native quantized Linear weight is still a meta tensor. "
                    "WanAnimatePlus should materialize native quant weights from the model state_dict "
                    "before forward; reload the model or report the layer path that skipped load_weights."
                )
            if contains_meta_tensor(getattr(weight, "_params", None)):
                raise RuntimeError(
                    "ComfyUI-native quantized Linear weight has meta quantization params. "
                    "WanAnimatePlus should materialize native quant weights before forward."
                )
            if weight.device != input.device and getattr(self, "_comfy_quant_format", None) == "nvfp4":
                self._raise_comfy_quant_device_mismatch(input, [("weight", weight.device)])
            elif weight.device != input.device:
                weight = weight.to(device=input.device)
            self._check_comfy_quant_forward_devices(weight, input)
            return weight
        if self.is_gguf:
            weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        else:
            raw_weight = self.weight
            if (
                getattr(self, "_comfy_quant_format", None) is None
                and isinstance(raw_weight, torch.Tensor)
                and raw_weight.ndim == 2
                and raw_weight.dtype == torch.uint8
                and raw_weight.shape[-1] * 2 == input.shape[-1]
            ):
                raise RuntimeError(
                    "Detected a half-width uint8 Linear weight entering a non-quantized path: "
                    f"weight={tuple(raw_weight.shape)}, input={tuple(input.shape)}. "
                    "This looks like an NVFP4 packed weight, but WanAnimatePlus did not bind "
                    "its weight_scale/weight_scale_2 metadata before forward."
                )
            weight = self.weight.to(input)
        return weight

    def _has_lora_diffs(self):
        return hasattr(self, "lora_diff_0_0")

    def _dequantize_comfy_quant_weight(self, weight, input):
        fmt = getattr(self, "_comfy_quant_format", None)
        if fmt is None:
            return weight
        scale = getattr(self, "_comfy_quant_weight_scale", None)
        block_scale = getattr(self, "_comfy_quant_block_scale", None)
        if scale is not None and scale.device != weight.device:
            scale = scale.to(device=weight.device)
        if block_scale is not None and block_scale.device != weight.device:
            block_scale = block_scale.to(device=weight.device)
        logical_shape = getattr(self, "_comfy_quant_logical_shape", None)
        dtype = self.compute_dtype if self.compute_dtype is not None else input.dtype
        weight = dequantize_comfy_quant_weight(
            weight,
            fmt,
            dtype,
            scale=scale,
            block_scale=block_scale,
            logical_shape=logical_shape,
        )
        return weight.to(input)

    def _quantize_raw_comfy_quant_weight(self, weight, input):
        fmt = getattr(self, "_comfy_quant_format", None)
        if fmt is None:
            return weight
        scale = getattr(self, "_comfy_quant_weight_scale", None)
        block_scale = getattr(self, "_comfy_quant_block_scale", None)
        if scale is not None and scale.device != weight.device:
            scale = scale.to(device=weight.device)
        if block_scale is not None and block_scale.device != weight.device:
            block_scale = block_scale.to(device=weight.device)
        dtype = self.compute_dtype if self.compute_dtype is not None else input.dtype
        return quantize_raw_comfy_quant_weight(
            weight,
            fmt,
            dtype,
            scale=scale,
            block_scale=block_scale,
            logical_shape=getattr(self, "_comfy_quant_logical_shape", None),
            layout_name=getattr(self, "_comfy_quant_layout", None),
            quant_params=getattr(self, "_comfy_quant_params", None),
        )

    def _is_comfy_quant_tensor(self, weight):
        return hasattr(weight, "_qdata") and hasattr(weight, "_params")

    def _is_raw_comfy_quant_weight(self, weight):
        fmt = getattr(self, "_comfy_quant_format", None)
        if fmt is None or self._is_comfy_quant_tensor(weight):
            return False
        if fmt == "nvfp4":
            return weight.dtype == torch.uint8
        if fmt in ("float8_e4m3fn", "mxfp8"):
            return weight.dtype == torch.float8_e4m3fn
        if fmt == "float8_e5m2":
            return weight.dtype == torch.float8_e5m2
        return False

    def _is_materialized_regular_weight(self, weight, input):
        if not isinstance(weight, torch.Tensor):
            return False
        if self._is_comfy_quant_tensor(weight) or self._is_raw_comfy_quant_weight(weight):
            return False
        if weight.device.type == "meta" or weight.ndim != 2:
            return False
        if weight.shape[-1] != input.shape[-1]:
            return False
        return torch.is_floating_point(weight)

    def _dequantize_comfy_quant_for_lora(self, weight, input):
        if getattr(self, "_comfy_quant_format", None) is None or not self._has_lora_diffs():
            return weight
        if self._is_comfy_quant_tensor(weight) or self._is_raw_comfy_quant_weight(weight):
            return self._dequantize_comfy_quant_weight(weight, input)
        return weight

    def _ensure_comfy_quant_forward_weight(self, weight, input):
        if getattr(self, "_comfy_quant_format", None) is None:
            return weight
        if self._is_raw_comfy_quant_weight(weight):
            weight = self._quantize_raw_comfy_quant_weight(weight, input)
        elif weight.shape[-1] == input.shape[-1]:
            return weight
        else:
            weight = self._dequantize_comfy_quant_weight(weight, input)
        if weight.shape[-1] != input.shape[-1]:
            raise RuntimeError(
                "ComfyUI-native quantized Linear weight has incompatible shape after "
                f"fallback dequantization: weight={tuple(weight.shape)}, input={tuple(input.shape)}"
            )
        return weight

    def _align_scale_weight_to_weight(self, weight):
        sw = self.scale_weight.to(weight.device, weight.dtype)

        # Block-wise scales may be stored in compressed form. Expand only
        # along the input-feature axis so the scale stays attached to weight.
        if sw.ndim > 1 and sw.shape[-1] != weight.shape[-1]:
            if weight.shape[-1] % sw.shape[-1] == 0:
                sw = sw.repeat_interleave(weight.shape[-1] // sw.shape[-1], dim=-1)
        elif sw.ndim == 1 and sw.shape[0] != weight.shape[-1]:
            if sw.shape[0] == weight.shape[0]:
                sw = sw.unsqueeze(-1)
            elif weight.shape[-1] % sw.shape[0] == 0:
                sw = sw.repeat_interleave(weight.shape[-1] // sw.shape[0])
        return sw

    def _maybe_quantize_input_for_qqgemm(self, input_2d: torch.Tensor, weight) -> torch.Tensor:
        """Quantize a 2-D input for Q×Q GEMM when hardware and format support it.

        Caller must ensure input_2d is already reshaped to (M, K).
        Returns QuantizedTensor on success, original tensor on any failure.
        """
        fmt = getattr(self, "_comfy_quant_format", None)
        if fmt is None:
            return input_2d
        try:
            from .comfy_quant_linear import QuantizedTensor, get_layout_class, QUANT_ALGOS
        except Exception:
            return input_2d
        if QuantizedTensor is None or get_layout_class is None:
            return input_2d
        device = input_2d.device
        if fmt in ("float8_e4m3fn", "float8_e5m2"):
            if not supports_fp8_compute(device):
                return input_2d
        elif fmt == "nvfp4":
            if not supports_nvfp4_compute(device):
                return input_2d
        else:
            # int8_tensorwise: quantize_input=False by design
            return input_2d
        qcfg = QUANT_ALGOS.get(fmt)
        if qcfg is None:
            return input_2d
        layout_name = qcfg.get("comfy_tensor_layout")
        if layout_name is None:
            return input_2d
        try:
            return QuantizedTensor.from_float(input_2d, layout_name)
        except Exception:
            return input_2d

    def _check_linear_weight_shape(self, weight, input):
        if weight.shape[-1] == input.shape[-1]:
            return
        if (
            weight.ndim == 2
            and weight.shape[-1] * 2 == input.shape[-1]
            and getattr(self, "_comfy_quant_format", None) is None
        ):
            raise RuntimeError(
                "Half-width Linear weight reached F.linear without NVFP4 metadata: "
                f"weight={tuple(weight.shape)}, input={tuple(input.shape)}. "
                "The model loader must bind native quant scale metadata for this layer."
            )
        raise RuntimeError(
            "Linear weight/input feature mismatch before F.linear: "
            f"weight={tuple(weight.shape)}, input={tuple(input.shape)}"
        )

    def forward(self, input):
        weight = self._prepare_weight(input)
        stale_comfy_quant_metadata = (
            getattr(self, "_comfy_quant_format", None) is not None
            and self._is_materialized_regular_weight(weight, input)
        )
        if not stale_comfy_quant_metadata:
            weight = self._ensure_comfy_quant_forward_weight(weight, input)

        if self.bias is not None:
            bias = self.bias.to(input if not self.is_gguf else self.compute_dtype)
        else:
            bias = None
        if not stale_comfy_quant_metadata:
            self._check_comfy_quant_forward_devices(weight, input)

        if not stale_comfy_quant_metadata:
            lora_output_reject_reason = self._native_quant_lora_output_reject_reason(weight, input)
            if lora_output_reject_reason is None:
                self._check_linear_weight_shape(weight, input)
                input_shape = input.shape
                if input.ndim == 3:
                    # Reshape to 2D for Q×Q GEMM (mirroring ComfyUI-master)
                    inp_2d = input.reshape(-1, input_shape[2])
                    q_inp = self._maybe_quantize_input_for_qqgemm(inp_2d, weight)
                    out = self._linear_forward_impl(q_inp, weight, bias)
                    del q_inp
                    out = out.reshape((-1, input_shape[1], weight.shape[0]))
                else:
                    q_inp = self._maybe_quantize_input_for_qqgemm(input, weight)
                    out = self._linear_forward_impl(q_inp, weight, bias)
                    del q_inp
                out = self._apply_lora_to_output(input, out)
                del weight, input, bias
                return out
            if getattr(self, "_comfy_quant_format", None) is not None and self._has_lora_diffs():
                self._raise_native_quant_lora_fallback(weight, input, lora_output_reject_reason)

            weight = self._dequantize_comfy_quant_for_lora(weight, input)

        # Only apply scale_weight for non-GGUF models
        if not self.is_gguf and self.scale_weight is not None:
            sw = self._align_scale_weight_to_weight(weight)
            try:
                weight = weight * sw
            except RuntimeError as e:
                raise RuntimeError(
                    f"scale_weight shape {tuple(sw.shape)} is not compatible with weight shape {tuple(weight.shape)}"
                ) from e

        weight = self._get_weight_with_lora(weight)
        self._check_linear_weight_shape(weight, input)
        out = self._linear_forward_impl(input, weight, bias)
        del weight, input, bias
        return out

def update_lora_step(module, step):
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear) and hasattr(submodule, "_step"):
            submodule._step.fill_(step)

def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")
