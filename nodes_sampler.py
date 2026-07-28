# Copyright (c) 2025 kijai
# Modified from nodes_sampler.py in ComfyUI-WanVideoWrapper.
# Original project: https://github.com/kijai/ComfyUI-WanVideoWrapper
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Added APG, APG-chain, and Bernini CFG-chain guidance modes.
#   - Added sampler graph-detach protection to prevent denoising graph accumulation.
#   - Added Bernini/SCAIL context_latents/context_roles propagation,
#     context-window slicing, and simple T2V/Bernini fast-path routing.
#   - Threaded rope_function, context_window_start, and context stream metadata
#     through added Bernini/SCAIL paths, including local context-window starts.
#   - Added prefix frame support in context windows, looping, face/pose slices, image conditions, and noise predictions.
#   - Added transition_video hard conditioning and canvas_expansion_px-aware Uni3C/render/output handling.
#   - Added EverAnimate segmented sampling with anchors, generated/random/user-first anchors, repeat-anchor padding, bg/mask conditioning, motion latents, and internal context-option blocking.
#   - Added EverAnimate VAE re-encode clamp/uint8 preprocessing to match the reference path.
#   - Added variable WanAnimate anchor-count forwarding for pose/face alignment.
#   - Added SCAIL-2 sampler-side freeze_mask handling for prefix/transition latents.
#   - Added SCAIL-2 loop output tensor caching, background cache saves, and tail-frame anchor encoding.
#   - Added SCAIL-2 auto_drift loop colormatch for lightweight seam correction.
#   - Added SCAIL-2 loop two-phase handoff sampling controls; thanks to
#     checknickname/ComfyUI-Scail2-Sampler-Helper for the idea and
#     user2318/ComfyUI-CustomNodeKit as an MIT-licensed reference project.
#   - Added official-compatible SCAIL-2 Flow sampler with legacy-aligned
#     reference/freeze masks, loop colormatch, random chunk seeds, and
#     two-phase freeze handling for official MODEL/CONDITIONING/LATENT chains.
#   - Added simplified WanAnimatePlus Easy Sampler and Easy SamplerSettings wrapper nodes.
#   RoPE math/mechanisms, upstream Wan/Bernini source-id RoPE mechanisms, and
#   Comfy RoPE implementations remain upstream/third-party work.
# Licensed under the Apache License, Version 2.0
import os, gc, math, copy, shutil, time
import torch
import numpy as np
from tqdm import tqdm
import inspect
import folder_paths
import comfy.sample
import comfy.samplers
import comfy.model_sampling
from .wanvideo.modules.model import rope_params
from .custom_linear import remove_lora_from_module, set_lora_params, _replace_linear
from .wanvideo.schedulers import get_scheduler, scheduler_list
from .gguf.gguf import set_lora_params_gguf
from .comfy_quant_linear import is_comfy_quant_state_dict, rebind_comfy_quant_metadata
from .multitalk.multitalk import add_noise
from .utils import(log, print_memory, apply_lora, fourier_filter, optimized_scale, setup_radial_attention,
                   compile_model, dict_to_device, tangential_projection, get_raag_guidance, temporal_score_rescaling, offload_transformer, init_blockswap)
from .multitalk.multitalk_loop import multitalk_loop
from .cache_methods.cache_methods import cache_report
from .nodes_model_loading import load_weights
from .enhance_a_video.globals import set_enhance_weight, set_num_frames
from .WanMove.trajectory import replace_feature
from contextlib import nullcontext

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale
from comfy.cli_args import args, LatentPreviewMethod
from .scail2_flow import (
    FLOW_DEFERRED_BUILD_KEY,
    FLOW_FREEZE_MASK_KEY,
    FLOW_HANDOFF_MASK_KEY,
    FLOW_RUNTIME_KEY,
    FLOW_RUNTIME_VAE_KEY,
    align_4n1,
    auto_drift_frames,
    auto_drift_tail_means,
    build_conditioning_and_latent,
    clean_flow_runtime_for_output,
    color_match_frames,
    decode_latent_to_images,
    release_flow_vae,
    take_tail_with_front_pad,
)

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

rope_functions = ["default", "comfy", "comfy_chunked"]

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


# --- APG (Adaptive Projected Guidance) helpers, adapted from Bernini ---

class MomentumBuffer:
    """EMA buffer for smoothing guidance differences across timesteps."""

    def __init__(self, momentum: float = -0.5):
        self.momentum = momentum
        self.running_average = 0

    def update(self, value: torch.Tensor):
        self.running_average = value + self.momentum * self.running_average


def _normalize_diff(diff: torch.Tensor, base_pred: torch.Tensor,
                    momentum_buffer: MomentumBuffer | None = None,
                    eta: float = 1.0, norm_threshold: float = 0.0) -> torch.Tensor:
    """Project diff onto/off base_pred and recombine with weight eta.

    Operates on 4-D tensors [C, F, H, W] (spatial latent format).
    Norm is computed over the spatio-temporal dims [-1, -2, -3] = [F, H, W].
    """
    # Momentum
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    # Norm clipping
    if norm_threshold > 0:
        diff_n = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale = torch.minimum(torch.ones_like(diff_n), norm_threshold / diff_n)
        diff = diff * scale

    # Parallel / orthogonal projection in double precision
    v0 = diff.double()
    v1 = base_pred.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return (v0_orthogonal + eta * v0_parallel).to(diff.dtype)


def _normalized_guidance(pred_cond: torch.Tensor, pred_uncond: torch.Tensor,
                         guidance_scale: float,
                         momentum_buffer: MomentumBuffer | None = None,
                         eta: float = 1.0, norm_threshold: float = 0.0) -> torch.Tensor:
    """Single-condition APG: project (cond - uncond) onto cond with eta weight."""
    nd = _normalize_diff(pred_cond - pred_uncond, pred_cond,
                         momentum_buffer, eta, norm_threshold)
    return pred_uncond + guidance_scale * nd


def _normalized_guidance_chain(pred_uncond, preds, scales,
                               momentum_buffers, eta, norm_threshold):
    """Chained APG. Each condition's diff is taken against the previous prediction.
       preds = [eps_I, eps_TI], scales = [omega_I, omega_TI]
       Returns: uncond + sum_i scale_i * normalize_diff(cond_i - base_i)"""
    bases = [pred_uncond] + list(preds)
    result = pred_uncond
    for i, cond in enumerate(preds):
        nd = _normalize_diff(cond - bases[i], cond,
                             momentum_buffers[i], eta, norm_threshold)
        result = result + scales[i] * nd
    return result


class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": "unipc",}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", {"tooltip": "FlowEdit support has been deprecated"}),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Start step for the sampling, 0 means full sampling, otherwise samples only from this step"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "End step for the sampling, -1 means full sampling, otherwise samples only until this step"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
                "guidance_mode": (["cfg", "apg", "apg_chain", "cfg_chain"], {"default": "cfg", "tooltip": "Guidance mode: cfg (standard CFG), apg (single-condition APG), apg_chain (image-reference APG chain), or cfg_chain (Bernini chained CFG)"}),
                "apg_eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "APG: parallel/orthogonal balance (0=orthogonal only, 1=full)"}),
                "apg_momentum": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "APG: EMA momentum for smoothing guidance differences"}),
                "apg_norm_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "APG: L2 norm clipping threshold (0=disabled)"}),
                "apg_omega": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG: guidance strength (equivalent to CFG scale, replaces cfg when APG is active)"}),
                "apg_omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG chain: image-only guidance strength"}),
                "apg_omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG chain: text+image guidance strength"}),
                "chain_omega_V": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: video-only guidance strength"}),
                "chain_omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: extra reference context strength (VI - V; reference video/images)"}),
                "chain_omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: text+image guidance strength"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None,
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None,
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, freeinit_args=None, start_step=0, end_step=-1, add_noise_to_samples=False,
        guidance_mode="cfg", apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0,
        apg_omega=4.0, apg_omega_I=4.5, apg_omega_TI=4.0,
        chain_omega_V=1.25, chain_omega_I=4.5, chain_omega_TI=4.0):
        if flowedit_args is not None:
            raise Exception("FlowEdit support has been deprecated and removed due to lack of use and code maintainability")
        patcher = model
        model = model.model
        transformer = model.diffusion_model

        dtype = model["base_dtype"]
        weight_dtype = model["weight_dtype"]
        fp8_matmul = model["fp8_matmul"]
        gguf_reader = model["gguf_reader"]
        control_lora = model["control_lora"]

        vae = image_embeds.get("vae", None)
        tiled_vae = image_embeds.get("tiled_vae", False)

        transformer_options = copy.deepcopy(patcher.model_options.get("transformer_options", None))
        merge_loras = transformer_options["merge_loras"]

        block_swap_args = transformer_options.get("block_swap_args", None)
        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", False)
            transformer.blocks_to_swap = block_swap_args.get("blocks_to_swap", 0)
            transformer.vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", 0)
            transformer.prefetch_blocks = block_swap_args.get("prefetch_blocks", 0)
            transformer.block_swap_debug = block_swap_args.get("block_swap_debug", False)
            transformer.offload_img_emb = block_swap_args.get("offload_img_emb", False)
            transformer.offload_txt_emb = block_swap_args.get("offload_txt_emb", False)

        is_5b = transformer.out_dim == 48
        vae_upscale_factor = 16 if is_5b else 8

        # Load weights
        if transformer.audio_model is not None:
            for block in transformer.blocks:
                if hasattr(block, 'audio_block'):
                    block.audio_block = None

        if not transformer.patched_linear and patcher.model["sd"] is not None and len(patcher.patches) != 0 and gguf_reader is None:
            transformer = _replace_linear(transformer, dtype, patcher.model["sd"], compile_args=model["compile_args"])
            transformer.patched_linear = True
        if patcher.model["sd"] is not None and gguf_reader is None:
            load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device,
                         block_swap_args=block_swap_args, compile_args=model["compile_args"])

        if gguf_reader is not None: #handle GGUF
            load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True,
                         reader=gguf_reader, block_swap_args=block_swap_args, compile_args=model["compile_args"])
            set_lora_params_gguf(transformer, patcher.patches)
            transformer.patched_linear = True
        elif len(patcher.patches) != 0: #handle patched linear layers (unmerged loras, fp8 scaled)
            log.info(f"Using {len(patcher.patches)} LoRA weight patches for WanVideo model")
            if not merge_loras and fp8_matmul:
                raise NotImplementedError("FP8 matmul with unmerged LoRAs is not supported")
            if is_comfy_quant_state_dict(patcher.model["sd"]):
                rebind_comfy_quant_metadata(transformer, patcher.model["sd"], dtype, device)
            set_lora_params(transformer, patcher.patches, state_dict=patcher.model["sd"], compute_dtype=dtype)
        else:
            remove_lora_from_module(transformer) #clear possible unmerged lora weights

        transformer.lora_scheduling_enabled = transformer_options.get("lora_scheduling_enabled", False)

        #torch.compile
        if model["auto_cpu_offload"] is False:
            transformer = compile_model(transformer, model["compile_args"])

        multitalk_sampling = image_embeds.get("multitalk_sampling", False)
        everanimate_sampling = image_embeds.get("everanimate", False)

        if multitalk_sampling and context_options is not None:
            raise Exception("context_options are not compatible or necessary with 'WanVideoImageToVideoMultiTalk' node, since it's already an alternative method that creates the video in a loop.")
        if everanimate_sampling and context_options is not None:
            raise Exception("context_options are not compatible with EverAnimate sampling, since it creates the video in internal segments.")

        if not multitalk_sampling and scheduler == "multitalk":
            raise Exception("multitalk scheduler is only for multitalk sampling when using ImagetoVideoMultiTalk -node")

        if text_embeds == None:
            text_embeds = {
                "prompt_embeds": [],
                "negative_prompt_embeds": [],
            }
        else:
            text_embeds = dict_to_device(text_embeds, device)

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        #region Scheduler
        if denoise_strength < 1.0:
            if start_step != 0:
                raise ValueError("start_step must be 0 when denoise_strength is used")
            start_step = steps - int(steps * denoise_strength) - 1
            add_noise_to_samples = True #for now to not break old workflows

        sample_scheduler = None
        if isinstance(scheduler, dict):
            sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
            timesteps = scheduler["timesteps"]
            start_step = scheduler.get("start_step", start_step)
        elif scheduler != "multitalk":
            sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, denoise_strength, sigmas=sigmas, log_timesteps=True)
        else:
            timesteps = torch.tensor([1000, 750, 500, 250], device=device)

        total_steps = steps
        steps = len(timesteps)

        is_pusa = "pusa" in sample_scheduler.__class__.__name__.lower()

        if scheduler != "multitalk":
            scheduler_step_args = {"generator": seed_g}
            step_sig = inspect.signature(sample_scheduler.step)
            for arg in list(scheduler_step_args.keys()):
                if arg not in step_sig.parameters:
                    scheduler_step_args.pop(arg)

        # Ovi
        if transformer.audio_model is not None: # temporary workaround (...nothing more permanent)
            for i, block in enumerate(transformer.blocks):
                block.audio_block = transformer.audio_model.blocks[i]
            sample_scheduler_ovi = copy.deepcopy(sample_scheduler)
            rope_function = "default" # comfy rope not implemented for ovi model yet
        ovi_negative_text_embeds = text_embeds.get("ovi_negative_prompt_embeds", None)
        ovi_audio_cfg = text_embeds.get("ovi_audio_cfg", None)
        if ovi_audio_cfg is not None:
            if not isinstance(ovi_audio_cfg, list):
                ovi_audio_cfg = [ovi_audio_cfg] * (steps + 1)

        if isinstance(cfg, list):
            if steps < len(cfg):
                log.info(f"Received {len(cfg)} cfg values, but only {steps} steps. Slicing cfg list to match steps.")
                cfg = cfg[:steps]
            elif steps > len(cfg):
                log.info(f"Received only {len(cfg)} cfg values, but {steps} steps. Extending cfg list to match steps.")
                cfg.extend([cfg[-1]] * (steps - len(cfg)))
            log.info(f"Using per-step cfg list: {cfg}")
        else:
            cfg = [cfg] * (steps + 1)

        control_latents = control_camera_latents = clip_fea = clip_fea_neg = end_image = recammaster = camera_embed = unianim_data = mocha_embeds = image_cond_neg =None
        vace_data = vace_context = vace_scale = None
        fun_or_fl2v_model = drop_last = False
        phantom_latents = fun_ref_image = ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg = noise_pred_flipped = None
        humo_audio = humo_audio_neg = None
        has_ref = image_embeds.get("has_ref", False)
        wananim_static_ref_latents = int(image_embeds.get("wananim_static_ref_latents", 0) or 0)
        wananim_main_ref_index = int(image_embeds.get("wananim_main_ref_index", max(wananim_static_ref_latents - 1, 0)) or 0)
        wananim_decode_ref_latents = int(image_embeds.get("wananim_decode_ref_latents", 0) or 0)
        wananim_num_anchor_latents = int(image_embeds.get("wananim_num_anchor_latents", max(wananim_static_ref_latents, 1)) or 1)

        #I2V
        story_mem_latents = image_embeds.get("story_mem_latents", None)
        image_cond = image_embeds.get("image_embeds", None)
        image_cond_mask = None
        if image_cond is not None:
            if transformer.in_dim == 16:
                raise ValueError("T2V (text to video) model detected, encoded images only work with I2V (Image to video) models")
            elif transformer.in_dim not in [48, 32]: # fun 2.1 models don't use the mask
                image_cond_mask = image_embeds.get("mask", None)
            # StoryMem
            if story_mem_latents is not None:
                image_cond = torch.cat([story_mem_latents.to(image_cond), image_cond], dim=1)
                image_cond_mask = torch.cat([torch.ones_like(story_mem_latents)[:4], image_cond_mask], dim=1) if image_cond_mask is not None else None

            if image_cond_mask is not None:
                image_cond = torch.cat([image_cond_mask, image_cond])
            else:
                image_cond[:, 1:] = 0

            #ATI tracks
            if transformer_options is not None:
                ATI_tracks = transformer_options.get("ati_tracks", None)
                if ATI_tracks is not None:
                    from .ATI.motion_patch import patch_motion
                    topk = transformer_options.get("ati_topk", 2)
                    temperature = transformer_options.get("ati_temperature", 220.0)
                    ati_start_percent = transformer_options.get("ati_start_percent", 0.0)
                    ati_end_percent = transformer_options.get("ati_end_percent", 1.0)
                    image_cond_ati = patch_motion(ATI_tracks.to(image_cond.device, image_cond.dtype), image_cond, topk=topk, temperature=temperature)
                    log.info(f"ATI tracks shape: {ATI_tracks.shape}")

            add_cond_latents = image_embeds.get("add_cond_latents", None)
            if add_cond_latents is not None:
                add_cond = add_cond_latents["pose_latent"]
                attn_cond = add_cond_latents["ref_latent"]
                attn_cond_neg = add_cond_latents["ref_latent_neg"]
                add_cond_start_percent = add_cond_latents["pose_cond_start_percent"]
                add_cond_end_percent = add_cond_latents["pose_cond_end_percent"]

            end_image = image_embeds.get("end_image", None)
            fun_or_fl2v_model = image_embeds.get("fun_or_fl2v_model", False)
            latent_frames = (image_embeds["num_frames"] - 1) // 4
            latent_frames = latent_frames + (2 if end_image is not None and not fun_or_fl2v_model else 1)
            latent_frames = latent_frames + story_mem_latents.shape[1] if story_mem_latents is not None else latent_frames
            noise = torch.randn( #C, T, H, W
                48 if is_5b else 16,
                latent_frames,
                image_embeds["lat_h"],
                image_embeds["lat_w"],
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                if transformer.in_dim not in [148, 52, 48, 36, 32]:
                    raise ValueError("Control signal only works with Fun-Control model")

                control_latents = control_embeds.get("control_images", None)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)

            drop_last = image_embeds.get("drop_last", False)
        else: #t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V models")

            # VACE
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            if not isinstance(vace_scale, list):
                vace_scale = [vace_scale] * (steps+1)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
            vace_seqlen = image_embeds.get("vace_seq_len", None)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {"context": vace_context,
                     "scale": vace_scale,
                     "start": vace_start_percent,
                     "end": vace_end_percent,
                     "seq_len": vace_seqlen
                     }
                ]
                if len(vace_additional_embeds) > 0:
                    for i in range(len(vace_additional_embeds)):
                        if vace_additional_embeds[i].get("has_ref", False):
                            has_ref = True
                        vace_scale = vace_additional_embeds[i]["vace_scale"]
                        if not isinstance(vace_scale, list):
                            vace_scale = [vace_scale] * (steps+1)
                        vace_data.append({
                            "context": vace_additional_embeds[i]["vace_context"],
                            "scale": vace_scale,
                            "start": vace_additional_embeds[i]["vace_start_percent"],
                            "end": vace_additional_embeds[i]["vace_end_percent"],
                            "seq_len": vace_additional_embeds[i]["vace_seq_len"]
                        })

            noise_ref_extra = 0 if wananim_static_ref_latents > 0 else (1 if has_ref else 0)
            noise = torch.randn(
                    48 if is_5b else 16,
                    target_shape[1] + noise_ref_extra,
                    target_shape[2] // 2 if is_5b else target_shape[2], #todo make this smarter
                    target_shape[3] // 2 if is_5b else target_shape[3], #todo make this smarter
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            recammaster = image_embeds.get("recammaster", None)
            if recammaster is not None:
                camera_embed = recammaster.get("camera_embed", None)
                recam_latents = recammaster.get("source_latents", None)
                orig_noise_len = noise.shape[1]
                log.info(f"RecamMaster camera embed shape: {camera_embed.shape}")
                log.info(f"RecamMaster source video shape: {recam_latents.shape}")
                seq_len *= 2

            if image_embeds.get("mocha_embeds", None) is not None:
                mocha_embeds = image_embeds.get("mocha_embeds", None)
                mocha_num_refs = image_embeds.get("mocha_num_refs", 0)
                orig_noise_len = noise.shape[1]
                seq_len = image_embeds.get("seq_len", seq_len)
                log.info(f"MoCha embeds shape: {mocha_embeds.shape}")

            # Fun control and control lora
            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds.get("control_images", None)
                if control_latents is not None:
                    control_latents = control_latents.to(device)

                control_camera_latents = control_embeds.get("control_camera_latents", None)
                if control_camera_latents is not None:
                    if transformer.control_adapter is None:
                        raise ValueError("Control camera latents are only supported with Fun-Control-Camera model")
                    control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                    control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)

                if control_lora:
                    image_cond = control_latents.to(device)
                    if not patcher.model.is_patched:
                        log.info("Re-loading control LoRA...")
                        patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                        patcher.model.is_patched = True
                else:
                    if transformer.in_dim not in [148, 48, 36, 32, 52]:
                        raise ValueError("Control signal only works with Fun-Control model")
                    image_cond = torch.zeros_like(noise).to(device) #fun control
                    if transformer.in_dim in [148, 52] or transformer.control_adapter is not None: #fun 2.2 control
                        mask_latents = torch.tile(
                            torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                        )
                        masked_video_latents_input = torch.zeros_like(noise)
                        image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)
                    clip_fea = None
                    fun_ref_image = control_embeds.get("fun_ref_image", None)
                    if fun_ref_image is not None:
                        if transformer.ref_conv.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            raise ValueError("Fun-Control reference image won't work with this specific fp8_scaled model, it's been fixed in latest version of the model")
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            else:
                if transformer.in_dim in [148, 52]: #fun inp
                    mask_latents = torch.tile(
                        torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                    )
                    masked_video_latents_input = torch.zeros_like(noise)
                    image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)

            # Phantom inputs
            phantom_latents = image_embeds.get("phantom_latents", None)
            phantom_cfg_scale = image_embeds.get("phantom_cfg_scale", None)
            if not isinstance(phantom_cfg_scale, list):
                phantom_cfg_scale = [phantom_cfg_scale] * (steps +1)
            phantom_start_percent = image_embeds.get("phantom_start_percent", 0.0)
            phantom_end_percent = image_embeds.get("phantom_end_percent", 1.0)

        # CLIP image features
        clip_fea = image_embeds.get("clip_context", None)
        if clip_fea is not None:
            clip_fea = clip_fea.to(dtype)
        clip_fea_neg = image_embeds.get("negative_clip_context", None)
        if clip_fea_neg is not None:
            clip_fea_neg = clip_fea_neg.to(dtype)

        num_frames = image_embeds.get("num_frames", 0)

        #HuMo inputs
        humo_audio = image_embeds.get("humo_audio_emb", None)
        humo_audio_neg = image_embeds.get("humo_audio_emb_neg", None)
        humo_reference_count = image_embeds.get("humo_reference_count", 0)

        if humo_audio is not None:
            from .HuMo.nodes import get_audio_emb_window
            if not multitalk_sampling:
                humo_audio, _ = get_audio_emb_window(humo_audio, num_frames, frame0_idx=0)
                zero_audio_pad = torch.zeros(humo_reference_count, *humo_audio.shape[1:]).to(humo_audio.device)
                humo_audio = torch.cat([humo_audio, zero_audio_pad], dim=0)
                humo_audio_neg = torch.zeros_like(humo_audio, dtype=humo_audio.dtype, device=humo_audio.device)
            humo_audio = humo_audio.to(device, dtype)

        if humo_audio_neg is not None:
            humo_audio_neg = humo_audio_neg.to(device, dtype)
        humo_audio_scale = image_embeds.get("humo_audio_scale", 1.0)
        humo_image_cond = image_embeds.get("humo_image_cond", None)
        humo_image_cond_neg = image_embeds.get("humo_image_cond_neg", None)

        pos_latent = neg_latent = None

        # Ovi
        noise_audio = latent_ovi = seq_len_ovi = None
        if transformer.audio_model is not None:
            noise_audio = samples.get("latent_ovi_audio", None) if samples is not None else None
            if noise_audio is not None:
                if not torch.any(noise_audio):
                    noise_audio = torch.randn(noise_audio.shape, device=torch.device("cpu"), dtype=torch.float32, generator=seed_g)
                else:
                    noise_audio = noise_audio.squeeze().movedim(0, 1).to(device, dtype)
            else:
                noise_audio = torch.randn((157, 20), device=torch.device("cpu"), dtype=torch.float32, generator=seed_g)  # T C
            log.info(f"Ovi audio latent shape: {noise_audio.shape}")
            latent_ovi = noise_audio
            seq_len_ovi = noise_audio.shape[0]

        if transformer.dim == 1536 and humo_image_cond is not None: #small humo model
            #noise = torch.cat([noise[:, :-humo_reference_count], humo_image_cond[4:, -humo_reference_count:]], dim=1)
            pos_latent = humo_image_cond[4:, -humo_reference_count:].to(device, dtype)
            neg_latent = torch.zeros_like(pos_latent)
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
            humo_image_cond = humo_image_cond_neg = None

        humo_audio_cfg_scale = image_embeds.get("humo_audio_cfg_scale", 1.0)
        humo_start_percent = image_embeds.get("humo_start_percent", 0.0)
        humo_end_percent = image_embeds.get("humo_end_percent", 1.0)
        if not isinstance(humo_audio_cfg_scale, list):
            humo_audio_cfg_scale = [humo_audio_cfg_scale] * (steps + 1)

        # region WanAnim inputs
        frame_window_size = image_embeds.get("frame_window_size", 77)
        wananimate_loop = image_embeds.get("looping", False)
        scail2_looping = bool(image_embeds.get("scail2_looping", False))
        scail2_two_phase = bool(image_embeds.get("scail2_two_phase", False))
        if scail2_two_phase and not scail2_looping:
            log.warning("SCAIL-2 two-phase settings are only used in SCAIL-2 loop mode; ignoring them for this run.")
            scail2_two_phase = False
        scail2_two_phase_start_step = int(image_embeds.get("scail2_two_phase_start_step", 0) or 0)
        scail2_two_phase_phase1_mask = max(
            0.0, min(1.0, float(image_embeds.get("scail2_two_phase_phase1_mask", 0.5)))
        )
        scail2_two_phase_phase2_mask = max(
            0.0, min(1.0, float(image_embeds.get("scail2_two_phase_phase2_mask", 1.0)))
        )
        scail2_requested_frames = int(image_embeds.get("scail2_requested_frames", image_embeds.get("num_frames", 0)))
        scail2_previous_frame_count = int(image_embeds.get("scail2_previous_frame_count", 5))
        # ============ Global transition_video read ============
        transition_latent = image_embeds.get("transition_latent", None)
        transition_mask_values = image_embeds.get("transition_mask_values", None)
        transition_len = transition_latent.shape[1] if transition_latent is not None else 0
        has_transition = transition_latent is not None
        scail_prefix_prepend_latents = int(image_embeds.get("scail_prefix_prepend_latents", 0))


        if has_transition:
            log.info(f"Transition video: {transition_len} latent frames")
            if transition_mask_values is not None:
                log.info(f"Linear decay mask values: {transition_mask_values.tolist()}")
        # ================================================================
        has_prefix = image_embeds.get("has_prefix", False)
        canvas_expansion_px = image_embeds.get("canvas_expansion_px", 0)
        prefix_prepend_latents = image_embeds.get("prefix_prepend_latents", 6 if has_prefix else 0)
        context_latents = image_embeds.get("context_latents", None)
        context_roles = image_embeds.get("context_roles", None)
        if has_prefix:
            log.info(f"Prefix frames: detected, will prepend {prefix_prepend_latents} latent frames to each non-first context window")
        if wananim_static_ref_latents > 0:
            log.info(f"WanAnimate static reference latents: {wananim_static_ref_latents}, main reference index: {wananim_main_ref_index}")
        if wananimate_loop and context_options is not None:
            raise Exception("context_options are not compatible or necessary with WanAnim looping, since it creates the video in a loop.")
        if scail2_looping and context_options is not None:
            raise Exception("context_options are not compatible with SCAIL-2 loop mode. Set frame_window_size equal to the normalized SCAIL-2 frame count to use context mode.")
        if scail2_looping and samples is not None:
            raise ValueError("Input latent samples are not supported with SCAIL-2 loop mode. Disconnect samples or set frame_window_size equal to the normalized SCAIL-2 frame count.")
        wananim_pose_latents = image_embeds.get("pose_latents", None)
        wananim_pose_strength = image_embeds.get("pose_strength", 1.0)
        wananim_face_strength = image_embeds.get("face_strength", 1.0)
        wananim_face_pixels = image_embeds.get("face_pixels", None)
        wananim_ref_masks = image_embeds.get("ref_masks", None)
        wananim_is_masked = image_embeds.get("is_masked", False)
        if not wananimate_loop: # create zero face pixels if mask is provided without face pixels, as masking seems to require face input to work properly
            if wananim_face_pixels is None and wananim_is_masked:
                if context_options is None:
                    wananim_face_pixels = torch.zeros(1, 3, num_frames-1, 512, 512, dtype=torch.float32, device=offload_device)
                else:
                    wananim_face_pixels = torch.zeros(1, 3, context_options["context_frames"]-1, 512, 512, dtype=torch.float32, device=device)

        if image_cond is None:
            image_cond = image_embeds.get("ref_latent", None)
            has_ref = image_cond is not None or has_ref

        latent_video_length = noise.shape[1]
        # Initialize FreeInit filter if enabled
        freq_filter = None
        if freeinit_args is not None:
            from .freeinit.freeinit_utils import get_freq_filter, freq_mix_3d
            filter_shape = list(noise.shape)  # [batch, C, T, H, W]
            freq_filter = get_freq_filter(
                filter_shape,
                device=device,
                filter_type=freeinit_args.get("freeinit_method", "butterworth"),
                n=freeinit_args.get("freeinit_n", 4) if freeinit_args.get("freeinit_method", "butterworth") == "butterworth" else None,
                d_s=freeinit_args.get("freeinit_s", 1.0),
                d_t=freeinit_args.get("freeinit_t", 1.0)
            )
            if samples is not None:
                saved_generator_state = samples.get("generator_state", None)
                if saved_generator_state is not None:
                    seed_g.set_state(saved_generator_state)

        # UniAnimate
        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device, dtype)
            dwpose_data = unianimate_poses["pose"].to(device, dtype)
            dwpose_data = torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
            dwpose_data = transformer.dwpose_embedding(dwpose_data)
            log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
            if not multitalk_sampling:
                if dwpose_data.shape[2] > latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating")
                    dwpose_data = dwpose_data[:,:, :latent_video_length]
                elif dwpose_data.shape[2] < latent_video_length:
                    log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose")
                    pad_len = latent_video_length - dwpose_data.shape[2]
                    pad = dwpose_data[:,:,:1].repeat(1,1,pad_len,1,1)
                    dwpose_data = torch.cat([dwpose_data, pad], dim=2)

            random_ref_dwpose_data = None
            if image_cond is not None:
                transformer.randomref_embedding_pose.to(device, dtype)
                random_ref_dwpose = unianimate_poses.get("ref", None)
                if random_ref_dwpose is not None:
                    random_ref_dwpose_data = transformer.randomref_embedding_pose(
                        random_ref_dwpose.to(device, dtype)
                        ).unsqueeze(2).to(dtype) # [1, 20, 104, 60]
                del random_ref_dwpose

            unianim_data = {
                "dwpose": dwpose_data,
                "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                "strength": unianimate_poses["strength"],
                "start_percent": unianimate_poses["start_percent"],
                "end_percent": unianimate_poses["end_percent"]
            }

        # FantasyTalking
        audio_proj = multitalk_audio_embeds = None
        audio_scale = 1.0
        if fantasytalking_embeds is not None:
            audio_proj = fantasytalking_embeds["audio_proj"].to(device)
            audio_scale = fantasytalking_embeds["audio_scale"]
            audio_cfg_scale = fantasytalking_embeds["audio_cfg_scale"]
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps +1)
            log.info(f"Audio proj shape: {audio_proj.shape}")


        # MultiTalk
        multitalk_audio_embeds = audio_emb_slice = audio_features_in = None
        multitalk_audio_stride = None
        multitalk_embeds = image_embeds.get("multitalk_embeds", multitalk_embeds)

        if multitalk_embeds is not None:
            audio_emb_slice = multitalk_embeds.get("audio_emb_slice", None) # if already sliced
            # Handle single or multiple speaker embeddings
            if audio_emb_slice is None:
                audio_features_in = multitalk_embeds.get("audio_features", None)
            if audio_features_in is not None:
                if isinstance(audio_features_in, list):
                    multitalk_audio_embeds = [emb.to(device, dtype) for emb in audio_features_in]
                else:
                    # keep backward-compatibility with single tensor input
                    multitalk_audio_embeds = [audio_features_in.to(device, dtype)]

                shapes = [tuple(e.shape) for e in multitalk_audio_embeds]
                log.info(f"Multitalk audio features shapes (per speaker): {shapes}")

            audio_scale = multitalk_embeds.get("audio_scale", 1.0)
            audio_cfg_scale = multitalk_embeds.get("audio_cfg_scale", 1.0)
            ref_target_masks = multitalk_embeds.get("ref_target_masks", None)
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

        # FantasyPortrait
        fantasy_portrait_input = None
        fantasy_portrait_embeds = image_embeds.get("portrait_embeds", None)
        if fantasy_portrait_embeds is not None:
            log.info("Using FantasyPortrait embeddings")
            fantasy_portrait_input = fantasy_portrait_embeds.copy()
            portrait_cfg = fantasy_portrait_input.get("cfg_scale", 1.0)
            if not isinstance(portrait_cfg, list):
                portrait_cfg = [portrait_cfg] * (steps + 1)

        # MiniMax Remover
        minimax_latents = image_embeds.get("minimax_latents", None)
        minimax_mask_latents = image_embeds.get("minimax_mask_latents", None)
        if minimax_latents is not None:
            log.info(f"minimax_latents: {minimax_latents.shape}, minimax_mask_latents: {minimax_mask_latents.shape}")
            minimax_latents = minimax_latents.to(device, dtype)
            minimax_mask_latents = minimax_mask_latents.to(device, dtype)

        # Context windows
        is_looped = False
        context_reference_latent = None
        if context_options is not None:
            if context_options["context_frames"] <= num_frames:
                context_schedule = context_options["context_schedule"]
                context_frames =  (context_options["context_frames"] - 1) // 4 + 1
                context_stride = context_options["context_stride"] // 4
                context_overlap = context_options["context_overlap"] // 4
                context_reference_latent = context_options.get("reference_latent", None)

                # Get total number of prompts
                num_prompts = len(text_embeds["prompt_embeds"])
                log.info(f"Number of prompts: {num_prompts}")
                # Calculate which section this context window belongs to
                section_length = max(latent_video_length - wananim_static_ref_latents, 1) if wananim_static_ref_latents > 0 else latent_video_length
                section_size = (section_length / num_prompts) if num_prompts != 0 else 1
                log.info(f"Section size: {section_size}")
                is_looped = context_schedule == "uniform_looped"

                if mocha_embeds is not None:
                    seq_len = (context_frames * 2 + 1 + mocha_num_refs) * (noise.shape[2] * noise.shape[3] // 4)
                else:
                    seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)
                base_patches_per_frame = math.ceil((noise.shape[2] * noise.shape[3]) / 4)
                log.info(f"context window seq len: {seq_len}")

                if context_options["freenoise"]:
                    log.info("Applying FreeNoise")
                    # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                    delta = context_frames - context_overlap
                    freenoise_offset = wananim_static_ref_latents if wananim_static_ref_latents > 0 else 0
                    freenoise_length = latent_video_length - freenoise_offset
                    for start_idx in range(0, freenoise_length-context_frames, delta):
                        place_idx = start_idx + context_frames
                        if place_idx >= freenoise_length:
                            break
                        end_idx = place_idx - 1

                        if end_idx + delta >= freenoise_length:
                            final_delta = freenoise_length - place_idx
                            list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                            list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                            noise[:, freenoise_offset + place_idx:freenoise_offset + place_idx + final_delta, :, :] = noise[:, freenoise_offset + list_idx, :, :]
                            break
                        list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                        list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                        noise[:, freenoise_offset + place_idx:freenoise_offset + place_idx + delta, :, :] = noise[:, freenoise_offset + list_idx, :, :]

                log.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
                from .context_windows.context import get_context_scheduler, create_window_mask, WindowTracker
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])
                context = get_context_scheduler(context_schedule)
            else:
                log.info("Context frames is larger than total num_frames, disabling context windows")
                context_options = None

        #MTV Crafter
        mtv_input = image_embeds.get("mtv_crafter_motion", None)
        mtv_motion_tokens = None
        if mtv_input is not None:
            from .MTV.mtv import prepare_motion_embeddings
            log.info("Using MTV Crafter embeddings")
            mtv_start_percent = mtv_input.get("start_percent", 0.0)
            mtv_end_percent = mtv_input.get("end_percent", 1.0)
            mtv_strength = mtv_input.get("strength", 1.0)
            mtv_motion_tokens = mtv_input.get("mtv_motion_tokens", None)
            if not isinstance(mtv_strength, list):
                mtv_strength = [mtv_strength] * (steps + 1)
            d = transformer.dim // transformer.num_heads
            mtv_freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
            motion_rotary_emb = prepare_motion_embeddings(
                latent_video_length if context_options is None else context_frames,
                24, mtv_input["global_mean"], [mtv_input["global_std"]], device=device)
            log.info(f"mtv_motion_rotary_emb: {motion_rotary_emb[0].shape}")
            mtv_freqs = mtv_freqs.to(device, dtype)

        #region S2V
        s2v_audio_input = s2v_ref_latent = s2v_pose = s2v_ref_motion = None
        framepack = False
        s2v_audio_embeds = image_embeds.get("audio_embeds", None)
        if s2v_audio_embeds is not None:
            log.info("Using S2V audio embeddings")
            framepack = s2v_audio_embeds.get("enable_framepack", False)
            if framepack and context_options is not None:
                raise ValueError("S2V framepack and context windows cannot be used at the same time")

            s2v_audio_input = s2v_audio_embeds.get("audio_embed_bucket", None)
            if s2v_audio_input is not None:
                #s2v_audio_input = s2v_audio_input[..., 0:image_embeds["num_frames"]]
                s2v_audio_input = s2v_audio_input.to(device, dtype)
            s2v_audio_scale = s2v_audio_embeds["audio_scale"]
            s2v_ref_latent = s2v_audio_embeds.get("ref_latent", None)
            if s2v_ref_latent is not None:
                s2v_ref_latent = s2v_ref_latent.to(device, dtype)
            s2v_ref_motion = s2v_audio_embeds.get("ref_motion", None)
            if s2v_ref_motion is not None:
                s2v_ref_motion = s2v_ref_motion.to(device, dtype)
            s2v_pose = s2v_audio_embeds.get("pose_latent", None)
            if s2v_pose is not None:
                s2v_pose = s2v_pose.to(device, dtype)
            s2v_pose_start_percent = s2v_audio_embeds.get("pose_start_percent", 0.0)
            s2v_pose_end_percent = s2v_audio_embeds.get("pose_end_percent", 1.0)
            s2v_num_repeat = s2v_audio_embeds.get("num_repeat", 1)
            vae = s2v_audio_embeds.get("vae", None)

        # vid2vid
        noise_mask = original_image = scail_freeze_mask = None
        if samples is not None and not multitalk_sampling and not wananimate_loop and not everanimate_sampling and not scail2_looping:
            saved_generator_state = samples.get("generator_state", None)
            if saved_generator_state is not None:
                seed_g.set_state(saved_generator_state)
            input_samples = samples.get("samples", None)
            if input_samples is not None:
                input_samples = input_samples.squeeze(0).to(noise)
                if input_samples.shape[1] != noise.shape[1]:
                    input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)

                if add_noise_to_samples:
                    latent_timestep = timesteps[:1].to(noise)
                    noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                else:
                    noise = input_samples

                noise_mask = samples.get("noise_mask", None)
                if noise_mask is not None:
                    log.info(f"Latent noise_mask shape: {noise_mask.shape}")
                    original_image = samples.get("original_image", None)
                    if original_image is None:
                        original_image = input_samples
                    if len(noise_mask.shape) == 4:
                        noise_mask = noise_mask.squeeze(1)
                    if noise_mask.shape[0] < noise.shape[1]:
                        noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)

                    noise_mask = torch.nn.functional.interpolate(
                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                        mode='trilinear',
                        align_corners=False
                    ).repeat(1, noise.shape[0], 1, 1, 1)

        scail_freeze_latents = image_embeds.get("scail_freeze_latents", None)
        if scail_freeze_latents is not None and not multitalk_sampling and not wananimate_loop and not everanimate_sampling and not scail2_looping:
            freeze_latents = scail_freeze_latents
            if freeze_latents.ndim == 5:
                freeze_latents = freeze_latents.squeeze(0)
            if freeze_latents.shape[0] != noise.shape[0]:
                raise ValueError(f"SCAIL-2 freeze latent channels {freeze_latents.shape[0]} do not match target channels {noise.shape[0]}")

            freeze_latents = freeze_latents.to(noise)
            if freeze_latents.shape[1] < noise.shape[1]:
                pad = torch.zeros(
                    freeze_latents.shape[0], noise.shape[1] - freeze_latents.shape[1],
                    freeze_latents.shape[2], freeze_latents.shape[3],
                    device=freeze_latents.device, dtype=freeze_latents.dtype,
                )
                freeze_latents = torch.cat([freeze_latents, pad], dim=1)
            elif freeze_latents.shape[1] > noise.shape[1]:
                freeze_latents = freeze_latents[:, :noise.shape[1]]

            if freeze_latents.shape[2:] != noise.shape[2:]:
                freeze_latents = torch.nn.functional.interpolate(
                    freeze_latents.unsqueeze(0),
                    size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                    mode="trilinear",
                    align_corners=False,
                )[0]

            raw_freeze_mask = image_embeds.get("scail_freeze_mask", None)
            if raw_freeze_mask is None:
                raw_freeze_mask = torch.ones(
                    freeze_latents.shape[1], freeze_latents.shape[2], freeze_latents.shape[3],
                    device=freeze_latents.device, dtype=freeze_latents.dtype,
                )
            if raw_freeze_mask.ndim == 5:
                raw_freeze_mask = raw_freeze_mask.squeeze(0).squeeze(0)
            elif raw_freeze_mask.ndim == 4:
                raw_freeze_mask = raw_freeze_mask.squeeze(0) if raw_freeze_mask.shape[0] == 1 else raw_freeze_mask[0]
            raw_freeze_mask = raw_freeze_mask.to(noise)
            if raw_freeze_mask.shape[0] < noise.shape[1]:
                pad = torch.zeros(
                    noise.shape[1] - raw_freeze_mask.shape[0], raw_freeze_mask.shape[1], raw_freeze_mask.shape[2],
                    device=raw_freeze_mask.device, dtype=raw_freeze_mask.dtype,
                )
                raw_freeze_mask = torch.cat([raw_freeze_mask, pad], dim=0)
            elif raw_freeze_mask.shape[0] > noise.shape[1]:
                raw_freeze_mask = raw_freeze_mask[:noise.shape[1]]

            scail_freeze_mask = torch.nn.functional.interpolate(
                raw_freeze_mask.unsqueeze(0).unsqueeze(0),
                size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                mode="trilinear",
                align_corners=False,
            ).repeat(1, noise.shape[0], 1, 1, 1).to(device)

            base_noise = noise.to(device)
            original_image = freeze_latents.to(device)
            freeze = scail_freeze_mask[0].to(base_noise)
            noise = (original_image * freeze + base_noise * (1 - freeze)).detach()
            log.info(f"SCAIL-2 freeze_mask active: protecting {int((raw_freeze_mask > 0).any(dim=(1, 2)).sum().item())} latent frames")

        # extra latents (Pusa) and 5b
        latents_to_insert = add_index = noise_multipliers = None
        extra_latents = image_embeds.get("extra_latents", None)
        clean_latent_indices = []
        noise_multiplier_list = image_embeds.get("pusa_noise_multipliers", None)
        if noise_multiplier_list is not None:
            if len(noise_multiplier_list) != latent_video_length:
                noise_multipliers = torch.zeros(latent_video_length)
            else:
                noise_multipliers = torch.tensor(noise_multiplier_list)
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")
        if extra_latents is not None and transformer.multitalk_model_type.lower() != "infinitetalk":
            if noise_multiplier_list is not None:
                noise_multiplier_list = list(noise_multiplier_list) + [1.0] * (len(clean_latent_indices) - len(noise_multiplier_list))
            for i, entry in enumerate(extra_latents):
                add_index = entry["index"]
                num_extra_frames = entry["samples"].shape[2]
                # Handle negative indices
                if add_index < 0:
                    add_index = noise.shape[1] + add_index
                add_index = max(0, min(add_index, noise.shape[1] - num_extra_frames))
                if start_step == 0:
                    noise[:, add_index:add_index+num_extra_frames] = entry["samples"].to(noise)
                    log.info(f"Adding extra samples to latent indices {add_index} to {add_index+num_extra_frames-1}")
                clean_latent_indices.extend(range(add_index, add_index+num_extra_frames))
            if noise_multipliers is not None and len(noise_multiplier_list) != latent_video_length:
                for i, idx in enumerate(clean_latent_indices):
                    noise_multipliers[idx] = noise_multiplier_list[i]
                log.info(f"Using Pusa noise multipliers: {noise_multipliers}")

        # lucy edit
        extra_channel_latents = image_embeds.get("extra_channel_latents", None)
        if extra_channel_latents is not None:
            extra_channel_latents = extra_channel_latents[0].to(noise)

        # FlashVSR
        flashvsr_LQ_latent = LQ_images = None
        flashvsr_LQ_images = image_embeds.get("flashvsr_LQ_images", None)
        flashvsr_strength = image_embeds.get("flashvsr_strength", 1.0)
        if flashvsr_LQ_images is not None:
            flashvsr_LQ_images = flashvsr_LQ_images[:num_frames]
            first_frame = flashvsr_LQ_images[:1]
            last_frame = flashvsr_LQ_images[-1:].repeat(3, 1, 1, 1)
            flashvsr_LQ_images = torch.cat([first_frame, flashvsr_LQ_images, last_frame], dim=0)
            LQ_images = flashvsr_LQ_images.unsqueeze(0).movedim(-1, 1).to(dtype) * 2 - 1
            if context_options is None:
                flashvsr_LQ_latent = transformer.LQ_proj_in(LQ_images.to(device))
                log.info(f"flashvsr_LQ_latent: {flashvsr_LQ_latent[0].shape}")
                seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

        latent = noise

        # LongCat-Avatar
        longcat_ref_latent = None
        longcat_num_ref_latents = longcat_num_cond_latents = 0
        longcat_avatar_options = image_embeds.get("longcat_avatar_options", None)

        if longcat_avatar_options is not None:
            longcat_ref_latent = longcat_avatar_options.get("longcat_ref_latent", None)
            if longcat_ref_latent is not None:
                log.info(f"LongCat-Avatar reference latent shape: {longcat_ref_latent.shape}")
                latent = torch.cat([longcat_ref_latent.to(latent), latent], dim=1)
                seq_len = math.ceil((latent.shape[2] * latent.shape[3]) / 4 * latent.shape[1])
                insert_len = longcat_ref_latent.shape[1]
                clean_latent_indices = list(range(0, insert_len)) + [i + insert_len for i in clean_latent_indices]
                longcat_num_ref_latents = longcat_ref_latent.shape[1]
                latent_video_length += insert_len
            longcat_num_cond_latents = len(clean_latent_indices)
            log.info(f"LongCat num_cond_latents: {longcat_num_cond_latents} num_ref_latents: {longcat_num_ref_latents}")
        # v1.5 (Whisper) embeds set audio_stride=1; v1.0 (wav2vec2) uses 2 for LongCat
        if multitalk_audio_stride is not None:
            audio_stride = multitalk_audio_stride
        else:
            audio_stride = 2 if transformer.is_longcat else 1

        #controlnet
        controlnet_latents = controlnet = None
        if transformer_options is not None:
            controlnet = transformer_options.get("controlnet", None)
            if controlnet is not None:
                self.controlnet = controlnet["controlnet"]
                controlnet_start = controlnet["controlnet_start"]
                controlnet_end = controlnet["controlnet_end"]
                controlnet_latents = controlnet["control_latents"]
                controlnet["controlnet_weight"] = controlnet["controlnet_strength"]
                controlnet["controlnet_stride"] = controlnet["control_stride"]

        #uni3c
        uni3c_data = uni3c_data_input = None
        if uni3c_embeds is not None:
            transformer.uni3c_controlnet = uni3c_embeds["controlnet"]
            render_latent = uni3c_embeds["render_latent"].to(offload_device if scail2_looping else device)
            uni3c_data = uni3c_embeds.copy()
            if not scail2_looping and render_latent.shape != noise.shape:
                # If temporal is shorter (prefix/transition expansion), pad with first frame at beginning
                if has_prefix and render_latent.shape[2] < noise.shape[1]:
                    pad_len = noise.shape[1] - render_latent.shape[2]
                    first_frame = render_latent[:, :, :1].repeat(1, 1, pad_len, 1, 1)
                    render_latent = torch.cat([first_frame, render_latent], dim=2)
                    log.info(f"Uni3C: padded render_latent by {pad_len} latent frames for prefix canvas expansion")
                if render_latent.shape != noise.shape:
                    render_latent = torch.nn.functional.interpolate(render_latent, size=(noise.shape[1], noise.shape[2], noise.shape[3]), mode='trilinear', align_corners=False)
            uni3c_data["render_latent"] = render_latent

        # Enhance-a-video (feta)
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            set_num_frames(latent_video_length) if context_options is None else set_num_frames(context_frames)
            enhance_enabled = True
        else:
            feta_args = None
            enhance_enabled = False

        # EchoShot https://github.com/D2I-ai/EchoShot
        echoshot = False
        shot_len = None
        if text_embeds is not None:
            echoshot = text_embeds.get("echoshot", False)
        if echoshot:
            shot_num = len(text_embeds["prompt_embeds"])
            shot_len = [latent_video_length//shot_num] * (shot_num-1)
            shot_len.append(latent_video_length-sum(shot_len))
            rope_function = "default" #echoshot does not support comfy rope function
            log.info(f"Number of shots in prompt: {shot_num}, Shot token lengths: {shot_len}")

        # Bindweave
        qwenvl_embeds_pos = image_embeds.get("qwenvl_embeds_pos", None)
        qwenvl_embeds_neg = image_embeds.get("qwenvl_embeds_neg", None)

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()

        #blockswap init
        init_blockswap(transformer, block_swap_args, model)

        # Initialize Cache if enabled
        previous_cache_states = None
        transformer.enable_teacache = transformer.enable_magcache = transformer.enable_easycache = False
        cache_args = teacache_args if teacache_args is not None else cache_args #for backward compatibility on old workflows
        if cache_args is not None:
            from .cache_methods.cache_methods import set_transformer_cache_method
            transformer = set_transformer_cache_method(transformer, timesteps, cache_args)

            # Initialize cache state
            if samples is not None:
                previous_cache_states = samples.get("cache_states", None)
                if previous_cache_states is not None:
                    log.info("Using cache states from previous sampler")
                    self.cache_state = previous_cache_states["cache_state"]
                    transformer.easycache_state = previous_cache_states["easycache_state"]
                    transformer.magcache_state = previous_cache_states["magcache_state"]
                    transformer.teacache_state = previous_cache_states["teacache_state"]

        if previous_cache_states is None:
            self.cache_state = [None, None]
            if phantom_latents is not None:
                log.info(f"Phantom latents shape: {phantom_latents.shape}")
                self.cache_state = [None, None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []

        # Skip layer guidance (SLG)
        if slg_args is not None:
            assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        # Setup radial attention
        if transformer.attention_mode == "radial_sage_attention":
            setup_radial_attention(transformer, transformer_options, latent, seq_len, latent_video_length, context_options=context_options)

        # Experimental args
        use_cfg_zero_star = use_tangential = use_fresca = bidirectional_sampling = use_tsr = False
        raag_alpha = 0.0
        transformer.video_attention_split_steps = []
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]

            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            use_tangential = experimental_args.get("use_tcfg", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)
            raag_alpha = experimental_args.get("raag_alpha", 0.0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

            bidirectional_sampling = experimental_args.get("bidirectional_sampling", False)
            if bidirectional_sampling:
                sample_scheduler_flipped = copy.deepcopy(sample_scheduler)
            use_tsr = experimental_args.get("temporal_score_rescaling", False)
            tsr_k = experimental_args.get("tsr_k", 1.0)
            tsr_sigma = experimental_args.get("tsr_sigma", 1.0)

        scail2_fast_path_blocked_by_experimental_sampling = (
            use_cfg_zero_star
            or use_tangential
            or raag_alpha > 0.0
            or use_fresca
            or use_tsr
            or bool(transformer.video_attention_split_steps)
        )

        # Rotary positional embeddings (RoPE)

        # RoPE base freq scaling as used with CineScale
        ntk_alphas = [1.0, 1.0, 1.0]
        if isinstance(rope_function, dict):
            ntk_alphas = rope_function["ntk_scale_f"], rope_function["ntk_scale_h"], rope_function["ntk_scale_w"]
            rope_function = rope_function["rope_function"]

        # Stand-In
        standin_input = image_embeds.get("standin_input", None)
        if standin_input is not None:
            rope_function = "comfy" # only works with this currently
        if context_latents is not None and "comfy" not in rope_function:
            log.info("Bernini context requires comfy RoPE; overriding rope_function to comfy")
            rope_function = "comfy"
        scail_embeds_for_rope = image_embeds.get("scail_embeds", None)
        if scail_embeds_for_rope is not None and "comfy" not in rope_function:
            log.info("SCAIL-2 requires comfy RoPE; overriding rope_function to comfy")
            rope_function = "comfy"

        freqs = None

        log.info(f"Rope function: {rope_function}")

        riflex_freq_index = 0 if riflex_freq_index is None else riflex_freq_index
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        d = transformer.dim // transformer.num_heads

        if mocha_embeds is not None and context_latents is None:
            from .mocha.nodes import rope_params_mocha
            log.info("Using Mocha RoPE")
            rope_function = 'mocha'

            freqs = torch.cat([
                rope_params_mocha(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index, start=-1),
                rope_params_mocha(1024, 2 * (d // 6), start=-1),
                rope_params_mocha(1024, 2 * (d // 6), start=-1)
            ],
            dim=1)
        elif context_latents is None and ("default" in rope_function or bidirectional_sampling): # original RoPE
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
        elif "comfy" in rope_function: # comfy's rope
            transformer.rope_embedder.k = riflex_freq_index
            transformer.rope_embedder.num_frames = latent_video_length

        transformer.rope_func = rope_function
        for block in transformer.blocks:
            block.rope_func = rope_function
        if transformer.vace_layers is not None:
            for block in transformer.vace_blocks:
                block.rope_func = rope_function

        # Lynx
        lynx_ref_buffer = None
        lynx_embeds = image_embeds.get("lynx_embeds", None)
        if lynx_embeds is not None:
            if lynx_embeds.get("ip_x", None) is not None:
                if transformer.blocks[0].cross_attn.ip_adapter is None:
                    raise ValueError("Lynx IP embeds provided, but the no lynx ip adapter layers found in the model.")
            lynx_embeds = lynx_embeds.copy()
            log.info("Using Lynx embeddings", lynx_embeds)
            lynx_ref_latent = lynx_embeds.get("ref_latent", None)
            lynx_ref_latent_uncond = lynx_embeds.get("ref_latent_uncond", None)
            lynx_ref_text_embed = lynx_embeds.get("ref_text_embed", None)
            lynx_ref_text_embed = dict_to_device(lynx_ref_text_embed, device)
            lynx_cfg_scale = lynx_embeds.get("cfg_scale", 1.0)
            if not isinstance(lynx_cfg_scale, list):
                lynx_cfg_scale = [lynx_cfg_scale] * (steps + 1)

            if lynx_ref_latent is not None:
                if transformer.blocks[0].self_attn.ref_adapter is None:
                    raise ValueError("Lynx reference provided, but the no lynx reference adapter layers found in the model.")
                lynx_ref_latent = lynx_ref_latent[0]
                lynx_ref_latent_uncond = lynx_ref_latent_uncond[0]
                lynx_embeds["ref_feature_extractor"] = True
                log.info(f"Lynx ref latent shape: {lynx_ref_latent.shape}")
                log.info("Extracting Lynx ref cond buffer...")
                if transformer.in_dim == 36:
                    mask_latents = torch.tile(torch.zeros_like(lynx_ref_latent[:1]), [4, 1, 1, 1])
                    empty_image_cond = torch.cat([mask_latents, torch.zeros_like(lynx_ref_latent)], dim=0).to(device)
                    lynx_ref_input = torch.cat([lynx_ref_latent, empty_image_cond], dim=0)
                else:
                    lynx_ref_input = lynx_ref_latent
                lynx_ref_buffer = transformer(
                    [lynx_ref_input.to(device, dtype)],
                    torch.tensor([0], device=device),
                    lynx_ref_text_embed["prompt_embeds"],
                    seq_len=math.ceil((lynx_ref_latent.shape[2] * lynx_ref_latent.shape[3]) / 4 * lynx_ref_latent.shape[1]),
                    lynx_embeds=lynx_embeds
                )
                log.info(f"Extracted {len(lynx_ref_buffer)} cond ref buffers")
                if any(not math.isclose(c, 1.0) for c in cfg):
                    log.info("Extracting Lynx ref uncond buffer...")
                    if transformer.in_dim == 36:
                        lynx_ref_input_uncond = torch.cat([lynx_ref_latent_uncond, empty_image_cond], dim=0)
                    else:
                        lynx_ref_input_uncond = lynx_ref_latent_uncond
                    lynx_ref_buffer_uncond = transformer(
                        [lynx_ref_input_uncond.to(device, dtype)],
                        torch.tensor([0], device=device),
                        lynx_ref_text_embed["prompt_embeds"],
                        seq_len=math.ceil((lynx_ref_latent.shape[2] * lynx_ref_latent.shape[3]) / 4 * lynx_ref_latent.shape[1]),
                        lynx_embeds=lynx_embeds,
                        is_uncond=True
                    )
                    log.info(f"Extracted {len(lynx_ref_buffer_uncond)} uncond ref buffers")

            if lynx_embeds.get("ip_x", None) is not None:
                lynx_embeds["ip_x"] = lynx_embeds["ip_x"].to(device, dtype)
                lynx_embeds["ip_x_uncond"] = lynx_embeds["ip_x_uncond"].to(device, dtype)
            lynx_embeds["ref_feature_extractor"] = False
            lynx_embeds["ref_latent"] = lynx_embeds["ref_text_embed"] = None
            lynx_embeds["ref_buffer"] = lynx_ref_buffer
            lynx_embeds["ref_buffer_uncond"] = lynx_ref_buffer_uncond if not math.isclose(cfg[0], 1.0) else None
            mm.soft_empty_cache()

        # UniLumos
        foreground_latents = image_embeds.get("foreground_latents", None)
        if foreground_latents is not None:
            log.info(f"UniLumos foreground latent input shape: {foreground_latents.shape}")
            foreground_latents = foreground_latents.to(device, dtype)
        background_latents = image_embeds.get("background_latents", None)
        if background_latents is not None:
            log.info(f"UniLumos background latent input shape: {background_latents.shape}")
            background_latents = background_latents.to(device, dtype)

        #Time-to-move (TTM)
        ttm_start_step = 0
        ttm_reference_latents = image_embeds.get("ttm_reference_latents", None)
        if ttm_reference_latents is not None:
            motion_mask = image_embeds["ttm_mask"].to(device, dtype)
            ttm_start_step = max(image_embeds["ttm_start_step"] - start_step, 0)
            ttm_end_step = image_embeds["ttm_end_step"] - start_step

            if ttm_start_step > steps:
                raise ValueError("TTM start step is beyond the total number of steps")

            if ttm_end_step > ttm_start_step:
                log.info("Using Time-to-move (TTM)")
                log.info(f"TTM reference latents shape: {ttm_reference_latents.shape}")
                log.info(f"TTM motion mask shape: {motion_mask.shape}")
                log.info(f"Applying TTM from step {ttm_start_step} to {ttm_end_step}")

                latent = add_noise(ttm_reference_latents, noise, timesteps[ttm_start_step].to(noise.device)).to(latent)

        # SteadyDancer
        sdancer_embeds = image_embeds.get("sdancer_embeds", None)
        sdancer_data = sdancer_input = None
        if sdancer_embeds is not None:
            log.info("Using SteadyDancer embeddings:")
            for k, v in sdancer_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            sdancer_data = sdancer_embeds.copy()
            sdancer_data = dict_to_device(sdancer_data, device, dtype)

        # One-to-all-Animation
        one_to_all_embeds = image_embeds.get("one_to_all_embeds", None)
        one_to_all_data = prev_latents = None
        latents_to_not_step = 0
        if one_to_all_embeds is not None:
            log.info("Using One-to-All embeddings:")
            for k, v in one_to_all_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            one_to_all_data = one_to_all_embeds.copy()
            one_to_all_data = dict_to_device(one_to_all_data, device, dtype)
            if one_to_all_embeds.get("pose_images") is not None:
                transformer.input_hint_block.to(device)
                pose_images_in = one_to_all_data.pop("pose_images")
                pose_images = transformer.input_hint_block(pose_images_in)
                if one_to_all_embeds.get("ref_latent_pos") is not None:
                    pose_prefix_image = transformer.input_hint_block(one_to_all_data.pop("pose_prefix_image"))
                    pose_images = torch.cat([pose_prefix_image, pose_images],dim=2)
                one_to_all_data["controlnet_tokens"] = pose_images.flatten(2).transpose(1, 2)
                transformer.input_hint_block.to(offload_device)

                one_to_all_pose_cfg_scale = one_to_all_embeds.get("pose_cfg_scale", 1.0)
                if not isinstance(one_to_all_pose_cfg_scale, list):
                    one_to_all_pose_cfg_scale = [one_to_all_pose_cfg_scale] * (steps + 1)

            prev_latents = one_to_all_data.get("prev_latents", None)
            if prev_latents is not None:
                log.info(f"Using previous latents for One-to-All Animation with shape: {prev_latents.shape}")
                latent[:, :prev_latents.shape[1]] = prev_latents.to(latent)
                one_to_all_data["token_replace"] = True
                latents_to_not_step = prev_latents.shape[1]
                one_to_all_data["num_latent_frames_to_replace"] = latents_to_not_step

        # SCAIL
        scail_embeds = image_embeds.get("scail_embeds", None)
        scail_data = None
        scail_context_windowed = scail_embeds is not None and context_options is not None and not scail2_looping
        if scail_embeds is not None:
            log.info("Using SCAIL embeddings:")
            for k, v in scail_embeds.items():
                log.info(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else v}")
            scail_data = scail_embeds.copy()
            if not scail_context_windowed:
                scail_data = dict_to_device(scail_data, device, dtype)
        scail2_fast_path_fallback_logged = False

        def _scail_index_dim1(tensor, indices):
            index = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
            return tensor.index_select(1, index)

        def _scail_to_like(value, like):
            if isinstance(value, torch.Tensor):
                return value.to(device=like.device, dtype=like.dtype)
            return value

        def _scail_window_dim1_to_like(tensor, indices, prepend_count, like):
            parts = []
            if prepend_count > 0:
                parts.append(tensor[:, :prepend_count].to(device=like.device, dtype=like.dtype))
            parts.append(_scail_index_dim1(tensor, indices).to(device=like.device, dtype=like.dtype))
            return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        def _scail_context_data_to_like(active_data, indices, prepend_count, like):
            out = {}
            windowed_keys = {"pose_latent", "sam_latents", "ref_mask_latents"}
            for key, value in active_data.items():
                if key in windowed_keys:
                    continue
                out[key] = _scail_to_like(value, like)

            pose_latent = active_data.get("pose_latent", None)
            if pose_latent is not None:
                out["pose_latent"] = _scail_window_dim1_to_like(pose_latent, indices, prepend_count, like)

            sam_latents = active_data.get("sam_latents", None)
            if sam_latents is not None:
                out["sam_latents"] = _scail_window_dim1_to_like(sam_latents, indices, prepend_count, like)

            ref_mask_latents = active_data.get("ref_mask_latents", None)
            if ref_mask_latents is not None:
                ref_latent = active_data.get("ref_latent_pos", active_data.get("ref_latent_neg", None))
                ref_count = min(ref_latent.shape[1], ref_mask_latents.shape[1]) if ref_latent is not None else 0
                ref_parts = []
                if ref_count > 0:
                    ref_parts.append(ref_mask_latents[:, :ref_count].to(device=like.device, dtype=like.dtype))
                ref_mask_target = ref_mask_latents[:, ref_count:]
                if ref_mask_target.shape[1] > 0:
                    ref_parts.append(_scail_window_dim1_to_like(ref_mask_target, indices, prepend_count, like))
                if ref_parts:
                    out["ref_mask_latents"] = torch.cat(ref_parts, dim=1) if len(ref_parts) > 1 else ref_parts[0]
                else:
                    out["ref_mask_latents"] = ref_mask_latents[:, :0].to(device=like.device, dtype=like.dtype)

            return out


        # WanMove
        wanmove_embeds = None
        if image_cond is not None:
            wanmove_embeds = image_embeds.get("wanmove_embeds", None)
            if wanmove_embeds is not None:
                track_pos = wanmove_embeds["track_pos"]
                if any(not math.isclose(c, 1.0) for c in cfg):
                    image_cond_neg = torch.cat([image_embeds["mask"], image_cond])
                if context_options is None:
                    image_cond = replace_feature(image_cond.unsqueeze(0).clone(), track_pos.unsqueeze(0), wanmove_embeds.get("strength", 1.0))[0]

        # LongVie2 dual control
        dual_control_embeds = image_embeds.get("dual_control", None)
        dual_control_input = None
        if dual_control_embeds is not None and context_options is None:
            dual_control_input = dict_to_device(dual_control_embeds.copy(), device, dtype) if dual_control_embeds is not None else None
            prev_latents = dual_control_input.get("prev_latent", None)
            if prev_latents is not None:
                _sigma = dual_control_embeds.get("first_frame_noise_level", 0.925926)
                log.info(f"Using dual control previous latents with first frame noise level: {_sigma}")
                latent[:, :1] = (1 - _sigma) * prev_latents[:, -1:].to(latent) + _sigma * noise[:, :1]
                prev_ones = torch.ones(20, *prev_latents.shape[1:], device=device, dtype=dtype)
                dual_control_input["prev_latent"] = torch.cat([prev_ones, prev_latents]).unsqueeze(0)

        # APG momentum buffers (persist across sampling steps)
        _momentum_buf = MomentumBuffer(apg_momentum) if guidance_mode == "apg" else None
        _momentum_buf_I  = MomentumBuffer(apg_momentum) if guidance_mode == "apg_chain" else None
        _momentum_buf_TI = MomentumBuffer(apg_momentum) if guidance_mode == "apg_chain" else None

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None,
                             control_latents=None, vace_data=None, unianim_data=None, audio_proj=None, control_camera_latents=None,
                             add_cond=None, cache_state=None, context_window=None, multitalk_audio_embeds=None, fantasy_portrait_input=None, reverse_time=False,
                             mtv_motion_tokens=None, s2v_audio_input=None, s2v_ref_motion=None, s2v_motion_frames=[1, 0], s2v_pose=None,
                             humo_image_cond=None, humo_image_cond_neg=None, humo_audio=None, humo_audio_neg=None, wananim_pose_latents=None,
                             wananim_face_pixels=None, wananim_num_anchor_latents=1, uni3c_data=None, latent_model_input_ovi=None, flashvsr_LQ_latent=None,
                             context_latents=None, context_roles=None, context_window_start=0, scail_context_prepend_latents=0,
                             scail_data_override=None,):
            nonlocal transformer
            nonlocal audio_cfg_scale
            nonlocal scail2_fast_path_fallback_logged

            autocast_enabled = ("fp8" in model["quantization"] and not transformer.patched_linear)
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype) if autocast_enabled else nullcontext():

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if control_embeds is not None and control_camera_latents is None:
                    if control_lora:
                        control_lora_enabled = True
                    else:
                        if ((control_start_percent <= current_step_percentage <= control_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_start_percent)) and \
                            (control_latents is not None):
                            image_cond_input = torch.cat([control_latents.to(z), image_cond.to(z)])
                        else:
                            image_cond_input = torch.cat([torch.zeros_like(noise, device=device, dtype=dtype), image_cond.to(z)])
                        if fun_ref_image is not None:
                            fun_ref_input = fun_ref_image.to(z)
                        else:
                            fun_ref_input = torch.zeros_like(z, dtype=z.dtype)[:, 0].unsqueeze(1)

                    if control_lora:
                        if not control_start_percent <= current_step_percentage <= control_end_percent:
                            control_lora_enabled = False
                            if patcher.model.is_patched:
                                log.info("Unloading LoRA...")
                                patcher.unpatch_model(device)
                                patcher.model.is_patched = False
                        else:
                            image_cond_input = control_latents.to(z)
                            if not patcher.model.is_patched:
                                log.info("Loading LoRA...")
                                patcher = apply_lora(patcher, device, device, low_mem_load=False, control_lora=True)
                                patcher.model.is_patched = True

                elif ATI_tracks is not None and ((ati_start_percent <= current_step_percentage <= ati_end_percent) or
                              (ati_end_percent > 0 and idx == 0 and current_step_percentage >= ati_start_percent)):
                    image_cond_input = image_cond_ati.to(z)
                elif humo_image_cond is not None:
                    humo_image_cond_neg_input = None
                    if context_window is not None:
                        image_cond_input = humo_image_cond[:, context_window].to(z)
                        humo_image_cond_neg_input = humo_image_cond_neg[:, context_window].to(z)
                        if humo_reference_count > 0:
                            image_cond_input[:, -humo_reference_count:] = humo_image_cond[:, -humo_reference_count:]
                            humo_image_cond_neg_input[:, -humo_reference_count:] = humo_image_cond_neg[:, -humo_reference_count:]
                    else:
                        if image_cond is not None:
                            image_cond_input = image_cond.to(z)
                            if humo_reference_count > 0:
                                image_cond_input = torch.cat([image_cond_input, humo_image_cond[:, -humo_reference_count:].to(z)], dim=1)
                                humo_image_cond_neg_input = torch.cat([image_cond_input, humo_image_cond_neg[:, -humo_reference_count:].to(z)], dim=1)
                        else:
                            image_cond_input = humo_image_cond.to(z)
                            humo_image_cond_neg_input = humo_image_cond_neg.to(z)

                elif image_cond is not None:
                    if reverse_time: # Flip the image condition
                        image_cond_input = torch.cat([
                            torch.flip(image_cond[:4], dims=[1]),
                            torch.flip(image_cond[4:], dims=[1])
                        ]).to(z)
                    else:
                        image_cond_input = image_cond.to(z)

                if control_camera_latents is not None:
                    if (control_camera_start_percent <= current_step_percentage <= control_camera_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_camera_start_percent):
                        control_camera_input = control_camera_latents.to(device, dtype)
                    else:
                        control_camera_input = None

                if recammaster is not None:
                    z = torch.cat([z, recam_latents.to(z)], dim=1)

                if mocha_embeds is not None:
                    if context_window is not None and mocha_embeds.shape[2] != context_frames:
                        latent_frames = len(context_window)
                        # [latent_frames, 1 mask frame, mocha_num_refs]
                        latent_end = latent_frames
                        mask_end = latent_end + 1
                        partial_latents = mocha_embeds[:, context_window]  # windowed latents
                        mask_frame = mocha_embeds[:, latent_end:mask_end]  # single mask frame
                        ref_frames = mocha_embeds[:, -mocha_num_refs:]     # reference frames

                        partial_mocha_embeds = torch.cat([partial_latents, mask_frame, ref_frames], dim=1)
                        z = torch.cat([z, partial_mocha_embeds.to(z)], dim=1)
                    else:
                        z = torch.cat([z, mocha_embeds.to(z)], dim=1)

                if mtv_input is not None:
                    if ((mtv_start_percent <= current_step_percentage <= mtv_end_percent) or \
                            (mtv_end_percent > 0 and idx == 0 and current_step_percentage >= mtv_start_percent)):
                        mtv_motion_tokens = mtv_motion_tokens.to(z)
                        mtv_motion_rotary_emb = motion_rotary_emb

                use_phantom = False
                phantom_ref = None
                if phantom_latents is not None:
                    if (phantom_start_percent <= current_step_percentage <= phantom_end_percent) or \
                        (phantom_end_percent > 0 and idx == 0 and current_step_percentage >= phantom_start_percent):
                        phantom_ref = phantom_latents.to(z)
                        use_phantom = True
                        if cache_state is not None and len(cache_state) != 3:
                            cache_state.append(None)

                if controlnet_latents is not None:
                    if (controlnet_start <= current_step_percentage < controlnet_end):
                        self.controlnet.to(device)
                        controlnet_states = self.controlnet(
                            hidden_states=z.unsqueeze(0).to(device, self.controlnet.dtype),
                            timestep=timestep,
                            encoder_hidden_states=positive_embeds[0].unsqueeze(0).to(device, self.controlnet.dtype),
                            attention_kwargs=None,
                            controlnet_states=controlnet_latents.to(device, self.controlnet.dtype),
                            return_dict=False,
                        )[0]
                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet["controlnet_states"] = [x.to(z) for x in controlnet_states]
                        else:
                            controlnet["controlnet_states"] = controlnet_states.to(z)

                add_cond_input = None
                if add_cond is not None:
                    if (add_cond_start_percent <= current_step_percentage <= add_cond_end_percent) or \
                        (add_cond_end_percent > 0 and idx == 0 and current_step_percentage >= add_cond_start_percent):
                        add_cond_input = add_cond

                if minimax_latents is not None:
                    if context_window is not None:
                        z = torch.cat([z, minimax_latents[:, context_window], minimax_mask_latents[:, context_window]], dim=0)
                    else:
                        z = torch.cat([z, minimax_latents, minimax_mask_latents], dim=0)

                multitalk_audio_input = None
                if audio_emb_slice is not None:
                    multitalk_audio_input = audio_emb_slice.to(z)
                elif not multitalk_sampling and multitalk_audio_embeds is not None:
                    audio_embedding = multitalk_audio_embeds
                    audio_embs = []
                    indices = (torch.arange(4 + 1) - 2) * 1
                    human_num = len(audio_embedding)
                    # split audio with window size
                    audio_end_idx = latent_video_length * 4 + 1 if add_cond is not None else (latent_video_length-1) * 4 + 1
                    audio_end_idx = audio_end_idx * audio_stride
                    if context_window is None:
                        for human_idx in range(human_num):
                            center_indices = torch.arange(0, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)

                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    else:
                        for human_idx in range(human_num):
                            audio_start = (context_window[0] * 4) * audio_stride
                            audio_end = (context_window[-1] * 4 + 1) * audio_stride
                            #print("audio_start: ", audio_start, "audio_end: ", audio_end)
                            center_indices = torch.arange(audio_start, audio_end, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    multitalk_audio_input = torch.concat(audio_embs, dim=0).to(dtype)

                elif multitalk_sampling and multitalk_audio_embeds is not None:
                    multitalk_audio_input = multitalk_audio_embeds

                if context_window is not None and uni3c_data is not None and uni3c_data["render_latent"].shape[2] != context_frames:
                    uni3c_data_input = {"render_latent": uni3c_data["render_latent"][:, :, context_window]}
                    for k in uni3c_data:
                        if k != "render_latent":
                            uni3c_data_input[k] = uni3c_data[k]
                else:
                    uni3c_data_input = uni3c_data

                if context_window is not None and sdancer_data is not None and sdancer_data["cond_pos"].shape[1] != context_frames:
                    sdancer_input = sdancer_data.copy()
                    sdancer_input["cond_pos"] = sdancer_data["cond_pos"][:, context_window]
                    sdancer_input["cond_neg"] = sdancer_data["cond_neg"][:, context_window] if sdancer_data.get("cond_neg", None) is not None else None
                else:
                    sdancer_input = sdancer_data

                if s2v_pose is not None:
                    if not ((s2v_pose_start_percent <= current_step_percentage <= s2v_pose_end_percent) or \
                            (s2v_pose_end_percent > 0 and idx == 0 and current_step_percentage >= s2v_pose_start_percent)):
                        s2v_pose = None


                if humo_audio is not None and ((humo_start_percent <= current_step_percentage <= humo_end_percent) or \
                            (humo_end_percent > 0 and idx == 0 and current_step_percentage >= humo_start_percent)):
                    if context_window is None:
                        humo_audio_input = humo_audio
                        humo_audio_input_neg = humo_audio_neg if humo_audio_neg is not None else None
                    else:
                        humo_audio_input = humo_audio[context_window].to(z)
                        if humo_audio_neg is not None:
                            humo_audio_input_neg = humo_audio_neg[context_window].to(z)
                        else:
                            humo_audio_input_neg = None
                else:
                    humo_audio_input = humo_audio_input_neg = None

                if extra_channel_latents is not None:
                    if context_window is not None:
                        extra_channel_latents_input = extra_channel_latents[:, context_window].to(z)
                    else:
                        extra_channel_latents_input = extra_channel_latents.to(z)
                    z = torch.cat([z, extra_channel_latents_input])

                if "rcm" in sample_scheduler.__class__.__name__.lower():
                    c_in = 1 / (torch.cos(timestep) + torch.sin(timestep))
                    c_noise = (torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))) * 1000
                    z = z * c_in
                    timestep = c_noise

                if image_cond is not None:
                    self.noise_front_pad_num = image_cond_input.shape[1] - z.shape[1]
                    if self.noise_front_pad_num > 0:
                        pad = torch.zeros((z.shape[0], self.noise_front_pad_num, z.shape[2], z.shape[3]), dtype=z.dtype, device=z.device)
                        z = torch.cat([pad, z], dim=1)
                        nonlocal seq_len
                        seq_len = math.ceil((z.shape[2] * z.shape[3]) / 4 * z.shape[1])

                if background_latents is not None or foreground_latents is not None:
                    z = torch.cat([z, foreground_latents.to(z), background_latents.to(z)], dim=0)

                active_scail_data = scail_data_override if scail_data_override is not None else scail_data
                scail_data_in = None
                if active_scail_data is not None:
                    ref_concat_mask = torch.zeros_like(z[:4])
                    if scail_freeze_mask is not None:
                        if context_window is not None:
                            if scail_context_windowed:
                                history_mask = _scail_window_dim1_to_like(
                                    scail_freeze_mask[0, :4],
                                    context_window,
                                    scail_context_prepend_latents,
                                    z,
                                )
                            else:
                                history_mask = scail_freeze_mask[0, :4].to(z)
                                history_parts = []
                                if scail_context_prepend_latents > 0:
                                    history_parts.append(history_mask[:, :scail_context_prepend_latents])
                                history_parts.append(history_mask[:, context_window])
                                history_mask = torch.cat(history_parts, dim=1)
                        else:
                            history_mask = scail_freeze_mask[0, :4].to(z)
                            history_mask = history_mask[:, :z.shape[1]]
                        if history_mask.shape[1] < z.shape[1]:
                            pad = torch.zeros(
                                history_mask.shape[0], z.shape[1] - history_mask.shape[1],
                                history_mask.shape[2], history_mask.shape[3],
                                device=history_mask.device, dtype=history_mask.dtype,
                            )
                            history_mask = torch.cat([history_mask, pad], dim=1)
                        elif history_mask.shape[1] > z.shape[1]:
                            history_mask = history_mask[:, :z.shape[1]]
                        if history_mask.shape[2:] != z.shape[2:]:
                            history_mask = torch.nn.functional.interpolate(
                                history_mask.unsqueeze(0),
                                size=(z.shape[1], z.shape[2], z.shape[3]),
                                mode="trilinear",
                                align_corners=False,
                            )[0]
                        ref_concat_mask = history_mask
                    z = torch.cat([z, ref_concat_mask])
                    if context_window is not None:
                        if scail_context_windowed:
                            scail_data_in = _scail_context_data_to_like(
                                active_scail_data,
                                context_window,
                                scail_context_prepend_latents,
                                z,
                            )
                        else:
                            scail_data_in = active_scail_data.copy()
                            if active_scail_data.get("pose_latent", None) is not None:
                                pose_parts = []
                                if scail_context_prepend_latents > 0:
                                    pose_parts.append(active_scail_data["pose_latent"][:, :scail_context_prepend_latents])
                                pose_parts.append(active_scail_data["pose_latent"][:, context_window])
                                scail_data_in["pose_latent"] = torch.cat(pose_parts, dim=1)
                            if active_scail_data.get("sam_latents", None) is not None:
                                sam_parts = []
                                if scail_context_prepend_latents > 0:
                                    sam_parts.append(active_scail_data["sam_latents"][:, :scail_context_prepend_latents])
                                sam_parts.append(active_scail_data["sam_latents"][:, context_window])
                                scail_data_in["sam_latents"] = torch.cat(sam_parts, dim=1)
                            ref_mask_latents = active_scail_data.get("ref_mask_latents", None)
                            if ref_mask_latents is not None:
                                ref_latent = active_scail_data.get("ref_latent_pos", active_scail_data.get("ref_latent_neg", None))
                                ref_count = min(ref_latent.shape[1], ref_mask_latents.shape[1]) if ref_latent is not None else 0
                                ref_mask_prefix = ref_mask_latents[:, :ref_count]
                                ref_mask_target = ref_mask_latents[:, ref_count:]
                                if ref_mask_target.shape[1] > 0:
                                    target_parts = []
                                    if scail_context_prepend_latents > 0:
                                        target_parts.append(ref_mask_target[:, :scail_context_prepend_latents])
                                    target_parts.append(ref_mask_target[:, context_window])
                                    ref_mask_target = torch.cat(target_parts, dim=1)
                                scail_data_in["ref_mask_latents"] = torch.cat([ref_mask_prefix, ref_mask_target], dim=1)
                    else:
                        scail_data_in = active_scail_data.copy()

                if wanmove_embeds is not None and context_window is not None:
                    image_cond_input = replace_feature(image_cond_input.unsqueeze(0), track_pos[:, context_window].unsqueeze(0), wanmove_embeds.get("strength", 1.0))[0]

                dual_control_in = None
                if dual_control_embeds is not None:
                    if context_window is not None:
                        dual_control_in = dual_control_embeds.copy()
                        dense_input_latent = dual_control_embeds.get("dense_input_latent", None)
                        if dense_input_latent is not None:
                            dual_control_in["dense_input_latent"] = dual_control_embeds["dense_input_latent"][:, :, context_window]
                        sparse_input_latent = dual_control_embeds.get("sparse_input_latent", None)
                        if sparse_input_latent is not None:
                            dual_control_in["sparse_input_latent"] = dual_control_embeds["sparse_input_latent"][:, :, context_window]
                    else:
                        dual_control_in = dual_control_input

                base_params = {
                    'x': [z], # latent
                    'y': [image_cond_input] if image_cond_input is not None else None, # image cond
                    'clip_fea': clip_fea, # clip features
                    'seq_len': seq_len, # sequence length
                    'device': device, # main device
                    'freqs': freqs, # rope freqs
                    't': timestep, # current timestep
                    'is_uncond': False, # is unconditional
                    'current_step': idx, # current step
                    'current_step_percentage': current_step_percentage, # current step percentage
                    'last_step': len(timesteps) - 1 == idx, # is last step
                    'control_lora_enabled': control_lora_enabled, # control lora toggle for patch embed selection
                    'enhance_enabled': enhance_enabled, # enhance-a-video toggle
                    'camera_embed': camera_embed, # recammaster embedding
                    'unianim_data': unianim_data, # unianimate input
                    'fun_ref': fun_ref_input if fun_ref_image is not None else None, # Fun model reference latent
                    'fun_camera': control_camera_input if control_camera_latents is not None else None, # Fun model camera embed
                    'audio_proj': audio_proj if fantasytalking_embeds is not None else None, # FantasyTalking audio projection
                    'audio_scale': audio_scale, # FantasyTalking audio scale
                    "uni3c_data": uni3c_data_input, # Uni3C input
                    "controlnet": controlnet, # TheDenk's controlnet input
                    "add_cond": add_cond_input, # additional conditioning input
                    "nag_params": text_embeds.get("nag_params", {}), # normalized attention guidance
                    "nag_context": text_embeds.get("nag_prompt_embeds", None), # normalized attention guidance context
                    "multitalk_audio": multitalk_audio_input, # Multi/InfiniteTalk audio input
                    "ref_target_masks": ref_target_masks if multitalk_audio_embeds is not None else None, # Multi/InfiniteTalk reference target masks
                    "inner_t": [shot_len] if shot_len else None, # inner timestep for EchoShot
                    "standin_input": standin_input, # Stand-in reference input
                    "fantasy_portrait_input": fantasy_portrait_input, # Fantasy portrait input
                    "phantom_ref": phantom_ref, # Phantom reference input
                    "reverse_time": reverse_time, # Reverse RoPE toggle
                    "ntk_alphas": ntk_alphas, # RoPE freq scaling values
                    "mtv_motion_tokens": mtv_motion_tokens if mtv_input is not None else None, # MTV-Crafter motion tokens
                    "mtv_motion_rotary_emb": mtv_motion_rotary_emb if mtv_input is not None else None, # MTV-Crafter RoPE
                    "mtv_strength": mtv_strength[idx] if mtv_input is not None else 1.0, # MTV-Crafter scaling
                    "mtv_freqs": mtv_freqs if mtv_input is not None else None, # MTV-Crafter extra RoPE freqs
                    "s2v_audio_input": s2v_audio_input, # official speech-to-video audio input
                    "s2v_ref_latent": s2v_ref_latent, # speech-to-video reference latent
                    "s2v_ref_motion": s2v_ref_motion, # speech-to-video reference motion latent
                    "s2v_audio_scale": s2v_audio_scale if s2v_audio_input is not None else 1.0, # speech-to-video audio scale
                    "s2v_pose": s2v_pose if s2v_pose is not None else None, # speech-to-video pose control
                    "s2v_motion_frames": s2v_motion_frames, # speech-to-video motion frames,
                    "humo_audio": humo_audio, # humo audio input
                    "humo_audio_scale": humo_audio_scale if humo_audio is not None else 1,
                    "wananim_pose_latents": wananim_pose_latents.to(device) if wananim_pose_latents is not None else None, # WanAnimate pose latents
                    "wananim_face_pixel_values": wananim_face_pixels.to(device, torch.float32) if wananim_face_pixels is not None else None, # WanAnimate face images
                    "wananim_pose_strength": wananim_pose_strength,
                    "wananim_face_strength": wananim_face_strength,
                    "wananim_num_anchor_latents": wananim_num_anchor_latents,
                    "lynx_embeds": lynx_embeds, # Lynx face and reference embeddings
                    "x_ovi": [latent_model_input_ovi.to(z)] if latent_model_input_ovi is not None else None, # Audio latent model input for Ovi
                    "seq_len_ovi": seq_len_ovi, # Audio latent model sequence length for Ovi
                    "ovi_negative_text_embeds": ovi_negative_text_embeds, # Audio latent model negative text embeds for Ovi
                    "flashvsr_LQ_latent": flashvsr_LQ_latent, # FlashVSR LQ latent for upsampling
                    "flashvsr_strength": flashvsr_strength, # FlashVSR strength
                    "longcat_num_cond_latents": longcat_num_cond_latents,
                    "longcat_num_ref_latents": longcat_num_ref_latents,
                    "longcat_avatar_options": longcat_avatar_options, # LongCat avatar attention options
                    "sdancer_input": sdancer_input, # SteadyDancer input
                    "one_to_all_input": one_to_all_data, # One-to-All input
                    "one_to_all_controlnet_strength": one_to_all_data["controlnet_strength"] if one_to_all_data is not None else 0.0,
                    "scail_input": scail_data_in, # SCAIL input
                    "dual_control_input": dual_control_in, # LongVie2 dual control input
                    "context_latents": context_latents, # Bernini in-context reference
                    "context_window_start": context_window_start, # Context window start frame index for temporal RoPE alignment
                    "simple_t2v": False, # enabled below for Bernini context or strict plain T2V
                    "transformer_options": transformer_options,
                    "rope_negative_offset": image_embeds.get("rope_negative_offset_frames", 0), # StoryMem rope negative offset
                    "num_memory_frames": story_mem_latents.shape[1] if story_mem_latents is not None else 0, # StoryMem memory frames
                }

                fast_path_no_extra_conditions = (
                    base_params['y'] is None
                    and base_params['clip_fea'] is None
                    and not reverse_time
                    and not enhance_enabled
                    and cache_args is None
                    and slg_args is None
                    and not batched_cfg
                    and not bidirectional_sampling
                    and control_latents is None
                    and vace_data is None
                    and unianim_data is None
                    and audio_proj is None
                    and control_camera_latents is None
                    and add_cond is None
                    and multitalk_audio_embeds is None
                    and fantasy_portrait_input is None
                    and mtv_motion_tokens is None
                    and s2v_audio_input is None
                    and s2v_ref_motion is None
                    and s2v_pose is None
                    and humo_image_cond is None
                    and humo_audio is None
                    and wananim_pose_latents is None
                    and wananim_face_pixels is None
                    and uni3c_data is None
                    and latent_model_input_ovi is None
                    and flashvsr_LQ_latent is None
                    and base_params['fun_ref'] is None
                    and base_params['fun_camera'] is None
                    and base_params['audio_proj'] is None
                    and base_params['uni3c_data'] is None
                    and base_params['controlnet'] is None
                    and base_params['add_cond'] is None
                    and not base_params['nag_params']
                    and base_params['nag_context'] is None
                    and base_params['multitalk_audio'] is None
                    and base_params['ref_target_masks'] is None
                    and base_params['inner_t'] is None
                    and base_params['standin_input'] is None
                    and base_params['fantasy_portrait_input'] is None
                    and base_params['phantom_ref'] is None
                    and base_params['mtv_motion_tokens'] is None
                    and base_params['mtv_motion_rotary_emb'] is None
                    and base_params['s2v_audio_input'] is None
                    and base_params['s2v_ref_latent'] is None
                    and base_params['s2v_ref_motion'] is None
                    and base_params['s2v_pose'] is None
                    and base_params['humo_audio'] is None
                    and base_params['wananim_pose_latents'] is None
                    and base_params['wananim_face_pixel_values'] is None
                    and base_params['lynx_embeds'] is None
                    and base_params['x_ovi'] is None
                    and base_params['flashvsr_LQ_latent'] is None
                    and base_params['longcat_num_cond_latents'] == 0
                    and base_params['longcat_num_ref_latents'] == 0
                    and base_params['longcat_avatar_options'] is None
                    and base_params['sdancer_input'] is None
                    and base_params['one_to_all_input'] is None
                    and base_params['scail_input'] is None
                    and base_params['dual_control_input'] is None
                    and base_params['rope_negative_offset'] == 0
                    and base_params['num_memory_frames'] == 0
                )
                strict_plain_t2v_fast_path = context_latents is None and context_window is None and fast_path_no_extra_conditions
                base_params["simple_t2v"] = (context_latents is not None and base_params['scail_input'] is None) or strict_plain_t2v_fast_path
                scail2_fast_path = (
                    base_params['scail_input'] is not None
                    and guidance_mode == "cfg"
                    and context_latents is None
                    and base_params['y'] is None
                    and image_cond_neg is None
                    and attn_cond is None
                    and attn_cond_neg is None
                    and not reverse_time
                    and not enhance_enabled
                    and cache_args is None
                    and freeinit_args is None
                    and not scail2_fast_path_blocked_by_experimental_sampling
                    and slg_args is None
                    and getattr(transformer, "slg_blocks", None) is None
                    and not getattr(transformer, "is_longcat", False)
                    and getattr(transformer, "audio_model", None) is None
                    and getattr(transformer, "vace_layers", None) is None
                    and not batched_cfg
                    and not bidirectional_sampling
                    and (loop_args is None or scail2_looping)
                    and qwenvl_embeds_pos is None
                    and qwenvl_embeds_neg is None
                    and control_latents is None
                    and vace_data is None
                    and unianim_data is None
                    and audio_proj is None
                    and control_camera_latents is None
                    and add_cond is None
                    and multitalk_audio_embeds is None
                    and fantasy_portrait_input is None
                    and mtv_motion_tokens is None
                    and s2v_audio_input is None
                    and s2v_ref_motion is None
                    and s2v_pose is None
                    and humo_image_cond is None
                    and humo_audio is None
                    and wananim_pose_latents is None
                    and wananim_face_pixels is None
                    and latent_model_input_ovi is None
                    and flashvsr_LQ_latent is None
                    and base_params['fun_ref'] is None
                    and base_params['fun_camera'] is None
                    and base_params['audio_proj'] is None
                    and base_params['controlnet'] is None
                    and base_params['add_cond'] is None
                    and not base_params['nag_params']
                    and base_params['nag_context'] is None
                    and base_params['multitalk_audio'] is None
                    and base_params['ref_target_masks'] is None
                    and base_params['inner_t'] is None
                    and base_params['standin_input'] is None
                    and base_params['fantasy_portrait_input'] is None
                    and base_params['phantom_ref'] is None
                    and base_params['mtv_motion_tokens'] is None
                    and base_params['mtv_motion_rotary_emb'] is None
                    and base_params['s2v_audio_input'] is None
                    and base_params['s2v_ref_latent'] is None
                    and base_params['s2v_ref_motion'] is None
                    and base_params['s2v_pose'] is None
                    and base_params['humo_audio'] is None
                    and base_params['wananim_pose_latents'] is None
                    and base_params['wananim_face_pixel_values'] is None
                    and base_params['lynx_embeds'] is None
                    and base_params['x_ovi'] is None
                    and base_params['flashvsr_LQ_latent'] is None
                    and base_params['longcat_num_cond_latents'] == 0
                    and base_params['longcat_num_ref_latents'] == 0
                    and base_params['longcat_avatar_options'] is None
                    and base_params['sdancer_input'] is None
                    and base_params['one_to_all_input'] is None
                    and base_params['dual_control_input'] is None
                    and base_params['rope_negative_offset'] == 0
                    and base_params['num_memory_frames'] == 0
                    and "comfy" in getattr(transformer, "rope_func", "")
                )
                if base_params['scail_input'] is not None and not scail2_fast_path:
                    if not scail2_fast_path_fallback_logged:
                        scail2_fast_path_fallback_logged = True
                        fallback_reasons = []
                        if guidance_mode != "cfg":
                            fallback_reasons.append(f"guidance_mode={guidance_mode!r}")
                        if context_latents is not None:
                            fallback_reasons.append("Bernini context_latents connected")
                        if base_params['y'] is not None:
                            fallback_reasons.append("image condition active")
                        if image_cond_neg is not None:
                            fallback_reasons.append("negative image condition active")
                        if attn_cond is not None or attn_cond_neg is not None:
                            fallback_reasons.append("attention condition active")
                        if reverse_time:
                            fallback_reasons.append("reverse_time enabled")
                        if enhance_enabled:
                            fallback_reasons.append("enhance enabled")
                        if cache_args is not None:
                            fallback_reasons.append("cache enabled")
                        if freeinit_args is not None:
                            fallback_reasons.append("FreeInit enabled")
                        if scail2_fast_path_blocked_by_experimental_sampling:
                            fallback_reasons.append("experimental sampling enabled")
                        if slg_args is not None or getattr(transformer, "slg_blocks", None) is not None:
                            fallback_reasons.append("SLG enabled")
                        if getattr(transformer, "is_longcat", False):
                            fallback_reasons.append("LongCat model active")
                        if getattr(transformer, "audio_model", None) is not None:
                            fallback_reasons.append("OVI audio model active")
                        if getattr(transformer, "vace_layers", None) is not None:
                            fallback_reasons.append("VACE layers active")
                        if batched_cfg:
                            fallback_reasons.append("batched CFG enabled")
                        if bidirectional_sampling:
                            fallback_reasons.append("bidirectional sampling enabled")
                        if loop_args is not None and not scail2_looping:
                            fallback_reasons.append("loop enabled")
                        if qwenvl_embeds_pos is not None or qwenvl_embeds_neg is not None:
                            fallback_reasons.append("QwenVL embeddings connected")
                        if control_latents is not None or control_camera_latents is not None:
                            fallback_reasons.append("control latents connected")
                        if vace_data is not None:
                            fallback_reasons.append("VACE input connected")
                        if unianim_data is not None:
                            fallback_reasons.append("UniAnimate input connected")
                        if audio_proj is not None:
                            fallback_reasons.append("audio projection connected")
                        if add_cond is not None:
                            fallback_reasons.append("additional condition connected")
                        if multitalk_audio_embeds is not None:
                            fallback_reasons.append("MultiTalk input connected")
                        if fantasy_portrait_input is not None:
                            fallback_reasons.append("FantasyPortrait input connected")
                        if mtv_motion_tokens is not None:
                            fallback_reasons.append("MTV motion tokens connected")
                        if s2v_audio_input is not None or s2v_ref_motion is not None or s2v_pose is not None:
                            fallback_reasons.append("S2V input connected")
                        if humo_image_cond is not None or humo_audio is not None:
                            fallback_reasons.append("HuMo input connected")
                        if wananim_pose_latents is not None or wananim_face_pixels is not None:
                            fallback_reasons.append("WanAnimate input connected")
                        if latent_model_input_ovi is not None:
                            fallback_reasons.append("OVI latent input connected")
                        if flashvsr_LQ_latent is not None:
                            fallback_reasons.append("FlashVSR input connected")
                        if base_params['fun_ref'] is not None or base_params['fun_camera'] is not None:
                            fallback_reasons.append("Fun reference/camera active")
                        if base_params['controlnet'] is not None:
                            fallback_reasons.append("ControlNet active")
                        if base_params['nag_params'] or base_params['nag_context'] is not None:
                            fallback_reasons.append("NAG active")
                        if base_params['ref_target_masks'] is not None:
                            fallback_reasons.append("reference target masks active")
                        if base_params['inner_t'] is not None:
                            fallback_reasons.append("EchoShot active")
                        if base_params['standin_input'] is not None:
                            fallback_reasons.append("Stand-In input connected")
                        if base_params['phantom_ref'] is not None:
                            fallback_reasons.append("Phantom reference connected")
                        if base_params['lynx_embeds'] is not None:
                            fallback_reasons.append("Lynx input connected")
                        if base_params['longcat_num_cond_latents'] != 0 or base_params['longcat_num_ref_latents'] != 0 or base_params['longcat_avatar_options'] is not None:
                            fallback_reasons.append("LongCat options active")
                        if base_params['sdancer_input'] is not None:
                            fallback_reasons.append("SteadyDancer input connected")
                        if base_params['one_to_all_input'] is not None:
                            fallback_reasons.append("One-to-All input connected")
                        if base_params['dual_control_input'] is not None:
                            fallback_reasons.append("Dual Control input connected")
                        if base_params['rope_negative_offset'] != 0 or base_params['num_memory_frames'] != 0:
                            fallback_reasons.append("StoryMem/memory RoPE active")
                        if "comfy" not in getattr(transformer, "rope_func", ""):
                            fallback_reasons.append(f"rope_func={getattr(transformer, 'rope_func', None)!r}")
                        if not fallback_reasons:
                            fallback_reasons.append("unsupported sampler/model option")
                        log.info("SCAIL-2 fast path disabled; using normal path. Reason(s): " + "; ".join(fallback_reasons))
                base_params["simple_scail2"] = scail2_fast_path
                if scail2_fast_path:
                    base_params["simple_t2v"] = False

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0):
                    if negative_embeds is None:
                        raise ValueError("Negative embeddings must be provided for CFG scale > 1.0")
                    if len(positive_embeds) > 1:
                        negative_embeds = negative_embeds * len(positive_embeds)

                try:
                    if not batched_cfg and guidance_mode not in ("cfg_chain", "apg_chain"):
                        #conditional (positive) pass
                        if pos_latent is not None: # for humo
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], pos_latent], dim=1)]
                        base_params["add_text_emb"] = qwenvl_embeds_pos.to(device) if qwenvl_embeds_pos is not None else None # QwenVL embeddings for Bindweave
                        noise_pred_cond, noise_pred_ovi, cache_state_cond = transformer(
                            context=positive_embeds,
                            pred_id=cache_state[0] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond,
                            **base_params
                        )
                        noise_pred_cond = noise_pred_cond[0]
                        noise_pred_ovi = noise_pred_ovi[0] if noise_pred_ovi is not None else None
                        if math.isclose(cfg_scale, 1.0):
                            if use_fresca:
                                noise_pred_cond = fourier_filter(noise_pred_cond, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                            if fantasy_portrait_input is not None and not math.isclose(portrait_cfg[idx], 1.0):
                                base_params["fantasy_portrait_input"] = None
                                noise_pred_no_portrait, noise_pred_ovi, cache_state_uncond = transformer(context=positive_embeds, pred_id=cache_state[0] if cache_state else None,
                                vace_data=vace_data, attn_cond=attn_cond, **base_params)
                                return noise_pred_no_portrait[0] + portrait_cfg[idx] * (noise_pred_cond - noise_pred_no_portrait[0]), noise_pred_ovi, [cache_state_cond, cache_state_uncond]
                            elif multitalk_audio_input is not None and not math.isclose(audio_cfg_scale[idx], 1.0):
                                base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:]
                                noise_pred_uncond_audio, _, cache_state_uncond = transformer(
                                context=positive_embeds, pred_id=cache_state[0] if cache_state else None,
                                vace_data=vace_data, attn_cond=attn_cond, **base_params)
                                return noise_pred_uncond_audio[0] + audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_uncond_audio[0]), noise_pred_ovi, [cache_state_cond, cache_state_uncond]
                            else:
                                return noise_pred_cond, noise_pred_ovi, [cache_state_cond]

                        #unconditional (negative) pass
                        base_params['is_uncond'] = True
                        base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                        base_params["add_text_emb"] = qwenvl_embeds_neg.to(device) if qwenvl_embeds_neg is not None else None # QwenVL embeddings for Bindweave
                        base_params['y'] = [image_cond_neg.to(z)] if image_cond_neg is not None else base_params['y']
                        if wananim_face_pixels is not None:
                            base_params['wananim_face_pixel_values'] = torch.zeros_like(wananim_face_pixels).to(device, torch.float32) - 1
                        if humo_audio_input_neg is not None:
                            base_params['humo_audio'] = humo_audio_input_neg
                        if neg_latent is not None:
                            base_params['x'] = [torch.cat([z[:, :-humo_reference_count], neg_latent], dim=1)]

                        noise_pred_uncond_text, noise_pred_ovi_uncond, cache_state_uncond = transformer(
                            context=negative_embeds if humo_audio_input_neg is None else positive_embeds, #ti #t
                            pred_id=cache_state[1] if cache_state else None,
                            vace_data=vace_data, attn_cond=attn_cond_neg,
                            **base_params)
                        noise_pred_uncond_text = noise_pred_uncond_text[0]
                        noise_pred_ovi_uncond = noise_pred_ovi_uncond[0] if noise_pred_ovi_uncond is not None else None

                        # HuMo
                        if not math.isclose(humo_audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            if humo_image_cond is not None and humo_audio_input_neg is not None:
                                if t > 980 and humo_image_cond_neg_input is not None: # use image cond for first timesteps
                                    base_params['y'] = [humo_image_cond_neg_input]

                                noise_pred_humo_audio_uncond, _, cache_state_humo = transformer(
                                context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                                **base_params)

                                noise_pred = (noise_pred_uncond_text + humo_audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_humo_audio_uncond[0])
                                            + (cfg_scale - 2.0) * (noise_pred_humo_audio_uncond[0] - noise_pred_uncond_text))
                                return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_humo]
                            elif humo_audio_input is not None:
                                if cache_state is not None and len(cache_state) != 4:
                                    cache_state.append(None)
                                # audio
                                noise_pred_humo_null, _, cache_state_humo = transformer(
                                context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                                **base_params)
                                # negative
                                if humo_audio_input is not None:
                                    base_params['humo_audio'] = humo_audio_input
                                noise_pred_humo_audio, _, cache_state_humo2 = transformer(
                                context=positive_embeds, pred_id=cache_state[3] if cache_state else None, vace_data=None,
                                **base_params)
                                noise_pred = (humo_audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_humo_audio[0])
                                    + cfg_scale * (noise_pred_humo_audio[0] - noise_pred_uncond_text)
                                    + cfg_scale * (noise_pred_uncond_text - noise_pred_humo_null[0])
                                    + noise_pred_humo_null[0])
                                return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_humo, cache_state_humo2]

                        #phantom
                        if use_phantom and not math.isclose(phantom_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_phantom, _, cache_state_phantom = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + phantom_cfg_scale[idx] * (noise_pred_phantom[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_phantom[0]))
                            return noise_pred, None,[cache_state_cond, cache_state_uncond, cache_state_phantom]
                        # audio cfg (fantasytalking and multitalk)
                        if (fantasytalking_embeds is not None or multitalk_audio_input is not None):
                            if not math.isclose(audio_cfg_scale[idx], 1.0):
                                if cache_state is not None and len(cache_state) != 3:
                                    cache_state.append(None)

                                base_params['audio_proj'] = None
                                base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:] if multitalk_audio_input is not None else None
                                base_params['is_uncond'] = False
                                noise_pred_uncond_audio, _, cache_state_audio = transformer(
                                    context=negative_embeds,
                                    pred_id=cache_state[2] if cache_state else None,
                                    vace_data=vace_data,
                                    **base_params)
                                noise_pred_uncond_audio = noise_pred_uncond_audio[0]

                                noise_pred = noise_pred_uncond_audio + cfg_scale * (
                                    (noise_pred_cond - noise_pred_uncond_text)
                                    + audio_cfg_scale[idx] * (noise_pred_uncond_text - noise_pred_uncond_audio))
                                return noise_pred, None,[cache_state_cond, cache_state_uncond, cache_state_audio]
                        # lynx
                        if lynx_embeds is not None and not math.isclose(lynx_cfg_scale[idx], 1.0):
                            base_params['is_uncond'] = False
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_lynx, _, cache_state_lynx = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + lynx_cfg_scale[idx] * (noise_pred_lynx[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_lynx[0]))
                            return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_lynx]
                        # one-to-all
                        if one_to_all_data is not None and not math.isclose(one_to_all_pose_cfg_scale[idx], 1.0):
                            tqdm.write("One-to-All pose CFG pass...")
                            base_params['is_uncond'] = False
                            base_params['one_to_all_controlnet_strength'] = 0.0
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            noise_pred_pose_uncond, _, cache_state_ref = transformer(
                            context=negative_embeds, pred_id=cache_state[2] if cache_state else None, vace_data=None,
                            **base_params)

                            noise_pred = (noise_pred_uncond_text + one_to_all_pose_cfg_scale[idx] * (noise_pred_pose_uncond[0] - noise_pred_uncond_text)
                                          + cfg_scale * (noise_pred_cond - noise_pred_pose_uncond[0]))
                            return noise_pred, None, [cache_state_cond, cache_state_uncond, cache_state_ref]

                    #batched
                    elif guidance_mode not in ("cfg_chain", "apg_chain"):
                        base_params['z'] = [z] * 2
                        base_params['y'] = [image_cond_input] * 2 if image_cond_input is not None else None
                        if clip_fea is not None:
                            uncond_clip_fea = clip_fea_neg if clip_fea_neg is not None else clip_fea
                            base_params['clip_fea'] = torch.cat([clip_fea, uncond_clip_fea], dim=0)
                        else:
                            base_params['clip_fea'] = None
                        cache_state_uncond = None
                        [noise_pred_cond, noise_pred_uncond_text], _, cache_state_cond = transformer(
                            context=positive_embeds + negative_embeds, is_uncond=False,
                            pred_id=cache_state[0] if cache_state else None,
                            **base_params
                        )
                except Exception as e:
                    log.error(f"Error during model prediction: {e}")
                    if force_offload:
                        if not model["auto_cpu_offload"]:
                            offload_transformer(transformer)
                    raise e

                # Post-processing (standard CFG only; chain modes use their own formulas)
                if guidance_mode not in ("cfg_chain", "apg_chain"):
                    #https://github.com/WeichenFan/CFG-Zero-star/
                    alpha = 1.0
                    if use_cfg_zero_star:
                        alpha = optimized_scale(
                            noise_pred_cond.view(batch_size, -1),
                            noise_pred_uncond_text.view(batch_size, -1)
                        ).view(batch_size, 1, 1, 1)

                    noise_pred_uncond_text = noise_pred_uncond_text * alpha

                    if use_tangential:
                        noise_pred_uncond_text = tangential_projection(noise_pred_cond, noise_pred_uncond_text)

                    # RAAG (RATIO-aware Adaptive Guidance)
                    if raag_alpha > 0.0:
                        cfg_scale = get_raag_guidance(noise_pred_cond, noise_pred_uncond_text, cfg_scale, raag_alpha)
                        log.info(f"RAAG modified cfg: {cfg_scale}")

                #https://github.com/WikiChao/FreSca
                if use_fresca and guidance_mode not in ("cfg_chain", "apg_chain"):
                    filtered_cond = fourier_filter(noise_pred_cond - noise_pred_uncond_text, fresca_scale_low, fresca_scale_high, fresca_freq_cutoff)
                    noise_pred = noise_pred_uncond_text + cfg_scale * filtered_cond * alpha
                elif guidance_mode == "apg":
                    sigma = sample_scheduler.sigmas[idx]
                    x_cond = z - sigma * noise_pred_cond
                    x_uncond = z - sigma * noise_pred_uncond_text
                    x_guided = _normalized_guidance(x_cond, x_uncond, apg_omega, _momentum_buf, apg_eta, apg_norm_threshold)
                    noise_pred = (z - x_guided) / sigma
                elif guidance_mode == "cfg_chain":
                    saved_ctx = base_params.pop("context_latents", None)
                    noise_pred_ovi_uncond = None  # not used in chain modes; prevents NameError
                    saved_clip_fea = base_params.get('clip_fea', None)
                    saved_add_text_emb = base_params.get("add_text_emb", None)
                    saved_y = base_params.get('y', None)
                    saved_x = base_params.get('x', None)
                    saved_wananim_face = base_params.get('wananim_face_pixel_values', None)
                    saved_humo_audio = base_params.get('humo_audio', None)
                    saved_ctx_list = list(saved_ctx or [])
                    if context_roles is not None and len(context_roles) == len(saved_ctx_list):
                        role_pairs = list(zip(saved_ctx_list, context_roles))
                        V_ctx = [lat for lat, role in role_pairs if role == "source_video"]
                        I_ctx = [lat for lat, role in role_pairs if role == "reference_image"]
                        VI_ctx = saved_ctx_list
                    else:
                        # Backward compatibility for old Bernini outputs without roles.
                        V_ctx = [lat for lat in saved_ctx_list if lat.shape[1] > 1]
                        I_ctx = [lat for lat in saved_ctx_list if lat.shape[1] == 1]
                        VI_ctx = saved_ctx_list
                    has_extra_context = len(VI_ctx) > len(V_ctx)

                    # Forward 1: ∅ + uncond text
                    base_params['is_uncond'] = True
                    base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                    base_params["add_text_emb"] = qwenvl_embeds_neg.to(device) if qwenvl_embeds_neg is not None else None
                    base_params['y'] = [image_cond_neg.to(z)] if image_cond_neg is not None else base_params['y']
                    if wananim_face_pixels is not None:
                        base_params['wananim_face_pixel_values'] = torch.zeros_like(wananim_face_pixels).to(device, torch.float32) - 1
                    if humo_audio_input_neg is not None:
                        base_params['humo_audio'] = humo_audio_input_neg
                    if neg_latent is not None:
                        base_params['x'] = [torch.cat([z[:, :-humo_reference_count], neg_latent], dim=1)]
                    base_params["context_latents"] = None
                    eps_uncond, _, cache_state_uncond = transformer(
                        context=negative_embeds,
                        pred_id=cache_state[1] if cache_state else None, **base_params)
                    eps_uncond = eps_uncond[0]

                    # Forward 2: V + uncond text. If V is empty, this equals eps_uncond.
                    if V_ctx:
                        base_params["context_latents"] = V_ctx
                        eps_V, _, _ = transformer(
                            context=negative_embeds, pred_id=None, **base_params)
                        eps_V = eps_V[0]
                    else:
                        eps_V = eps_uncond

                    if has_extra_context:
                        # VI combo: source video + reference video/images, uncond text
                        base_params["context_latents"] = VI_ctx if VI_ctx else None
                        eps_VI, _, _ = transformer(
                            context=negative_embeds, pred_id=None, **base_params)
                        eps_VI = eps_VI[0]

                    # Final forward: VI + cond text
                    base_params["context_latents"] = VI_ctx if VI_ctx else None
                    base_params['is_uncond'] = False
                    base_params['clip_fea'] = saved_clip_fea
                    base_params["add_text_emb"] = saved_add_text_emb
                    base_params['y'] = saved_y
                    base_params['x'] = saved_x
                    base_params['wananim_face_pixel_values'] = saved_wananim_face
                    base_params['humo_audio'] = saved_humo_audio
                    eps_VTI, npo, cache_state_cond = transformer(
                        context=positive_embeds, pred_id=None, **base_params)
                    eps_VTI = eps_VTI[0]
                    noise_pred_ovi = npo[0] if npo is not None else None

                    base_params["context_latents"] = saved_ctx
                    noise_pred_cond = eps_VTI
                    noise_pred_uncond_text = eps_uncond

                    if has_extra_context:
                        noise_pred = (eps_uncond
                                      + chain_omega_V * (eps_V - eps_uncond)
                                      + chain_omega_I * (eps_VI - eps_V)
                                      + chain_omega_TI * (eps_VTI - eps_VI))
                    else:
                        noise_pred = (eps_uncond
                                      + chain_omega_V * (eps_V - eps_uncond)
                                      + chain_omega_TI * (eps_VTI - eps_V))
                elif guidance_mode == "apg_chain":
                    sigma = sample_scheduler.sigmas[idx]

                    # ∅ combo: no context, uncond text
                    saved_ctx = base_params.pop("context_latents", None)
                    noise_pred_ovi_uncond = None  # not used in chain modes; prevents NameError
                    saved_clip_fea = base_params.get('clip_fea', None)
                    saved_add_text_emb = base_params.get("add_text_emb", None)
                    saved_y = base_params.get('y', None)
                    saved_x = base_params.get('x', None)
                    saved_wananim_face = base_params.get('wananim_face_pixel_values', None)
                    saved_humo_audio = base_params.get('humo_audio', None)
                    saved_ctx_list = list(saved_ctx or [])
                    if context_roles is not None and len(context_roles) == len(saved_ctx_list):
                        I_ctx = [lat for lat, role in zip(saved_ctx_list, context_roles) if role == "reference_image"]
                    else:
                        # Backward compatibility for old Bernini outputs without roles.
                        I_ctx = [lat for lat in saved_ctx_list if lat.shape[1] == 1]
                        if not I_ctx:
                            I_ctx = saved_ctx_list
                    base_params['is_uncond'] = True
                    base_params['clip_fea'] = clip_fea_neg if clip_fea_neg is not None else clip_fea
                    base_params["add_text_emb"] = qwenvl_embeds_neg.to(device) if qwenvl_embeds_neg is not None else None
                    base_params['y'] = [image_cond_neg.to(z)] if image_cond_neg is not None else base_params['y']
                    if wananim_face_pixels is not None:
                        base_params['wananim_face_pixel_values'] = torch.zeros_like(wananim_face_pixels).to(device, torch.float32) - 1
                    if humo_audio_input_neg is not None:
                        base_params['humo_audio'] = humo_audio_input_neg
                    if neg_latent is not None:
                        base_params['x'] = [torch.cat([z[:, :-humo_reference_count], neg_latent], dim=1)]
                    base_params["context_latents"] = None
                    noise_pred_0, _, cache_state_uncond = transformer(
                        context=negative_embeds,
                        pred_id=cache_state[1] if cache_state else None,
                        **base_params)
                    noise_pred_0 = noise_pred_0[0]

                    # I combo: reference images only, uncond text
                    base_params["context_latents"] = I_ctx if I_ctx else None
                    noise_pred_I, _, _ = transformer(
                        context=negative_embeds, pred_id=None, **base_params)
                    noise_pred_I = noise_pred_I[0]

                    # I+T combo: reference images only, cond text
                    base_params["context_latents"] = I_ctx if I_ctx else None
                    base_params['is_uncond'] = False
                    base_params['clip_fea'] = saved_clip_fea
                    base_params["add_text_emb"] = saved_add_text_emb
                    base_params['y'] = saved_y
                    base_params['x'] = saved_x
                    base_params['wananim_face_pixel_values'] = saved_wananim_face
                    base_params['humo_audio'] = saved_humo_audio
                    noise_pred_TI, npo, cache_state_cond = transformer(
                        context=positive_embeds, pred_id=None, **base_params)
                    noise_pred_TI = noise_pred_TI[0]
                    noise_pred_ovi = npo[0] if npo is not None else None

                    # Restore context_latents
                    base_params["context_latents"] = saved_ctx
                    noise_pred_cond = noise_pred_TI
                    noise_pred_uncond_text = noise_pred_0

                    # x-pred space → chain APG → v-pred space
                    x_0  = z - sigma * noise_pred_0
                    x_I  = z - sigma * noise_pred_I
                    x_TI = z - sigma * noise_pred_TI
                    x_guided = _normalized_guidance_chain(
                        x_0, [x_I, x_TI], [apg_omega_I, apg_omega_TI],
                        [_momentum_buf_I, _momentum_buf_TI],
                        apg_eta, apg_norm_threshold)
                    noise_pred = (z - x_guided) / sigma
                else:
                    noise_pred = noise_pred_uncond_text + cfg_scale * (noise_pred_cond - noise_pred_uncond_text)
                del noise_pred_uncond_text, noise_pred_cond

                if latent_model_input_ovi is not None and guidance_mode not in ("cfg_chain", "apg_chain"):
                    if ovi_audio_cfg is None:
                        audio_cfg_scale = cfg_scale - 1.0 if cfg_scale > 4.0 else cfg_scale
                    else:
                        audio_cfg_scale = ovi_audio_cfg[idx]
                    noise_pred_ovi = noise_pred_ovi_uncond + audio_cfg_scale * (noise_pred_ovi - noise_pred_ovi_uncond)

                return noise_pred, noise_pred_ovi, [cache_state_cond, cache_state_uncond]

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from .latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, len(timesteps))

        if not multitalk_sampling and not framepack and not wananimate_loop and not everanimate_sampling:
            log.info("-" * 10 + " Sampling start " + "-" * 10)
            log.info(f"{(latent_video_length-1) * 4 + 1} frames at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} (Input sequence length: {seq_len}) with {steps-ttm_start_step} steps")


        # Differential diffusion prep
        masks = None
        scail_context_freeze_direct_mask = (
            not multitalk_sampling
            and scail_freeze_mask is not None
            and context_options is not None
            and not scail2_looping
        )
        if not multitalk_sampling and scail_freeze_mask is not None and not scail_context_freeze_direct_mask:
            masks = scail_freeze_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device) > 0.5
        elif not multitalk_sampling and samples is not None and noise_mask is not None:
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
            thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
            noise_mask = noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device=device, dtype=thresholds.dtype)
            masks = (1.0 - noise_mask) > thresholds

        latent_shift_loop = False
        if loop_args is not None and not scail2_looping:
            latent_shift_loop = is_looped = True
            latent_skip = loop_args["shift_skip"]
            latent_shift_start_percent = loop_args["start_percent"]
            latent_shift_end_percent = loop_args["end_percent"]
            shift_idx = 0

        #clear memory before sampling
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

        if everanimate_sampling:
            if freeinit_args is not None:
                raise ValueError("FreeInit is not supported with EverAnimate sampling")
            if samples is not None:
                raise ValueError("Input latent samples are not supported with EverAnimate sampling yet")
            if vae is None:
                raise ValueError("EverAnimate sampling requires a VAE in image_embeds")

            segment_frames = int(image_embeds.get("frame_window_size", 77))
            overlap_frames = int(image_embeds.get("num_overlap_frame", 0))
            num_anchor_latents = int(image_embeds.get("num_video_anchor_latents", 4))
            num_motion_latents_ea = int(image_embeds.get("num_motion_latents", 1))
            total_output_frames = int(num_frames)
            if (segment_frames - 1) % 4 != 0:
                raise ValueError("frame_window_size must be 1 mod 4 for EverAnimate, e.g. 77, 81, 85")
            if overlap_frames >= segment_frames:
                raise ValueError("num_overlap_frame must be smaller than frame_window_size")
            if num_anchor_latents <= 0:
                raise ValueError("num_video_anchor_latents must be positive")

            stride_frames = segment_frames - overlap_frames
            num_segments = int(image_embeds.get(
                "everanimate_num_segments",
                1 if total_output_frames <= segment_frames else math.ceil((total_output_frames - segment_frames) / stride_frames) + 1,
            ))
            segment_latent_frames = int(image_embeds.get(
                "everanimate_segment_latent_frames",
                ((segment_frames + 4 * num_anchor_latents - 1) // 4) + 1,
            ))
            target_latent_frames = segment_latent_frames - num_anchor_latents
            if target_latent_frames <= 0:
                raise ValueError("EverAnimate segment latent length must be larger than the anchor latent count")

            # EverAnimate does not use CLIP Vision conditioning; "segment" is temporal-window terminology only.
            clip_fea = None
            clip_fea_neg = None

            outer_freqs = freqs
            outer_rope_num_frames = getattr(transformer.rope_embedder, "num_frames", None)
            outer_cached_freqs = getattr(transformer, "cached_freqs", None)
            outer_cached_key = getattr(transformer, "cached_key", None)

            def _ea_restore_rope_state():
                nonlocal freqs
                freqs = outer_freqs
                transformer.rope_embedder.num_frames = outer_rope_num_frames
                transformer.cached_freqs = outer_cached_freqs
                if hasattr(transformer, "cached_key"):
                    transformer.cached_key = outer_cached_key

            if context_latents is None and ("default" in rope_function or bidirectional_sampling):
                freqs = torch.cat([
                    rope_params(1024, d - 4 * (d // 6), L_test=segment_latent_frames, k=riflex_freq_index),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ], dim=1)
            elif "comfy" in rope_function:
                transformer.rope_embedder.num_frames = segment_latent_frames
                transformer.cached_freqs = None
                if hasattr(transformer, "cached_key"):
                    transformer.cached_key = None

            try:
                lat_h = int(image_embeds.get("lat_h", noise.shape[2]))
                lat_w = int(image_embeds.get("lat_w", noise.shape[3]))
                pose_pixels = image_embeds.get("pose_images", None)
                face_pixels = wananim_face_pixels
                bg_pixels = image_embeds.get("bg_images", None)
                mask_pixels = image_embeds.get("mask", None)
                if pose_pixels is None or face_pixels is None:
                    raise ValueError("EverAnimate sampling requires pose_images and face_images")

                anchor_latents = image_embeds.get("anchor_latents", None)
                if anchor_latents is None:
                    raise ValueError("EverAnimate sampling requires anchor_latents")
                manual_anchor_latents = image_embeds.get("manual_anchor_latents", anchor_latents)
                user_first_anchor_latent = image_embeds.get("user_first_anchor_latent", manual_anchor_latents[:, :1])
                use_image_anchor = bool(image_embeds.get("use_image_anchor", True))
                use_random_frame_anchor = bool(image_embeds.get("use_random_frame_anchor", True))
                random_anchor_with_user_first = bool(image_embeds.get("random_anchor_with_user_first", True))
                use_repeat_anchor = bool(image_embeds.get("use_repeat_anchor", False))

                def _ea_slice(tensor, start, length, dim):
                    if tensor is None:
                        return None
                    if tensor.shape[dim] <= 0:
                        raise ValueError("EverAnimate input sequence contains no frames")
                    take = max(min(length, tensor.shape[dim] - start), 0)
                    if take > 0:
                        out = tensor.narrow(dim, start, take)
                    else:
                        out = tensor.narrow(dim, tensor.shape[dim] - 1, 1).narrow(dim, 0, 0)
                    if out.shape[dim] < length:
                        pad_shape = [1] * tensor.ndim
                        pad_shape[dim] = length - out.shape[dim]
                        last = tensor.narrow(dim, tensor.shape[dim] - 1, 1).repeat(*pad_shape)
                        out = torch.cat([out, last], dim=dim)
                    return out

                def _ea_fit_latents(lat, count, repeat_sequence=False):
                    if count <= 0:
                        return lat[:, :0]
                    if lat.shape[1] > count:
                        return lat[:, :count]
                    if lat.shape[1] < count:
                        if repeat_sequence:
                            repeats = math.ceil(count / lat.shape[1])
                            lat = lat.repeat(1, repeats, 1, 1)[:, :count]
                        else:
                            lat = torch.cat([lat, lat[:, -1:].repeat(1, count - lat.shape[1], 1, 1)], dim=1)
                    return lat

                def _ea_encode_video_latents(video_cthw):
                    vae.to(device)
                    return vae.encode([video_cthw.to(device, vae.dtype)], device, tiled=tiled_vae, pbar=False)[0].to(device, dtype)

                def _ea_encode_pose_latents(pose_cthw):
                    vae.to(device)
                    return vae.encode([pose_cthw.to(device, vae.dtype)], device, tiled=tiled_vae, pbar=False).to(device, dtype)

                def _ea_mask_latents(mask_thw):
                    mask_thw = (1.0 - mask_thw.to(device, vae.dtype)).clamp(0.0, 1.0)
                    mask_thw = torch.nn.functional.interpolate(
                        mask_thw.unsqueeze(0).unsqueeze(0),
                        size=(mask_thw.shape[0], lat_h, lat_w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                    expanded = torch.cat([mask_thw[:1].repeat(4, 1, 1), mask_thw[1:]], dim=0)
                    latent_count = expanded.shape[0] // 4
                    expanded = expanded[:latent_count * 4]
                    mask_lat = expanded.view(latent_count, 4, lat_h, lat_w).permute(1, 0, 2, 3).contiguous()
                    if mask_lat.shape[1] < target_latent_frames:
                        mask_lat = torch.cat([mask_lat, mask_lat[:, -1:].repeat(1, target_latent_frames - mask_lat.shape[1], 1, 1)], dim=1)
                    return mask_lat[:, :target_latent_frames].to(device, dtype)

                def _ea_make_condition(current_anchor_latents, motion_latents, bg_latents=None, mask_latents=None):
                    cond_mask = torch.zeros(4, segment_latent_frames, lat_h, lat_w, device=device, dtype=dtype)
                    cond_latents = torch.zeros(16, segment_latent_frames, lat_h, lat_w, device=device, dtype=dtype)
                    current_anchor_latents = _ea_fit_latents(current_anchor_latents.to(device, dtype), num_anchor_latents)
                    cond_mask[:, :num_anchor_latents] = 1
                    cond_latents[:, :num_anchor_latents] = current_anchor_latents

                    target_start = num_anchor_latents
                    if bg_latents is not None:
                        bg_latents = _ea_fit_latents(bg_latents.to(device, dtype), target_latent_frames)
                        cond_latents[:, target_start:target_start + bg_latents.shape[1]] = bg_latents
                        if mask_latents is not None:
                            cond_mask[:, target_start:target_start + mask_latents.shape[1]] = mask_latents[:, :bg_latents.shape[1]]

                    if motion_latents is not None and motion_latents.shape[1] > 0:
                        motion_latents = _ea_fit_latents(motion_latents.to(device, dtype), min(num_motion_latents_ea, target_latent_frames))
                        cond_latents[:, target_start:target_start + motion_latents.shape[1]] = motion_latents
                    return torch.cat([cond_mask, cond_latents], dim=0)

                def _ea_pick_random_frame_indices(frame_count, count):
                    if count <= 0:
                        return []
                    if frame_count <= 1:
                        return [0] * count
                    if random_anchor_with_user_first:
                        candidates = torch.arange(1, frame_count, dtype=torch.long)
                        sample_count = min(count, candidates.numel())
                        if sample_count > 0:
                            perm = torch.randperm(candidates.numel(), generator=seed_g)[:sample_count]
                            indices = sorted(int(candidates[i]) for i in perm)
                        else:
                            indices = []
                    else:
                        candidates = torch.arange(1, frame_count, dtype=torch.long)
                        sample_count = min(max(count - 1, 0), candidates.numel())
                        if sample_count > 0:
                            perm = torch.randperm(candidates.numel(), generator=seed_g)[:sample_count]
                            indices = [0] + sorted(int(candidates[i]) for i in perm)
                        else:
                            indices = [0]
                    while len(indices) < count:
                        indices.append(indices[-1] if indices else 0)
                    return indices[:count]

                def _ea_pose_difference_score(pose_01, masks, i, j):
                    union = masks[i] | masks[j]
                    if not bool(union.any()):
                        return 0.0
                    xor_ratio = float((masks[i] ^ masks[j]).sum().item()) / float(union.sum().item())
                    color_diff = (pose_01[i] - pose_01[j]).abs().mean(dim=0)
                    fg_color_diff = float(color_diff[union].mean().item())
                    return xor_ratio + 0.25 * fg_color_diff

                def _ea_select_pose_triangle_indices(pose_cthw):
                    frame_count = int(pose_cthw.shape[1])
                    if frame_count <= 0:
                        return []
                    if frame_count == 1:
                        return [0, 0, 0]
                    if frame_count == 2:
                        return [0, 1, 1]

                    pose = pose_cthw[:, :frame_count].detach().to(torch.float32).clamp(-1, 1).cpu().movedim(0, 1)
                    max_side = max(int(pose.shape[-2]), int(pose.shape[-1]))
                    if max_side > 128:
                        scale = 128.0 / max_side
                        new_h = max(1, int(round(pose.shape[-2] * scale)))
                        new_w = max(1, int(round(pose.shape[-1] * scale)))
                        pose = torch.nn.functional.interpolate(pose, size=(new_h, new_w), mode="area")
                    pose_01 = (pose + 1.0) * 0.5
                    masks = torch.any(pose_01 > (20.0 / 255.0), dim=1)

                    best_pair = None
                    best_score = -1.0
                    diff_cache = {}

                    def diff(i, j):
                        key = (min(i, j), max(i, j))
                        if key not in diff_cache:
                            diff_cache[key] = _ea_pose_difference_score(pose_01, masks, key[0], key[1])
                        return diff_cache[key]

                    for i in range(1, frame_count):
                        diff_0_i = diff(0, i)
                        for j in range(i + 1, frame_count):
                            pairwise_sum = diff_0_i + diff(0, j) + diff(i, j)
                            if pairwise_sum > best_score:
                                best_score = pairwise_sum
                                best_pair = (i, j)

                    if best_pair is None:
                        fallback = min(2, frame_count - 1)
                        return [0, 1, fallback]
                    return [0, best_pair[0], best_pair[1]]

                def _ea_preprocess_decoded_anchor_frame(frame):
                    # Match the reference path: VAE decode -> RGB uint8/PIL -> preprocess_image [-1, 1].
                    frame = frame.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)
                    frame_u8 = ((frame + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
                    return frame_u8.to(device=device, dtype=torch.float32).div(127.5).sub(1.0).to(vae.dtype)

                def _ea_encode_anchor_frames(video_cthw, frame_indices):
                    encoded = []
                    vae.to(device)
                    for frame_idx in frame_indices:
                        frame = _ea_preprocess_decoded_anchor_frame(video_cthw[:, frame_idx:frame_idx + 1])
                        encoded.append(vae.encode([frame], device, tiled=tiled_vae, pbar=False)[0][:, :1].to(device, dtype))
                    return encoded

                def _ea_build_generated_anchor(video_cthw, pose_cthw):
                    if use_random_frame_anchor:
                        random_slots = num_anchor_latents - 1 if random_anchor_with_user_first else num_anchor_latents
                        frame_indices = _ea_pick_random_frame_indices(video_cthw.shape[1], random_slots)
                        encoded = _ea_encode_anchor_frames(video_cthw, frame_indices)
                        if random_anchor_with_user_first:
                            encoded.append(user_first_anchor_latent.to(device, dtype))
                        if not encoded:
                            encoded.append(user_first_anchor_latent.to(device, dtype))
                        return _ea_fit_latents(torch.cat(encoded, dim=1), num_anchor_latents, repeat_sequence=use_repeat_anchor)

                    num_user_slots = min(2, num_anchor_latents)
                    num_sample_slots = max(num_anchor_latents - num_user_slots, 0)
                    selected_triplet = _ea_select_pose_triangle_indices(pose_cthw)
                    sample_indices = selected_triplet[1:1 + num_sample_slots]
                    encoded = _ea_encode_anchor_frames(video_cthw, sample_indices)
                    if num_sample_slots > 0 and encoded:
                        sampled = _ea_fit_latents(torch.cat(encoded, dim=1), num_sample_slots)
                        anchor_latents_out = [sampled]
                    else:
                        anchor_latents_out = []
                    if num_user_slots > 0:
                        anchor_latents_out.append(user_first_anchor_latent.to(device, dtype).repeat(1, num_user_slots, 1, 1))
                    if not anchor_latents_out:
                        anchor_latents_out.append(user_first_anchor_latent.to(device, dtype))
                    return _ea_fit_latents(torch.cat(anchor_latents_out, dim=1), num_anchor_latents, repeat_sequence=use_repeat_anchor)

                log.info(
                    f"EverAnimate sampling: {total_output_frames} requested frames, {num_segments} segments, "
                    f"{segment_frames} frames/segment, {overlap_frames} overlap frames, {num_anchor_latents} anchor latents"
                )

                callback = prepare_callback(patcher, num_segments * len(timesteps))
                generated_segments = []
                video_anchor_latent = None
                prev_segment_latents = None
                step_iteration_count = 0

                for segment_idx in range(num_segments):
                    segment_start = segment_idx * stride_frames
                    current_anchor = anchor_latents if segment_idx == 0 or video_anchor_latent is None else video_anchor_latent
                    if segment_idx == 0 or prev_segment_latents is None or num_motion_latents_ea <= 0:
                        motion_latents = torch.zeros(16, max(num_motion_latents_ea, 0), lat_h, lat_w, device=device, dtype=dtype)
                    else:
                        motion_latents = _ea_fit_latents(prev_segment_latents[:, -num_motion_latents_ea:], num_motion_latents_ea)

                    pose_window = _ea_slice(pose_pixels, segment_start, segment_frames, 1)
                    face_window = _ea_slice(face_pixels, segment_start, segment_frames, 2).to(device, torch.float32)
                    pose_latents = _ea_encode_pose_latents(pose_window)

                    bg_latents = mask_latents = None
                    if bg_pixels is not None and mask_pixels is not None:
                        bg_window = _ea_slice(bg_pixels, segment_start, segment_frames, 1)
                        bg_latents = _ea_fit_latents(_ea_encode_video_latents(bg_window), target_latent_frames)
                        mask_window = _ea_slice(mask_pixels, segment_start, segment_frames, 0)
                        mask_latents = _ea_mask_latents(mask_window)

                    image_cond_in = _ea_make_condition(current_anchor, motion_latents, bg_latents, mask_latents)
                    latent = torch.randn(
                        16,
                        segment_latent_frames,
                        lat_h,
                        lat_w,
                        dtype=torch.float32,
                        generator=seed_g,
                        device=torch.device("cpu"),
                    ).to(device)
                    seq_len = math.ceil((latent.shape[2] * latent.shape[3]) / 4 * latent.shape[1])

                    if isinstance(scheduler, dict):
                        sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                        segment_timesteps = scheduler["timesteps"]
                    else:
                        sample_scheduler, segment_timesteps, _, _ = get_scheduler(
                            scheduler, total_steps, start_step, end_step, shift, device,
                            transformer.dim, denoise_strength, sigmas=sigmas,
                        )
                    if hasattr(sample_scheduler, "timesteps"):
                        sample_scheduler.timesteps = segment_timesteps

                    self.cache_state = [None, None]
                    sampling_pbar = tqdm(total=len(segment_timesteps), desc=f"EverAnimate segment {segment_idx + 1}/{num_segments}", position=0, leave=True)
                    for i, t in enumerate(segment_timesteps):
                        timestep = torch.tensor([t]).to(device)
                        latent_model_input = latent.to(device)
                        noise_pred, _, self.cache_state = predict_with_cfg(
                            latent_model_input,
                            cfg[min(i, len(cfg) - 1)],
                            text_embeds["prompt_embeds"],
                            text_embeds["negative_prompt_embeds"],
                            timestep,
                            i,
                            image_cond_in,
                            clip_fea,
                            cache_state=self.cache_state,
                            wananim_face_pixels=face_window,
                            wananim_pose_latents=pose_latents,
                            wananim_num_anchor_latents=num_anchor_latents,
                        )

                        if use_tsr:
                            noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)
                        latent = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            timestep,
                            latent.unsqueeze(0).to(noise_pred.device),
                            **scheduler_step_args,
                        )[0].squeeze(0).detach()

                        if callback is not None:
                            callback_latent = (latent_model_input - noise_pred.to(device) * timestep.to(device) / 1000).detach()
                            callback(step_iteration_count, callback_latent.permute(1, 0, 2, 3), None, num_segments * len(segment_timesteps))
                            del callback_latent
                        sampling_pbar.update(1)
                        step_iteration_count += 1
                        del noise_pred, latent_model_input, timestep
                    sampling_pbar.close()

                    segment_latents = latent[:, num_anchor_latents:].detach()
                    prev_segment_latents = segment_latents
                    vae.to(device)
                    segment_video = vae.decode(
                        segment_latents.unsqueeze(0).to(device, vae.dtype),
                        device=device,
                        tiled=tiled_vae,
                        pbar=False,
                    )[0].detach().cpu()
                    if segment_video.shape[1] > segment_frames:
                        segment_video = segment_video[:, :segment_frames]

                    if segment_idx == 0:
                        generated_segments.append(segment_video)
                        if use_image_anchor:
                            video_anchor_latent = _ea_build_generated_anchor(segment_video, pose_window)
                        else:
                            video_anchor_latent = segment_latents.to(device, dtype)
                    else:
                        generated_segments.append(segment_video[:, overlap_frames:])

                    del latent, segment_video, segment_latents, pose_latents, face_window, image_cond_in
                    mm.soft_empty_cache()
                    gc.collect()

                gen_video_samples = torch.cat(generated_segments, dim=1)
                if gen_video_samples.shape[1] > total_output_frames:
                    gen_video_samples = gen_video_samples[:, :total_output_frames]
                if force_offload:
                    vae.to(offload_device)
                    if not model["auto_cpu_offload"]:
                        offload_transformer(transformer)
                try:
                    print_memory(device)
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
                return {"video": gen_video_samples.permute(1, 2, 3, 0)},
            finally:
                _ea_restore_rope_state()

        if scail2_looping:
            if freeinit_args is not None:
                raise ValueError("FreeInit is not supported with SCAIL-2 loop mode")
            if vae is None:
                raise ValueError("SCAIL-2 loop mode requires a VAE in image_embeds")

            requested_output_frames = int(scail2_requested_frames)
            total_generation_frames = int(num_frames)
            chunk_frames = int(image_embeds.get("scail2_frame_window_size", frame_window_size))
            chunk_frames = ((chunk_frames - 1) // 4) * 4 + 1
            prev_frame_count = int(scail2_previous_frame_count)
            if prev_frame_count <= 0:
                raise ValueError("SCAIL-2 previous frame count must be positive")
            if chunk_frames <= prev_frame_count:
                raise ValueError("SCAIL-2 frame_window_size must be larger than the 5-frame handoff")
            stride_frames = chunk_frames - prev_frame_count
            chunk_latent_frames = (chunk_frames - 1) // 4 + 1
            prev_latent_count = (prev_frame_count - 1) // 4 + 1
            num_chunks = 1 if total_generation_frames <= chunk_frames else math.ceil((total_generation_frames - chunk_frames) / stride_frames) + 1

            lat_h = int(image_embeds.get("lat_h", noise.shape[2]))
            lat_w = int(image_embeds.get("lat_w", noise.shape[3]))
            pose_pixels_all = image_embeds.get("scail2_pose_pixels", None)
            pose_mask_pixels_all = image_embeds.get("scail2_pose_mask_pixels", None)
            if pose_pixels_all is not None:
                pose_pixels_all = pose_pixels_all.to(offload_device)
            if pose_mask_pixels_all is not None:
                pose_mask_pixels_all = pose_mask_pixels_all.to(offload_device)

            freeze_latents_global = image_embeds.get("scail_freeze_latents", None)
            freeze_mask_global = image_embeds.get("scail_freeze_mask", None)
            scail_condition_zero_mask_global = image_embeds.get("scail_condition_zero_mask", None)
            scail_sam_keep_mask_global = image_embeds.get("scail_sam_keep_mask", None)
            scail_transition_keep_mask_global = image_embeds.get("scail_transition_keep_mask", None)
            scail2_transition_colormatch = image_embeds.get("scail2_transition_colormatch", "disabled")
            scail2_transition_match_ref = image_embeds.get("scail2_transition_match_ref", None)
            scail2_transition_raw_last_frame = image_embeds.pop("scail2_transition_raw_last_frame", None)
            scail2_transition_raw_tail_means = image_embeds.pop("scail2_transition_raw_tail_means", None)
            scail2_loop_colormatch_reference = image_embeds.get("scail2_loop_colormatch_reference", "previous_matched_frame")
            if scail2_loop_colormatch_reference not in ("previous_matched_frame", "main_ref_image"):
                log.warning(f"Unknown SCAIL-2 loop_colormatch_reference={scail2_loop_colormatch_reference!r}; using previous_matched_frame")
                scail2_loop_colormatch_reference = "previous_matched_frame"
            scail2_has_transition_video = bool(image_embeds.get("scail2_has_transition_video", False))

            outer_freqs = freqs
            outer_rope_num_frames = getattr(transformer.rope_embedder, "num_frames", None)
            outer_cached_freqs = getattr(transformer, "cached_freqs", None)
            outer_cached_key = getattr(transformer, "cached_key", None)

            def _scail2_restore_rope_state():
                nonlocal freqs
                freqs = outer_freqs
                transformer.rope_embedder.num_frames = outer_rope_num_frames
                transformer.cached_freqs = outer_cached_freqs
                if hasattr(transformer, "cached_key"):
                    transformer.cached_key = outer_cached_key

            def _slice_with_last_pad(tensor, start, length, dim):
                if tensor is None:
                    return None
                if tensor.shape[dim] <= 0:
                    raise ValueError("SCAIL-2 loop input sequence contains no frames")
                take = max(min(length, tensor.shape[dim] - start), 0)
                if take > 0:
                    out = tensor.narrow(dim, start, take)
                else:
                    out = tensor.narrow(dim, tensor.shape[dim] - 1, 1).narrow(dim, 0, 0)
                if out.shape[dim] < length:
                    pad_shape = [1] * tensor.ndim
                    pad_shape[dim] = length - out.shape[dim]
                    last = tensor.narrow(dim, tensor.shape[dim] - 1, 1).repeat(*pad_shape)
                    out = torch.cat([out, last], dim=dim)
                return out

            def _fit_latent_time(tensor, length):
                if tensor is None:
                    return None
                if tensor.shape[1] > length:
                    return tensor[:, :length]
                if tensor.shape[1] < length:
                    pad = tensor[:, -1:].repeat(1, length - tensor.shape[1], 1, 1)
                    return torch.cat([tensor, pad], dim=1)
                return tensor

            def _resize_latent_spatial(tensor, h, w):
                if tensor is None or tensor.shape[2:] == (h, w):
                    return tensor
                return torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(tensor.shape[1], h, w),
                    mode="trilinear",
                    align_corners=False,
                )[0]

            def _maybe_offload_loop_vae():
                if force_offload:
                    vae.to(offload_device)

            def _encode_video_window(video_bhwc, width, height, scale=1.0):
                video_bhwc = video_bhwc[:, :, :, :3]
                pixels = common_upscale(video_bhwc.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
                pixels = pixels.permute(3, 0, 1, 2).to(device=device, dtype=vae.dtype) * 2 - 1
                vae.to(device)
                lat = vae.encode([pixels], device, tiled=tiled_vae, pbar=False)[0].to(device, dtype)
                del pixels
                _maybe_offload_loop_vae()
                mask = torch.ones_like(lat[:4])
                if scale != 1.0:
                    lat = lat * scale
                return torch.cat([lat, mask], dim=0)

            def _mask_pixels_to_latents(mask_bhwc):
                T, Hm, Wm, _ = mask_bhwc.shape
                on_thresh = 225.0 / 255.0
                mask = mask_bhwc[:, :, :, :3].movedim(-1, 1).float()
                r = (mask[:, 0:1] > on_thresh).float()
                g = (mask[:, 1:2] > on_thresh).float()
                b = (mask[:, 2:3] > on_thresh).float()
                nr, ng, nb = 1 - r, 1 - g, 1 - b
                binary_7ch = torch.cat([
                    r * g * b,
                    r * ng * nb,
                    nr * g * nb,
                    nr * ng * b,
                    r * g * nb,
                    r * ng * b,
                    nr * g * b,
                ], dim=1)
                h_lat, w_lat = Hm, Wm
                for _ in range(3):
                    h_lat = (h_lat + 1) // 2
                    w_lat = (w_lat + 1) // 2
                binary_7ch = torch.nn.functional.interpolate(binary_7ch, size=(h_lat, w_lat), mode="area")
                t_lat = (T - 1) // 4 + 1
                padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
                if padded.shape[0] < t_lat * 4:
                    padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0], 1, 1, 1)], dim=0)
                padded = padded[:t_lat * 4]
                return padded.view(t_lat, 28, h_lat, w_lat).movedim(0, 1).contiguous().to(device=device, dtype=dtype)

            def _frame_mask_to_latent_mask(frame_mask, latent_count):
                frame_mask = frame_mask.flatten().to(device=device, dtype=dtype)
                t_lat = (frame_mask.shape[0] - 1) // 4 + 1
                padded = torch.cat([frame_mask[:1].repeat(4), frame_mask[1:]], dim=0)
                if padded.shape[0] < t_lat * 4:
                    padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0])], dim=0)
                latent_mask = padded[:t_lat * 4].view(t_lat, 4).amax(dim=1)
                if latent_mask.shape[0] < latent_count:
                    latent_mask = torch.cat([latent_mask, torch.zeros(latent_count - latent_mask.shape[0], device=device, dtype=dtype)], dim=0)
                return latent_mask[:latent_count]

            def _slice_mask_window(mask, chunk_start, keep_mask=None):
                if mask is None:
                    return None
                sliced = _slice_with_last_pad(mask, chunk_start // 4, chunk_latent_frames, 0).to(device=device, dtype=torch.bool)
                if keep_mask is not None:
                    keep = _slice_with_last_pad(keep_mask, chunk_start // 4, chunk_latent_frames, 0).to(device=device, dtype=torch.bool)
                    sliced &= ~keep
                return sliced

            def _zero_condition_latents(latents, zero_mask):
                if latents is None or zero_mask is None:
                    return latents
                copy_len = min(latents.shape[1], zero_mask.shape[0])
                if copy_len > 0 and bool(zero_mask[:copy_len].any()):
                    latents = latents.clone()
                    zero_idx = zero_mask[:copy_len].nonzero(as_tuple=True)[0]
                    latents[:, zero_idx] = 0
                return latents

            def _color_match_ref_to_numpy(ref_frame):
                if ref_frame is None:
                    return None
                if not isinstance(ref_frame, torch.Tensor):
                    return None
                ref = ref_frame.detach().cpu().float()
                if ref.numel() == 0:
                    return None
                if ref.ndim == 4:
                    if ref.shape[-1] in (3, 4):
                        ref = ref[0, :, :, :3]
                    elif ref.shape[1] in (3, 4):
                        ref = ref[0, :3].permute(1, 2, 0)
                    else:
                        return None
                elif ref.ndim == 3:
                    if ref.shape[-1] in (3, 4):
                        ref = ref[:, :, :3]
                    elif ref.shape[0] in (3, 4):
                        ref = ref[:3].permute(1, 2, 0)
                    else:
                        return None
                else:
                    return None
                if ref.min().item() < 0.0:
                    ref = ref.clamp(-1.0, 1.0).add(1.0).div(2.0)
                else:
                    ref = ref.clamp(0.0, 1.0)
                return ref.contiguous().numpy()

            def _color_match_video_frames(video_cthw, ref_frame):
                if video_cthw is None or video_cthw.shape[1] == 0:
                    return video_cthw
                ref_np = _color_match_ref_to_numpy(ref_frame)
                if ref_np is None:
                    return video_cthw
                from color_matcher import ColorMatcher
                cm = ColorMatcher()
                video_bhwc = video_cthw.float().clamp(-1.0, 1.0).permute(1, 2, 3, 0).add(1.0).div(2.0)
                matched = []
                warned_error = False
                warned_shape = False
                warned_nonfinite = False
                for frame in video_bhwc:
                    frame_np = frame.detach().cpu().contiguous().numpy()
                    try:
                        out = np.asarray(cm.transfer(src=frame_np, ref=ref_np, method=scail2_transition_colormatch), dtype=np.float32)
                    except Exception as e:
                        if not warned_error:
                            log.warning(f"SCAIL-2 colormatch method {scail2_transition_colormatch!r} failed; keeping original frame: {e}")
                            warned_error = True
                        out = frame_np
                    if out.shape != frame_np.shape:
                        if not warned_shape:
                            log.warning(
                                f"SCAIL-2 colormatch method {scail2_transition_colormatch!r} returned shape {out.shape}, "
                                f"expected {frame_np.shape}; keeping original frame"
                            )
                            warned_shape = True
                        out = frame_np
                    elif not np.isfinite(out).all():
                        if not warned_nonfinite:
                            log.warning(f"SCAIL-2 colormatch method {scail2_transition_colormatch!r} returned non-finite values; keeping finite pixels only")
                            warned_nonfinite = True
                        out = np.where(np.isfinite(out), out, frame_np)
                    matched.append(torch.from_numpy(out).to(device=video_cthw.device, dtype=torch.float32))
                return torch.stack(matched, dim=0).clamp(0.0, 1.0).mul(2.0).sub(1.0).permute(3, 0, 1, 2).to(dtype=video_cthw.dtype)

            auto_drift_max_frames = 5
            auto_drift_jump_threshold = 0.0005
            auto_drift_max_offset = 0.02
            auto_drift_residual_max = 0.004

            def _normalize_auto_drift_means(means):
                if means is None or not isinstance(means, torch.Tensor):
                    return None
                means = means.detach().cpu().float()
                if means.ndim == 1:
                    means = means.unsqueeze(0)
                if means.ndim != 2 or means.shape[0] <= 0 or means.shape[1] < 3:
                    return None
                return means[:, :3].clamp(0.0, 1.0).contiguous()

            def _auto_drift_tail_means(video_cthw):
                if video_cthw is None or video_cthw.shape[1] == 0:
                    return None
                count = min(auto_drift_max_frames, int(video_cthw.shape[1]))
                tail = video_cthw[:, -count:].float().clamp(-1.0, 1.0).add(1.0).div(2.0)
                return tail.mean(dim=(2, 3)).permute(1, 0).detach().cpu().contiguous()

            def _auto_drift_video_frames(video_cthw, ref_means, chunk_index):
                ref_means = _normalize_auto_drift_means(ref_means)
                if video_cthw is None or video_cthw.shape[1] == 0 or ref_means is None:
                    return video_cthw

                current = video_cthw.float().clamp(-1.0, 1.0).permute(1, 2, 3, 0).add(1.0).div(2.0)
                frame_count = int(current.shape[0])
                compare_count = min(auto_drift_max_frames, int(ref_means.shape[0]), frame_count)
                if compare_count <= 0:
                    return video_cthw

                ref_means = ref_means.to(device=current.device, dtype=current.dtype)
                ref_window_mean = ref_means[-compare_count:].mean(dim=0)
                current_head_means = current[:compare_count].mean(dim=(1, 2))
                current_window_mean = current_head_means.mean(dim=0)
                jump_vec = current_window_mean - ref_window_mean
                local_jump = float(jump_vec.abs().max().item())

                global_applied = local_jump > auto_drift_jump_threshold
                if global_applied:
                    base = jump_vec.clamp(-auto_drift_max_offset, auto_drift_max_offset)
                    current = current - base.view(1, 1, 1, 3)
                    max_global = float(base.abs().max().item())
                else:
                    max_global = 0.0

                ref_last_mean = ref_means[-1]
                current_means = current.mean(dim=(1, 2))
                residual_offsets = (ref_last_mean.view(1, 3) - current_means).clamp(
                    -auto_drift_residual_max,
                    auto_drift_residual_max,
                )
                max_residual = float(residual_offsets.abs().max().item()) if residual_offsets.numel() else 0.0
                current = (current + residual_offsets.view(-1, 1, 1, 3)).clamp(0.0, 1.0)

                jump_values = ", ".join(f"{float(v):+.6f}" for v in jump_vec.detach().cpu())
                log.info(
                    f"SCAIL-2 auto_drift chunk {chunk_index + 1}/{num_chunks}: "
                    f"frames={compare_count}, jump=[{jump_values}], "
                    f"global={'yes' if global_applied else 'no'}, "
                    f"max_global={max_global:.6f}, max_residual={max_residual:.6f}"
                )

                return current.mul(2.0).sub(1.0).permute(3, 0, 1, 2).to(device=video_cthw.device, dtype=video_cthw.dtype)

            def _select_loop_colormatch_ref(chunk_idx, last_matched_ref_frame):
                if scail2_transition_colormatch in ("disabled", "auto_drift"):
                    return None
                if chunk_idx == 0:
                    if not scail2_has_transition_video:
                        return None
                    if scail2_loop_colormatch_reference == "main_ref_image":
                        return scail2_transition_match_ref
                    return scail2_transition_raw_last_frame
                if scail2_loop_colormatch_reference == "main_ref_image":
                    return scail2_transition_match_ref
                if chunk_idx > 0 and last_matched_ref_frame is not None:
                    return last_matched_ref_frame
                return scail2_transition_match_ref

            loop_cache_dir_prefix = "scail2_loop_cache_"
            loop_cache_marker = ".wananimateplus_scail2_loop_cache"
            loop_cache_mib = 1024 * 1024

            def _loop_chunk_label(chunk_idx):
                if isinstance(chunk_idx, int) and chunk_idx >= 0:
                    return f"chunk {chunk_idx + 1}/{num_chunks}"
                return "final"

            def _loop_tensor_mib(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return 0.0
                return tensor.numel() * tensor.element_size() / loop_cache_mib

            def _loop_file_mib(path):
                try:
                    return os.path.getsize(path) / loop_cache_mib
                except Exception:
                    return None

            def _loop_cache_log(stage, elapsed, chunk_idx=None, extra=None):
                message = f"SCAIL-2 loop cache: {_loop_chunk_label(chunk_idx)} {stage} took {elapsed:.2f}s"
                if extra:
                    message += f"; {extra}"
                log.info(message)

            def _is_scail2_loop_cache_dir_name(name):
                if not name.startswith(loop_cache_dir_prefix):
                    return False
                suffix = name[len(loop_cache_dir_prefix):]
                parts = suffix.split("_")
                return (
                    len(parts) == 4
                    and len(parts[0]) == 8 and parts[0].isdigit()
                    and len(parts[1]) == 6 and parts[1].isdigit()
                    and len(parts[2]) == 6 and parts[2].isdigit()
                    and parts[3].isdigit()
                )

            def _cleanup_stale_loop_temp_output_paths():
                output_dir = folder_paths.get_output_directory()
                try:
                    entries = list(os.scandir(output_dir))
                except Exception as e:
                    log.warning(f"SCAIL-2 loop: failed to scan temporary cache folders in {output_dir}: {e}")
                    return

                for entry in entries:
                    try:
                        if not entry.is_dir():
                            continue
                        marker_path = os.path.join(entry.path, loop_cache_marker)
                        if not (_is_scail2_loop_cache_dir_name(entry.name) or os.path.exists(marker_path)):
                            continue
                        shutil.rmtree(entry.path)
                        log.info(f"SCAIL-2 loop: removed stale temporary cache folder {entry.path}")
                    except Exception as e:
                        log.warning(f"SCAIL-2 loop: failed to remove stale temporary cache folder {entry.path}: {e}")

            def _remove_loop_temp_output_path(path):
                if path is None:
                    return None
                remove_start = time.perf_counter()
                try:
                    shutil.rmtree(path)
                    _loop_cache_log("temporary folder cleanup", time.perf_counter() - remove_start, extra=f"path={path}")
                    return None
                except Exception as e:
                    log.warning(
                        f"SCAIL-2 loop cache: temporary folder cleanup failed after "
                        f"{time.perf_counter() - remove_start:.2f}s; path={path}; error={e}"
                    )
                    return path

            def _create_loop_temp_output_path():
                from datetime import datetime
                path = os.path.join(
                    folder_paths.get_output_directory(),
                    f"scail2_loop_cache_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}",
                )
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, loop_cache_marker), "w", encoding="utf-8") as marker_file:
                    marker_file.write("WanAnimatePlus SCAIL-2 loop temporary tensor cache\n")
                return path

            def _save_loop_cache_tensor(cached_chunk, cache_path, frames_to_cache, chunk_idx, estimated_mib):
                save_start = time.perf_counter()
                try:
                    torch.save(cached_chunk, cache_path)
                except Exception as e:
                    elapsed = time.perf_counter() - save_start
                    log.warning(
                        f"SCAIL-2 loop cache: {_loop_chunk_label(chunk_idx)} save failed after {elapsed:.2f}s; "
                        f"frames={frames_to_cache}; tensor={estimated_mib:.2f} MiB; keeping chunk in memory: {e}"
                    )
                    return {
                        "tensor": cached_chunk,
                        "frames": frames_to_cache,
                        "chunk_idx": chunk_idx,
                        "estimated_mib": estimated_mib,
                    }
                del cached_chunk
                file_mib = _loop_file_mib(cache_path)
                file_extra = f"{file_mib:.2f} MiB" if file_mib is not None else "unknown"
                _loop_cache_log(
                    "cache save",
                    time.perf_counter() - save_start,
                    chunk_idx,
                    extra=f"frames={frames_to_cache}; tensor={estimated_mib:.2f} MiB; file={file_extra}; path={cache_path}",
                )
                return {
                    "path": cache_path,
                    "frames": frames_to_cache,
                    "chunk_idx": chunk_idx,
                    "estimated_mib": estimated_mib,
                    "file_mib": file_mib,
                }

            def _resolve_loop_cache_entry(entry):
                if entry is None:
                    return None
                future = entry.pop("future", None)
                if future is None:
                    return entry
                chunk_idx = entry.get("chunk_idx")
                wait_start = time.perf_counter()
                if not future.done():
                    submitted_at = entry.get("submitted_at")
                    age = time.perf_counter() - submitted_at if submitted_at is not None else None
                    age_text = f"; submitted {age:.2f}s ago" if age is not None else ""
                    log.info(
                        f"SCAIL-2 loop cache: waiting for pending save for {_loop_chunk_label(chunk_idx)}"
                        f"{age_text}; frames={entry.get('frames', '?')}; tensor={entry.get('estimated_mib', 0.0):.2f} MiB"
                    )
                resolved = future.result()
                _loop_cache_log(
                    "pending cache wait",
                    time.perf_counter() - wait_start,
                    chunk_idx,
                    extra=f"frames={entry.get('frames', '?')}; tensor={entry.get('estimated_mib', 0.0):.2f} MiB",
                )
                entry.clear()
                entry.update(resolved)
                return entry

            def _cache_loop_output_chunk(video_cthw, output_dir, chunk_idx, max_frames, executor=None):
                if video_cthw is None or max_frames <= 0:
                    return None
                frames_to_cache = min(int(video_cthw.shape[1]), int(max_frames))
                if frames_to_cache <= 0:
                    return None
                prepare_start = time.perf_counter()
                cached_chunk = video_cthw[:, :frames_to_cache].detach().cpu().contiguous()
                estimated_mib = _loop_tensor_mib(cached_chunk)
                _loop_cache_log(
                    "cache prepare",
                    time.perf_counter() - prepare_start,
                    chunk_idx,
                    extra=f"frames={frames_to_cache}; tensor={estimated_mib:.2f} MiB",
                )
                if output_dir is None:
                    _loop_cache_log(
                        "cache keep in memory",
                        0.0,
                        chunk_idx,
                        extra=f"frames={frames_to_cache}; tensor={estimated_mib:.2f} MiB",
                    )
                    return {
                        "tensor": cached_chunk,
                        "frames": frames_to_cache,
                        "chunk_idx": chunk_idx,
                        "estimated_mib": estimated_mib,
                    }

                cache_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.pt")
                if executor is None:
                    return _save_loop_cache_tensor(cached_chunk, cache_path, frames_to_cache, chunk_idx, estimated_mib)
                submit_start = time.perf_counter()
                submitted_at = time.perf_counter()
                future = executor.submit(
                    _save_loop_cache_tensor,
                    cached_chunk,
                    cache_path,
                    frames_to_cache,
                    chunk_idx,
                    estimated_mib,
                )
                _loop_cache_log(
                    "cache save submit",
                    time.perf_counter() - submit_start,
                    chunk_idx,
                    extra=f"frames={frames_to_cache}; tensor={estimated_mib:.2f} MiB; path={cache_path}",
                )
                return {
                    "future": future,
                    "frames": frames_to_cache,
                    "chunk_idx": chunk_idx,
                    "estimated_mib": estimated_mib,
                    "path": cache_path,
                    "submitted_at": submitted_at,
                }

            def _load_loop_cache_entry(entry):
                entry = _resolve_loop_cache_entry(entry)
                if "tensor" in entry:
                    _loop_cache_log(
                        "cache load from memory",
                        0.0,
                        entry.get("chunk_idx"),
                        extra=f"frames={entry.get('frames', '?')}; tensor={entry.get('estimated_mib', 0.0):.2f} MiB",
                    )
                    return entry["tensor"]
                load_start = time.perf_counter()
                try:
                    chunk = torch.load(entry["path"], map_location="cpu", weights_only=True)
                except TypeError:
                    chunk = torch.load(entry["path"], map_location="cpu")
                file_mib = _loop_file_mib(entry["path"])
                file_extra = f"{file_mib:.2f} MiB" if file_mib is not None else "unknown"
                _loop_cache_log(
                    "cache load",
                    time.perf_counter() - load_start,
                    entry.get("chunk_idx"),
                    extra=f"frames={entry.get('frames', '?')}; file={file_extra}; path={entry['path']}",
                )
                return chunk

            def _assemble_loop_cached_chunks(entries, max_frames):
                if not entries or max_frames <= 0:
                    return None
                assemble_start = time.perf_counter()
                total_frames = min(sum(int(entry["frames"]) for entry in entries), int(max_frames))
                if total_frames <= 0:
                    return None

                first_entry = entries[0]
                first_chunk = _load_loop_cache_entry(first_entry)
                if not isinstance(first_chunk, torch.Tensor) or first_chunk.ndim != 4:
                    raise RuntimeError("SCAIL-2 loop tensor cache entry is invalid")
                gen_video = torch.empty(
                    first_chunk.shape[0],
                    total_frames,
                    first_chunk.shape[2],
                    first_chunk.shape[3],
                    dtype=first_chunk.dtype,
                    device=torch.device("cpu"),
                )

                def _copy_cached_chunk(chunk, entry, write_pos):
                    if chunk.shape[0] != gen_video.shape[0] or chunk.shape[2:] != gen_video.shape[2:]:
                        raise RuntimeError(
                            f"SCAIL-2 loop tensor cache shape mismatch: expected "
                            f"{tuple(gen_video.shape[0:1] + gen_video.shape[2:])}, got {tuple(chunk.shape)}"
                    )
                    take = min(int(chunk.shape[1]), total_frames - write_pos)
                    if take > 0:
                        copy_start = time.perf_counter()
                        gen_video[:, write_pos:write_pos + take].copy_(chunk[:, :take])
                        _loop_cache_log(
                            "assemble copy",
                            time.perf_counter() - copy_start,
                            entry.get("chunk_idx"),
                            extra=f"frames={take}; write_pos={write_pos}",
                        )
                        write_pos += take
                    entry.pop("tensor", None)
                    return write_pos

                write_pos = 0
                write_pos = _copy_cached_chunk(first_chunk, first_entry, write_pos)
                del first_chunk
                for entry in entries[1:]:
                    if write_pos >= total_frames:
                        break
                    chunk = _load_loop_cache_entry(entry)
                    write_pos = _copy_cached_chunk(chunk, entry, write_pos)
                    del chunk
                _loop_cache_log(
                    "assemble complete",
                    time.perf_counter() - assemble_start,
                    extra=f"frames={write_pos}; entries={len(entries)}",
                )
                return gen_video

            def _make_local_freeze(global_latents, global_mask, chunk_start, prev_anchor_latents, first_chunk):
                local_latents = torch.zeros(16, chunk_latent_frames, lat_h, lat_w, device=device, dtype=dtype)
                local_mask = torch.zeros(chunk_latent_frames, lat_h, lat_w, device=device, dtype=dtype)

                if first_chunk and global_latents is not None:
                    g_lat = _fit_latent_time(global_latents.to(device, dtype), chunk_latent_frames)
                    g_lat = _resize_latent_spatial(g_lat, lat_h, lat_w)
                    copy_len = min(chunk_latent_frames, g_lat.shape[1])
                    local_latents[:, :copy_len] = g_lat[:, :copy_len]
                    if global_mask is not None:
                        g_mask = global_mask
                        if g_mask.ndim == 5:
                            g_mask = g_mask.squeeze(0).squeeze(0)
                        elif g_mask.ndim == 4:
                            g_mask = g_mask.squeeze(0) if g_mask.shape[0] == 1 else g_mask[0]
                        g_mask = g_mask.to(device, dtype)
                        if g_mask.shape[0] < chunk_latent_frames:
                            pad = torch.zeros(chunk_latent_frames - g_mask.shape[0], g_mask.shape[1], g_mask.shape[2], device=device, dtype=dtype)
                            g_mask = torch.cat([g_mask, pad], dim=0)
                        g_mask = g_mask[:chunk_latent_frames]
                        if g_mask.shape[1:] != (lat_h, lat_w):
                            g_mask = torch.nn.functional.interpolate(
                                g_mask.unsqueeze(0).unsqueeze(0),
                                size=(chunk_latent_frames, lat_h, lat_w),
                                mode="trilinear",
                                align_corners=False,
                            )[0, 0]
                        local_mask[:copy_len] = torch.maximum(local_mask[:copy_len], g_mask[:copy_len])

                if prev_anchor_latents is not None:
                    anchor = _fit_latent_time(prev_anchor_latents.to(device, dtype), prev_latent_count)
                    anchor = _resize_latent_spatial(anchor, lat_h, lat_w)
                    copy_len = min(prev_latent_count, chunk_latent_frames, anchor.shape[1])
                    local_latents[:, :copy_len] = anchor[:, :copy_len]
                    local_mask[:copy_len] = 1.0

                if local_mask.any():
                    return local_latents, local_mask
                return None, None

            def _expand_local_freeze_mask(local_mask, channels, strength=1.0):
                if local_mask is None:
                    return None
                strength = max(0.0, min(1.0, float(strength)))
                if strength <= 0.0:
                    return None
                mask = (local_mask * strength).clamp(0.0, 1.0)
                return mask.unsqueeze(0).repeat(1, channels, 1, 1, 1).to(device)

            def _apply_local_freeze(latent_in, local_latents, local_mask, strength=1.0):
                if local_latents is None or local_mask is None:
                    return latent_in
                strength = max(0.0, min(1.0, float(strength)))
                if strength <= 0.0:
                    return latent_in
                mask = (local_mask.unsqueeze(0).to(latent_in) * strength).clamp(0.0, 1.0)
                return local_latents.to(latent_in) * mask + latent_in * (1.0 - mask)

            def _make_local_freeze_base(local_latents, local_mask, latent_ref):
                if local_latents is None or local_mask is None:
                    return None
                return {
                    "latents": local_latents.to(latent_ref),
                    "latent_mask": local_mask.unsqueeze(0).to(latent_ref),
                    "scail_mask": local_mask,
                }

            def _make_local_freeze_state(freeze_base, strength, channels):
                if freeze_base is None:
                    return None
                strength = max(0.0, min(1.0, float(strength)))
                if strength <= 0.0:
                    return None
                latent_mask = (freeze_base["latent_mask"] * strength).clamp(0.0, 1.0)
                scail_mask = (freeze_base["scail_mask"] * strength).clamp(0.0, 1.0)
                return {
                    "frozen_part": freeze_base["latents"] * latent_mask,
                    "inverse_mask": 1.0 - latent_mask,
                    "scail_mask": scail_mask.unsqueeze(0).repeat(1, channels, 1, 1, 1).to(device),
                }

            def _apply_local_freeze_state(latent_in, freeze_state):
                if freeze_state is None:
                    return latent_in
                frozen_part = freeze_state["frozen_part"]
                inverse_mask = freeze_state["inverse_mask"]
                if frozen_part.device != latent_in.device or frozen_part.dtype != latent_in.dtype:
                    frozen_part = frozen_part.to(latent_in)
                    inverse_mask = inverse_mask.to(latent_in)
                return frozen_part + latent_in * inverse_mask

            def _build_chunk_scail_data(base_scail_data, chunk_start, local_freeze_mask, first_chunk):
                chunk_data = base_scail_data.copy() if base_scail_data is not None else {}
                chunk_latent_start = chunk_start // 4
                zero_mask_source = scail_condition_zero_mask_global if first_chunk else None
                if pose_pixels_all is not None:
                    pose_window = _slice_with_last_pad(pose_pixels_all, chunk_start, chunk_frames, 0).to(device)
                    pose_latent = _encode_video_window(
                        pose_window,
                        max(1, lat_w * vae_upscale_factor // 2),
                        max(1, lat_h * vae_upscale_factor // 2),
                        scale=float((base_scail_data or {}).get("pose_strength", 1.0)),
                    )
                    pose_zero_mask = _slice_mask_window(
                        zero_mask_source,
                        chunk_start,
                        keep_mask=scail_transition_keep_mask_global,
                    )
                    pose_latent = _zero_condition_latents(pose_latent, pose_zero_mask)
                    chunk_data["pose_latent"] = pose_latent
                else:
                    chunk_data.pop("pose_latent", None)

                if pose_mask_pixels_all is not None:
                    mask_window = _slice_with_last_pad(pose_mask_pixels_all, chunk_start, chunk_frames, 0).to(device)
                    sam_latents = _mask_pixels_to_latents(mask_window)
                    sam_zero_mask = _slice_mask_window(
                        zero_mask_source,
                        chunk_start,
                        keep_mask=scail_transition_keep_mask_global,
                    )
                    sam_keep_mask = _slice_mask_window(
                        scail_sam_keep_mask_global,
                        chunk_start,
                    )
                    if sam_zero_mask is not None and sam_keep_mask is not None:
                        sam_zero_mask &= ~sam_keep_mask
                    sam_latents = _zero_condition_latents(sam_latents, sam_zero_mask)
                    chunk_data["sam_latents"] = sam_latents
                else:
                    chunk_data.pop("sam_latents", None)

                ref_mask_latents = chunk_data.get("ref_mask_latents", None)
                if ref_mask_latents is not None:
                    ref_latent = chunk_data.get("ref_latent_pos", chunk_data.get("ref_latent_neg", None))
                    ref_count = min(ref_latent.shape[1], ref_mask_latents.shape[1]) if ref_latent is not None else 0
                    ref_mask_prefix = ref_mask_latents[:, :ref_count]
                    ref_mask_target = ref_mask_latents[:, ref_count:]
                    target_window = _slice_with_last_pad(ref_mask_target, chunk_latent_start, chunk_latent_frames, 1) if ref_mask_target.shape[1] > 0 else ref_mask_target
                    chunk_data["ref_mask_latents"] = torch.cat([ref_mask_prefix, target_window], dim=1)
                return chunk_data

            def _build_chunk_uni3c_data(chunk_start, first_chunk):
                if uni3c_data is None:
                    return None
                render_latent = uni3c_data.get("render_latent", None)
                if render_latent is None:
                    return None

                handoff_latents = 0 if first_chunk else min(prev_latent_count, chunk_latent_frames)
                valid_len = chunk_latent_frames - handoff_latents
                valid_start = chunk_start // 4 + handoff_latents

                if valid_len > 0:
                    valid = _slice_with_last_pad(render_latent, valid_start, valid_len, 2)
                else:
                    valid = _slice_with_last_pad(render_latent, chunk_start // 4, 1, 2).narrow(2, 0, 0)

                if handoff_latents > 0:
                    if valid.shape[2] > 0:
                        pad_source = valid[:, :, :1]
                    else:
                        pad_source = _slice_with_last_pad(render_latent, chunk_start // 4, 1, 2)
                    handoff = pad_source.repeat(1, 1, handoff_latents, 1, 1)
                    local_render_latent = torch.cat([handoff, valid], dim=2)
                else:
                    local_render_latent = valid

                if local_render_latent.shape[2] != chunk_latent_frames:
                    local_render_latent = _slice_with_last_pad(local_render_latent, 0, chunk_latent_frames, 2)
                if local_render_latent.shape[2:] != (chunk_latent_frames, lat_h, lat_w):
                    local_render_latent = torch.nn.functional.interpolate(
                        local_render_latent,
                        size=(chunk_latent_frames, lat_h, lat_w),
                        mode="trilinear",
                        align_corners=False,
                    )

                chunk_data = {"render_latent": local_render_latent.to(device)}
                for k, v in uni3c_data.items():
                    if k != "render_latent":
                        chunk_data[k] = v
                return chunk_data

            def _encode_anchor_from_video(video_cthw):
                anchor = video_cthw[:, -prev_frame_count:].to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)
                vae.to(device)
                anchor_pixels = anchor.to(device, vae.dtype)
                anchor_latent = vae.encode([anchor_pixels], device, tiled=tiled_vae, pbar=False)[0].to(device, dtype)
                del anchor, anchor_pixels
                _maybe_offload_loop_vae()
                return anchor_latent

            if "comfy" in rope_function:
                transformer.rope_embedder.num_frames = chunk_latent_frames
                transformer.cached_freqs = None
                if hasattr(transformer, "cached_key"):
                    transformer.cached_key = None
            elif context_latents is None and "default" in rope_function:
                freqs = torch.cat([
                    rope_params(1024, d - 4 * (d // 6), L_test=chunk_latent_frames, k=riflex_freq_index),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ], dim=1)

            log.info(
                f"SCAIL-2 loop sampling: {requested_output_frames} requested frames, {total_generation_frames} canvas frames, {num_chunks} chunks, "
                f"{chunk_frames} frames/chunk, stride {stride_frames}, {prev_frame_count} frame handoff"
            )

            callback = prepare_callback(patcher, num_chunks * len(timesteps))
            cached_output_chunks = []
            cached_loop_frame_count = 0
            loop_cache_frame_limit = requested_output_frames
            if canvas_expansion_px and not scail2_has_transition_video:
                loop_cache_frame_limit += max(int(canvas_expansion_px), 0)
            prev_anchor_latents = None
            last_matched_ref_frame = None
            last_auto_drift_ref_means = _normalize_auto_drift_means(scail2_transition_raw_tail_means)
            chunk_seeds = []
            step_iteration_count = 0
            loop_cache_output_path = None
            loop_cache_executor = None
            pending_loop_cache_entry = None
            try:
                try:
                    _cleanup_stale_loop_temp_output_paths()
                    loop_cache_output_path = _create_loop_temp_output_path()
                    log.info(f"SCAIL-2 loop: caching completed output chunks to temporary folder {loop_cache_output_path}")
                except Exception as e:
                    loop_cache_output_path = None
                    log.warning(f"SCAIL-2 loop tensor cache will use in-memory fallback; could not create temporary folder: {e}")
                if loop_cache_output_path is not None:
                    try:
                        from concurrent.futures import ThreadPoolExecutor
                        loop_cache_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scail2_loop_cache")
                    except Exception as e:
                        loop_cache_executor = None
                        log.warning(f"SCAIL-2 loop tensor cache will save synchronously; could not start background worker: {e}")
            except BaseException:
                if loop_cache_executor is not None:
                    loop_cache_executor.shutdown(wait=True)
                    loop_cache_executor = None
                loop_cache_output_path = _remove_loop_temp_output_path(loop_cache_output_path)
                raise

            try:
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * stride_frames
                    chunk_seed = int.from_bytes(os.urandom(8), "little")
                    chunk_seeds.append(chunk_seed)
                    chunk_generator = torch.Generator(device=torch.device("cpu"))
                    chunk_generator.manual_seed(chunk_seed)
                    log.info(f"SCAIL-2 chunk {chunk_idx + 1}/{num_chunks}: start={chunk_start}, seed={chunk_seed}")

                    if isinstance(scheduler, dict):
                        chunk_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                        chunk_timesteps = scheduler["timesteps"]
                    else:
                        chunk_scheduler, chunk_timesteps, _, _ = get_scheduler(
                            scheduler, total_steps, start_step, end_step, shift, device,
                            transformer.dim, denoise_strength, sigmas=sigmas,
                        )
                    if hasattr(chunk_scheduler, "timesteps"):
                        chunk_scheduler.timesteps = chunk_timesteps
                    chunk_step_args = dict(scheduler_step_args)
                    chunk_step_args["generator"] = chunk_generator
                    step_sig = inspect.signature(chunk_scheduler.step)
                    for arg in list(chunk_step_args.keys()):
                        if arg not in step_sig.parameters:
                            chunk_step_args.pop(arg)

                    chunk_two_phase = scail2_two_phase and scail2_two_phase_start_step > 0
                    if chunk_two_phase:
                        if scail2_two_phase_start_step >= len(chunk_timesteps):
                            raise ValueError(
                                "SCAIL-2 two-phase phase2_start_step must be smaller than the chunk step count."
                            )
                        log.info(
                            "SCAIL-2 two-phase chunk %d/%d: phase2 starts at step %d, phase1_mask=%.3f, phase2_mask=%.3f",
                            chunk_idx + 1,
                            num_chunks,
                            scail2_two_phase_start_step,
                            scail2_two_phase_phase1_mask,
                            scail2_two_phase_phase2_mask,
                        )

                    latent = torch.randn(
                        16,
                        chunk_latent_frames,
                        lat_h,
                        lat_w,
                        dtype=torch.float32,
                        generator=chunk_generator,
                        device=torch.device("cpu"),
                    ).to(device)
                    local_freeze_latents, local_freeze_mask = _make_local_freeze(
                        freeze_latents_global,
                        freeze_mask_global,
                        chunk_start,
                        prev_anchor_latents,
                        chunk_idx == 0,
                    )
                    if chunk_idx > 0:
                        prev_anchor_latents = None

                    freeze_base = _make_local_freeze_base(local_freeze_latents, local_freeze_mask, latent)
                    full_freeze_state = _make_local_freeze_state(freeze_base, 1.0, latent.shape[0])
                    phase1_freeze_state = (
                        _make_local_freeze_state(freeze_base, scail2_two_phase_phase1_mask, latent.shape[0])
                        if chunk_two_phase else None
                    )
                    phase2_freeze_state = (
                        _make_local_freeze_state(freeze_base, scail2_two_phase_phase2_mask, latent.shape[0])
                        if chunk_two_phase else None
                    )
                    if full_freeze_state is not None:
                        latent = _apply_local_freeze_state(latent, full_freeze_state).detach()

                    local_scail_data = _build_chunk_scail_data(scail_data, chunk_start, local_freeze_mask, chunk_idx == 0)
                    local_scail_data = dict_to_device(local_scail_data, device, dtype)
                    local_uni3c_data = _build_chunk_uni3c_data(chunk_start, chunk_idx == 0)
                    local_scail_freeze_mask = full_freeze_state["scail_mask"] if full_freeze_state is not None else None

                    self.cache_state = [None, None]
                    seq_len = math.ceil((latent.shape[2] * latent.shape[3]) / 4 * latent.shape[1])
                    chunk_pbar = tqdm(total=len(chunk_timesteps), desc=f"SCAIL-2 chunk {chunk_idx + 1}/{num_chunks}", position=0, leave=True)
                    old_scail_freeze_mask = scail_freeze_mask
                    sample_start = time.perf_counter()
                    try:
                        for i, t in enumerate(chunk_timesteps):
                            if chunk_two_phase:
                                phase_freeze_state = (
                                    phase1_freeze_state
                                    if i < scail2_two_phase_start_step
                                    else phase2_freeze_state
                                )
                                scail_freeze_mask = (
                                    phase_freeze_state["scail_mask"] if phase_freeze_state is not None else None
                                )
                            else:
                                phase_freeze_state = full_freeze_state
                                scail_freeze_mask = local_scail_freeze_mask

                            timestep = torch.tensor([t]).to(device)
                            latent_model_input = latent.to(device)
                            noise_pred, _, self.cache_state = predict_with_cfg(
                                latent_model_input,
                                cfg[min(i, len(cfg) - 1)],
                                text_embeds["prompt_embeds"],
                                text_embeds["negative_prompt_embeds"],
                                timestep,
                                i,
                                clip_fea=clip_fea,
                                cache_state=self.cache_state,
                                scail_data_override=local_scail_data,
                                uni3c_data=local_uni3c_data,
                            )
                            if use_tsr:
                                noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)
                            latent = chunk_scheduler.step(
                                noise_pred.unsqueeze(0),
                                timestep,
                                latent.unsqueeze(0).to(noise_pred.device),
                                **chunk_step_args,
                            )[0].squeeze(0).detach()
                            if freeze_base is not None:
                                latent = _apply_local_freeze_state(latent, phase_freeze_state).detach()
                                if chunk_two_phase and i == scail2_two_phase_start_step - 1:
                                    latent = _apply_local_freeze_state(latent, full_freeze_state).detach()
                                    self.cache_state = [None, None]

                            if callback is not None:
                                callback_latent = (latent_model_input - noise_pred.to(device) * timestep.to(device) / 1000).detach()
                                callback(step_iteration_count, callback_latent.permute(1, 0, 2, 3), None, num_chunks * len(chunk_timesteps))
                                del callback_latent
                            chunk_pbar.update(1)
                            step_iteration_count += 1
                            del noise_pred, latent_model_input, timestep
                    finally:
                        scail_freeze_mask = old_scail_freeze_mask
                        chunk_pbar.close()
                        self.cache_state = [None, None]
                        _loop_cache_log(
                            "sampling loop",
                            time.perf_counter() - sample_start,
                            chunk_idx,
                            extra=f"steps={len(chunk_timesteps)}",
                        )

                    if full_freeze_state is not None and not chunk_two_phase:
                        latent = _apply_local_freeze_state(latent, full_freeze_state).detach()
                    del (
                        local_scail_data,
                        local_uni3c_data,
                        local_freeze_latents,
                        local_freeze_mask,
                        local_scail_freeze_mask,
                        freeze_base,
                        full_freeze_state,
                        phase1_freeze_state,
                        phase2_freeze_state,
                    )
                    chunk_boundary_start = time.perf_counter()
                    pre_decode_cleanup_start = time.perf_counter()
                    _maybe_offload_loop_vae()
                    mm.soft_empty_cache()
                    _loop_cache_log("pre-decode cleanup", time.perf_counter() - pre_decode_cleanup_start, chunk_idx)

                    vae_to_start = time.perf_counter()
                    vae.to(device)
                    _loop_cache_log("vae to device", time.perf_counter() - vae_to_start, chunk_idx)

                    decode_start = time.perf_counter()
                    decoded_video = vae.decode(
                        latent.unsqueeze(0).to(device, vae.dtype),
                        device=device,
                        tiled=tiled_vae,
                        pbar=False,
                    )[0].detach()
                    _loop_cache_log(
                        "vae decode",
                        time.perf_counter() - decode_start,
                        chunk_idx,
                        extra=f"shape={tuple(decoded_video.shape)}; dtype={decoded_video.dtype}",
                    )

                    cpu_transfer_start = time.perf_counter()
                    chunk_video = decoded_video.cpu()
                    del decoded_video
                    _loop_cache_log(
                        "decode cpu transfer",
                        time.perf_counter() - cpu_transfer_start,
                        chunk_idx,
                        extra=f"tensor={_loop_tensor_mib(chunk_video):.2f} MiB",
                    )
                    del latent
                    vae_offload_start = time.perf_counter()
                    _maybe_offload_loop_vae()
                    _loop_cache_log("vae offload after decode", time.perf_counter() - vae_offload_start, chunk_idx)
                    output_slice_start = time.perf_counter()
                    if chunk_video.shape[1] > chunk_frames:
                        chunk_video = chunk_video[:, :chunk_frames]

                    if chunk_idx == 0:
                        overlap_frames = min(max(int(canvas_expansion_px), 0), chunk_video.shape[1]) if scail2_has_transition_video else 0
                        output_chunk = chunk_video[:, overlap_frames:].contiguous() if overlap_frames > 0 else chunk_video
                    else:
                        output_chunk = chunk_video[:, prev_frame_count:].contiguous()
                    del chunk_video
                    _loop_cache_log(
                        "output slice",
                        time.perf_counter() - output_slice_start,
                        chunk_idx,
                        extra=f"frames={int(output_chunk.shape[1])}; tensor={_loop_tensor_mib(output_chunk):.2f} MiB",
                    )

                    if output_chunk.shape[1] > 0:
                        if scail2_transition_colormatch == "auto_drift":
                            colormatch_start = time.perf_counter()
                            output_chunk = _auto_drift_video_frames(output_chunk, last_auto_drift_ref_means, chunk_idx)
                            _loop_cache_log(
                                "colormatch",
                                time.perf_counter() - colormatch_start,
                                chunk_idx,
                                extra=f"method={scail2_transition_colormatch}; tensor={_loop_tensor_mib(output_chunk):.2f} MiB",
                            )
                        else:
                            match_ref = _select_loop_colormatch_ref(chunk_idx, last_matched_ref_frame)
                            if match_ref is not None:
                                colormatch_start = time.perf_counter()
                                output_chunk = _color_match_video_frames(output_chunk, match_ref)
                                _loop_cache_log(
                                    "colormatch",
                                    time.perf_counter() - colormatch_start,
                                    chunk_idx,
                                    extra=f"method={scail2_transition_colormatch}; tensor={_loop_tensor_mib(output_chunk):.2f} MiB",
                                )
                            else:
                                _loop_cache_log(
                                    "colormatch skipped",
                                    0.0,
                                    chunk_idx,
                                    extra=f"method={scail2_transition_colormatch}",
                                )

                        ref_update_start = time.perf_counter()
                        last_matched_ref_frame = (
                            output_chunk[:, -1:]
                            .float()
                            .clamp(-1.0, 1.0)
                            .permute(1, 2, 3, 0)
                            .add(1.0)
                            .div(2.0)
                            .detach()
                            .cpu()
                        )
                        last_auto_drift_ref_means = _auto_drift_tail_means(output_chunk)
                        _loop_cache_log("last matched ref update", time.perf_counter() - ref_update_start, chunk_idx)
                        remaining_cache_frames = loop_cache_frame_limit - cached_loop_frame_count
                        if pending_loop_cache_entry is not None:
                            _resolve_loop_cache_entry(pending_loop_cache_entry)
                            pending_loop_cache_entry = None
                        cache_entry = _cache_loop_output_chunk(
                            output_chunk,
                            loop_cache_output_path,
                            chunk_idx,
                            remaining_cache_frames,
                            loop_cache_executor,
                        )
                        if cache_entry is not None:
                            cached_output_chunks.append(cache_entry)
                            cached_loop_frame_count += int(cache_entry["frames"])
                            if "future" in cache_entry:
                                pending_loop_cache_entry = cache_entry

                        anchor_frames = None
                        if chunk_idx + 1 < num_chunks:
                            anchor_frame_count = min(prev_frame_count, int(output_chunk.shape[1]))
                            if anchor_frame_count > 0:
                                anchor_slice_start = time.perf_counter()
                                anchor_frames = output_chunk[:, -anchor_frame_count:].contiguous()
                                _loop_cache_log(
                                    "anchor slice",
                                    time.perf_counter() - anchor_slice_start,
                                    chunk_idx,
                                    extra=f"frames={anchor_frame_count}; tensor={_loop_tensor_mib(anchor_frames):.2f} MiB",
                                )
                                anchor_encode_start = time.perf_counter()
                                prev_anchor_latents = _encode_anchor_from_video(anchor_frames)
                                _loop_cache_log(
                                    "anchor encode",
                                    time.perf_counter() - anchor_encode_start,
                                    chunk_idx,
                                    extra=f"latents={tuple(prev_anchor_latents.shape)}",
                                )
                        if anchor_frames is not None:
                            del anchor_frames
                        del output_chunk
                    else:
                        del output_chunk
                    if chunk_idx == 0:
                        scail2_transition_raw_last_frame = None
                        scail2_transition_raw_tail_means = None

                    chunk_cleanup_start = time.perf_counter()
                    mm.soft_empty_cache()
                    gc.collect()
                    _loop_cache_log("chunk cleanup", time.perf_counter() - chunk_cleanup_start, chunk_idx)
                    _loop_cache_log(
                        "post-sampling boundary total",
                        time.perf_counter() - chunk_boundary_start,
                        chunk_idx,
                    )

                if pending_loop_cache_entry is not None:
                    _resolve_loop_cache_entry(pending_loop_cache_entry)
                    pending_loop_cache_entry = None
                gen_video_samples = _assemble_loop_cached_chunks(cached_output_chunks, loop_cache_frame_limit)
                if gen_video_samples is None:
                    raise RuntimeError("SCAIL-2 loop produced no output frames")
                if canvas_expansion_px and not scail2_has_transition_video:
                    if gen_video_samples.shape[1] > canvas_expansion_px:
                        gen_video_samples = gen_video_samples[:, canvas_expansion_px:]
                        log.info(f"SCAIL-2 loop: trimmed {canvas_expansion_px} canvas expansion pixel frames from output")
                    else:
                        log.warning(
                            f"SCAIL-2 loop: canvas_expansion_px={canvas_expansion_px} is not smaller than output length {gen_video_samples.shape[1]}, skipping trim"
                        )
                if gen_video_samples.shape[1] > requested_output_frames:
                    gen_video_samples = gen_video_samples[:, :requested_output_frames]
                log.info(f"SCAIL-2 chunk seeds: {chunk_seeds}")
                if loop_cache_output_path is not None:
                    log.info(f"SCAIL-2 loop: cached {cached_loop_frame_count} output frames to {loop_cache_output_path}")
                    loop_cache_output_path = _remove_loop_temp_output_path(loop_cache_output_path)

                if force_offload:
                    vae.to(offload_device)
                    if not model["auto_cpu_offload"]:
                        offload_transformer(transformer)
                try:
                    print_memory(device)
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
                return ({
                    "video": gen_video_samples.permute(1, 2, 3, 0),
                    "scail2_chunk_seeds": chunk_seeds,
                },)
            finally:
                if pending_loop_cache_entry is not None:
                    try:
                        _resolve_loop_cache_entry(pending_loop_cache_entry)
                    except Exception as e:
                        log.warning(f"SCAIL-2 loop: failed to resolve pending tensor cache save during cleanup: {e}")
                    pending_loop_cache_entry = None
                if loop_cache_executor is not None:
                    loop_cache_executor.shutdown(wait=True)
                    loop_cache_executor = None
                loop_cache_output_path = _remove_loop_temp_output_path(loop_cache_output_path)
                _scail2_restore_rope_state()

        # Main sampling loop with FreeInit iterations
        iterations = freeinit_args.get("freeinit_num_iters", 3) if freeinit_args is not None else 1
        current_latent = latent
        initial_noise_saved = None

        for iter_idx in range(iterations):

            # FreeInit noise reinitialization (after first iteration)
            if freeinit_args is not None and iter_idx > 0:
                # restart scheduler for each iteration
                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer.dim, denoise_strength, sigmas=sigmas)

                # Re-apply start_step and end_step logic to timesteps and sigmas
                if end_step != -1:
                    timesteps = timesteps[:end_step]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[:end_step+1]
                if start_step > 0:
                    timesteps = timesteps[start_step:]
                    sample_scheduler.sigmas = sample_scheduler.sigmas[start_step:]
                if hasattr(sample_scheduler, 'timesteps'):
                    sample_scheduler.timesteps = timesteps

                # Diffuse current latent to t=999
                diffuse_timesteps = torch.full((noise.shape[0],), 999, device=device, dtype=torch.long)
                z_T = add_noise(
                    current_latent.to(device),
                    initial_noise_saved.to(device),
                    diffuse_timesteps
                )

                # Generate new random noise
                z_rand = torch.randn(z_T.shape, dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))
                # Apply frequency mixing
                current_latent = (freq_mix_3d(z_T.to(torch.float32), z_rand.to(device), LPF=freq_filter)).to(dtype)

            # Store initial noise for first iteration
            if freeinit_args is not None and iter_idx == 0:
                initial_noise_saved = current_latent.detach().clone()
                if input_samples is not None:
                    current_latent = input_samples.to(device)
                    continue

            # Reset per-iteration states
            self.cache_state = [None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []
            if context_options is not None:
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])

            # Set latent for denoising
            latent = current_latent

            if is_pusa and clean_latent_indices:
                pusa_noisy_steps = image_embeds.get("pusa_noisy_steps", -1)
                if pusa_noisy_steps == -1:
                    pusa_noisy_steps = len(timesteps)
            try:
                pbar = ProgressBar(len(timesteps) - ttm_start_step)
                #region main loop start
                for idx, t in enumerate(tqdm(timesteps[ttm_start_step:], disable=multitalk_sampling or wananimate_loop)):

                    if bidirectional_sampling:
                        latent_flipped = torch.flip(latent, dims=[1])
                        latent_model_input_flipped = latent_flipped.to(device)

                    self.noise_front_pad_num = 0

                    #InfiniteTalk first frame handling
                    if (extra_latents is not None
                        and not multitalk_sampling
                        and transformer.multitalk_model_type=="InfiniteTalk"):
                        for entry in extra_latents:
                            add_index = entry["index"]
                            num_extra_frames = entry["samples"].shape[2]
                            latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                    latent_model_input = latent.to(device)
                    latent_model_input_ovi = latent_ovi.to(device) if latent_ovi is not None else None

                    current_step_percentage = idx / len(timesteps)

                    timestep = torch.tensor([t]).to(device)
                    if is_pusa or ((is_5b or transformer.is_longcat) and clean_latent_indices):
                        orig_timestep = timestep
                        timestep = timestep.unsqueeze(1).repeat(1, latent_video_length)
                        if extra_latents is not None:
                            if clean_latent_indices and noise_multipliers is not None:
                                if is_pusa:
                                    scheduler_step_args["cond_frame_latent_indices"] = clean_latent_indices
                                    scheduler_step_args["noise_multipliers"] = noise_multipliers
                                for latent_idx in clean_latent_indices:
                                    timestep[:, latent_idx] = timestep[:, latent_idx] * noise_multipliers[latent_idx]
                                    # add noise for conditioning frames if multiplier > 0
                                    if idx < pusa_noisy_steps and noise_multipliers[latent_idx] > 0:
                                        latent_size = (1, latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])
                                        noise_for_cond = torch.randn(latent_size, generator=seed_g, device=torch.device("cpu"))
                                        timestep_cond = torch.ones_like(timestep) * timestep.max()
                                        if is_pusa:
                                            latent[:, latent_idx:latent_idx+1] = sample_scheduler.add_noise_for_conditioning_frames(
                                                latent[:, latent_idx:latent_idx+1].to(device),
                                                noise_for_cond[:, :, latent_idx:latent_idx+1].to(device),
                                                timestep_cond[:, latent_idx:latent_idx+1].to(device),
                                                noise_multiplier=noise_multipliers[latent_idx])
                            else:
                                timestep[:, clean_latent_indices] = 0
                            #print("timestep: ", timestep)

                    ### latent shift
                    if latent_shift_loop:
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            latent_model_input = torch.cat([latent_model_input[:, shift_idx:]] + [latent_model_input[:, :shift_idx]], dim=1)

                    #enhance-a-video
                    enhance_enabled = False
                    if feta_args is not None and feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enhance_enabled = True
                    #region context windowing
                    if context_options is not None:
                        # ============ Get latent spatial dimensions ============
                        lat_h = latent.shape[-2]
                        lat_w = latent.shape[-1]
                        # ============================================
                        counter = torch.zeros_like(latent_model_input, device=device)
                        noise_pred = torch.zeros_like(latent_model_input, device=device)
                        context_latent_offset = wananim_static_ref_latents
                        context_target_length = latent_video_length - context_latent_offset
                        if image_cond is not None and image_cond.ndim > 1:
                            if context_latent_offset > 0:
                                context_target_length = min(context_target_length, max(image_cond.shape[1] - context_latent_offset, 0))
                            else:
                                latent_video_length = min(latent_video_length, image_cond.shape[1])
                                context_target_length = latent_video_length
                        if context_target_length <= 0:
                            raise ValueError(f"Context windows need target latents, got latent_video_length={latent_video_length}, static_refs={context_latent_offset}")
                        context_queue = list(context(idx, steps, context_target_length, context_frames, context_stride, context_overlap))
                        fraction_per_context = 1.0 / len(context_queue)
                        context_pbar = ProgressBar(steps)
                        step_start_progress = idx

                        # Validate all context windows before processing
                        max_idx = context_target_length
                        for window_indices in context_queue:
                            if not all(0 <= idx < max_idx for idx in window_indices):
                                raise ValueError(f"Invalid context window indices {window_indices} for target latent length {max_idx}")

                        context_full_length = context_latent_offset + context_target_length

                        def _index_temporal(tensor, temporal_dim, indices):
                            index = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
                            return tensor.index_select(temporal_dim, index)

                        def _prepend_static_temporal(tensor, temporal_dim, count):
                            if count <= 0:
                                return tensor
                            static_shape = list(tensor.shape)
                            static_shape[temporal_dim] = count
                            static = tensor.new_zeros(static_shape)
                            return torch.cat([static, tensor], dim=temporal_dim)

                        def _slice_context_temporal(tensor, temporal_dim, c, target_c, name, prepend_static=False):
                            if context_latent_offset <= 0:
                                return _index_temporal(tensor, temporal_dim, c)

                            temporal_len = tensor.shape[temporal_dim]
                            if temporal_len in (latent_video_length, context_full_length):
                                static = _index_temporal(tensor, temporal_dim, range(context_latent_offset))
                                target = _index_temporal(tensor, temporal_dim, target_c)
                                return torch.cat([static, target], dim=temporal_dim)

                            if temporal_len >= (max(c) + 1 if c else 0):
                                target = _index_temporal(tensor, temporal_dim, c)
                                if prepend_static:
                                    target = _prepend_static_temporal(target, temporal_dim, context_latent_offset)
                                return target

                            raise ValueError(
                                f"{name} temporal length {temporal_len} cannot cover context window {c}; "
                                f"expected full length {context_full_length} or target length at least {max(c) + 1 if c else 0}"
                            )

                        def _wananim_face_pixel_frames_to_latents(frame_count):
                            return (int(frame_count) + 3) // 4

                        def _slice_wananim_face_pixels_for_context(face_pixels, c):
                            if context_latent_offset <= 0:
                                start = c[0] * 4
                                end = c[-1] * 4
                                frame_indices = torch.arange(start, end, device=face_pixels.device, dtype=torch.long)
                            else:
                                if face_pixels.shape[2] <= 0:
                                    raise ValueError("WanAnimate face pixels must contain at least one frame")
                                frame_indices = []
                                for latent_idx in c:
                                    base = int(latent_idx) * 4
                                    frame_indices.extend([base, base + 1, base + 2, base + 3])
                                frame_indices = torch.tensor(frame_indices, device=face_pixels.device, dtype=torch.long)

                            if frame_indices.numel() > 0:
                                frame_indices = torch.clamp(frame_indices, min=0, max=face_pixels.shape[2] - 1)
                            return face_pixels.index_select(2, frame_indices)

                        def _slice_wananim_pose_latents_for_context(pose_latents, c):
                            if context_latent_offset <= 0:
                                start = c[0]
                                end = c[-1]
                                latent_indices = torch.arange(start, end, device=pose_latents.device, dtype=torch.long)
                                if latent_indices.numel() > 0:
                                    latent_indices = torch.clamp(latent_indices, min=0, max=pose_latents.shape[2] - 1)
                                pose_limit = context_frames - 1
                                return pose_latents.index_select(2, latent_indices)[:, :, :pose_limit]

                            if pose_latents.shape[2] <= 0:
                                raise ValueError("WanAnimate pose latents must contain at least one latent frame")
                            latent_indices = torch.as_tensor(c, device=pose_latents.device, dtype=torch.long)
                            latent_indices = torch.clamp(latent_indices, min=0, max=pose_latents.shape[2] - 1)
                            return pose_latents.index_select(2, latent_indices)

                        for i, c in enumerate(context_queue):
                            window_id = self.window_tracker.get_window_id(c)
                            target_c = [context_latent_offset + ci for ci in c] if context_latent_offset > 0 else c

                            if cache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                            else:
                                current_teacache = None

                            prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                            else:
                                positive = text_embeds["prompt_embeds"]

                            partial_img_emb = partial_control_latents = None
                            if image_cond is not None:
                                if context_latent_offset > 0:
                                    static_img_emb = image_cond[:, :context_latent_offset].to(device)
                                    target_img_emb = image_cond[:, target_c].to(device)
                                    partial_img_emb = torch.cat([static_img_emb, target_img_emb], dim=1)
                                else:
                                    partial_img_emb = image_cond[:, c].to(device)
                                # ============ Build msk channels ============
                                window_frames = len(c)
                                window_msk = partial_img_emb[:4].clone()
                                # ========================================

                                # ============ Transition replacement (1st window only) ============
                                if has_transition and c[0] == 0 and not has_prefix:
                                    transition_start = context_latent_offset if context_latent_offset > 0 else 1
                                    transition_count = min(transition_len, max(partial_img_emb.shape[1] - transition_start, 0))
                                    if transition_count > 0:
                                        partial_img_emb[4:20, transition_start:transition_start + transition_count] = transition_latent[:, :transition_count].to(device, dtype=transition_latent.dtype)
                                        log.info(f"Replaced first {transition_count} latent frames with transition_video (context_options mode)")
                                # ============================================================

                                # ============ Set transition mask ============
                                if has_transition and c[0] == 0 and not has_prefix:
                                    window_trans_start = context_latent_offset if context_latent_offset > 0 else 1
                                    window_trans_end = min(window_trans_start + transition_len, window_msk.shape[1])
                                    if window_trans_start < window_trans_end:
                                        mask_slice = transition_mask_values[0:window_trans_end - window_trans_start]
                                        window_msk[:, window_trans_start:window_trans_end] = mask_slice.view(1, -1, 1, 1)
                                # ================================================================

                                # ============ Prefix: prepend cached prefix context to non-first windows ============
                                if has_prefix and c[0] != 0 and prefix_prepend_latents > 0:
                                    prefix_ctx = image_cond[:, :prefix_prepend_latents].to(device, dtype)
                                    partial_img_emb = torch.cat([prefix_ctx, partial_img_emb], dim=1)
                                    prefix_msk = image_cond[:4, :prefix_prepend_latents].to(device, dtype)
                                    window_msk = torch.cat([prefix_msk, window_msk], dim=1)
                                # ======================================================================

                                # ============ ref_latent replacement (skip when prefix active) ============
                                if context_latent_offset == 0 and not has_prefix:
                                    if c[0] != 0 and context_reference_latent is not None:
                                        if context_reference_latent.shape[0] == 1:
                                            new_init_image = context_reference_latent[0, :, 0].to(device)
                                            partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0].to(device), new_init_image], dim=0)
                                            window_msk[:, 0] = image_cond[:4, 0].to(device)
                                        elif context_reference_latent.shape[0] > 1:
                                            num_extra_inits = context_reference_latent.shape[0]
                                            section_size = (latent_video_length / num_extra_inits)
                                            extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                            if context_options["verbose"]:
                                                log.info(f"extra init image index: {extra_init_index}")
                                            new_init_image = context_reference_latent[extra_init_index, :, 0].to(device)
                                            partial_img_emb[:, 0] = torch.cat([image_cond[:4, 0].to(device), new_init_image], dim=0)
                                            window_msk[:, 0] = image_cond[:4, 0].to(device)
                                    else:
                                        new_init_image = image_cond[:, 0].to(device)
                                        partial_img_emb[:, 0] = new_init_image
                                        window_msk[:, 0] = image_cond[:4, 0].to(device)
                                elif context_latent_offset > 0:
                                    main_ref_local_index = min(max(wananim_main_ref_index, 0), context_latent_offset - 1)
                                    if c[0] != 0 and context_reference_latent is not None:
                                        if context_reference_latent.shape[0] == 1:
                                            new_init_image = context_reference_latent[0, :, 0].to(device)
                                            partial_img_emb[:, main_ref_local_index] = torch.cat([image_cond[:4, wananim_main_ref_index].to(device), new_init_image], dim=0)
                                            window_msk[:, main_ref_local_index] = image_cond[:4, wananim_main_ref_index].to(device)
                                        elif context_reference_latent.shape[0] > 1:
                                            num_extra_inits = context_reference_latent.shape[0]
                                            section_size = (context_target_length / num_extra_inits)
                                            extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                            if context_options["verbose"]:
                                                log.info(f"extra init image index: {extra_init_index}")
                                            new_init_image = context_reference_latent[extra_init_index, :, 0].to(device)
                                            partial_img_emb[:, main_ref_local_index] = torch.cat([image_cond[:4, wananim_main_ref_index].to(device), new_init_image], dim=0)
                                            window_msk[:, main_ref_local_index] = image_cond[:4, wananim_main_ref_index].to(device)
                                    else:
                                        partial_img_emb[:, main_ref_local_index] = image_cond[:, wananim_main_ref_index].to(device)
                                        window_msk[:, main_ref_local_index] = image_cond[:4, wananim_main_ref_index].to(device)
                                # ==========================================================================

                                if control_latents is not None:
                                    partial_control_latents = _slice_context_temporal(
                                        control_latents, 1, c, target_c, "control_latents", prepend_static=True
                                    )

                            partial_control_camera_latents = None
                            if control_camera_latents is not None:
                                partial_control_camera_latents = _slice_context_temporal(
                                    control_camera_latents, 2, c, target_c, "control_camera_latents", prepend_static=True
                                )

                            partial_vace_context = None
                            if vace_data is not None:
                                window_vace_data = []
                                for vace_entry in vace_data:
                                    partial_context = _slice_context_temporal(
                                        vace_entry["context"][0], 1, c, target_c, "vace_context", prepend_static=True
                                    )
                                    if has_ref:
                                        ref_context_index = wananim_main_ref_index if context_latent_offset > 0 else 0
                                        if c[0] != 0 and context_reference_latent is not None:
                                            if context_reference_latent.shape[0] == 1: #only single extra init latent
                                                partial_context[16:32, ref_context_index:ref_context_index + 1] = context_reference_latent[0, :, :1].to(device)
                                            elif context_reference_latent.shape[0] > 1:
                                                num_extra_inits = context_reference_latent.shape[0]
                                                section_size = (context_target_length / num_extra_inits)
                                                extra_init_index = min(int(max(c) / section_size), num_extra_inits - 1)
                                                if context_options["verbose"]:
                                                    log.info(f"extra init image index: {extra_init_index}")
                                                partial_context[16:32, ref_context_index:ref_context_index + 1] = context_reference_latent[extra_init_index, :, :1].to(device)
                                        elif context_latent_offset == 0:
                                            partial_context[:, ref_context_index] = vace_entry["context"][0][:, ref_context_index]

                                    window_vace_data.append({
                                        "context": [partial_context],
                                        "scale": vace_entry["scale"],
                                        "start": vace_entry["start"],
                                        "end": vace_entry["end"],
                                        "seq_len": vace_entry["seq_len"]
                                    })

                                partial_vace_context = window_vace_data

                            partial_audio_proj = None
                            if fantasytalking_embeds is not None:
                                partial_audio_proj = audio_proj[:, c]

                            partial_fantasy_portrait_input = None
                            if fantasy_portrait_input is not None:
                                partial_fantasy_portrait_input = fantasy_portrait_input.copy()
                                partial_fantasy_portrait_input["adapter_proj"] = fantasy_portrait_input["adapter_proj"][:, c]

                            if context_latent_offset > 0:
                                partial_latent_model_input = torch.cat([
                                    latent_model_input[:, :context_latent_offset],
                                    latent_model_input[:, target_c],
                                ], dim=1)
                            else:
                                partial_latent_model_input = latent_model_input[:, c]
                            if latents_to_insert is not None and c[0] != 0:
                                insert_index = wananim_main_ref_index if context_latent_offset > 0 else 0
                                partial_latent_model_input[:, insert_index:insert_index + 1] = latents_to_insert

                            scail_window_prepend_latents = scail_prefix_prepend_latents if c[0] != 0 else 0
                            if scail_window_prepend_latents > 0:
                                prefix_noise = latent_model_input[:, :scail_window_prepend_latents].to(device)
                                partial_latent_model_input = torch.cat([prefix_noise, partial_latent_model_input], dim=1)

                            # ============ Prefix: prepend noise for cached prefix context ============
                            if has_prefix and c[0] != 0 and prefix_prepend_latents > 0:
                                prefix_noise = latent_model_input[:, :prefix_prepend_latents].to(device)
                                partial_latent_model_input = torch.cat([prefix_noise, partial_latent_model_input], dim=1)
                            # =============================================================

                            partial_unianim_data = None
                            if unianim_data is not None:
                                partial_dwpose = _slice_context_temporal(
                                    dwpose_data, 2, c, target_c, "dwpose_data", prepend_static=True
                                )
                                partial_unianim_data = {
                                    "dwpose": partial_dwpose,
                                    "random_ref": unianim_data["random_ref"],
                                    "strength": unianimate_poses["strength"],
                                    "start_percent": unianimate_poses["start_percent"],
                                    "end_percent": unianimate_poses["end_percent"]
                                }

                            partial_mtv_motion_tokens = None
                            if mtv_input is not None:
                                start_token_index = c[0] * 24
                                end_token_index = (c[-1] + 1) * 24
                                partial_mtv_motion_tokens = mtv_motion_tokens[:, start_token_index:end_token_index, :]
                                if context_options["verbose"]:
                                    log.info(f"context window: {c}")
                                    log.info(f"motion_token_indices: {start_token_index}-{end_token_index}")

                            partial_s2v_audio_input = None
                            if s2v_audio_input is not None:
                                audio_start = c[0] * 4
                                audio_end = c[-1] * 4 + 1
                                center_indices = torch.arange(audio_start, audio_end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=s2v_audio_input.shape[-1] - 1)
                                partial_s2v_audio_input = s2v_audio_input[..., center_indices]

                            partial_s2v_pose = None
                            if s2v_pose is not None:
                                partial_s2v_pose = _slice_context_temporal(
                                    s2v_pose, 2, c, target_c, "s2v_pose", prepend_static=True
                                ).to(device, dtype)

                            partial_add_cond = None
                            if add_cond is not None:
                                partial_add_cond = _slice_context_temporal(
                                    add_cond, 2, c, target_c, "add_cond", prepend_static=True
                                ).to(device, dtype)

                            partial_wananim_face_pixels = partial_wananim_pose_latents = None
                            if wananim_face_pixels is not None and partial_wananim_face_pixels is None:
                                partial_wananim_face_pixels = _slice_wananim_face_pixels_for_context(
                                    wananim_face_pixels, c
                                ).to(device, dtype)
                            if wananim_pose_latents is not None:
                                partial_wananim_pose_latents = _slice_wananim_pose_latents_for_context(
                                    wananim_pose_latents, c
                                ).to(device, dtype)

                            # ============ Prefix: prepend face/pose for cached prefix context ============
                            if has_prefix and c[0] != 0 and prefix_prepend_latents > 0:
                                if partial_wananim_face_pixels is not None:
                                    prefix_face = wananim_face_pixels[:, :, :prefix_prepend_latents * 4].to(device, dtype)
                                    partial_wananim_face_pixels = torch.cat([prefix_face, partial_wananim_face_pixels], dim=2)
                                if partial_wananim_pose_latents is not None:
                                    prefix_pose = wananim_pose_latents[:, :, :prefix_prepend_latents].to(device, dtype)
                                    partial_wananim_pose_latents = torch.cat([prefix_pose, partial_wananim_pose_latents], dim=2)
                            # ================================================================

                            if context_latent_offset > 0 and (
                                partial_wananim_face_pixels is not None or partial_wananim_pose_latents is not None
                            ):
                                anchor_latents = int(wananim_num_anchor_latents)
                                model_latents = int(partial_latent_model_input.shape[1])
                                target_latents = model_latents - anchor_latents
                                if anchor_latents != context_latent_offset or target_latents < 0:
                                    raise RuntimeError(
                                        "WanAnimate context anchor mismatch: "
                                        f"context_latent_offset={context_latent_offset}, "
                                        f"wananim_num_anchor_latents={anchor_latents}, "
                                        f"model_latents={model_latents}"
                                    )
                                if partial_wananim_face_pixels is not None:
                                    face_frames = int(partial_wananim_face_pixels.shape[2])
                                    face_target_latents = _wananim_face_pixel_frames_to_latents(face_frames)
                                    face_total_latents = anchor_latents + face_target_latents
                                    if face_total_latents != model_latents:
                                        raise RuntimeError(
                                            "WanAnimate face/context latent mismatch: "
                                            f"context_latent_offset={context_latent_offset}, "
                                            f"window_latents={len(c)}, model_latents={model_latents}, "
                                            f"face_pixel_frames={face_frames}, "
                                            f"face_target_latents={face_target_latents}, "
                                            f"wananim_num_anchor_latents={anchor_latents}"
                                        )
                                if partial_wananim_pose_latents is not None:
                                    pose_target_latents = int(partial_wananim_pose_latents.shape[2])
                                    if pose_target_latents != target_latents:
                                        raise RuntimeError(
                                            "WanAnimate pose/context latent mismatch: "
                                            f"context_latent_offset={context_latent_offset}, "
                                            f"window_latents={len(c)}, model_latents={model_latents}, "
                                            f"pose_target_latents={pose_target_latents}, "
                                            f"expected_target_latents={target_latents}, "
                                            f"wananim_num_anchor_latents={anchor_latents}"
                                        )

                            partial_flashvsr_LQ_latent = None
                            if LQ_images is not None:
                                start = c[0] * 4
                                end = c[-1] * 4 + 1 + 4
                                center_indices = torch.arange(start, end, 1)
                                center_indices = torch.clamp(center_indices, min=0, max=LQ_images.shape[2] - 1)
                                partial_flashvsr_LQ_images = LQ_images[:, :, center_indices].to(device)
                                partial_flashvsr_LQ_latent = transformer.LQ_proj_in(partial_flashvsr_LQ_images)

                            if len(timestep.shape) != 1:
                                if context_latent_offset > 0:
                                    static_timestep = torch.zeros(
                                        timestep.shape[0],
                                        context_latent_offset,
                                        device=timestep.device,
                                        dtype=timestep.dtype,
                                    )
                                    partial_timestep = torch.cat([static_timestep, timestep[:, target_c]], dim=1)
                                else:
                                    partial_timestep = timestep[:, c]
                                    partial_timestep[:, :1] = 0
                            else:
                                partial_timestep = timestep

                            orig_model_input_frames = partial_latent_model_input.shape[1]

                            # ============ Replace msk placeholder channels ============
                            # image_cond is optional in this path. Only build image_cond_in
                            # when partial_img_emb is available.
                            if partial_img_emb is not None:
                                # image_cond structure: [image_cond_mask (4) + image_embeds (32)] = 36 channels
                                # Replace the first 4 channels of image_cond_mask with window_msk
                                image_cond_in = partial_img_emb.clone()
                                image_cond_in[:4] = window_msk  # Replace the first 4 channels
                            else:
                                image_cond_in = None
                            # ========================================
                            original_seq_len = seq_len
                            prepend_seq_latents = 0
                            if context_latent_offset > 0:
                                prepend_seq_latents += context_latent_offset
                            if has_prefix and c[0] != 0 and prefix_prepend_latents > 0:
                                prepend_seq_latents += prefix_prepend_latents
                            if scail_window_prepend_latents > 0:
                                prepend_seq_latents += scail_window_prepend_latents
                            if prepend_seq_latents > 0:
                                seq_len = original_seq_len + prepend_seq_latents * base_patches_per_frame
                            # Slice context_latents for current context window
                            sliced_context_latents = None
                            if context_latents is not None:
                                sliced_context_latents = []
                                for lat_idx, lat in enumerate(context_latents):
                                    context_role = (
                                        context_roles[lat_idx]
                                        if context_roles is not None and lat_idx < len(context_roles)
                                        else None
                                    )
                                    if lat.shape[1] > 1 and lat.shape[1] == noise.shape[1]:
                                        if context_latent_offset > 0:
                                            sliced_context_latents.append(torch.cat([
                                                lat[:, :context_latent_offset],
                                                lat[:, target_c],
                                            ], dim=1).to(device))
                                        else:
                                            sliced_context_latents.append(lat[:, c].to(device))
                                    elif (
                                        context_latent_offset > 0
                                        and context_role == "source_video"
                                        and lat.shape[1] >= (max(c) + 1 if c else 0)
                                    ):
                                        sliced_context_latents.append(lat[:, c].to(device))
                                    else:
                                        sliced_context_latents.append(lat.to(device))
                            # Context latents are already sliced to this local window. The target
                            # window RoPE also starts at 0, so keep Bernini context RoPE local.
                            noise_pred_context, _, new_teacache = predict_with_cfg(
                                partial_latent_model_input,
                                cfg[idx], positive,
                                text_embeds["negative_prompt_embeds"],
                                partial_timestep, idx, image_cond_in, clip_fea, partial_control_latents, partial_vace_context, partial_unianim_data,partial_audio_proj,
                                partial_control_camera_latents, partial_add_cond, current_teacache, context_window=c, fantasy_portrait_input=partial_fantasy_portrait_input,
                                mtv_motion_tokens=partial_mtv_motion_tokens, s2v_audio_input=partial_s2v_audio_input, s2v_motion_frames=[1, 0], s2v_pose=partial_s2v_pose,
                                humo_image_cond=humo_image_cond, humo_image_cond_neg=humo_image_cond_neg, humo_audio=humo_audio, humo_audio_neg=humo_audio_neg,
                                wananim_face_pixels=partial_wananim_face_pixels, wananim_pose_latents=partial_wananim_pose_latents,
                                wananim_num_anchor_latents=wananim_num_anchor_latents, multitalk_audio_embeds=multitalk_audio_embeds,
                                uni3c_data=uni3c_data, flashvsr_LQ_latent=partial_flashvsr_LQ_latent, context_latents=sliced_context_latents,
                                context_roles=context_roles, context_window_start=0, scail_context_prepend_latents=scail_window_prepend_latents)

                            seq_len = original_seq_len  # restore seq_len

                            # ============ Prefix: slice off prepended predictions ============
                            if has_prefix and c[0] != 0 and prefix_prepend_latents > 0:
                                noise_pred_context = noise_pred_context[:, prefix_prepend_latents:]
                            # ==============================================================
                            if scail_window_prepend_latents > 0:
                                noise_pred_context = noise_pred_context[:, scail_window_prepend_latents:]
                            if context_latent_offset > 0:
                                noise_pred_context = noise_pred_context[:, context_latent_offset:]

                            if cache_args is not None:
                                self.window_tracker.cache_states[window_id] = new_teacache

                            if mocha_embeds is not None:
                                noise_pred_context = noise_pred_context[:, :orig_model_input_frames]

                            window_mask = create_window_mask(noise_pred_context, c, context_target_length, context_overlap, looped=is_looped, window_type=context_options["fuse_method"])
                            write_c = target_c if context_latent_offset > 0 else c
                            noise_pred[:, write_c] += noise_pred_context * window_mask
                            counter[:, write_c] += window_mask
                            context_pbar.update_absolute(step_start_progress + (i + 1) * fraction_per_context, len(timesteps))
                        if context_latent_offset > 0:
                            noise_pred[:, :context_latent_offset] = 0
                            counter[:, :context_latent_offset] = 1
                        noise_pred /= counter
                    #region multitalk
                    elif multitalk_sampling:
                        return multitalk_loop(**locals())
                    # region framepack loop
                    elif framepack:
                        framepack_out = []
                        ref_motion_image = None
                        #infer_frames = image_embeds["num_frames"]
                        infer_frames = s2v_audio_embeds.get("frame_window_size", 80)
                        motion_frames = infer_frames - 7 #73 default
                        lat_motion_frames = (motion_frames + 3) // 4
                        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames

                        step_iteration_count = 0
                        total_frames = s2v_audio_input.shape[-1]

                        s2v_motion_frames = [motion_frames, lat_motion_frames]

                        noise = torch.randn( #C, T, H, W
                            48 if is_5b else 16,
                                lat_target_frames,
                                target_shape[2],
                                target_shape[3],
                                dtype=torch.float32,
                                generator=seed_g,
                                device=torch.device("cpu"))

                        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

                        if ref_motion_image is None:
                            ref_motion_image = torch.zeros(
                                [1, 3, motion_frames, latent.shape[2]*vae_upscale_factor, latent.shape[3]*vae_upscale_factor],
                                dtype=vae.dtype,
                                device=device)
                        videos_last_frames = ref_motion_image

                        if s2v_pose is not None:
                            pose_cond_list = []
                            for r in range(s2v_num_repeat):
                                pose_start = r * (infer_frames // 4)
                                pose_end = pose_start + (infer_frames // 4)

                                cond_lat = s2v_pose[:, :, pose_start:pose_end]

                                pad_len = (infer_frames // 4) - cond_lat.shape[2]
                                if pad_len > 0:
                                    pad = -torch.ones(cond_lat.shape[0], cond_lat.shape[1], pad_len, cond_lat.shape[3], cond_lat.shape[4], device=cond_lat.device, dtype=cond_lat.dtype)
                                    cond_lat = torch.cat([cond_lat, pad], dim=2)
                                pose_cond_list.append(cond_lat.cpu())

                        log.info(f"Sampling {total_frames} frames in {s2v_num_repeat} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")
                        # sample
                        for r in range(s2v_num_repeat):

                            mm.soft_empty_cache()
                            gc.collect()
                            if ref_motion_image is not None:
                                vae.to(device)
                                ref_motion = vae.encode(ref_motion_image.to(vae.dtype), device=device, pbar=False).to(dtype)[0]

                                vae.to(offload_device)

                            left_idx = r * infer_frames
                            right_idx = r * infer_frames + infer_frames

                            s2v_audio_input_slice = s2v_audio_input[..., left_idx:right_idx]
                            if s2v_audio_input_slice.shape[-1] < (right_idx - left_idx):
                                pad_len = (right_idx - left_idx) - s2v_audio_input_slice.shape[-1]
                                pad_shape = list(s2v_audio_input_slice.shape)
                                pad_shape[-1] = pad_len
                                pad = torch.zeros(pad_shape, device=s2v_audio_input_slice.device, dtype=s2v_audio_input_slice.dtype)
                                log.info(f"Padding s2v_audio_input_slice from {s2v_audio_input_slice.shape[-1]} to {right_idx - left_idx}")
                                s2v_audio_input_slice = torch.cat([s2v_audio_input_slice, pad], dim=-1)

                            if ref_motion_image is not None:
                                input_motion_latents = ref_motion.clone().unsqueeze(0)
                            else:
                                input_motion_latents = None

                            s2v_pose_slice = None
                            if s2v_pose is not None:
                                s2v_pose_slice = pose_cond_list[r].to(device)

                            if isinstance(scheduler, dict):
                                sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                timesteps = scheduler["timesteps"]
                            else:
                                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, denoise_strength, sigmas=sigmas)

                            latent = noise.to(device)
                            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling audio indices {left_idx}-{right_idx}", position=0)):
                                latent_model_input = latent.to(device)
                                timestep = torch.tensor([t]).to(device)
                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input,
                                    cfg[idx],
                                    text_embeds["prompt_embeds"],
                                    text_embeds["negative_prompt_embeds"],
                                    timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                                    cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens,
                                    context_latents=context_latents, context_roles=context_roles,
                                    s2v_audio_input=s2v_audio_input_slice, s2v_ref_motion=input_motion_latents, s2v_motion_frames=s2v_motion_frames, s2v_pose=s2v_pose_slice)

                                latent = sample_scheduler.step(
                                        noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0),
                                        **scheduler_step_args)[0].squeeze(0).detach()
                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, s2v_num_repeat*(len(timesteps)))
                                    del callback_latent
                                step_iteration_count += 1
                                del latent_model_input, noise_pred


                            vae.to(device)
                            decode_latents = torch.cat([ref_motion.unsqueeze(0), latent.unsqueeze(0)], dim=2)
                            image = vae.decode(decode_latents.to(device, vae.dtype), device=device, pbar=False)[0]
                            del decode_latents
                            image = image.unsqueeze(0)[:, :, -infer_frames:]
                            if r == 0:
                                image = image[:, :, 3:]

                            framepack_out.append(image.cpu())

                            overlap_frames_num = min(motion_frames, image.shape[2])

                            videos_last_frames = torch.cat([
                                videos_last_frames[:, :, overlap_frames_num:],
                                image[:, :, -overlap_frames_num:]], dim=2).to(device, vae.dtype)

                            ref_motion_image = videos_last_frames

                        vae.to(offload_device)

                        mm.soft_empty_cache()
                        gen_video_samples = torch.cat(framepack_out, dim=2).squeeze(0).permute(1, 2, 3, 0)

                        if force_offload:
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except Exception:
                            pass
                        return {"video": gen_video_samples},
                    # region wananimate loop
                    elif wananimate_loop:
                        # calculate frame counts
                        total_frames = num_frames
                        refert_num = 1

                        real_clip_len = frame_window_size - refert_num
                        last_clip_num = (total_frames - refert_num) % real_clip_len
                        extra = 0 if last_clip_num == 0 else real_clip_len - last_clip_num
                        target_len = total_frames + extra
                        estimated_iterations = target_len // real_clip_len
                        target_latent_len = (target_len - 1) // 4 + estimated_iterations
                        latent_window_size = (frame_window_size - 1) // 4 + 1

                        from .utils import tensor_pingpong_pad

                        ref_latent = image_embeds.get("ref_latent", None)
                        ref_images = image_embeds.get("ref_image", None)
                        bg_images = image_embeds.get("bg_images", None)
                        pose_images = image_embeds.get("pose_images", None)
                        prefix_ctx = image_embeds.get("prefix_ctx", None)
                        prefix_T = image_embeds.get("prefix_T", 0)

                        current_ref_images = image_embeds.get("start_ref_image", None)
                        if current_ref_images is not None:
                            log.info(
                                "WanAnimate: Detected manual start reference image, enabling continuous generation across windows.")
                        face_images = face_images_in = None

                        if wananim_face_pixels is not None:
                            face_images = tensor_pingpong_pad(wananim_face_pixels, target_len)
                            log.info(f"WanAnimate: Face input {wananim_face_pixels.shape} padded to shape {face_images.shape}")
                        if wananim_ref_masks is not None:
                            ref_masks_in = tensor_pingpong_pad(wananim_ref_masks, target_latent_len)
                            log.info(f"WanAnimate: Ref masks {wananim_ref_masks.shape} padded to shape {ref_masks_in.shape}")
                        if bg_images is not None:
                            bg_images_in = tensor_pingpong_pad(bg_images, target_len)
                            log.info(f"WanAnimate: BG images {bg_images.shape} padded to shape {bg_images.shape}")
                        if pose_images is not None:
                            pose_images_in = tensor_pingpong_pad(pose_images, target_len)
                            log.info(f"WanAnimate: Pose images {pose_images.shape} padded to shape {pose_images_in.shape}")

                        # init variables
                        offloaded = False

                        colormatch = image_embeds.get("colormatch", "disabled")
                        output_path = image_embeds.get("output_path", "")
                        offload = image_embeds.get("force_offload", False)

                        lat_h, lat_w = noise.shape[2], noise.shape[3]
                        start = start_latent = img_counter = step_iteration_count = iteration_count = 0
                        end = frame_window_size
                        end_latent = latent_window_size


                        callback = prepare_callback(patcher, estimated_iterations)
                        log.info(f"Sampling {total_frames} frames in {estimated_iterations} windows, at {latent.shape[3]*vae_upscale_factor}x{latent.shape[2]*vae_upscale_factor} with {steps} steps")

                        # outer WanAnimate loop
                        gen_video_list = []
                        while True:
                            if start + refert_num >= total_frames:
                                break

                            mm.soft_empty_cache()

                            if current_ref_images is not None:
                                mask_reft_len = refert_num
                            else:
                                mask_reft_len = 0 if start == 0 else refert_num

                            self.cache_state = [None, None]

                            loop_ref_latents = wananim_static_ref_latents if wananim_static_ref_latents > 0 else 1
                            loop_extra_prefix_latents = 0 if wananim_static_ref_latents > 0 else prefix_T
                            loop_target_latents = latent_window_size + loop_ref_latents
                            noise = torch.randn(16, latent_window_size + loop_ref_latents + loop_extra_prefix_latents, lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)
                            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

                            def _trim_loop_prefix_latents(tensor):
                                if tensor is not None and loop_extra_prefix_latents > 0 and tensor.shape[1] > loop_target_latents:
                                    return tensor[:, -loop_target_latents:]
                                return tensor

                            if current_ref_images is not None or bg_images is not None or ref_latent is not None or has_transition or prefix_ctx is not None:
                                if offload:
                                    offload_transformer(transformer, remove_lora=False)
                                    offloaded = True
                                vae.to(device)
                                if wananim_ref_masks is not None:
                                    msk = ref_masks_in[:, start_latent:end_latent].to(device, dtype)
                                else:
                                    msk = torch.zeros(4, latent_window_size, lat_h, lat_w, device=device, dtype=dtype)
                                if bg_images is not None:
                                    bg_image_slice = bg_images_in[:, start:end].to(device)
                                else:
                                    bg_image_slice = torch.zeros(3, frame_window_size-refert_num, lat_h * 8, lat_w * 8, device=device, dtype=vae.dtype)
                                if mask_reft_len == 0 and not has_transition:
                                    temporal_ref_latents = vae.encode([bg_image_slice], device,tiled=tiled_vae)[0]
                                else:
                                    # Build concatenated image
                                    concat_parts = []
                                    
                                    # 1. Add start_ref_image (if any)
                                    if current_ref_images is not None:
                                        concat_parts.append(current_ref_images.to(device, dtype=vae.dtype))
                                    
                                    # 2. Add the remaining part of bg_image_slice
                                    bg_start_idx = mask_reft_len if current_ref_images is not None else 0
                                    if bg_image_slice.shape[1] > bg_start_idx:
                                        concat_parts.append(bg_image_slice[:, bg_start_idx:])
                                    
                                    # 3. VAE encode the image part
                                    if len(concat_parts) > 0:
                                        image_concat = torch.cat(concat_parts, dim=1)
                                        temporal_ref_latents = vae.encode([image_concat.to(device, vae.dtype)], device, tiled=tiled_vae, pbar=False)[0]
                                    else:
                                        temporal_ref_latents = None

                                    # 4. Initialize mask
                                    msk = torch.zeros(4, latent_window_size, lat_h, lat_w, device=device, dtype=dtype)
                                    
                                    # 4a. Hard mask for start_ref_image
                                    if current_ref_images is not None:
                                        msk[:, :mask_reft_len] = 1
                                    
                                    # 4b. Hard mask for transition_latent (replaces first 8 target latent frames ON THE FIRST CHUNK ONLY)
                                    if has_transition and start == 0:  
                                        if temporal_ref_latents is not None:
                                            # ============ Mod: Replace first transition_len frames of target latent ============
                                            # Keep target latent after transition_len
                                            remaining_target = temporal_ref_latents[:, transition_len:]
                                            # transition_latent replaces the first transition_len target latent frames
                                            temporal_ref_latents = torch.cat([transition_latent.to(device, dtype=transition_latent.dtype), remaining_target], dim=1)
                                            log.info(f"Replaced first {transition_len} target latent frames with transition_video in Chunk 0")
                                            # ========================================================================
                                        else:
                                            # Only transition_latent, no image part
                                            temporal_ref_latents = transition_latent.to(device, dtype=dtype)
                                        
                                        # Set hard mask. 
                                        transition_start = 0 
                                        msk[:, transition_start:transition_start+transition_len] = transition_mask_values.to(device).view(1, -1, 1, 1)
                                    
                                # [FIX] Unified shape alignment for temporal_ref_latents and msk
                                # Handles VAE time downsampling offset: (T-1)//4 + 1 vs latent_window_size
                                # Moved outside branches to ensure execution for both transition and non-transition modes
                                if temporal_ref_latents is not None and msk.shape[1] != temporal_ref_latents.shape[1]:
                                    if temporal_ref_latents.shape[1] < msk.shape[1]:
                                        pad_len = msk.shape[1] - temporal_ref_latents.shape[1]
                                        pad_tensor = temporal_ref_latents[:, -1:].repeat(1, pad_len, 1, 1)
                                        temporal_ref_latents = torch.cat([temporal_ref_latents, pad_tensor], dim=1)
                                    else:
                                        temporal_ref_latents = temporal_ref_latents[:, :msk.shape[1]]

                                if ref_latent is not None or prefix_ctx is not None:
                                    ref_part = prefix_ctx if prefix_ctx is not None else ref_latent
                                    temporal_ref_latents = torch.cat([msk, temporal_ref_latents], dim=0) # 4+C T H W
                                    image_cond_in = torch.cat([ref_part.to(device), temporal_ref_latents], dim=1) # 4+C T+trefs H W
                                    del temporal_ref_latents, msk, bg_image_slice
                                else:
                                    image_cond_in = torch.cat([torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1]), torch.zeros_like(noise)], dim=0).to(device)
                            else:
                                image_cond_in = torch.cat([torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1]), torch.zeros_like(noise)], dim=0).to(device)

                            pose_input_slice = None
                            if pose_images is not None:
                                vae.to(device)
                                pose_image_slice = pose_images_in[:, start:end].to(device)
                                pose_input_slice = vae.encode([pose_image_slice], device,tiled=tiled_vae, pbar=False).to(dtype)

                            vae.to(offload_device)

                            if wananim_face_pixels is None and wananim_ref_masks is not None:
                                face_images_in = torch.zeros(1, 3, frame_window_size, 512, 512, device=device, dtype=torch.float32)
                            elif wananim_face_pixels is not None:
                                face_images_in = face_images[:, :, start:end].to(device, torch.float32) if face_images is not None else None

                            # ============ Prefix prepend for looping: copy first frame of current chunk ============
                            if loop_extra_prefix_latents > 0:
                                if pose_input_slice is not None:
                                    first_pose = pose_input_slice[:, :, :1].repeat(1, 1, loop_extra_prefix_latents, 1, 1)
                                    pose_input_slice = torch.cat([first_pose, pose_input_slice], dim=2)
                                if face_images_in is not None:
                                    first_face = face_images_in[:, :, :1].repeat(1, 1, loop_extra_prefix_latents * 4, 1, 1)
                                    face_images_in = torch.cat([first_face, face_images_in], dim=2)
                            # ========================================================================================

                            if samples is not None:
                                input_samples = samples["samples"]
                                if input_samples is not None:
                                    input_samples = input_samples.squeeze(0).to(noise)
                                    # Check if we have enough frames in input_samples
                                    # if latent_end_idx > input_samples.shape[1]:
                                    #     # We need more frames than available - pad the input_samples at the end
                                    #     pad_length = latent_end_idx - input_samples.shape[1]
                                    #     last_frame = input_samples[:, -1:].repeat(1, pad_length, 1, 1)
                                    #     input_samples = torch.cat([input_samples, last_frame], dim=1)
                                    # input_samples = input_samples[:, latent_start_idx:latent_end_idx]
                                    if noise_mask is not None:
                                        original_image = input_samples.to(device)

                                    assert input_samples.shape[1] == noise.shape[1], f"Slice mismatch: {input_samples.shape[1]} vs {noise.shape[1]}"

                                    if add_noise_to_samples:
                                        latent_timestep = timesteps[0]
                                        noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples
                                    else:
                                        noise = input_samples

                                # diff diff prep
                                noise_mask = samples.get("noise_mask", None)
                                if noise_mask is not None:
                                    if len(noise_mask.shape) == 4:
                                        noise_mask = noise_mask.squeeze(1)
                                    if noise_mask.shape[0] < noise.shape[1]:
                                        noise_mask = noise_mask.repeat(noise.shape[1] // noise_mask.shape[0], 1, 1)
                                    else:
                                        noise_mask = noise_mask[start_latent:end_latent]
                                    noise_mask = torch.nn.functional.interpolate(
                                        noise_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                                        size=(noise.shape[1], noise.shape[2], noise.shape[3]),
                                        mode='trilinear',
                                        align_corners=False
                                    ).repeat(1, noise.shape[0], 1, 1, 1)

                                    thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
                                    thresholds = thresholds.reshape(-1, 1, 1, 1, 1).to(device)
                                    noise_mask = noise_mask.repeat(len(timesteps), 1, 1, 1, 1).to(device=device, dtype=thresholds.dtype)
                                    masks = (1.0 - noise_mask) > thresholds

                            if isinstance(scheduler, dict):
                                sample_scheduler = copy.deepcopy(scheduler["sample_scheduler"])
                                timesteps = scheduler["timesteps"]
                            else:
                                sample_scheduler, timesteps,_,_ = get_scheduler(scheduler, total_steps, start_step, end_step, shift, device, transformer.dim, denoise_strength, sigmas=sigmas)

                            # sample videos
                            latent = noise

                            if offloaded:
                                # Load weights
                                if transformer.patched_linear and gguf_reader is None:
                                    load_weights(patcher.model.diffusion_model, patcher.model["sd"], weight_dtype, base_dtype=dtype, transformer_load_device=device, block_swap_args=block_swap_args)
                                elif gguf_reader is not None: #handle GGUF
                                    load_weights(transformer, patcher.model["sd"], base_dtype=dtype, transformer_load_device=device, patcher=patcher, gguf=True, reader=gguf_reader, block_swap_args=block_swap_args)
                                #blockswap init
                                init_blockswap(transformer, block_swap_args, model)

                            # Use the appropriate prompt for this section
                            if len(text_embeds["prompt_embeds"]) > 1:
                                prompt_index = min(iteration_count, len(text_embeds["prompt_embeds"]) - 1)
                                positive = [text_embeds["prompt_embeds"][prompt_index]]
                                log.info(f"Using prompt index: {prompt_index}")
                            else:
                                positive = text_embeds["prompt_embeds"]

                            # uni3c slices
                            uni3c_data_input = None
                            if uni3c_embeds is not None:
                                render_latent = uni3c_embeds["render_latent"][:,:,start_latent:end_latent].to(device)
                                if render_latent.shape[2] < noise.shape[1]:
                                    render_latent = torch.nn.functional.interpolate(render_latent, size=(noise.shape[1], noise.shape[2], noise.shape[3]), mode='trilinear', align_corners=False)
                                uni3c_data_input = {"render_latent": render_latent}
                                for k in uni3c_data:
                                    if k != "render_latent":
                                        uni3c_data_input[k] = uni3c_data[k]

                            mm.soft_empty_cache()
                            gc.collect()
                            # inner WanAnimate sampling loop
                            sampling_pbar = tqdm(total=len(timesteps), desc=f"Frames {start}-{end}", position=0, leave=True)
                            for i in range(len(timesteps)):
                                timestep = timesteps[i]
                                latent_model_input = latent.to(device)

                                noise_pred, _, self.cache_state = predict_with_cfg(
                                    latent_model_input, cfg[min(i, len(timesteps)-1)], positive, text_embeds["negative_prompt_embeds"],
                                    timestep, i, cache_state=self.cache_state, image_cond=image_cond_in, clip_fea=clip_fea, wananim_face_pixels=face_images_in,
                                    wananim_pose_latents=pose_input_slice, wananim_num_anchor_latents=wananim_num_anchor_latents, uni3c_data=uni3c_data_input,
                                    context_latents=context_latents, context_roles=context_roles,
                                 )
                                if loop_extra_prefix_latents > 0:
                                    noise_pred = _trim_loop_prefix_latents(noise_pred)
                                    latent = _trim_loop_prefix_latents(latent)
                                    latent_model_input = _trim_loop_prefix_latents(latent_model_input)
                                if callback is not None:
                                    callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                    callback(step_iteration_count, callback_latent, None, estimated_iterations*(len(timesteps)))
                                    del callback_latent

                                sampling_pbar.update(1)
                                step_iteration_count += 1

                                if use_tsr:
                                    noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)

                                latent = sample_scheduler.step(noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0).to(noise_pred.device), **scheduler_step_args)[0].squeeze(0).detach()
                                del noise_pred, latent_model_input, timestep

                                # differential diffusion inpaint
                                if masks is not None:
                                    if i < len(timesteps) - 1:
                                        image_latent = add_noise(original_image.to(device), noise.to(device), timesteps[i+1])
                                        mask = masks[i].to(device=latent.device, dtype=latent.dtype)
                                        latent = image_latent * mask + latent * (1.0 - mask)

                            del noise
                            if offload:
                                offload_transformer(transformer, remove_lora=False)
                                offloaded = True

                            vae.to(device)
                            decode_skip = loop_ref_latents
                            videos = vae.decode(latent[:, decode_skip:].unsqueeze(0).to(device, vae.dtype), device=device, tiled=tiled_vae, pbar=False)[0].cpu()
                            del latent

                            if start != 0 or current_ref_images is not None:
                                videos = videos[:, refert_num:]

                            sampling_pbar.close()

                            # optional color correction
                            if colormatch != "disabled":
                                videos = videos.permute(1, 2, 3, 0).float().numpy()
                                from color_matcher import ColorMatcher
                                cm = ColorMatcher()
                                cm_result_list = []
                                for img in videos:
                                    cm_result = cm.transfer(src=img, ref=ref_images.permute(1, 2, 3, 0).squeeze(0).cpu().float().numpy(), method=colormatch)
                                    cm_result_list.append(torch.from_numpy(cm_result).to(vae.dtype))
                                videos = torch.stack(cm_result_list, dim=0).permute(3, 0, 1, 2)
                                del cm_result_list

                            current_ref_images = videos[:, -refert_num:].clone().detach()

                            # optionally save generated samples to disk
                            # if output_path:
                            #     video_np = videos.clamp(-1.0, 1.0).add(1.0).div(2.0).mul(255).cpu().float().numpy().transpose(1, 2, 3, 0).astype('uint8')
                            #     num_frames_to_save = video_np.shape[0] if is_first_clip else video_np.shape[0] - cur_motion_frames_num
                            #     log.info(f"Saving {num_frames_to_save} generated frames to {output_path}")
                            #     start_idx = 0 if is_first_clip else cur_motion_frames_num
                            #     for i in range(start_idx, video_np.shape[0]):
                            #         im = Image.fromarray(video_np[i])
                            #         im.save(os.path.join(output_path, f"frame_{img_counter:05d}.png"))
                            #         img_counter += 1
                            # else:
                            gen_video_list.append(videos)

                            del videos

                            iteration_count += 1
                            start += frame_window_size - refert_num
                            end += frame_window_size - refert_num
                            start_latent += latent_window_size - ((refert_num - 1)// 4 + 1)
                            end_latent += latent_window_size - ((refert_num - 1)// 4 + 1)

                        if not output_path:
                            gen_video_samples = torch.cat(gen_video_list, dim=1)
                            if canvas_expansion_px:
                                if gen_video_samples.shape[1] > canvas_expansion_px:
                                    gen_video_samples = gen_video_samples[:, canvas_expansion_px:]
                                    log.info(f"WanAnimate loop: trimmed {canvas_expansion_px} canvas expansion pixel frames from output")
                                else:
                                    log.warning(
                                        f"WanAnimate loop: canvas_expansion_px={canvas_expansion_px} is not smaller than output length {gen_video_samples.shape[1]}, skipping trim"
                                    )
                        else:
                            gen_video_samples = torch.zeros(3, 1, 64, 64) # dummy output

                        if force_offload:
                            vae.to(offload_device)
                            if not model["auto_cpu_offload"]:
                                offload_transformer(transformer)
                        try:
                            print_memory(device)
                            torch.cuda.reset_peak_memory_stats(device)
                        except Exception:
                            pass
                        return {"video": gen_video_samples.permute(1, 2, 3, 0), "output_path": output_path},

                    #region normal inference
                    else:
                        noise_pred, noise_pred_ovi, self.cache_state = predict_with_cfg(
                            latent_model_input,
                            cfg[idx], text_embeds["prompt_embeds"], text_embeds["negative_prompt_embeds"],
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, multitalk_audio_embeds=multitalk_audio_embeds, mtv_motion_tokens=mtv_motion_tokens, s2v_audio_input=s2v_audio_input,
                            humo_image_cond=humo_image_cond, humo_image_cond_neg=humo_image_cond_neg, humo_audio=humo_audio, humo_audio_neg=humo_audio_neg,
                            wananim_face_pixels=wananim_face_pixels, wananim_pose_latents=wananim_pose_latents,
                            wananim_num_anchor_latents=wananim_num_anchor_latents, uni3c_data = uni3c_data, latent_model_input_ovi=latent_model_input_ovi, flashvsr_LQ_latent=flashvsr_LQ_latent,
                            context_latents=context_latents, context_roles=context_roles,
                        )
                        if bidirectional_sampling:
                            noise_pred_flipped, _,self.cache_state = predict_with_cfg(
                            latent_model_input_flipped,
                            cfg[idx], text_embeds["prompt_embeds"], text_embeds["negative_prompt_embeds"],
                            timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                            cache_state=self.cache_state, fantasy_portrait_input=fantasy_portrait_input, mtv_motion_tokens=mtv_motion_tokens,reverse_time=True,
                            wananim_face_pixels=wananim_face_pixels, wananim_pose_latents=wananim_pose_latents,
                            wananim_num_anchor_latents=wananim_num_anchor_latents,
                            context_latents=context_latents, context_roles=context_roles,)

                    if latent_shift_loop:
                        #reverse latent shift
                        if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                            noise_pred = torch.cat([noise_pred[:, latent_video_length - shift_idx:]] + [noise_pred[:, :latent_video_length - shift_idx]], dim=1)
                            shift_idx = (shift_idx + latent_skip) % latent_video_length

                    latent = latent.to(device)

                    if self.noise_front_pad_num > 0:
                        noise_pred = noise_pred[:, self.noise_front_pad_num:]

                    if use_tsr:
                        noise_pred = temporal_score_rescaling(noise_pred, latent, timestep, tsr_k, tsr_sigma)

                    if transformer.is_longcat:
                        noise_pred = -noise_pred

                    if len(timestep.shape) != 1 and clean_latent_indices and not is_pusa: #5b and longcat, skip clean latents for scheduler step
                        step_process_indices = [i for i in range(latent.shape[1]) if i not in clean_latent_indices]
                        latent[:, step_process_indices] = sample_scheduler.step(noise_pred[:, step_process_indices].unsqueeze(0), orig_timestep,
                                                        latent[:, step_process_indices].unsqueeze(0), **scheduler_step_args)[0].squeeze(0).detach()
                    else:
                        if latents_to_not_step > 0:
                            raw_latent = latent[:, :latents_to_not_step]
                            noise_pred_in = noise_pred[:, latents_to_not_step:]
                            latent = latent[:, latents_to_not_step:]
                        elif recammaster is not None or mocha_embeds is not None:
                            noise_pred_in = noise_pred[:, :orig_noise_len]
                            latent = latent[:, :orig_noise_len]
                        else:
                            noise_pred_in = noise_pred
                        latent = sample_scheduler.step(noise_pred_in.unsqueeze(0), timestep, latent.unsqueeze(0), **scheduler_step_args)[0].squeeze(0).detach()
                        if noise_pred_flipped is not None:
                            latent_backwards = sample_scheduler_flipped.step(noise_pred_flipped.unsqueeze(0), timestep, latent_flipped.unsqueeze(0), **scheduler_step_args)[0].squeeze(0).detach()
                            latent_backwards = torch.flip(latent_backwards, dims=[1])
                            latent = latent * 0.5 + latent_backwards * 0.5
                        if latents_to_not_step > 0:
                            latent = torch.cat([raw_latent, latent], dim=1)

                    latent = latent.detach()

                    if latent_ovi is not None:
                        latent_ovi = sample_scheduler_ovi.step(noise_pred_ovi.unsqueeze(0), t, latent_ovi.to(device).unsqueeze(0), **scheduler_step_args)[0].squeeze(0).detach()

                    #InfiniteTalk first frame handling
                    if (extra_latents is not None
                        and not multitalk_sampling
                        and transformer.multitalk_model_type=="InfiniteTalk"):
                        for entry in extra_latents:
                            add_index = entry["index"]
                            num_extra_frames = entry["samples"].shape[2]
                            latent[:, add_index:add_index+num_extra_frames] = entry["samples"].to(latent)

                    # differential diffusion inpaint
                    if scail_context_freeze_direct_mask:
                        image_latent = original_image.to(device)
                        mask = (scail_freeze_mask[0] > 0.5).to(device=latent.device, dtype=latent.dtype)
                        latent = image_latent * mask + latent * (1.0 - mask)
                    elif masks is not None:
                        image_latent = None
                        if scail_freeze_mask is not None:
                            image_latent = original_image.to(device)
                        elif idx < len(timesteps) - 1:
                            noise_timestep = timesteps[idx+1]
                            image_latent = sample_scheduler.scale_noise(
                                original_image.to(device), torch.tensor([noise_timestep]), noise.to(device)
                            )
                        if image_latent is not None:
                            mask = masks[idx].to(device=latent.device, dtype=latent.dtype)
                            latent = image_latent * mask + latent * (1.0 - mask)

                    # TTM
                    if ttm_reference_latents is not None and (idx + ttm_start_step) < ttm_end_step:
                        ttm_motion_mask = motion_mask.to(device=latent.device, dtype=latent.dtype)
                        if idx + ttm_start_step + 1 < len(sample_scheduler.all_timesteps):
                            noisy_latents = add_noise(ttm_reference_latents, noise, sample_scheduler.all_timesteps[idx + ttm_start_step + 1].to(noise.device)).to(latent)
                            latent = latent * (1.0 - ttm_motion_mask) + noisy_latents * ttm_motion_mask
                        else:
                            latent = latent * (1.0 - ttm_motion_mask) + ttm_reference_latents.to(latent) * ttm_motion_mask

                    if freeinit_args is not None:
                        current_latent = latent.clone()

                    if callback is not None:
                        if recammaster is not None or mocha_embeds is not None:
                            callback_latent = (latent_model_input[:, :orig_noise_len].to(device) - noise_pred[:, :orig_noise_len].to(device) * t.to(device) / 1000).detach()
                        #elif phantom_latents is not None:
                        #    callback_latent = (latent_model_input[:,:-phantom_latents.shape[1]].to(device) - noise_pred[:,:-phantom_latents.shape[1]].to(device) * t.to(device) / 1000).detach()
                        elif humo_reference_count > 0:
                            callback_latent = (latent_model_input[:,:-humo_reference_count].to(device) - noise_pred[:,:-humo_reference_count].to(device) * t.to(device) / 1000).detach()
                        elif "rcm" in sample_scheduler.__class__.__name__.lower():
                            callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device)).detach()
                        else:
                            callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach()
                        callback(idx, callback_latent.permute(1,0,2,3), None, len(timesteps))
                    else:
                        pbar.update(1)

                    # Release per-step tensor references after all scheduler/callback work is done.
                    noise_pred = noise_pred_in = noise_pred_ovi = None
                    noise_pred_flipped = None
                    latent_model_input = latent_model_input_ovi = None
                    latent_model_input_flipped = None
                    latent_flipped = latent_backwards = raw_latent = None
                    timestep = orig_timestep = None
                    noise_for_cond = timestep_cond = None
                    counter = partial_latent_model_input = prefix_noise = None
                    partial_img_emb = partial_control_latents = partial_control_camera_latents = None
                    partial_vace_context = partial_audio_proj = partial_s2v_audio_input = None
                    partial_s2v_pose = partial_add_cond = None
                    partial_wananim_face_pixels = partial_wananim_pose_latents = None

            except Exception as e:
                log.error(f"Error during sampling: {e}")
                raise
            finally:
                if force_offload and not model["auto_cpu_offload"]:
                    offload_transformer(transformer)

        if phantom_latents is not None:
            latent = latent[:,:-phantom_latents.shape[1]]
        if humo_reference_count > 0:
            latent = latent[:,:-humo_reference_count]
        if longcat_ref_latent is not None:
            latent = latent[:, longcat_ref_latent.shape[1]:]
        if story_mem_latents is not None:
            latent = latent[:, story_mem_latents.shape[1]:]

        log.info("-" * 10 + " Sampling end " + "-" * 12)

        cache_states = None
        if cache_args is not None:
            cache_report(transformer, cache_args)
            if end_step != -1 and end_step < total_steps:
                cache_states = {
                    "cache_state": self.cache_state,
                    "easycache_state": transformer.easycache_state,
                    "teacache_state": transformer.teacache_state,
                    "magcache_state": transformer.magcache_state,
                }

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        samples_out = {
            "samples": latent.unsqueeze(0).cpu(),
            "looped": is_looped,
            "end_image": end_image if not fun_or_fl2v_model else None,
            "has_ref": has_ref,
            "has_prefix": has_prefix,
            "canvas_expansion_px": canvas_expansion_px,
            "drop_last": drop_last,
            "generator_state": seed_g.get_state(),
            "original_image": original_image.cpu() if original_image is not None else None,
            "cache_states": cache_states,
            "latent_ovi_audio": latent_ovi.unsqueeze(0).transpose(1, 2).cpu() if latent_ovi is not None else None,
            "flashvsr_LQ_images": LQ_images,
        }
        if wananim_decode_ref_latents > 0:
            samples_out["wananim_decode_ref_latents"] = wananim_decode_ref_latents
        return (samples_out,{
            "samples": callback_latent.unsqueeze(0).cpu() if callback is not None else None,
        })

class WanVideoSamplerSettings(WanVideoSampler):
    RETURN_TYPES = ("SAMPLER_ARGS",)
    RETURN_NAMES = ("sampler_inputs", )
    DESCRIPTION = "Node to output all settings and inputs for the WanVideoSamplerFromSettings -node"
    def process(self, *args, **kwargs):
        import inspect
        params = inspect.signature(WanVideoSampler.process).parameters
        args_dict = {name: kwargs.get(name, param.default if param.default is not inspect.Parameter.empty else None)
                     for name, param in params.items() if name != "self"}
        return args_dict,

class WanVideoSamplerFromSettings(WanVideoSampler):
    DESCRIPTION = "Utility node with no other functionality than to look cleaner, useful for the live preview as the main sampler node has become a messy monster"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_inputs": ("SAMPLER_ARGS",),},
        }

    def process(self, sampler_inputs):
        return super().process(**sampler_inputs)


def _wananimateplus_easy_sampler_input_types():
    return {
        "required": {
            "model": ("WANVIDEOMODEL",),
            "image_embeds": ("WANVIDIMAGE_EMBEDS",),
            "steps": ("INT", {"default": 30, "min": 1}),
            "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
            "scheduler": (scheduler_list, {"default": "unipc"}),
        },
        "optional": {
            "text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"}),
            "context_options": ("WANVIDCONTEXT",),
            "uni3c_embeds": ("UNI3C_EMBEDS",),
        },
    }


def _build_wananimateplus_easy_sampler_inputs(
    model,
    image_embeds,
    steps,
    cfg,
    shift,
    seed,
    force_offload,
    scheduler,
    text_embeds=None,
    samples=None,
    context_options=None,
    uni3c_embeds=None,
):
    params = inspect.signature(WanVideoSampler.process).parameters
    sampler_inputs = {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in params.items()
        if name != "self"
    }
    sampler_inputs.update(
        {
            "model": model,
            "image_embeds": image_embeds,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "seed": seed,
            "force_offload": force_offload,
            "scheduler": scheduler,
            "riflex_freq_index": 0,
            "text_embeds": text_embeds,
            "samples": samples,
            "context_options": context_options,
            "uni3c_embeds": uni3c_embeds,
            "rope_function": "comfy",
        }
    )
    return sampler_inputs


class WanAnimatePlusEasySampler(WanVideoSamplerFromSettings):
    DESCRIPTION = "Simplified WanAnimatePlus sampler exposing the common controls while using the same sampler settings path."

    @classmethod
    def INPUT_TYPES(s):
        return _wananimateplus_easy_sampler_input_types()

    def process(
        self,
        model,
        image_embeds,
        steps,
        cfg,
        shift,
        seed,
        force_offload,
        scheduler,
        text_embeds=None,
        samples=None,
        context_options=None,
        uni3c_embeds=None,
    ):
        sampler_inputs = _build_wananimateplus_easy_sampler_inputs(
            model,
            image_embeds,
            steps,
            cfg,
            shift,
            seed,
            force_offload,
            scheduler,
            text_embeds,
            samples,
            context_options,
            uni3c_embeds,
        )
        return super().process(sampler_inputs)


class WanAnimatePlusEasySamplerSettings:
    RETURN_TYPES = ("SAMPLER_ARGS",)
    RETURN_NAMES = ("sampler_inputs",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Simplified WanAnimatePlus sampler settings exposing the common controls while preserving the full sampler settings path."

    @classmethod
    def INPUT_TYPES(s):
        return _wananimateplus_easy_sampler_input_types()

    def process(
        self,
        model,
        image_embeds,
        steps,
        cfg,
        shift,
        seed,
        force_offload,
        scheduler,
        text_embeds=None,
        samples=None,
        context_options=None,
        uni3c_embeds=None,
    ):
        return (
            _build_wananimateplus_easy_sampler_inputs(
                model,
                image_embeds,
                steps,
                cfg,
                shift,
                seed,
                force_offload,
                scheduler,
                text_embeds,
                samples,
                context_options,
                uni3c_embeds,
            ),
        )


def _wanap_flow_patch_shift(model, shift):
    if shift is None:
        return model
    try:
        m = model.clone()
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=float(shift), multiplier=1000)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)
        return m
    except Exception as e:
        log.warning(f"WanAnimatePlus SCAIL-2 Flow Sampler could not apply shift={shift}: {e}")
        return model


def _wanap_flow_prepare_callback(model, steps):
    steps = max(1, int(steps))
    if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]:
        from latent_preview import prepare_callback
    else:
        from .latent_preview import prepare_callback
    return prepare_callback(model, steps)


def _wanap_flow_sample_once(
    model,
    positive,
    negative,
    latent,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=True,
    final_freeze_strength=1.0,
    callback=None,
    callback_offset=0,
    callback_total=None,
):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    sample_callback = None
    if callback is not None:
        callback_offset = int(callback_offset or 0)
        callback_total = max(1, int(callback_total if callback_total is not None else steps))

        def sample_callback(step, x0, x, total_steps):
            callback(callback_offset + int(step), x0, x, callback_total)

    samples = comfy.sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=sample_callback,
        disable_pbar=False,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    out = _wanap_flow_apply_freeze_strength(latent, out, final_freeze_strength)
    return out


def _wanap_flow_apply_freeze_strength(source_latent, sampled_latent, strength):
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return sampled_latent
    freeze = source_latent.get(FLOW_FREEZE_MASK_KEY, None)
    if freeze is None:
        base = source_latent.get("noise_mask", None)
        if base is None:
            return sampled_latent
        freeze = base != 1.0
    samples = sampled_latent["samples"]
    source = source_latent["samples"].to(device=samples.device, dtype=samples.dtype)
    freeze = freeze.to(device=samples.device, dtype=samples.dtype)
    if freeze.shape[0] != samples.shape[0]:
        freeze = freeze.repeat(math.ceil(samples.shape[0] / freeze.shape[0]), 1, 1, 1, 1)[:samples.shape[0]]
    if source.shape[0] != samples.shape[0]:
        source = source.repeat(math.ceil(samples.shape[0] / source.shape[0]), 1, 1, 1, 1)[:samples.shape[0]]
    mask = (freeze * strength).clamp(0.0, 1.0)
    out = sampled_latent.copy()
    out["samples"] = source * mask + samples * (1.0 - mask)
    return out


def _wanap_flow_phase_noise_mask(latent, phase_protect):
    samples = latent["samples"]
    base = latent.get("noise_mask", None)
    if base is None:
        base = torch.ones((samples.shape[0], 1, samples.shape[2], samples.shape[-2], samples.shape[-1]), device=samples.device, dtype=samples.dtype)
    else:
        base = base.to(device=samples.device, dtype=samples.dtype)
    freeze = latent.get(FLOW_FREEZE_MASK_KEY, None)
    if freeze is None:
        handoff = latent.get(FLOW_HANDOFF_MASK_KEY, None)
        freeze = handoff if handoff is not None else (base != 1.0)
    freeze = freeze.to(device=base.device, dtype=torch.bool)
    out = base.clone()
    denoise_value = max(0.0, min(1.0, 1.0 - float(phase_protect)))
    out = torch.where(freeze, torch.full_like(out, denoise_value), out)
    return out


def _wanap_flow_sample_two_phase(
    model,
    positive,
    negative,
    latent,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    phase1_mask,
    phase2_mask,
    phase2_start_step,
    allow_two_phase,
    callback=None,
    callback_offset=0,
    callback_total=None,
):
    phase2_start_step = int(phase2_start_step or 0)
    callback_total = max(1, int(callback_total if callback_total is not None else steps))
    if not allow_two_phase or phase2_start_step <= 0:
        return _wanap_flow_sample_once(
            model,
            positive,
            negative,
            latent,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            callback=callback,
            callback_offset=callback_offset,
            callback_total=callback_total,
        )
    if phase2_start_step >= int(steps):
        log.warning("WanAnimatePlus SCAIL-2 Flow two-phase start step is outside the sampling range; using one-phase sampling.")
        return _wanap_flow_sample_once(
            model,
            positive,
            negative,
            latent,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            callback=callback,
            callback_offset=callback_offset,
            callback_total=callback_total,
        )

    phase1_latent = latent.copy()
    phase1_latent["noise_mask"] = _wanap_flow_phase_noise_mask(latent, phase1_mask)
    phase1 = _wanap_flow_sample_once(
        model,
        positive,
        negative,
        phase1_latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        last_step=phase2_start_step,
        force_full_denoise=False,
        final_freeze_strength=0.0,
        callback=callback,
        callback_offset=callback_offset,
        callback_total=callback_total,
    )
    phase1 = _wanap_flow_apply_freeze_strength(latent, phase1, 1.0)

    phase2_latent = latent.copy()
    phase2_latent["samples"] = phase1["samples"]
    phase2_latent["noise_mask"] = _wanap_flow_phase_noise_mask(latent, phase2_mask)
    phase2 = _wanap_flow_sample_once(
        model,
        positive,
        negative,
        phase2_latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        disable_noise=True,
        start_step=phase2_start_step,
        force_full_denoise=True,
        final_freeze_strength=0.0,
        callback=callback,
        callback_offset=int(callback_offset or 0) + phase2_start_step,
        callback_total=callback_total,
    )
    return _wanap_flow_apply_freeze_strength(latent, phase2, phase2_mask)


class WanAnimatePlusSCAIL2FlowSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "phase1_mask": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "phase2_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "phase2_start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = "Official ComfyUI-compatible SCAIL-2 sampler. Uses MODEL/CONDITIONING/LATENT and comfy.sample.sample."

    def process(
        self,
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        sampler_name,
        scheduler,
        shift,
        seed,
        force_offload,
        phase1_mask,
        phase2_mask,
        phase2_start_step,
        vae=None,
    ):
        runtime = latent.get(FLOW_RUNTIME_KEY, None)
        model = _wanap_flow_patch_shift(model, shift)
        has_context_handler = bool(getattr(model, "model_options", {}).get("context_handler", None))
        flow_vae = vae
        if flow_vae is None and runtime is not None:
            flow_vae = runtime.get(FLOW_RUNTIME_VAE_KEY, None)
        deferred_mode = runtime.get(FLOW_DEFERRED_BUILD_KEY, None) if runtime is not None else None

        try:
            if runtime is not None and runtime.get("looping", False) and not has_context_handler:
                if flow_vae is None:
                    raise ValueError(
                        "WanAnimatePlus SCAIL-2 Flow internal loop requires a VAE. "
                        "Connect VAE or use official context mode."
                    )
                out = self._process_loop(
                    model,
                    positive,
                    negative,
                    latent,
                    flow_vae,
                    runtime,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    int(seed),
                    phase1_mask,
                    phase2_mask,
                    phase2_start_step,
                )
            else:
                if runtime is not None and runtime.get("looping", False) and has_context_handler:
                    log.info("WanAnimatePlus SCAIL-2 Flow: official model context handler detected; disabling internal loop.")
                if runtime is not None and deferred_mode is not None:
                    if flow_vae is None:
                        raise ValueError("WanAnimatePlus SCAIL-2 Flow deferred build requires a VAE.")
                    positive, negative, latent = build_conditioning_and_latent(
                        positive,
                        negative,
                        flow_vae,
                        runtime,
                        start_frame=0,
                        length=runtime["num_frames"],
                        include_runtime=True,
                    )
                    release_flow_vae(flow_vae)
                    runtime = clean_flow_runtime_for_output(runtime)
                    latent[FLOW_RUNTIME_KEY] = runtime
                if int(phase2_start_step or 0) > 0:
                    log.warning("WanAnimatePlus SCAIL-2 Flow two-phase settings only affect internal loop handoff chunks; ignoring them for this sample.")
                if runtime is not None:
                    sample_mode = "context handler" if runtime.get("looping", False) and has_context_handler else "one-shot"
                    log.info(
                        f"WanAnimatePlus SCAIL-2 Flow {sample_mode} sampling: "
                        f"{int(runtime.get('requested_output_frames', runtime.get('num_frames', 0)))} frames "
                        f"at {runtime['width']}x{runtime['height']} with {steps} steps, "
                        f"sampler={sampler_name}, scheduler={scheduler}"
                    )
                callback = _wanap_flow_prepare_callback(model, steps)
                out = _wanap_flow_sample_once(
                    model,
                    positive,
                    negative,
                    latent,
                    int(seed),
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    callback=callback,
                    callback_total=steps,
                )
        finally:
            if force_offload:
                if hasattr(mm, "unload_all_models"):
                    mm.unload_all_models()
                mm.soft_empty_cache()
                gc.collect()

        return (out,)

    def _main_ref_frames(self, runtime):
        ref = runtime.get("ref_image", None)
        if ref is None:
            return None
        return ref[:1, :, :, :3]

    def _process_loop(
        self,
        model,
        positive,
        negative,
        latent,
        vae,
        runtime,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        phase1_mask,
        phase2_mask,
        phase2_start_step,
    ):
        total_frames = int(runtime["num_frames"])
        requested_output_frames = int(runtime.get("requested_output_frames", total_frames))
        canvas_expansion_px = int(runtime.get("canvas_expansion_px", 0) or 0)
        window_frames = int(runtime["frame_window_size"])
        prev_count = int(runtime.get("previous_frame_count", 5))
        if window_frames <= prev_count:
            raise ValueError("WanAnimatePlus SCAIL-2 Flow frame_window_size must be larger than the 5-frame handoff.")
        if canvas_expansion_px and window_frames <= canvas_expansion_px:
            raise ValueError("WanAnimatePlus SCAIL-2 Flow frame_window_size must be larger than the 21-frame transition canvas.")
        stride = max(1, window_frames - prev_count)
        num_chunks = 1 if total_frames <= window_frames else math.ceil((total_frames - window_frames) / stride) + 1
        log.info(
            f"WanAnimatePlus SCAIL-2 Flow loop sampling: "
            f"{requested_output_frames} requested frames, {total_frames} sample frames, "
            f"{num_chunks} chunks, {window_frames} frames/chunk, stride {stride}, {prev_count} frame handoff"
        )

        previous_frames = None
        output_chunks = []
        chunk_seeds = []
        transition_match_ref = runtime.get("transition_match_ref", None)
        transition_raw_last_frame = runtime.get("transition_raw_last_frame", None)
        last_matched_ref_frame = transition_raw_last_frame
        last_auto_drift_means = runtime.get("transition_raw_tail_means", None)

        def _select_loop_colormatch_ref(chunk_idx, last_ref_frame):
            method = runtime.get("transition_colormatch", "disabled")
            if method in ("disabled", "auto_drift"):
                return None
            has_transition = runtime.get("transition_video", None) is not None
            loop_ref = runtime.get("loop_colormatch_reference", "previous_matched_frame")
            if chunk_idx == 0:
                if not has_transition:
                    return None
                if loop_ref == "main_ref_image":
                    return transition_match_ref
                return transition_raw_last_frame
            if loop_ref == "main_ref_image":
                return transition_match_ref
            if last_ref_frame is not None:
                return last_ref_frame
            return transition_match_ref

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * stride
            has_handoff = chunk_idx > 0 and previous_frames is not None and previous_frames.shape[0] > 0
            if has_handoff:
                chunk_frames = window_frames
                cond_start = chunk_start
            else:
                chunk_frames = window_frames
                cond_start = chunk_start

            aligned_chunk_frames = align_4n1(chunk_frames)
            chunk_positive, chunk_negative, chunk_latent = build_conditioning_and_latent(
                positive,
                negative,
                vae,
                runtime,
                start_frame=cond_start,
                length=aligned_chunk_frames,
                previous_frames=previous_frames,
                include_runtime=False,
            )
            release_flow_vae(vae)

            chunk_seed = int.from_bytes(os.urandom(8), "little")
            chunk_seeds.append(chunk_seed)
            log.info(
                f"WanAnimatePlus SCAIL-2 Flow chunk {chunk_idx + 1}: "
                f"start={chunk_start}, frames={chunk_frames}, seed={chunk_seed}"
            )
            if has_handoff and 0 < int(phase2_start_step or 0) < int(steps):
                log.info(
                    f"WanAnimatePlus SCAIL-2 Flow two-phase chunk {chunk_idx + 1}/{num_chunks}: "
                    f"phase2 starts at step {int(phase2_start_step)}, "
                    f"phase1_mask={float(phase1_mask):.3f}, phase2_mask={float(phase2_mask):.3f}"
                )
            chunk_callback = _wanap_flow_prepare_callback(model, steps)
            sampled = _wanap_flow_sample_two_phase(
                model,
                chunk_positive,
                chunk_negative,
                chunk_latent,
                chunk_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                phase1_mask,
                phase2_mask,
                phase2_start_step,
                allow_two_phase=has_handoff,
                callback=chunk_callback,
                callback_total=steps,
            )

            decoded = decode_latent_to_images(vae, sampled, tiled_vae=runtime.get("tiled_vae", False))
            decoded = decoded[:chunk_frames]
            if chunk_idx == 0 and canvas_expansion_px > 0:
                output_chunk = decoded[canvas_expansion_px:]
            elif has_handoff:
                output_chunk = decoded[prev_count:]
            else:
                output_chunk = decoded

            method = runtime.get("transition_colormatch", "disabled")
            if method != "disabled":
                if method == "auto_drift":
                    output_chunk = auto_drift_frames(output_chunk, last_auto_drift_means, chunk_idx, num_chunks)
                else:
                    ref_frames = _select_loop_colormatch_ref(chunk_idx, last_matched_ref_frame)
                    output_chunk = color_match_frames(output_chunk, ref_frames, method)

            output_chunks.append(output_chunk.detach().cpu())
            previous_frames = take_tail_with_front_pad(output_chunk.detach().cpu(), prev_count)
            last_matched_ref_frame = previous_frames[-1:, :, :, :3]
            last_auto_drift_means = auto_drift_tail_means(output_chunk)

        video = torch.cat(output_chunks, dim=0)[:requested_output_frames].clamp(0.0, 1.0)
        log.info(f"WanAnimatePlus SCAIL-2 Flow chunk seeds: {chunk_seeds}")
        return {
            "video": video.mul(2.0).sub(1.0),
            "output_frame_count": requested_output_frames,
            "scail2_chunk_seeds": chunk_seeds,
            FLOW_RUNTIME_KEY: clean_flow_runtime_for_output(runtime),
        }


class WanVideoSamplerExtraArgs():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
            }
        }
    RETURN_TYPES = ("WANVIDSAMPLEREXTRAARGS",)
    RETURN_NAMES = ("extra_args", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, *args, **kwargs):
        return kwargs,


class WanVideoSamplerv2(WanVideoSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": ("WANVIDEOSCHEDULER",),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
                "guidance_mode": (["cfg", "apg", "apg_chain", "cfg_chain"], {"default": "cfg", "tooltip": "Guidance mode: cfg (standard CFG), apg (single-condition APG), apg_chain (image-reference APG chain), or cfg_chain (Bernini chained CFG)"}),
                "apg_eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "APG: parallel/orthogonal balance (0=orthogonal only, 1=full)"}),
                "apg_momentum": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "APG: EMA momentum for smoothing guidance differences"}),
                "apg_norm_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "APG: L2 norm clipping threshold (0=disabled)"}),
                "apg_omega": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG: guidance strength"}),
                "apg_omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG chain: image-only guidance strength"}),
                "apg_omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "APG chain: text+image guidance strength"}),
                "chain_omega_V": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: video-only guidance strength"}),
                "chain_omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: extra reference context strength (VI - V; reference video/images)"}),
                "chain_omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Chain CFG: text+image guidance strength"}),
                "extra_args": ("WANVIDSAMPLEREXTRAARGS", ),
            }
        }

    def process(self, *args, extra_args=None, **kwargs):
        import inspect
        params = inspect.signature(WanVideoSampler.process).parameters
        args_dict = {name: kwargs.get(name, param.default if param.default is not inspect.Parameter.empty else None)
                     for name, param in params.items() if name != "self"}

        if extra_args is not None:
            args_dict.update(extra_args)
        else:
            args_dict["rope_function"] = "comfy"

        return super().process(**args_dict)


class WanVideoScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of steps for the scheduler"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "Starting step for the scheduler"}),
                "end_step": ("INT", {"default": -1, "min": -1, "tooltip": "Ending step for the scheduler"})
            },
            "optional": {
                "sigmas": ("SIGMAS", ),
                "enhance_hf": ("BOOLEAN", {"default": False, "tooltip": "Enhanced high-frequency denoising schedule"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SIGMAS", "INT", "FLOAT", scheduler_list, "INT", "INT",)
    RETURN_NAMES = ("sigmas", "steps", "shift", "scheduler", "start_step", "end_step")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, scheduler, steps, start_step, end_step, shift, unique_id, sigmas=None, enhance_hf=False):
        sample_scheduler, timesteps, start_idx, end_idx = get_scheduler(
            scheduler, steps, start_step, end_step, shift, device, sigmas=sigmas, log_timesteps=True, enhance_hf=enhance_hf)

        scheduler_dict = {
            "sample_scheduler": sample_scheduler,
            "timesteps": timesteps,
        }

        try:
            from server import PromptServer
            import io
            import base64
            import matplotlib.pyplot as plt
        except Exception:
            PromptServer = None
        if unique_id and PromptServer is not None:
            try:
                # Plot sigmas and save to a buffer
                sigmas_np = sample_scheduler.full_sigmas.cpu().numpy()
                if not np.isclose(sigmas_np[-1], 0.0, atol=1e-6):
                    sigmas_np = np.append(sigmas_np, 0.0)
                buf = io.BytesIO()
                fig = plt.figure(facecolor='#353535')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#353535')  # Set axes background color
                x_values = range(0, len(sigmas_np))
                ax.plot(x_values, sigmas_np)
                # Annotate each sigma value
                ax.scatter(x_values, sigmas_np, color='white', s=20, zorder=3)  # Small dots at each sigma
                for x, y in zip(x_values, sigmas_np):
                    # Show all annotations if few steps, or just show split step annotations
                    show_annotation = len(sigmas_np) <= 10
                    is_split_step = (start_idx > 0 and x == start_idx) or (end_idx != -1 and x == end_idx + 1)

                    if show_annotation or is_split_step:
                        color = 'orange'
                        if is_split_step:
                            color = 'yellow'
                        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(10, 1), ha='center', color=color, fontsize=12)
                ax.set_xticks(x_values)
                ax.set_title("Sigmas", color='white')           # Title font color
                ax.set_xlabel("Step", color='white')            # X label font color
                ax.set_ylabel("Sigma Value", color='white')     # Y label font color
                ax.tick_params(axis='x', colors='white', labelsize=10)        # X tick color
                ax.tick_params(axis='y', colors='white', labelsize=10)        # Y tick color
                # Add split point if end_step is defined
                end_idx += 1
                if end_idx != -1 and 0 <= end_idx < len(sigmas_np) - 1:
                    ax.axvline(end_idx, color='red', linestyle='--', linewidth=2, label='end_step split')
                # Add split point if start_step is defined
                if start_idx > 0 and 0 <= start_idx < len(sigmas_np):
                    ax.axvline(start_idx, color='green', linestyle='--', linewidth=2, label='start_step split')
                if (end_idx != -1 and 0 <= end_idx < len(sigmas_np)) or (start_idx > 0 and 0 <= start_idx < len(sigmas_np)):
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:
                        ax.legend()
                # Draw shaded range
                range_start_idx = start_idx if start_idx > 0 else 0
                range_end_idx = end_idx if end_idx > 0 and end_idx < len(sigmas_np) else len(sigmas_np) - 1
                if range_start_idx < range_end_idx:
                    ax.axvspan(range_start_idx, range_end_idx, color='lightblue', alpha=0.1, label='Sampled Range')


                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Send as HTML img tag with base64 data
                html_img = f"<img src='data:image/png;base64,{img_base64}' alt='Sigmas Plot' style='max-width:100%; height:100%; overflow:hidden; display:block;'>"
                PromptServer.instance.send_progress_text(html_img, unique_id)
            except Exception as e:
                log.error(f"Failed to send sigmas plot: {e}")
                pass

        return (sigmas, steps, shift, scheduler_dict, start_step, end_step)

class WanVideoSchedulerv2(WanVideoScheduler):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of steps for the scheduler"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "Starting step for the scheduler"}),
                "end_step": ("INT", {"default": -1, "min": -1, "tooltip": "Ending step for the scheduler"})
            },
            "optional": {
                "sigmas": ("SIGMAS", ),
                "enhance_hf": ("BOOLEAN", {"default": False, "tooltip": "Enhanced high-frequency denoising schedule"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WANVIDEOSCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, *args, **kwargs):
        sigmas, steps, shift, scheduler_dict, start_step, end_step = super().process(*args, **kwargs)
        return scheduler_dict,

NODE_CLASS_MAPPINGS = {
    "WanVideoSampler": WanVideoSampler,
    "WanVideoSamplerSettings": WanVideoSamplerSettings,
    "WanVideoSamplerFromSettings": WanVideoSamplerFromSettings,
    "WanAnimatePlusEasySampler": WanAnimatePlusEasySampler,
    "WanAnimatePlusEasySamplerSettings": WanAnimatePlusEasySamplerSettings,
    "WanAnimatePlusSCAIL2FlowSampler": WanAnimatePlusSCAIL2FlowSampler,
    "WanVideoSamplerv2": WanVideoSamplerv2,
    "WanVideoSamplerExtraArgs": WanVideoSamplerExtraArgs,
    "WanVideoScheduler": WanVideoScheduler,
    "WanVideoSchedulerv2": WanVideoSchedulerv2,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSampler": "WanVideo Sampler",
    "WanVideoSamplerSettings": "WanVideo Sampler Settings",
    "WanVideoSamplerFromSettings": "WanVideo Sampler From Settings",
    "WanAnimatePlusEasySampler": "WanAnimatePlus Easy Sampler",
    "WanAnimatePlusEasySamplerSettings": "WanAnimatePlus Easy SamplerSettings",
    "WanAnimatePlusSCAIL2FlowSampler": "WanAnimatePlus SCAIL_2 Flow Sampler",
    "WanVideoSamplerv2": "WanVideo Sampler v2",
    "WanVideoSamplerExtraArgs": "WanVideoSampler v2 Extra Args",
    "WanVideoScheduler": "WanVideo Scheduler",
    "WanVideoSchedulerv2": "WanVideo Scheduler v2",
}
