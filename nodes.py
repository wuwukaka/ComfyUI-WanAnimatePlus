# Copyright (c) 2025 kijai
# Modified from nodes.py in ComfyUI-WanVideoWrapper.
# Original project: https://github.com/kijai/ComfyUI-WanVideoWrapper
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Added prefix_frames support, reversed-prefix sampling, and single-frame prefix reference encoding for WanVideoAnimateEmbeds.
#   - Added transition_video, transition/outfit layouts, and canvas_expansion_px output trimming.
#   - Added prefix/transition metadata for sampler-side context-window and looping continuation.
#   - Added WanAnimatePlusEverAnimateEmbeds with anchors, pose/face, bg/mask, pingpong, random/user-first anchors, repeat-anchor padding, and offload controls.
#   - Added WanAnimatePlusBernini with context_latents/context_roles for source video, reference video, and reference images.
#   - Added WanAnimatePlusSCAIL2Embeds for wrapper-native SCAIL-2 ref/pose/mask conditioning and prefix/transition freeze latents.
#   - Added SCAIL-2 bg_image reference/prefix composition for animation mode.
#   - Added Bernini task guidance recommendations and native-aspect reference resizing.
#   - Added WanAnimatePlus signature widget to the embeds node.
#   - Added SCAIL-2 auto_drift loop colormatch metadata for sampler-side seam correction.
#   - Added SCAIL-2 loop two-phase sampling settings; thanks to
#     checknickname/ComfyUI-Scail2-Sampler-Helper for the idea and
#     user2318/ComfyUI-CustomNodeKit as an MIT-licensed reference project.
#   - Added official-ComfyUI-compatible SCAIL-2 Flow embeds and VAE decode nodes
#     using MODEL/VAE/CONDITIONING/LATENT/IMAGE interfaces while preserving
#     WanAnimatePlus SCAIL-2 bg/prefix/reference/transition behavior.
# Licensed under the Apache License, Version 2.0
import os, gc, math
import torch
import torch.nn.functional as F
import hashlib
from tqdm import tqdm

from .utils import(log, clip_encode_image_tiled, add_noise_to_reference_video, set_module_tensor_to_device)
from .taehv import TAEHV
from .scail2_flow import (
    FLOW_DEFERRED_BUILD_KEY,
    FLOW_RUNTIME_VAE_KEY,
    build_conditioning_and_latent,
    build_deferred_latent,
    decode_latent_to_images,
    make_runtime,
    release_flow_vae,
)

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


def _sample_reversed_prefix_frames(frames, count):
    if count <= 0:
        return frames[:0]
    indices = torch.arange(count, device=frames.device) * 2
    indices = torch.clamp(indices, max=frames.shape[0] - 1)
    return torch.flip(frames.index_select(0, indices), [0])


class WanVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoSetBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
               },
            "optional": {
                "block_swap_args": ("BLOCKSWAPARGS", ),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, block_swap_args=None):
        if block_swap_args is None:
            return (model,)
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args     

        return (patcher,)

class WanVideoSetRadialAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "dense_attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sparse_sage_attention",
                    ], {"default": "sageattn", "tooltip": "The attention mode for dense attention"}),
                "dense_blocks": ("INT",  {"default": 1, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to apply normal attention to"}),
                "dense_vace_blocks": ("INT",  {"default": 1, "min": 0, "max": 15, "step": 1, "tooltip": "Number of vace blocks to apply normal attention to"}),
                "dense_timesteps": ("INT",  {"default": 2, "min": 0, "max": 100, "step": 1, "tooltip": "The step to start applying sparse attention"}),
                "decay_factor": ("FLOAT",  {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "tooltip": "Controls how quickly the attention window shrinks as the distance between frames increases in the sparse attention mask."}),
                "block_size":([128, 64], {"default": 128, "tooltip": "Radial attention block size, larger blocks are faster but restricts usable dimensions more."}),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Sets radial attention parameters, dense attention refers to normal attention"

    def loadmodel(self, model, dense_attention_mode, dense_blocks, dense_vace_blocks, dense_timesteps, decay_factor, block_size):
        if "radial" not in model.model.diffusion_model.attention_mode:
            raise Exception("Enable radial attention first in the model loader.")
            
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}

        patcher.model_options["transformer_options"]["dense_attention_mode"] = dense_attention_mode
        patcher.model_options["transformer_options"]["dense_blocks"] = dense_blocks
        patcher.model_options["transformer_options"]["dense_vace_blocks"] = dense_vace_blocks
        patcher.model_options["transformer_options"]["dense_timesteps"] = dense_timesteps
        patcher.model_options["transformer_options"]["decay_factor"] = decay_factor
        patcher.model_options["transformer_options"]["block_size"] = block_size

        return (patcher,)

class WanVideoBlockList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks": ("STRING",  {"default": "1", "multiline":True}),
               }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("block_list", )
    FUNCTION = "create_list"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Comma separated list of blocks to apply block swap to, can also use ranges like '0-5' or '0,2,3-5' etc., can be connected to the dense_blocks input of 'WanVideoSetRadialAttention' node"

    def create_list(self, blocks):
        block_list = []
        for line in blocks.splitlines():
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-", 1))
                        block_list.extend(range(start, end + 1))
                    except Exception:
                        raise ValueError(f"Invalid range: '{part}'")
                else:
                    try:
                        block_list.append(int(part))
                    except Exception:
                        raise ValueError(f"Invalid integer: '{part}'")
        return (block_list,)



# In-memory cache for prompt extender output
_extender_cache = {}

cache_dir = os.path.join(script_directory, 'text_embed_cache')

def get_cache_path(prompt):
    cache_key = prompt.strip()
    cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{cache_hash}.pt")

def get_cached_text_embeds(positive_prompt, negative_prompt):
    
    os.makedirs(cache_dir, exist_ok=True)

    context = None
    context_null = None

    pos_cache_path = get_cache_path(positive_prompt)
    neg_cache_path = get_cache_path(negative_prompt)

    # Try to load positive prompt embeds
    if os.path.exists(pos_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {pos_cache_path}")
            context = torch.load(pos_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    # Try to load negative prompt embeds
    if os.path.exists(neg_cache_path):
        try:
            log.info(f"Loading prompt embeds from cache: {neg_cache_path}")
            context_null = torch.load(neg_cache_path)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}, will re-encode.")

    return context, context_null

class WanVideoTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
            "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
            "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS", {"tooltip": "Use this node to extend the prompt with additional text."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    OUTPUT_TOOLTIPS = ("The text embeddings for both prompts", "The text embeddings for the negative prompt only (for NAG)", "Positive prompt to display prompt extender results")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """Encodes text prompts into text embeddings. This node loads and completely unloads the T5 after done,  
leaving no VRAM or RAM imprint. If prompts have been cached before T5 is not loaded at all.  
negative output is meant to be used with NAG, it contains only negative prompt embeddings.  

Additionally you can provide a Qwen LLM model to extend the positive prompt with either one  
of the original Wan templates or a custom system prompt.  
"""


    def process(self, model_name, precision, positive_prompt, negative_prompt, quantization='disabled', use_disk_cache=True, device="gpu", extender_args=None):
        from .nodes_model_loading import LoadWanVideoT5TextEncoder
        pbar = ProgressBar(3)

        echoshot = True if "[1]" in positive_prompt else False

        # Handle prompt extension with in-memory cache
        orig_prompt = positive_prompt
        if extender_args is not None:
            extender_key = (orig_prompt, str(extender_args))
            if extender_key in _extender_cache:
                positive_prompt = _extender_cache[extender_key]
                log.info(f"Loaded extended prompt from in-memory cache: {positive_prompt}")
            else:
                from .qwen.qwen import QwenLoader, WanVideoPromptExtender
                log.info("Using WanVideoPromptExtender to process prompts")
                qwen, = QwenLoader().load(
                    extender_args["model"], 
                    load_device="main_device" if device == "gpu" else "cpu", 
                    precision=precision)
                positive_prompt, = WanVideoPromptExtender().generate(
                    qwen=qwen,
                    max_new_tokens=extender_args["max_new_tokens"],
                    prompt=orig_prompt,
                    device=device,
                    force_offload=False,
                    custom_system_prompt=extender_args["system_prompt"],
                    seed=extender_args["seed"]
                )
                log.info(f"Extended positive prompt: {positive_prompt}")
                _extender_cache[extender_key] = positive_prompt
                del qwen
            pbar.update(1)

        # Now check disk cache using the (possibly extended) prompt
        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },{"prompt_embeds": context_null}, positive_prompt

        t5, = LoadWanVideoT5TextEncoder().loadmodel(model_name, precision, "main_device", quantization)
        pbar.update(1)

        prompt_embeds_dict, = WanVideoTextEncode().process(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            t5=t5,
            force_offload=False,
            model_to_offload=None,
            use_disk_cache=use_disk_cache,
            device=device
        )
        pbar.update(1)
        del t5
        mm.soft_empty_cache()
        gc.collect() 
        return (prompt_embeds_dict, {"prompt_embeds": prompt_embeds_dict["negative_prompt_embeds"]}, positive_prompt)

#region TextEncode
class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"


    def process(self, positive_prompt, negative_prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        if t5 is None and not use_disk_cache:
            raise ValueError("T5 encoder is required for text encoding. Please provide a valid T5 encoder or enable disk cache.")

        echoshot = True if "[1]" in positive_prompt else False

        if use_disk_cache:
            context, context_null = get_cached_text_embeds(positive_prompt, negative_prompt)
            if context is not None and context_null is not None:
                return{
                    "prompt_embeds": context,
                    "negative_prompt_embeds": context_null,
                    "echoshot": echoshot,
                },
            
        if t5 is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if model_to_offload is not None and device == "gpu":
            try:
                log.info(f"Moving video model to {offload_device}")
                model_to_offload.model.to(offload_device)
            except:
                pass

        encoder = t5["model"]
        dtype = t5["dtype"]
        
        positive_prompts = []
        all_weights = []

        # Split positive prompts and process each with weights
        if "|" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by '|'")
            positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        elif "[1]" in positive_prompt:
            log.info("Multiple positive prompts detected, splitting by [#] and enabling EchoShot")
            import re
            segments = re.split(r'\[\d+\]', positive_prompt)
            positive_prompts_raw = [segment.strip() for segment in segments if segment.strip()]
            assert len(positive_prompts_raw) > 1 and len(positive_prompts_raw) < 7, 'Input shot num must between 2~6 !'
        else:
            positive_prompts_raw = [positive_prompt.strip()]
            
        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        mm.soft_empty_cache()

        if device == "gpu":
            device_to = mm.get_torch_device()
        else:
            device_to = torch.device("cpu")

        if encoder.quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = encoder.dtype

        params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
        if hasattr(encoder, 'state_dict'):
            model_state_dict = encoder.state_dict
        else:
            model_state_dict = encoder.model.state_dict()

        params_list = list(encoder.model.named_parameters())
        pbar = tqdm(params_list, desc="Loading T5 parameters", leave=True)
        for name, param in pbar:
            dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
            value = model_state_dict[name]
            set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
        del model_state_dict
        if hasattr(encoder, 'state_dict'):
            del encoder.state_dict
            mm.soft_empty_cache()
            gc.collect()

        with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
            # Encode positive if not loaded from cache
            if use_disk_cache and context is not None:
                pass
            else:
                context = encoder(positive_prompts, device_to)
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    for text, weight in weights.items():
                        log.info(f"Applying weight {weight} to prompt: {text}")
                        if len(weights) > 0:
                            context[i] = context[i] * weight

            # Encode negative if not loaded from cache
            if use_disk_cache and context_null is not None:
                pass
            else:
                context_null = encoder([negative_prompt], device_to)

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
            "echoshot": echoshot,
        }

        # Save each part to its own cache file if needed
        if use_disk_cache:
            pos_cache_path = get_cache_path(positive_prompt)
            neg_cache_path = get_cache_path(negative_prompt)
            try:
                if not os.path.exists(pos_cache_path):
                    torch.save(context, pos_cache_path)
                    log.info(f"Saved prompt embeds to cache: {pos_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")
            try:
                if not os.path.exists(neg_cache_path):
                    torch.save(context_null, neg_cache_path)
                    log.info(f"Saved prompt embeds to cache: {neg_cache_path}")
            except Exception as e:
                log.warning(f"Failed to save cache: {e}")

        return (prompt_embeds_dict,)
    
    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re
        
        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)
        
        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}
        
        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)
            
        return cleaned_prompt, weights
    
class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompt into text embedding."

    def process(self, prompt, t5=None, force_offload=True, model_to_offload=None, use_disk_cache=False, device="gpu"):
        # Unified cache logic: use a single cache file per unique prompt
        encoded = None
        echoshot = True if "[1]" in prompt else False
        if use_disk_cache:
            cache_dir = os.path.join(script_directory, 'text_embed_cache')
            os.makedirs(cache_dir, exist_ok=True)
            def get_cache_path(prompt):
                cache_key = prompt.strip()
                cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
                return os.path.join(cache_dir, f"{cache_hash}.pt")
            cache_path = get_cache_path(prompt)
            if os.path.exists(cache_path):
                try:
                    log.info(f"Loading prompt embeds from cache: {cache_path}")
                    encoded = torch.load(cache_path)
                except Exception as e:
                    log.warning(f"Failed to load cache: {e}, will re-encode.")

        if t5 is None and encoded is None:
            raise ValueError("No cached text embeds found for prompts, please provide a T5 encoder.")

        if encoded is None:
            try:
                if model_to_offload is not None and device == "gpu":
                    log.info(f"Moving video model to {offload_device}")
                    model_to_offload.model.to(offload_device)
                    mm.soft_empty_cache()
            except:
                pass

            encoder = t5["model"]
            dtype = t5["dtype"]

            if device == "gpu":
                device_to = mm.get_torch_device()
            else:
                device_to = torch.device("cpu")

            if encoder.quantization == "fp8_e4m3fn":
                cast_dtype = torch.float8_e4m3fn
            else:
                cast_dtype = encoder.dtype
            params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
            for name, param in encoder.model.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
                value = encoder.state_dict[name] if hasattr(encoder, 'state_dict') else encoder.model.state_dict()[name]
                set_module_tensor_to_device(encoder.model, name, device=device_to, dtype=dtype_to_use, value=value)
            if hasattr(encoder, 'state_dict'):
                del encoder.state_dict
                mm.soft_empty_cache()
                gc.collect()
            with torch.autocast(device_type=mm.get_autocast_device(device_to), dtype=encoder.dtype, enabled=encoder.quantization != 'disabled'):
                encoded = encoder([prompt], device_to)

            if force_offload:
                encoder.model.to(offload_device)
                mm.soft_empty_cache()

            # Save to cache if enabled
            if use_disk_cache:
                try:
                    if not os.path.exists(cache_path):
                        torch.save(encoded, cache_path)
                        log.info(f"Saved prompt embeds to cache: {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to save cache: {e}")

        prompt_embeds_dict = {
            "prompt_embeds": encoded,
            "negative_prompt_embeds": None,
            "echoshot": echoshot
        }
        return (prompt_embeds_dict,)
    
class WanVideoApplyNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "inplace": ("BOOLEAN", {"default": True, "tooltip": "If true, modifies tensors in place to save memory. Leads to different numerical results which may change the output slightly."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds NAG prompt embeds to original prompt embeds: 'https://github.com/ChenDarYen/Normalized-Attention-Guidance'"

    def process(self, original_text_embeds, nag_text_embeds, nag_scale, nag_tau, nag_alpha, inplace=True):
        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "nag_prompt_embeds": nag_text_embeds["prompt_embeds"],
                "nag_params": {
                    "nag_scale": nag_scale,
                    "nag_tau": nag_tau,
                    "nag_alpha": nag_alpha,
                    "inplace": inplace,
                }
            })
        return (prompt_embeds_dict_copy,)
    
class WanVideoTextEmbedBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Bridge between ComfyUI native text embedding and WanVideoWrapper text embedding"

    def process(self, positive, negative=None):
        prompt_embeds_dict = {
                "prompt_embeds": positive[0][0].to(device),
                "negative_prompt_embeds": negative[0][0].to(device) if negative is not None else None,
            }
        return (prompt_embeds_dict,)
    
#region clip vision
class WanVideoClipVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE", {"tooltip": "Image to encode"}),
            "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}), 
            "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "negative_image": ("IMAGE", {"tooltip": "image to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip_vision, image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2=None, negative_image=None, tiles=0, ratio=1.0):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        if image_2 is not None:
            image = torch.cat([image_1, image_2], dim=0)
        else:
            image = image_1

        clip_vision.model.to(device)
        
        negative_clip_embeds = None

        if tiles > 0:
            log.info("Using tiled image encoding")
            clip_embeds = clip_encode_image_tiled(clip_vision, image.to(device), tiles=tiles, ratio=ratio)
            if negative_image is not None:
                negative_clip_embeds = clip_encode_image_tiled(clip_vision, negative_image.to(device), tiles=tiles, ratio=ratio)
        else:
            if isinstance(clip_vision, ClipVisionModel):
                clip_embeds = clip_vision.encode_image(image).penultimate_hidden_states.to(device)
                if negative_image is not None:
                    negative_clip_embeds = clip_vision.encode_image(negative_image).penultimate_hidden_states.to(device)
            else:
                pixel_values = clip_preprocess(image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                clip_embeds = clip_vision.visual(pixel_values)
                if negative_image is not None:
                    pixel_values = clip_preprocess(negative_image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                    negative_clip_embeds = clip_vision.visual(pixel_values)
    
        log.info(f"Clip embeds shape: {clip_embeds.shape}, dtype: {clip_embeds.dtype}")

        weighted_embeds = []
        weighted_embeds.append(clip_embeds[0:1] * strength_1)

        # Handle all additional embeddings
        if clip_embeds.shape[0] > 1:
            weighted_embeds.append(clip_embeds[1:2] * strength_2)
            
            if clip_embeds.shape[0] > 2:
                for i in range(2, clip_embeds.shape[0]):
                    weighted_embeds.append(clip_embeds[i:i+1])  # Add as-is without strength modifier
            
            # Combine all weighted embeddings
            if combine_embeds == "average":
                clip_embeds = torch.mean(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "sum":
                clip_embeds = torch.sum(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "concat":
                clip_embeds = torch.cat(weighted_embeds, dim=1)
            elif combine_embeds == "batch":
                clip_embeds = torch.cat(weighted_embeds, dim=0)
        else:
            clip_embeds = weighted_embeds[0]
                

        log.info(f"Combined clip embeds shape: {clip_embeds.shape}")
        
        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        clip_embeds_dict = {
            "clip_embeds": clip_embeds,
            "negative_clip_embeds": negative_clip_embeds
        }

        return (clip_embeds_dict,)


class WanVideoClipVisionEncodeV2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "images": ("IMAGE", {"tooltip": "Image sequence to encode. All frames in the IMAGE batch are accepted."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier applied to every image embed"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_image": ("IMAGE", {"tooltip": "Image batch to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip_vision, images, strength, force_offload, crop, combine_embeds, negative_image=None, tiles=0, ratio=1.0):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        if images is None or images.shape[0] == 0:
            raise ValueError("At least one image is required")

        clip_vision.model.to(device)

        negative_clip_embeds = None

        if tiles > 0:
            log.info("Using tiled image encoding")
            clip_embeds = clip_encode_image_tiled(clip_vision, images.to(device), tiles=tiles, ratio=ratio)
            if negative_image is not None:
                negative_clip_embeds = clip_encode_image_tiled(clip_vision, negative_image.to(device), tiles=tiles, ratio=ratio)
        else:
            if isinstance(clip_vision, ClipVisionModel):
                clip_embeds = clip_vision.encode_image(images).penultimate_hidden_states.to(device)
                if negative_image is not None:
                    negative_clip_embeds = clip_vision.encode_image(negative_image).penultimate_hidden_states.to(device)
            else:
                pixel_values = clip_preprocess(images.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                clip_embeds = clip_vision.visual(pixel_values)
                if negative_image is not None:
                    pixel_values = clip_preprocess(negative_image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                    negative_clip_embeds = clip_vision.visual(pixel_values)

        log.info(f"Clip embeds V2 shape: {clip_embeds.shape}, dtype: {clip_embeds.dtype}")

        weighted_embeds = []
        for i in range(clip_embeds.shape[0]):
            weighted_embeds.append(clip_embeds[i:i + 1] * strength)

        if len(weighted_embeds) == 1:
            clip_embeds = weighted_embeds[0]
        elif combine_embeds == "average":
            clip_embeds = torch.mean(torch.stack(weighted_embeds), dim=0)
        elif combine_embeds == "sum":
            clip_embeds = torch.sum(torch.stack(weighted_embeds), dim=0)
        elif combine_embeds == "concat":
            clip_embeds = torch.cat(weighted_embeds, dim=1)
        elif combine_embeds == "batch":
            clip_embeds = torch.cat(weighted_embeds, dim=0)
        else:
            raise ValueError(f"Unsupported combine_embeds mode: {combine_embeds}")

        log.info(f"Combined clip embeds V2 shape: {clip_embeds.shape}")

        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        clip_embeds_dict = {
            "clip_embeds": clip_embeds,
            "negative_clip_embeds": negative_clip_embeds
        }

        return (clip_embeds_dict,)
        
class WanVideoRealisDanceLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
            "pose_cond_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the SMPL model"}),
            "pose_cond_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the SMPL model"}),
            },
            "optional": {
                "smpl_latent": ("LATENT", {"tooltip": "SMPL pose image to encode"}),
                "hamer_latent": ("LATENT", {"tooltip": "Hamer hand pose image to encode"}),
            },
        }

    RETURN_TYPES = ("ADD_COND_LATENTS",)
    RETURN_NAMES = ("add_cond_latents",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, ref_latent, pose_cond_start_percent, pose_cond_end_percent, hamer_latent=None, smpl_latent=None):
        if smpl_latent is None and hamer_latent is None:
            raise Exception("At least one of smpl_latent or hamer_latent must be provided")
        if smpl_latent is None:
            smpl = torch.zeros_like(hamer_latent["samples"])
        else:
            smpl = smpl_latent["samples"]
        if hamer_latent is None:
            hamer = torch.zeros_like(smpl_latent["samples"])
        else:
            hamer = hamer_latent["samples"]

        pose_latent = torch.cat((smpl, hamer), dim=1)
        
        add_cond_latents = {
            "ref_latent": ref_latent["samples"],
            "pose_latent": pose_latent,
            "pose_cond_start_percent": pose_cond_start_percent,
            "pose_cond_end_percent": pose_cond_end_percent,
        }

        return (add_cond_latents,)

    
class WanVideoAddStandInLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "ip_image_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
                    "freq_offset": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "EXPERIMENTAL: RoPE frequency offset between the reference and rest of the sequence"}),
                    #"start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    #"end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, ip_image_latent, freq_offset):
        # Prepare the new extra latent entry
        new_entry = {
            "ip_image_latent": ip_image_latent["samples"],
            "freq_offset": freq_offset,
            #"ip_start_percent": start_percent,
            #"ip_end_percent": end_percent,
        }    

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["standin_input"] = new_entry
        return (updated,)
    
class WanVideoAddBindweaveEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "reference_latents": ("LATENT", {"tooltip": "Reference image to encode"}),
                }, 
                "optional": {
                    "ref_masks": ("MASK", {"tooltip": "Reference mask to encode"}),
                    "qwenvl_embeds_pos": ("QWENVL_EMBEDS", {"tooltip": "Qwen-VL image embeddings for the reference image"}),
                    "qwenvl_embeds_neg": ("QWENVL_EMBEDS", {"tooltip": "Qwen-VL image embeddings for the reference image"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "LATENT", "MASK",)
    RETURN_NAMES = ("image_embeds", "image_embed_preview", "mask_preview",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, reference_latents, ref_masks=None, qwenvl_embeds_pos=None, qwenvl_embeds_neg=None):
        updated = dict(embeds)
        image_embeds = embeds["image_embeds"]
        max_refs = 4
        num_refs = reference_latents["samples"].shape[0]
        pad = torch.zeros(image_embeds.shape[0], max_refs-num_refs, image_embeds.shape[2], image_embeds.shape[3], device=image_embeds.device, dtype=image_embeds.dtype)
        if num_refs < max_refs:
            image_embeds = torch.cat([pad, image_embeds], dim=1)
        ref_latents = [ref_latent for ref_latent in reference_latents["samples"]]
        image_embeds = torch.cat([*ref_latents, image_embeds], dim=1)
        
        mask = embeds.get("mask", None)
        if mask is not None:
            mask_pad = torch.zeros(mask.shape[0], max_refs-num_refs, mask.shape[2], mask.shape[3], device=mask.device, dtype=mask.dtype)
            if num_refs < max_refs:
                mask = torch.cat([mask_pad, mask], dim=1)
            if ref_masks is not None:
                ref_mask_ = common_upscale(ref_masks.unsqueeze(1), mask.shape[3], mask.shape[2], "nearest", "disabled").movedim(0,1)
                ref_mask_ = torch.cat([ref_mask_, torch.zeros(3, ref_mask_.shape[1], ref_mask_.shape[2], ref_mask_.shape[3], device=ref_mask_.device, dtype=ref_mask_.dtype)])
                mask = torch.cat([ref_mask_, mask], dim=1)
            else:
                mask = torch.cat([torch.ones(mask.shape[0], num_refs, mask.shape[2], mask.shape[3], device=mask.device, dtype=mask.dtype), mask], dim=1)

            updated["mask"] = mask

        clip_embeds = updated.get("clip_context", None)
        if clip_embeds is not None:
            B, T, C = clip_embeds.shape
            target_len = max_refs * 257  # 4 * 257 = 1028
            if T < target_len:
                pad = torch.zeros(B, target_len - T, C, device=clip_embeds.device, dtype=clip_embeds.dtype)
                padded_embeds = torch.cat([clip_embeds, pad], dim=1)
                log.info(f"Padded clip embeds from {clip_embeds.shape} to {padded_embeds.shape} for Bindweave")
                updated["clip_context"] = padded_embeds
            else:
                updated["clip_context"] = clip_embeds

        updated["image_embeds"] = image_embeds
        updated["qwenvl_embeds_pos"] = qwenvl_embeds_pos
        updated["qwenvl_embeds_neg"] = qwenvl_embeds_neg
        return (updated, {"samples": image_embeds.unsqueeze(0)}, mask[0].float())
    
class TextImageEncodeQwenVL():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP",),
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                }, 
                "optional": {
                    "image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("QWENVL_EMBEDS",)
    RETURN_NAMES = ("qwenvl_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(cls, clip, prompt, image=None):
        if image is None:
            input_images = []
            llama_template = None
        else:
            input_images = [image[:, :, :, :3]]

            llama_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

        tokens = clip.tokenize(prompt, images=input_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        print("Qwen-VL embeds shape:", conditioning[0][0].shape)
        return (conditioning[0][0],)

class WanVideoAddMTVMotion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "mtv_crafter_motion": ("MTVCRAFTERMOTION",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the MTV motion"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, mtv_crafter_motion, strength, start_percent, end_percent):
        # Prepare the new extra latent entry
        new_entry = {
            "mtv_motion_tokens": mtv_crafter_motion["mtv_motion_tokens"],
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "global_mean": mtv_crafter_motion["global_mean"],
            "global_std": mtv_crafter_motion["global_std"]
        }

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["mtv_crafter_motion"] = new_entry
        return (updated,)

class WanVideoAddStoryMemLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "memory_images": ("IMAGE",),
                    "rope_negative_offset": ("BOOLEAN", {"default": False, "tooltip": "Use positive RoPE frequency offset for the memory latents"}),
                    "rope_negative_offset_frames": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "tooltip": "RoPE frequency offset for the memory latents"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, vae, embeds, memory_images, rope_negative_offset, rope_negative_offset_frames):
        updated = dict(embeds)
        story_mem_latents, = WanVideoEncodeLatentBatch().encode(vae, memory_images)
        updated["story_mem_latents"] = story_mem_latents["samples"].squeeze(2).permute(1, 0, 2, 3)  # [C, T, H, W]
        updated["rope_negative_offset_frames"] = rope_negative_offset_frames if rope_negative_offset else 0
        return (updated,)


class WanVideoSVIProEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "anchor_samples": ("LATENT", {"tooltip": "Initial start image encoded"}),
                    "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
                },
                "optional": {
                    "prev_samples": ("LATENT", {"tooltip": "Last latent from previous generation"}),
                    "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "Number of latents used to continue"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, anchor_samples, num_frames, prev_samples=None, motion_latent_count=1):

        anchor_latent = anchor_samples["samples"][0].clone()

        C, T, H, W = anchor_latent.shape

        total_latents = (num_frames - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        if prev_samples is None or motion_latent_count == 0:
            padding_size = total_latents - anchor_latent.shape[1]
            padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
            y = torch.concat([anchor_latent, padding], dim=1)
        else:
            prev_latent = prev_samples["samples"][0].clone()
            motion_latent = prev_latent[:, -motion_latent_count:]
            padding_size = total_latents - anchor_latent.shape[1] - motion_latent.shape[1]
            padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
            y = torch.concat([anchor_latent, motion_latent, padding], dim=1)

        msk = torch.ones(1, num_frames, H, W, device=device, dtype=dtype)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, H, W)
        msk = msk.transpose(1, 2)[0]

        image_embeds = {
            "image_embeds": y,
            "num_frames": num_frames,
            "lat_h": H,
            "lat_w": W,
            "mask": msk
        }

        return (image_embeds,)

#region I2V encode
class WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
                "augment_empty_frames": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "EXPERIMENTAL: Augment empty frames with the difference to the start image to force more motion"}),
                "empty_frame_pad_image": ("IMAGE", {"tooltip": "Use this image to pad empty frames instead of gray, used with SVI-shot and SVI 2.0 LoRAs"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False,
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None, augment_empty_frames=0.0, empty_frame_pad_image=None):

        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True # end image alone only works with this option

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        pixel_mask = mask.clone()

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_image = start_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)

        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end_image = end_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)

        # Concatenate image with zero frames and encode
        if start_image is not None and end_image is None:
            zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames], dim=1)
            del resized_start_image, zero_frames
        elif start_image is None and end_image is not None:
            zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
            del zero_frames
        elif start_image is None and end_image is None:
            concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
        else:
            if fun_or_fl2v_model:
                zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
            else:
                zero_frames = torch.zeros(3, num_frames-1, H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
            del resized_start_image, zero_frames

        if empty_frame_pad_image is not None:
            pad_img = empty_frame_pad_image.clone()[..., :3]
            if pad_img.shape[1] != H or pad_img.shape[2] != W:
                pad_img = common_upscale(pad_img.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
            pad_img = (pad_img.movedim(-1, 0) * 2 - 1).to(device, dtype=vae.dtype)

            num_pad_frames = pad_img.shape[1]
            num_target_frames = concatenated.shape[1]
            if num_pad_frames < num_target_frames:
                pad_img = torch.cat([pad_img, pad_img[:, -1:].expand(-1, num_target_frames - num_pad_frames, -1, -1)], dim=1)
            else:
                pad_img = pad_img[:, :num_target_frames]

            frame_is_empty = (pixel_mask[0].mean(dim=(-2, -1)) < 0.5)[:concatenated.shape[1]].clone()
            if start_image is not None:
                frame_is_empty[:start_image.shape[0]] = False
            if end_image is not None:
                frame_is_empty[-end_image.shape[0]:] = False

            concatenated[:, frame_is_empty] = pad_img[:, frame_is_empty]

        mm.soft_empty_cache()
        gc.collect()

        vae.to(device)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        del concatenated

        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength
        if augment_empty_frames > 0.0:
            frame_is_empty = (mask[0].mean(dim=(-2, -1)) < 0.5).view(1, -1, 1, 1)
            y = y[:, :1] + (y - y[:, :1]) * ((augment_empty_frames+1) * frame_is_empty + ~frame_is_empty)

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "image_embeds": y.cpu(),
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask.cpu()
        }

        return (image_embeds,)

# region WanAnimate
class WanVideoAnimateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of frames to use for temporal attention window"}),
            "colormatch": (
            [
                'disabled',
                'mkl',
                'hm',
                'reinhard',
                'mvgd',
                'hm-mvgd-hm',
                'hm-mkl-hm',
            ], {
               "default": 'disabled', "tooltip": "Color matching method to use between the windows"
            },),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the pose"}),
            "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the face"}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "ref_images": ("IMAGE", {"tooltip": "Image to encode"}),
                "pose_images": ("IMAGE", {"tooltip": "end frame"}),
                "face_images": ("IMAGE", {"tooltip": "end frame"}),
                "bg_images": ("IMAGE", {"tooltip": "background images"}),
                "mask": ("MASK", {"tooltip": "mask"}),
                "start_ref_image": ("IMAGE", {"tooltip": "start ref image"}),
                "transition_video": ("IMAGE", {"default": None, "tooltip": "Transition video frames (32 images, encoded to 8 latent frames). Acts as hard conditioning guide for seamless connection."}),
                "prefix_frames": ("IMAGE", {"default": None, "tooltip": "Up to 5 prefix images. Image 0 is used once; images 1-4 are repeated 4 times each, for a max 17-frame prefix."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "transition mode": ("BOOLEAN", {"default": True, "label_on": "Transition mode", "label_off": "Outfit mode", "tooltip": "Transition mode: 37-frame layout (17 prefix + 20 transition). Outfit mode: 45-frame layout (17 prefix + 8 reserve + 20 transition)."}),
                "single_frame_prefix_encoding": ("BOOLEAN", {"default": False, "tooltip": "Encode prefix images as individual reference latents instead of expanding the beginning of the canvas."}),
                "Prefix & Transition Video by wuwukasi(bilibili)": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "ON"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, colormatch, pose_strength, face_strength,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None, tiled_vae=False, bg_images=None, mask=None, start_ref_image=None,
                transition_video=None, prefix_frames=None, single_frame_prefix_encoding=False, **kwargs):
        
        W = (width // 16) * 16
        H = (height // 16) * 16

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        wananim_single_frame_prefix = prefix_frames is not None and bool(single_frame_prefix_encoding)
        legacy_prefix_active = prefix_frames is not None and not wananim_single_frame_prefix
        if wananim_single_frame_prefix and ref_images is None:
            raise ValueError("single_frame_prefix_encoding requires ref_images; the main reference is taken from the last ref image.")
        prefix_ref_count = min(prefix_frames.shape[0], 5) if wananim_single_frame_prefix else 0
        if wananim_single_frame_prefix and prefix_frames.shape[0] > 5:
            log.warning(f"Prefix has {prefix_frames.shape[0]} images, max 5. Truncating.")
        num_refs = prefix_ref_count + 1 if wananim_single_frame_prefix else (ref_images.shape[0] if ref_images is not None else 0)
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        transition_mode = kwargs.get("transition mode", True)
        prefix_canvas_extra = 37 if transition_mode else 45
        prefix_transition_start = 17 if transition_mode else 25
        prefix_transition_end = prefix_canvas_extra

        if transition_video is not None and not legacy_prefix_active:

            # --- [Core Mod] Reserve space for insertion logic and shift subsequent actions ---
            # 1. Expand canvas: Add space for 21 pixel frames (corresponding to 6 Latent frames).
            num_frames += 21
            trim = (num_frames - 1) % 4
            num_frames -= trim

            # 2. Shift control signals by 21 pixel frames with sampled+reversed padding
            if pose_images is not None:
                sampled = _sample_reversed_prefix_frames(pose_images, 21)
                pose_images = torch.cat([sampled, pose_images], dim=0)
            if face_images is not None:
                sampled = _sample_reversed_prefix_frames(face_images, 21)
                face_images = torch.cat([sampled, face_images], dim=0)
            if bg_images is not None:
                sampled = _sample_reversed_prefix_frames(bg_images, 21)
                bg_images = torch.cat([sampled, bg_images], dim=0)
            if mask is not None:
                sampled = _sample_reversed_prefix_frames(mask, 21)
                mask = torch.cat([sampled, mask], dim=0)
            # ----------------------------------------------------

            if start_ref_image is not None:
                log.warning("Both transition_video and start_ref_image provided. Using transition_video only (loop disabled).")
        # ============ Prefix frames: expand canvas and shift control signals ============
        if legacy_prefix_active:
            # Expand canvas: transition mode 37 = 17 prefix + 20 transition; outfit mode 45 = 17 prefix + 8 reserve + 20 transition
            extra = prefix_canvas_extra
            num_frames += extra
            # Trim 1-3 frames from end to keep num_frames % 4 == 1 (required by repeat_interleave + view)
            trim = (num_frames - 1) % 4
            num_frames -= trim

            # Pad beginning with sampled+reversed frames for pose/face (sparse temporal context),
            # clamping to the last available frame if the source is shorter than the sampled range.
            if pose_images is not None:
                sampled = _sample_reversed_prefix_frames(pose_images, extra)
                pose_images = torch.cat([sampled, pose_images], dim=0)
            if face_images is not None:
                sampled = _sample_reversed_prefix_frames(face_images, extra)
                face_images = torch.cat([sampled, face_images], dim=0)
            if bg_images is not None:
                sampled = _sample_reversed_prefix_frames(bg_images, extra)
                bg_images = torch.cat([sampled, bg_images], dim=0)
            if mask is not None:
                sampled = _sample_reversed_prefix_frames(mask, extra)
                mask = torch.cat([sampled, mask], dim=0)
        # -----------------------------------------------------------------

        if legacy_prefix_active:
            effective_frames = num_frames - prefix_canvas_extra
        elif transition_video is not None:
            effective_frames = num_frames - 21
        else:
            effective_frames = num_frames
        looping = effective_frames > frame_window_size or start_ref_image is not None

        if num_frames < frame_window_size:
            frame_window_size = num_frames

        target_shape = (16, (num_frames - 1) // 4 + 1 + num_refs, lat_h, lat_w)
        latent_window_size = ((frame_window_size - 1) // 4)

        if not looping:
            if not wananim_single_frame_prefix:
                num_frames = num_frames + num_refs * 4
            # latent_window_size must cover the full bg latent range (including prefix/transition expansion),
            # otherwise context windows that reach past the original frame count will clamp pose indices
            latent_window_size = target_shape[1] - num_refs
        else:
            latent_window_size = latent_window_size + 1

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)
        # Resize and rearrange the input image dimensions
        pose_latents = ref_latent = None
        if pose_images is not None:
            pose_images = pose_images[..., :3]
            if pose_images.shape[1] != H or pose_images.shape[2] != W:
                resized_pose_images = common_upscale(pose_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_pose_images = pose_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_pose_images = resized_pose_images * 2 - 1
            if not looping:
                pose_latents = vae.encode([resized_pose_images.to(device, vae.dtype)], device,tiled=tiled_vae)
                pose_latents = pose_latents.to(offload_device)
            
                if pose_latents.shape[2] < latent_window_size:
                    log.info(f"WanAnimate: Padding pose latents from {pose_latents.shape} to length {latent_window_size}")
                    pad_len = latent_window_size - pose_latents.shape[2]
                    pad = torch.zeros(pose_latents.shape[0], pose_latents.shape[1], pad_len, pose_latents.shape[3], pose_latents.shape[4], device=pose_latents.device, dtype=pose_latents.dtype)
                    pose_latents = torch.cat([pose_latents, pad], dim=2)
                del resized_pose_images
            else:
                resized_pose_images = resized_pose_images.to(offload_device, dtype=vae.dtype)            

        bg_latents = None
        if bg_images is not None:
            if bg_images.shape[1] != H or bg_images.shape[2] != W:
                resized_bg_images = common_upscale(bg_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_bg_images = bg_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_bg_images = (resized_bg_images[:3] * 2 - 1)

        actual_prefix_px = 0
        prefix_pixel_data = None  # holds [C, actual_prefix_px, H, W] normalized pixel data for reuse
        if legacy_prefix_active:
            pf = prefix_frames
            b_pf, h_pf, w_pf, c_pf = pf.shape
            log.info(f"Prefix frames input: {b_pf} frames, {h_pf}x{w_pf}")
            if b_pf > 5:
                log.warning(f"Prefix has {b_pf} images, max 5. Truncating.")
                pf = pf[:5]
                b_pf = 5
            pf_frames = pf[0:1]
            for i in range(1, b_pf):
                pf_frames = torch.cat([pf_frames, pf[i:i+1].repeat(4, 1, 1, 1)], dim=0)
            actual_prefix_px = pf_frames.shape[0]
            log.info(f"Prefix: {b_pf} images -> {actual_prefix_px} pixel frames")
            if h_pf != H or w_pf != W:
                pf_frames = common_upscale(pf_frames.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
            prefix_pixel_data = pf_frames.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, actual_prefix_px, H, W]
            del pf, pf_frames

        if not looping:
            if bg_images is None:
                bg_frame_count = num_frames if wananim_single_frame_prefix else num_frames - num_refs
                resized_bg_images = torch.zeros(3, bg_frame_count, H, W, device=device, dtype=vae.dtype)

            # ============ Prefix: replace first N pixel frames of canvas ============
            if prefix_pixel_data is not None:
                resized_bg_images[:, :actual_prefix_px] = prefix_pixel_data.to(device, dtype=resized_bg_images.dtype)
                log.info(f"Prefix: replaced first {actual_prefix_px} pixel frames of black canvas")
                # If transition_video also present, embed last 20 frames into prefix transition area
                if transition_video is not None:
                    tv = transition_video  # [B, H, W, C]
                    b_tv = tv.shape[0]
                    if b_tv >= 20:
                        tv = tv[-20:]
                    else:
                        tv = torch.cat([tv[0:1].repeat(20 - b_tv, 1, 1, 1), tv], dim=0)
                    if tv.shape[1] != H or tv.shape[2] != W:
                        tv = common_upscale(tv.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
                    tv = tv.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, 20, H, W]
                    resized_bg_images[:, prefix_transition_start:prefix_transition_end] = tv.to(device, dtype=resized_bg_images.dtype)
                    log.info(f"Prefix+Transition: embedded last 20 transition frames into canvas positions {prefix_transition_start}-{prefix_transition_end}")
            # ==========================================================================

            # ============ Transition (no prefix): embed into canvas first 21 frames ============
            if transition_video is not None and not legacy_prefix_active:
                tv = transition_video  # [B, H, W, C]
                b_tv = tv.shape[0]
                if b_tv >= 21:
                    tv = tv[-21:]
                else:
                    tv = torch.cat([tv[0:1].repeat(21 - b_tv, 1, 1, 1), tv], dim=0)
                if tv.shape[1] != H or tv.shape[2] != W:
                    tv = common_upscale(tv.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
                tv = tv.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, 21, H, W]
                resized_bg_images[:, :21] = tv.to(device, dtype=resized_bg_images.dtype)
                log.info("Transition: embedded first 21 pixel frames of black canvas")
            # ==========================================================================

            bg_latents = vae.encode([resized_bg_images.to(device, vae.dtype)], device,tiled=tiled_vae)[0].to(offload_device)
            del resized_bg_images
        elif bg_images is not None:
            resized_bg_images = resized_bg_images.to(offload_device, dtype=vae.dtype)
        elif transition_video is not None or prefix_frames is not None:
            # Looping mode: create canvas (transition and/or prefix handled separately via prefix_ctx)
            bg_frame_count = num_frames if wananim_single_frame_prefix else num_frames - num_refs
            resized_bg_images = torch.zeros(3, bg_frame_count, H, W, device=offload_device, dtype=vae.dtype)
            if transition_video is not None:
                tv = transition_video  # [B, H, W, C]
                b_tv = tv.shape[0]
                if legacy_prefix_active:
                    if b_tv >= 20:
                        tv = tv[-20:]
                    else:
                        tv = torch.cat([tv[0:1].repeat(20 - b_tv, 1, 1, 1), tv], dim=0)
                    if tv.shape[1] != H or tv.shape[2] != W:
                        tv = common_upscale(tv.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
                    tv = tv.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, 20, H, W]
                    resized_bg_images[:, prefix_transition_start:prefix_transition_end] = tv.to(offload_device, dtype=resized_bg_images.dtype)
                    log.info(f"Prefix+Transition (loop): embedded last 20 transition frames into canvas positions {prefix_transition_start}-{prefix_transition_end}")
                else:
                    if b_tv >= 21:
                        tv = tv[-21:]
                    else:
                        tv = torch.cat([tv[0:1].repeat(21 - b_tv, 1, 1, 1), tv], dim=0)
                    if tv.shape[1] != H or tv.shape[2] != W:
                        tv = common_upscale(tv.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
                    tv = tv.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, 21, H, W]
                    resized_bg_images[:, :21] = tv.to(offload_device, dtype=resized_bg_images.dtype)
                    log.info("Transition (loop): embedded first 21 pixel frames of canvas")
            # Prefix NOT embedded in canvas for looping — handled via prefix_ctx prepend later

        prefix_ctx = None
        prefix_T = 0
        bg_mask = None
        resized_ref_images = None
        wananim_static_ref_latents = 0
        wananim_main_ref_index = 0

        if ref_images is not None:
            def _prepare_ref_frame(frame_bhwc):
                frame_bhwc = frame_bhwc[:, :, :, :3]
                if frame_bhwc.shape[1] != H or frame_bhwc.shape[2] != W:
                    resized = common_upscale(frame_bhwc.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
                else:
                    resized = frame_bhwc.permute(3, 0, 1, 2)
                return resized[:3] * 2 - 1

            if wananim_single_frame_prefix:
                if ref_images.shape[0] > 1:
                    log.warning("single_frame_prefix_encoding uses the last ref_images frame as the main reference.")
                ref_latent_parts = []
                prefix_ref_images = prefix_frames[:prefix_ref_count, :, :, :3]
                for i in range(prefix_ref_count):
                    prefix_pixels = _prepare_ref_frame(prefix_ref_images[i:i + 1])
                    prefix_latent = vae.encode([prefix_pixels.to(device, vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device)
                    ref_latent_parts.append(prefix_latent)
                main_ref_pixels = _prepare_ref_frame(ref_images[-1:])
                main_ref_latent = vae.encode([main_ref_pixels.to(device, vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device)
                ref_latent_parts.append(main_ref_latent)
                ref_latent = torch.cat(ref_latent_parts, dim=1)
                ref_latent_for_output = ref_latent
                resized_ref_images = main_ref_pixels
                msk = torch.ones(4, ref_latent.shape[1], lat_h, lat_w, device=offload_device, dtype=vae.dtype)
                ref_latent_masked = torch.cat([msk, ref_latent_for_output], dim=0).to(offload_device)
                wananim_static_ref_latents = ref_latent.shape[1]
                wananim_main_ref_index = wananim_static_ref_latents - 1
                log.info(f"WanAnimate single-frame prefix references: {prefix_ref_count} prefix + 1 main = {wananim_static_ref_latents} latents")
                del ref_latent_parts
            else:
                if ref_images.shape[1] != H or ref_images.shape[2] != W:
                    resized_ref_images = common_upscale(ref_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
                else:
                    resized_ref_images = ref_images.permute(3, 0, 1, 2) # C, T, H, W
                resized_ref_images = resized_ref_images[:3] * 2 - 1

                ref_latent = vae.encode([resized_ref_images.to(device, vae.dtype)], device,tiled=tiled_vae)[0]
                msk = torch.zeros(4, 1, lat_h, lat_w, device=device, dtype=vae.dtype)
                msk[:, :num_refs] = 1
                ref_latent_masked = torch.cat([msk, ref_latent], dim=0).to(offload_device) # 4+C 1 H W

            # ============ Prefix: VAE encode for looping (prepended to each chunk like ref) ============
            if legacy_prefix_active and looping:
                vae.to(device)
                prefix_latent = vae.encode([prefix_pixel_data.to(device, vae.dtype)], device, tiled=tiled_vae)[0]
                prefix_T = prefix_latent.shape[1]
                prefix_msk = torch.ones(4, prefix_T, lat_h, lat_w, device=offload_device, dtype=vae.dtype)
                prefix_latent_masked = torch.cat([prefix_msk, prefix_latent.to(offload_device)], dim=0)  # [20, prefix_T, ...]
                prefix_ctx = torch.cat([ref_latent_masked, prefix_latent_masked], dim=1)  # [20, 1+prefix_T, ...]
                log.info(f"Prefix (loop): encoded {actual_prefix_px}px -> {prefix_T} latent, prefix_ctx: {prefix_ctx.shape}")
                if force_offload:
                    vae.to(offload_device)
            # ===========================================================================================

            if mask is None:
                bg_mask = torch.zeros(1, num_frames, lat_h, lat_w, device=offload_device, dtype=vae.dtype)
            else:
                bg_mask = 1 - mask[:num_frames]
                if bg_mask.shape[0] < num_frames and not looping:
                    bg_mask = torch.cat([bg_mask, bg_mask[-1:].repeat(num_frames - bg_mask.shape[0], 1, 1)], dim=0)
                bg_mask = common_upscale(bg_mask.unsqueeze(1), lat_w, lat_h, "nearest", "disabled").squeeze(1)
                bg_mask = bg_mask.unsqueeze(-1).permute(3, 0, 1, 2).to(offload_device, vae.dtype) # C, T, H, W

            # ============ Prefix: set mask=1 for actual prefix frames and optionally transition ============
            if legacy_prefix_active:
                bg_mask[:, :actual_prefix_px] = 1.0  # only actual prefix pixel frames
                if transition_video is not None:
                    bg_mask[:, prefix_transition_start:prefix_transition_end] = 1.0
            # ======= Transition (no prefix): set mask=1 for first 21 pixel frames =======
            elif transition_video is not None:
                bg_mask[:, :21] = 1.0
            # ======================================================================================

            if bg_images is None and looping and not wananim_single_frame_prefix:
                bg_mask[:, :num_refs] = 1
            bg_mask_mask_repeated = torch.repeat_interleave(bg_mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
            bg_mask = torch.cat([bg_mask_mask_repeated, bg_mask[:, 1:]], dim=1)
            bg_mask = bg_mask.view(1, bg_mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
            bg_mask = bg_mask.movedim(1, 2)[0]# C, T, H, W

            if not looping:
                bg_latents_masked = torch.cat([bg_mask[:, :bg_latents.shape[1]], bg_latents], dim=0)
                del bg_mask, bg_latents
                ref_latent = torch.cat([ref_latent_masked, bg_latents_masked], dim=1)
            else:
                ref_latent = ref_latent_masked

        if face_images is not None:
            face_images = face_images[..., :3]
            if face_images.shape[1] != 512 or face_images.shape[2] != 512:
                resized_face_images = common_upscale(face_images.movedim(-1, 1), 512, 512, "lanczos", "center").movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2) # B, C, T, H, W
            resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
            resized_face_images = resized_face_images.to(offload_device, dtype=vae.dtype)

        if start_ref_image is not None:
            if start_ref_image.shape[1] != H or start_ref_image.shape[2] != W:
                resized_start_ref_image = common_upscale(start_ref_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_ref_image = start_ref_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_ref_image = resized_start_ref_image[:3] * 2 - 1

        # ============ Transition video processing ============
        transition_latent = None
        transition_mask_values = None

        if False:  # Transition now embedded in canvas (non-looping) or bg_images (looping), no independent VAE encode needed
            # transition_video input: 32 images [B, H, W, C]
            # Expecting B=32, which encodes to 8 latent frames
            b, h, w, c = transition_video.shape
            log.info(f"Transition video input: {b} frames, {h}x{w}")
            
            # Verify frame count to ensure it is exactly 32 frames
            expected_input_frames = 32
            if b != expected_input_frames:
                log.warning(f"Transition video has {b} frames, expected {expected_input_frames}. Resizing time dimension.")
                if b > expected_input_frames:
                    # Downsample to 32 frames
                    indices = torch.linspace(0, b-1, expected_input_frames).long()
                    transition_video = transition_video[indices]
                else:
                    # Repeat frames to reach 32 frames
                    repeat_factor = math.ceil(expected_input_frames / b)
                    transition_video = transition_video.repeat(repeat_factor, 1, 1, 1)[:expected_input_frames]
            
            b, h, w, c = transition_video.shape  # It should be 32 now
            
            # Adjust spatial dimensions to target WxH.
            # Keep the same semantic flow as the matched-size path:
            # BHWC -> (optional resize in BCHW) -> BHWC -> CTHW -> normalize -> encode
            if h != H or w != W:
                transition_video = common_upscale(
                    transition_video.movedim(-1, 1), W, H, "lanczos", "disabled"
                ).movedim(1, -1)
            
            # Normalize to [-1, 1]
            transition_video = transition_video.permute(3, 0, 1, 2)  # [C, T, H, W]
            transition_video = transition_video[:3] * 2 - 1  # Keep only RGB channels
            
            
            # VAE Encoding (32 pixel frames -> 8 latent frames)
            vae.to(device)
            transition_latent = vae.encode([transition_video.to(device, vae.dtype)], device, tiled=tiled_vae)[0]
            log.info(f"Transition latent encoded: {transition_latent.shape[1] if len(transition_latent.shape) > 1 else transition_latent.shape[0]} frames, shape {transition_latent.shape}")
            transition_len = transition_latent.shape[1]  # It should be 8
            log.info(f"Transition latent encoded: {transition_len} frames, shape {transition_latent.shape}")
            
            # ============ Generate Mask values ============
            # Force mask to all 1s, making it act purely as a hard conditioning guide.
            # The model will strictly follow these frames without altering them.
            transition_mask_values = torch.ones(transition_len)
            
            log.info("Transition mask: forced to all 1s for hard conditioning.")
            log.info(f"Mask values: {transition_mask_values.tolist()}")
            # ==========================================
            
            if force_offload:
                transition_latent = transition_latent.to(offload_device)
                transition_mask_values = transition_mask_values.to(offload_device)
        # ================================================

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / 4 * target_shape[1])
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": seq_len,
            "pose_latents": pose_latents,
            "pose_images": resized_pose_images if pose_images is not None and looping else None,
            "bg_images": resized_bg_images if (bg_images is not None or transition_video is not None or prefix_frames is not None) and looping else None,
            "ref_masks": bg_mask if (mask is not None or legacy_prefix_active or transition_video is not None) and looping else None,
            "is_masked": mask is not None,
            "ref_latent": ref_latent,
            "ref_image": resized_ref_images if ref_images is not None else None,
            "start_ref_image": resized_start_ref_image if start_ref_image is not None else None,
            "transition_latent": transition_latent,
            "transition_mask_values": transition_mask_values,
            "has_prefix": legacy_prefix_active,
            "canvas_expansion_px": prefix_canvas_extra if legacy_prefix_active else (21 if transition_video is not None else 0),
            "prefix_ctx": prefix_ctx,
            "prefix_T": prefix_T,
            "prefix_prepend_latents": 6 if legacy_prefix_active else 0,
            "face_pixels": resized_face_images if face_images is not None else None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "frame_window_size": frame_window_size,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "vae": vae,
            "colormatch": colormatch,
            "looping": looping,
            "pose_strength": pose_strength,
            "face_strength": face_strength,
        }
        if wananim_single_frame_prefix:
            image_embeds.update({
                "wananim_single_frame_prefix_encoding": True,
                "wananim_static_ref_latents": wananim_static_ref_latents,
                "wananim_main_ref_index": wananim_main_ref_index,
                "wananim_decode_ref_latents": wananim_static_ref_latents,
                "wananim_num_anchor_latents": wananim_static_ref_latents,
            })

        return (image_embeds,)


class WanAnimatePlusSCAIL2Embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 32, "tooltip": "Width of the video to generate. SCAIL-2 inputs are aligned to multiples of 32."}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 32, "tooltip": "Height of the video to generate. SCAIL-2 inputs are aligned to multiples of 32."}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to generate"}),
            "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "SCAIL-2 chunk window length. Automatically rounds down to 4n+1. Values different from the normalized num_frames enable built-in loop generation with 5-frame handoff; oversized values are clamped to the largest valid window that fits."}),
            "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload VAE after encoding to save VRAM"}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the SCAIL pose stream"}),
            "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of the SCAIL reference stream"}),
            "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False = animation mode (pose mask black bg, reference mask white bg). True = replacement mode (pose mask white bg, reference mask black bg)."}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "ref_image": ("IMAGE", {"tooltip": "Reference image for SCAIL conditioning. If a sequence is connected, only the first frame is used."}),
                "bg_image": ("IMAGE", {"tooltip": "Optional single background image for animation mode. In single-frame prefix mode it is encoded as an extra background reference latent; in legacy prefix mode it is placed after prefix_frames. Ignored in replacement mode."}),
                "pose_images": ("IMAGE", {"tooltip": "Driving pose video. Encoded at half resolution for SCAIL."}),
                "prefix_frames": ("IMAGE", {"tooltip": "Optional prefix images. In single-frame prefix mode these are encoded as reference latents; in legacy mode they hard-freeze the beginning of the canvas."}),
                "prefix_mask": ("IMAGE", {"tooltip": "Optional colored mask images matching prefix_frames. In single-frame prefix mode this follows the reference-mask path; in legacy canvas-prefix mode it is expanded as 1+4+4... and written into the prefix mask frames."}),
                "transition_video": ("IMAGE", {"tooltip": "Optional transition frames to hard-freeze at the beginning of the canvas. In legacy canvas-prefix mode, transition frames are placed after the prefix frames."}),
                "pose_image_mask": ("IMAGE", {"tooltip": "SCAIL-2 colored per-identity driving pose mask. Background is normalized to black in animation mode and white in replacement mode."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "SCAIL-2 colored per-identity reference mask image. Background is normalized to white in animation mode and black in replacement mode."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "transition_colormatch": ([
                    'disabled',
                    'auto_drift',
                    'mkl',
                    'hm',
                    'reinhard',
                    'mvgd',
                    'hm-mvgd-hm',
                    'hm-mkl-hm',
                ], {"default": 'disabled', "tooltip": "Color match transition_video to ref_image."}),
                "loop_colormatch_reference": ([
                    'previous_matched_frame',
                    'main_ref_image',
                ], {"default": 'previous_matched_frame', "tooltip": "SCAIL-2 loop color match reference. The first chunk is not color matched when transition_video is not connected."}),
                "prefix_alpha_crop": ("BOOLEAN", {"default": False, "tooltip": "Off keeps prefix masks as white-background reference masks in animation mode. On uses black-background masks and alpha-crops prefix_frames. Replacement mode always uses black-background reference masks."}),
                "preserve_main_ref_background": ("BOOLEAN", {"default": True, "tooltip": "Animation mode only. Keep the main reference image background. When off, reference_image_mask is normalized to black background and used to alpha-crop ref_image. Ignored in replacement mode."}),
                "single_frame_prefix_encoding": ("BOOLEAN", {"default": True, "tooltip": "Encode prefix images as individual full-resolution reference latents instead of expanding the canvas."}),
                "by wuwukasi（bilibili）": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "ON", "tooltip": "Follow wuwukasi on bilibili"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    @staticmethod
    def _fit_sequence(seq, target_len):
        if seq is None:
            return None
        if seq.shape[0] == target_len:
            return seq
        if seq.shape[0] > target_len:
            return seq[:target_len]
        if seq.shape[0] == 0:
            raise ValueError("Input sequence must contain at least one frame")
        return torch.cat([seq, seq[-1:].repeat(target_len - seq.shape[0], *([1] * (seq.ndim - 1)))], dim=0)

    @staticmethod
    def _resize_bhwc(images, width, height, mode="lanczos", crop="disabled"):
        images = images[:, :, :, :3]
        if images.shape[1] == height and images.shape[2] == width:
            return images
        chunk_size = WanAnimatePlusSCAIL2Embeds._safe_frame_chunk_size(
            images.shape[0],
            images.shape[1],
            images.shape[2],
            width,
            height,
            3,
        )
        if images.shape[0] <= chunk_size:
            return common_upscale(images.movedim(-1, 1), width, height, mode, crop).movedim(1, -1)
        resized = []
        for start in range(0, images.shape[0], chunk_size):
            chunk = images[start:start + chunk_size]
            resized.append(common_upscale(chunk.movedim(-1, 1), width, height, mode, crop).movedim(1, -1))
        return torch.cat(resized, dim=0)

    @staticmethod
    def _safe_frame_chunk_size(frame_count, in_height, in_width, out_width, out_height, channels):
        # Keep CUDA pooling/interpolate calls well below the int32 element limit.
        element_budget = 64 * 1024 * 1024
        per_frame = max(in_height * in_width * channels, out_height * out_width * channels, 1)
        return max(1, min(frame_count, element_budget // per_frame))

    @staticmethod
    def _to_cthw_pixels(images):
        return images[:, :, :, :3].permute(3, 0, 1, 2) * 2 - 1

    @staticmethod
    def _encode_scail_20ch(vae, images, width, height, tiled_vae, scale=1.0):
        images = WanAnimatePlusSCAIL2Embeds._resize_bhwc(images, width, height)
        pixels = WanAnimatePlusSCAIL2Embeds._to_cthw_pixels(images).to(device, vae.dtype)
        latent = vae.encode([pixels], device, tiled=tiled_vae)[0]
        mask = torch.ones_like(latent[:4])
        if scale != 1.0:
            latent = latent * scale
        return torch.cat([latent, mask], dim=0)

    @staticmethod
    def _encode_freeze_latents(vae, images, width, height, target_latents, tiled_vae, frame_mask=None):
        if images is None or images.shape[0] == 0:
            return None, None
        images = images[:, :, :, :3]
        if images.shape[1] != height or images.shape[2] != width:
            images = WanAnimatePlusSCAIL2Embeds._resize_bhwc(images, width, height)
        pixels = WanAnimatePlusSCAIL2Embeds._to_cthw_pixels(images).to(device, vae.dtype)
        latent = vae.encode([pixels], device, tiled=tiled_vae)[0]
        if latent.shape[1] > target_latents:
            log.warning(f"SCAIL-2 freeze latents longer than target ({latent.shape[1]} > {target_latents}), truncating")
            latent = latent[:, :target_latents]
        if frame_mask is None:
            freeze_mask = torch.ones(latent.shape[1], latent.shape[2], latent.shape[3], device=latent.device, dtype=latent.dtype)
        else:
            frame_mask = frame_mask.to(latent.device, latent.dtype).flatten()
            t_lat = (frame_mask.shape[0] - 1) // 4 + 1
            padded = torch.cat([frame_mask[:1].repeat(4), frame_mask[1:]], dim=0)
            if padded.shape[0] < t_lat * 4:
                padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0])], dim=0)
            latent_mask = padded[:t_lat * 4].view(t_lat, 4).amax(dim=1)[:latent.shape[1]]
            freeze_mask = latent_mask.view(-1, 1, 1).expand(-1, latent.shape[2], latent.shape[3]).contiguous()
        return latent, freeze_mask

    @staticmethod
    def _frame_mask_to_latent_mask(frame_mask, target_latents=None):
        frame_mask = frame_mask.flatten()
        t_lat = (frame_mask.shape[0] - 1) // 4 + 1
        padded = torch.cat([frame_mask[:1].repeat(4), frame_mask[1:]], dim=0)
        if padded.shape[0] < t_lat * 4:
            padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0])], dim=0)
        latent_mask = padded[:t_lat * 4].view(t_lat, 4).amax(dim=1) > 0
        if target_latents is not None:
            if latent_mask.shape[0] < target_latents:
                pad = torch.zeros(target_latents - latent_mask.shape[0], device=latent_mask.device, dtype=torch.bool)
                latent_mask = torch.cat([latent_mask, pad], dim=0)
            elif latent_mask.shape[0] > target_latents:
                latent_mask = latent_mask[:target_latents]
        return latent_mask

    @staticmethod
    def _take_tail_with_front_pad(images, count):
        if images.shape[0] >= count:
            return images[-count:]
        return torch.cat([images[:1].repeat(count - images.shape[0], 1, 1, 1), images], dim=0)

    @staticmethod
    def _build_prefix_pixels(prefix_frames):
        pf = prefix_frames[:, :, :, :3]
        if pf.shape[0] > 5:
            log.warning(f"SCAIL-2 prefix has {pf.shape[0]} images, max 5. Truncating.")
            pf = pf[:5]
        frames = pf[0:1]
        for i in range(1, pf.shape[0]):
            frames = torch.cat([frames, pf[i:i + 1].repeat(4, 1, 1, 1)], dim=0)
        return frames

    @staticmethod
    def _normalize_mask_background(mask, white_background):
        mask = mask[:, :, :, :3]
        if white_background:
            bg = mask.amax(dim=-1, keepdim=True) <= 0.05
            return torch.where(bg, torch.ones_like(mask), mask)
        bg = mask.amin(dim=-1, keepdim=True) >= 0.95
        return torch.where(bg, torch.zeros_like(mask), mask)

    @staticmethod
    def _alpha_crop_with_mask(images, masks):
        if images is None or masks is None or images.shape[0] == 0 or masks.shape[0] == 0:
            return images
        count = min(images.shape[0], masks.shape[0])
        crop_mask = masks[:count, :, :, :3]
        if crop_mask.shape[1] != images.shape[1] or crop_mask.shape[2] != images.shape[2]:
            crop_mask = WanAnimatePlusSCAIL2Embeds._resize_bhwc(crop_mask, images.shape[2], images.shape[1], mode="nearest-exact")
        crop_mask = WanAnimatePlusSCAIL2Embeds._normalize_mask_background(crop_mask, white_background=False)
        is_char = (crop_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(device=images.device, dtype=images.dtype)
        out = images.clone()
        out[:count] = out[:count] * is_char
        return out

    @staticmethod
    def _color_match_frames(frames, ref_frame, method):
        if method in ("disabled", "auto_drift") or ref_frame is None or frames is None or frames.shape[0] == 0:
            return frames
        from color_matcher import ColorMatcher
        cm = ColorMatcher()
        ref_np = ref_frame[:1, :, :, :3].detach().cpu().float().numpy()[0]
        matched = []
        for frame in frames[:, :, :, :3]:
            out = cm.transfer(src=frame.detach().cpu().float().numpy(), ref=ref_np, method=method)
            matched.append(torch.from_numpy(out).to(device=frames.device, dtype=frames.dtype))
        return torch.stack(matched, dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _frame_rgb_means_bhwc(frames, max_frames=5):
        if frames is None or frames.shape[0] == 0:
            return None
        count = min(max(1, int(max_frames)), int(frames.shape[0]))
        tail = frames[-count:, :, :, :3].detach().float().clamp(0.0, 1.0)
        if tail.numel() == 0:
            return None
        return tail.mean(dim=(1, 2)).cpu()

    @staticmethod
    def _extract_mask_to_28ch(rgb_video):
        # Colored RGB mask (T,H,W,3) in [0,1] -> [28,T_lat,H/8,W/8].
        T, H, W, _ = rgb_video.shape
        on_thresh = 225.0 / 255.0
        h_lat, w_lat = H, W
        for _ in range(3):
            h_lat = (h_lat + 1) // 2
            w_lat = (w_lat + 1) // 2

        chunk_size = WanAnimatePlusSCAIL2Embeds._safe_frame_chunk_size(T, H, W, w_lat, h_lat, 7)
        binary_chunks = []
        for start in range(0, T, chunk_size):
            mask = rgb_video[start:start + chunk_size, :, :, :3].movedim(-1, 1).float()
            r = (mask[:, 0:1] > on_thresh).float()
            g = (mask[:, 1:2] > on_thresh).float()
            b = (mask[:, 2:3] > on_thresh).float()
            nr, ng, nb = 1 - r, 1 - g, 1 - b
            binary_7ch = torch.cat([
                r * g * b,      # white
                r * ng * nb,    # red
                nr * g * nb,    # green
                nr * ng * b,    # blue
                r * g * nb,     # yellow
                r * ng * b,     # magenta
                nr * g * b,     # cyan
            ], dim=1)
            binary_chunks.append(F.interpolate(binary_7ch, size=(h_lat, w_lat), mode="area"))
        binary_7ch = torch.cat(binary_chunks, dim=0)

        t_lat = (T - 1) // 4 + 1
        padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
        if padded.shape[0] < t_lat * 4:
            padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0], 1, 1, 1)], dim=0)
        padded = padded[:t_lat * 4]
        return padded.view(t_lat, 28, h_lat, w_lat).movedim(0, 1).contiguous()

    @staticmethod
    def _empty_ref_mask_like(ref_latent):
        return torch.zeros(
            28, ref_latent.shape[1], ref_latent.shape[-2], ref_latent.shape[-1],
            device=ref_latent.device, dtype=ref_latent.dtype,
        )

    def process(self, vae, width, height, num_frames, frame_window_size, force_offload, pose_strength, ref_strength,
                replacement_mode, clip_embeds=None, ref_image=None, bg_image=None, pose_images=None, prefix_frames=None, prefix_mask=None,
                transition_video=None, pose_image_mask=None, reference_image_mask=None, tiled_vae=False,
                transition_colormatch='disabled', prefix_alpha_crop=False, preserve_main_ref_background=True,
                single_frame_prefix_encoding=True, loop_colormatch_reference='previous_matched_frame', **kwargs):
        W = (width // 32) * 32
        H = (height // 32) * 32
        raw_num_frames = max(1, int(num_frames))
        raw_frame_window_size = max(1, int(frame_window_size))

        def _align_4n_plus_1(frames):
            return ((frames - 1) // 4) * 4 + 1

        def _align_4n(frames):
            return max(4, (frames // 4) * 4)

        def _clamp_window_to_requested(window_size, requested_size):
            return min(window_size, _align_4n_plus_1(requested_size))

        legacy_canvas_encoding = (
            not bool(single_frame_prefix_encoding)
            and (prefix_frames is not None or transition_video is not None)
        )
        frame_window_size = _align_4n_plus_1(raw_frame_window_size)
        if legacy_canvas_encoding:
            requested_frames = _align_4n(raw_num_frames)
        else:
            requested_frames = _align_4n_plus_1(raw_num_frames)

        frame_window_size = _clamp_window_to_requested(frame_window_size, requested_frames)
        scail2_looping = frame_window_size != requested_frames
        if scail2_looping and not single_frame_prefix_encoding:
            log.info("SCAIL-2 loop mode forces single_frame_prefix_encoding on; legacy canvas prefix is not used for loop handoff.")
            single_frame_prefix_encoding = True
            requested_frames = _align_4n_plus_1(raw_num_frames)
            frame_window_size = _align_4n_plus_1(raw_frame_window_size)
            frame_window_size = _clamp_window_to_requested(frame_window_size, requested_frames)
            scail2_looping = frame_window_size != requested_frames
        num_frames = requested_frames
        bg_prefix_mask_pixel_frames = 0
        bg_prefix_mask_index = None
        crop_main_ref_background = (not replacement_mode) and (not preserve_main_ref_background)

        user_prefix_frames = prefix_frames
        user_prefix_mask = prefix_mask if user_prefix_frames is not None else None
        bg_prefix_active = bg_image is not None and not replacement_mode
        if bg_image is not None and replacement_mode:
            log.info("SCAIL-2 bg_image is ignored in replacement mode")

        bg_prefix_image = None
        bg_mask = None
        user_prefix_count = 0
        if bg_prefix_active:
            if bg_image.shape[0] > 1:
                log.warning("SCAIL-2 bg_image accepts one image; using the first frame")
            bg_prefix_image = bg_image[:1, :, :, :3]

            if user_prefix_frames is not None:
                if user_prefix_frames.shape[0] > 4:
                    log.warning("SCAIL-2 bg_image uses one prefix slot; truncating prefix_frames to 4 images")
                user_prefix_frames = user_prefix_frames[:4, :, :, :3]
                user_prefix_count = user_prefix_frames.shape[0]
                if bg_prefix_image.shape[1] != user_prefix_frames.shape[1] or bg_prefix_image.shape[2] != user_prefix_frames.shape[2]:
                    bg_prefix_image = self._resize_bhwc(bg_prefix_image, user_prefix_frames.shape[2], user_prefix_frames.shape[1])
                bg_prefix_image = bg_prefix_image.to(device=user_prefix_frames.device, dtype=user_prefix_frames.dtype)

            if user_prefix_mask is not None:
                if user_prefix_mask.shape[0] > 4:
                    log.warning("SCAIL-2 bg_image uses one prefix-mask slot; truncating prefix_mask to 4 images")
                user_prefix_mask = user_prefix_mask[:4, :, :, :3]
                bg_mask = torch.ones(
                    1, user_prefix_mask.shape[1], user_prefix_mask.shape[2], 3,
                    device=user_prefix_mask.device, dtype=user_prefix_mask.dtype,
                )
            else:
                bg_mask = torch.ones_like(bg_prefix_image[:, :, :, :3])

        if user_prefix_frames is not None and user_prefix_mask is not None and (replacement_mode or prefix_alpha_crop):
            crop_count = min(user_prefix_frames.shape[0], user_prefix_mask.shape[0], 5)
            if crop_count > 0:
                cropped_prefix_frames = user_prefix_frames.clone()
                cropped_prefix_frames[:crop_count] = self._alpha_crop_with_mask(
                    cropped_prefix_frames[:crop_count],
                    user_prefix_mask[:crop_count],
                )
                user_prefix_frames = cropped_prefix_frames
                # Composite cropped prefix frames onto bg when bg + prefix_alpha_crop + crop_main_ref_bg are all active
                if bg_prefix_active and crop_main_ref_background:
                    comp_bg = bg_prefix_image
                    if comp_bg.shape[1] != user_prefix_frames.shape[1] or comp_bg.shape[2] != user_prefix_frames.shape[2]:
                        comp_bg = self._resize_bhwc(comp_bg, user_prefix_frames.shape[2], user_prefix_frames.shape[1])
                    comp_bg = comp_bg.to(device=user_prefix_frames.device, dtype=user_prefix_frames.dtype)
                    for ci in range(crop_count):
                        mi = user_prefix_mask[ci:ci + 1, :, :, :3]
                        mi = self._normalize_mask_background(mi, white_background=False)
                        is_bg = (mi[..., :3].max(dim=-1, keepdim=True).values <= 0.1).to(
                            device=user_prefix_frames.device, dtype=user_prefix_frames.dtype
                        )
                        user_prefix_frames[ci:ci + 1] = user_prefix_frames[ci:ci + 1] + comp_bg * is_bg
                    log.info("SCAIL-2 prefix frames composited onto bg; masks will be inverted")

        prefix_frames = user_prefix_frames
        prefix_mask_for_prefix = user_prefix_mask
        if bg_prefix_active:
            if single_frame_prefix_encoding:
                if user_prefix_mask is not None:
                    if user_prefix_mask.shape[0] < user_prefix_count:
                        empty_user_masks = torch.zeros(
                            user_prefix_count - user_prefix_mask.shape[0],
                            user_prefix_mask.shape[1], user_prefix_mask.shape[2], 3,
                            device=user_prefix_mask.device, dtype=user_prefix_mask.dtype,
                        )
                        user_prefix_mask = torch.cat([user_prefix_mask, empty_user_masks], dim=0)
                    elif user_prefix_mask.shape[0] > user_prefix_count:
                        user_prefix_mask = user_prefix_mask[:user_prefix_count]
                    prefix_mask_for_prefix = user_prefix_mask
                elif user_prefix_count > 0:
                    prefix_mask_for_prefix = torch.zeros(
                        user_prefix_count, bg_mask.shape[1], bg_mask.shape[2], 3,
                        device=bg_mask.device, dtype=bg_mask.dtype,
                    )
                else:
                    prefix_mask_for_prefix = None
            else:
                if user_prefix_frames is not None:
                    prefix_frames = torch.cat([user_prefix_frames, bg_prefix_image], dim=0)
                else:
                    prefix_frames = bg_prefix_image
                if user_prefix_mask is not None:
                    if user_prefix_mask.shape[0] < user_prefix_count:
                        empty_user_masks = torch.zeros(
                            user_prefix_count - user_prefix_mask.shape[0],
                            user_prefix_mask.shape[1], user_prefix_mask.shape[2], 3,
                            device=user_prefix_mask.device, dtype=user_prefix_mask.dtype,
                        )
                        user_prefix_mask = torch.cat([user_prefix_mask, empty_user_masks], dim=0)
                    elif user_prefix_mask.shape[0] > user_prefix_count:
                        user_prefix_mask = user_prefix_mask[:user_prefix_count]
                    prefix_mask_for_prefix = torch.cat([user_prefix_mask, bg_mask], dim=0)
                elif user_prefix_count > 0:
                    empty_user_masks = torch.zeros(
                        user_prefix_count, bg_mask.shape[1], bg_mask.shape[2], 3,
                        device=bg_mask.device, dtype=bg_mask.dtype,
                    )
                    prefix_mask_for_prefix = torch.cat([empty_user_masks, bg_mask], dim=0)
                else:
                    prefix_mask_for_prefix = bg_mask
                bg_prefix_mask_pixel_frames = 1 if user_prefix_count == 0 else 4
                bg_prefix_mask_index = prefix_frames.shape[0] - 1

        canvas_expansion_px = 0
        freeze_canvas = None
        canvas_prefix_frames = prefix_frames if not single_frame_prefix_encoding else None
        if canvas_prefix_frames is not None:
            canvas_expansion_px = 37
        elif transition_video is not None:
            canvas_expansion_px = 21
        transition_px_range = None
        if transition_video is not None:
            transition_px_range = (17, 37) if canvas_prefix_frames is not None else (0, 21)
        transition_match_ref = self._resize_bhwc(ref_image[:1, :, :, :3], W, H) if ref_image is not None else None
        transition_raw_last_frame = (
            transition_video[-1:, :, :, :3].detach().to(offload_device)
            if transition_video is not None and scail2_looping and transition_colormatch != "auto_drift" else None
        )
        transition_raw_tail_means = None
        if transition_colormatch not in ("disabled", "auto_drift") and transition_match_ref is None and (
            transition_video is not None or (scail2_looping and loop_colormatch_reference == "main_ref_image")
        ):
            log.warning("SCAIL-2 transition_colormatch is enabled but ref_image is not connected. Skipping color match.")

        if canvas_expansion_px:
            num_frames += canvas_expansion_px
            trim = (num_frames - 1) % 4
            if trim:
                if scail2_looping:
                    num_frames += 4 - trim
                else:
                    num_frames -= trim
            if pose_images is not None:
                pose_images = torch.cat([_sample_reversed_prefix_frames(pose_images, canvas_expansion_px), pose_images], dim=0)
            if pose_image_mask is not None:
                pose_image_mask = torch.cat([_sample_reversed_prefix_frames(pose_image_mask, canvas_expansion_px), pose_image_mask], dim=0)

        prefix_mask_pixels = prefix_mask_frame_mask = scail_sam_keep_mask = scail_transition_keep_mask = None
        if prefix_mask_for_prefix is not None:
            if prefix_frames is None:
                log.warning("SCAIL-2 prefix_mask was provided without prefix_frames. Ignoring prefix_mask.")
            else:
                prefix_count = min(prefix_frames.shape[0], 5)
                if prefix_mask_for_prefix.shape[0] > prefix_count:
                    log.warning(f"SCAIL-2 prefix_mask has {prefix_mask_for_prefix.shape[0]} images, but prefix_frames has {prefix_count}. Truncating prefix_mask.")
                mask_count = min(prefix_mask_for_prefix.shape[0], prefix_count)
                if mask_count > 0:
                    prefix_mask_pixels = self._build_prefix_pixels(prefix_mask_for_prefix[:mask_count])

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor
        target_latents = (num_frames - 1) // 4 + 1

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        scail_embeds = {
            "pose_strength": pose_strength,
            "pose_start_percent": 0.0,
            "pose_end_percent": 1.0,
            "ref_start_percent": 0.0,
            "ref_end_percent": 1.0,
            "ref_mask_flag": not replacement_mode,
        }

        ref_latents = []
        ref_masks = []
        ref_mask_condition_used = False
        prefix_ref_count = 0
        prefix_ref_black_background = replacement_mode or prefix_alpha_crop
        if single_frame_prefix_encoding and prefix_frames is not None:
            prefix_ref_images = prefix_frames[:5, :, :, :3]
            prefix_ref_source_indices = list(range(prefix_ref_images.shape[0]))
            prefix_ref_count = prefix_ref_images.shape[0]
            prefix_ref_masks = [None] * prefix_ref_count
            if prefix_mask_for_prefix is not None:
                for out_idx, src_idx in enumerate(prefix_ref_source_indices):
                    if src_idx >= prefix_mask_for_prefix.shape[0]:
                        continue
                    mask = prefix_mask_for_prefix[src_idx:src_idx + 1, :, :, :3]
                    if src_idx != bg_prefix_mask_index:
                        mask = self._normalize_mask_background(mask, white_background=not prefix_ref_black_background)
                        # Invert mask when bg compositing is active: black bg->white, keep character colors
                        if bg_prefix_active and crop_main_ref_background:
                            is_bg = mask.amax(dim=-1, keepdim=True) <= 0.1
                            mask = torch.where(is_bg, torch.ones_like(mask), mask)
                    prefix_ref_masks[out_idx] = mask
                ref_mask_condition_used = any(mask is not None for mask in prefix_ref_masks)
            for i in range(prefix_ref_images.shape[0]):
                prefix_latent = self._encode_scail_20ch(vae, prefix_ref_images[i:i + 1], W, H, tiled_vae, scale=ref_strength).to(offload_device)
                ref_latents.append(prefix_latent)
                if i < len(prefix_ref_masks) and prefix_ref_masks[i] is not None:
                    ref_mask = self._resize_bhwc(prefix_ref_masks[i], W, H, mode="bicubic")
                    ref_masks.append(self._extract_mask_to_28ch(ref_mask).to(offload_device, vae.dtype))
                else:
                    ref_masks.append(self._empty_ref_mask_like(prefix_latent))
            log.info(f"SCAIL-2 prefix reference latents: {prefix_ref_count}")

        if ref_image is not None:
            ref_image = ref_image[:1, :, :, :3]
            if (replacement_mode or crop_main_ref_background) and reference_image_mask is not None:
                ref_mask = self._resize_bhwc(reference_image_mask[:1, :, :, :3], W, H, mode="nearest-exact")
                ref_mask = self._normalize_mask_background(ref_mask, white_background=False)
                is_char = (ref_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_image.dtype)
                ref_image = self._resize_bhwc(ref_image, W, H)
                is_char = is_char.to(device=ref_image.device, dtype=ref_image.dtype)
                ref_image = ref_image * is_char
                # Composite ref foreground onto bg
                if bg_prefix_active and crop_main_ref_background:
                    comp_bg = bg_prefix_image
                    if comp_bg.shape[1] != W or comp_bg.shape[2] != H:
                        comp_bg = self._resize_bhwc(comp_bg, W, H)
                    comp_bg = comp_bg.to(device=ref_image.device, dtype=ref_image.dtype)
                    bg_is_char = 1.0 - is_char
                    ref_image = ref_image + comp_bg * bg_is_char
                    log.info("SCAIL-2 ref image composited onto bg")
            ref_latent = self._encode_scail_20ch(vae, ref_image, W, H, tiled_vae, scale=ref_strength).to(offload_device)
            ref_latents.append(ref_latent)
            if reference_image_mask is not None:
                ref_mask_condition_used = True
                ref_mask = self._fit_sequence(reference_image_mask[:, :, :, :3], 1)
                ref_mask = self._normalize_mask_background(ref_mask, white_background=not (replacement_mode or crop_main_ref_background))
                # Invert mask when bg compositing is active: black bg->white, keep character colors
                if bg_prefix_active and crop_main_ref_background:
                    is_bg = ref_mask.amax(dim=-1, keepdim=True) <= 0.1
                    ref_mask = torch.where(is_bg, torch.ones_like(ref_mask), ref_mask)
                ref_mask = self._resize_bhwc(ref_mask, W, H, mode="bicubic")
                ref_masks.append(self._extract_mask_to_28ch(ref_mask).to(offload_device, vae.dtype))
            else:
                ref_masks.append(self._empty_ref_mask_like(ref_latent))
            log.info(f"SCAIL-2 reference latent shape: {ref_latent.shape}")

        if bg_prefix_active and single_frame_prefix_encoding:
            bg_ref_latent = self._encode_scail_20ch(vae, bg_prefix_image, W, H, tiled_vae, scale=ref_strength).to(offload_device)
            ref_latents.append(bg_ref_latent)
            bg_ref_mask = self._resize_bhwc(bg_mask, W, H, mode="bicubic")
            ref_masks.append(self._extract_mask_to_28ch(bg_ref_mask).to(offload_device, vae.dtype))
            ref_mask_condition_used = True
            log.info(f"SCAIL-2 bg reference latent shape: {bg_ref_latent.shape}")

        if ref_latents:
            ref_latent = torch.cat(ref_latents, dim=1)
            scail_embeds["ref_latent_pos"] = ref_latent
            scail_embeds["ref_latent_neg"] = ref_latent
            scail_embeds["ref_prefix_latents"] = prefix_ref_count
            log.info(f"SCAIL-2 combined reference latent shape: {ref_latent.shape}")

        scail_freeze_latents = scail_freeze_mask = scail_condition_zero_mask = None
        actual_prefix_px = 0
        if canvas_expansion_px:
            freeze_canvas = torch.zeros(canvas_expansion_px, H, W, 3, device=device, dtype=vae.dtype)
            freeze_frame_mask = torch.zeros(canvas_expansion_px, device=device, dtype=vae.dtype)
            if canvas_prefix_frames is not None:
                prefix_pixels = self._build_prefix_pixels(canvas_prefix_frames)
                actual_prefix_px = min(prefix_pixels.shape[0], 17)
                prefix_pixels = self._resize_bhwc(prefix_pixels[:actual_prefix_px], W, H)
                freeze_canvas[:actual_prefix_px] = prefix_pixels.to(device, dtype=freeze_canvas.dtype)
                freeze_frame_mask[:actual_prefix_px] = 1.0
                if transition_video is not None:
                    tv = self._take_tail_with_front_pad(transition_video[:, :, :, :3], 20)
                    tv = self._resize_bhwc(tv, W, H)
                    if scail2_looping and transition_colormatch == "auto_drift":
                        transition_raw_tail_means = self._frame_rgb_means_bhwc(tv, 5)
                    tv = self._color_match_frames(tv, transition_match_ref, transition_colormatch)
                    freeze_canvas[17:37] = tv.to(device, dtype=freeze_canvas.dtype)
                    freeze_frame_mask[17:37] = 1.0
            elif transition_video is not None:
                tv = self._take_tail_with_front_pad(transition_video[:, :, :, :3], 21)
                tv = self._resize_bhwc(tv, W, H)
                if scail2_looping and transition_colormatch == "auto_drift":
                    transition_raw_tail_means = self._frame_rgb_means_bhwc(tv, 5)
                tv = self._color_match_frames(tv, transition_match_ref, transition_colormatch)
                freeze_canvas[:21] = tv.to(device, dtype=freeze_canvas.dtype)
                freeze_frame_mask[:21] = 1.0
            scail_freeze_latents, scail_freeze_mask = self._encode_freeze_latents(
                vae, freeze_canvas, W, H, target_latents, tiled_vae, frame_mask=freeze_frame_mask
            )
            scail_freeze_latents = scail_freeze_latents.to(offload_device)
            scail_freeze_mask = scail_freeze_mask.to(offload_device)
            scail_condition_zero_mask = (scail_freeze_mask > 0).any(dim=(1, 2))
            log.info(f"SCAIL-2 freeze latents shape: {scail_freeze_latents.shape}")
            if canvas_prefix_frames is not None and actual_prefix_px > 0:
                prefix_frame_mask = torch.zeros(canvas_expansion_px, device=device, dtype=vae.dtype)
                prefix_frame_mask[:actual_prefix_px] = 1.0
                scail_prefix_prepend_latents = int(self._frame_mask_to_latent_mask(prefix_frame_mask, target_latents).sum().item())
            else:
                scail_prefix_prepend_latents = 0
        else:
            scail_prefix_prepend_latents = 0

        if transition_px_range is not None:
            trans_start, trans_end = transition_px_range
            trans_end = min(trans_end, num_frames)
            if trans_start < trans_end:
                transition_frame_mask = torch.zeros(num_frames, device=device, dtype=vae.dtype)
                transition_frame_mask[trans_start:trans_end] = 1
                scail_transition_keep_mask = self._frame_mask_to_latent_mask(transition_frame_mask, target_latents)

        scail2_pose_pixels = None
        scail2_pose_mask_pixels = None
        if pose_images is not None:
            pose_images = self._fit_sequence(pose_images[:, :, :, :3], num_frames)
            if scail2_looping:
                scail2_pose_pixels = self._resize_bhwc(pose_images, W // 2, H // 2).to(offload_device)
            else:
                pose_latent = self._encode_scail_20ch(vae, pose_images, W // 2, H // 2, tiled_vae, scale=pose_strength).to(offload_device)
                if scail_condition_zero_mask is not None:
                    target_len = pose_latent.shape[1]
                    zero_mask = torch.zeros(target_len, dtype=torch.bool, device=pose_latent.device)
                    copy_len = min(len(scail_condition_zero_mask), target_len)
                    zero_mask[:copy_len] = scail_condition_zero_mask[:copy_len].to(device=pose_latent.device, dtype=torch.bool)
                    if scail_transition_keep_mask is not None:
                        keep_len = min(len(scail_transition_keep_mask), target_len)
                        keep_mask = torch.zeros(target_len, dtype=torch.bool, device=pose_latent.device)
                        keep_mask[:keep_len] = scail_transition_keep_mask[:keep_len].to(device=pose_latent.device, dtype=torch.bool)
                        zero_mask &= ~keep_mask
                    pose_latent[:, zero_mask] = 0
                scail_embeds["pose_latent"] = pose_latent
                log.info(f"SCAIL-2 pose latent shape: {pose_latent.shape}")

        if pose_image_mask is not None:
            pose_image_mask = self._fit_sequence(pose_image_mask[:, :, :, :3], num_frames)
            pose_image_mask = self._normalize_mask_background(pose_image_mask, white_background=replacement_mode)
        elif prefix_mask_pixels is not None and not single_frame_prefix_encoding:
            pose_bg = 1.0 if replacement_mode else 0.0
            pose_image_mask = torch.full(
                (
                num_frames, prefix_mask_pixels.shape[1], prefix_mask_pixels.shape[2], 3,
                ),
                pose_bg, device=prefix_mask_pixels.device, dtype=prefix_mask_pixels.dtype,
            )

        if pose_image_mask is not None and prefix_mask_pixels is not None and not single_frame_prefix_encoding:
            prefix_mask_len = min(prefix_mask_pixels.shape[0], pose_image_mask.shape[0])
            prefix_mask_in = prefix_mask_pixels[:prefix_mask_len, :, :, :3]
            if prefix_mask_in.shape[1] != pose_image_mask.shape[1] or prefix_mask_in.shape[2] != pose_image_mask.shape[2]:
                prefix_mask_in = self._resize_bhwc(prefix_mask_in, pose_image_mask.shape[2], pose_image_mask.shape[1], mode="nearest-exact")
            bg_keep = min(bg_prefix_mask_pixel_frames, prefix_mask_in.shape[0])
            user_prefix_end = prefix_mask_in.shape[0] - bg_keep
            if user_prefix_end > 0:
                user_prefix_mask_in = self._normalize_mask_background(
                    prefix_mask_in[:user_prefix_end],
                    white_background=not replacement_mode and not prefix_alpha_crop,
                )
                if bg_keep > 0:
                    prefix_mask_in = torch.cat([user_prefix_mask_in, prefix_mask_in[user_prefix_end:]], dim=0)
                else:
                    prefix_mask_in = user_prefix_mask_in
            pose_image_mask = pose_image_mask.clone()
            pose_image_mask[:prefix_mask_len] = prefix_mask_in.to(device=pose_image_mask.device, dtype=pose_image_mask.dtype)
            prefix_mask_frame_mask = torch.zeros(num_frames, device=pose_image_mask.device, dtype=pose_image_mask.dtype)
            prefix_mask_frame_mask[:prefix_mask_len] = 1
            scail_sam_keep_mask = self._frame_mask_to_latent_mask(prefix_mask_frame_mask, target_latents)

        if pose_image_mask is not None:
            mask_video = self._resize_bhwc(pose_image_mask, W // 2, H // 2, mode="area")
            if scail2_looping:
                scail2_pose_mask_pixels = mask_video.to(offload_device)
            else:
                sam_latents = self._extract_mask_to_28ch(mask_video).to(offload_device, vae.dtype)
                if scail_condition_zero_mask is not None:
                    target_len = sam_latents.shape[1]
                    zero_mask = torch.zeros(target_len, dtype=torch.bool, device=sam_latents.device)
                    copy_len = min(len(scail_condition_zero_mask), target_len)
                    zero_mask[:copy_len] = scail_condition_zero_mask[:copy_len].to(device=sam_latents.device, dtype=torch.bool)
                    if scail_sam_keep_mask is not None:
                        keep_len = min(len(scail_sam_keep_mask), target_len)
                        keep_mask = torch.zeros(target_len, dtype=torch.bool, device=sam_latents.device)
                        keep_mask[:keep_len] = scail_sam_keep_mask[:keep_len].to(device=sam_latents.device, dtype=torch.bool)
                        zero_mask &= ~keep_mask
                    if scail_transition_keep_mask is not None:
                        keep_len = min(len(scail_transition_keep_mask), target_len)
                        keep_mask = torch.zeros(target_len, dtype=torch.bool, device=sam_latents.device)
                        keep_mask[:keep_len] = scail_transition_keep_mask[:keep_len].to(device=sam_latents.device, dtype=torch.bool)
                        zero_mask &= ~keep_mask
                    sam_latents[:, zero_mask] = 0
                scail_embeds["sam_latents"] = sam_latents
                log.info(f"SCAIL-2 driving mask latents shape: {sam_latents.shape}")

        if ref_masks and ref_mask_condition_used:
            ref_mask_prefix = torch.cat(ref_masks, dim=1)
            zeros = torch.zeros(
                28, target_latents, ref_mask_prefix.shape[-2], ref_mask_prefix.shape[-1],
                device=offload_device, dtype=ref_mask_prefix.dtype,
            )
            scail_embeds["ref_mask_latents"] = torch.cat([ref_mask_prefix, zeros], dim=1)
            log.info(f"SCAIL-2 reference mask latents shape: {scail_embeds['ref_mask_latents'].shape}")

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        window_latents = (frame_window_size - 1) // 4 + 1
        target_shape = (16, window_latents if scail2_looping else target_latents, lat_h, lat_w)
        image_embeds = {
            "target_shape": target_shape,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": math.ceil((lat_h * lat_w) / 4 * target_shape[1]),
            "num_frames": num_frames,
            "vae": vae,
            "tiled_vae": tiled_vae,
            "scail2_requested_frames": requested_frames,
            "frame_window_size": frame_window_size,
            "scail2_frame_window_size": frame_window_size,
            "scail2_looping": scail2_looping,
            "scail2_previous_frame_count": 5,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "scail_embeds": scail_embeds,
            "canvas_expansion_px": canvas_expansion_px,
            "scail_prefix_prepend_latents": scail_prefix_prepend_latents,
            "scail2_transition_colormatch": transition_colormatch,
            "scail2_loop_colormatch_reference": loop_colormatch_reference,
            "scail2_has_transition_video": transition_video is not None,
        }
        if transition_match_ref is not None:
            image_embeds["scail2_transition_match_ref"] = transition_match_ref.to(offload_device)
        if transition_raw_last_frame is not None:
            image_embeds["scail2_transition_raw_last_frame"] = transition_raw_last_frame
        if transition_raw_tail_means is not None:
            image_embeds["scail2_transition_raw_tail_means"] = transition_raw_tail_means
        if scail_freeze_latents is not None:
            image_embeds["scail_freeze_latents"] = scail_freeze_latents
            image_embeds["scail_freeze_mask"] = scail_freeze_mask
        if scail_condition_zero_mask is not None:
            image_embeds["scail_condition_zero_mask"] = scail_condition_zero_mask
        if scail_sam_keep_mask is not None:
            image_embeds["scail_sam_keep_mask"] = scail_sam_keep_mask
        if scail_transition_keep_mask is not None:
            image_embeds["scail_transition_keep_mask"] = scail_transition_keep_mask
        if scail2_pose_pixels is not None:
            image_embeds["scail2_pose_pixels"] = scail2_pose_pixels
        if scail2_pose_mask_pixels is not None:
            image_embeds["scail2_pose_mask_pixels"] = scail2_pose_mask_pixels
        return (image_embeds,)


class WanAnimatePlusSCAIL2TwoPhaseSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image_embeds": ("WANVIDIMAGE_EMBEDS",),
            "phase1_mask": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                "tooltip": "SCAIL-2 loop handoff protection for phase 1. 1=freeze/protect, 0=free."}),
            "phase2_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                "tooltip": "SCAIL-2 loop handoff protection for phase 2. 1=freeze/protect, 0=free."}),
            "phase2_start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1,
                "tooltip": "Chunk-local step where phase 2 starts. 0 disables two-phase sampling."}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"

    def process(self, image_embeds, phase1_mask, phase2_mask, phase2_start_step):
        updated = dict(image_embeds)
        updated["scail2_two_phase"] = True
        updated["scail2_two_phase_phase1_mask"] = float(phase1_mask)
        updated["scail2_two_phase_phase2_mask"] = float(phase2_mask)
        updated["scail2_two_phase_start_step"] = int(phase2_start_step)
        return (updated,)


class WanAnimatePlusSCAIL2FlowEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 32}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
                "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "replacement_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "ref_image": ("IMAGE",),
                "bg_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "prefix_frames": ("IMAGE",),
                "prefix_mask": ("IMAGE",),
                "transition_video": ("IMAGE",),
                "pose_image_mask": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "transition_colormatch": ([
                    "disabled",
                    "auto_drift",
                    "mkl",
                    "hm",
                    "reinhard",
                    "mvgd",
                    "hm-mvgd-hm",
                    "hm-mkl-hm",
                ], {"default": "disabled"}),
                "loop_colormatch_reference": ([
                    "previous_matched_frame",
                    "main_ref_image",
                ], {"default": "previous_matched_frame"}),
                "prefix_alpha_crop": ("BOOLEAN", {"default": False}),
                "preserve_main_ref_background": ("BOOLEAN", {"default": True}),
                "by wuwukasi(bilibili)": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "ON", "tooltip": "Follow wuwukasi on bilibili"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = "Official ComfyUI-compatible SCAIL-2 conditioning bundle. It outputs CONDITIONING and LATENT and does not depend on legacy WanAnimatePlus node types."

    @staticmethod
    def _flow_input_summary(label, value):
        if value is None:
            return f"{label}=none"
        if isinstance(value, torch.Tensor):
            if value.ndim >= 4:
                return f"{label}={value.shape[0]}@{value.shape[2]}x{value.shape[1]}"
            return f"{label}=shape{tuple(value.shape)}"
        return f"{label}=yes"

    def process(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        num_frames,
        frame_window_size,
        batch_size,
        pose_strength,
        ref_strength,
        replacement_mode,
        clip_vision_output=None,
        ref_image=None,
        bg_image=None,
        pose_images=None,
        prefix_frames=None,
        prefix_mask=None,
        transition_video=None,
        pose_image_mask=None,
        reference_image_mask=None,
        tiled_vae=False,
        transition_colormatch="disabled",
        loop_colormatch_reference="previous_matched_frame",
        prefix_alpha_crop=False,
        preserve_main_ref_background=True,
        **kwargs,
    ):
        runtime = make_runtime(
            width,
            height,
            num_frames,
            frame_window_size,
            batch_size,
            pose_strength,
            ref_strength,
            replacement_mode,
            tiled_vae,
            transition_colormatch,
            loop_colormatch_reference,
            prefix_alpha_crop,
            preserve_main_ref_background,
            ref_image=ref_image,
            bg_image=bg_image,
            pose_images=pose_images,
            prefix_frames=prefix_frames,
            prefix_mask=prefix_mask,
            transition_video=transition_video,
            pose_image_mask=pose_image_mask,
            reference_image_mask=reference_image_mask,
            clip_vision_output=clip_vision_output,
        )
        mode = "loop deferred build" if runtime.get("looping", False) else "context full build"
        log.info(
            f"SCAIL-2 Flow Embeds: {mode}, "
            f"{runtime['requested_output_frames']} requested frames, {runtime['num_frames']} sample frames, "
            f"window={runtime['frame_window_size']}, {runtime['width']}x{runtime['height']}, "
            f"batch={runtime['batch_size']}, tiled_vae={runtime['tiled_vae']}, replacement={runtime['replacement_mode']}"
        )
        log.info(
            "SCAIL-2 Flow inputs: "
            + ", ".join([
                self._flow_input_summary("ref", ref_image),
                self._flow_input_summary("bg", bg_image),
                self._flow_input_summary("pose", pose_images),
                self._flow_input_summary("prefix", prefix_frames),
                self._flow_input_summary("transition", transition_video),
                self._flow_input_summary("pose_mask", pose_image_mask),
                self._flow_input_summary("ref_mask", reference_image_mask),
                self._flow_input_summary("clip_vision", clip_vision_output),
            ])
        )
        if runtime.get("looping", False):
            runtime[FLOW_DEFERRED_BUILD_KEY] = "loop"
            runtime[FLOW_RUNTIME_VAE_KEY] = vae
            latent = build_deferred_latent(
                runtime,
                length=runtime["frame_window_size"],
                include_runtime=True,
            )
        else:
            positive, negative, latent = build_conditioning_and_latent(
                positive,
                negative,
                vae,
                runtime,
                start_frame=0,
                length=runtime["num_frames"],
                include_runtime=True,
            )
            release_flow_vae(vae)
        return (positive, negative, latent)


class WanAnimatePlusEverAnimateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the video to generate"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the video to generate"}),
            "num_frames": ("INT", {"default": 734, "min": 1, "max": 10000, "step": 1, "tooltip": "Total output frames. Segment count is computed from this, frame_window_size, and num_overlap_frame."}),
            "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload VAE after encoding to save VRAM"}),
            "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 10000, "step": 4, "tooltip": "Effective output frames per EverAnimate segment. Must be 1 mod 4, e.g. 77, 81, 85."}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the pose adapter"}),
            "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the face adapter"}),
            "pose_images": ("IMAGE", {"tooltip": "Pose control video"}),
            "face_images": ("IMAGE", {"tooltip": "Face control video. Resized to 512x512 internally."}),
            "num_video_anchor_latents": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1, "tooltip": "Number of anchor latent slots prepended to each EverAnimate segment"}),
            "num_motion_latents": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1, "tooltip": "Number of previous-segment motion latents used for continuity"}),
            "num_overlap_frame": ("INT", {"default": 4, "min": 0, "max": 10000, "step": 1, "tooltip": "Overlapping output frames between adjacent segments"}),
            "use_pingpong": ("BOOLEAN", {"default": True, "tooltip": "Ping-pong extend pose, face, bg, and mask sequences when more frames are needed"}),
            "use_image_anchor": ("BOOLEAN", {"default": True, "tooltip": "Use generated frames from the first segment to build later video anchors"}),
            "use_random_frame_anchor": ("BOOLEAN", {"default": True, "tooltip": "Randomly sample generated segment-0 frames for later video anchors"}),
            "random_anchor_with_user_first": ("BOOLEAN", {"default": True, "tooltip": "In random-anchor mode, reserve the first manual anchor frame as the user anchor"}),
            "use_repeat_anchor": ("BOOLEAN", {"default": False, "tooltip": "When fewer manual anchor frames are provided than anchor slots, repeat the provided sequence to fill the missing slots"}),
            "anchor_images": ("IMAGE", {"tooltip": "Manual identity/anchor image frames. Video inputs are treated as an anchor-frame sequence, not as source-video conditioning."}),
            },
            "optional": {
                "bg_images": ("IMAGE", {"tooltip": "Optional background/inpaint video"}),
                "mask": ("MASK", {"tooltip": "Optional mask paired with bg_images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = "EverAnimate conditioning bundle for segmented WanAnimatePlus sampling."

    @staticmethod
    def _fit_sequence(seq, target_len, use_pingpong):
        if seq is None:
            return None
        if target_len <= 0:
            return seq[:0]
        frames = seq.shape[0]
        if frames == target_len:
            return seq
        if frames > target_len:
            return seq[:target_len]
        if frames == 0:
            raise ValueError("Input sequence contains no frames")
        if frames == 1 or not use_pingpong:
            return torch.cat([seq, seq[-1:].repeat(target_len - frames, *([1] * (seq.ndim - 1)))], dim=0)

        indices = []
        idx = 0
        step = 1
        while len(indices) < target_len:
            indices.append(idx)
            idx += step
            if idx == frames - 1 or idx == 0:
                step *= -1
        return seq.index_select(0, torch.tensor(indices, device=seq.device))

    @staticmethod
    def _resize_video_bhwc(images, width, height, mode="lanczos", crop="disabled"):
        if images.shape[1] == height and images.shape[2] == width:
            return images[:, :, :, :3]
        return common_upscale(images[:, :, :, :3].movedim(-1, 1), width, height, mode, crop).movedim(1, -1)

    @staticmethod
    def _to_cthw_pixels(images):
        return images.permute(3, 0, 1, 2)[:3] * 2 - 1

    @staticmethod
    def _encode_single_frame_latents(vae, images, width, height, tiled_vae):
        if images is None or images.shape[0] == 0:
            raise ValueError("At least one anchor_images frame is required")

        latents = []
        for i in range(images.shape[0]):
            frame = WanAnimatePlusEverAnimateEmbeds._resize_video_bhwc(images[i:i + 1], width, height)
            frame = WanAnimatePlusEverAnimateEmbeds._to_cthw_pixels(frame).to(device=device, dtype=vae.dtype)
            lat = vae.encode([frame], device, tiled=tiled_vae)[0][:, :1].to(offload_device)
            latents.append(lat)
        return torch.cat(latents, dim=1)

    @staticmethod
    def _normalize_anchor_slots(anchor_latents, target_count, repeat_sequence=False):
        if anchor_latents.shape[1] == target_count:
            return anchor_latents
        if anchor_latents.shape[1] > target_count:
            return anchor_latents[:, :target_count]
        if repeat_sequence:
            repeats = math.ceil(target_count / anchor_latents.shape[1])
            return anchor_latents.repeat(1, repeats, 1, 1)[:, :target_count]
        pad = anchor_latents[:, -1:].repeat(1, target_count - anchor_latents.shape[1], 1, 1)
        return torch.cat([anchor_latents, pad], dim=1)

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, pose_strength, face_strength,
                pose_images, face_images, num_video_anchor_latents, num_motion_latents, num_overlap_frame,
                use_pingpong, use_image_anchor, use_random_frame_anchor, random_anchor_with_user_first,
                use_repeat_anchor, anchor_images, bg_images=None, mask=None, tiled_vae=False):

        W = (width // 16) * 16
        H = (height // 16) * 16
        if (frame_window_size - 1) % 4 != 0:
            raise ValueError("frame_window_size must be 1 mod 4 for EverAnimate, e.g. 77, 81, 85")
        if num_overlap_frame >= frame_window_size:
            raise ValueError("num_overlap_frame must be smaller than frame_window_size")
        if (bg_images is None) != (mask is None):
            raise ValueError("bg_images and mask must be connected together")

        stride = frame_window_size - num_overlap_frame
        if num_frames <= frame_window_size:
            num_segments = 1
        else:
            num_segments = math.ceil((num_frames - frame_window_size) / stride) + 1
        generated_frames = frame_window_size + (num_segments - 1) * stride

        manual_images = anchor_images
        anchor_source = "anchor_images"
        if manual_images is None:
            raise ValueError("EverAnimate requires anchor_images")

        manual_images = manual_images[:, :, :, :3]
        manual_count = min(manual_images.shape[0], num_video_anchor_latents)
        if manual_count <= 0:
            raise ValueError(f"{anchor_source} contains no frames")

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        manual_anchor_latents = self._encode_single_frame_latents(
            vae, manual_images[:manual_count], W, H, tiled_vae
        )
        user_first_anchor_latent = manual_anchor_latents[:, :1].clone()
        anchor_latents = self._normalize_anchor_slots(
            manual_anchor_latents,
            num_video_anchor_latents,
            repeat_sequence=use_repeat_anchor,
        )

        pose_images = self._fit_sequence(pose_images[:, :, :, :3], generated_frames, use_pingpong)
        pose_images = self._resize_video_bhwc(pose_images, W, H)
        pose_pixels = self._to_cthw_pixels(pose_images).to(offload_device, dtype=vae.dtype)

        face_images = self._fit_sequence(face_images[:, :, :, :3], generated_frames, use_pingpong)
        face_images = self._resize_video_bhwc(face_images, 512, 512, crop="center")
        face_pixels = self._to_cthw_pixels(face_images).unsqueeze(0).to(offload_device, dtype=vae.dtype)

        bg_pixels = None
        mask_pixels = None
        if bg_images is not None:
            bg_images = self._fit_sequence(bg_images[:, :, :, :3], generated_frames, use_pingpong)
            bg_images = self._resize_video_bhwc(bg_images, W, H)
            bg_pixels = self._to_cthw_pixels(bg_images).to(offload_device, dtype=vae.dtype)

            mask = self._fit_sequence(mask, generated_frames, use_pingpong)
            if mask.shape[1] != H or mask.shape[2] != W:
                mask_pixels = common_upscale(mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            else:
                mask_pixels = mask
            mask_pixels = mask_pixels.to(offload_device, dtype=vae.dtype)

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor
        target_shape = (16, (num_frames - 1) // 4 + 1, lat_h, lat_w)
        segment_latent_frames = (frame_window_size + 4 * num_video_anchor_latents - 1) // 4 + 1

        if force_offload:
            vae.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "vae": vae,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,

            "everanimate": True,
            "everanimate_num_segments": num_segments,
            "everanimate_generated_frames": generated_frames,
            "everanimate_segment_latent_frames": segment_latent_frames,
            "frame_window_size": frame_window_size,
            "num_overlap_frame": num_overlap_frame,
            "num_video_anchor_latents": num_video_anchor_latents,
            "num_motion_latents": num_motion_latents,
            "use_pingpong": use_pingpong,
            "use_image_anchor": use_image_anchor,
            "use_random_frame_anchor": use_random_frame_anchor,
            "random_anchor_with_user_first": random_anchor_with_user_first,
            "use_repeat_anchor": use_repeat_anchor,

            "anchor_source": anchor_source,
            "manual_anchor_count": manual_count,
            "manual_anchor_latents": manual_anchor_latents,
            "anchor_latents": anchor_latents,
            "user_first_anchor_latent": user_first_anchor_latent,
            "random_user_anchor_latent": user_first_anchor_latent,
            "manual_random_user_anchor_index": 0 if random_anchor_with_user_first else None,

            "pose_images": pose_pixels,
            "face_pixels": face_pixels,
            "bg_images": bg_pixels,
            "mask": mask_pixels,
            "is_masked": mask_pixels is not None,
            "pose_strength": pose_strength,
            "face_strength": face_strength,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "looping": False,
        }

        log.info(
            f"EverAnimate Embeds: {num_frames} output frames, {num_segments} segments, "
            f"{generated_frames} generated frames before trim, anchors={anchor_latents.shape[1]} from {anchor_source}"
        )

        return (image_embeds,)

# region UniLumos
class WanVideoUniLumosEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "foreground_latents": ("LATENT", {"tooltip": "Video foreground latents"}),
                "background_latents": ("LATENT", {"tooltip": "Video background latents"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, foreground_latents=None, background_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
        }
        if foreground_latents is not None:
            embeds["foreground_latents"] = foreground_latents["samples"][0]
        else:
            embeds["foreground_latents"] = torch.zeros(target_shape[0], target_shape[1], target_shape[2], target_shape[3], device=torch.device("cpu"), dtype=torch.float32)
        if background_latents is not None:
            embeds["background_latents"] = background_latents["samples"][0]
        else:
            embeds["background_latents"] = torch.zeros(target_shape[0], target_shape[1], target_shape[2], target_shape[3], device=torch.device("cpu"), dtype=torch.float32)

        return (embeds,)
    
class WanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "control signal for the Fun -model"}),
                "extra_latents": ("LATENT", {"tooltip": "First latent to use for the Pusa -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, control_embeds=None, extra_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
        }
        if extra_latents is not None:
            embeds["extra_latents"] = [{
                "samples": extra_latents["samples"],
                "index": 0,
            }]

        return (embeds,)
    
class WanVideoAddExtraLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "extra_latents": ("LATENT",),
                    "latent_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1, "tooltip": "Index to insert the extra latents at in latent space"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, extra_latents, latent_index):
        # Prepare the new extra latent entry
        new_entry = {
            "samples": extra_latents["samples"],
            "index": latent_index,
        }
        # Get previous extra_latents list, or start a new one
        prev_extra_latents = embeds.get("extra_latents", None)
        if prev_extra_latents is None:
            extra_latents_list = [new_entry]
        elif isinstance(prev_extra_latents, list):
            extra_latents_list = prev_extra_latents + [new_entry]
        else:
            extra_latents_list = [prev_extra_latents, new_entry]

        # Return a new dict with updated extra_latents
        updated = dict(embeds)
        updated["extra_latents"] = extra_latents_list
        return (updated,)
    
class WanVideoAddLucyEditLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "extra_latents": ("LATENT",),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, extra_latents):
        updated = dict(embeds)
        updated["extra_channel_latents"] = extra_latents["samples"]
        return (updated,)

class WanVideoMiniMaxRemoverEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            "mask_latents": ("LATENT", {"tooltip": "Encoded latents to use as mask"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, latents, mask_latents):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "minimax_latents": latents["samples"].squeeze(0),
            "minimax_mask_latents": mask_latents["samples"].squeeze(0),
        }
    
        return (embeds,)
    
# region phantom
class WanVideoPhantomEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "phantom_latent_1": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
            
            "phantom_cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFG scale for the extra phantom cond pass"}),
            "phantom_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the phantom model"}),
            "phantom_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the phantom model"}),
            },
            "optional": {
                "phantom_latent_2": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_3": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_4": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "vace_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "VACE embeds"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, phantom_cfg_scale, phantom_start_percent, phantom_end_percent, phantom_latent_1, phantom_latent_2=None, phantom_latent_3=None, phantom_latent_4=None, vace_embeds=None):
        samples = phantom_latent_1["samples"].squeeze(0)
        if phantom_latent_2 is not None:
            samples = torch.cat([samples, phantom_latent_2["samples"].squeeze(0)], dim=1)
        if phantom_latent_3 is not None:
            samples = torch.cat([samples, phantom_latent_3["samples"].squeeze(0)], dim=1)
        if phantom_latent_4 is not None:
            samples = torch.cat([samples, phantom_latent_4["samples"].squeeze(0)], dim=1)
        C, T, H, W = samples.shape

        log.info(f"Phantom latents shape: {samples.shape}")

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        H * 8 // VAE_STRIDE[1],
                        W * 8 // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "phantom_latents": samples,
            "phantom_cfg_scale": phantom_cfg_scale,
            "phantom_start_percent": phantom_start_percent,
            "phantom_end_percent": phantom_end_percent,
        }
        if vace_embeds is not None:
            vace_input = {
                "vace_context": vace_embeds["vace_context"],
                "vace_scale": vace_embeds["vace_scale"],
                "has_ref": vace_embeds["has_ref"],
                "vace_start_percent": vace_embeds["vace_start_percent"],
                "vace_end_percent": vace_embeds["vace_end_percent"],
                "vace_seq_len": vace_embeds["vace_seq_len"],
                "additional_vace_inputs": vace_embeds["additional_vace_inputs"],
                }
            embeds.update(vace_input)
    
        return (embeds,)
    
class WanVideoControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            },
            "optional": {
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, latents, start_percent, end_percent, fun_ref_image=None):
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape

        num_frames = (T - 1) * 4 + 1
        seq_len = math.ceil((H * W) / 4 * ((num_frames - 1) // 4 + 1))
      
        embeds = {
            "max_seq_len": seq_len,
            "target_shape": samples.shape,
            "num_frames": num_frames,
            "control_embeds": {
                "control_images": samples,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
            }
        }
    
        return (embeds,)
    
class WanVideoAddControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
            "optional": {
                "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, embeds, start_percent, end_percent, fun_ref_image=None, latents=None):      
        new_entry = {
            "control_images": latents["samples"].squeeze(0) if latents is not None else None,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
        }

        updated = dict(embeds)
        updated["control_embeds"] = new_entry

        return (updated,)
    
class WanVideoAddPusaNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "noise_multipliers": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Noise multipliers for Pusa, can be a list of floats"}),
            "noisy_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Number steps to apply the extra noise"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds latent and timestep noise multipliers when using flowmatch_pusa"

    def add(self, embeds, noise_multipliers, noisy_steps):
        updated = dict(embeds)
        updated["pusa_noise_multipliers"] = noise_multipliers
        updated["pusa_noisy_steps"] = noisy_steps

        return (updated,)
    
class WanVideoSLG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "blocks": ("STRING", {"default": "10", "tooltip": "Blocks to skip uncond on, separated by comma, index starts from 0"}),
            "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
        }

    RETURN_TYPES = ("SLGARGS", )
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Skips uncond on the selected blocks"

    def process(self, blocks, start_percent, end_percent):
        slg_block_list = [int(x.strip()) for x in blocks.split(",")]

        slg_args = {
            "blocks": slg_block_list,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        return (slg_args,)

#region VACE
class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, tiled_vae=False):
        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        # vace context encode
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=vae.dtype)
        else:
            input_frames = input_frames.clone()[:num_frames, :, :, :3]
            input_frames = common_upscale(input_frames.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1
        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=device)
        else:
            log.info(f"input_masks shape: {input_masks.shape}")
            input_masks = input_masks[:num_frames]
            input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            input_masks = input_masks.to(vae.dtype).to(device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        if ref_images is not None:
            ref_images = ref_images.clone()[..., :3]
            # Create padded image
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
            
            B, H, W, C = ref_images.shape
            current_aspect = W / H
            target_aspect = width / height
            if current_aspect > target_aspect:
                # Image is wider than target, pad height
                new_h = int(W / target_aspect)
                pad_h = (new_h - H) // 2
                padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, pad_h:pad_h+H, :, :] = ref_images
                ref_images = padded
            elif current_aspect < target_aspect:
                # Image is taller than target, pad width
                new_w = int(H * target_aspect)
                pad_w = (new_w - W) // 2
                padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, :, pad_w:pad_w+W, :] = ref_images
                ref_images = padded
            ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            
            ref_images = ref_images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1

        vae = vae.to(device)
        z0 = self.vace_encode_frames(vae, input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
        
        m0 = self.vace_encode_masks(input_masks, ref_images)
        z = self.vace_latent(z0, m0)
        vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_input["additional_vace_inputs"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)
    
        return (vace_input,)
    
    def vace_encode_frames(self, vae, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        pbar = ProgressBar(len(frames))
        if masks is None:
            latents = vae.encode(frames, device=device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            del frames
            inactive = vae.encode(inactive, device=device, tiled=tiled_vae)
            reactive = vae.encode(reactive, device=device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
            del inactive, reactive
        
        
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                else:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
            pbar.update(1)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        pbar = ProgressBar(len(masks))
        for mask, refs in zip(masks, ref_images):
            _c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
            pbar.update(1)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]


#region context options
class WanVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
            "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 16, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print debug output"}),
            },
            "optional": {
                "fuse_method": (["linear", "pyramid"], {"default": "linear", "tooltip": "Window weight function: linear=ramps at edges only, pyramid=triangular weights peaking in middle"}),
                "reference_latent": ("LATENT", {"tooltip": "Image to be used as init for I2V models for windows where first frame is not the actual first frame. Mostly useful with MAGREF model"}),
            }
        }

    RETURN_TYPES = ("WANVIDCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Context options for WanVideo, allows splitting the video into context windows and attemps blending them for longer generations than the model and memory otherwise would allow."

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise, verbose, image_cond_start_step=6, image_cond_window_count=2, vae=None, fuse_method="linear", reference_latent=None):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise,
            "verbose":verbose,
            "fuse_method":fuse_method,
            "reference_latent":reference_latent["samples"] if reference_latent is not None else None,
        }

        return (context_options,)

class WanVideoLoopArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "shift_skip": ("INT", {"default": 6, "min": 0, "tooltip": "Skip step of latent shift"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the looping effect"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the looping effect"}),
            },
        }

    RETURN_TYPES = ("LOOPARGS", )
    RETURN_NAMES = ("loop_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Looping through latent shift as shown in https://github.com/YisuiTT/Mobius/"

    def process(self, **kwargs):
        return (kwargs,)

class WanVideoExperimentalArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "video_attention_split_steps": ("STRING", {"default": "", "tooltip": "Steps to split self attention when using multiple prompts"}),
                "cfg_zero_star": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WeichenFan/CFG-Zero-star"}),
                "use_zero_init": ("BOOLEAN", {"default": False}),
                "zero_star_steps": ("INT", {"default": 0, "min": 0, "tooltip": "Steps to split self attention when using multiple prompts"}),
                "use_fresca": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WikiChao/FreSca"}),
                "fresca_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_scale_high": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_freq_cutoff": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "use_tcfg": ("BOOLEAN", {"default": False, "tooltip": "https://arxiv.org/abs/2503.18137 TCFG: Tangential Damping Classifier-free Guidance. CFG artifacts reduction."}),
                "raag_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Alpha value for RAAG, 1.0 is default, 0.0 is disabled."}),
                "bidirectional_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable bidirectional sampling, based on https://github.com/ff2416/WanFM"}),
                "temporal_score_rescaling": ("BOOLEAN", {"default": False, "tooltip": "Enable temporal score rescaling: https://github.com/temporalscorerescaling/TSR/"}),
                "tsr_k": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "The sampling temperature"}),
                "tsr_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How early TSR steer the sampling process"}),
            },
        }

    RETURN_TYPES = ("EXPERIMENTALARGS", )
    RETURN_NAMES = ("exp_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Experimental stuff"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoFreeInitArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "freeinit_num_iters": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Number of FreeInit iterations"}),
                "freeinit_method": (["butterworth", "ideal", "gaussian", "none"], {"default": "ideal", "tooltip": "Frequency filter type"}),
                "freeinit_n": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Butterworth filter order (only for butterworth)"}),
                "freeinit_d_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Spatial filter cutoff"}),
                "freeinit_d_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Temporal filter cutoff"}),
            },
        }

    RETURN_TYPES = ("FREEINITARGS", )
    RETURN_NAMES = ("freeinit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/TianxingWu/FreeInit; FreeInit, a concise yet effective method to improve temporal consistency of videos generated by diffusion models"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)

rope_functions = ["default", "comfy", "comfy_chunked"]
class WanVideoRoPEFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "rope_function": (rope_functions, {"default": "comfy"}),
                "ntk_scale_f": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "ntk_scale_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (rope_functions, )
    RETURN_NAMES = ("rope_function",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, rope_function, ntk_scale_f, ntk_scale_h, ntk_scale_w):
        if ntk_scale_f != 1.0 or ntk_scale_h != 1.0 or ntk_scale_w != 1.0:
            rope_func_dict = {
                "rope_function": rope_function,
                "ntk_scale_f": ntk_scale_f,
                "ntk_scale_h": ntk_scale_h,
                "ntk_scale_w": ntk_scale_w,
            }
            return (rope_func_dict,)
        return (rope_function,)

#region TTM
class WanVideoAddTTMLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("WANVIDIMAGE_EMBEDS",),
            "reference_latents": ("LATENT", {"tooltip": "Latents used as reference for TTM"}),
            "mask": ("MASK", {"tooltip": "Mask used for TTM"}),
            "start_step": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1, "tooltip": "Start step for whole denoising process"}),
            "end_step": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "tooltip": "The step to stop applying TTM"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds", )
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/time-to-move/TTM"

    def add(self, embeds, reference_latents, mask, start_step, end_step):

        if end_step < max(0, start_step):
            raise ValueError(f"`end_step` ({end_step}) must be >= `start_step` ({start_step}).")

        mask_sampled = mask[::4]
        mask_sampled = mask_sampled.unsqueeze(1).unsqueeze(0)  # [1, T, 1, H, W]

        vae_upscale_factor = 8
        if reference_latents["samples"].shape[1] == 48:
            vae_upscale_factor = 16

        # Upsample spatially to latent resolution
        H_latent = mask_sampled.shape[-2] // vae_upscale_factor
        W_latent = mask_sampled.shape[-1] // vae_upscale_factor
        mask_latent = F.interpolate(
            mask_sampled.float(),
            size=(mask_sampled.shape[2], H_latent, W_latent),
            mode="nearest"
        )

        updated = dict(embeds)
        updated["ttm_reference_latents"] = reference_latents["samples"].squeeze(0)
        updated["ttm_mask"] = mask_latent.squeeze(0).movedim(1, 0)  # [T, 1, H, W]
        updated["ttm_start_step"] = start_step
        updated["ttm_end_step"] = end_step

        return (updated,)

#region VideoDecode
class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax", "none"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        mm.soft_empty_cache()
        video = samples.get("video", None)
        if video is not None:
            video.clamp_(-1.0, 1.0)
            video.add_(1.0).div_(2.0)
            return video.cpu().float(),
        latents = samples["samples"].clone()
        end_image = samples.get("end_image", None)
        has_ref = samples.get("has_ref", False)
        has_prefix = samples.get("has_prefix", False)
        ref_skip_value = samples.get("wananim_decode_ref_latents", None)
        ref_skip = int(ref_skip_value if ref_skip_value is not None else (1 if has_ref else 0))
        canvas_expansion_px = samples.get("canvas_expansion_px", 0)
        output_frame_count = samples.get("output_frame_count", None)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        flashvsr_LQ_images = samples.get("flashvsr_LQ_images", None)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        if ref_skip > 0:
            latents = latents[:, :, ref_skip:]
        if drop_last:
            latents = latents[:, :, :-1]

        if type(vae).__name__ == "TAEHV":
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4), cond=flashvsr_LQ_images.to(vae.dtype) if flashvsr_LQ_images is not None else None)[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            if canvas_expansion_px:
                images = images[:, canvas_expansion_px:]
            if output_frame_count is not None and images.shape[1] > output_frame_count:
                images = images[:, :output_frame_count]
            images = images.permute(1, 2, 3, 0).cpu().float()
            vae.to(offload_device)
            mm.soft_empty_cache()
            return (images,)
        else:
            images = vae.decode(latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]


        images = images.cpu().float()

        if normalization != "none":
            if normalization == "minmax":
                images.sub_(images.min()).div_(images.max() - images.min())
            else:
                images.clamp_(-1.0, 1.0)
                images.add_(1.0).div_(2.0)

        if is_looped:
            temp_latents = torch.cat([latents[:, :, -3:]] + [latents[:, :, :2]], dim=2)
            temp_images = vae.decode(temp_latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))[0]
            temp_images = temp_images.cpu().float()
            temp_images = (temp_images - temp_images.min()) / (temp_images.max() - temp_images.min())
            images = torch.cat([temp_images[:, 9:].to(images), images[:, 5:]], dim=1)

        if end_image is not None:
            images = images[:, 0:-1]

        if canvas_expansion_px:
            images = images[:, canvas_expansion_px:]
        if output_frame_count is not None and images.shape[1] > output_frame_count:
            images = images[:, :output_frame_count]


        vae.to(offload_device)
        mm.soft_empty_cache()

        images.clamp_(0.0, 1.0)

        return (images.permute(1, 2, 3, 0),)


class WanAnimatePlusVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = "Official-compatible VAE decode. If SCAIL-2 Flow Sampler returns decoded video in the LATENT object, this node returns it directly."

    def decode(self, vae, samples, tiled_vae=False):
        video = samples.get("video", None)
        if video is not None:
            images = video.detach().cpu().float()
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            if images.numel() > 0 and float(images.min()) < 0.0:
                images = images.clamp(-1.0, 1.0).add(1.0).div(2.0)
            else:
                images = images.clamp(0.0, 1.0)
            return (images,)

        images = decode_latent_to_images(vae, samples, tiled_vae=tiled_vae)
        canvas_expansion_px = int(samples.get("canvas_expansion_px", 0) or 0)
        if canvas_expansion_px and images.shape[0] > canvas_expansion_px:
            images = images[canvas_expansion_px:]
        output_frame_count = samples.get("output_frame_count", None)
        if output_frame_count is not None and images.shape[0] > int(output_frame_count):
            images = images[:int(output_frame_count)]
        return (images,)

#region VideoEncode
class WanVideoEncodeLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "images": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes a batch of images individually to create a latent video batch where each video is a single frame, useful for I2V init purposes, for example as multiple context window inits"

    def encode(self, vae, images, enable_vae_tiling=False, tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128, latent_strength=1.0):
        vae.to(device)

        images = images.clone()

        B, H, W, C = images.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            images = common_upscale(images.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if images.shape[-1] == 4:
            images = images[..., :3]
        images = images.to(vae.dtype).to(device) * 2.0 - 1.0

        latent_list = []
        for img in images:
            if enable_vae_tiling and tile_x is not None:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))
            else:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling)

            if latent_strength != 1.0:
                latent *= latent_strength
            latent_list.append(latent.squeeze(0).cpu())
        latents_out = torch.stack(latent_list, dim=0)

        log.info(f"WanVideoEncode: Encoded latents shape {latents_out.shape}")
        vae.to(offload_device)
        mm.soft_empty_cache()

        return ({"samples": latents_out},)

class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None):
        vae.to(device)

        image = image.clone()

        B, H, W, C = image.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            image = common_upscale(image.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W        

        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

        if isinstance(vae, TAEHV):
            latents = vae.encode_video(image.permute(0, 2, 1, 3, 4), parallel=False)# B, T, C, H, W
            latents = latents.permute(0, 2, 1, 3, 4)
        else:
            latents = vae.encode(image * 2.0 - 1.0, device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))

            vae.to(offload_device)
        if latent_strength != 1.0:
            latents *= latent_strength

        latents = latents.cpu()

        log.info(f"WanVideoEncode: Encoded latents shape {latents.shape}")
        mm.soft_empty_cache()

        return ({"samples": latents, "noise_mask": mask},)

class WanAnimatePlusBernini:
    """Bernini in-context conditioning for WanAnimatePlus.

    VAE-encodes source video / reference images / reference video and attaches
    them as extra in-context tokens (context_latents) to the image embeds.
    The Wan model appends these tokens with per-stream source_id RoPE.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE",),
                "task_type": (["t2v", "v2v", "r2v", "rv2v"], {"default": "t2v", "tooltip": "Select your task type to see the recommended sampler guidance settings"}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
            },
            "optional": {
                "source_video": ("IMAGE", {"tooltip": "Source video to edit/restyle (v2v/rv2v). Resized to width/height. Acts as the edit base."}),
                "reference_video": ("IMAGE", {"tooltip": "Moving content to composite into the source (video insertion), kept at native aspect."}),
                "reference_image_1": ("IMAGE", {"tooltip": "Reference image 1, native aspect preserved."}),
                "reference_image_2": ("IMAGE", {"tooltip": "Reference image 2, native aspect preserved."}),
                "reference_image_3": ("IMAGE", {"tooltip": "Reference image 3, native aspect preserved."}),
                "reference_image_4": ("IMAGE", {"tooltip": "Reference image 4, native aspect preserved."}),
                "reference_image_5": ("IMAGE", {"tooltip": "Reference image 5, native aspect preserved."}),
                "reference_image_6": ("IMAGE", {"tooltip": "Reference image 6, native aspect preserved."}),
                "reference_image_7": ("IMAGE", {"tooltip": "Reference image 7, native aspect preserved."}),
                "reference_image_8": ("IMAGE", {"tooltip": "Reference image 8, native aspect preserved."}),
                "reference_image_9": ("IMAGE", {"tooltip": "Reference image 9, native aspect preserved."}),
                "reference_image_10": ("IMAGE", {"tooltip": "Reference image 10, native aspect preserved."}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16, "tooltip": "Max long-edge size for reference_video and reference_images."}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload VAE after encoding to save VRAM"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "by wuwukasi(bilibili)": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "ON", "tooltip": "Follow wuwukasi on bilibili"}),
            },
        }

    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = (
        "Bernini in-context conditioning for Wan2.x models. "
        "Recommended sampler guidance_mode by task:\n"
        "  t2v (no media): apg, apg_omega=4.0\n"
        "  v2v (source_video): apg, apg_omega=4.0 | or cfg_chain, chain_omega_V=1.25 chain_omega_TI=4.0\n"
        "  r2v (ref_images): apg_chain, apg_omega_I=4.5 apg_omega_TI=4.0 | or cfg_chain, chain_omega_I=4.5 chain_omega_TI=4.0\n"
        "  rv2v (src+ref): cfg_chain, chain_omega_V=1.25 chain_omega_I=4.5 chain_omega_TI=4.0 | alt apg, apg_omega=4.0\n"
        "Shared APG params: apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0"
    )
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING",)
    RETURN_NAMES = ("image_embeds", "recommended_guidance",)

    def process(self, vae, task_type, width, height, num_frames,
                source_video=None, reference_video=None, ref_max_size=848,
                force_offload=True, tiled_vae=False, **kwargs):

        def _resize_long_edge(image, max_size, stride=16):
            """Resize keeping aspect ratio so long edge ≤ max_size, snapped to stride."""
            h, w = image.shape[1], image.shape[2]
            if max(h, w) <= max_size:
                return image[:, :, :, :3]
            ratio = max_size / max(h, w)
            # Snap long edge to stride, then derive short edge proportionally
            # to preserve aspect ratio (independent rounding destroys it for extreme ratios)
            if h >= w:
                nh = max(stride, round(h * ratio / stride) * stride)
                nw = max(stride, round(nh * w / h / stride) * stride)
            else:
                nw = max(stride, round(w * ratio / stride) * stride)
                nh = max(stride, round(nw * h / w / stride) * stride)
            return common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "bicubic", "disabled").movedim(1, -1)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Ordered context list: source_video (1), reference_video (2), reference_images (3,4,...)
        context = []
        context_roles = []

        if source_video is not None:
            vid = common_upscale(source_video[:num_frames, :, :, :3].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            vae.to(device)
            context.append(vae.encode([(vid.permute(3, 0, 1, 2) * 2 - 1).to(device=device, dtype=vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device))
            context_roles.append("source_video")
            if force_offload:
                vae.to(offload_device)

        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:num_frames], ref_max_size)
            vae.to(device)
            context.append(vae.encode([(ref_vid.permute(3, 0, 1, 2) * 2 - 1).to(device=device, dtype=vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device))
            context_roles.append("reference_video")
            if force_offload:
                vae.to(offload_device)

        # Collect reference images from named slots (reference_image_1 to reference_image_10)
        for i in range(1, 11):
            ref_img = kwargs.get(f"reference_image_{i}", None)
            if ref_img is not None:
                img = _resize_long_edge(ref_img[0:1], ref_max_size)  # each slot is 1 image
                vae.to(device)
                context.append(vae.encode([(img.permute(3, 0, 1, 2) * 2 - 1).to(device=device, dtype=vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device))
                context_roles.append("reference_image")
                if force_offload:
                    vae.to(offload_device)

        if context:
            log.info(f"Bernini: attached {len(context)} context streams")

        target_shape = (16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
        recommendations = {
            "t2v":    "guidance_mode=apg, apg_omega=4.0, apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0\n"
                      "  alt: guidance_mode=cfg, cfg=4.0",
            "v2v":    "guidance_mode=apg, apg_omega=4.0, apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0\n"
                      "  alt: guidance_mode=cfg_chain, chain_omega_V=1.25, chain_omega_TI=4.0",
            "r2v":    "guidance_mode=apg_chain, apg_omega_I=4.5, apg_omega_TI=4.0, apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0\n"
                      "  alt: guidance_mode=cfg_chain, chain_omega_I=4.5, chain_omega_TI=4.0",
            "rv2v":   "guidance_mode=cfg_chain, chain_omega_V=1.25, chain_omega_I=4.5, chain_omega_TI=4.0\n"
                      "  alt: guidance_mode=apg, apg_omega=4.0, apg_eta=0.5, apg_momentum=0.0, apg_norm_threshold=50.0",
        }
        rec = recommendations.get(task_type, "")
        image_embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "context_latents": list(context) if context else None,
            "context_roles": list(context_roles) if context else None,
        }
        return (image_embeds, rec)


NODE_CLASS_MAPPINGS = {
    "WanVideoDecode": WanVideoDecode,
    "WanAnimatePlusVAEDecode": WanAnimatePlusVAEDecode,
    "WanVideoTextEncode": WanVideoTextEncode,
    "WanVideoTextEncodeSingle": WanVideoTextEncodeSingle,
    "WanVideoClipVisionEncode": WanVideoClipVisionEncode,
    "WanVideoClipVisionEncodeV2": WanVideoClipVisionEncodeV2,
    "WanVideoImageToVideoEncode": WanVideoImageToVideoEncode,
    "WanVideoEncode": WanVideoEncode,
    "WanVideoEncodeLatentBatch": WanVideoEncodeLatentBatch,
    "WanVideoEmptyEmbeds": WanVideoEmptyEmbeds,
    "WanVideoEnhanceAVideo": WanVideoEnhanceAVideo,
    "WanVideoContextOptions": WanVideoContextOptions,
    "WanVideoTextEmbedBridge": WanVideoTextEmbedBridge,
    "WanVideoControlEmbeds": WanVideoControlEmbeds,
    "WanVideoSLG": WanVideoSLG,
    "WanVideoLoopArgs": WanVideoLoopArgs,
    "WanVideoSetBlockSwap": WanVideoSetBlockSwap,
    "WanVideoExperimentalArgs": WanVideoExperimentalArgs,
    "WanVideoVACEEncode": WanVideoVACEEncode,
    "WanVideoPhantomEmbeds": WanVideoPhantomEmbeds,
    "WanVideoRealisDanceLatents": WanVideoRealisDanceLatents,
    "WanVideoApplyNAG": WanVideoApplyNAG,
    "WanVideoMiniMaxRemoverEmbeds": WanVideoMiniMaxRemoverEmbeds,
    "WanVideoFreeInitArgs": WanVideoFreeInitArgs,
    "WanVideoSetRadialAttention": WanVideoSetRadialAttention,
    "WanVideoBlockList": WanVideoBlockList,
    "WanVideoTextEncodeCached": WanVideoTextEncodeCached,
    "WanVideoAddExtraLatent": WanVideoAddExtraLatent,
    "WanVideoAddStandInLatent": WanVideoAddStandInLatent,
    "WanVideoAddControlEmbeds": WanVideoAddControlEmbeds,
    "WanVideoAddMTVMotion": WanVideoAddMTVMotion,
    "WanVideoRoPEFunction": WanVideoRoPEFunction,
    "WanVideoAddPusaNoise": WanVideoAddPusaNoise,
    "WanVideoAnimateEmbeds": WanVideoAnimateEmbeds,
    "WanAnimatePlusSCAIL2Embeds": WanAnimatePlusSCAIL2Embeds,
    "WanAnimatePlusSCAIL2TwoPhaseSettings": WanAnimatePlusSCAIL2TwoPhaseSettings,
    "WanAnimatePlusSCAIL2FlowEmbeds": WanAnimatePlusSCAIL2FlowEmbeds,
    "WanAnimatePlusEverAnimateEmbeds": WanAnimatePlusEverAnimateEmbeds,
    "WanVideoAddLucyEditLatents": WanVideoAddLucyEditLatents,
    "WanVideoAddBindweaveEmbeds": WanVideoAddBindweaveEmbeds,
    "TextImageEncodeQwenVL": TextImageEncodeQwenVL,
    "WanVideoUniLumosEmbeds": WanVideoUniLumosEmbeds,
    "WanVideoAddTTMLatents": WanVideoAddTTMLatents,
    "WanVideoAddStoryMemLatents": WanVideoAddStoryMemLatents,
    "WanVideoSVIProEmbeds": WanVideoSVIProEmbeds,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDecode": "WanVideo Decode",
    "WanAnimatePlusVAEDecode": "WanAnimatePlus VAE Decode",
    "WanVideoTextEncode": "WanVideo TextEncode",
    "WanVideoTextEncodeSingle": "WanVideo TextEncodeSingle",
    "WanVideoTextImageEncode": "WanVideo TextImageEncode (IP2V)",
    "WanVideoClipVisionEncode": "WanVideo ClipVision Encode",
    "WanVideoClipVisionEncodeV2": "WanVideo ClipVision Encode V2",
    "WanVideoImageToVideoEncode": "WanVideo ImageToVideo Encode",
    "WanVideoEncode": "WanVideo Encode",
    "WanVideoEncodeLatentBatch": "WanVideo Encode Latent Batch",
    "WanVideoEmptyEmbeds": "WanVideo Empty Embeds",
    "WanVideoEnhanceAVideo": "WanVideo Enhance-A-Video",
    "WanVideoContextOptions": "WanVideo Context Options",
    "WanVideoTextEmbedBridge": "WanVideo TextEmbed Bridge",
    "WanVideoControlEmbeds": "WanVideo Control Embeds",
    "WanVideoSLG": "WanVideo SLG",
    "WanVideoLoopArgs": "WanVideo Loop Args",
    "WanVideoSetBlockSwap": "WanVideo Set BlockSwap",
    "WanVideoExperimentalArgs": "WanVideo Experimental Args",
    "WanVideoVACEEncode": "WanVideo VACE Encode",
    "WanVideoPhantomEmbeds": "WanVideo Phantom Embeds",
    "WanVideoRealisDanceLatents": "WanVideo RealisDance Latents",
    "WanVideoApplyNAG": "WanVideo Apply NAG",
    "WanVideoMiniMaxRemoverEmbeds": "WanVideo MiniMax Remover Embeds",
    "WanVideoFreeInitArgs": "WanVideo Free Init Args",
    "WanVideoSetRadialAttention": "WanVideo Set Radial Attention",
    "WanVideoBlockList": "WanVideo Block List",
    "WanVideoTextEncodeCached": "WanVideo TextEncode Cached",
    "WanVideoAddExtraLatent": "WanVideo Add Extra Latent",
    "WanVideoAddStandInLatent": "WanVideo Add StandIn Latent",
    "WanVideoAddControlEmbeds": "WanVideo Add Control Embeds",
    "WanVideoAddMTVMotion": "WanVideo MTV Crafter Motion",
    "WanVideoRoPEFunction": "WanVideo RoPE Function",
    "WanVideoAddPusaNoise": "WanVideo Add Pusa Noise",
    "WanVideoAnimateEmbeds": "WanVideo Animate Embeds",
    "WanAnimatePlusSCAIL2Embeds": "WanAnimatePlus SCAIL_2 Embeds",
    "WanAnimatePlusSCAIL2TwoPhaseSettings": "WanAnimatePlus SCAIL_2 TwoPhase Settings",
    "WanAnimatePlusSCAIL2FlowEmbeds": "WanAnimatePlus SCAIL_2 Flow Embeds",
    "WanAnimatePlusEverAnimateEmbeds": "WanAnimatePlus EverAnimate Embeds",
    "WanVideoAddLucyEditLatents": "WanVideo Add LucyEdit Latents",
    "WanVideoAddBindweaveEmbeds": "WanVideo Add Bindweave Embeds",
    "WanVideoUniLumosEmbeds": "WanVideo UniLumos Embeds",
    "WanVideoAddTTMLatents": "WanVideo Add TTMLatents",
    "WanVideoAddStoryMemLatents": "WanVideo Add StoryMem Latents",
    "WanVideoSVIProEmbeds": "WanVideo SVIPro Embeds",
}
