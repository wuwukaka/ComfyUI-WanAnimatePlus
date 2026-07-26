# Copyright (c) 2025 kijai
# Modified from __init__.py in ComfyUI-WanVideoWrapper.
# Original project: https://github.com/kijai/ComfyUI-WanVideoWrapper
# Modified portions Copyright (c) 2026 wuwukasi/wuwukaka.
#   - Registers a WanAnimatePlus-specific workflow node chain with renamed node keys.
#   - Adds WanAnimatePlus Bernini and EverAnimate embeds nodes to the public mappings.
#   - Adds WanAnimatePlus SCAIL-2 embeds node to the public mappings.
#   - Adds WanAnimatePlus SCAIL-2 two-phase settings node to the public mappings.
#   - Adds official-compatible SCAIL-2 Flow embeds/sampler/VAE decode nodes for
#     MODEL/VAE/CONDITIONING/LATENT workflow chains.
#   - Adds WanAnimatePlus Easy Sampler and Easy SamplerSettings nodes to the public mappings.
#   - Includes WanAnimatePlus Uni3C loader/embeds nodes for same-package sampling compatibility.
#   - Forces registered node categories to WanAnimatePlus.
#   - Keeps WanAnimatePlus nodes isolated from original WanVideoWrapper node names.
# Licensed under the Apache License, Version 2.0
try:
    from .utils import check_duplicate_nodes, log, color_text
    duplicate_dirs = check_duplicate_nodes()
    if duplicate_dirs:
        warning_msg = f"WARNING:  Found {len(duplicate_dirs)} other WanAnimatePlus directories:\n"
        for dir_path in duplicate_dirs:
            warning_msg += f"  - {color_text(dir_path, 'yellow')}\n"
        log.warning(color_text(warning_msg + "Please remove duplicates to avoid possible conflicts.", "red"))
except Exception:
    pass

from .nodes import (
    WanAnimatePlusBernini,
    WanAnimatePlusEverAnimateEmbeds,
    WanAnimatePlusSCAIL2Embeds,
    WanAnimatePlusSCAIL2FlowEmbeds,
    WanAnimatePlusSCAIL2TwoPhaseSettings,
    WanAnimatePlusVAEDecode,
    WanVideoAnimateEmbeds,
    WanVideoClipVisionEncode,
    WanVideoClipVisionEncodeV2,
    WanVideoContextOptions,
    WanVideoDecode,
    WanVideoEncode,
    WanVideoSetBlockSwap,
    WanVideoTextEncodeCached,
)
from .nodes_sampler import (
    WanAnimatePlusEasySampler,
    WanAnimatePlusEasySamplerSettings,
    WanAnimatePlusSCAIL2FlowSampler,
    WanVideoSampler,
    WanVideoSamplerv2,
    WanVideoScheduler,
    WanVideoSchedulerv2,
    WanVideoSamplerSettings,
    WanVideoSamplerFromSettings,
    WanVideoSamplerExtraArgs,
)
from .nodes_model_loading import (
    WanVideoBlockSwap,
    WanVideoLoraSelect,
    WanVideoLoraSelectMulti,
    WanVideoModelLoader,
    WanVideoSetLoRAs,
    WanVideoTorchCompileSettings,
    WanVideoVAELoader,
)
from .uni3c.nodes import WanVideoUni3C_ControlnetLoader, WanVideoUni3C_embeds

NODE_CLASS_MAPPINGS = {
    "WanAnimatePlus AnimateEmbeds": WanVideoAnimateEmbeds,
    "WanAnimatePlus Sampler": WanVideoSampler,
    "WanAnimatePlus Samplerv2": WanVideoSamplerv2,
    "WanAnimatePlus Easy Sampler": WanAnimatePlusEasySampler,
    "WanAnimatePlus Easy SamplerSettings": WanAnimatePlusEasySamplerSettings,
    "WanAnimatePlus SCAIL_2 Flow Sampler": WanAnimatePlusSCAIL2FlowSampler,
    "WanAnimatePlus Scheduler": WanVideoScheduler,
    "WanAnimatePlus Schedulerv2": WanVideoSchedulerv2,
    "WanAnimatePlus SamplerSettings": WanVideoSamplerSettings,
    "WanAnimatePlus SamplerFromSettings": WanVideoSamplerFromSettings,
    "WanAnimatePlus Encode": WanVideoEncode,
    "WanAnimatePlus Decode": WanVideoDecode,
    "WanAnimatePlus VAE Decode": WanAnimatePlusVAEDecode,
    "WanAnimatePlus ModelLoader": WanVideoModelLoader,
    "WanAnimatePlus VAELoader": WanVideoVAELoader,
    "WanAnimatePlus ContextOptions": WanVideoContextOptions,
    "WanAnimatePlus TextEncodeCached": WanVideoTextEncodeCached,
    "WanAnimatePlus ClipVisionEncode": WanVideoClipVisionEncode,
    "WanAnimatePlus ClipVisionEncode V2": WanVideoClipVisionEncodeV2,
    "WanAnimatePlus LoraSelect": WanVideoLoraSelect,
    "WanAnimatePlus LoraSelectMulti": WanVideoLoraSelectMulti,
    "WanAnimatePlus SetLoRAs": WanVideoSetLoRAs,
    "WanAnimatePlus BlockSwap": WanVideoBlockSwap,
    "WanAnimatePlus SetBlockSwap": WanVideoSetBlockSwap,
    "WanAnimatePlus TorchCompileSettings": WanVideoTorchCompileSettings,
    "WanAnimatePlus Uni3C ControlnetLoader": WanVideoUni3C_ControlnetLoader,
    "WanAnimatePlus Uni3C Embeds": WanVideoUni3C_embeds,
    "WanAnimatePlus SamplerExtraArgs": WanVideoSamplerExtraArgs,
    "WanAnimatePlus Bernini": WanAnimatePlusBernini,
    "WanAnimatePlus EverAnimate Embeds": WanAnimatePlusEverAnimateEmbeds,
    "WanAnimatePlus SCAIL_2 Embeds": WanAnimatePlusSCAIL2Embeds,
    "WanAnimatePlus SCAIL_2 Flow Embeds": WanAnimatePlusSCAIL2FlowEmbeds,
    "WanAnimatePlus SCAIL_2 TwoPhase Settings": WanAnimatePlusSCAIL2TwoPhaseSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}

for node_class in NODE_CLASS_MAPPINGS.values():
    node_class.CATEGORY = "WanAnimatePlus"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
