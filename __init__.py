# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Registers a complete WanAnimatePlus workflow node chain with renamed node keys
#   - Includes WanAnimatePlus Uni3C loader/embeds nodes for same-package sampling compatibility
#   - Keeps WanAnimatePlus nodes isolated from original WanVideoWrapper node names
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
    WanVideoAnimateEmbeds,
    WanVideoClipVisionEncode,
    WanVideoContextOptions,
    WanVideoDecode,
    WanVideoEncode,
    WanVideoSetBlockSwap,
    WanVideoTextEncodeCached,
)
from .nodes_sampler import (
    WanVideoSampler,
    WanVideoSamplerv2,
    WanVideoScheduler,
    WanVideoSchedulerv2,
    WanVideoSamplerSettings,
    WanVideoSamplerFromSettings,
)
from .nodes_model_loading import (
    WanVideoBlockSwap,
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
    "WanAnimatePlus Scheduler": WanVideoScheduler,
    "WanAnimatePlus Schedulerv2": WanVideoSchedulerv2,
    "WanAnimatePlus SamplerSettings": WanVideoSamplerSettings,
    "WanAnimatePlus SamplerFromSettings": WanVideoSamplerFromSettings,
    "WanAnimatePlus Encode": WanVideoEncode,
    "WanAnimatePlus Decode": WanVideoDecode,
    "WanAnimatePlus ModelLoader": WanVideoModelLoader,
    "WanAnimatePlus VAELoader": WanVideoVAELoader,
    "WanAnimatePlus ContextOptions": WanVideoContextOptions,
    "WanAnimatePlus TextEncodeCached": WanVideoTextEncodeCached,
    "WanAnimatePlus ClipVisionEncode": WanVideoClipVisionEncode,
    "WanAnimatePlus LoraSelectMulti": WanVideoLoraSelectMulti,
    "WanAnimatePlus SetLoRAs": WanVideoSetLoRAs,
    "WanAnimatePlus BlockSwap": WanVideoBlockSwap,
    "WanAnimatePlus SetBlockSwap": WanVideoSetBlockSwap,
    "WanAnimatePlus TorchCompileSettings": WanVideoTorchCompileSettings,
    "WanAnimatePlus Uni3C ControlnetLoader": WanVideoUni3C_ControlnetLoader,
    "WanAnimatePlus Uni3C Embeds": WanVideoUni3C_embeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}

for node_class in NODE_CLASS_MAPPINGS.values():
    node_class.CATEGORY = "WanAnimatePlus"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
