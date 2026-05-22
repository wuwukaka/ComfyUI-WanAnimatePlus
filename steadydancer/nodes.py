import os
import torch
import numpy as np
from ..utils import log

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


class WanVideoAddSteadyDancerEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "pose_latents_positive": ("LATENT",),
                    "pose_strength_spatial": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the pose embedding"}),
                    "pose_strength_temporal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the pose embedding"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                },
                "optional": {
                    "pose_latents_negative": ("LATENT",),
                    "clip_vision_embeds": ("WANVIDIMAGE_CLIPEMBEDS",),
                    
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, pose_latents_positive, pose_strength_spatial, pose_strength_temporal, start_percent=0.0, end_percent=1.0, pose_latents_negative=None, clip_vision_embeds=None):
        sdancer_embeds = {
            "cond_pos": pose_latents_positive["samples"][0],
            "cond_neg": pose_latents_negative["samples"][0] if pose_latents_negative else None,
            "pose_strength_spatial": pose_strength_spatial,
            "pose_strength_temporal": pose_strength_temporal,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "clip_fea": clip_vision_embeds,
        }

        updated = dict(embeds)
        updated["sdancer_embeds"] = sdancer_embeds
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddSteadyDancerEmbeds": WanVideoAddSteadyDancerEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddSteadyDancerEmbeds": "WanVideo Add SteadyDancer Embeds",
    }
