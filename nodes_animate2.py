# Copyright (c) 2026 wuwukasi/wuwukaka.
# Official-ComfyUI-compatible Animate2 embeds node.
# Licensed under the Apache License, Version 2.0
import logging

import torch

from .animate2_flow import (
    ANIMATE2_CACHE_PATH_KEY,
    ANIMATE2_DEFERRED_BUILD_KEY,
    ANIMATE2_RUNTIME_VAE_KEY,
    build_conditioning_and_latent,
    build_deferred_latent,
    make_runtime,
    release_flow_vae,
)
from .scail2_loop_inputs import remove_input_cache_dir


log = logging.getLogger(__name__)


class WanAnimatePlusAnimate2Embeds:
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
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "positive_pose": ("CONDITIONING",),
                "ref_image": ("IMAGE",),
                "bg_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "prefix_frames": ("IMAGE",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePlus"
    DESCRIPTION = (
        "Official ComfyUI-compatible Animate2 conditioning. Outputs CONDITIONING and LATENT. "
        "prefix_frames are frozen identity latents trimmed after sampling. "
        "bg_image fills the unknown canvas instead of mid-grey. "
        "Internal looping is handled by WanAnimatePlus Animate2 Sampler."
    )

    @staticmethod
    def _input_summary(label, value):
        if value is None:
            return f"{label}=none"
        if isinstance(value, torch.Tensor):
            if value.ndim >= 4:
                return f"{label}={value.shape[0]}@{value.shape[2]}x{value.shape[1]}"
            return f"{label}=shape{tuple(value.shape)}"
        if isinstance(value, dict) and value.get("type") == "scail2_loop_sequence":
            shape = tuple(value.get("shape", ()))
            return f"{label}=seq{shape}"
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
        clip_vision_output=None,
        clip_vision_output_pose=None,
        positive_pose=None,
        ref_image=None,
        bg_image=None,
        pose_images=None,
        prefix_frames=None,
        tiled_vae=False,
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
            tiled_vae,
            ref_image=ref_image,
            bg_image=bg_image,
            pose_images=pose_images,
            prefix_frames=prefix_frames,
            clip_vision_output=clip_vision_output,
            clip_vision_output_pose=clip_vision_output_pose,
            positive_pose=positive_pose,
        )
        mode = "loop deferred build" if runtime.get("looping", False) else "one-shot full build"
        log.info(
            f"Animate2 Embeds: {mode}, "
            f"{runtime['requested_output_frames']} frames, "
            f"window={runtime['frame_window_size']}, {runtime['width']}x{runtime['height']}, "
            f"batch={runtime['batch_size']}, tiled_vae={runtime['tiled_vae']}"
        )
        log.info(
            "Animate2 inputs: "
            + ", ".join([
                self._input_summary("ref", ref_image),
                self._input_summary("bg", bg_image),
                self._input_summary("pose", runtime.get("pose_images")),
                self._input_summary("prefix", prefix_frames),
                self._input_summary("clip_vision", clip_vision_output),
                self._input_summary("clip_vision_pose", clip_vision_output_pose),
                self._input_summary("positive_pose", positive_pose),
            ])
        )
        try:
            if runtime.get("looping", False):
                runtime[ANIMATE2_DEFERRED_BUILD_KEY] = "loop"
                runtime[ANIMATE2_RUNTIME_VAE_KEY] = vae
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
        except BaseException:
            remove_input_cache_dir(runtime.get(ANIMATE2_CACHE_PATH_KEY), log)
            raise
        return (positive, negative, latent)
