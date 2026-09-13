# Copyright (c) 2026 wuwukasi/wuwukaka.
# Official-ComfyUI-compatible Animate2 flow helpers for WanAnimatePlus.
#   - Builds WanAnimate2 CONDITIONING (concat_latent_image / concat_mask /
#     pose_video_latent) from Plus-style ref / bg / prefix inputs.
#   - Internal loop uses a 5-frame concat/mask handoff, not official 1-frame
#     continue_motion node wiring.
# Licensed under the Apache License, Version 2.0
import gc
import logging

import torch

import node_helpers
from comfy import model_management as mm
from comfy.utils import common_upscale

from .scail2_flow import (
    align_4n1,
    decode_latent_to_images,
    latent_frames_for_pixels,
    release_flow_vae,
    slice_fit_sequence,
    take_tail_with_front_pad,
)
from .scail2_loop_inputs import (
    ANIMATE2_CACHE_DIR_PREFIX,
    ANIMATE2_CACHE_MARKER,
    LoopSequenceReader,
    build_resized_bhwc_sequence,
    estimate_resized_bhwc_bytes,
    is_loop_sequence,
    memory_allows_allocation,
    remove_input_cache_dir,
    safe_frame_chunk_size,
)


ANIMATE2_RUNTIME_KEY = "_wananimateplus_animate2_flow"
ANIMATE2_DEFERRED_BUILD_KEY = "_wananimateplus_animate2_deferred_build"
ANIMATE2_RUNTIME_VAE_KEY = "_wananimateplus_animate2_vae"
ANIMATE2_STATIC_CACHE_KEY = "_wananimateplus_animate2_static_cache"
ANIMATE2_LOGGED_KEYS = "_wananimateplus_animate2_logged_keys"
ANIMATE2_POSE_READER_KEY = "_wananimateplus_animate2_pose_reader"
ANIMATE2_CACHE_PATH_KEY = "_wananimateplus_animate2_cache_path"

PREVIOUS_FRAME_COUNT = 5
MAX_PREFIX_FRAMES = 5


log = logging.getLogger(__name__)


def _maybe_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _log_once(runtime, key, message):
    logged = runtime.setdefault(ANIMATE2_LOGGED_KEYS, set())
    if key in logged:
        return
    logged.add(key)
    log.info(message)


def resize_bhwc_area(images, width, height):
    images = images[:, :, :, :3]
    if images.shape[1] == height and images.shape[2] == width:
        return images
    chunk_size = safe_frame_chunk_size(
        images.shape[0],
        images.shape[1],
        images.shape[2],
        width,
        height,
        3,
    )
    if images.shape[0] <= chunk_size:
        return common_upscale(images.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
    resized = []
    for start in range(0, images.shape[0], chunk_size):
        chunk = images[start:start + chunk_size]
        resized.append(common_upscale(chunk.movedim(-1, 1), width, height, "area", "center").movedim(1, -1))
    return torch.cat(resized, dim=0)


def encode_pixels(vae, images, tiled_vae=False):
    images = images[:, :, :, :3]
    if tiled_vae and hasattr(vae, "encode_tiled"):
        return vae.encode_tiled(images)
    return vae.encode(images)


def clean_animate2_runtime_for_output(runtime):
    if runtime is None:
        return runtime
    cleaned = dict(runtime)
    cleaned.pop(ANIMATE2_RUNTIME_VAE_KEY, None)
    cleaned.pop(ANIMATE2_STATIC_CACHE_KEY, None)
    cleaned.pop(ANIMATE2_LOGGED_KEYS, None)
    cleaned.pop(ANIMATE2_POSE_READER_KEY, None)
    return cleaned


def prepare_pose_sequence(pose_images, width, height, frame_window_size, looping, logger=None):
    if pose_images is None:
        return None, None
    pose_images = pose_images[:, :, :, :3]
    if not looping:
        return _maybe_cpu(pose_images), None

    expected_bytes = estimate_resized_bhwc_bytes(pose_images, width, height)
    cache_state = {"path": None, "cleaned": False}
    segment_frames = max(1, int(frame_window_size) - PREVIOUS_FRAME_COUNT)
    try:
        if memory_allows_allocation(expected_bytes) and pose_images.shape[1] == height and pose_images.shape[2] == width:
            stored = pose_images.detach().to(mm.unet_offload_device())
            try:
                if stored.untyped_storage().data_ptr() == pose_images.untyped_storage().data_ptr():
                    stored = stored.clone()
            except Exception:
                pass
            return stored.contiguous(), None

        sequence = build_resized_bhwc_sequence(
            pose_images,
            width,
            height,
            "area",
            "center",
            cache_state,
            "pose",
            segment_frames,
            logger=logger,
            cache_prefix=ANIMATE2_CACHE_DIR_PREFIX,
            cache_marker=ANIMATE2_CACHE_MARKER,
            cache_note="WanAnimatePlus Animate2 loop temporary input cache\n",
        )
    except BaseException:
        remove_input_cache_dir(cache_state.get("path"), logger)
        raise

    cache_path = cache_state.get("path")
    if cache_path is not None and logger is not None:
        logger.info(f"Animate2 loop: using mixed resized pose inputs with temporary cache {cache_path}")
    elif logger is not None:
        logger.info("Animate2 loop: using mixed resized pose inputs fully in memory")
    return sequence, cache_path


def make_runtime(
    width,
    height,
    num_frames,
    frame_window_size,
    batch_size,
    pose_strength,
    ref_strength,
    tiled_vae,
    ref_image=None,
    bg_image=None,
    pose_images=None,
    prefix_frames=None,
    clip_vision_output=None,
    clip_vision_output_pose=None,
    positive_pose=None,
):
    width = (int(width) // 32) * 32
    height = (int(height) // 32) * 32
    requested_frames = align_4n1(num_frames)
    frame_window_size = min(align_4n1(frame_window_size), requested_frames)
    looping = frame_window_size != requested_frames

    pose_source, cache_path = prepare_pose_sequence(
        pose_images,
        width,
        height,
        frame_window_size,
        looping,
        logger=log,
    )

    if prefix_frames is not None:
        prefix_frames = prefix_frames[:, :, :, :3]
        if prefix_frames.shape[0] > MAX_PREFIX_FRAMES:
            log.warning(f"Animate2 prefix has {prefix_frames.shape[0]} images, max {MAX_PREFIX_FRAMES}. Truncating.")
            prefix_frames = prefix_frames[:MAX_PREFIX_FRAMES]

    if bg_image is not None:
        if bg_image.shape[0] > 1:
            log.warning("Animate2 bg_image accepts one image; using the first frame")
        bg_image = bg_image[:1, :, :, :3]

    runtime = {
        "width": width,
        "height": height,
        "num_frames": requested_frames,
        "requested_output_frames": requested_frames,
        "frame_window_size": frame_window_size,
        "looping": looping,
        "previous_frame_count": PREVIOUS_FRAME_COUNT,
        "batch_size": max(1, int(batch_size)),
        "pose_strength": float(pose_strength),
        "ref_strength": float(ref_strength),
        "tiled_vae": bool(tiled_vae),
        "ref_image": _maybe_cpu(ref_image),
        "bg_image": _maybe_cpu(bg_image),
        "pose_images": pose_source if is_loop_sequence(pose_source) else _maybe_cpu(pose_source),
        "prefix_frames": _maybe_cpu(prefix_frames),
        "clip_vision_output": clip_vision_output,
        "clip_vision_output_pose": clip_vision_output_pose,
        "positive_pose": positive_pose,
        ANIMATE2_CACHE_PATH_KEY: cache_path,
    }
    return runtime


def _empty_ref_image(width, height, ref_image):
    if ref_image is None:
        return torch.zeros((1, height, width, 3))
    return ref_image[:1, :, :, :3]


def _get_static_cache(runtime, vae):
    cache = runtime.get(ANIMATE2_STATIC_CACHE_KEY, None)
    if cache is not None:
        return cache

    width = runtime["width"]
    height = runtime["height"]
    tiled_vae = runtime["tiled_vae"]
    ref_image = resize_bhwc_area(_empty_ref_image(width, height, runtime.get("ref_image")), width, height)
    ref_latent = encode_pixels(vae, ref_image, tiled_vae=tiled_vae)
    trim_latent = int(ref_latent.shape[2])
    parts = [ref_latent]

    prefix_frames = runtime.get("prefix_frames")
    prefix_count = 0
    if prefix_frames is not None and prefix_frames.shape[0] > 0:
        for i in range(prefix_frames.shape[0]):
            prefix = resize_bhwc_area(prefix_frames[i:i + 1, :, :, :3], width, height)
            prefix_latent = encode_pixels(vae, prefix, tiled_vae=tiled_vae)
            parts.append(prefix_latent)
            trim_latent += int(prefix_latent.shape[2])
            prefix_count += int(prefix_latent.shape[2])

    identity_latents = torch.cat(parts, dim=2)
    cache = {
        "identity_latents": identity_latents.detach().cpu(),
        "trim_latent": trim_latent,
        "prefix_latent_count": prefix_count,
    }
    runtime["trim_latent"] = trim_latent
    runtime[ANIMATE2_STATIC_CACHE_KEY] = cache
    _log_once(
        runtime,
        "identity_latents",
        f"Animate2 identity latents shape: {tuple(identity_latents.shape)} (trim_latent={trim_latent})",
    )
    return cache


def _canvas_image(runtime, length, previous_frames=None):
    width = runtime["width"]
    height = runtime["height"]
    bg = runtime.get("bg_image")
    if bg is not None:
        bg = resize_bhwc_area(bg[:1, :, :, :3], width, height)
        image = bg.repeat(length, 1, 1, 1)
    else:
        image = torch.ones((length, height, width, 3)) * 0.5

    handoff_pixels = 0
    if previous_frames is not None and previous_frames.shape[0] > 0:
        prev = take_tail_with_front_pad(previous_frames, runtime["previous_frame_count"])
        prev = resize_bhwc_area(prev[:, :, :, :3], width, height)
        take = min(prev.shape[0], length)
        image = image.to(device=prev.device, dtype=prev.dtype)
        image[:take] = prev[:take]
        handoff_pixels = take
    return image, handoff_pixels


def _pose_slice(runtime, start_frame, length):
    pose = runtime.get("pose_images")
    if pose is None:
        return None
    reader = runtime.get(ANIMATE2_POSE_READER_KEY)
    if reader is None and is_loop_sequence(pose):
        reader = LoopSequenceReader(pose, 0, mm.unet_offload_device(), "pose")
        runtime[ANIMATE2_POSE_READER_KEY] = reader
    if reader is not None:
        return reader.slice(start_frame, length)
    return slice_fit_sequence(pose, start_frame, length)


def _apply_pose_values(positive, negative, pose_values):
    if not pose_values:
        return positive, negative
    return (
        node_helpers.conditioning_set_values(positive, pose_values),
        node_helpers.conditioning_set_values(negative, pose_values),
    )


def build_conditioning_and_latent(
    positive,
    negative,
    vae,
    runtime,
    start_frame=0,
    length=None,
    previous_frames=None,
    include_runtime=True,
):
    width = runtime["width"]
    height = runtime["height"]
    if length is None:
        length = runtime["num_frames"]
    length = align_4n1(length)
    start_frame = max(0, int(start_frame))

    video_t = latent_frames_for_pixels(length)
    lat_h = height // 8
    lat_w = width // 8
    batch_size = runtime["batch_size"]
    tiled_vae = runtime["tiled_vae"]

    cache = _get_static_cache(runtime, vae)
    identity_latents = cache["identity_latents"]
    trim_latent = int(cache["trim_latent"])
    prefix_latent_count = int(cache.get("prefix_latent_count", 0))

    canvas, handoff_pixels = _canvas_image(runtime, length, previous_frames=previous_frames)
    canvas_latent = encode_pixels(vae, canvas, tiled_vae=tiled_vae)
    concat_latent_image = torch.cat(
        (identity_latents.to(device=canvas_latent.device, dtype=canvas_latent.dtype), canvas_latent),
        dim=2,
    )

    handoff_latents = latent_frames_for_pixels(handoff_pixels) if handoff_pixels > 0 else 0
    concat_mask = torch.ones(
        (1, 1, video_t + trim_latent, lat_h, lat_w),
        device=concat_latent_image.device,
        dtype=concat_latent_image.dtype,
    )
    concat_mask[:, :, :trim_latent + handoff_latents] = 0.0

    values = {
        "concat_latent_image": concat_latent_image,
        "concat_mask": concat_mask,
    }
    clip_vision_output = runtime.get("clip_vision_output")
    if clip_vision_output is not None:
        values["clip_vision_output"] = clip_vision_output
    if runtime["ref_strength"] != 1.0:
        values["reference_strength"] = runtime["ref_strength"]

    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)

    pose_values = {}
    pose_slice = _pose_slice(runtime, start_frame, length)
    if pose_slice is not None:
        pose_slice = resize_bhwc_area(pose_slice[:, :, :, :3], width, height)
        pose_latent = encode_pixels(vae, pose_slice, tiled_vae=tiled_vae)
        if prefix_latent_count > 0:
            pad = pose_latent[:, :, :1].repeat(1, 1, prefix_latent_count, 1, 1)
            pose_latent = torch.cat([pad, pose_latent], dim=2)
        expected_pose_t = (video_t + trim_latent) - 1
        if pose_latent.shape[2] < expected_pose_t:
            pad_t = expected_pose_t - pose_latent.shape[2]
            pose_latent = torch.cat([pose_latent, pose_latent[:, :, -1:].repeat(1, 1, pad_t, 1, 1)], dim=2)
        elif pose_latent.shape[2] > expected_pose_t:
            pose_latent = pose_latent[:, :, :expected_pose_t]
        pose_values["pose_video_latent"] = pose_latent
        _log_once(runtime, "pose_latent", f"Animate2 pose latent shape: {tuple(pose_latent.shape)}")

    pose_clip = runtime.get("clip_vision_output_pose")
    if pose_clip is None:
        pose_clip = clip_vision_output
    if pose_clip is not None:
        pose_values["clip_vision_output_pose"] = pose_clip

    pose_cond = runtime.get("positive_pose")
    if pose_cond is None:
        pose_cond = positive
    if pose_cond is not None and len(pose_cond) > 0:
        pose_values["cross_attn_pose"] = pose_cond[0][0]

    if runtime["pose_strength"] != 1.0:
        pose_values["pose_strength"] = runtime["pose_strength"]

    positive, negative = _apply_pose_values(positive, negative, pose_values)

    latent = torch.zeros(
        (batch_size, 16, video_t + trim_latent, lat_h, lat_w),
        device=mm.intermediate_device(),
    )
    out_latent = {"samples": latent, "trim_latent": trim_latent}
    if include_runtime:
        out_latent[ANIMATE2_RUNTIME_KEY] = runtime
    out_latent["output_frame_count"] = int(runtime["requested_output_frames"])
    return positive, negative, out_latent


def build_deferred_latent(runtime, length=None, include_runtime=True):
    width = runtime["width"]
    height = runtime["height"]
    if length is None:
        length = runtime["frame_window_size"] if runtime.get("looping", False) else runtime["num_frames"]
    length = align_4n1(length)
    video_t = latent_frames_for_pixels(length)
    prefix_count = 0
    prefix_frames = runtime.get("prefix_frames")
    if prefix_frames is not None:
        prefix_count = min(int(prefix_frames.shape[0]), MAX_PREFIX_FRAMES)
    trim_latent = 1 + prefix_count
    runtime["trim_latent"] = trim_latent
    latent = torch.zeros(
        (runtime["batch_size"], 16, video_t + trim_latent, height // 8, width // 8),
        device=mm.intermediate_device(),
    )
    out_latent = {"samples": latent, "trim_latent": trim_latent}
    if include_runtime:
        out_latent[ANIMATE2_RUNTIME_KEY] = runtime
    out_latent["output_frame_count"] = int(runtime["requested_output_frames"])
    return out_latent


def trim_and_decode(vae, sampled, runtime):
    samples = sampled["samples"] if isinstance(sampled, dict) else sampled
    trim_latent = int(runtime.get("trim_latent", sampled.get("trim_latent", 0) if isinstance(sampled, dict) else 0) or 0)
    if trim_latent > 0:
        if samples.shape[2] <= trim_latent:
            raise ValueError(
                f"Animate2 sampled latent time {samples.shape[2]} is too short to trim {trim_latent} identity frames"
            )
        samples = samples[:, :, trim_latent:].contiguous()
    return decode_latent_to_images(vae, {"samples": samples}, tiled_vae=runtime.get("tiled_vae", False))


def attach_pose_reader(runtime):
    pose = runtime.get("pose_images")
    if pose is None:
        return None
    reader = runtime.get(ANIMATE2_POSE_READER_KEY)
    if reader is None:
        reader = LoopSequenceReader(pose, 0, mm.unet_offload_device(), "pose")
        runtime[ANIMATE2_POSE_READER_KEY] = reader
    return reader


def cleanup_loop_inputs(runtime, logger=None):
    runtime.pop(ANIMATE2_POSE_READER_KEY, None)
    path = runtime.get(ANIMATE2_CACHE_PATH_KEY)
    runtime[ANIMATE2_CACHE_PATH_KEY] = remove_input_cache_dir(path, logger)
