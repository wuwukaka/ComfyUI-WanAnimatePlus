# Copyright (c) 2026 wuwukasi/wuwukaka.
# Official-ComfyUI-compatible SCAIL-2 flow helpers for WanAnimatePlus.
#   - Bridges WanAnimatePlus SCAIL-2 conditioning to official ComfyUI
#     MODEL/VAE/CONDITIONING/LATENT/IMAGE workflow types.
#   - Preserves legacy WanAnimatePlus SCAIL-2 bg/prefix/reference ordering,
#     transition hard-freeze trimming, loop colormatch, and freeze-mask metadata.
#   - Provides shared Flow helper logic used by the official-compatible embeds,
#     sampler, and VAE decode nodes.
# Licensed under the Apache License, Version 2.0
import gc
import math
import logging
import numpy as np

import torch
import torch.nn.functional as F

import node_helpers
from comfy import model_management as mm
from comfy.utils import common_upscale


FLOW_RUNTIME_KEY = "_wananimateplus_scail2_flow"
FLOW_HANDOFF_MASK_KEY = "_wananimateplus_scail2_handoff_mask"
FLOW_FREEZE_MASK_KEY = "_wananimateplus_scail2_freeze_mask"
FLOW_DEFERRED_BUILD_KEY = "_wananimateplus_scail2_deferred_build"
FLOW_RUNTIME_VAE_KEY = "_wananimateplus_scail2_vae"
FLOW_STATIC_CACHE_KEY = "_wananimateplus_scail2_static_cache"
FLOW_LOGGED_KEYS = "_wananimateplus_scail2_logged_keys"


log = logging.getLogger(__name__)


def align_4n1(frames):
    frames = max(1, int(frames))
    return ((frames - 1) // 4) * 4 + 1


def latent_frames_for_pixels(frames):
    return ((max(1, int(frames)) - 1) // 4) + 1


def frame_to_latent_index(frame):
    frame = max(0, int(frame))
    return 0 if frame <= 0 else ((frame - 1) // 4) + 1


def fit_sequence(seq, target_len):
    if seq is None:
        return None
    target_len = max(1, int(target_len))
    if seq.shape[0] == target_len:
        return seq
    if seq.shape[0] > target_len:
        return seq[:target_len]
    if seq.shape[0] == 0:
        raise ValueError("Input sequence must contain at least one frame")
    return torch.cat([seq, seq[-1:].repeat(target_len - seq.shape[0], *([1] * (seq.ndim - 1)))], dim=0)


def slice_fit_sequence(seq, start, length):
    if seq is None:
        return None
    start = max(0, int(start))
    length = max(1, int(length))
    if seq.shape[0] <= start:
        return seq[-1:].repeat(length, *([1] * (seq.ndim - 1)))
    out = seq[start:start + length]
    if out.shape[0] < length:
        out = torch.cat([out, out[-1:].repeat(length - out.shape[0], *([1] * (out.ndim - 1)))], dim=0)
    return out


def resize_bhwc(images, width, height, mode="lanczos", crop="disabled"):
    images = images[:, :, :, :3]
    if images.shape[1] == height and images.shape[2] == width:
        return images
    return common_upscale(images.movedim(-1, 1), width, height, mode, crop).movedim(1, -1)


def normalize_mask_background(mask, white_background):
    mask = mask[:, :, :, :3]
    if white_background:
        bg = mask.amax(dim=-1, keepdim=True) <= 0.05
        return torch.where(bg, torch.ones_like(mask), mask)
    bg = mask.amin(dim=-1, keepdim=True) >= 0.95
    return torch.where(bg, torch.zeros_like(mask), mask)


def alpha_crop_with_mask(images, masks):
    if images is None or masks is None or images.shape[0] == 0 or masks.shape[0] == 0:
        return images
    count = min(images.shape[0], masks.shape[0])
    crop_mask = masks[:count, :, :, :3]
    if crop_mask.shape[1] != images.shape[1] or crop_mask.shape[2] != images.shape[2]:
        crop_mask = resize_bhwc(crop_mask, images.shape[2], images.shape[1], mode="nearest-exact")
    crop_mask = normalize_mask_background(crop_mask, white_background=False)
    is_char = (crop_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(device=images.device, dtype=images.dtype)
    out = images.clone()
    out[:count] = out[:count] * is_char
    return out


def take_tail_with_front_pad(images, count):
    count = max(1, int(count))
    images = images[:, :, :, :3]
    if images.shape[0] >= count:
        return images[-count:]
    return torch.cat([images[:1].repeat(count - images.shape[0], 1, 1, 1), images], dim=0)


def sample_reversed_prefix_frames(frames, count):
    count = max(0, int(count))
    if frames is None or count <= 0:
        return None if frames is None else frames[:0]
    indices = torch.arange(count, device=frames.device) * 2
    indices = torch.clamp(indices, max=frames.shape[0] - 1)
    return torch.flip(frames.index_select(0, indices), [0])


def encode_video_latent(vae, images, width, height, tiled_vae=False, scale=1.0, mode="lanczos"):
    pixels = resize_bhwc(images, width, height, mode=mode)
    if tiled_vae and hasattr(vae, "encode_tiled"):
        latent = vae.encode_tiled(pixels)
    else:
        latent = vae.encode(pixels)
    if scale != 1.0:
        latent = latent * float(scale)
    return latent


def extract_mask_to_28ch(rgb_video):
    # Colored RGB mask (T,H,W,3) in [0,1] -> (1,T_lat,28,H/8,W/8).
    rgb_video = rgb_video[:, :, :, :3]
    t, h, w, _ = rgb_video.shape
    on_thresh = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
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
    h_lat, w_lat = h, w
    for _ in range(3):
        h_lat = (h_lat + 1) // 2
        w_lat = (w_lat + 1) // 2
    binary_7ch = F.interpolate(binary_7ch, size=(h_lat, w_lat), mode="area")
    t_lat = latent_frames_for_pixels(t)
    padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    if padded.shape[0] < t_lat * 4:
        padded = torch.cat([padded, padded[-1:].repeat(t_lat * 4 - padded.shape[0], 1, 1, 1)], dim=0)
    padded = padded[:t_lat * 4]
    return padded.view(t_lat, 28, h_lat, w_lat).unsqueeze(0).contiguous()


def _empty_ref_mask_like(latent):
    return torch.zeros(
        1, 1, 28, latent.shape[-2], latent.shape[-1],
        device=latent.device, dtype=latent.dtype,
    )


def _maybe_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def release_flow_vae(vae):
    if vae is None:
        return
    patcher = getattr(vae, "patcher", None)
    if patcher is not None and hasattr(mm, "unload_model_and_clones"):
        try:
            mm.unload_model_and_clones(patcher)
        except Exception as e:
            log.debug(f"SCAIL-2 Flow: VAE targeted unload skipped: {e}")
    mm.soft_empty_cache()
    gc.collect()


def clean_flow_runtime_for_output(runtime):
    if runtime is None:
        return runtime
    cleaned = dict(runtime)
    cleaned.pop(FLOW_RUNTIME_VAE_KEY, None)
    cleaned.pop(FLOW_STATIC_CACHE_KEY, None)
    cleaned.pop(FLOW_LOGGED_KEYS, None)
    return cleaned


def _log_once(runtime, key, message):
    logged = runtime.setdefault(FLOW_LOGGED_KEYS, set())
    if key in logged:
        return
    logged.add(key)
    log.info(message)


def make_runtime(
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
    ref_image=None,
    bg_image=None,
    pose_images=None,
    prefix_frames=None,
    prefix_mask=None,
    transition_video=None,
    pose_image_mask=None,
    reference_image_mask=None,
    clip_vision_output=None,
):
    width = (int(width) // 32) * 32
    height = (int(height) // 32) * 32
    requested_frames = align_4n1(num_frames)
    frame_window_size = min(align_4n1(frame_window_size), requested_frames)
    looping = frame_window_size != requested_frames
    canvas_expansion_px = 21 if transition_video is not None else 0
    sample_frames = requested_frames
    if canvas_expansion_px:
        sample_frames += canvas_expansion_px
        trim = (sample_frames - 1) % 4
        if trim:
            if looping:
                sample_frames += 4 - trim
            else:
                sample_frames -= trim
        if pose_images is not None:
            pose_images = torch.cat([sample_reversed_prefix_frames(pose_images, canvas_expansion_px), pose_images], dim=0)
        if pose_image_mask is not None:
            pose_image_mask = torch.cat([sample_reversed_prefix_frames(pose_image_mask, canvas_expansion_px), pose_image_mask], dim=0)

    transition_match_ref = resize_bhwc(ref_image[:1, :, :, :3], width, height) if ref_image is not None else None
    transition_raw_last_frame = (
        transition_video[-1:, :, :, :3].detach().cpu()
        if transition_video is not None and looping and transition_colormatch != "auto_drift" else None
    )
    transition_raw_tail_means = None
    if transition_video is not None and looping and transition_colormatch == "auto_drift":
        transition_tail = take_tail_with_front_pad(transition_video[:, :, :, :3], canvas_expansion_px or 21)
        transition_tail = resize_bhwc(transition_tail, width, height)
        transition_raw_tail_means = auto_drift_tail_means(transition_tail)
    if transition_colormatch not in ("disabled", "auto_drift") and transition_match_ref is None and (
        transition_video is not None or (looping and loop_colormatch_reference == "main_ref_image")
    ):
        log.warning("SCAIL-2 transition_colormatch is enabled but ref_image is not connected. Skipping color match.")

    return {
        "width": width,
        "height": height,
        "num_frames": sample_frames,
        "requested_output_frames": requested_frames,
        "canvas_expansion_px": canvas_expansion_px,
        "frame_window_size": frame_window_size,
        "looping": looping,
        "previous_frame_count": 5,
        "batch_size": max(1, int(batch_size)),
        "pose_strength": float(pose_strength),
        "ref_strength": float(ref_strength),
        "replacement_mode": bool(replacement_mode),
        "tiled_vae": bool(tiled_vae),
        "transition_colormatch": transition_colormatch,
        "loop_colormatch_reference": loop_colormatch_reference,
        "prefix_alpha_crop": bool(prefix_alpha_crop),
        "preserve_main_ref_background": bool(preserve_main_ref_background),
        "single_frame_prefix_encoding": True,
        "ref_image": _maybe_cpu(ref_image),
        "bg_image": _maybe_cpu(bg_image),
        "pose_images": _maybe_cpu(pose_images),
        "prefix_frames": _maybe_cpu(prefix_frames),
        "prefix_mask": _maybe_cpu(prefix_mask),
        "transition_video": _maybe_cpu(transition_video),
        "pose_image_mask": _maybe_cpu(pose_image_mask),
        "reference_image_mask": _maybe_cpu(reference_image_mask),
        "transition_match_ref": _maybe_cpu(transition_match_ref),
        "transition_raw_last_frame": _maybe_cpu(transition_raw_last_frame),
        "transition_raw_tail_means": _maybe_cpu(transition_raw_tail_means),
        "clip_vision_output": clip_vision_output,
    }


def _prepare_reference_inputs(runtime):
    replacement_mode = runtime["replacement_mode"]
    prefix_alpha_crop = runtime["prefix_alpha_crop"]
    preserve_main_ref_background = runtime["preserve_main_ref_background"]
    crop_main_ref_background = (not replacement_mode) and (not preserve_main_ref_background)

    ref_image = runtime.get("ref_image")
    bg_image = runtime.get("bg_image")
    prefix_frames = runtime.get("prefix_frames")
    prefix_mask = runtime.get("prefix_mask")
    reference_image_mask = runtime.get("reference_image_mask")

    bg_active = bg_image is not None and not replacement_mode
    bg = bg_image[:1, :, :, :3] if bg_active else None
    if bg_image is not None and replacement_mode:
        log.info("SCAIL-2 bg_image is ignored in replacement mode")

    user_prefix_frames = prefix_frames
    user_prefix_mask = prefix_mask if user_prefix_frames is not None else None
    user_prefix_count = 0
    bg_mask = None

    if bg_active:
        if bg_image.shape[0] > 1:
            log.warning("SCAIL-2 bg_image accepts one image; using the first frame")
        if user_prefix_frames is not None:
            if user_prefix_frames.shape[0] > 4:
                log.warning("SCAIL-2 bg_image uses one prefix slot; truncating prefix_frames to 4 images")
            user_prefix_frames = user_prefix_frames[:4, :, :, :3]
            user_prefix_count = user_prefix_frames.shape[0]
            if bg.shape[1] != user_prefix_frames.shape[1] or bg.shape[2] != user_prefix_frames.shape[2]:
                bg = resize_bhwc(bg, user_prefix_frames.shape[2], user_prefix_frames.shape[1])
            bg = bg.to(device=user_prefix_frames.device, dtype=user_prefix_frames.dtype)

        if user_prefix_mask is not None:
            if user_prefix_mask.shape[0] > 4:
                log.warning("SCAIL-2 bg_image uses one prefix-mask slot; truncating prefix_mask to 4 images")
            user_prefix_mask = user_prefix_mask[:4, :, :, :3]
            bg_mask = torch.ones(
                1,
                user_prefix_mask.shape[1],
                user_prefix_mask.shape[2],
                3,
                device=user_prefix_mask.device,
                dtype=user_prefix_mask.dtype,
            )
        else:
            bg_mask = torch.ones_like(bg[:, :, :, :3])

    if user_prefix_frames is not None and user_prefix_mask is not None and (replacement_mode or prefix_alpha_crop):
        crop_count = min(user_prefix_frames.shape[0], user_prefix_mask.shape[0], 5)
        if crop_count > 0:
            cropped_prefix_frames = user_prefix_frames.clone()
            cropped_prefix_frames[:crop_count] = alpha_crop_with_mask(
                cropped_prefix_frames[:crop_count],
                user_prefix_mask[:crop_count],
            )
            user_prefix_frames = cropped_prefix_frames
            if bg_active and crop_main_ref_background:
                comp_bg = bg
                if comp_bg.shape[1] != user_prefix_frames.shape[1] or comp_bg.shape[2] != user_prefix_frames.shape[2]:
                    comp_bg = resize_bhwc(comp_bg, user_prefix_frames.shape[2], user_prefix_frames.shape[1])
                comp_bg = comp_bg.to(device=user_prefix_frames.device, dtype=user_prefix_frames.dtype)
                for i in range(crop_count):
                    mask = normalize_mask_background(user_prefix_mask[i:i + 1, :, :, :3], white_background=False)
                    is_bg = (mask[..., :3].max(dim=-1, keepdim=True).values <= 0.1).to(
                        device=user_prefix_frames.device, dtype=user_prefix_frames.dtype
                    )
                    user_prefix_frames[i:i + 1] = user_prefix_frames[i:i + 1] + comp_bg * is_bg
                log.info("SCAIL-2 prefix frames composited onto bg; masks will be inverted")

    prefix_frames = user_prefix_frames
    prefix_mask_for_prefix = user_prefix_mask
    if bg_active:
        if user_prefix_mask is not None:
            if user_prefix_mask.shape[0] < user_prefix_count:
                empty_user_masks = torch.zeros(
                    user_prefix_count - user_prefix_mask.shape[0],
                    user_prefix_mask.shape[1],
                    user_prefix_mask.shape[2],
                    3,
                    device=user_prefix_mask.device,
                    dtype=user_prefix_mask.dtype,
                )
                prefix_mask_for_prefix = torch.cat([user_prefix_mask, empty_user_masks], dim=0)
            elif user_prefix_mask.shape[0] > user_prefix_count:
                prefix_mask_for_prefix = user_prefix_mask[:user_prefix_count]
            else:
                prefix_mask_for_prefix = user_prefix_mask
        elif user_prefix_count > 0:
            prefix_mask_for_prefix = torch.zeros(
                user_prefix_count,
                bg_mask.shape[1],
                bg_mask.shape[2],
                3,
                device=bg_mask.device,
                dtype=bg_mask.dtype,
            )
        else:
            prefix_mask_for_prefix = None

    refs = []
    masks = []

    if prefix_frames is not None:
        pf = prefix_frames[:5, :, :, :3]
        pm = prefix_mask_for_prefix[:pf.shape[0], :, :, :3] if prefix_mask_for_prefix is not None else None
        for i in range(pf.shape[0]):
            refs.append(pf[i:i + 1])
            if pm is not None and i < pm.shape[0]:
                mask = pm[i:i + 1]
                mask = normalize_mask_background(mask, white_background=not (replacement_mode or prefix_alpha_crop))
                if bg_active and crop_main_ref_background:
                    is_bg = mask.amax(dim=-1, keepdim=True) <= 0.1
                    mask = torch.where(is_bg, torch.ones_like(mask), mask)
                masks.append(mask)
            else:
                masks.append(None)

    if ref_image is not None:
        ri = ref_image[:1, :, :, :3]
        rm = reference_image_mask[:1, :, :, :3] if reference_image_mask is not None else None
        if (replacement_mode or crop_main_ref_background) and rm is not None:
            rm_norm = normalize_mask_background(rm, white_background=False)
            if rm_norm.shape[1] != ri.shape[1] or rm_norm.shape[2] != ri.shape[2]:
                rm_norm = resize_bhwc(rm_norm, ri.shape[2], ri.shape[1], mode="nearest-exact")
            is_char = (rm_norm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ri)
            ri = ri * is_char
            if bg_active and crop_main_ref_background:
                bg_resized = bg
                if bg_resized.shape[1] != ri.shape[1] or bg_resized.shape[2] != ri.shape[2]:
                    bg_resized = resize_bhwc(bg_resized, ri.shape[2], ri.shape[1])
                ri = ri + bg_resized.to(ri) * (1.0 - is_char)
                log.info("SCAIL-2 ref image composited onto bg")
        refs.append(ri)
        if rm is not None:
            rm = normalize_mask_background(rm, white_background=not (replacement_mode or crop_main_ref_background))
            if bg_active and crop_main_ref_background:
                is_bg = rm.amax(dim=-1, keepdim=True) <= 0.1
                rm = torch.where(is_bg, torch.ones_like(rm), rm)
            masks.append(rm)
        else:
            masks.append(None)

    if bg_active:
        refs.append(bg)
        masks.append(bg_mask)

    return refs, masks


def _rotate_reference_latents_for_official(final_order_latents):
    if len(final_order_latents) <= 1:
        return final_order_latents
    return final_order_latents[-1:] + final_order_latents[:-1]


def _get_static_conditioning_cache(runtime, vae):
    cache = runtime.get(FLOW_STATIC_CACHE_KEY, None)
    if cache is not None:
        return cache

    cache = {"ref_mask_flag": not runtime["replacement_mode"]}
    refs, ref_masks = _prepare_reference_inputs(runtime)
    ref_latents = []
    mask_latents = []
    width = runtime["width"]
    height = runtime["height"]
    for ref, mask in zip(refs, ref_masks):
        ref_lat = encode_video_latent(
            vae,
            ref,
            width,
            height,
            runtime["tiled_vae"],
            scale=runtime["ref_strength"],
            mode="lanczos",
        )
        ref_latents.append(ref_lat)
        if mask is not None:
            mask_resized = resize_bhwc(mask, width, height, mode="bicubic")
            mask_latents.append(extract_mask_to_28ch(mask_resized).to(device=ref_lat.device, dtype=ref_lat.dtype))
        else:
            mask_latents.append(_empty_ref_mask_like(ref_lat))

    if ref_latents:
        cache["reference_latents"] = [
            _maybe_cpu(lat) for lat in _rotate_reference_latents_for_official(ref_latents)
        ]
        _log_once(
            runtime,
            "reference_latents",
            f"SCAIL-2 Flow reference latents: {len(ref_latents)} item(s), first shape {tuple(ref_latents[0].shape)}",
        )
        if any(mask is not None for mask in ref_masks):
            cache["ref_mask_prefix"] = _maybe_cpu(torch.cat(mask_latents, dim=1))
            _log_once(
                runtime,
                "reference_mask_latents",
                f"SCAIL-2 Flow reference mask latents shape: {tuple(cache['ref_mask_prefix'].shape)}",
            )

    clip_vision_output = runtime.get("clip_vision_output")
    if clip_vision_output is not None:
        cache["clip_vision_output"] = clip_vision_output

    runtime[FLOW_STATIC_CACHE_KEY] = cache
    return cache


def _static_conditioning_values_for_length(runtime, vae, t_lat):
    cache = _get_static_conditioning_cache(runtime, vae)
    values = {"ref_mask_flag": cache.get("ref_mask_flag", not runtime["replacement_mode"])}
    if "reference_latents" in cache:
        values["reference_latents"] = cache["reference_latents"]
    ref_mask_prefix = cache.get("ref_mask_prefix", None)
    if ref_mask_prefix is not None:
        zeros = torch.zeros(
            1, t_lat, 28, ref_mask_prefix.shape[-2], ref_mask_prefix.shape[-1],
            device=ref_mask_prefix.device, dtype=ref_mask_prefix.dtype,
        )
        values["ref_mask_28ch"] = torch.cat([ref_mask_prefix, zeros], dim=1)
    if "clip_vision_output" in cache:
        values["clip_vision_output"] = cache["clip_vision_output"]
    return values


def _prepare_transition_freeze(runtime, vae):
    canvas_expansion_px = int(runtime.get("canvas_expansion_px", 0) or 0)
    transition_video = runtime.get("transition_video", None)
    if canvas_expansion_px <= 0 or transition_video is None:
        return None, None
    if runtime.get("transition_freeze_latents", None) is not None:
        return runtime["transition_freeze_latents"], runtime["transition_freeze_mask"]

    width = runtime["width"]
    height = runtime["height"]
    transition = take_tail_with_front_pad(transition_video[:, :, :, :3], canvas_expansion_px)
    transition = resize_bhwc(transition, width, height)
    method = runtime.get("transition_colormatch", "disabled")
    ref = runtime.get("transition_match_ref", None)
    if method not in ("disabled", "auto_drift") and ref is not None:
        transition = color_match_frames(transition, ref, method)

    freeze_latents = encode_video_latent(vae, transition, width, height, runtime["tiled_vae"])
    freeze_mask = torch.ones(
        freeze_latents.shape[0],
        1,
        freeze_latents.shape[2],
        freeze_latents.shape[-2],
        freeze_latents.shape[-1],
        device=freeze_latents.device,
        dtype=freeze_latents.dtype,
    )
    runtime["transition_freeze_latents"] = freeze_latents.detach().cpu()
    runtime["transition_freeze_mask"] = freeze_mask.detach().cpu()
    _log_once(
        runtime,
        "freeze_latents",
        f"SCAIL-2 Flow freeze latents shape: {tuple(freeze_latents.shape)}",
    )
    return runtime["transition_freeze_latents"], runtime["transition_freeze_mask"]


def _apply_transition_hard_freeze(vae, runtime, latent, noise_mask, start_frame):
    freeze_latents, freeze_mask = _prepare_transition_freeze(runtime, vae)
    if freeze_latents is None:
        return

    global_start = frame_to_latent_index(start_frame)
    local_count = latent.shape[2]
    freeze_count = freeze_latents.shape[2]
    overlap_start = max(global_start, 0)
    overlap_end = min(global_start + local_count, freeze_count)
    if overlap_end <= overlap_start:
        return

    src_start = overlap_start
    dst_start = overlap_start - global_start
    count = overlap_end - overlap_start
    src = freeze_latents[:, :, src_start:src_start + count].to(device=latent.device, dtype=latent.dtype)
    src_mask = freeze_mask[:, :, src_start:src_start + count].to(device=noise_mask.device, dtype=noise_mask.dtype)
    if src.shape[0] != latent.shape[0]:
        src = src.repeat(math.ceil(latent.shape[0] / src.shape[0]), 1, 1, 1, 1)[:latent.shape[0]]
        src_mask = src_mask.repeat(math.ceil(noise_mask.shape[0] / src_mask.shape[0]), 1, 1, 1, 1)[:noise_mask.shape[0]]
    latent[:, :, dst_start:dst_start + count] = src
    noise_mask[:, :, dst_start:dst_start + count] = torch.where(
        src_mask > 0,
        torch.zeros_like(noise_mask[:, :, dst_start:dst_start + count]),
        noise_mask[:, :, dst_start:dst_start + count],
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
    total_frames = runtime["num_frames"]
    if length is None:
        length = total_frames
    length = align_4n1(length)
    start_frame = max(0, int(start_frame))

    t_lat = latent_frames_for_pixels(length)
    lat_h = height // 8
    lat_w = width // 8
    batch_size = runtime["batch_size"]
    latent = torch.zeros((batch_size, 16, t_lat, lat_h, lat_w), device=mm.intermediate_device())
    noise_mask = torch.ones((batch_size, 1, t_lat, lat_h, lat_w), device=latent.device, dtype=latent.dtype)
    handoff_mask = torch.zeros_like(noise_mask)
    _apply_transition_hard_freeze(vae, runtime, latent, noise_mask, start_frame)

    if previous_frames is not None and previous_frames.shape[0] > 0:
        prev = take_tail_with_front_pad(previous_frames, runtime["previous_frame_count"])
        prev_latent = encode_video_latent(vae, prev, width, height, runtime["tiled_vae"])
        prev_latent = prev_latent.to(device=latent.device, dtype=latent.dtype)
        prev_latents = min(prev_latent.shape[2], latent.shape[2])
        latent[:, :, :prev_latents] = prev_latent[:, :, :prev_latents]
        noise_mask[:, :, :prev_latents] = 0.0
        handoff_mask[:, :, :prev_latents] = 1.0

    values = _static_conditioning_values_for_length(runtime, vae, t_lat)

    pose_images = runtime.get("pose_images")
    if pose_images is not None:
        pose_slice = slice_fit_sequence(pose_images, start_frame, length)
        pose_latent = encode_video_latent(
            vae,
            pose_slice,
            width // 2,
            height // 2,
            runtime["tiled_vae"],
            scale=runtime["pose_strength"],
            mode="area",
        )
        values["pose_video_latent"] = pose_latent
        _log_once(
            runtime,
            "pose_latent",
            f"SCAIL-2 Flow pose latent shape: {tuple(pose_latent.shape)}",
        )

    pose_mask = runtime.get("pose_image_mask")
    if pose_mask is not None:
        mask_slice = slice_fit_sequence(pose_mask, start_frame, length)
        mask_slice = normalize_mask_background(mask_slice, white_background=runtime["replacement_mode"])
        mask_video = resize_bhwc(mask_slice, width // 2, height // 2, mode="area")
        values["driving_mask_28ch"] = extract_mask_to_28ch(mask_video)
        _log_once(
            runtime,
            "driving_mask",
            f"SCAIL-2 Flow driving mask latents shape: {tuple(values['driving_mask_28ch'].shape)}",
        )

    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)

    out_latent = {"samples": latent}
    if torch.count_nonzero(noise_mask != 1.0):
        out_latent["noise_mask"] = noise_mask
        out_latent[FLOW_FREEZE_MASK_KEY] = noise_mask != 1.0
    if torch.count_nonzero(handoff_mask):
        out_latent[FLOW_HANDOFF_MASK_KEY] = handoff_mask
    if include_runtime:
        out_latent[FLOW_RUNTIME_KEY] = runtime
    canvas_expansion_px = int(runtime.get("canvas_expansion_px", 0) or 0)
    if canvas_expansion_px:
        out_latent["canvas_expansion_px"] = canvas_expansion_px
    if runtime.get("requested_output_frames", None) is not None:
        out_latent["output_frame_count"] = int(runtime["requested_output_frames"])
    return positive, negative, out_latent


def build_deferred_latent(runtime, length=None, include_runtime=True):
    width = runtime["width"]
    height = runtime["height"]
    if length is None:
        length = runtime["frame_window_size"] if runtime.get("looping", False) else runtime["num_frames"]
    length = align_4n1(length)
    t_lat = latent_frames_for_pixels(length)
    lat_h = height // 8
    lat_w = width // 8
    batch_size = runtime["batch_size"]
    latent = torch.zeros((batch_size, 16, t_lat, lat_h, lat_w), device=mm.intermediate_device())
    out_latent = {"samples": latent}
    if include_runtime:
        out_latent[FLOW_RUNTIME_KEY] = runtime
    canvas_expansion_px = int(runtime.get("canvas_expansion_px", 0) or 0)
    if canvas_expansion_px:
        out_latent["canvas_expansion_px"] = canvas_expansion_px
    if runtime.get("requested_output_frames", None) is not None:
        out_latent["output_frame_count"] = int(runtime["requested_output_frames"])
    return out_latent


def decode_latent_to_images(vae, latent, tiled_vae=False):
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if tiled_vae and hasattr(vae, "decode_tiled"):
        compression = vae.spacial_compression_decode()
        temporal_compression = vae.temporal_compression_decode()
        tile_t = None if temporal_compression is None else max(2, 64 // temporal_compression)
        overlap_t = None if temporal_compression is None else max(1, min(tile_t // 2, 8 // temporal_compression))
        images = vae.decode_tiled(
            samples,
            tile_x=512 // compression,
            tile_y=512 // compression,
            overlap=64 // compression,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )
    else:
        images = vae.decode(samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images.clamp(0.0, 1.0)


def _color_match_ref_to_numpy(ref_frame):
    if ref_frame is None or not isinstance(ref_frame, torch.Tensor):
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


def color_match_frames(frames, ref_frames, method):
    if method in (None, "disabled", "auto_drift") or frames is None or ref_frames is None or frames.shape[0] == 0:
        return frames
    ref_np = _color_match_ref_to_numpy(ref_frames)
    if ref_np is None:
        return frames
    try:
        from color_matcher import ColorMatcher
    except Exception as e:
        log.warning(f"SCAIL-2 colormatch method {method!r} unavailable; keeping original frames: {e}")
        return frames

    cm = ColorMatcher()
    matched = []
    warned_error = False
    warned_shape = False
    warned_nonfinite = False
    for frame in frames[:, :, :, :3]:
        src_np = frame.detach().cpu().float().contiguous().numpy()
        try:
            out = np.asarray(cm.transfer(src=src_np, ref=ref_np, method=method), dtype=np.float32)
        except Exception as e:
            if not warned_error:
                log.warning(f"SCAIL-2 colormatch method {method!r} failed; keeping original frame: {e}")
                warned_error = True
            out = src_np
        if out.shape != src_np.shape:
            if not warned_shape:
                log.warning(
                    f"SCAIL-2 colormatch method {method!r} returned shape {out.shape}, "
                    f"expected {src_np.shape}; keeping original frame"
                )
                warned_shape = True
            out = src_np
        elif not np.isfinite(out).all():
            if not warned_nonfinite:
                log.warning(f"SCAIL-2 colormatch method {method!r} returned non-finite values; keeping finite pixels only")
                warned_nonfinite = True
            out = np.where(np.isfinite(out), out, src_np)
        matched.append(torch.from_numpy(out).to(device=frames.device, dtype=frames.dtype))
    return torch.stack(matched, dim=0).clamp(0.0, 1.0)


def auto_drift_tail_means(frames, max_frames=5):
    if frames is None or frames.shape[0] == 0:
        return None
    count = min(int(max_frames), int(frames.shape[0]))
    tail = frames[-count:, :, :, :3].detach().float().clamp(0.0, 1.0)
    return tail.mean(dim=(1, 2)).cpu().contiguous()


def _normalize_auto_drift_means(means):
    if means is None or not isinstance(means, torch.Tensor):
        return None
    means = means.detach().cpu().float()
    if means.ndim == 1:
        means = means.unsqueeze(0)
    if means.ndim != 2 or means.shape[0] <= 0 or means.shape[1] < 3:
        return None
    return means[:, :3].clamp(0.0, 1.0).contiguous()


def auto_drift_frames(frames, ref_means, chunk_idx=0, num_chunks=1):
    ref_means = _normalize_auto_drift_means(ref_means)
    if frames is None or frames.shape[0] == 0 or ref_means is None:
        return frames

    auto_drift_max_frames = 5
    auto_drift_jump_threshold = 0.0005
    auto_drift_max_offset = 0.02
    auto_drift_residual_max = 0.004

    current = frames[:, :, :, :3].float().clamp(0.0, 1.0)
    frame_count = int(current.shape[0])
    compare_count = min(auto_drift_max_frames, int(ref_means.shape[0]), frame_count)
    if compare_count <= 0:
        return frames

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
        f"SCAIL-2 auto_drift chunk {chunk_idx + 1}/{num_chunks}: "
        f"frames={compare_count}, jump=[{jump_values}], "
        f"global={'yes' if global_applied else 'no'}, "
        f"max_global={max_global:.6f}, max_residual={max_residual:.6f}"
    )
    return current.to(device=frames.device, dtype=frames.dtype)
