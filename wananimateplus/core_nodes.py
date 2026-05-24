# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Added prefix_frames support to WanVideoAnimateEmbeds_plus
#   - Added canvas_expansion_px trimming to WanVideoDecode_plus
#   - Unified canvas expansion for prefix and transition video
# Licensed under the Apache License, Version 2.0
import os, gc, math
import torch
import torch.nn.functional as F

from ..utils import(log, add_noise_to_reference_video)
from ..taehv import TAEHV

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


class WanVideoAnimateEmbeds_plus:
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
                "prefix_frames": ("IMAGE", {"default": None, "tooltip": "3 reference images. Expands canvas by 17 pixel frames, encoded together with bg frames. Image 0 ×5, image 1 ×4, image 2 ×4, image 0 ×4. Shifts pose/face by 17 frames."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "Prefix & Transition Video by wuwukasi(bilibili)": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "ON"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, colormatch, pose_strength, face_strength,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None, tiled_vae=False, bg_images=None, mask=None, start_ref_image=None,
                transition_video=None, prefix_frames=None, **kwargs):
        
        W = (width // 16) * 16
        H = (height // 16) * 16

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_refs = ref_images.shape[0] if ref_images is not None else 0
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        if transition_video is not None and prefix_frames is None:

            # --- [Core Mod] Reserve space for insertion logic and shift subsequent actions ---
            # 1. Expand canvas: Add space for 21 pixel frames (corresponding to 6 Latent frames).
            num_frames += 21
            trim = (num_frames - 1) % 4
            num_frames -= trim

            # 2. Shift control signals by 21 pixel frames with sampled+reversed padding
            if pose_images is not None:
                sampled = pose_images[0:42:2]               # 21 frames from indices 0,2,...,40
                sampled = torch.flip(sampled, [0])           # reverse order
                pose_images = torch.cat([sampled, pose_images], dim=0)
            if face_images is not None:
                sampled = face_images[0:42:2]
                sampled = torch.flip(sampled, [0])
                face_images = torch.cat([sampled, face_images], dim=0)
            if bg_images is not None:
                sampled = bg_images[0:42:2]
                sampled = torch.flip(sampled, [0])
                bg_images = torch.cat([sampled, bg_images], dim=0)
            if mask is not None:
                sampled = mask[0:42:2]
                sampled = torch.flip(sampled, [0])
                mask = torch.cat([sampled, mask], dim=0)
            # ----------------------------------------------------

            if start_ref_image is not None:
                log.warning("Both transition_video and start_ref_image provided. Using transition_video only (loop disabled).")
        # ============ Prefix frames: expand canvas and shift control signals ============
        if prefix_frames is not None:
            # Expand canvas: always 37 = 17 prefix + 20 reserve (for optional transition)
            extra = 37
            num_frames += extra
            # Trim 1-3 frames from end to keep num_frames % 4 == 1 (required by repeat_interleave + view)
            trim = (num_frames - 1) % 4
            num_frames -= trim

            # Pad beginning with sampled+reversed frames for pose/face (sparse temporal context),
            # repeat frame 0 for bg/mask
            if pose_images is not None:
                sampled = pose_images[0:extra*2:2]        # indices 0,2,4,... (extra frames)
                sampled = torch.flip(sampled, [0])         # reverse order
                pose_images = torch.cat([sampled, pose_images], dim=0)
            if face_images is not None:
                sampled = face_images[0:extra*2:2]
                sampled = torch.flip(sampled, [0])
                face_images = torch.cat([sampled, face_images], dim=0)
            if bg_images is not None:
                sampled = bg_images[0:extra*2:2]
                sampled = torch.flip(sampled, [0])
                bg_images = torch.cat([sampled, bg_images], dim=0)
            if mask is not None:
                sampled = mask[0:extra*2:2]
                sampled = torch.flip(sampled, [0])
                mask = torch.cat([sampled, mask], dim=0)
        # -----------------------------------------------------------------

        if prefix_frames is not None:
            effective_frames = num_frames - 37
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
        if prefix_frames is not None:
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
                resized_bg_images = torch.zeros(3, num_frames - num_refs, H, W, device=device, dtype=vae.dtype)

            # ============ Prefix: replace first N pixel frames of canvas ============
            if prefix_pixel_data is not None:
                resized_bg_images[:, :actual_prefix_px] = prefix_pixel_data.to(device, dtype=resized_bg_images.dtype)
                log.info(f"Prefix: replaced first {actual_prefix_px} pixel frames of black canvas")
                # If transition_video also present, embed last 20 frames into canvas positions 17-37
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
                    resized_bg_images[:, 17:37] = tv.to(device, dtype=resized_bg_images.dtype)
                    log.info("Prefix+Transition: embedded last 20 transition frames into canvas positions 17-37")
            # ==========================================================================

            # ============ Transition (no prefix): embed into canvas first 21 frames ============
            if transition_video is not None and prefix_frames is None:
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
            resized_bg_images = torch.zeros(3, num_frames - num_refs, H, W, device=offload_device, dtype=vae.dtype)
            if transition_video is not None:
                tv = transition_video  # [B, H, W, C]
                b_tv = tv.shape[0]
                if b_tv >= 21:
                    tv = tv[-21:]
                else:
                    tv = torch.cat([tv[0:1].repeat(21 - b_tv, 1, 1, 1), tv], dim=0)
                tv = tv.permute(3, 0, 1, 2)[:3] * 2 - 1  # [C, 21, H, W]
                resized_bg_images[:, :21] = tv.to(offload_device, dtype=resized_bg_images.dtype)
                log.info("Transition (loop): embedded first 21 pixel frames of canvas")
            # Prefix NOT embedded in canvas for looping — handled via prefix_ctx prepend later

        if ref_images is not None:
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
            prefix_ctx = None
            prefix_T = 0
            if prefix_frames is not None and looping:
                vae.to(device)
                prefix_latent = vae.encode([prefix_pixel_data.to(device, vae.dtype).unsqueeze(0)], device, tiled=tiled_vae)[0]
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
            if prefix_frames is not None:
                bg_mask[:, :actual_prefix_px] = 1.0  # only actual prefix pixel frames
                if transition_video is not None:
                    bg_mask[:, 17:37] = 1.0
            # ======= Transition (no prefix): set mask=1 for first 21 pixel frames =======
            elif transition_video is not None:
                bg_mask[:, :21] = 1.0
            # ======================================================================================

            if bg_images is None and looping:
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
            "ref_masks": bg_mask if (mask is not None or prefix_frames is not None) and looping else None,
            "is_masked": mask is not None,
            "ref_latent": ref_latent,
            "ref_image": resized_ref_images if ref_images is not None else None,
            "start_ref_image": resized_start_ref_image if start_ref_image is not None else None,
            "transition_latent": transition_latent,
            "transition_mask_values": transition_mask_values,
            "has_prefix": prefix_frames is not None,
            "canvas_expansion_px": 37 if prefix_frames is not None else (21 if transition_video is not None else 0),
            "prefix_ctx": prefix_ctx,
            "prefix_T": prefix_T,
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

        return (image_embeds,)

# region UniLumos


class WanVideoDecode_plus:
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
        canvas_expansion_px = samples.get("canvas_expansion_px", 0)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        flashvsr_LQ_images = samples.get("flashvsr_LQ_images", None)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        if has_ref:
            latents = latents[:, :, 1:]
        if drop_last:
            latents = latents[:, :, :-1]

        if type(vae).__name__ == "TAEHV":
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4), cond=flashvsr_LQ_images.to(vae.dtype) if flashvsr_LQ_images is not None else None)[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            images = images.permute(1, 2, 3, 0).cpu().float()
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

        if canvas_expansion_px and not is_looped:
            images = images[:, canvas_expansion_px:]


        vae.to(offload_device)
        mm.soft_empty_cache()

        images.clamp_(0.0, 1.0)

        return (images.permute(1, 2, 3, 0),)

#region VideoEncode


class WanVideoEncode_plus:
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

        log.info(f"WanVideoEncode_plus: Encoded latents shape {latents.shape}")
        mm.soft_empty_cache()

        return ({"samples": latents, "noise_mask": mask},)
