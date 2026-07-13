# ComfyUI-WanAnimatePlus

[English](./README.md) | [中文](./README_ZH.md)

Multi-reference image injection and seamless video connection for ComfyUI's WanAnimate pipeline.

## Overview

`ComfyUI-WanAnimatePlus` adds four major feature groups to the WanVideo workflow:

### prefix_frames & transition_video

- **prefix_frames**: allows passing 1–5 additional reference images for multi-reference guided generation
- **transition_video**: allows passing the last 21 frames of the previous video segment for seamless video connection

When used together, canvas layout and frame offsets are automatically coordinated without conflicts.

### Bernini

Supports Bernini models. Allows passing source video, reference video, or reference images as generation conditions. Supports v2v, rv2v, r2v, and t2v tasks.

Use cases:

- Multi-shot video sequence generation
- Video continuation / extension
- Motion transfer with multi-reference control
- Video editing with source video + reference images
- Reference-to-video generation

### SCAIL-2 Embeds

Adds a wrapper-native `WanAnimatePlus SCAIL_2 Embeds` node for SCAIL-2 models. It prepares reference image, driving pose, colored pose mask, reference mask, optional prefix/transition hard-freeze latents, and prefix-aligned colored masks for the WanAnimatePlus sampler.

## Demo

### prefix_frames & transition_video usage

![Usage](docs/images/image_001.png)

### prefix_frames demo

[](https://github.com/user-attachments/assets/6df01023-5daa-42ab-9817-27a3b49bd6af)

### transition_video demo

[](https://github.com/user-attachments/assets/4c6d2d29-dc21-406c-8ae9-5201d4cc416b)

## Features

### prefix_frames (Multi-Reference Injection)

Allows 1–5 additional reference images. Internally expands the canvas pixel space and encodes reference images across the front frames, with automatic frame offset coordination for control signals (pose / face).

- Supports 1–5 reference images (truncated if exceeding 5)
- Auto-resizes reference images to target resolution
- Automatically aligns frame offsets for pose / face / bg / mask signals

### transition_video (Seamless Video Connection)

Allows passing the last 21 frames of the previous video segment. Writes these pixel frames directly into the front of the generation canvas, with sampled+reversed padding for control signal offsets.

- Automatically coordinates with prefix when both are used

### Bernini

Generates condition latents from source video, reference video, and/or reference images via VAE encoding. Supports v2v, rv2v, r2v, and t2v — task is auto-detected from connected inputs.

- Reference images kept at native aspect ratio
- Compatible with context windows

### SCAIL-2

Provides SCAIL-2 ref / pose / mask conditioning through `WanAnimatePlus SCAIL_2 Embeds`.

- Encodes `ref_image`, `pose_images`, `pose_image_mask`, `prefix_frames`, `prefix_mask`, `bg_image`, and `reference_image_mask`
- Aligns SCAIL-2 inputs to 32-pixel multiples before VAE encoding
- Supports animation and replacement modes
- Supports single-frame prefix reference encoding and optional `transition_video` hard-freeze conditioning
- By default, `prefix_frames` are encoded as full-resolution reference latents and do not expand the output canvas; disable `single_frame_prefix_encoding` to use the legacy 37 front pixel-frame prefix layout
- In single-frame prefix mode, `prefix_mask` follows the same reference-mask path as `reference_image_mask`
- Supports context-window sampling; non-first windows can see prepended prefix/transition context without fusing those prepended predictions

## Installation

Place this repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuwukaka/ComfyUI-WanAnimatePlus.git
```

Restart ComfyUI after installation.

> **Important**: To use `prefix_frames`, `transition_video`, `Bernini`, or `SCAIL_2 Embeds`, you **must** replace the full workflow chain with WanAnimatePlus nodes. Mixing WanAnimatePlus nodes with original WanVideoWrapper nodes in the same workflow will result in degraded output.

## Quick Start

1. Start ComfyUI and confirm the WanAnimatePlus nodes appear under the `WanAnimatePlus` category
2. **Replace the entire workflow chain** with WanAnimatePlus counterparts: `ModelLoader`, `VAELoader`, `ContextOptions`, `AnimateEmbeds`, `Sampler`, `Decode`, and supporting nodes
3. Do **not** mix original WanVideoWrapper nodes in the same workflow
4. Connect `prefix_frames` and/or `transition_video` inputs as needed
5. Example workflows are available in the `example_workflows/` directory

## Nodes

WanAnimatePlus exposes a complete workflow chain to avoid cross-package object mixing with the original WanVideoWrapper nodes.

Core nodes:

- `WanAnimatePlus ModelLoader`
- `WanAnimatePlus VAELoader`
- `WanAnimatePlus TextEncodeCached`
- `WanAnimatePlus ClipVisionEncode`
- `WanAnimatePlus ContextOptions`
- `WanAnimatePlus AnimateEmbeds`
- `WanAnimatePlus Sampler` / `WanAnimatePlus Samplerv2`
- `WanAnimatePlus Scheduler` / `WanAnimatePlus Schedulerv2`
- `WanAnimatePlus Decode` / `WanAnimatePlus Encode`
- `WanAnimatePlus LoraSelect` / `WanAnimatePlus LoraSelectMulti` / `WanAnimatePlus SetLoRAs`
- `WanAnimatePlus BlockSwap` / `WanAnimatePlus SetBlockSwap`
- `WanAnimatePlus TorchCompileSettings`
- `WanAnimatePlus SamplerExtraArgs`
- `WanAnimatePlus Uni3C ControlnetLoader` / `WanAnimatePlus Uni3C Embeds`
- `WanAnimatePlus Bernini`
- `WanAnimatePlus SCAIL_2 Embeds`

### WanAnimatePlus AnimateEmbeds

Core node, replaces the original `WanVideoAnimateEmbeds`.

**New inputs:**

| Input | Description |
|------|------|
| `prefix_frames` | 1–5 additional reference images for multi-reference guided generation |
| `transition_video` | Last 21 frames of the previous video segment for seamless video connection |

Other inputs are identical to the original WanVideoAnimateEmbeds: `vae`, `width`, `height`, `num_frames`, `ref_images`, `pose_images`, `face_images`, `bg_images`, `mask`, `start_ref_image`, `clip_embeds`, etc.

### WanAnimatePlus Bernini

Generates condition latents from source video, reference video, and/or reference images for Bernini models.

**Inputs:**

| Input | Description |
|------|------|
| `vae` | VAE model for encoding |
| `width` / `height` / `num_frames` | Output dimensions |
| `source_video` | Source video to edit/restyle (v2v/rv2v). Resized to width/height |
| `reference_video` | Moving content to composite (video insertion), native aspect |
| `reference_images` | Reference image(s) as in-context tokens (r2v/rv2v). Native aspect |
| `ref_max_size` | Max long-edge size for reference media (default 848) |
| `force_offload` | Offload VAE after encoding to save VRAM |
| `tiled_vae` | Use tiled VAE encoding for memory savings |

The task (v2v, rv2v, r2v, t2v) is automatically inferred from which inputs are connected.

### WanAnimatePlus SCAIL_2 Embeds

Creates SCAIL-2 conditioning for WanAnimatePlus sampling. Use this node with SCAIL-2 checkpoints that include the pose and mask streams.

**Inputs:**

| Input | Description |
|------|------|
| `vae` | VAE model for encoding |
| `width` / `height` / `num_frames` | Target dimensions; width and height are aligned to multiples of 32 |
| `ref_image` | Reference image for SCAIL-2 conditioning |
| `bg_image` | Optional single background image for animation mode. In single-frame mode it is encoded as an extra background reference latent; in legacy canvas-prefix mode it is appended after user `prefix_frames`. Ignored in replacement mode |
| `pose_images` | Driving pose video/images, encoded at half resolution |
| `pose_image_mask` | Colored per-identity pose mask sequence |
| `prefix_mask` | Optional colored mask images matching `prefix_frames`; in single-frame mode it follows the reference-mask path, and in legacy canvas-prefix mode it is expanded as `1+4+4...` and written into prefix pixel mask frames |
| `reference_image_mask` | Colored reference mask image |
| `replacement_mode` | Enables SCAIL-2 replacement-mode RoPE and reference-mask compositing |
| `preserve_main_ref_background` | Animation mode only; keeps the main reference image background when enabled, or uses `reference_image_mask` as a black-background alpha crop when disabled. Ignored in replacement mode |
| `single_frame_prefix_encoding` | Encodes `prefix_frames` as individual full-resolution reference latents instead of expanding the canvas; enabled by default |
| `prefix_frames` | Optional prefix images. In default single-frame mode these become reference-stream latents; with single-frame mode disabled they hard-freeze the front canvas |
| `transition_video` | Optional transition frames to hard-freeze at the front of the latent sequence |
| `clip_embeds` | Optional CLIP vision features from `WanAnimatePlus ClipVisionEncode` |
| `force_offload` / `tiled_vae` | Memory controls for VAE encoding |

For short generations, context windows are optional. For long generations or low VRAM, context windows are recommended. In context-window mode, single-frame prefix references remain visible through the SCAIL-2 reference stream. Legacy canvas prefixes and transition latents are prepended for model context and removed before overlap fusion.

For SCAIL-2, the default `single_frame_prefix_encoding` mode does not expand or trim the output for `prefix_frames`. If `transition_video` is connected, the front canvas expands by 21 pixel frames and those 21 frames are trimmed after decoding. With `single_frame_prefix_encoding` disabled, `prefix_frames` use the legacy 37 front pixel-frame canvas layout, with transition frames placed at frames 17-36 when `transition_video` is also connected.

When `bg_image` is connected in animation mode, the node internally adds a white mask for that background image. User-provided `prefix_frames` and `prefix_mask` are limited to four images each so the background can occupy the remaining reference/prefix capacity. In replacement mode, `bg_image` is ignored.

## Project Structure

```text
ComfyUI-WanAnimatePlus/
├─ wanvideo/                 # WanVideo core model code
├─ nodes.py                  # Core WanAnimatePlus embeds / encode / decode nodes
├─ nodes_sampler.py          # Core WanAnimatePlus sampler / scheduler nodes
├─ nodes_model_loading.py    # Core WanAnimatePlus model / VAE / LoRA / block swap nodes
├─ context_windows/          # Context-window scheduling
├─ cache_methods/            # Cache acceleration
├─ utils.py                  # Shared utilities
├─ docs/
│  └─ images/                # Documentation images
├─ example_workflows/        # Example workflows
├─ __init__.py               # Node registration entry point
├─ pyproject.toml
├─ requirements.txt
└─ LICENSE
```

## FAQ

### 1. Nodes not showing after installation

- Verify the repo path is `ComfyUI/custom_nodes/ComfyUI-WanAnimatePlus`
- Ensure the original `ComfyUI-WanVideoWrapper` is also installed
- Restart ComfyUI and search for `WanAnimatePlus` in the node list

### 2. Conflicts with original nodes?

No. All node names use the `WanAnimatePlus` prefix, completely avoiding conflicts with the original `WanVideo` prefixed nodes. Both can be installed simultaneously.

### 3. How many images for prefix_frames?

3 is recommended. Up to 5 are accepted (excess is truncated). The node works with fewer than 3 as well, but the coverage range will be smaller.

### 4. How many frames for transition_video?

Input is automatically cropped to 21 frames (padded with the first frame if insufficient).

## Acknowledgments

Modified from [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). Deep respect to the original author for their tremendous contributions to the WanVideo ecosystem.

Thanks to [checknickname/ComfyUI-Scail2-Sampler-Helper](https://github.com/checknickname/ComfyUI-Scail2-Sampler-Helper) for the SCAIL-2 two-phase sampling idea, and to [user2318/ComfyUI-CustomNodeKit](https://github.com/user2318/ComfyUI-CustomNodeKit/) for implementation reference. See [NOTICE](NOTICE) for attribution details.

## Contact

- Bilibili: [@wuwukasi](https://space.bilibili.com/670281046)
- Email: wuwukawayi@gmail.com

## Sponsorship

If you find this project helpful, consider supporting me! Your support is what keeps this project going and motivates me to continue improving it.

Every contribution, no matter how small, means a lot and helps me dedicate more time to development and new features. Thank you!

<p align="center">
  <img src="docs/images/image_003.png" alt="WeChat Sponsor QR" width="400"/>
  <img src="docs/images/image_002.jpg" alt="Alipay Sponsor QR" width="400"/>
</p>

## License

This project is an independently maintained fork / derivative project based on [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) and is released under the **Apache License, Version 2.0**. Thanks again to kijai and the original contributors for their work.

Modified portions and newly added source expression by wuwukasi/wuwukaka are Copyright (c) 2026 wuwukasi/wuwukaka. See [NOTICE](NOTICE) for attribution, authorship scope, and modification notice details.
