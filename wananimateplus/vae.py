# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Exposes WanAnimatePlus VAE encode/decode node wrappers
# Licensed under the Apache License, Version 2.0

from .core_nodes import WanVideoEncode_plus, WanVideoDecode_plus


class WanAnimatePlusEncode(WanVideoEncode_plus):
    CATEGORY = "WanAnimatePlus"


class WanAnimatePlusDecode(WanVideoDecode_plus):
    CATEGORY = "WanAnimatePlus"
