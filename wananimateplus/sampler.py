# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Exposes WanAnimatePlus sampler node wrappers
# Licensed under the Apache License, Version 2.0

from .core_sampler import WanVideoSampler_plus, WanVideoSamplerv2_plus


class WanAnimatePlusSampler(WanVideoSampler_plus):
    CATEGORY = "WanAnimatePlus"


class WanAnimatePlusSamplerv2(WanVideoSamplerv2_plus):
    CATEGORY = "WanAnimatePlus"
