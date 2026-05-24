# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Exposes WanAnimatePlus sampler settings node wrappers
# Licensed under the Apache License, Version 2.0

from .core_sampler import WanVideoSamplerSettings_plus, WanVideoSamplerFromSettings_plus


class WanAnimatePlusSamplerSettings(WanVideoSamplerSettings_plus):
    CATEGORY = "WanAnimatePlus"


class WanAnimatePlusSamplerFromSettings(WanVideoSamplerFromSettings_plus):
    CATEGORY = "WanAnimatePlus"
