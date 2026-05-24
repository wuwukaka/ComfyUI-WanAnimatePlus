# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Exposes WanAnimatePlus scheduler node wrappers
# Licensed under the Apache License, Version 2.0

from .core_sampler import WanVideoScheduler_plus, WanVideoSchedulerv2_plus


class WanAnimatePlusScheduler(WanVideoScheduler_plus):
    CATEGORY = "WanAnimatePlus"


class WanAnimatePlusSchedulerv2(WanVideoSchedulerv2_plus):
    CATEGORY = "WanAnimatePlus"
