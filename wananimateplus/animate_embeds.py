# Copyright (c) 2025 kijai
# Modified from the original work (https://github.com/kijai/ComfyUI-WanVideoWrapper)
#   - Exposes WanAnimatePlus AnimateEmbeds as an independent node wrapper
# Licensed under the Apache License, Version 2.0

from .core_nodes import WanVideoAnimateEmbeds_plus


class WanAnimatePlusAnimateEmbeds(WanVideoAnimateEmbeds_plus):
    CATEGORY = "WanAnimatePlus"
