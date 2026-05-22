from typing import Literal, Optional

import torch
import torch.nn as nn

from .vae import VAE, get_my_vae
from .distributions import DiagonalGaussianDistribution
from ..bigvgan import BigVGAN

from comfy.utils import load_torch_file

class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path,
                 vocoder_ckpt_path: Optional[str] = None,
                 mode: Literal['16k', '44k'],
                 need_vae_encoder: bool = True):
        super().__init__()
        self.vae: VAE = get_my_vae(mode).eval()
        #vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location='cpu')'
        vae_state_dict = load_torch_file(vae_ckpt_path)
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()

        if mode == '16k':
            assert vocoder_ckpt_path is not None
            self.vocoder = BigVGAN(vocoder_ckpt_path).eval()
        elif mode == '44k':
            raise NotImplementedError("44k mode requires BigVGANv2 which is not currently supported in this environment.")
            self.vocoder = BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x',
                                                     use_cuda_kernel=False)
            self.vocoder.remove_weight_norm()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
