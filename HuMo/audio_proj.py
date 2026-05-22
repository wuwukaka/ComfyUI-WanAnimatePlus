import torch
from einops import rearrange
from torch import nn
from einops import rearrange

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.to(self.weight.dtype)) * self.weight

    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)


class DummyAdapterLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)
    

class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=13,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=16,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.audio_proj_glob_1 = DummyAdapterLayer(nn.Linear(self.input_dim, intermediate_dim))
        self.audio_proj_glob_2 = DummyAdapterLayer(nn.Linear(intermediate_dim, intermediate_dim))
        self.audio_proj_glob_3 = DummyAdapterLayer(nn.Linear(intermediate_dim, context_tokens * output_dim))

        self.audio_proj_glob_norm = DummyAdapterLayer(nn.LayerNorm(output_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.audio_proj_glob_1(audio_embeds))
        audio_embeds = torch.relu(self.audio_proj_glob_2(audio_embeds))

        context_tokens = self.audio_proj_glob_3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)

        context_tokens = self.audio_proj_glob_norm(context_tokens.to(self.audio_proj_glob_norm.layer.weight.dtype)).to(audio_embeds.dtype)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens