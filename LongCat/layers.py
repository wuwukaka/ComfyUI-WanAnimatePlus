import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from einops import rearrange

from ..wanvideo.modules.model import WanRMSNorm, attention
from ..multitalk.multitalk import RotaryPositionalEmbedding1D, normalize_and_scale

class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
        attention_mode: str = "sdpa",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = WanRMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.k_norm = WanRMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attention_mode = attention_mode

        # multitalk related params
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def _process_cross_attn(self, x, cond, frames_num=None, x_ref_attn_map=None):

        N_t = frames_num
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3)) # [B, H, N, D]
        q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(q.dtype)

        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            max_values = x_ref_attn_map.max(1).values[:, None, None] 
            min_values = x_ref_attn_map.min(1).values[:, None, None] 
            max_min_values = torch.cat([max_values, min_values], dim=2) 
            human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
            human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

            human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
            human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
            back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
            max_indices = x_ref_attn_map.argmax(dim=0)
            normalized_map = torch.stack([human1, human2, back], dim=1)
            normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] 

            q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
            q = self.rope_1d(q, normalized_pos)
            q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # get kv from encoder_hidden_states
        _, N_a, _ = cond.shape
        encoder_kv = self.kv_linear(cond)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))

        encoder_k, encoder_v = encoder_kv.unbind(0)
        encoder_k = self.k_norm(encoder_k.to(self.k_norm.weight.dtype)).to(encoder_k.dtype)


        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
            per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
            per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
            encoder_pos = torch.concat([per_frame]*N_t, dim=0)
            encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
            encoder_k = self.rope_1d(encoder_k, encoder_pos)
            encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
        # the sequence length, H the number of heads, and K the embeding size per head

        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        x = attention(q, encoder_k, encoder_v, attention_mode=self.attention_mode)
        x = rearrange(x, "B M H K -> B H M K")

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x.type(out_dtype)

    def forward(self, x, cond, num_latent_frames=None, num_cond_latents=None, x_ref_attn_map=None, human_num=None):

        B, N, C = x.shape
        if (num_cond_latents is None or num_cond_latents == 0):
            # text to video
            output = self._process_cross_attn(x, cond, num_latent_frames, x_ref_attn_map)
            return None, output
        elif num_cond_latents is not None and num_cond_latents > 0:
            # image to video or video continuation
            num_cond_latents_thw = num_cond_latents * (N // num_latent_frames)
            x_noise = x[:, num_cond_latents_thw:]
            cond = rearrange(cond, "(B N_t) M C -> B N_t M C", B=B)
            cond = cond[:, num_cond_latents:]
            cond = rearrange(cond, "B N_t M C -> (B N_t) M C")
            frames_num = num_latent_frames - num_cond_latents
            if human_num is not None and human_num == 2:
                # multitalk mode
                output_noise = self._process_cross_attn(x_noise, cond, frames_num, x_ref_attn_map)
            else:
                # singletalk mode
                output_noise = self._process_cross_attn(x_noise, cond, frames_num)
            output_cond = torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device)
            return output_cond, output_noise
        else:
            raise NotImplementedError
