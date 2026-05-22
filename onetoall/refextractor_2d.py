from typing import Dict, Union

import torch
import torch.nn as nn

from ..wanvideo.modules.model import WanLayerNorm, WanSelfAttention, EmbedND_RifleX, sinusoidal_embedding_1d, apply_rotary_emb_split, apply_rope_comfy1

class WanAttentionBlock(nn.Module):
    def __init__(self, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6, attention_mode="sdpa", rope_func="comfy", rms_norm_function="default"):
        super().__init__()
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func

        # layers
        self.norm1 = WanLayerNorm(self.dim, eps)
        self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function, head_norm=False)
        self.norm2 = WanLayerNorm(self.dim, eps)
        self.ffn = nn.Sequential(nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn2_dim, out_features))

        self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)


    def get_mod(self, e, modulation):
        if e.dim() == 3:
            if e.shape[-1] == 512:
                e = self.modulation(e)
                return e.unsqueeze(2).chunk(6, dim=-1)
            return (modulation + e).chunk(6, dim=1) # 1, 6, dim
        elif e.dim() == 4:
            e_mod = modulation.unsqueeze(2) + e
            return [ei.squeeze(1) for ei in e_mod.unbind(dim=1)]


    def modulate(self, norm_x, shift_msa, scale_msa):
        return torch.addcmul(shift_msa, norm_x, 1 + scale_msa)

    def ffn_chunked(self, mod_x, num_chunks=4):
        seq_len = mod_x.shape[1]
        if seq_len <= 8192 or num_chunks <= 1:
            return self.ffn(mod_x)
        return torch.cat([self.ffn(chunk.contiguous()) for chunk in mod_x.chunk(num_chunks, dim=1)], dim=1)

    #region attention forward
    def forward(self, x, e, seq_lens, freqs, split_rope=True, e_tr=None, tr_start=0, tr_num=0):

        use_token_replace = False
        if e_tr is not None and tr_num > 0:
            tr_shift_msa, tr_scale_msa, tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp = self.get_mod(e_tr.to(x.device), self.modulation)
            use_token_replace = True
            tr_start = tr_start or 0
            tr_end = tr_start + (tr_num or 0)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e.to(x.device), self.modulation)
        del e
        input_dtype = x.dtype

        if use_token_replace:
            norm_x = self.norm1(x.to(shift_msa.dtype))
            input_x = torch.cat([
                torch.addcmul(shift_msa, norm_x[:, :tr_start], 1 + scale_msa),           # before replace → T
                torch.addcmul(tr_shift_msa, norm_x[:, tr_start:tr_end], 1 + tr_scale_msa), # replace segment → t=0
                torch.addcmul(shift_msa, norm_x[:, tr_end:], 1 + scale_msa)              # after replace → T
            ], dim=1).to(input_dtype)
        else:
            input_x = self.modulate(self.norm1(x.to(shift_msa.dtype)), shift_msa, scale_msa).to(input_dtype)
        del shift_msa, scale_msa

        b, s, n, d = *x.shape[:2], self.self_attn.num_heads, self.self_attn.head_dim
        h_dim = w_dim = 2 * (self.head_dim  // 6)
        t_dim = self.head_dim - h_dim - w_dim

        q = self.self_attn.norm_q(self.self_attn.q(input_x)).to(self.self_attn.norm_q.weight.dtype).view(b, s, n, d)
        if split_rope:
            q = apply_rotary_emb_split(q, freqs, t_dim) # Apply split rotary embedding (only to H/W dimensions, leaving T unchanged)
        else:
            q = apply_rope_comfy1(q, freqs)

        k = self.self_attn.norm_k(self.self_attn.k(input_x).to(self.self_attn.norm_k.weight.dtype)).to(input_x.dtype).view(b, s, n, d)
        if split_rope:
            k = apply_rotary_emb_split(k, freqs, t_dim)
        else:
            k = apply_rope_comfy1(k, freqs)

        v = self.self_attn.v(input_x).view(b, s, n, d)
        del input_x

        y = self.self_attn.forward(q, k, v, seq_lens)
        del q, k, v
        if use_token_replace:
            x = x + torch.cat([
                y[:, :tr_start] * gate_msa,
                y[:, tr_start:tr_end] * tr_gate_msa,
                y[:, tr_end:] * gate_msa
            ], dim=1).to(input_dtype)
        else:
            x = x.addcmul(y, gate_msa)
        del y, gate_msa

        # ffn
        if use_token_replace:
            norm2_x = self.norm2(x.to(shift_mlp.dtype))
            mod_x = torch.cat([
                torch.addcmul(shift_mlp, norm2_x[:, :tr_start], 1 + scale_mlp),
                torch.addcmul(tr_shift_mlp, norm2_x[:, tr_start:tr_end], 1 + tr_scale_mlp),
                torch.addcmul(shift_mlp, norm2_x[:, tr_end:], 1 + scale_mlp)
            ], dim=1)
        else:
            mod_x = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp)
        del shift_mlp, scale_mlp
        x_ffn = self.ffn_chunked(mod_x.to(input_dtype), num_chunks=1)
        del mod_x

        # gate_mlp
        if use_token_replace:
            x = x + torch.cat([
                x_ffn[:, :tr_start] * gate_mlp,
                x_ffn[:, tr_start:tr_end] * tr_gate_mlp,
                x_ffn[:, tr_end:] * gate_mlp
            ], dim=1).to(input_dtype)
        else:
            x = x.addcmul(x_ffn.to(gate_mlp.dtype), gate_mlp).to(input_dtype)
        del gate_mlp

        return x


class WanRefextractor(nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_dim=16, dim=5120, in_features=5120, out_features=5120, ffn_dim=8192, ffn2_dim=8192,
                freq_dim=256, num_heads=16, num_layers=32, eps=1e-6,
                qk_norm=True, cross_attn_norm=True,
                attention_mode='sdpa', rope_func='comfy', rms_norm_function='default',
                main_device=torch.device('cuda'), offload_device=torch.device('cpu'), dtype=torch.float16):
        super().__init__()
        self.patch_size = patch_size
        self.freq_dim = freq_dim
        self.dim = dim
        self.main_device = main_device
        self.base_dtype = dtype
        self.attention_mode = attention_mode
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
                WanAttentionBlock(in_features, out_features, ffn_dim, ffn2_dim, num_heads,
                                qk_norm, cross_attn_norm, eps, attention_mode="sdpa", rope_func=rope_func, rms_norm_function=rms_norm_function)
                for i in range(num_layers)
            ])

        self.ref_blocks = nn.ModuleList([])
        for _ in range(len(self.blocks)+1):
            self.ref_blocks.append(nn.Linear(in_features, out_features))

        d = dim // num_heads
        self.rope_embedder = EmbedND_RifleX(d,10000.0, [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)], num_frames=1, k=0)

    def rope_encode_comfy(self, t, h, w, freq_offset=0, t_start=0, steps_t=None, steps_h=None, steps_w=None, ntk_alphas=[1,1,1], device=None, dtype=None):
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start+freq_offset, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(freq_offset, h_len - 1, steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(freq_offset, w_len - 1, steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids, ntk_alphas).movedim(1, 2)
        return freqs

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.LongTensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, F, H, W = x.shape

        freqs = self.rope_encode_comfy(F, H, W, device=x.device, dtype=x.dtype)

        self.patch_embedding.to(self.main_device)
        x = self.patch_embedding(x.float()).to(x.dtype).flatten(2).transpose(1, 2).to(self.base_dtype)

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32)

        time_embed_dtype = self.time_embedding[0].weight.dtype
        if time_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            time_embed_dtype = self.base_dtype
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(time_embed_dtype))  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).to(self.base_dtype)  # b, 6, dim
        del e

        # 4. Transformer blocks
        block_samples = ()
        for block in self.blocks:
            block_samples = block_samples + (x, )
            x = block(x, e0, seq_lens, freqs)

        block_samples = block_samples + (x, )

        ref_block_samples = ()
        for block_sample, ref_block in zip(block_samples, self.ref_blocks):
            block_sample = ref_block(block_sample)
            ref_block_samples = ref_block_samples + (block_sample, )

        return ref_block_samples, freqs
