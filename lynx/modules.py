import torch
import torch.nn as nn
from ..wanvideo.modules.attention import attention

def vector_to_list(tensor, lens, dim):
    return list(torch.split(tensor, lens, dim=dim))

def list_to_vector(tensor_list, dim):
    lens = [tensor.shape[dim] for tensor in tensor_list]
    tensor = torch.cat(tensor_list, dim)
    return tensor, lens

def merge_token_lists(list1, list2, dim):
    assert(len(list1) == len(list2))
    return [torch.cat((t1, t2), dim) for t1, t2 in zip(list1, list2)]

try:
    from sageattention import sageattn_varlen
except ImportError:
    sageattn_varlen = None

class WanLynxIPCrossAttention(nn.Module):
    def __init__(self, cross_attention_dim=5120, dim=5120, n_registers=16, bias=True):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        if n_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, n_registers, cross_attention_dim) / dim**0.5)
        else:
            self.registers = None

    def forward(self, block, q, ip_x):
        b, s, n, d = q.shape
        ip_lens = [ip_x.shape[1]]

        if self.registers is not None and ip_x is not None and ip_x.shape[0] == 1:
            ip_x = torch.cat([ip_x, self.registers], dim=1)
            ip_lens[0] += self.registers.shape[1]
        elif self.registers is not None and ip_x.shape[0] > 1:
            ip_x_list = vector_to_list(ip_x, ip_lens, 1)
            ip_x_list = merge_token_lists(ip_x_list, [self.registers] * len(ip_x_list), 1)
            ip_x, ip_lens = list_to_vector(ip_x_list, 1)

        ip_key = self.to_k_ip(ip_x)
        ip_value = self.to_v_ip(ip_x)

        if self.registers is None: # lite model normalization
            ip_key = ip_key * torch.rsqrt(ip_key.pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(ip_key.dtype)
        else: # full model
            ip_key = block.norm_k(ip_key)

        return attention(
            q,
            ip_key.view(b, -1, n, d),
            ip_value.view(b, -1, n, d)
        ).flatten(2)


#@torch.compiler.disable()
class WanLynxRefAttention(nn.Module):
    def __init__(self, dim=5120, bias=True, attention_mode="sdpa"):
        super().__init__()
        self.to_k_ref = nn.Linear(dim, dim, bias=bias)
        self.to_v_ref = nn.Linear(dim, dim, bias=bias)
        self.attention_mode = attention_mode
        # Pre-compute attention mode flags to avoid string operations in forward
        self.use_flash_attn = "flash_attn" in attention_mode
        self.use_sageattn = sageattn_varlen is not None

    def forward(self, block, q, ref_feature):
        b, s, n, d = q.shape

        ref_key = self.to_k_ref(ref_feature)
        ref_value = self.to_v_ref(ref_feature)
        ref_key = block.norm_k(ref_key)

        # Use pre-computed flags instead of runtime string checks
        if not self.use_flash_attn and not self.use_sageattn:
            # Pad ref_key and ref_value to match q's sequence length (s)
            seq_len = ref_key.shape[1]
            pad_len = s - seq_len
            if pad_len > 0:
                # Pad on the sequence dimension (dim=1)
                ref_key = torch.nn.functional.pad(ref_key, (0, 0, 0, pad_len))
                ref_value = torch.nn.functional.pad(ref_value, (0, 0, 0, pad_len))

            # Create attention mask: True for real tokens, False for padded
            attn_mask = torch.zeros((b, s), dtype=torch.bool, device=ref_key.device)
            attn_mask[:, :seq_len] = True

            ref_key = ref_key.view(b, s, n, d)
            ref_value = ref_value.view(b, s, n, d)

            ref_x = attention(
                q,
                ref_key,
                ref_value,
                attention_mode="sdpa",
                attn_mask=attn_mask,
            )
        else:
            q_lens = [s] * b
            k_lens = [ref_key.shape[1]] * b

            ref_x = attention(
                q.view(-1, n, d),
                ref_key.view(-1, n, d),
                ref_value.view(-1, n, d),
                q_lens=q_lens,
                k_lens=k_lens,
                max_seqlen_k=ref_key.shape[1],
                max_seqlen_q=s,
                attention_mode='sageattn_varlen' if self.use_sageattn else self.attention_mode,
                )

        return ref_x