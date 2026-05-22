# source https://github.com/thu-ml/DiT-Extrapolation/blob/ultra-wan/sageattn/core.py

import torch
import triton.language as tl

from .quant_per_block import per_block_int8
from .attn_qk_int8_per_block import forward as attn_false

from typing import Optional

def sage_attention(
    qkv: list[torch.Tensor],
    tensor_layout: str ="HND",
    is_causal=False,
    sm_scale: Optional[float] = None,
    smooth_k: bool =True,
    xpos_xi: tl.constexpr = 0.9999934149894527,
    flags = None,
    block_bias = None,
    sigmoid_a: float = 1.0,
    alpha_xpos_xi: float = 0.97,
    beta_xpos_xi: float = 0.8,
    decay_mask = None,
    sink_width: int = 4,
    window_width: int = 21,
    multi_factor: Optional[float] = None,
    entropy_factor: Optional[float] = None,
    block_size : int = 64,
    **kwargs
) -> torch.Tensor:
    dtype = qkv[0].dtype
    q, k, v = qkv[0].transpose(1, 2), qkv[1].transpose(1, 2), qkv[2].transpose(1, 2) # to HND

    if flags == None:
        flags = torch.zeros([q.shape[0],q.shape[1]], dtype=torch.int32, device=q.device)

    seq_dim = 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k -= km
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if q.dtype != k.dtype or q.dtype != v.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout, BLKQ=block_size, BLKK=block_size)
    del q, k

    o = attn_false(q_int8, k_int8, v, flags, block_bias, decay_mask, q_scale, k_scale,
        tensor_layout=tensor_layout, output_dtype=dtype, xpos_xi=xpos_xi, sigmoid_a=sigmoid_a,
        alpha_xpos_xi=alpha_xpos_xi, beta_xpos_xi=beta_xpos_xi,
        BLOCK_M=block_size, BLOCK_N=block_size,
        sink_width=sink_width,
        window_width=window_width,
        multi_factor=multi_factor,
        entropy_factor=entropy_factor,
    )

    return o.transpose(1, 2).contiguous()
