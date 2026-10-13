"""BF16 Q/K/V to BF16 attention on SM120 using the persistent NVFP4 kernel.

Supports non-causal self-attention, head dimension 128, and sequence lengths
that are multiples of 128. Preprocessing computes Q-block and K-sequence means,
quantizes Q/K/V, and forms the BF16 compensation term with torch.matmul.
Inputs are not modified. Short sequences use a separate torch K subtraction.
"""

import torch

import tilelang
import tilelang.language as T

from examples.attention_sm120 import sageattn3_prep as prep
from examples.attention_sm120 import sm120_sageattn3_fwd_ws as ws

# below this many tokens the smoothed K is cheaper as a torch subtraction than as a kernel output
KSM_FROM_KERNEL_MIN_SEQ = 2048
_KERNELS: dict = {}


def _kernels(batch: int, heads: int, seq_len: int, dim: int):
    key = (batch, heads, seq_len, dim)
    if key not in _KERNELS:
        emit_ksm = seq_len >= KSM_FROM_KERNEL_MIN_SEQ
        attn = ws.sm120_sageattn3_fwd(
            batch,
            heads,
            seq_len,
            dim,
            ds_dtype=T.bfloat16,
            pass_configs={tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 5},
        )
        _KERNELS[key] = (
            prep.sa3_means(batch, heads, seq_len, dim),
            prep.sa3_quant(batch, heads, seq_len, dim, emit_ksm=emit_ksm),
            attn,
            emit_ksm,
        )
    return _KERNELS[key]


def sageattn3_tl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """bf16 [B, H, N, 128] q, k, v -> bf16 [B, H, N, 128] attention output."""
    for name, x in (("q", q), ("k", k), ("v", v)):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 4:
            raise ValueError(f"{name} must be a 4-D CUDA bfloat16 tensor, got {tuple(x.shape)} {x.dtype} on {x.device}")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"q, k, v must have the same shape (non-causal self-attention), got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    b, h, n, d = q.shape
    if d != 128:
        raise ValueError(f"head_dim must be 128, got {d}")
    if min(b, h, n) <= 0:
        raise ValueError("batch, heads and seq_len must be positive")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same CUDA device")
    if n % 128 != 0:
        raise ValueError(f"seq_len must be a multiple of 128, got {n} (padding is not supported yet)")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    means, quant, attn, emit_ksm = _kernels(b, h, n, d)
    qm, km = means(q, k)
    qp, kp, vtp, sfq, sfk, sfv, ksm = quant(q, k, v, qm, km)
    if not emit_ksm:
        ksm = k - km
    ds = torch.matmul(qm, ksm.transpose(-2, -1))
    # Release the smoothed K before the attention call: the output has the same size, so the
    # allocator can reuse the allocation for the attention output.
    del ksm
    return attn(
        qp.view(torch.int8),
        kp.view(torch.int8),
        vtp.view(torch.int8),
        sfq.view(torch.uint32),
        sfk.view(torch.uint32),
        sfv.view(torch.uint32),
        ds,
    )
