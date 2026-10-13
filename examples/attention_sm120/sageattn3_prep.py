"""Device-side SageAttention3 means and NVFP4 quantization on SM120.

Q is smoothed by its 128-row block mean and K by its sequence mean, using
FP32 reductions rounded to BF16. The quantizer writes Q, permuted K, and
transposed V directly into the persistent attention kernel's packed layouts.
The Q mean reduction differs from the original implementation's BF16 reduction;
end-to-end tests check the resulting numerical agreement.
"""

import tilelang
import tilelang.language as T

QUANT_SRC = r"""
// Four pairs -> one uint32 of e2m1 nibbles (pair t in byte t, even element in the low nibble).
__device__ __forceinline__ unsigned int tl_q_e2m1x8_rn(float e0, float o0, float e1, float o1,
                                                      float e2, float o2, float e3, float o3) {
  unsigned int out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(e0), "f"(o0), "f"(e1), "f"(o1), "f"(e2), "f"(o2), "f"(e3), "f"(o3));
  return out;
}
// ue4m3 byte of x, round to nearest even, saturating (in the low byte of the result).
__device__ __forceinline__ unsigned int tl_q_e4m3_rn(float x) {
  unsigned int out;
  asm volatile("{\n.reg .b16 d;\ncvt.rn.satfinite.e4m3x2.f32 d, %1, %1;\nmov.b32 %0, {d, 0x0000};\n}" : "=r"(out) : "f"(x));
  return out;
}
"""


def _e4m3_value(byte):
    """Exact fp32 value of a ue4m3 byte (e4m3fn, sign bit ignored)."""
    e = (byte >> 3) & 15
    m = byte & 7
    sub = T.cast(m, T.float32) * T.float32(2.0**-9)
    nrm = T.cast(T.shift_left(8 + m, e), T.float32) * T.float32(2.0**-10)
    return T.if_then_else(e == 0, sub, nrm)


def build_sa3_means(batch: int, heads: int, seq_len: int, dim: int = 128, threads: int = 128):
    if dim != 128 or seq_len % 128 != 0:
        raise ValueError(f"sa3_means needs head_dim 128 and seq_len % 128 == 0, got dim={dim} seq_len={seq_len}")
    nblk = seq_len // 128
    bf16 = T.bfloat16
    f32 = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), bf16),
        K: T.Tensor((batch, heads, seq_len, dim), bf16),
        QM: T.Tensor((batch, heads, nblk, dim), bf16),
        KM: T.Tensor((batch, heads, 1, dim), bf16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (by, bz):
            q_t = T.alloc_fragment((128, dim), f32)
            k_t = T.alloc_fragment((128, dim), f32)
            q_acc = T.alloc_fragment((dim,), f32)
            k_acc = T.alloc_fragment((dim,), f32)
            T.fill(k_acc, 0.0)
            for blk in T.serial(nblk):
                T.copy(Q[bz, by, blk * 128, 0], q_t)
                T.copy(K[bz, by, blk * 128, 0], k_t)
                T.reduce_sum(q_t, q_acc, dim=0, clear=True)
                T.reduce_sum(k_t, k_acc, dim=0, clear=False)
                for d in T.Parallel(dim):
                    QM[bz, by, blk, d] = T.cast(q_acc[d] * T.float32(1.0 / 128.0), bf16)
            for d in T.Parallel(dim):
                KM[bz, by, 0, d] = T.cast(k_acc[d] / T.cast(seq_len, f32), bf16)

    return main


def build_sa3_quant(batch: int, heads: int, seq_len: int, dim: int = 128, threads: int = 1024, emit_ksm: bool = False):
    if dim != 128 or seq_len % 128 != 0 or threads != 1024:
        raise ValueError(f"sa3_quant needs head_dim 128, seq_len % 128 == 0 and 1024 threads, got {dim}, {seq_len}, {threads}")
    nblk = seq_len // 128
    bf16 = T.bfloat16
    # the smoothed-K output exists either way (a fixed signature); without emit_ksm it is 1 row wide
    ksm_rows = seq_len if emit_ksm else 1

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), bf16),
        K: T.Tensor((batch, heads, seq_len, dim), bf16),
        V: T.Tensor((batch, heads, seq_len, dim), bf16),
        QM: T.Tensor((batch, heads, nblk, dim), bf16),
        KM: T.Tensor((batch, heads, 1, dim), bf16),
        Qp: T.Tensor((batch, heads, seq_len, 8), T.uint64),
        Kp: T.Tensor((batch, heads, seq_len, 8), T.uint64),
        VTp: T.Tensor((batch, heads, dim, seq_len // 16), T.uint64),
        SFQ: T.Tensor((batch, heads, seq_len, 8), T.uint8),
        SFK: T.Tensor((batch, heads, seq_len, 8), T.uint8),
        SFV: T.Tensor((batch, heads, dim, seq_len // 16), T.uint8),
        KSM: T.Tensor((batch, heads, ksm_rows, dim), bf16),
    ):
        with T.Kernel(3 * nblk, heads, batch, threads=threads) as (bx, by, bz):
            T.import_source(QUANT_SRC)
            tx = T.get_thread_binding()
            role = bx % 3  # 0: Q, 1: K, 2: V^T
            blk = bx // 3
            t = tx // 8  # token (Q/K) or head-dim row (V^T) inside the 128 block
            g = tx % 8  # scale group
            xl = T.alloc_local((16,), T.float32)
            xb = T.alloc_local((16,), bf16)
            mb = T.alloc_local((16,), bf16)
            if role == 0:
                for i in T.vectorized(16):
                    xb[i] = Q[bz, by, blk * 128 + t, 16 * g + i]
                for i in T.vectorized(16):
                    mb[i] = QM[bz, by, blk, 16 * g + i]
                for i in T.serial(16):
                    xl[i] = T.cast(T.cast(T.cast(xb[i], T.float32) - T.cast(mb[i], T.float32), bf16), T.float32)
            elif role == 1:
                for i in T.vectorized(16):
                    xb[i] = K[bz, by, blk * 128 + t, 16 * g + i]
                for i in T.vectorized(16):
                    mb[i] = KM[bz, by, 0, 16 * g + i]
                if emit_ksm:
                    for i in T.serial(16):
                        xb[i] = T.cast(T.cast(xb[i], T.float32) - T.cast(mb[i], T.float32), bf16)
                    for i in T.vectorized(16):
                        KSM[bz, by, blk * 128 + t, 16 * g + i] = xb[i]
                    for i in T.serial(16):
                        xl[i] = T.cast(xb[i], T.float32)
                else:
                    for i in T.serial(16):
                        xl[i] = T.cast(T.cast(T.cast(xb[i], T.float32) - T.cast(mb[i], T.float32), bf16), T.float32)
            else:
                for i in T.serial(16):
                    xl[i] = T.cast(V[bz, by, blk * 128 + 16 * g + i, t], T.float32)
            amax = T.alloc_var(T.float32, init=T.abs(xl[0]))
            for i in T.serial(1, 16):
                amax = T.max(amax, T.abs(xl[i]))
            sfb = T.alloc_var(T.uint32, init=T.call_extern("uint32", "tl_q_e4m3_rn", amax / T.float32(6.0)) & 255)
            sfv = T.alloc_var(T.float32, init=_e4m3_value(sfb))
            inv = T.alloc_var(T.float32, init=T.if_then_else(sfv == T.float32(0.0), T.float32(0.0), T.float32(1.0) / sfv))
            w0 = T.call_extern(
                "uint32",
                "tl_q_e2m1x8_rn",
                xl[0] * inv,
                xl[1] * inv,
                xl[2] * inv,
                xl[3] * inv,
                xl[4] * inv,
                xl[5] * inv,
                xl[6] * inv,
                xl[7] * inv,
            )
            w1 = T.call_extern(
                "uint32",
                "tl_q_e2m1x8_rn",
                xl[8] * inv,
                xl[9] * inv,
                xl[10] * inv,
                xl[11] * inv,
                xl[12] * inv,
                xl[13] * inv,
                xl[14] * inv,
                xl[15] * inv,
            )
            packed = T.cast(w0, T.uint64) | T.shift_left(T.cast(w1, T.uint64), T.uint64(32))
            sf8 = T.cast(sfb, T.uint8)
            tp = (t // 32) * 32 + ((t % 32) // 8) * 2 + ((t % 8) // 2) * 8 + t % 2  # PERM32 is an involution
            if role == 0:
                Qp[bz, by, blk * 128 + t, g] = packed
                SFQ[bz, by, blk * 128 + t, g] = sf8
            elif role == 1:
                Kp[bz, by, blk * 128 + tp, g] = packed
                SFK[bz, by, blk * 128 + (tp % 8) * 16 + tp // 8, g] = sf8  # swizzled scale row of the permuted row
            else:
                VTp[bz, by, t, blk * 8 + g] = packed
                SFV[bz, by, (t % 8) * 16 + t // 8, blk * 8 + g] = sf8

    return main


def sa3_means(batch: int, heads: int, seq_len: int, dim: int = 128):
    """JIT kernel: ``QM, KM = kernel(q, k)``."""
    return tilelang.jit(out_idx=[2, 3])(build_sa3_means)(batch, heads, seq_len, dim)


def sa3_quant(batch: int, heads: int, seq_len: int, dim: int = 128, emit_ksm: bool = False):
    """JIT kernel: ``Qp, Kp, VTp, SFQ, SFK, SFV, KSM = kernel(q, k, v, qm, km)``.

    Packed codes come back as uint64 words (``.view(torch.int8)`` gives the ``[..., D // 2]``
    nibble bytes) and scales as bytes (``.view(torch.uint32)`` gives the row-major words).
    Fast math stays off so the bf16 rounding and the scale reciprocal are exact.
    """
    cfg = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}
    return tilelang.jit(out_idx=[5, 6, 7, 8, 9, 10, 11], pass_configs=cfg)(build_sa3_quant)(batch, heads, seq_len, dim, emit_ksm=emit_ksm)
