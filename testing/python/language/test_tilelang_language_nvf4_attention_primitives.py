import pytest

import tilelang

import tilelang.language as T

import tilelang.testing


torch = pytest.importorskip("torch")


def _make_rs_vs_ss_kernel(M, N, K, threads):
    """Same NVFP4 blockscaled GEMM twice: A from shared (ss) and A as a uint32 word fragment
    (rs, a_packed_words) filled from the same shared tile; both accumulate into separate outputs."""
    fp4 = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((M, K), fp4),
        B: T.Tensor((N, K), fp4),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C_ss: T.Tensor((M, N), T.float32),
        C_rs: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_sh = T.alloc_shared((M, K), fp4)
            B_sh = T.alloc_shared((N, K), fp4)
            SFA_sh = T.alloc_shared((M, K // 64), T.uint32)
            SFB_sh = T.alloc_shared((N, K // 64), T.uint32)
            A_words = T.view(A_sh, (M, K // 8), dtype=T.uint32)
            Aw = T.alloc_fragment((M, K // 8), T.uint32)
            acc_ss = T.alloc_fragment((M, N), T.float32)
            acc_rs = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_sh)
            T.copy(B, B_sh)
            T.copy(SFA, SFA_sh)
            T.copy(SFB, SFB_sh)
            T.clear(acc_ss)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_ss,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
            )
            for i, w in T.Parallel(M, K // 8):
                Aw[i, w] = A_words[i, w]
            T.clear(acc_rs)
            T.mma_gemm_blockscaled(
                Aw,
                B_sh,
                acc_rs,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
                a_packed_words=True,
            )
            T.copy(acc_ss, C_ss)
            T.copy(acc_rs, C_rs)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("M,N,K,threads", [(16, 32, 64, 32), (32, 32, 128, 32), (128, 128, 128, 256)])
def test_blockscaled_register_a_packed_words_matches_shared_a(M, N, K, threads):
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    # ue4m3 scale bytes in the normal range, packed 4 per word
    sfa_b = torch.randint(0x30, 0x40, (M, K // 16), device="cuda", dtype=torch.uint8).to(torch.int64).reshape(M, K // 64, 4)
    sfb_b = torch.randint(0x30, 0x40, (N, K // 16), device="cuda", dtype=torch.uint8).to(torch.int64).reshape(N, K // 64, 4)
    sfa = (sfa_b[..., 0] | (sfa_b[..., 1] << 8) | (sfa_b[..., 2] << 16) | (sfa_b[..., 3] << 24)).to(torch.uint32).contiguous()
    sfb = (sfb_b[..., 0] | (sfb_b[..., 1] << 8) | (sfb_b[..., 2] << 16) | (sfb_b[..., 3] << 24)).to(torch.uint32).contiguous()
    kernel = tilelang.compile(_make_rs_vs_ss_kernel(M, N, K, threads), target="cuda", out_idx=[4, 5])
    c_ss, c_rs = kernel(a, b, sfa, sfb)
    assert torch.isfinite(c_ss).all() and float(c_ss.abs().max()) > 0
    assert torch.equal(c_ss.view(torch.int32), c_rs.view(torch.int32))
    src = kernel.get_kernel_source()
    assert src.count("sm120_mma_sync_blockscaled") >= 2


def _register_sfa_kernel(M: int, N: int, K: int, threads: int, register_k_start: int = 0):
    """One tile computed twice: A and its scale words in shared memory, and both copied into
    register fragments (A words with a_packed_words, A scales as a uint32 word fragment whose
    layout the GEMM dictates), the way a kernel keeps one Q tile in registers across many K tiles."""
    fp4 = T.float4_e2m1fn
    words = K // 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), fp4),
        B: T.Tensor((N, K), fp4),
        SFA: T.Tensor((M, words), T.uint32),
        SFB: T.Tensor((N, words), T.uint32),
        C_sh: T.Tensor((M, N), T.float32),
        C_reg: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_sh = T.alloc_shared((M, K), fp4)
            B_sh = T.alloc_shared((N, K), fp4)
            SFA_sh = T.alloc_shared((M, words), T.uint32)
            SFB_sh = T.alloc_shared((N, words), T.uint32)
            A_words = T.view(A_sh, (M, K // 8), dtype=T.uint32)
            Aw = T.alloc_fragment((M, K // 8), T.uint32)
            SFAw = T.alloc_fragment((M, words), T.uint32)
            acc_sh = T.alloc_fragment((M, N), T.float32)
            acc_reg = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_sh)
            T.copy(B, B_sh)
            T.copy(SFA, SFA_sh)
            T.copy(SFB, SFB_sh)
            T.clear(acc_sh)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_sh,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
            )
            T.copy(A_words, Aw)
            T.copy(SFA_sh, SFAw)
            T.clear(acc_reg)
            T.mma_gemm_blockscaled(
                Aw,
                B_sh,
                acc_reg,
                SFAw,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=register_k_start,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
                a_packed_words=True,
            )
            T.copy(acc_sh, C_sh)
            T.copy(acc_reg, C_reg)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("M,N,K,threads", [(16, 32, 64, 32), (32, 32, 128, 32), (128, 128, 128, 256)])
def test_blockscaled_register_a_scales_match_shared_scales(M, N, K, threads):
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)

    def _scale_words(rows, cols):  # four ue4m3 bytes per word, all finite; distinct per row and group
        by = torch.randint(0x30, 0x48, (rows, cols, 4), dtype=torch.int64, device="cuda")
        return (by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)).to(torch.uint32).contiguous()

    sfa = _scale_words(M, K // 64)
    sfb = _scale_words(N, K // 64)
    kernel = tilelang.compile(_register_sfa_kernel(M, N, K, threads), target="cuda", out_idx=[4, 5])
    c_sh, c_reg = kernel(a, b, sfa, sfb)
    assert torch.isfinite(c_sh).all() and float(c_sh.abs().max()) > 0
    assert torch.equal(c_sh.view(torch.int32), c_reg.view(torch.int32)), (c_sh - c_reg).abs().max()


def _swizzled_sfb_kernel(M: int, N: int, K: int, threads: int):
    """One tile computed twice: row-major B scales, and the swizzled layout that lets each lane
    load its scale rows contiguously (T.mma_gemm_blockscaled(..., sf_b_swizzled=True))."""
    fp4 = T.float4_e2m1fn
    words = K // 64
    rows = N // 2  # the swizzled buffer holds 4 words (2 original rows) per row
    padded = rows + rows // 8  # one pad row per lane block, to keep the blocks off the same banks

    @T.prim_func
    def main(
        A: T.Tensor((M, K), fp4),
        B: T.Tensor((N, K), fp4),
        SFA: T.Tensor((M, words), T.uint32),
        SFB: T.Tensor((N, words), T.uint32),
        SFB_sw: T.Tensor((N, words), T.uint32),
        C_plain: T.Tensor((M, N), T.float32),
        C_sw: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_sh = T.alloc_shared((M, K), fp4)
            B_sh = T.alloc_shared((N, K), fp4)
            SFA_sh = T.alloc_shared((M, words), T.uint32)
            SFB_sh = T.alloc_shared((N, words), T.uint32)
            SFB_sw_sh = T.alloc_shared((padded, 2 * words), T.uint32)
            acc_plain = T.alloc_fragment((M, N), T.float32)
            acc_sw = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_sh)
            T.copy(B, B_sh)
            T.copy(SFA, SFA_sh)
            T.copy(SFB, SFB_sh)
            for r, w in T.Parallel(rows, 2 * words):
                SFB_sw_sh[(r // 8) * 9 + r % 8, w] = SFB_sw[2 * r + w // words, w % words]
            T.clear(acc_plain)
            T.mma_gemm_blockscaled(
                A_sh,
                B_sh,
                acc_plain,
                SFA_sh,
                SFB_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
            )
            Aw = T.alloc_fragment((M, K // 8), T.uint32)
            T.copy(T.view(A_sh, (M, K // 8), dtype=T.uint32), Aw)
            T.clear(acc_sw)
            T.mma_gemm_blockscaled(
                Aw,
                B_sh,
                acc_sw,
                SFA_sh,
                SFB_sw_sh,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="rowmajor",
                sf_b_swizzled=True,
                a_packed_words=True,
            )
            T.copy(acc_plain, C_plain)
            T.copy(acc_sw, C_sw)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_blockscaled_swizzled_b_scales_match_rowmajor():
    M = N = K = 128
    threads = 256
    kernel = tilelang.jit(out_idx=[5, 6])(_swizzled_sfb_kernel)(M, N, K, threads)
    torch.manual_seed(0)
    a = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")

    def _scale_words(rows, cols):  # four ue4m3 bytes per word, all finite (1.0 .. 3.5)
        by = torch.randint(0x38, 0x42, (rows, cols, 4), dtype=torch.int64, device="cuda")
        return (by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)).to(torch.uint32)

    sfa = _scale_words(M, K // 64)
    sfb = _scale_words(N, K // 64)
    # row n of the B scales moves to (n % 8) * 16 + n // 8, i.e. gather new[p] = old[(p % 16) * 8 + p // 16]
    p_idx = torch.arange(N, device="cuda")
    sfb_sw = sfb.index_select(0, (p_idx % 16) * 8 + p_idx // 16).contiguous()
    c_plain, c_sw = kernel(a.view(torch.int8), b.view(torch.int8), sfa, sfb, sfb_sw)
    assert torch.equal(c_plain, c_sw), (c_plain - c_sw).abs().max()
    assert c_plain.abs().sum() > 0


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_register_scale_offset_rejected():
    with pytest.raises(ValueError, match="require k_start=0"):
        tilelang.compile(_register_sfa_kernel(16, 32, 128, 32, register_k_start=64), target="cuda", out_idx=[4, 5])
