"""V1: schedule-as-data reproduction of the SM120 package pingpong.

The consumer body does NOT call T.mma_gemm_blockscaled. Instead it allocates
register packages as plain local uint32 arrays and emits the pingpong as a
Python-generated sequence of call_extern's to thin injected wrappers around
the existing L2 primitives in gemm_sm120.h. Zero repository changes: the
wrappers enter the kernel TU via T.import_source (pragma_import_c).

Validation: bitwise-identical C vs the reference example kernel, then perf.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout
from tilelang.profiler import do_bench
from tilelang.quantize import swizzle_blockscaled_chunk_kmajor_scale_words

_WRAPPERS = r"""
#include <tl_templates/cuda/gemm_sm120.h>

extern "C" {

TL_DEVICE void tl_rf_copy_ab(unsigned int *pkg, const void *a_base,
                             const void *b_base, int kblock) {
  tl::sm120_copy_fulltile_ab_owner_wide_package(
      *reinterpret_cast<tl::SM120FulltileABOwnerWidePackage *>(pkg),
      static_cast<const char *>(a_base), static_cast<const char *>(b_base),
      kblock);
}

TL_DEVICE void tl_rf_copy_sf(unsigned int *spkg, const unsigned int *sfa,
                             const unsigned int *sfb, int kblock) {
  tl::detail::sm120_copy_scale_tv_package(
      *reinterpret_cast<tl::detail::SM120ScaleTVPackage *>(spkg), sfa, sfb,
      kblock);
}

TL_DEVICE void tl_rf_gemm(float *c, const unsigned int *pkg,
                          const unsigned int *spkg) {
  tl::sm120_gemm_fulltile_ab_owner_wide_package(
      c, *reinterpret_cast<const tl::SM120FulltileABOwnerWidePackage *>(pkg),
      *reinterpret_cast<const tl::detail::SM120ScaleTVPackage *>(spkg));
}

}
"""

# The pingpong schedule as data: (op, package_idx, kblock)
PINGPONG_4 = [
    ("cp", 0, 0), ("sf", 0, 0),
    ("cp", 1, 1), ("sf", 1, 1),
    ("mma", 0), ("cp", 0, 2), ("sf", 0, 2),
    ("mma", 1), ("cp", 1, 3), ("sf", 1, 3),
    ("mma", 0), ("mma", 1),
]

_SM120_SCALE_LAYOUT = "blockscaled_chunk_kmajor"


def _emit_schedule(ops, pkgs, sps, A_shared, B_shared, SFA_shared, SFB_shared, C_local):
    """Trace-time emission: plain python loop over compile-time schedule data."""
    for op in ops:
        if op[0] == "cp":
            _, p, kb = op
            T.call_extern("handle", "tl_rf_copy_ab", T.address_of(pkgs[p][0]),
                          T.address_of(A_shared[0, 0]), T.address_of(B_shared[0, 0]), kb)
        elif op[0] == "sf":
            _, p, kb = op
            T.call_extern("handle", "tl_rf_copy_sf", T.address_of(sps[p][0]),
                          T.address_of(SFA_shared[0, 0]), T.address_of(SFB_shared[0, 0]), kb)
        else:
            _, p = op
            T.call_extern("handle", "tl_rf_gemm",
                          T.access_ptr(C_local, "w"),
                          T.address_of(pkgs[p][0]), T.address_of(sps[p][0]))


@tilelang.jit(out_idx=None)
def rf_sched_gemm(M, N, K, schedule_ops, block_M=128, block_N=128, block_K=256, num_stages=2, out_dtype=T.bfloat16):
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    wpk = block_K // 64
    k_blocks = K // block_K

    # Fragment store layout must match what the package gemm writes; reuse the
    # same emitter helper the production lowering uses.
    emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=block_K,
        is_blockscaled=True,
        kind="mxf4nvf4",
        scale_vec_size=4,
        stype="ue4m3",
    )

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M * k_blocks, wpk), T.uint32),
        SFB: T.Tensor((N * k_blocks, wpk), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, wpk), T.uint32)
            SFB_shared = T.alloc_shared((block_N, wpk), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            pkg0 = T.alloc_local((32,), T.uint32)
            pkg1 = T.alloc_local((32,), T.uint32)
            sp0 = T.alloc_local((4,), T.uint32)
            sp1 = T.alloc_local((4,), T.uint32)

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                B_shared: make_swizzled_layout(B_shared),
                C_local: emitter.make_mma_store_layout(C_local),
            })
            T.import_source(_WRAPPERS)

            T.clear(C_local)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared, annotations={"prefer_instruction": "tma"})
                T.copy(B[bx * block_N, ko * block_K], B_shared, annotations={"prefer_instruction": "tma"})
                for r, w in T.Parallel(block_M, wpk):
                    SFA_shared[r, w] = SFA[(by * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, wpk):
                    SFB_shared[r, w] = SFB[(bx * k_blocks + ko) * block_N + r, w]

                # ---- the schedule, emitted from data (trace-time python) ----
                _emit_schedule(
                    schedule_ops, (pkg0, pkg1), (sp0, sp1),
                    A_shared, B_shared, SFA_shared, SFB_shared, C_local,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "ex", str(REPO_ROOT / "examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py"))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["ex"]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def main():
    mod = _load_example()
    shapes = [(256, 256, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    for M, N, K in shapes:
        A = mod._make_packed_fp4(M, K, seed=71)
        B = mod._make_packed_fp4(N, K, seed=72)
        SFA_sem = mod._make_binary_scale_words(M, K, seed=73)
        SFB_sem = mod._make_binary_scale_words(N, K, seed=74)
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_sem).reshape(-1, 4)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_sem).reshape(-1, 4)

        ref_kernel = mod.sm120_nvfp4_blockscaled_gemm(M, N, K)
        C_ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        ref_kernel(A, B, SFA, SFB, C_ref)

        v1_kernel = rf_sched_gemm(M, N, K, tuple(PINGPONG_4))
        C_v1 = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        v1_kernel(A, B, SFA, SFB, C_v1)
        torch.cuda.synchronize()

        bitwise = torch.equal(C_v1.view(torch.uint16), C_ref.view(torch.uint16))
        print(f"{M}x{N}x{K}: bitwise vs example = {'IDENTICAL' if bitwise else 'MISMATCH'}", flush=True)
        if not bitwise:
            diff = (C_v1.float() - C_ref.float()).abs()
            print(f"  max abs diff {diff.max().item()}, mismatch frac {(diff > 0).float().mean().item():.4f}", flush=True)

    # perf at 2048^3 vs reference
    M = N = K = 2048
    A = mod._make_packed_fp4(M, K, seed=71)
    B = mod._make_packed_fp4(N, K, seed=72)
    SFA = swizzle_blockscaled_chunk_kmajor_scale_words(mod._make_binary_scale_words(M, K, seed=73)).reshape(-1, 4)
    SFB = swizzle_blockscaled_chunk_kmajor_scale_words(mod._make_binary_scale_words(N, K, seed=74)).reshape(-1, 4)
    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    ref_kernel = mod.sm120_nvfp4_blockscaled_gemm(M, N, K)
    v1_kernel = rf_sched_gemm(M, N, K, tuple(PINGPONG_4))
    for name, k in (("example", ref_kernel), ("v1-sched", v1_kernel)):
        ms = do_bench(lambda kk=k: kk(A, B, SFA, SFB, C), _n_warmup=25, _n_repeat=100)
        print(f"{name}: {ms:.4f} ms  {2.0 * M * N * K / (ms * 1e-3) / 1e12:.1f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
