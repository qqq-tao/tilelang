"""V1: reproduce the SM120 package pingpong from a schedule expressed as data.

The consumer body does not call T.mma_gemm_blockscaled. It allocates the
register packages as plain local uint32 arrays and emits the pingpong as a
Python-generated sequence of calls into thin C wrappers around the L2
primitives already in gemm_sm120.h. The wrappers enter the kernel translation
unit through T.import_source, so nothing in the shipped code changes.

Emission mechanism (see probe_emit.py): @T.prim_func is an AST transform, so a
plain Python helper cannot emit and a Python `for` in the body is parsed as a
TIR loop. A list comprehension is left alone by the transform and runs at trace
time, and @T.macro helpers emit from wherever they are called. Compile-time
expansion is therefore: [ macro(...) for op in SCHEDULE ].

Acceptance: bitwise-identical output vs the example kernel, and the same
throughput as the hand-written template.
"""

import importlib.util
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
# Run against this worktree even when an editable install of tilelang pins
# imports elsewhere: drop the editable meta-path finder and any sys.path entry
# from another checkout, then put this tree (and its vendored tvm) first.
sys.meta_path[:] = [
    f for f in sys.meta_path
    if "_tilelang_editable" not in (getattr(type(f), "__module__", "") or "")
]
_OTHER_TREE = str(REPO_ROOT.parent) + "/tilelang-"
sys.path[:] = [p for p in sys.path if not (p.startswith(_OTHER_TREE) and not p.startswith(str(REPO_ROOT)))]
for extra in (str(REPO_ROOT / "3rdparty/tvm/python"), str(REPO_ROOT)):
    if extra in sys.path:
        sys.path.remove(extra)
    sys.path.insert(0, extra)

import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout
from tilelang.profiler import do_bench
from tilelang.quantize import swizzle_blockscaled_chunk_kmajor_scale_words

# Thin wrappers so the schedule can call the L2 primitives by name. Note the
# scale copy lives in tl::detail, the other two at tl:: scope.
_WRAPPERS = r"""
#include <tl_templates/cuda/gemm_sm120.h>

extern "C" {

TL_DEVICE void tl_rf_copy_ab(void *pkg, const void *a_base,
                             const void *b_base, int kblock) {
  tl::sm120_copy_fulltile_ab_owner_wide_package(
      *reinterpret_cast<tl::SM120FulltileABOwnerWidePackage *>(pkg),
      static_cast<const char *>(a_base), static_cast<const char *>(b_base),
      kblock);
}

TL_DEVICE void tl_rf_copy_sf(void *spkg, const void *sfa, const void *sfb,
                             int kblock) {
  tl::detail::sm120_copy_scale_tv_package(
      *reinterpret_cast<tl::detail::SM120ScaleTVPackage *>(spkg),
      static_cast<const unsigned int *>(sfa),
      static_cast<const unsigned int *>(sfb), kblock);
}

TL_DEVICE void tl_rf_full(void *c, const void *a, const void *b,
                          const void *sfa, const void *sfb) {
  tl::sm120_mma_blockscaled_kblock_fulltile_package_pingpong(
      static_cast<float *>(c), a, b,
      static_cast<const uint32_t *>(sfa), static_cast<const uint32_t *>(sfb));
}

TL_DEVICE void tl_rf_gemm(void *c, const void *pkg, const void *spkg) {
  tl::sm120_gemm_fulltile_ab_owner_wide_package(
      static_cast<float *>(c),
      *reinterpret_cast<const tl::SM120FulltileABOwnerWidePackage *>(pkg),
      *reinterpret_cast<const tl::detail::SM120ScaleTVPackage *>(spkg));
}

}
"""

# The hand-written template's schedule, as data: (op, package, kblock).
# This is the entire information content of its ~300-line L3 layer.
PINGPONG_4 = (
    ("cp", 0, 0), ("sf", 0, 0),
    ("cp", 1, 1), ("sf", 1, 1),
    ("mma", 0), ("cp", 0, 2), ("sf", 0, 2),
    ("mma", 1), ("cp", 1, 3), ("sf", 1, 3),
    ("mma", 0), ("mma", 1),
)


@T.macro
def _emit_cp(pkg, A_shared, B_shared, kblock):
    # access_ptr, not .data: it carries buffer identity and the access mask, so
    # the pipeline pass can see that this consumes A/B and build the dependency.
    T.call_extern("handle", "tl_rf_copy_ab", T.access_ptr(pkg, "w"),
                  T.access_ptr(A_shared, "r"), T.access_ptr(B_shared, "r"), kblock)


@T.macro
def _emit_sf(spkg, SFA_shared, SFB_shared, kblock):
    T.call_extern("handle", "tl_rf_copy_sf", T.access_ptr(spkg, "w"),
                  T.access_ptr(SFA_shared, "r"), T.access_ptr(SFB_shared, "r"), kblock)


@T.macro
def _emit_full(C_local, A_shared, B_shared, SFA_shared, SFB_shared):
    T.call_extern("handle", "tl_rf_full", T.access_ptr(C_local, "w"),
                  T.access_ptr(A_shared, "r"), T.access_ptr(B_shared, "r"),
                  T.access_ptr(SFA_shared, "r"), T.access_ptr(SFB_shared, "r"))


@T.macro
def _emit_mma(C_local, pkg, spkg):
    T.call_extern("handle", "tl_rf_gemm", T.access_ptr(C_local, "w"),
                  T.access_ptr(pkg, "r"), T.access_ptr(spkg, "r"))


@tilelang.jit(out_idx=None)
def rf_sched_gemm(M, N, K, schedule, block_M=128, block_N=128, block_K=256,
                  num_stages=2, out_dtype=T.bfloat16):
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    wpk = block_K // 64
    k_blocks = K // block_K

    # The package gemm writes the accumulator in the layout the production
    # emitter defines, so reuse that layout for the fragment.
    emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype, b_dtype=in_dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=2, block_col_warps=2,
        warp_row_tiles=64, warp_col_tiles=64, chunk=block_K,
        is_blockscaled=True, kind="mxf4nvf4", scale_vec_size=4, stype="ue4m3",
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
            # T.serial, not T.Pipelined: the warp-specialization pass assigns
            # the schedule calls to the consumer branch but does not recognise
            # an opaque extern as the producer of C_local, so the epilogue (and
            # T.clear, and the C parameter itself) get eliminated. Correctness
            # first here; recovering the pipeline requires the primitives to
            # carry access semantics the pass understands, which is the
            # "fold into the emitter" step of the roadmap.
            for ko in T.serial(K // block_K):
                T.copy(A[by * block_M, ko * block_K], A_shared,
                       annotations={"prefer_instruction": "tma"})
                T.copy(B[bx * block_N, ko * block_K], B_shared,
                       annotations={"prefer_instruction": "tma"})
                for r, w in T.Parallel(block_M, wpk):
                    SFA_shared[r, w] = SFA[(by * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, wpk):
                    SFB_shared[r, w] = SFB[(bx * k_blocks + ko) * block_N + r, w]

                # The schedule, expanded at trace time. A list comprehension is
                # not rewritten by the AST transform, so this runs as real
                # Python; each @T.macro call emits into the kernel body.
                pkgs = (pkg0, pkg1)
                sps = (sp0, sp1)
                _ = [
                    (_emit_cp(pkgs[op[1]], A_shared, B_shared, op[2]) if op[0] == "cp" else
                     _emit_sf(sps[op[1]], SFA_shared, SFB_shared, op[2]) if op[0] == "sf" else
                     _emit_full(C_local, A_shared, B_shared, SFA_shared, SFB_shared) if op[0] == "full" else
                     _emit_mma(C_local, pkgs[op[1]], sps[op[1]]))
                    for op in schedule
                ]

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "ex", str(REPO_ROOT / "examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py"))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["ex"]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def main():
    mod = _load_example()

    print("=== correctness (bitwise vs the hand-written template) ===")
    for M, N, K in [(256, 256, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        A = mod._make_packed_fp4(M, K, seed=71)
        B = mod._make_packed_fp4(N, K, seed=72)
        SFA_sem = mod._make_binary_scale_words(M, K, seed=73)
        SFB_sem = mod._make_binary_scale_words(N, K, seed=74)
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_sem).reshape(-1, 4)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_sem).reshape(-1, 4)

        C_ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        mod.sm120_nvfp4_blockscaled_gemm(M, N, K)(A, B, SFA, SFB, C_ref)
        C_v1 = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        rf_sched_gemm(M, N, K, PINGPONG_4)(A, B, SFA, SFB, C_v1)
        torch.cuda.synchronize()

        same = torch.equal(C_v1.view(torch.uint16), C_ref.view(torch.uint16))
        print(f"  {M}x{N}x{K}: {'BITWISE IDENTICAL' if same else 'MISMATCH'}", flush=True)
        if not same:
            d = (C_v1.float() - C_ref.float()).abs()
            print(f"    max |diff| {d.max().item()}, mismatch frac {(d > 0).float().mean().item():.4f}")

    print("=== throughput @ 2048^3 (median of 3) ===")
    M = N = K = 2048
    A = mod._make_packed_fp4(M, K, seed=71)
    B = mod._make_packed_fp4(N, K, seed=72)
    SFA = swizzle_blockscaled_chunk_kmajor_scale_words(mod._make_binary_scale_words(M, K, seed=73)).reshape(-1, 4)
    SFB = swizzle_blockscaled_chunk_kmajor_scale_words(mod._make_binary_scale_words(N, K, seed=74)).reshape(-1, 4)
    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    kernels = {
        "template (T.mma_gemm_blockscaled)": mod.sm120_nvfp4_blockscaled_gemm(M, N, K),
        "schedule-as-data (V1)": rf_sched_gemm(M, N, K, PINGPONG_4),
    }
    for name, kern in kernels.items():
        kern(A, B, SFA, SFB, C)
        torch.cuda.synchronize()
        ms = statistics.median(
            do_bench(lambda k=kern: k(A, B, SFA, SFB, C), _n_warmup=25, _n_repeat=100)
            for _ in range(3))
        print(f"  {name:36s} {ms:.4f} ms  {2.0 * M * N * K / (ms * 1e-3) / 1e12:7.1f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
