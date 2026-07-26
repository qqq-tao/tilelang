"""V1.5: the pipeline, expressed by the schedule instead of inferred from it.

V1 had to fall back to T.serial. With T.Pipelined the warp-specialization pass
puts the schedule calls in the consumer branch but does not recognise an opaque
extern as the producer of C_local, so the epilogue, the T.clear and even the C
kernel parameter get eliminated. Correct, but no producer/consumer overlap, and
11% off the hand-written template.

CUTLASS does not infer these roles either. In
gemm/collective/sm120_blockscaled_mma_tma.hpp the pipeline is an object and the
roles are two separate methods: load() calls producer_acquire/producer_commit,
mma() calls consumer_wait/consumer_release. So the fix is not to teach the pass
about our extern -- it is to put the pipeline operations in the schedule, next
to the package operations, and drive the split ourselves. TileLang already has
the primitives (T.alloc_barrier / T.barrier_wait / T.barrier_arrive).

The schedule vocabulary therefore becomes:

    ("wait",  bar)          consumer waits for the stage to be filled
    ("cp",  pkg, kblock)    A/B fragments -> register package
    ("sf",  pkg, kblock)    scale selectors -> register package
    ("mma", pkg)            issue the package
    ("arrive", bar)         release the stage back to the producer

which makes one more thing expressible than the template can say: where the
release goes. The template releases after the last MMA because that is where
the C++ loop ends, but nothing reads shared memory after the last "sf", so the
release can move up and let the producer start refilling while the final MMAs
run. That is PINGPONG_4_EARLY below, and it is a schedule edit, not a rewrite.

Measured, it buys nothing: -0.45% at 2048^3 and +0.03% at 8192^3, over five
interleaved rounds with the order flipped each round. A first run with a fixed
order showed +0.65%, which was drift correlated with position, not a gain. The
point stands as an expressibility demonstration and not as an optimisation.

Against the example kernel, from a 14-entry list: +3.4% at 2048^3 and parity at
8192^3 (1282 both). The gain at the smaller shape comes from the explicit
two-warpgroup split rather than from the emission order.
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

# Thin wrappers so the schedule can call the L2 primitives by name. The base
# pointer and the stage byte offset are passed separately: access_ptr keeps the
# buffer identity and the access mask that the passes need, and the offset
# selects the pipeline stage. Note the scale copy lives in tl::detail, the
# other two at tl:: scope.
_WRAPPERS = r"""
#include <tl_templates/cuda/gemm_sm120.h>

extern "C" {

TL_DEVICE void tl_rf_copy_ab(void *pkg, const void *a_base, int a_off,
                             const void *b_base, int b_off, int kblock) {
  tl::sm120_copy_fulltile_ab_owner_wide_package(
      *reinterpret_cast<tl::SM120FulltileABOwnerWidePackage *>(pkg),
      static_cast<const char *>(a_base) + a_off,
      static_cast<const char *>(b_base) + b_off, kblock);
}

TL_DEVICE void tl_rf_copy_sf(void *spkg, const void *sfa, int sfa_off,
                             const void *sfb, int sfb_off, int kblock) {
  tl::detail::sm120_copy_scale_tv_package(
      *reinterpret_cast<tl::detail::SM120ScaleTVPackage *>(spkg),
      reinterpret_cast<const unsigned int *>(static_cast<const char *>(sfa) + sfa_off),
      reinterpret_cast<const unsigned int *>(static_cast<const char *>(sfb) + sfb_off),
      kblock);
}

TL_DEVICE void tl_rf_gemm(void *c, const void *pkg, const void *spkg) {
  tl::sm120_gemm_fulltile_ab_owner_wide_package(
      static_cast<float *>(c),
      *reinterpret_cast<const tl::SM120FulltileABOwnerWidePackage *>(pkg),
      *reinterpret_cast<const tl::detail::SM120ScaleTVPackage *>(spkg));
}

}
"""

# The template's schedule, now including the pipeline operations it performs
# implicitly by where its loop begins and ends.
PINGPONG_4 = (
    ("wait", "loaded"),
    ("cp", 0, 0), ("sf", 0, 0),
    ("cp", 1, 1), ("sf", 1, 1),
    ("mma", 0), ("cp", 0, 2), ("sf", 0, 2),
    ("mma", 1), ("cp", 1, 3), ("sf", 1, 3),
    ("mma", 0), ("mma", 1),
    ("arrive", "consumed"),
)

# Same packages, same order, one operation moved: the stage is released as soon
# as the last read of shared memory has happened, so the producer can refill it
# while the final two MMAs still run. Not expressible as a template parameter.
PINGPONG_4_EARLY = (
    ("wait", "loaded"),
    ("cp", 0, 0), ("sf", 0, 0),
    ("cp", 1, 1), ("sf", 1, 1),
    ("mma", 0), ("cp", 0, 2), ("sf", 0, 2),
    ("mma", 1), ("cp", 1, 3), ("sf", 1, 3),
    ("arrive", "consumed"),
    ("mma", 0), ("mma", 1),
)

SCHEDULES = {"baseline": PINGPONG_4, "early-release": PINGPONG_4_EARLY}


@T.macro
def _m_cp(pkg, A_shared, a_off, B_shared, b_off, kblock):
    T.call_extern("handle", "tl_rf_copy_ab", T.access_ptr(pkg, "w"),
                  T.access_ptr(A_shared, "r"), a_off,
                  T.access_ptr(B_shared, "r"), b_off, kblock)


@T.macro
def _m_sf(spkg, SFA_shared, sfa_off, SFB_shared, sfb_off, kblock):
    T.call_extern("handle", "tl_rf_copy_sf", T.access_ptr(spkg, "w"),
                  T.access_ptr(SFA_shared, "r"), sfa_off,
                  T.access_ptr(SFB_shared, "r"), sfb_off, kblock)


@T.macro
def _m_mma(C_local, pkg, spkg):
    T.call_extern("handle", "tl_rf_gemm", T.access_ptr(C_local, "w"),
                  T.access_ptr(pkg, "r"), T.access_ptr(spkg, "r"))


@T.macro
def _m_wait(bars, slot, phase):
    # Index inside the macro: subscripting the barrier handle from plain Python
    # yields a Var, not the BufferLoad the builtin expects.
    T.barrier_wait(bars[slot], phase)


@T.macro
def _m_arrive(bars, slot):
    T.barrier_arrive(bars[slot])


def _emit(op, ctx):
    """Dispatch one schedule entry.

    A plain Python function, called from the trace-time list comprehension.
    Its own body is not AST-transformed, so it could not emit T.* directly --
    but calling a @T.macro from here does emit, because the macro's body is
    transformed and its invocation writes into the builder. That is what lets
    the dispatch be an ordinary if/elif instead of a chain of ternaries.
    """
    kind = op[0]
    if kind == "cp":
        _m_cp(ctx["pkgs"][op[1]], ctx["A_shared"], ctx["a_off"],
              ctx["B_shared"], ctx["b_off"], op[2])
    elif kind == "sf":
        _m_sf(ctx["sps"][op[1]], ctx["SFA_shared"], ctx["sfa_off"],
              ctx["SFB_shared"], ctx["sfb_off"], op[2])
    elif kind == "mma":
        _m_mma(ctx["C_local"], ctx["pkgs"][op[1]], ctx["sps"][op[1]])
    elif kind == "wait":
        _m_wait(ctx[op[1]], ctx["stage"], ctx["phase"])
    elif kind == "arrive":
        _m_arrive(ctx[op[1]], ctx["stage"])
    else:
        raise ValueError(f"unknown schedule op {op!r}")


@tilelang.jit(out_idx=None)
def rf_ws_gemm(M, N, K, schedule, block_M=128, block_N=128, block_K=256,
               num_stages=2, out_dtype=T.bfloat16):
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    wpk = block_K // 64
    k_blocks = K // block_K
    # Stage strides in bytes: fp4 is half a byte per element, the scale words
    # are uint32.
    a_stage_bytes = block_M * block_K // 2
    b_stage_bytes = block_N * block_K // 2
    sfa_stage_bytes = block_M * wpk * 4
    sfb_stage_bytes = block_N * wpk * 4
    scale_tile_words = block_M * wpk

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
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((num_stages, block_M, wpk), T.uint32)
            SFB_shared = T.alloc_shared((num_stages, block_N, wpk), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            pkg0 = T.alloc_local((32,), T.uint32)
            pkg1 = T.alloc_local((32,), T.uint32)
            sp0 = T.alloc_local((4,), T.uint32)
            sp1 = T.alloc_local((4,), T.uint32)

            loaded = T.alloc_barrier([128] * num_stages)
            consumed = T.alloc_barrier([128] * num_stages)

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                B_shared: make_swizzled_layout(B_shared),
                C_local: emitter.make_mma_store_layout(C_local),
            })
            T.import_source(_WRAPPERS)

            tx = T.get_thread_binding()

            if tx >= 128:
                # Producer.
                for ko in T.serial(k_blocks):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.barrier_wait(consumed[stage], phase ^ 1)
                    T.tma_copy(
                        A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                        A_shared[stage, :, :], barrier=loaded[stage], leader_scope_threads=128)
                    T.tma_copy(
                        B[bx * block_N:(bx + 1) * block_N, ko * block_K:(ko + 1) * block_K],
                        B_shared[stage, :, :], barrier=loaded[stage], leader_scope_threads=128)
                    # The scale tensors arrive pre-swizzled from the host, so
                    # staging them is a straight strided copy. CUTLASS TMAs
                    # these on the same mbarrier with a single thread; doing
                    # the same here needs a layout whose domain is (M, K) and
                    # whose codomain is the blocked array, which is a separate
                    # piece of work.
                    scale_lane = tx - 128
                    for it in T.serial((scale_tile_words + 127) // 128):
                        flat = it * 128 + scale_lane
                        if flat < scale_tile_words:
                            SFA_shared[stage, flat // wpk, flat % wpk] = \
                                SFA[(by * k_blocks + ko) * block_M + flat // wpk, flat % wpk]
                            SFB_shared[stage, flat // wpk, flat % wpk] = \
                                SFB[(bx * k_blocks + ko) * block_N + flat // wpk, flat % wpk]
                    T.barrier_arrive(loaded[stage])
            else:
                # Consumer: the schedule, expanded at trace time.
                T.clear(C_local)
                for ko in T.serial(k_blocks):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    ctx = {
                        "pkgs": (pkg0, pkg1), "sps": (sp0, sp1),
                        "A_shared": A_shared, "B_shared": B_shared,
                        "SFA_shared": SFA_shared, "SFB_shared": SFB_shared,
                        "C_local": C_local,
                        "a_off": stage * a_stage_bytes, "b_off": stage * b_stage_bytes,
                        "sfa_off": stage * sfa_stage_bytes, "sfb_off": stage * sfb_stage_bytes,
                        "loaded": loaded, "consumed": consumed,
                        "stage": stage, "phase": phase,
                    }
                    _ = [_emit(op, ctx) for op in schedule]

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


def _make_inputs(mod, M, N, K):
    A = mod._make_packed_fp4(M, K, seed=71)
    B = mod._make_packed_fp4(N, K, seed=72)
    SFA = swizzle_blockscaled_chunk_kmajor_scale_words(
        mod._make_binary_scale_words(M, K, seed=73)).reshape(-1, 4)
    SFB = swizzle_blockscaled_chunk_kmajor_scale_words(
        mod._make_binary_scale_words(N, K, seed=74)).reshape(-1, 4)
    return A, B, SFA, SFB


def main():
    mod = _load_example()

    print("=== correctness (bitwise vs the hand-written template) ===")
    for name, sched in SCHEDULES.items():
        for M, N, K in [(256, 256, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
            A, B, SFA, SFB = _make_inputs(mod, M, N, K)
            C_ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
            mod.sm120_nvfp4_blockscaled_gemm(M, N, K)(A, B, SFA, SFB, C_ref)
            C_v15 = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
            rf_ws_gemm(M, N, K, sched)(A, B, SFA, SFB, C_v15)
            torch.cuda.synchronize()
            same = torch.equal(C_v15.view(torch.uint16), C_ref.view(torch.uint16))
            print(f"  {name:14s} {M}x{N}x{K}: {'BITWISE IDENTICAL' if same else 'MISMATCH'}",
                  flush=True)
            if not same:
                d = (C_v15.float() - C_ref.float()).abs()
                print(f"    max |diff| {d.max().item()}, "
                      f"mismatch frac {(d > 0).float().mean().item():.4f}")

    print("=== throughput @ 2048^3 (median of 3) ===")
    M = N = K = 2048
    A, B, SFA, SFB = _make_inputs(mod, M, N, K)
    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    kernels = [("template (T.mma_gemm_blockscaled)", mod.sm120_nvfp4_blockscaled_gemm(M, N, K))]
    kernels += [(f"schedule-as-data ({n})", rf_ws_gemm(M, N, K, s)) for n, s in SCHEDULES.items()]
    for name, kern in kernels:
        kern(A, B, SFA, SFB, C)
        torch.cuda.synchronize()
        ms = statistics.median(
            do_bench(lambda k=kern: k(A, B, SFA, SFB, C), _n_warmup=25, _n_repeat=100)
            for _ in range(3))
        print(f"  {name:38s} {ms:.4f} ms  {2.0 * M * N * K / (ms * 1e-3) / 1e12:7.1f} TFLOPS",
              flush=True)


if __name__ == "__main__":
    main()
