"""V3: Mt=256 cooperative, as a parameter change rather than a second kernel.

Why cooperative and not pingpong: at 256x128 a pingpong warpgroup owns
BLK_M*BLK_N/128 = 256 accumulators per thread, past CUTLASS's 208 threshold, so
the register quota drops to 24/240 and the accumulators spill. Measured on this
box through the CUTLASS C++ path: 256x128x128 pingpong reaches 120 TFLOPS
against 1251 for cooperative, a 90% loss. Mt=256 is cooperative-only in
hardware, not by convention.

Why we split M and CUTLASS splits N: their AtomLayoutMNK is <4,2,1>, i.e. the
two consumer warpgroups take half of N each, because PermTileN is chosen to let
a warp own everything it needs for scale-factor reduction. We do not generate
scale factors here, so the constraint does not apply, and with block_M=256 >
block_N=128 splitting M moves less data: each warpgroup then reads its own half
of A and all of B (256 + 256 rows of traffic) rather than all of A and half of B
(512 + 128).

Splitting M also costs nothing to implement. The package primitives index with
threadIdx.x & 127, so any 128-thread group runs them identically, and the swizzle
phase repeats every Cols16 rows: for row 128 + r the XOR term is
((128 + r) >> shift) & (Cols16 - 1) = (r >> shift) & (Cols16 - 1), because 128 is
a multiple of the period. So warpgroup 1 is warpgroup 0 with the base pointer
advanced by 128 rows, and the same schedule runs on both.

Shared memory per stage: A 16384 + B 8192 + SFA 2048 + SFB 1024 = 27648 B, so
three stages fit in the 99 KiB budget where block_M=128 with block_K=256 only
allowed two.

STATUS: blocked, and the blocker is in the experiment channel rather than in the
design. Warpgroup 0 is exactly right; warpgroup 1 computes but never writes,
because its T.clear and its epilogue are dropped during lowering. See
repro_two_branch_store_layout.py: two thread branches whose accumulators are
annotated with make_mma_store_layout keep only the first branch's analyzable
statements. Without the annotation both survive.

We have to annotate because the MMA here is an opaque call_extern, so there is
no op for layout inference to read the accumulator layout off. The production
kernel has the same two consumer warpgroups and does not hit this, because
T.mma_gemm_blockscaled infers the layout instead.

So call_extern, which carried V1, V1.5 and V2, runs out here: a second consumer
warpgroup needs the accumulator layout to come from inference, which means the
package operations have to become real ops rather than externs. That is the
emitter-side step of the roadmap, and it is now the prerequisite for Mt=256
rather than a later cleanup.

Numbers below are therefore for half the output and are not a throughput claim.
"""

import importlib.util
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
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
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench

_spec = importlib.util.spec_from_file_location(
    "sched_gemm", str(Path(__file__).with_name("sched_gemm.py")))
sched_gemm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sched_gemm)

def store_layout(emitter, buf, thread_offset):
    """make_mma_store_layout, rebased onto a warpgroup's thread bounds.

    Fragment.forward_thread maps an element to an *absolute* thread id, and
    make_mma_store_layout produces 0..127 because it is normally used in a
    region that starts at thread 0. Annotate a second warpgroup's accumulator
    with those same ids and no thread in [128, 256) ever matches, so every
    element's ownership predicate is false: LowerTileOp turns the T.clear and
    the epilogue into empty loops and that half of C is never written, with no
    diagnostic. Adding the region's thread base is the whole fix.

    The production kernel does not need this only because T.mma_gemm_blockscaled
    infers the layout in place; an annotated one has to say which threads it
    means.
    """
    base = emitter.make_mma_store_layout(buf)
    if thread_offset == 0:
        return base
    return T.Fragment(
        buf.shape,
        forward_fn=lambda i, j: (base.map_forward_thread([i, j])[0] + thread_offset,
                                 base.map_forward_index([i, j])))


_WRAPPERS_TEMPLATE = sched_gemm._WRAPPERS_TEMPLATE
pingpong_schedule = sched_gemm.pingpong_schedule
_emit = sched_gemm._emit


@tilelang.jit(out_idx=None)
def coop_gemm(M, N, K, schedule=None, block_M=256, block_N=128, block_K=128,
              num_stages=3, out_dtype=T.bfloat16, producer_regs=40,
              consumer_regs=224, persistent=True, group_m=8):
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    assert block_M == 256, "this kernel is the two-consumer-warpgroup M split"
    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    wpk = block_K // 64
    k_blocks = K // block_K
    half_M = block_M // 2
    if schedule is None:
        schedule = pingpong_schedule(block_K // 64)

    # One CTA per SM, streaming over tiles. Without this the tail wave dominates:
    # 4096^3 at Mt=128 is 6.02 waves over 170 SMs, so a 7-wave runtime for 6.02
    # waves of work costs 14%, and 14% is what the vendor gap measured there.
    sm_num = driver.get_num_sms()
    n_blocks = N // block_N
    m_blocks = M // block_M
    total_tiles = n_blocks * m_blocks
    stream_iters = (total_tiles + sm_num - 1) // sm_num
    # Non-persistent is the same code with one tile per CTA: grid = total_tiles,
    # stride = total_tiles, one iteration, so tile_id == block_id.
    grid = sm_num if persistent else total_tiles
    tile_stride = sm_num if persistent else total_tiles
    tile_iters = stream_iters if persistent else 1

    a_stage_bytes = block_M * block_K // 2
    b_stage_bytes = block_N * block_K // 2
    sfa_stage_bytes = block_M * wpk * 4
    sfb_stage_bytes = block_N * wpk * 4
    a_half_bytes = half_M * block_K // 2
    sfa_half_bytes = half_M * wpk * 4

    emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype, b_dtype=in_dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=2, block_col_warps=2,
        warp_row_tiles=64, warp_col_tiles=64, chunk=block_K,
        is_blockscaled=True, kind="mxf4nvf4", scale_vec_size=4, stype="ue4m3",
    )

    def consumer_ctx(wg, bufs, stage, phase):
        A_shared, B_shared, SFA_shared, SFB_shared, C_local, pkgs, sps, bars = bufs
        return {
            "pkgs": pkgs, "sps": sps,
            "A_shared": A_shared, "B_shared": B_shared,
            "SFA_shared": SFA_shared, "SFB_shared": SFB_shared,
            "C_local": C_local,
            "a_off": stage * a_stage_bytes + wg * a_half_bytes,
            "b_off": stage * b_stage_bytes,
            "sfa_off": stage * sfa_stage_bytes + wg * sfa_half_bytes,
            "sfb_off": stage * sfb_stage_bytes,
            "loaded": bars[0], "consumed": bars[1],
            "stage": stage, "phase": phase,
        }


    def tile_coords(tile_id):
        """Grouped-M rasterisation: sweep N inside a band of group_m tile rows."""
        if group_m <= 1:
            return tile_id % n_blocks, tile_id // n_blocks
        per_group = group_m * n_blocks
        group = tile_id // per_group
        first_m = group * group_m
        rows = T.min(group_m, m_blocks - first_m)
        within = tile_id % per_group
        return within // rows, first_m + within % rows

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M * k_blocks, wpk), T.uint32),
        SFB: T.Tensor((N * k_blocks, wpk), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(grid, threads=384) as block_id:
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((num_stages, block_M, wpk), T.uint32)
            SFB_shared = T.alloc_shared((num_stages, block_N, wpk), T.uint32)
            pkg0 = T.alloc_local((32,), T.uint32)
            pkg1 = T.alloc_local((32,), T.uint32)
            sp0 = T.alloc_local((4,), T.uint32)
            sp1 = T.alloc_local((4,), T.uint32)

            loaded = T.alloc_barrier([128] * num_stages)
            consumed = T.alloc_barrier([256] * num_stages)

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                B_shared: make_swizzled_layout(B_shared),
            })
            T.import_source(_WRAPPERS_TEMPLATE.replace("__BLOCK_K__", str(block_K)))

            tx = T.get_thread_binding()

            if tx >= 256:
                # At 384 threads the register file caps every thread at
                # 65536/384 = 170, but the kernel wants 218, so a uniform cap
                # spills. Splitting it asymmetrically -- the producer only
                # addresses TMA descriptors -- fits: 128*40 + 256*224 = 62464.
                if producer_regs > 0:
                    T.set_max_nreg(producer_regs, 0)
                for stream in T.serial(tile_iters):
                    tile_id = block_id + stream * tile_stride
                    if tile_id < total_tiles:
                        bx, by = tile_coords(tile_id)
                        for ko in T.serial(k_blocks):
                            # The phase counter runs over the CTA's whole
                            # lifetime rather than per tile. Barriers live in
                            # shared memory, so producer and consumers only have
                            # to agree with each other, and they skip the same
                            # iterations.
                            gko = stream * k_blocks + ko
                            stage = gko % num_stages
                            phase = (gko // num_stages) & 1
                            T.barrier_wait(consumed[stage], phase ^ 1)
                            T.tma_copy(
                                A[by * block_M:(by + 1) * block_M,
                                  ko * block_K:(ko + 1) * block_K],
                                A_shared[stage, :, :], barrier=loaded[stage],
                                leader_scope_threads=128)
                            T.tma_copy(
                                B[bx * block_N:(bx + 1) * block_N,
                                  ko * block_K:(ko + 1) * block_K],
                                B_shared[stage, :, :], barrier=loaded[stage],
                                leader_scope_threads=128)
                            # The scale source blocks rows by 128 whatever
                            # block_M is, so a 256-row tile spans row blocks
                            # 2*by and 2*by+1, each a contiguous rectangle.
                            for rb in T.serial(2):
                                T.copy(SFA[((2 * by + rb) * k_blocks + ko) * 128:
                                           ((2 * by + rb) * k_blocks + ko) * 128 + 128, :],
                                       SFA_shared[stage, rb * 128:(rb + 1) * 128, :])
                            T.copy(SFB[(bx * k_blocks + ko) * block_N:
                                       (bx * k_blocks + ko) * block_N + block_N, :],
                                   SFB_shared[stage, :, :])
                            T.barrier_arrive(loaded[stage])
            elif tx < 128:
                if consumer_regs > 0:
                    T.set_max_nreg(consumer_regs, 1)
                C0_local = T.alloc_fragment((half_M, block_N), accum_dtype)
                T.annotate_layout({C0_local: store_layout(emitter, C0_local, 0)})
                for stream in T.serial(tile_iters):
                    tile_id = block_id + stream * tile_stride
                    if tile_id < total_tiles:
                        bx, by = tile_coords(tile_id)
                        T.clear(C0_local)
                        for ko in T.serial(k_blocks):
                            gko = stream * k_blocks + ko
                            stage = gko % num_stages
                            phase = (gko // num_stages) & 1
                            _ = [_emit(op, consumer_ctx(
                                0, (A_shared, B_shared, SFA_shared, SFB_shared,
                                    C0_local, (pkg0, pkg1), (sp0, sp1),
                                    (loaded, consumed)), stage, phase))
                                 for op in schedule]
                        T.copy(C0_local,
                               C[by * block_M:by * block_M + half_M,
                                 bx * block_N:(bx + 1) * block_N])
            else:
                # Same body, other half of M. Written out rather than shared:
                # the accumulators must not coexist in one thread's register
                # budget, and each needs its own thread-rebased store layout.
                C1_local = T.alloc_fragment((half_M, block_N), accum_dtype)
                T.annotate_layout({C1_local: store_layout(emitter, C1_local, 128)})
                for stream in T.serial(tile_iters):
                    tile_id = block_id + stream * tile_stride
                    if tile_id < total_tiles:
                        bx, by = tile_coords(tile_id)
                        T.clear(C1_local)
                        for ko in T.serial(k_blocks):
                            gko = stream * k_blocks + ko
                            stage = gko % num_stages
                            phase = (gko // num_stages) & 1
                            _ = [_emit(op, consumer_ctx(
                                1, (A_shared, B_shared, SFA_shared, SFB_shared,
                                    C1_local, (pkg0, pkg1), (sp0, sp1),
                                    (loaded, consumed)), stage, phase))
                                 for op in schedule]
                        T.copy(C1_local,
                               C[by * block_M + half_M:(by + 1) * block_M,
                                 bx * block_N:(bx + 1) * block_N])

    return main


def main():
    mod = sched_gemm._load_example()

    print("=== correctness: bitwise vs the block_K=256 template ===")
    for M, N, K in [(256, 256, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        A_r, B_r, SFA_r, SFB_r = sched_gemm._make_inputs(mod, M, N, K, 256)
        C_ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        mod.sm120_nvfp4_blockscaled_gemm(M, N, K)(A_r, B_r, SFA_r, SFB_r, C_ref)

        A, B, SFA, SFB = sched_gemm._make_inputs(mod, M, N, K, 128)
        C_out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        coop_gemm(M, N, K)(A, B, SFA, SFB, C_out)
        torch.cuda.synchronize()
        same = torch.equal(C_out.view(torch.uint16), C_ref.view(torch.uint16))
        print(f"  Mt=256 coop {M}x{N}x{K}: {'BITWISE IDENTICAL' if same else 'MISMATCH'}",
              flush=True)
        if not same:
            d = (C_out.float() - C_ref.float()).abs()
            print(f"    max |diff| {d.max().item()}, "
                  f"mismatch frac {(d > 0).float().mean().item():.4f}")

    print("=== throughput: high-entropy scales, 200 warmup, median of 3 ===")
    for M, N, K in ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384)):
        C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        ins128 = sched_gemm._make_inputs(mod, M, N, K, 128, high_entropy=True)
        ins256 = sched_gemm._make_inputs(mod, M, N, K, 256, high_entropy=True)
        arms = [("Mt=128 bK=256 st=2", sched_gemm.rf_ws_gemm(M, N, K, block_K=256), ins256),
                ("Mt=128 bK=128 st=2", sched_gemm.rf_ws_gemm(M, N, K, block_K=128), ins128),
                ("Mt=128 bK=128 st=4",
                 sched_gemm.rf_ws_gemm(M, N, K, block_K=128, num_stages=4), ins128)]
        for st in (3,):
            arms.append((f"Mt=256 coop bK=128 st={st}",
                         coop_gemm(M, N, K, num_stages=st), ins128))
        for name, kern, ins in arms:
            kern(*ins, C)
            torch.cuda.synchronize()
            ms = statistics.median(
                do_bench(lambda k=kern, i=ins: k(*i, C), _n_warmup=200, _n_repeat=200)
                for _ in range(3))
            print(f"  {M}^3  {name:26s} {2.0 * M * N * K / (ms * 1e-3) / 1e12:7.1f} TFLOPS",
                  flush=True)


if __name__ == "__main__":
    main()
