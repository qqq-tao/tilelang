"""Warpgroup-level pingpong: two consumers on alternating tiles, to hide the epilogue.

Measured cause (see README, A3): with a persistent kernel the mainloop pipeline
runs continuously across tile boundaries, so the only fixed cost left per tile is
the epilogue. Holding the tile count at 1024 and varying only K shows what that
costs: 1060.5 / 1132.4 / 1213.7 / 1213.8 TFLOPS at 16 / 32 / 64 / 128
k-iterations. Short k-loops cannot amortise it, and 4096^3 has 16.

CUTLASS hides it by ordering the two math warpgroups against each other so that
one is issuing MMAs while the other stores -- MMA occupies the tensor cores,
the epilogue occupies CUDA cores and the store path, and they do not contend.
The ordering is an OrderedSequenceBarrier<2,2>: two stages (mainloop, epilogue)
across two groups, where group i's arrive releases group i+1 at the same stage.

Here that is two pairs of barriers, mirroring the production kernel:

    wg_order[0]    consumer 1 signals, consumer 0 waits     mainloop order
    wg_order[1]    consumer 0 signals, consumer 1 waits
    store_order[*] the same ring, one stage later           epilogue order

The two consumers take alternating tiles and share one mainloop pipeline, so a
single global k-iteration counter drives the stage rotation for all three
warpgroups: producer, consumer 0 and consumer 1 all compute
gko = tile_index * k_blocks + ko and derive stage and phase from it.

Unlike v3_coop.py, which splits M inside one tile and therefore has both
consumers epilogue at the same moment, this splits tiles between them.
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
from tilelang.carver.arch import driver
from tilelang.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout
from tilelang.profiler import do_bench

_spec = importlib.util.spec_from_file_location(
    "sched_gemm", str(Path(__file__).with_name("sched_gemm.py")))
sched_gemm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sched_gemm)

_WRAPPERS_TEMPLATE = sched_gemm._WRAPPERS_TEMPLATE
pingpong_schedule = sched_gemm.pingpong_schedule
_emit = sched_gemm._emit

_v3spec = importlib.util.spec_from_file_location(
    "v3_coop", str(Path(__file__).with_name("v3_coop.py")))
_v3 = importlib.util.module_from_spec(_v3spec)
_v3spec.loader.exec_module(_v3)
store_layout = _v3.store_layout


@tilelang.jit(out_idx=None)
def wg_pingpong_gemm(M, N, K, schedule=None, block_M=128, block_N=128, block_K=256,
                     num_stages=2, out_dtype=T.bfloat16, group_m=1, order=True,
                     split_stages=False):
    # order=False is not a valid configuration, only a way to measure the
    # barriers' cost: the producer fills stages in one global order, so the two
    # consumers have to take them in that order or deadlock. Dropping the
    # ordering gives an unspecified launch failure, as it should.
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0
    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    wpk = block_K // 64
    k_blocks = K // block_K
    if schedule is None:
        schedule = pingpong_schedule(block_K // 64)

    sm_num = driver.get_num_sms()
    n_blocks = N // block_N
    m_blocks = M // block_M
    total_tiles = n_blocks * m_blocks
    # Each CTA takes two tiles per round, one per consumer warpgroup.
    tiles_per_round = 2 * sm_num
    rounds = (total_tiles + tiles_per_round - 1) // tiles_per_round

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

    def tile_coords(tile_id):
        if group_m <= 1:
            return tile_id % n_blocks, tile_id // n_blocks
        per_group = group_m * n_blocks
        group = tile_id // per_group
        first_m = group * group_m
        rows = T.min(group_m, m_blocks - first_m)
        within = tile_id % per_group
        return within // rows, first_m + within % rows

    def ctx_for(bufs, C_local, stage, phase):
        A_shared, B_shared, SFA_shared, SFB_shared, pkgs, sps, loaded, consumed = bufs
        return {
            "pkgs": pkgs, "sps": sps,
            "A_shared": A_shared, "B_shared": B_shared,
            "SFA_shared": SFA_shared, "SFB_shared": SFB_shared,
            "C_local": C_local,
            "a_off": stage * a_stage_bytes, "b_off": stage * b_stage_bytes,
            "sfa_off": stage * sfa_stage_bytes, "sfb_off": stage * sfb_stage_bytes,
            "loaded": loaded, "consumed": consumed,
            "stage": stage, "phase": phase,
        }

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M * k_blocks, wpk), T.uint32),
        SFB: T.Tensor((N * k_blocks, wpk), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(sm_num, threads=384) as block_id:
            # split_stages gives each consumer its own set, so the producer
            # can fill one consumer's next tile while the other is still
            # consuming -- the handover no longer waits for a pipeline fill.
            sets = 2 if split_stages else 1
            all_stages = sets * num_stages
            A_shared = T.alloc_shared((all_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((all_stages, block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((all_stages, block_M, wpk), T.uint32)
            SFB_shared = T.alloc_shared((all_stages, block_N, wpk), T.uint32)
            pkg0 = T.alloc_local((32,), T.uint32)
            pkg1 = T.alloc_local((32,), T.uint32)
            sp0 = T.alloc_local((4,), T.uint32)
            sp1 = T.alloc_local((4,), T.uint32)

            # One consumer warpgroup owns any given stage, so 128 arrivals.
            loaded = T.alloc_barrier([128] * all_stages)
            consumed = T.alloc_barrier([128] * all_stages)
            # OrderedSequenceBarrier<2,2>: a two-element ring per stage.
            wg_order = T.alloc_barrier([128] * 2)
            store_order = T.alloc_barrier([128] * 2)

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                B_shared: make_swizzled_layout(B_shared),
            })
            T.import_source(_WRAPPERS_TEMPLATE.replace("__BLOCK_K__", str(block_K)))

            tx = T.get_thread_binding()
            bufs = (A_shared, B_shared, SFA_shared, SFB_shared,
                    (pkg0, pkg1), (sp0, sp1), loaded, consumed)

            if tx >= 256:
                # Producer: streams both consumers' tiles, in the order they
                # will be consumed, on one pipeline.
                for rnd in T.serial(rounds):
                    for half in T.serial(2):
                        tile_id = (block_id + rnd * sm_num) * 2 + half
                        if tile_id < total_tiles:
                            bx, by = tile_coords(tile_id)
                            for ko in T.serial(k_blocks):
                                if split_stages:
                                    # Each set sees exactly one tile per round.
                                    gko = rnd * k_blocks + ko
                                    stage = half * num_stages + gko % num_stages
                                    phase = (gko // num_stages) & 1
                                else:
                                    gko = (rnd * 2 + half) * k_blocks + ko
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
                                T.copy(SFA[(by * k_blocks + ko) * block_M:
                                           (by * k_blocks + ko) * block_M + block_M, :],
                                       SFA_shared[stage, :, :])
                                T.copy(SFB[(bx * k_blocks + ko) * block_N:
                                           (bx * k_blocks + ko) * block_N + block_N, :],
                                       SFB_shared[stage, :, :])
                                T.barrier_arrive(loaded[stage])
            elif tx < 128:
                # Consumer 0: even tiles. Starts un-blocked.
                C0 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.annotate_layout({C0: store_layout(emitter, C0, 0)})
                for rnd in T.serial(rounds):
                    tile_id = (block_id + rnd * sm_num) * 2
                    if tile_id < total_tiles:
                        bx, by = tile_coords(tile_id)
                        T.clear(C0)
                        if order:
                            T.barrier_wait(wg_order[0], (rnd - 1) & 1)
                        for ko in T.serial(k_blocks):
                            gko = (rnd * 2) * k_blocks + ko
                            stage = gko % num_stages
                            phase = (gko // num_stages) & 1
                            _ = [_emit(op, ctx_for(bufs, C0, stage, phase))
                                 for op in schedule]
                        if order:
                            T.barrier_arrive(wg_order[1])
                            T.barrier_wait(store_order[0], (rnd - 1) & 1)
                        T.copy(C0, C[by * block_M:(by + 1) * block_M,
                                     bx * block_N:(bx + 1) * block_N])
                        if order:
                            T.barrier_arrive(store_order[1])
            else:
                # Consumer 1: odd tiles. Its MMAs are released by consumer 0's
                # arrive, so its epilogue lands under consumer 0's next MMAs.
                C1 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.annotate_layout({C1: store_layout(emitter, C1, 128)})
                for rnd in T.serial(rounds):
                    tile_id = (block_id + rnd * sm_num) * 2 + 1
                    if tile_id < total_tiles:
                        bx, by = tile_coords(tile_id)
                        T.clear(C1)
                        if order:
                            T.barrier_wait(wg_order[1], rnd & 1)
                        for ko in T.serial(k_blocks):
                            if split_stages:
                                gko = rnd * k_blocks + ko
                                stage = 1 * num_stages + gko % num_stages
                                phase = (gko // num_stages) & 1
                            else:
                                gko = (rnd * 2 + 1) * k_blocks + ko
                                stage = gko % num_stages
                                phase = (gko // num_stages) & 1
                            _ = [_emit(op, ctx_for(bufs, C1, stage, phase))
                                 for op in schedule]
                        if order:
                            T.barrier_arrive(wg_order[0])
                            T.barrier_wait(store_order[1], rnd & 1)
                        T.copy(C1, C[by * block_M:(by + 1) * block_M,
                                     bx * block_N:(bx + 1) * block_N])
                        if order:
                            T.barrier_arrive(store_order[0])

    return main


def main():
    mod = sched_gemm._load_example()

    print("=== correctness vs the template ===")
    for M, N, K in [(1024, 1024, 1024), (2048, 2048, 2048)]:
        A, B, SFA, SFB = sched_gemm._make_inputs(mod, M, N, K, 256)
        ref = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        mod.sm120_nvfp4_blockscaled_gemm(M, N, K)(A, B, SFA, SFB, ref)
        for order in (True,):
            out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
            wg_pingpong_gemm(M, N, K, order=order)(A, B, SFA, SFB, out)
            torch.cuda.synchronize()
            same = torch.equal(out.view(torch.uint16), ref.view(torch.uint16))
            print(f"  order={order!s:5s} {M}x{N}x{K}: "
                  f"{'BITWISE IDENTICAL' if same else 'MISMATCH'}", flush=True)

    print("=== throughput: high-entropy, 200 warmup, median of 3 ===")
    for S, target in ((4096, 1247.7), (8192, 1251.0), (16384, 1058.7)):
        M = N = K = S
        ins = sched_gemm._make_inputs(mod, M, N, K, 256, high_entropy=True)
        C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        arms = {
            "single consumer (best so far)":
                sched_gemm.rf_ws_gemm(M, N, K, block_K=256, group_m=1 if S < 16384 else 16),
            "wg pingpong, ordered": wg_pingpong_gemm(M, N, K, group_m=1 if S < 16384 else 16),
        }
        for k in arms.values():
            k(*ins, C)
        torch.cuda.synchronize()
        names = list(arms)
        res = {n: [] for n in names}
        for r in range(3):
            for n in (names if r % 2 == 0 else names[::-1]):
                ms = do_bench(lambda k=arms[n]: k(*ins, C), _n_warmup=200, _n_repeat=200)
                res[n].append(2.0 * M * N * K / (ms * 1e-3) / 1e12)
        print(f"--- {S}^3  target {target} ---")
        for n in names:
            v = statistics.median(res[n])
            print(f"  {n:30s} {v:7.1f}  ({100 * (v / target - 1):+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
