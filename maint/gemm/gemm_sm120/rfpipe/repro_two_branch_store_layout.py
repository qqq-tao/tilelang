"""Two thread branches, two annotated accumulators: the second one's chain is dropped.

Minimal reproduction of what blocks the Mt=256 cooperative kernel in v3_coop.py.

Setup: 256 threads split into two 128-thread branches. Each branch has its own
fragment, annotates it with TensorCoreIntrinEmitter.make_mma_store_layout,
clears it, has an opaque extern write it, and copies it to its own half of the
output.

Expected: two clears and two global stores in the generated CUDA.
Actual:   one of each. The second branch keeps only its extern call -- the
          T.clear and the epilogue are both gone, so that half of the output is
          never written.

What matters:
  - without the layout annotation the same kernel keeps both branches intact,
    so the annotation is the trigger, not the extern and not the two branches
  - moving the declaration and the annotation inside each branch does not help
  - it follows the second *branch*, not the second buffer: swapping which
    fragment is used in which branch moves the failure with the branch
  - no diagnostic is emitted

Why it matters here: the production SM120 kernel has two consumer warpgroups and
does not hit this, because it never annotates -- T.mma_gemm_blockscaled infers
the accumulator layout. A schedule built from call_extern has no layout to infer
from, so it must annotate, and then it cannot have two consumer warpgroups.

Run:
    python maint/gemm/gemm_sm120/rfpipe/repro_two_branch_store_layout.py
"""

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

import tilelang
import tilelang.language as T
from tilelang.intrinsics import TensorCoreIntrinEmitter

_SRC = r"""
extern "C" { __device__ void tl_touch(void *c) { ((float *)c)[0] += 1.0f; } }
"""


def _emitter():
    return TensorCoreIntrinEmitter(
        a_dtype=T.float4_e2m1fn, b_dtype=T.float4_e2m1fn, accum_dtype=T.float32,
        a_transposed=False, b_transposed=True,
        block_row_warps=2, block_col_warps=2,
        warp_row_tiles=64, warp_col_tiles=64, chunk=128,
        is_blockscaled=True, kind="mxf4nvf4", scale_vec_size=4, stype="ue4m3",
    )


def build(annotate):

    @tilelang.jit(out_idx=None)
    def kern(_tag=annotate):

        @T.prim_func
        def main(C: T.Tensor((256, 128), T.float32)):
            with T.Kernel(1, 1, threads=256) as (bx, by):
                c0 = T.alloc_fragment((128, 128), T.float32)
                c1 = T.alloc_fragment((128, 128), T.float32)
                if annotate:
                    em = _emitter()
                    T.annotate_layout({c0: em.make_mma_store_layout(c0),
                                       c1: em.make_mma_store_layout(c1)})
                T.import_source(_SRC)
                tx = T.get_thread_binding()
                if tx < 128:
                    T.clear(c0)
                    T.call_extern("handle", "tl_touch", T.access_ptr(c0, "w"))
                    T.copy(c0, C[0:128, 0:128])
                else:
                    T.clear(c1)
                    T.call_extern("handle", "tl_touch", T.access_ptr(c1, "w"))
                    T.copy(c1, C[128:256, 0:128])

        return main

    return kern().get_kernel_source()


def main():
    for annotate in (False, True):
        src = build(annotate)
        # Count how many statements each accumulator takes part in. A complete
        # branch has the clear, the extern and the store; a dropped one keeps
        # only the extern.
        body = src[src.index("main_kernel("):]
        n0 = sum(l.count("c0") for l in body.splitlines())
        n1 = sum(l.count("c1") for l in body.splitlines())
        stores = sum(1 for l in body.splitlines() if "(C +" in l or "C[" in l)
        label = "with make_mma_store_layout" if annotate else "no layout annotation"
        verdict = "both branches complete" if n0 == n1 else "SECOND BRANCH DROPPED"
        print(f"  {label:28s} c0 refs={n0:2d}  c1 refs={n1:2d}  stores to C={stores}"
              f"  -> {verdict}")


if __name__ == "__main__":
    main()
