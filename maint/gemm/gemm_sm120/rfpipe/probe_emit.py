"""Probe: which construct lets a python-level (trace-time) loop emit TIR calls?

A) direct inline call in prim_func body
B) plain python helper function
C) list comprehension + T.evaluate
D) @T.macro helper called N times from an unrolled python source
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
import tilelang.language as T

SRC = r"""
extern "C" { __device__ void tl_probe(unsigned int *p, int tag) { p[0] += (unsigned)tag; } }
"""


def _helper_plain(loc):
    T.call_extern("handle", "tl_probe", loc.data, 2)


def _helper_eval(loc, tag):
    T.evaluate(T.call_extern("handle", "tl_probe", loc.data, tag))


@T.macro
def _helper_macro(loc, tag):
    T.call_extern("handle", "tl_probe", loc.data, tag)


@tilelang.jit(out_idx=None)
def probe(N):
    @T.prim_func
    def main(Out: T.Tensor((N,), T.uint32)):
        with T.Kernel(1, threads=32) as _:
            loc = T.alloc_local((4,), T.uint32)
            T.import_source(SRC)
            loc[0] = T.uint32(0)
            # A) direct
            T.call_extern("handle", "tl_probe", loc.data, 1)
            # B) plain helper
            _helper_plain(loc)
            # C) comprehension + evaluate (trace-time python loop)
            [_helper_eval(loc, t) for t in (3, 4)]
            # D) macro called from a comprehension
            [_helper_macro(loc, t) for t in (5, 6)]
            Out[0] = loc[0]

    return main


if __name__ == "__main__":
    k = probe(4)
    src = k.get_kernel_source()
    tags = []
    for ln in src.splitlines():
        if "tl_probe(" in ln and "__device__" not in ln:
            tags.append(ln.strip()[:110])
    print("emitted tl_probe calls:", len(tags))
    for t in tags:
        print("   ", t)
