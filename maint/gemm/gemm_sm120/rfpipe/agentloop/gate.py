"""Gate 0: resource limits and stall breakdown, before any structural theory.

    python agentloop/gate.py <harness.py> [kernel-regex]

Prints registers per thread, spill bytes, shared memory, the binding occupancy
limit, and the ranked stall reasons. Run this before proposing why a kernel is
slow. If spill bytes are non-zero or achieved occupancy is under ~30%, that is
the answer and no further theory is needed.
"""
import csv
import io
import os
import subprocess
import sys
import tempfile

PY = "/data/public/envs/miniconda3/envs/test_tilelang/bin/python"
# tilelang resolves its native libs relative to the working directory, so the
# profiled process has to start at the repo root or it picks up another
# worktree's build and dies on a GLIBCXX mismatch.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0",
           LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

# Occupancy limiters, in the order ncu reports them.
_LIMITS = [("Block Limit Registers", "registers"),
           ("Block Limit Shared Mem", "shared memory"),
           ("Block Limit Warps", "warps"),
           ("Block Limit Barriers", "barriers")]


def _ncu(harness, regex, outdir):
    rep = os.path.join(outdir, "gate")
    # Profile a wrapper rather than python directly: ncu does not carry an
    # LD_PRELOAD passed through the subprocess environment, and this env needs
    # the system libstdc++ to load tilelang's native runtime at all.
    wrap = os.path.join(outdir, "run.sh")
    with open(wrap, "w") as f:
        f.write("#!/bin/sh\n"
                "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
                f'exec "{PY}" "{os.path.abspath(harness)}"\n')
    os.chmod(wrap, 0o755)
    cmd = ["ncu", "--set", "full", "-k", f"regex:{regex}", "-c", "1",
           "--replay-mode", "kernel", "-o", rep, "--force-overwrite", wrap]
    r = subprocess.run(cmd, cwd=REPO_ROOT,
                       env=dict(ENV, TILELANG_CACHE_DIR=os.path.join(outdir, "cache")),
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
        raise SystemExit(f"ncu failed ({r.returncode}). Harness must run standalone "
                         f"and launch the kernel at least once.\n{tail}")
    return rep + ".ncu-rep"


def _rows(rep):
    out = subprocess.run(["ncu", "--import", rep, "--csv", "--page", "raw"],
                         capture_output=True, text=True).stdout
    r = list(csv.reader(io.StringIO(out)))
    hdr = r[0]
    row = next(x for x in r[2:] if len(x) == len(hdr))   # r[1] is units
    return dict(zip(hdr, row))


def _details(rep):
    return subprocess.run(["ncu", "--import", rep, "--page", "details"],
                          capture_output=True, text=True).stdout


def main():
    harness = sys.argv[1]
    regex = sys.argv[2] if len(sys.argv) > 2 else "main_kernel"
    with tempfile.TemporaryDirectory() as td:
        rep = _ncu(harness, regex, td)
        raw, det = _rows(rep), _details(rep)

    def num(key, default=None):
        v = raw.get(key, "")
        try:
            return float(v)
        except ValueError:
            return default

    def line(label):
        for l in det.splitlines():
            if label in l:
                return l.split()[-1].replace(",", "")
        return "?"

    regs = line("Registers Per Thread")
    spill_store = num("launch__local_mem_per_thread_spill_stores", 0) or 0
    spill_load = num("launch__local_mem_per_thread_spill_loads", 0) or 0
    smem = line("Static Shared Memory Per Block") if "Static Shared" in det else "?"
    dyn = num("launch__shared_mem_per_block_dynamic", 0) or 0   # KiB
    threads = num("launch__block_size", 0) or 0
    occ = line("Achieved Occupancy")

    print("== Gate 0 ==")
    print(f"  registers/thread   {regs}")
    print(f"  spill              {spill_store:.0f} store / {spill_load:.0f} load bytes per thread"
          f"{'   <== FIX THIS FIRST' if spill_store or spill_load else ''}")
    print(f"  threads/CTA        {threads:.0f}")
    print(f"  dynamic smem/CTA   {dyn:.0f} KiB")
    print(f"  achieved occupancy {occ} %")
    binding = [n for k, n in _LIMITS if line(k) == "1"]
    print(f"  binding limit      {', '.join(binding) if binding else 'none at 1 CTA'}")

    sel = []
    for k, v in raw.items():
        if "average_warps_issue_stalled" in k and "per_issue_active" in k:
            try:
                sel.append((k.split("stalled_")[1].split("_per_issue")[0], float(v)))
            except ValueError:
                pass
    tot = sum(v for _, v in sel) or 1.0
    print("  stalls (cycles per issue):")
    for n, v in sorted(sel, key=lambda x: -x[1])[:6]:
        print(f"    {n:22s} {v:6.2f}  {100 * v / tot:5.1f}%")

    try:
        o = float(occ)
        if o < 30:
            print(f"\n  Occupancy is {o:.1f}%. Latency cannot be hidden at this warp count;")
            print("  raising it dominates any structural change. Do not theorise further.")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
