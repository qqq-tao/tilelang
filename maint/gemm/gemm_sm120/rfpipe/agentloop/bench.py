"""Gate 1: the one measurement protocol. Import it; do not hand-roll timing.

    from agentloop.bench import measure
    measure({"name": kernel, ...}, inputs_fn, M, N, K)

Fixes the four things that drifted in the session this came out of: input
entropy, warmup counted in iterations, arm order rotated between rounds, and
median rather than max.
"""
import statistics

WARMUP_ITERS = 200      # 5090 needs ~100-200 to settle at the 575 W clock
REPEAT_ITERS = 200
ROUNDS = 3              # odd, so the median is a measured value


def measure(arms, inputs, out, rounds=ROUNDS, flops_fn=None):
    """arms: {name: (kernel, args)}. Returns {name: (median, min, max)}.

    Order is rotated every round so thermal drift is not correlated with the
    arm under test -- block ordering has produced sign-flipped conclusions here.
    """
    import torch
    from tilelang.profiler import do_bench

    for k, a in arms.values():
        k(*a, out)
    torch.cuda.synchronize()

    names = list(arms)
    res = {n: [] for n in names}
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            k, a = arms[n]
            ms = do_bench(lambda kk=k, aa=a: kk(*aa, out),
                          _n_warmup=WARMUP_ITERS, _n_repeat=REPEAT_ITERS)
            res[n].append(flops_fn(ms) if flops_fn else ms)
    return {n: (statistics.median(v), min(v), max(v)) for n, v in res.items()}


def tflops_fn(M, N, K):
    return lambda ms: 2.0 * M * N * K / (ms * 1e-3) / 1e12


def report(results, target=None):
    """Median is the number that counts. Max is printed only as a spread check."""
    for n, (med, lo, hi) in sorted(results.items(), key=lambda x: -x[1][0]):
        s = f"  {n:34s} {med:8.1f}   [{lo:.1f}, {hi:.1f}]"
        if target:
            s += f"   {100 * (med / target - 1):+5.1f}% vs {target}"
        print(s, flush=True)


def high_entropy_scales(rows, K, seed):
    """Full-entropy UE4M3: exponents 6..8, uniform mantissa, 24 values.

    The correctness generators emit one or two values, one of them 0x00 -- a
    zero scale that blanks half the blocks. Timing with those understates power
    draw, which on a hard-capped board reads directly as a higher clock.
    """
    import torch
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    b = torch.randint(0x30, 0x48, (rows, K // 16), device="cuda",
                      dtype=torch.int64, generator=g).reshape(rows, -1, 4)
    w = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16) | (b[:, :, 3] << 24)
    return w.to(torch.uint32).contiguous()
