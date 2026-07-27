# SM120 kernel work loop

A process for tuning SM120 block-scaled GEMM kernels in this repo. Written after
a session that produced six consecutive wrong structural explanations for a
kernel being slow, all of which `ptxas -v` would have refuted in one line, and
that never once ran a profiler. Every rule below exists because its absence cost
that session real hours.

References: the shape of the profiling phase follows
`mit-han-lab/ncu-report-skill`; the candidate-ledger and promotion structure
follows `mit-han-lab/kernel-design-agents`. Both are adapted to SM120 (RTX 5090,
170 SMs, 99 KiB smem/CTA, 575 W hard cap) rather than B200.

## Gate -1 — before writing any kernel code

Read `tilelang-mechanics.md`. TileLang has a set of behaviours that are
invisible from the API and each cost hours to find: a plain Python helper's
`T.*` calls are silently dropped, `.data` loses the dependencies that create
double buffering, an annotated fragment layout names absolute thread ids so a
second warpgroup's output is silently never written, barrier subscripts must be
taken inside a macro, `block_N` is not a free parameter, and the shared-memory
swizzle has a shift term that is correct at `block_K=256` and wrong at 128.

None of these announce themselves. They present as a missing call, a wrong
number, or half an output tensor, and they will be re-derived from scratch by
anyone who does not read that file first.

## Gate -0.5 — for instruction semantics, read the PTX ISA before measuring

Anything about what a PTX instruction *means* -- operand roles, selector rules,
layouts, which combinations exist -- is specified in the PTX ISA document, under
"Warp Level Matrix Multiply-Accumulate Instructions" and its block-scaling
subsection. Read that first.

Measuring is a legitimate fallback and the probes in `mxf8f6f4/` are good ones,
but a probe only tells you about the cases you probed. The spec tells you the
rule, its edge cases, and the variants you did not think to try -- for
`.block_scale` that is `scale_vec::1X` versus `2X` versus `4X`, which of A and B
each selector applies to, and what happens for lanes the mapping does not name.
The measurement that produced the mxf8f6f4 selector mapping in this repo took
two probes and answered exactly two questions; the spec section answers those
and about six more.

Order: spec, then a probe to confirm you read it correctly, then build. Not
probe-only. Confirming a documented rule takes one run; discovering it takes
several, and leaves you unsure whether what you found generalises.

## Gate 0 — before any hypothesis about why something is slow

Run `python agentloop/gate.py <script> <kernel-regex>`. It reports, in order:

1. **registers per thread and spill bytes** (`ptxas -v`)
2. **shared memory per CTA** and what it implies for CTAs/SM
3. **achieved occupancy** and which of {registers, shared memory, warps,
   barriers} is the binding limit
4. **the stall breakdown**, ranked

**You may not propose a structural explanation until this has run.** If spill
bytes are non-zero, or achieved occupancy is under ~30%, stop: that is the
answer, and any structural theory is describing its symptoms. In the session
that motivated this file, the answer was `218 registers/thread against a
65536/384 = 170 ceiling` and it was invisible to fifteen rounds of black-box A/B.

## Gate 1 — one measurement protocol, not a habit

All throughput numbers come from `agentloop/bench.py`. It fixes what drifted:

- **high-entropy inputs**: full-byte-domain FP4 *and* `random_ue4m3` scales.
  The scale generators used for correctness emit one or two values, one of them
  a zero scale; timing with them reads several percent high on a power-capped
  board.
- **200 warmup iterations**, counted in iterations rather than milliseconds.
  5090 needs ~100-200 to settle at the 575 W clock; a 25 ms warmup is ~30
  iterations at 8192³ and reads 6% high.
- **order rotated between arms each round, median of >= 3**. Block ordering
  correlates thermal drift with whatever you are testing, which has produced
  sign-flipped conclusions here and in the CUTLASS study.
- **median, never max**. Reporting max-of-2 put an 8192³ number 1.4% closer to
  target than the median did.

## Gate 2 — task contract, before writing code

One line each, into `candidates.jsonl` as a `contract` record:

- **what changes**, in one sentence
- **which acceptance number it targets**, and the current value
- **what the profiler says should improve**, quantitatively — a change with no
  predicted metric movement is a guess
- **promotion criterion**: the number that decides keep-or-revert, decided now
- **certain or uncertain**: does this add capability that is known to work
  (a new dtype path, a new tile), or does it chase a number that may not
  converge? Do the certain ones first. The motivating session spent a night on
  uncertain optimisation and never started the certain increment.

## Loop

```
gate.py  ->  contract  ->  implement  ->  correctness  ->  bench.py  ->  record
   ^                                            |                          |
   |                                     fails: revert                     |
   +---------------- profile again if the prediction missed ---------------+
```

Correctness before throughput, always: every kernel here has a bitwise reference
in `sched_gemm.py`. A throughput number from a kernel that has not been checked
bitwise is worthless -- one in the motivating session read 1604.9 TFLOPS while
skipping half the epilogue.

If the measured change does not match the prediction from the contract, do not
try a variation. Profile again. The prediction missing means the model is wrong,
and the next variation will be built on the same wrong model.

## Recording

`candidates.jsonl`, one JSON object per line, appended never rewritten. Types:
`contract`, `result`, `profile`, `revert`. This is the state that survives a
context reset; the transcript does not. Anything that only exists in the
conversation is lost work.

Commit messages carry the reasoning, including negative results and the
hypotheses that were eliminated. A negative result that is not written down gets
re-tried.
