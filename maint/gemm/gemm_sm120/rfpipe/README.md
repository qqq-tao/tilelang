# RF pipeline: toward a compiler-analyzable path for mxf8f6f4 / mxf4nvf4

Working area for the refactor branch `refactor/mx-blockscale-compiler-path`.

## Goal

Support the SM120 block-scaled MMA families (`mxf4nvf4`, `mxf8f6f4`) through a
path the compiler can *analyze and schedule*, instead of a hand-written fixed
pipeline template.

Today `src/tl_templates/cuda/gemm_sm120.h` ships
`sm120_mma_blockscaled_kblock_fulltile_package_pingpong`: ~300 lines of macro
expansion that hard-code one register-level double-buffered issue order for one
tile shape (128x128x256, NVFP4 only). It reaches vendor parity (8192^3 within
1% of an isomorphic CUTLASS config), but it was picked by hand-searching **16
pipeline variants** — and that search is exactly the job a compiler should own.

Target end state: the SM120 NVFP4 kernel becomes a ~50-line schedule
description; a new tile shape, a new dtype family, or a new issue order is a
schedule parameter rather than a new template. MXFP8 support then costs an
operand-type entry, not a second monolith.


### What "done" means, and how it is checked

All figures below are the **sustained, high-entropy** protocol: full-byte-domain
FP4 operands, `--scale-mode random_ue4m3`, 200 warmup iterations, order rotated
across rounds, median of three. Numbers taken any other way are not comparable;
a short warmup alone reads ~6% high on this board, and low-entropy scales
another ~4%.

**A1 -- expressible.** `block_K` in {128, 256} crossed with `Mt` in {128, 256},
all bitwise identical to the shipped template, from one schedule generator with
no per-configuration offset table.

**A2 -- matches the vendor where the vendor is strongest.** CUTLASS C++ 79a,
same protocol, `256x128x128` cooperative: **1251 TFLOPS at 8192^3** and **1058.7
at 16384^3**. Those are the targets for Mt=256.

**A3 -- fixes our weak shape.** At 4096^3 we currently reach 1069.9 against
CUTLASS's 1247.7 on the same tile, **-14.3%**. This is the one shape where we
are clearly behind and it has never been investigated.

**A4 -- upstreamable.** No new hand-written address tables; the swizzle comes
from the 4-line atom rule, the accumulator layout from inference, and the
schedule from a generator.

**A5 -- both families.** `mxf8f6f4` as well as `mxf4nvf4`. Only NVFP4 has been
touched so far, and MXFP8 is not a free ride: its shared-memory elements are 1.0
bytes rather than 0.5 (fp4 and fp6 are unpacked on that path), which halves the
stage budget, and CUTLASS applies an `fp4_shift` after the fragment load that we
have no equivalent for.


### Status against the acceptance criteria (2026-07-27)

Protocol as stated above. `pers gm=N` is persistent with N-tile-row bands.

| | ours | config | target | |
|---|---|---|---|---|
| 4096³ | 1062.5 | Mt=128 bK=256 pers gm=1 | 1247.7 | **-14.8%** |
| 8192³ | 1224.0 | Mt=128 bK=256 pers gm=1 | 1251.0 | -2.2% |
| 16384³ | **1149.0** | Mt=128 bK=256 pers gm=16 | 1058.7 | **+8.5%** |

- **A1 met.** `block_K` in {128, 256} crossed with `Mt` in {128, 256}, all bitwise
  identical to the template, from one generator, no per-configuration table.
- **A2 met at 16384³, 2.2% short at 8192³.**
- **A3 open.** 4096³ does not move: 1062.5 / 984.9 / 980.8 / 984.8 / 1053.2 across
  tile, stage-count and band-width combinations. The production kernel measures
  1069.9 there under the same protocol, so this is inherited, not introduced, and
  it needs profiling rather than tuning. Note the shape of it: CUTLASS is
  *faster* at 4096³ (1247.7) than at 8192³ (1191.8) while we are slower
  (1062.5 vs 1224.0), so whatever it is scales with tile count or with k-loop
  length, not with the working set.
- **A4 met for addressing** (swizzle from the 4-line atom rule, no hand tables);
  the accumulator layout is annotated rather than inferred, which works but
  needs the thread rebase below.
- **A5 not started.** `mxf8f6f4`.

### Constraints found the hard way

- **`block_N` is not a free parameter.** The package primitives are built for a
  2x2 warp grid of 64x64 tiles, i.e. 128 columns. `block_N=64` compiles and then
  reads out of bounds -- the TMA descriptor fails with an illegal address.
- **An annotated accumulator layout must be rebased onto its warpgroup's
  threads.** `Fragment.forward_thread` returns an absolute thread id and
  `make_mma_store_layout` emits 0..127, so a second warpgroup annotated with it
  matches no thread, and `LowerTileOp` silently turns that warpgroup's `T.clear`
  and epilogue into empty loops. See `repro_two_branch_store_layout.py`.
- **Rasterisation order is worth more than persistence.** At 16384³, same tile
  and stages: 881.2 row-major, 1118.5 grouped, 1149.0 persistent with 16-row
  bands.

### Order of work, revised by what the prototype found

1. **A3: why 4096³ does not move.** The one gap that tuning did not touch, and
   the only one where the vendor's shape curve and ours disagree in direction.
   Needs a profiler, not another sweep.
2. **`mxf8f6f4`.** Untouched, and the stage budget halves on that path.
3. **Fold the package operations into the emitter.** Demoted from prerequisite
   back to cleanup: annotation can express the second warpgroup after all, once
   the layout is rebased. Still the right end state for upstreaming, and still
   needs the interface agreed with the maintainer first.
4. **Shape routing** as a kernel-selection policy rather than a benchmark
   parameter. Mt=128 with block_K=256 wins every shape we measure once the
   rasterisation is grouped, so this is currently cheap.

### Not goals, on evidence

- **Searching the emission order.** The one reordering the vocabulary made
  newly expressible -- releasing the pipeline stage after the last shared-memory
  read instead of after the last MMA -- measured **-0.45% / +0.03%** over five
  interleaved rounds. The CUTLASS study puts schedule choice at +3~11% against
  +31~41% for pipeline depth. The value of schedule-as-data is that it makes
  K=128 and Mt=256 *expressible* (K=128 block-scaled is otherwise a hard
  `ValueError`), not that a better permutation is waiting to be found.
- **Beating the vendor at 128x128x256.** Already parity: schedule-as-data is
  within 1.3% of the hand-written template in an identical skeleton, and the
  kernel is +3.3% on CUTLASS's same tile at 8192^3. Not the bottleneck.

### Known risks

- Folding into the emitter is an intrusive upstream change; the interface should
  be agreed with the maintainer before it is built, not after.
- Staging scale factors by TMA the way CUTLASS does needs a layout whose domain
  is (M, K) and whose codomain is the blocked array. If TileLang cannot express
  that, the producer keeps its 128-lane strided copy and the gap stays.
- `mxf8f6f4` has never been compiled on this path. Halving the stage budget puts
  it near the 1->2 stage cliff, which is the steepest part of the curve.

### Why Mt=256 forces this (the concrete driver)

```
Mt=256 measured to win on both 5090 and RTX Pro 6000 (8192^3 +8%, 16384^3 +48%)
  -> Mt=256 with K=256 double buffering needs ~108KB smem > 99KB      infeasible
  -> so Mt=256 implies K=128 (27KB/stage x 3 = 81KB, same as CUTLASS m256_st3)
  -> K=128 means 2 kblocks, which today routes to the generic serial path (-17%)
  -> package-quality 2-kblock code requires composing the L2 primitives
  => the schedule abstraction is a precondition for Mt=256, not a nice-to-have
```

## What the template already is (layer analysis)

`gemm_sm120.h` is not actually monolithic — it is four layers, and only the top
one is hard-coded:

| Layer | Contents | Location |
|---|---|---|
| L0 ISA | `sm120_mma_m16n8k64_mxf4nvf4_4x_ue4m3_regs`, `sm120_ldmatrix_x4_*` | :127, :101 |
| L1 addressing (pure fns) | `sm120_blockscaled_chunk_kmajor_sf_word`, `sm120_fulltile_k_swizzle_offset`, `sm120_fulltile_package_a/b_offset` | :47, :158 |
| L2 package ops | `sm120_copy_fulltile_ab_owner_wide_package`, `detail::sm120_copy_scale_tv_package`, `sm120_gemm_fulltile_ab_owner_wide_package` | :227, :83, :260 |
| L3 schedule | **a fixed sequence of 12 L2 calls** (4 cp + 4 sf + 4 mma, two packages alternating) | :304 |

L3 carries no more information than a Python list:

```python
PINGPONG_4 = [
    ("cp", 0, 0), ("sf", 0, 0), ("cp", 1, 1), ("sf", 1, 1),
    ("mma", 0), ("cp", 0, 2), ("sf", 0, 2),
    ("mma", 1), ("cp", 1, 3), ("sf", 1, 3),
    ("mma", 0), ("mma", 1),
]
```

The 16-variant search space is the permutation space of that list.

Composability facts already verified in the current code:

- L2 package helpers index with `threadIdx.x & 127`, so any 128-thread warp
  group (including `tx` in `[128, 256)`) can call them — cooperative consumers
  come for free.
- Region slicing already reaches the package path (the maint WS benchmark's
  panel mode calls the MMA with `B_shared[stage, 0:64, :]`).
- The lowering normalizes thread bounds (`local_thread_var = tx - bounds.min`),
  which is what makes the existing pingpong consumer #1 work.

## Emission mechanism (read this before writing prototypes)

`@T.prim_func` goes through the eager builder, which is a **source-level AST
transform** (`tilelang/language/eager/ast.py: mutate()` ->
`inspect.getsourcelines()`), not a dynamic trace. Consequences, measured with
`probe_emit.py`:

| Construct | Emits into TIR? |
|---|---|
| `T.call_extern(...)` directly in the prim_func body | yes |
| the same call inside a plain Python helper | **no — silently dropped** |
| list comprehension + `T.evaluate(T.call_extern(...))` | yes |
| list comprehension + `@T.macro` helper | yes |

The transform intercepts `ast.For` (a Python loop in the body raises
`Invalid for loop, got tuple`) but not comprehensions, which run as real Python
at trace time. So **compile-time expansion = list comprehension driving
`@T.macro` helpers**. That is the mechanism the schedule generator is built on.

Debugging note: if generated code is missing your calls, check
`kernel.prim_func` first — that distinguishes "never emitted" from "emitted then
optimized away". A dropped emission also drags its consumers out with it (the
output tensor can disappear from the kernel signature entirely).

## Contents

- `probe_emit.py` — the emission-mechanism probe (four constructs, prints which
  ones survive). Run it after any TileLang frontend bump.
- `v1_sched_repro.py` — V1 prototype: reproduce the production pingpong from the
  schedule list, via injected C wrappers around the L2 primitives
  (`T.import_source` + `call_extern`), zero changes to shipped code.

## Roadmap

1. **V1** — schedule-as-data reproduction of the current pingpong.
   Acceptance: bitwise-identical output vs the example kernel, ~1284 TFLOPS at
   2048^3 (the current template's number). Proves 300 lines can be generated
   from ~50.
2. **V2** — same generator with `kblocks=2` (K=128), schedule
   `[cp0, sf0, cp1, sf1, mma0, mma1]`. Measures what this recovers versus the
   generic serial path (1069 TFLOPS).
3. **V3** — cooperative Mt=256: 384 threads (256 consumer + 128 producer), each
   consumer WG owning a 128-row half of a 256-row tile, K=128 x 3 stages,
   persistent single-tile stream. Target 1431 TFLOPS (CUTLASS m256_st3 on 5090).
4. **Upstreaming** — fold the generator into the emitter
   (`emit_package_schedule(schedule)`), keep L2 in the header, retire L3 to a
   default schedule constant. At that point the dedicated TIR op and its codegen
   special case can be deleted.
5. **Abstraction** — derive the L1 swizzle inverse from the layout object rather
   than a hand-written case table; expose the schedule through a lowering
   annotation; eventually an IR-level RF multi-buffer (a fragment-level
   `T.Pipelined`) with the permutation space handed to a search.

## Known blocker for V2/V3

L1 addressing hard-codes `block_K=256`: `sm120_fulltile_package_a/b_offset`
assume a 128-byte row pitch (`(tx & 15) * 128`, `row * 2048`, warp-m `8192`),
and `sm120_fulltile_k_swizzle_offset` is the inverse of the `T.copy` swizzle for
a (128, 256) FP4 tile, with a case table for 4 slices only. K=128 has a 64-byte
pitch and a different (2-slice) swizzle pattern, so both need to be
K-parameterized and the 64-byte-row swizzle inverse has to be re-derived —
which is precisely the argument for generating it from the layout object.
