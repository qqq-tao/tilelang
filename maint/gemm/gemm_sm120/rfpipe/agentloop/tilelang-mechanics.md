# Writing TileLang for SM120 block-scaled GEMM: mechanics that are not obvious

Read this before writing kernel code in this area. Every entry below was found
by debugging, cost between twenty minutes and two hours, and is invisible from
the API surface. None of it is in TileLang's documentation.

## Where the authority lives

TileLang behaviour: this file, found by debugging, because it is not documented.

**PTX instruction semantics: the PTX ISA document, not this file and not a
probe.** Operand roles, block-scale selector rules, layouts and which variants
exist are all specified. Probe to confirm you read it right, not to discover it.

## Emission: `@T.prim_func` is an AST transform, not a trace

`tilelang/language/eager/ast.py: mutate()` rewrites the function's *source*
(via `inspect.getsourcelines`). Consequences:

| you write | does it emit? |
|---|---|
| `T.call_extern(...)` directly in the prim_func body | yes |
| the same call inside a plain Python helper | **no — silently dropped** |
| list comprehension calling `T.evaluate(...)` | yes |
| list comprehension calling a `@T.macro` | yes |
| a plain Python function that *calls* a `@T.macro` | yes |

So a plain Python function cannot emit, but it can *dispatch*: it may contain
ordinary `if/elif` and call macros, and those macros emit normally. That is how
schedule dispatch is written here.

A Python `for` in a prim_func body is intercepted as a TIR loop and fails with
`Invalid for loop, got tuple`. Compile-time expansion must be a list
comprehension — `ast.ListComp` is not intercepted.

Experiment code must live in a **file**. A function defined in a heredoc has no
source and dies with `OSError: could not get source code`.

**Debugging rule**: when a call does not appear in the generated CUDA, print
`kernel.prim_func` first. That distinguishes "never entered the IR" from
"entered and was eliminated", and they have completely different causes.

## Buffers: `T.access_ptr`, never `.data`

`T.access_ptr(buf, "r"|"w"|"rw")` carries buffer identity and an access mask, so
the pipeline and layout passes can build dependencies. `.data` is a raw pointer
and hides them: with `.data` the double buffering silently disappears — shared
memory allocated at one stage's size instead of `num_stages`, and no stage
offset on the pointers. Pass the base with `access_ptr` and the stage offset as
a separate integer argument.

Shared buffers arrive at extern wrappers as `void *`. Take `void *` and cast
inside the wrapper; taking `const unsigned int *` fails to compile.

## Fragments: an annotated layout names *absolute* thread ids

`Fragment.forward_thread` returns an absolute thread index, and
`make_mma_store_layout` emits 0..127 because it is normally used in a region
starting at thread 0. Annotate a second warpgroup's accumulator with the same
layout and **no thread in [128, 256) matches any element**: every ownership
predicate is false, `LowerTileOp` turns the `T.clear` and the epilogue into
`for i in unroll(128): evaluate(0)`, and that warpgroup's output is never
written — with no diagnostic at any stage. Layout inference does not do this;
the module is symmetric up to and including `LayoutInference` and asymmetric
immediately after `LowerTileOp`.

Rebase it:

```python
base = emitter.make_mma_store_layout(buf)
T.Fragment(buf.shape,
           forward_fn=lambda i, j: (base.map_forward_thread([i, j])[0] + thread_base,
                                    base.map_forward_index([i, j])))
```

`map_forward_thread` returns an Array; index `[0]` or `+ n` concatenates instead
of adding. Repro: `repro_two_branch_store_layout.py`.

The production kernel never hits this because `T.mma_gemm_blockscaled` infers
the layout in place. Only annotated layouts need the rebase.

## Barriers

`T.alloc_barrier([n] * stages)`; `n` is the expected arrival count, so it is the
number of *arriving* threads, not waiting ones. Waiting does not consume an
arrival.

`T.barrier_wait(bar, p)` passes when the parity differs from `p`. A fresh
barrier has parity 0, so `wait(bar, 1)` falls through and `wait(bar, 0)` blocks:
the producer's first wait on `consumed` should use `phase ^ 1`.

**Subscript the barrier inside the macro.** `bars[i]` evaluated in plain Python
yields a `Var`, and `T.barrier_wait` wants a `BufferLoad`
(`mbarrier must be an tirx.BufferLoad or a tirx.Buffer`). Pass `bars` and `slot`
separately and index in the macro body.

Phase counters must span the whole CTA lifetime, not one tile: with a persistent
kernel use `gko = tile_index * k_blocks + ko`. Barriers are per-CTA, so only the
warpgroups inside one CTA have to agree, and they must skip the same iterations.

## Warp specialization: express it, do not let it be inferred

`T.Pipelined` triggers the warp-specialization pass, which places tile ops in a
consumer branch but does not recognise an opaque `call_extern` as the producer
of an accumulator — so it eliminates the epilogue, the `T.clear`, and the `C`
kernel parameter itself. CUTLASS does not infer these roles either; its `load()`
and `mma()` call `producer_acquire`/`consumer_wait` explicitly. Write the split
by hand with `T.get_thread_binding()` and the barrier primitives, and put the
pipeline operations in the schedule alongside the package operations.

## Register quotas at 384 threads

The register file is 65536 per SM, so 384 threads caps at 170 per thread; these
kernels want 218 and ptxas spills the difference. The non-uniform quota is
mandatory, not a tuning knob:

```python
if tx >= 256: T.set_max_nreg(producer_regs, 0)   # 40  -- producer only addresses TMA
else:         T.set_max_nreg(consumer_regs, 1)   # 224
```

Without it a 384-thread kernel measures 30-50% slow and every structural theory
you form about why will be describing spill traffic. At 256 threads the same
knob does nothing: that kernel settles at 218 with no spill and gets one CTA per
SM regardless. **Check spill before reaching for it.** See `gate.py`.

## Fixed shapes

`block_N` is not free: the package primitives assume a 2x2 warp grid of 64x64
tiles, i.e. 128 columns. `block_N=64` compiles, reads out of bounds, and
surfaces as a TMA descriptor init failure with `CUDA_ERROR_ILLEGAL_ADDRESS`.

`block_K` is free in {128, 256} — the swizzle follows the atom rule (see below)
and the schedule follows `pingpong_schedule(block_K // 64)`.

## Swizzle: one rule, and a shift that is easy to miss

TileLang's `MakeGemmABLayout` and CUTLASS's `sm120_rr_smem_selector` pick the
widest atom whose extent divides the row — verified identical across 12 points
of dtype x block_K. The address is one XOR:

```
addr = row * (Cols16 * 16) + ((col16 ^ ((row >> shift) & (Cols16 - 1))) * 16)
Cols16 = block_K / 32   (packed FP4)
shift  = 3 - log2(Cols16)      0 for a 128 B row, 1 for 64 B, 2 for 32 B
```

The shift is the trap. All three atoms are `Swizzle<B,4,3>` — S is always 3 — so
the XOR source is always bits [7, 7+B) of the linear offset while the row field
starts at bit `4 + log2(Cols16)`. Writing `row & (Cols16 - 1)` is correct only
for the 128-byte row, which means it passes every `block_K=256` test and
produces garbage at 128. `sm120_swizzled_byte_offset` in `gemm_sm120.h` has it.

## Running against this worktree

An editable install pins `import tilelang` to another worktree. Every script
here needs the bypass at the top: drop meta-path finders whose class module
contains `_tilelang_editable`, drop other worktrees from `sys.path`, prepend
this tree and its `3rdparty/tvm/python`. Keep `_apache_tvm_ffi_editable`.

Run with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, and set a fresh
`TILELANG_CACHE_DIR` when validating a change to a `.h` template — the JIT cache
key does not include template contents. `ncu` drops an `LD_PRELOAD` passed
through the subprocess environment, so profile a wrapper script instead.
