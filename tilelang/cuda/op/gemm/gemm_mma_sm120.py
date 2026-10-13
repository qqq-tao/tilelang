from __future__ import annotations

from tilelang import language as T
from tilelang.layout.swizzle import make_swizzled_layout
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import (
    TensorCoreIntrinEmitterSM120 as TensorCoreIntrinEmitterBlockScaled,
)
from tilelang.cuda.target import target_is_cuda, target_is_sm120
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment, is_full_region
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target

from .gemm_mma import GemmMMA


GEMM_INST_MMA_BLOCK_SCALED = "cuda.mma.blockscaled"


def _is_explicit_non_sm120_cuda(target: Target) -> bool:
    if not target_is_cuda(target):
        return False
    arch = target.attrs.get("arch", None)
    if arch is None or not str(arch).startswith("sm_"):
        return False
    return not target_is_sm120(target)


class GemmMMASm120BlockScaled(GemmBlockScaledMixin, GemmMMA):
    """SM120 warp-level block-scaled MMA lowering.

    A may be shared or a full fragment; B remains shared. Fragment A and
    fragment scales use row-major packed uint32 scales. Compact K-major
    scales retain the shared/shared ping-pong path, including odd atom grids.
    """

    supported_sf_layouts = ("rowmajor", "blockscaled_chunk_kmajor")

    intrin_emitter_cls = TensorCoreIntrinEmitterBlockScaled

    @staticmethod
    def _validate_target(target: Target) -> None:
        if _is_explicit_non_sm120_cuda(target):
            raise ValueError("T.mma_gemm_blockscaled requires SM120 CUDA target")

    def _sf_b_swizzled(self) -> bool:
        annotations = getattr(self.gemm_node, "annotations", {})
        return bool(int(annotations.get("sf_b_swizzled", 0)))

    def _a_packed_words(self) -> bool:
        annotations = getattr(self.gemm_node, "annotations", {})
        return bool(int(annotations.get("a_packed_words", 0)))

    def _sfa_in_registers(self) -> bool:
        """A-operand scale words given as a register fragment (e.g. computed in-lane) rather than
        loaded from shared memory."""
        region = self.SFARegion
        return region is not None and is_fragment(region.buffer)

    def __post_init__(self) -> None:
        # A uint32 fragment of packed e2m1 words is a storage form of float4_e2m1fn, not a
        # mixed-dtype GEMM: skip the base A/B dtype validation for that case.
        if self._a_packed_words():
            if str(self.B.dtype) != "float4_e2m1fn":
                raise ValueError(f"a_packed_words requires a packed float4_e2m1fn B operand, got {self.B.dtype}")
            return
        super().__post_init__()

    def _validate_operands(self) -> None:
        if self._a_packed_words():
            if not self.is_gemm_rs() or str(self.a_dtype) != "uint32" or str(self.b_dtype) != "float4_e2m1fn":
                raise ValueError("a_packed_words requires uint32 fragment A and shared float4_e2m1fn B")
            if self.trans_A or not is_full_region(self.ARegion) or self.sf_layout != "rowmajor":
                raise ValueError("packed-word A requires a full non-transposed fragment and rowmajor scales")
            if self.sf_a_granularity_k != 16 or self.sf_b_granularity_k != 16:
                raise ValueError("packed-word A requires NVFP4 scale granularity 16")
            for region in (self.SFARegion, self.SFBRegion):
                if str(region.buffer.dtype) != "uint32":
                    raise ValueError("packed-word A requires uint32 scale words")
            if not self.SFBRegion.buffer.scope().startswith("shared"):
                raise ValueError("packed-word A requires shared B scales")
            if self._sfa_in_registers():
                sf = self.SFARegion.buffer
                if (
                    not is_full_region(self.SFARegion)
                    or len(sf.shape) != 2
                    or tuple(map(int, sf.shape)) != (int(self.M), int(self.chunk) // 64)
                ):
                    raise ValueError("packed-word A requires a full A-scale fragment of shape [M, K // 64]")
            if (self._sfa_in_registers() or self._sf_b_swizzled()) and (
                not isinstance(self.sf_k_start, tirx.IntImm) or int(self.sf_k_start) != 0
            ):
                raise ValueError("register A scales and swizzled B scales require k_start=0")
            return
        if self._sf_b_swizzled():
            raise ValueError("sf_b_swizzled requires a_packed_words")
        if not (self.is_gemm_ss() or self.is_gemm_rs()):
            raise ValueError("T.mma_gemm_blockscaled requires shared or fragment A and shared B")
        if self.is_gemm_rs():
            if not is_full_region(self.ARegion):
                raise ValueError("SM120 block-scaled fragment input A must be a full region")
            if self.sf_layout != "rowmajor":
                raise ValueError("SM120 block-scaled fragment A currently requires sf_layout='rowmajor'")

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tirx.Var | None = None):
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M,
            self.N,
            thread_nums,
            target,
            GEMM_INST_MMA_BLOCK_SCALED,
        )
        return self.intrin_emitter_cls(
            a_dtype=T.float4_e2m1fn if self._a_packed_words() else self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=int(self.M // m_warp),
            warp_col_tiles=int(self.N // n_warp),
            chunk=self.chunk,
            thread_var=thread_var,
            is_blockscaled=True,
        )

    def infer_layout(self, target: Target, thread_nums: int):
        self._validate_target(target)
        self._validate_operands()
        if self.is_gemm_rs() and self._a_packed_words():
            mma_emitter = self._make_mma_emitter(target, thread_nums)
            if self._sf_b_swizzled() and mma_emitter.block_col_warps != 1:
                raise ValueError("swizzled B scales require a FullRow warp partition")
            layouts = {
                self.A: mma_emitter.make_mma_load_word_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
            if self._sfa_in_registers():
                n_ksteps = int(mma_emitter.chunk // mma_emitter.micro_size_k)
                layouts[self.SFARegion.buffer] = mma_emitter.make_sfa_register_layout(self.SFARegion.buffer, n_ksteps)
            return layouts
        layouts = super().infer_layout(target, thread_nums)
        emitter = self._make_mma_emitter(target, thread_nums)
        for matrix, region, rows in (("A", self.SFARegion, self.M), ("B", self.SFBRegion, self.N)):
            buffer = region.buffer
            if is_fragment(buffer):
                if self.sf_layout != "rowmajor":
                    raise ValueError("SM120 fragment scales currently require sf_layout='rowmajor'")
                if not is_full_region(region) or len(buffer.shape) != 2 or int(buffer.shape[0]) != int(rows):
                    raise ValueError(f"SM120 fragment SF{matrix} must be a full 2D region with {rows} rows")
                scale_layout = emitter.make_scale_load_layout(buffer, matrix)
                # Preserve both operand requirements when SFA and SFB alias.
                if buffer in layouts and not layouts[buffer].is_equal(scale_layout):
                    raise ValueError(
                        f"SM120 block-scaled GEMM cannot reuse fragment buffer '{buffer.name}' as SF{matrix}: "
                        "its operands require different fragment layouts. "
                        "Use separate SFA and SFB fragment buffers, or shared-memory scales."
                    )
                layouts[buffer] = scale_layout
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._validate_target(target)
        self._validate_operands()

        thread_nums = thread_bounds.extent
        local_thread_var = thread_index - thread_bounds.min
        mma_emitter = self._make_mma_emitter(target, thread_nums, thread_var=local_thread_var)

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert block_K % micro_size_k == 0, f"block_K ({block_K}) must be a multiple of micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"

        sf_layout = self.sf_layout
        if sf_layout not in self.supported_sf_layouts:
            raise ValueError(f"Unsupported SM120 scale layout: {sf_layout}")
        sf_a_granularity_k = self.sf_a_granularity_k
        sf_b_granularity_k = self.sf_b_granularity_k

        if self.is_gemm_rs() and self._a_packed_words():
            if sf_layout != "rowmajor":
                raise ValueError("register-A blockscaled GEMM supports sf_layout='rowmajor' only")
            assert is_full_region(A_region), "Fragment input A must be a full region"
            A_buf = A_region.buffer

            n_ksteps = int(block_K // micro_size_k)
            use_words = mma_emitter.supports_scale_words(int(sf_a_granularity_k), int(sf_b_granularity_k))
            sfb_swizzled = self._sf_b_swizzled()
            sfb_words_len = (2 * warp_cols * n_ksteps) if sfb_swizzled else (warp_cols * n_ksteps)
            sfa_in_registers = self._sfa_in_registers()
            if sfa_in_registers and not use_words:
                raise ValueError("register A scales require one scale word per k64 step")
            SFA_register_buf = self.SFARegion.buffer if sfa_in_registers else None

            @T.prim_func
            def _gemm_rs_blockscaled() -> None:
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                if use_words:
                    # one vector load per atom row fetches the scale words of every k step
                    SFA_words = T.alloc_local((warp_rows * n_ksteps), "uint32")
                    SFB_words = T.alloc_local((sfb_words_len), "uint32")
                    SFB_rep_words = T.alloc_local((warp_cols * n_ksteps), "uint32")
                    mma_emitter.ldscale_words(
                        SFA_words,
                        SFB_words,
                        SFB_rep_words,
                        self.SFARegion,
                        self.SFBRegion,
                        n_ksteps,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=int(sf_a_granularity_k),
                        sf_b_granularity_k=int(sf_b_granularity_k),
                        sfb_swizzled=sfb_swizzled,
                        sfa_in_registers=sfa_in_registers,
                    )
                    for ki in T.unroll(n_ksteps):
                        mma_emitter.ldmatrix_b(B_local, B_region, ki)
                        mma_emitter.mma_packed_words(
                            A_buf,
                            B_local,
                            C_buf,
                            ki,
                            SFA_buf=self.SFARegion,
                            SFB_buf=self.SFBRegion,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                            SFA_words=SFA_words,
                            SFB_words=SFB_words,
                            SFB_rep_words=SFB_rep_words,
                            n_ksteps=n_ksteps,
                            sfb_swizzled=sfb_swizzled,
                            SFA_register_buf=SFA_register_buf,
                        )
                else:
                    for ki in T.serial(0, n_ksteps):
                        mma_emitter.ldmatrix_b(B_local, B_region, ki)
                        mma_emitter.mma_packed_words(
                            A_buf,
                            B_local,
                            C_buf,
                            ki,
                            SFA_buf=self.SFARegion,
                            SFB_buf=self.SFBRegion,
                            k_start=self.sf_k_start,
                            sf_a_granularity_k=int(sf_a_granularity_k),
                            sf_b_granularity_k=int(sf_b_granularity_k),
                        )

            return _Simplify(_gemm_rs_blockscaled, inline_let=True)

        if self.is_gemm_rs():
            A_buf = A_region.buffer

            @T.prim_func
            def _gemm_rs_blockscaled() -> None:
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(block_K // micro_size_k):
                    mma_emitter.ldmatrix_b(B_local, B_region, ki)
                    mma_emitter.mma(
                        A_buf,
                        B_local,
                        C_buf,
                        ki,
                        SFA_buf=self.SFARegion,
                        SFB_buf=self.SFBRegion,
                        k_start=self.sf_k_start,
                        sf_a_granularity_k=sf_a_granularity_k,
                        sf_b_granularity_k=sf_b_granularity_k,
                    )

            return _Simplify(_gemm_rs_blockscaled, inline_let=True)

        if sf_layout == "blockscaled_chunk_kmajor":

            @T.prim_func
            def _gemm_ss_blockscaled_kmajor() -> None:
                if clear_accum:
                    T.clear(C_buf)
                mma_emitter.mma_blockscaled_fulltile(
                    A_region,
                    B_region,
                    C_buf,
                    self.SFARegion,
                    self.SFBRegion,
                    sf_layout=sf_layout,
                )

            return _Simplify(_gemm_ss_blockscaled_kmajor, inline_let=True)

        if int(block_K // micro_size_k) == 4:

            @T.prim_func
            def _gemm_ss_blockscaled_static_kblock() -> None:
                A_local_0 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                A_local_1 = T.alloc_local((warp_rows * local_size_a), a_dtype)
                B_local_0 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                B_local_1 = T.alloc_local((warp_cols * local_size_b), b_dtype)
                SFA_local_0 = T.alloc_local((warp_rows), "uint32")
                SFA_local_1 = T.alloc_local((warp_rows), "uint32")
                SFB_local_0 = T.alloc_local((warp_cols), "uint32")
                SFB_local_1 = T.alloc_local((warp_cols), "uint32")
                SFB_rep_local_0 = T.alloc_local((warp_cols), "uint32")
                SFB_rep_local_1 = T.alloc_local((warp_cols), "uint32")
                if clear_accum:
                    T.clear(C_buf)

                mma_emitter.ldmatrix_a(A_local_0, A_region, 0)
                mma_emitter.ldmatrix_b(B_local_0, B_region, 0)
                mma_emitter.ldscale_fragment(
                    SFA_local_0,
                    SFB_local_0,
                    SFB_rep_local_0,
                    self.SFARegion,
                    self.SFBRegion,
                    ki=0,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    sf_layout=sf_layout,
                )
                mma_emitter.ldmatrix_a(A_local_1, A_region, 1)
                mma_emitter.ldmatrix_b(B_local_1, B_region, 1)
                mma_emitter.ldscale_fragment(
                    SFA_local_1,
                    SFB_local_1,
                    SFB_rep_local_1,
                    self.SFARegion,
                    self.SFBRegion,
                    ki=1,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    sf_layout=sf_layout,
                )
                for i in T.unroll(warp_rows):
                    for j in T.unroll(warp_cols):
                        mma_emitter.mma_full_b_atom_with_scale_fragments(
                            A_local_0,
                            B_local_0,
                            C_buf,
                            SFA_local_0,
                            SFB_local_0,
                            SFB_rep_local_0,
                            i,
                            j,
                        )

                mma_emitter.ldmatrix_a(A_local_0, A_region, 2)
                mma_emitter.ldmatrix_b(B_local_0, B_region, 2)
                mma_emitter.ldscale_fragment(
                    SFA_local_0,
                    SFB_local_0,
                    SFB_rep_local_0,
                    self.SFARegion,
                    self.SFBRegion,
                    ki=2,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    sf_layout=sf_layout,
                )
                for i in T.unroll(warp_rows):
                    for j in T.unroll(warp_cols):
                        mma_emitter.mma_full_b_atom_with_scale_fragments(
                            A_local_1,
                            B_local_1,
                            C_buf,
                            SFA_local_1,
                            SFB_local_1,
                            SFB_rep_local_1,
                            i,
                            j,
                        )

                mma_emitter.ldmatrix_a(A_local_1, A_region, 3)
                mma_emitter.ldmatrix_b(B_local_1, B_region, 3)
                mma_emitter.ldscale_fragment(
                    SFA_local_1,
                    SFB_local_1,
                    SFB_rep_local_1,
                    self.SFARegion,
                    self.SFBRegion,
                    ki=3,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                    sf_layout=sf_layout,
                )
                for i in T.unroll(warp_rows):
                    for j in T.unroll(warp_cols):
                        mma_emitter.mma_full_b_atom_with_scale_fragments(
                            A_local_0,
                            B_local_0,
                            C_buf,
                            SFA_local_0,
                            SFB_local_0,
                            SFB_rep_local_0,
                            i,
                            j,
                        )
                for i in T.unroll(warp_rows):
                    for j in T.unroll(warp_cols):
                        mma_emitter.mma_full_b_atom_with_scale_fragments(
                            A_local_1,
                            B_local_1,
                            C_buf,
                            SFA_local_1,
                            SFB_local_1,
                            SFB_rep_local_1,
                            i,
                            j,
                        )

            return _Simplify(_gemm_ss_blockscaled_static_kblock, inline_let=True)

        @T.prim_func
        def _gemm_ss_blockscaled() -> None:
            A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
            if clear_accum:
                T.clear(C_buf)
            for ki in T.serial(0, (block_K // micro_size_k)):
                mma_emitter.ldmatrix_a(A_local, A_region, ki)
                mma_emitter.ldmatrix_b(B_local, B_region, ki)
                mma_emitter.mma(
                    A_local,
                    B_local,
                    C_buf,
                    ki,
                    SFA_buf=self.SFARegion,
                    SFB_buf=self.SFBRegion,
                    k_start=self.sf_k_start,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_k=sf_b_granularity_k,
                )

        return _Simplify(_gemm_ss_blockscaled, inline_let=True)
