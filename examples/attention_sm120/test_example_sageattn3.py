import pytest

import tilelang

import tilelang.testing

torch = pytest.importorskip("torch")

from examples.attention_sm120 import sageattn3_quant as sq  # noqa: E402

from examples.attention_sm120.sm120_sageattn3_fwd_ws import prepare_inputs  # noqa: E402

from examples.attention_sm120 import sm120_sageattn3_fwd_ws as ws  # noqa: E402


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_ws_matches_original_kernel():
    """Alignment to the original kernel on the same inputs, at a batch with more tiles than SMs."""
    fp4attn_cuda = pytest.importorskip("fp4attn_cuda")
    torch.manual_seed(0)
    b, h, n, d = 4, 32, 1024, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    canon, kargs, delta_s = prepare_inputs(q, k, v)
    out = ws.sm120_sageattn3_fwd(b, h, n, d)(*kargs).float()
    s = sq.export_for_sage(canon)
    o_sage, _ = fp4attn_cuda.fwd(s["q"], s["k"], s["v"], s["sfq"], s["sfk"], s["sfv"], delta_s, n, None, d**-0.5, False, True, True)
    o_ref = sq.reference_attention(q, k, v, d**-0.5)
    ratio = sq.alignment_ratio(out, o_sage.float(), o_ref)
    assert ratio < 0.25, ratio
    assert abs(sq.cos_sim(out, o_ref) - sq.cos_sim(o_sage.float(), o_ref)) < 1e-3


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sageattn3_ws_codegen_contract():
    """The producer/consumer shape actually materialized (no silent fallback to the plain form)."""
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 256, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    _, kargs, _ = prepare_inputs(q, k, v)
    src = ws.sm120_sageattn3_fwd(b, h, n, d).get_kernel_source()
    assert "tl::tma_load(" in src  # Q/K/V/delta_s ride TMA, which only the producer path selects
    assert "cp_async_gs" not in src  # ... so nothing is left on the cp.async path
    assert "tl::warpgroup_reg_dealloc<" in src and "tl::warpgroup_reg_alloc<" in src
    assert "__launch_bounds__(384, 1)" in src  # the second argument is what makes setmaxnreg bind
    assert "SM120MmaBlockScaledKind::kMxf4nvf4, 4, tl::SM120MmaScaleType::kUE4M3" in src
    assert src.count("tl::tma_load(") == 4  # Q, K, V^T, delta_s
    assert "tl_cvt_e2m1x8_rn(" in src  # four key pairs per packed P word, one asm sequence
    assert "tl_sts_ue4m3(" in src and "tl_syncwarp()" in src  # warp-private P-scale store, no block barrier
    # Q and its scale words enter registers once per q tile and feed the QK MMAs from there
    assert "reinterpret_cast<const uint32_t*>(Qw" in src and "(&(SFQw[" in src
    assert ws.DEFAULT_PASS_CONFIGS[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] == 8


def test_sageattn3_ws_register_split_leaves_slack():
    """setmaxnreg.inc blocks forever if the two roles ask for the whole register file.

    A CPU-only guard on the defaults: 32/240 is exactly 65536 and hangs the kernel, so the
    product must stay strictly under it. Regression for a deadlock that costs hours to find.
    """
    import inspect

    sig = inspect.signature(ws.build_sm120_sageattn3_fwd)
    threads = sig.parameters["threads"].default
    producer_regs = sig.parameters["producer_regs"].default
    consumer_regs = sig.parameters["consumer_regs"].default
    consumer_threads = 256
    producer_threads = threads - consumer_threads
    total = producer_threads * producer_regs + consumer_threads * consumer_regs
    assert total <= 65536 - 1024, (total, producer_regs, consumer_regs)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("n", [128, 256, 512, 1024, 4096])
def test_sageattn3_ws_matches_golden(n):
    # Short sequences cover pipeline fill/drain; longer ones exercise reuse.
    torch.manual_seed(0)
    b, h, d = 1, 2, 128
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    canon, kargs, delta_s = prepare_inputs(q, k, v)
    kernel = ws.sm120_sageattn3_fwd(b, h, n, d)
    out = kernel(*kargs).float()
    for _ in range(3):
        assert torch.equal(out, kernel(*kargs).float())
    ref = sq.reference_attention(q, k, v, d**-0.5)
    golden = sq.golden_attention(canon, delta_s, d**-0.5)
    assert sq.alignment_ratio(out, golden, ref) < 0.02
    assert sq.cos_sim(out, ref) > 0.975
