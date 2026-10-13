# SageAttention3 on SM120

Standalone TileLang port extracted from qqq-tao's #3081, on current main.
The implementation follows thu-ml/SageAttention's SageAttention3 NVFP4 algorithm.
It does not depend on the MXFP4/MXFP8 additions in #3495.

## Scope

- Forward non-causal self-attention, BF16 Q/K/V and BF16 output.
- Equal Q/K/V shapes `[B, H, N, 128]`, with positive N divisible by 128.
- Persistent producer/consumer kernel, NVFP4 QK/PV, TMA staging, and
  register-resident Q/scale words.
- TileLang mean/quantization preprocessing; PyTorch matmul for the correction term.
- CUDA code generation on SM120. CuTeDSL, backward, causal masking,
  arbitrary head dimensions and sequence padding are not implemented.

```python
from examples.attention_sm120.sageattn3_e2e import sageattn3_tl

out = sageattn3_tl(q, k, v)
```

## Validation

```bash
pytest testing/python/quantize/test_tilelang_quantize_sageattn3.py \
       testing/python/language/test_tilelang_language_nvf4_attention_primitives.py \
       examples/attention_sm120/test_example_sageattn3.py \
       examples/attention_sm120/test_example_sageattn3_e2e.py -q
```

Original-kernel comparisons require the `sageattention3_blackwell` extension from
https://github.com/thu-ml/SageAttention. They skip when it is not installed.
The Q block mean uses FP32 reduction rather than the original BF16 reduction;
outputs are compared numerically, not claimed bit-identical to the original.

## Performance measurements

`bench_sageattn3.py` separates pre-quantized kernel-only (`ws,orig`) and BF16
end-to-end (`tl_e2e,orig_e2e`) comparisons. Historical measurements from the old
branch are not results for this port. Record revision, import path, GPU state,
warmup, timing backend and raw output when comparing builds.
