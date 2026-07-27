"""One launch of the 4096^3 single-consumer kernel, for ncu."""
import importlib.util
from pathlib import Path
D = Path("/data/home/qutao/refactor/tilelang-nvf4-pr2364-current/maint/gemm/gemm_sm120/rfpipe")
spec = importlib.util.spec_from_file_location("sg", str(D / "sched_gemm.py"))
sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
import torch
mod = sg._load_example()
M = N = K = 4096
ins = sg._make_inputs(mod, M, N, K, 256, high_entropy=True)
C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
k = sg.rf_ws_gemm(M, N, K, block_K=256, group_m=1)
for _ in range(3):          # warm the JIT and the clocks a little
    k(*ins, C)
torch.cuda.synchronize()
k(*ins, C)                  # the launch ncu captures
torch.cuda.synchronize()
