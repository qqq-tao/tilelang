// Which scale byte feeds which accumulator element, for
// mma.kind::mxf8f6f4.scale_vec::1X.m16n8k32 with tid=bid=0 (CUTLASS's choice)?
// A = B = all 1.0 (e4m3 0x38). All scales ue8m0 127 (=2^0) except one lane's
// byte 0, set to 128 (=2^1). Whatever doubles is what that byte scales.
#include <cstdint>
#include <cstdio>

__global__ void probe(float *out, int hot_lane, int hot_side) {
  int lane = threadIdx.x;
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0x38383838u, 0x38383838u};
  uint32_t sfa = 127u, sfb = 127u;                 // ue8m0 1.0
  if (lane == hot_lane) { if (hot_side == 0) sfa = 128u; else sfb = 128u; }
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row."
      "col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "{%14},{%15,%16}, {%17},{%18,%19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = d[i];
}

int main() {
  float *o; cudaMallocManaged(&o, 128 * sizeof(float));
  // m16n8: accumulator element (lane,i) is row = 8*(i/2) + lane/4,
  // col = 2*(lane%4) + i%2  -- the standard m16n8 C layout.
  for (int side = 0; side < 2; ++side) {
    printf("=== hot side %s ===\n", side ? "B" : "A");
    for (int hl : {0, 1, 2, 3, 4, 8}) {
      probe<<<1, 32>>>(o, hl, side); cudaDeviceSynchronize();
      printf("  lane %2d doubles: ", hl);
      int n = 0;
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
          if (o[lane * 4 + i] > 33.0f) {          // baseline is 32.0
            int row = 8 * (i / 2) + lane / 4, col = 2 * (lane % 4) + i % 2;
            if (n < 6) printf("(r%d,c%d) ", row, col);
            ++n;
          }
      printf(" [%d elems]\n", n);
    }
  }
  return 0;
}
