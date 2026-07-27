// Which scale byte feeds which accumulator element, for
// mma.kind::mxf8f6f4.scale_vec::1X.m16n8k32 with tid=bid=0 (CUTLASS's choice)?
// A = B = all 1.0 (e4m3 0x38). All scales ue8m0 127 (=2^0) except one lane's
// byte 0, set to 128 (=2^1). Whatever doubles is what that byte scales.
#include <cstdint>
#include <cstdio>

__global__ void probe(float *out, int hot_lane, int hot_side, int bid_sel) {
  int lane = threadIdx.x;
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0x38383838u, 0x38383838u};
  uint32_t sfa = 127u, sfb = 127u;                 // ue8m0 1.0
  // put the hot ue8m0 byte at position bid_sel, keep the rest at 1.0
  if (lane == hot_lane) {
    uint32_t h = 0x7f7f7f7fu & ~(0xffu << (8 * bid_sel));
    h |= 128u << (8 * bid_sel);
    if (hot_side == 0) sfa = h; else sfb = h;
  } else { if (hot_side == 0) sfa = 0x7f7f7f7fu; else sfb = 0x7f7f7f7fu; }
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  uint16_t z = 0;
  uint16_t ba = uint16_t(hot_side == 0 ? bid_sel : 0);
  uint16_t bb = uint16_t(hot_side == 1 ? bid_sel : 0);
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row."
      "col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "{%14},{%15,%16}, {%17},{%18,%19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
        "r"(sfa), "h"(ba), "h"(z), "r"(sfb), "h"(bb), "h"(z));
  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = d[i];
}

int main() {
  float *o; cudaMallocManaged(&o, 128 * sizeof(float));
  // m16n8: accumulator element (lane,i) is row = 8*(i/2) + lane/4,
  // col = 2*(lane%4) + i%2  -- the standard m16n8 C layout.
  for (int side = 0; side < 2; ++side) {
    printf("=== hot side %s: does byte_id select the byte? ===\n", side ? "B" : "A");
    for (int bid = 0; bid < 4; ++bid) {
      int hl = 0;
      probe<<<1, 32>>>(o, hl, side, bid); cudaDeviceSynchronize();
      printf("  bid %d, lane %d doubles: ", bid, hl);
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
