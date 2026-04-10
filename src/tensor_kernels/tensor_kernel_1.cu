#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include "kernel_helper.h"

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// output tile: BM * BN = 64 * 64 = 4096 results
// threads per block: 128 (4 warps)
// results per thread: 4096 / 128 = 32
// each warp: 16 rows x 64 cols = 1024 results, 1024/32 threads = 32 per thread
// per warp: 8 MMA tiles along N, each MMA yields 4 results per thread -> 8*4 = 32
constexpr int NUM_THREADS = 128;
constexpr int RESULTS_PER_THREAD = (BM * BN) / NUM_THREADS; // 32
constexpr int MMA_TILES_N = BN / MMA_N; // 8

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__
void mma_m16n8k16_bf16(float (&d)[4],
                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                        uint32_t b0, uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
  );
}

// tensor kernel 1: TMA + mma.sync.aligned m16n8k16 for sm_120.
// A is M x K row-major bf16 (K contiguous).
// B is N x K row-major bf16 (K contiguous, pre-transposed from K x N).
// C is M x N row-major f32 output.
__global__ void __launch_bounds__(NUM_THREADS)
tensor_kernel_1(const __grid_constant__ CUtensorMap tma_desc_a,
                const __grid_constant__ CUtensorMap tma_desc_b,
                float* __restrict__ c,
                int M, int N, int K) {

  extern __shared__ __align__(128) uint8_t smem[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(smem + BM * BK * sizeof(__nv_bfloat16));

  __shared__ __align__(8) uint64_t mbar;

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = threadIdx.x;
  const int warpId = tid / 32;
  const int laneId = tid % 32;

  if (tid == 0) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n"
      :: "r"(mbar_addr), "r"(1)
    );
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  float d[RESULTS_PER_THREAD] = {};

  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
  uint32_t phase = 0;

  for (int k = 0; k < K; k += BK) {
    if (tid == 0) {
      uint32_t expected_bytes = (BM * BK + BN * BK) * sizeof(__nv_bfloat16);
      asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(expected_bytes) : "memory"
      );

      uint32_t sA_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(sA_addr), "l"(&tma_desc_a),
           "r"(k), "r"(by * BM), "r"(mbar_addr)
      );

      uint32_t sB_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(sB_addr), "l"(&tma_desc_b),
           "r"(k), "r"(bx * BN), "r"(mbar_addr)
      );
    }

    asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n"
      "@!P bra LAB_WAIT;\n"
      "}\n"
      :: "r"(mbar_addr), "r"(phase)
    );
    phase ^= 1;

    // fragment mapping for mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32:
    //   groupID = laneId / 4  (0..7)
    //   tid_in_grp = laneId % 4  (0..3)
    //
    // A (.row, m16 x k16): rows use groupID and groupID+8
    //   a[0] = {A[groupID,   tid_in_grp*2],   A[groupID,   tid_in_grp*2+1]}
    //   a[1] = {A[groupID+8, tid_in_grp*2],   A[groupID+8, tid_in_grp*2+1]}
    //   a[2] = {A[groupID,   tid_in_grp*2+8], A[groupID,   tid_in_grp*2+9]}
    //   a[3] = {A[groupID+8, tid_in_grp*2+8], A[groupID+8, tid_in_grp*2+9]}
    //
    // B (.col, k16 x n8): N-index = groupID, K-index = tid_in_grp*2
    //   b[0] = {B_col[tid_in_grp*2,   groupID], B_col[tid_in_grp*2+1,   groupID]}
    //   b[1] = {B_col[tid_in_grp*2+8, groupID], B_col[tid_in_grp*2+9, groupID]}
    //
    // D (m16 x n8): same row pattern as A
    //   d[0] = D[groupID,   tid_in_grp*2],  d[1] = D[groupID,   tid_in_grp*2+1]
    //   d[2] = D[groupID+8, tid_in_grp*2],  d[3] = D[groupID+8, tid_in_grp*2+1]

    const int groupID = laneId / 4;
    const int tid_in_grp = laneId % 4;
    const int a_m0 = warpId * MMA_M + groupID;
    const int a_m1 = a_m0 + 8;
    const int a_k0 = 2 * tid_in_grp;
    const int a_k1 = a_k0 + 8;

    uint32_t a_reg[4];
    a_reg[0] = *reinterpret_cast<const uint32_t*>(&sA[a_m0 * BK + a_k0]);
    a_reg[1] = *reinterpret_cast<const uint32_t*>(&sA[a_m1 * BK + a_k0]);
    a_reg[2] = *reinterpret_cast<const uint32_t*>(&sA[a_m0 * BK + a_k1]);
    a_reg[3] = *reinterpret_cast<const uint32_t*>(&sA[a_m1 * BK + a_k1]);

    #pragma unroll
    for (int n = 0; n < MMA_TILES_N; n++) {
      const int b_n = n * MMA_N + groupID;
      const int b_k0 = 2 * tid_in_grp;
      const int b_k1 = b_k0 + 8;

      uint32_t b_reg[2];
      b_reg[0] = *reinterpret_cast<const uint32_t*>(&sB[b_n * BK + b_k0]);
      b_reg[1] = *reinterpret_cast<const uint32_t*>(&sB[b_n * BK + b_k1]);

      mma_m16n8k16_bf16(
        *reinterpret_cast<float(*)[4]>(&d[n * 4]),
        a_reg[0], a_reg[1], a_reg[2], a_reg[3],
        b_reg[0], b_reg[1]);
    }
  }

  // D fragment store: same row pattern (groupID and groupID+8)
  const int groupID_s = laneId / 4;
  const int row0 = by * BM + warpId * MMA_M + groupID_s;
  const int row1 = row0 + 8;
  const int colBase = (laneId % 4) * 2;

  #pragma unroll
  for (int n = 0; n < MMA_TILES_N; n++) {
    int col = bx * BN + n * MMA_N + colBase;
    c[row0 * N + col]     = d[n * 4 + 0];
    c[row0 * N + col + 1] = d[n * 4 + 1];
    c[row1 * N + col]     = d[n * 4 + 2];
    c[row1 * N + col + 1] = d[n * 4 + 3];
  }
}
