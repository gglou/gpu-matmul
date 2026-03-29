#ifndef PIPELINING_KERNEL_H
#define PIPELINING_KERNEL_H

#include <cuda/pipeline>

template <int BM, int BN, int BK, int WM, int WN>
__device__ void load_tile(float *a, float *b, float (&As_stage)[BK][BM + 4],
                          float (&Bs_stage)[BK][BN], int kOffset, int by,
                          int bx, int K, int N,
                          cuda::pipeline<cuda::thread_scope_thread> &pipe) {
  constexpr int numThreads = ((BM / WM) * (BN / WN)) * 32;
  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;

  const int innerRowA = linearThreadId / (BK / 4);
  const int innerColA = linearThreadId % (BK / 4);
  constexpr int rowStrideA = (numThreads * 4) / BK;
  const int innerRowB = linearThreadId / (BN / 4);
  const int innerColB = linearThreadId % (BN / 4);
  constexpr int rowStrideB = numThreads / (BN / 4);

  // Load A tile, transposing on-the-fly into As[k][m] via registers.
  for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 val = *reinterpret_cast<const float4 *>(
        &a[(BM * by + innerRowA + offset) * K + kOffset + innerColA * 4]);
    As_stage[innerColA * 4 + 0][innerRowA + offset] = val.x;
    As_stage[innerColA * 4 + 1][innerRowA + offset] = val.y;
    As_stage[innerColA * 4 + 2][innerRowA + offset] = val.z;
    As_stage[innerColA * 4 + 3][innerRowA + offset] = val.w;
  }

  // Load B tile with 16-byte async copies.
  for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cuda::memcpy_async(
        &Bs_stage[innerRowB + offset][innerColB * 4],
        &b[(kOffset + innerRowB + offset) * N + BN * bx + innerColB * 4],
        cuda::aligned_size_t<16>(16), pipe);
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
__global__ void __launch_bounds__(((BM / WM) * (BN / WN)) * 32)
                blocktiling_2d_transpose_kernel(float *a, // A: M×K row-major
                                                float *b, float *c, int M,
                                                int N, int K, float alpha,
                                                float beta) {
  const int NUM_STAGES = 2;

  // As stored transposed: As[k][m] — stride-1 column reads during compute
  __shared__ float As[NUM_STAGES][BK][BM + 4];
  __shared__ float Bs[NUM_STAGES][BK][BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int linearThreadId = ty * blockDim.x + tx;

  // Warp specific calculations. Useful for warp-tiling.
  constexpr int warpSize = 32;
  const int warpId = linearThreadId / warpSize;
  const int laneId = linearThreadId % 32;
  // How many TM x TN subtiles per thread.
  constexpr int threadTiles = (WM * WN) / (TM * TN * warpSize);
  // WMITER and WNITER are the "dimensions" of the thread tiles.
  constexpr int WNITER = WN / (WSUBN * TN);
  constexpr int WMITER = threadTiles / WNITER;
  // WSUBN * WSUBM == 32.
  constexpr int WSUBM = warpSize / WSUBN;
  // This is the warpColumn and warpRow inside the block.
  const int warpRow = (warpId * WN) / BN;
  const int warpCol = warpId % (BN / WN);
  // Thread row / column inside the warp.
  const int threadWarpCol = laneId % WSUBN;
  const int threadWarpRow = laneId / WSUBN;

  // Advance c to the warp's output tile
  c += (by * BM + warpRow * WM) * N + bx * BN + warpCol * WN;

  // 2D block tiling on register file.
  float threadSum[TM * TN * WNITER * WMITER] = {0.0f};
  float regM[TM * WMITER] = {0.0f};
  float regN[TN * WNITER] = {0.0f};

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Load 0th tile.
  pipe.producer_acquire();
  load_tile<BM, BN, BK, WM, WN>(a, b, As[0], Bs[0], 0, by, bx, K, N, pipe);
  pipe.producer_commit();

  int compute_stage = 0;

  for (int i = 0; i < K; i += BK) {
    const int load_stage = compute_stage ^ 1;

    // Load next tile so we have 2 pending pipeline stages.
    if (i + BK < K) {
      pipe.producer_acquire();
      load_tile<BM, BN, BK, WM, WN>(a, b, As[load_stage], Bs[load_stage],
                                      i + BK, by, bx, K, N, pipe);
      pipe.producer_commit();
    }

    // Wait until the current compute_stage's async copies are done.
    pipe.consumer_wait();
    __syncthreads();

    // Compute stage.
    for (int j = 0; j < BK; j++) {
      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int tid_m = 0; tid_m < TM; tid_m++) {
          const int aRow =
              warpRow * WM + wSubRow * WSUBM * TM + threadWarpRow * TM + tid_m;
          regM[wSubRow * TM + tid_m] = As[compute_stage][j][aRow];
        }
      }

      for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
        for (int tid_n = 0; tid_n < TN; tid_n++) {
          const int bCol =
              warpCol * WN + wSubCol * WSUBN * TN + threadWarpCol * TN + tid_n;
          regN[wSubCol * TN + tid_n] = Bs[compute_stage][j][bCol];
        }
      }

      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
          for (int tid_m = 0; tid_m < TM; tid_m++) {
            for (int tid_n = 0; tid_n < TN; tid_n++) {
              const int regMIdx = wSubRow * TM + tid_m;
              const int regNIdx = wSubCol * TN + tid_n;
              const int resIdx =
                  (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;
              threadSum[resIdx] += regM[regMIdx] * regN[regNIdx];
            }
          }
        }
      }
    }

    __syncthreads();
    pipe.consumer_release();

    compute_stage ^= 1;
  }

  // C = alpha * (A*B) + beta * C  — float4 stores (TN must be a multiple of 4).
  for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
    for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
      float *c_interim =
          c + (wSubRow * WSUBM * TM) * N + wSubCol * WSUBN * TN;
      for (int tid_m = 0; tid_m < TM; tid_m++) {
        for (int tid_n = 0; tid_n < TN; tid_n += 4) {
          const int resIdx =
              (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;

          float4 c_reg = *reinterpret_cast<const float4 *>(
              &c_interim[(threadWarpRow * TM + tid_m) * N +
                         threadWarpCol * TN + tid_n]);

          float4 result =
              make_float4(alpha * threadSum[resIdx + 0] + beta * c_reg.x,
                          alpha * threadSum[resIdx + 1] + beta * c_reg.y,
                          alpha * threadSum[resIdx + 2] + beta * c_reg.z,
                          alpha * threadSum[resIdx + 3] + beta * c_reg.w);

          *reinterpret_cast<float4 *>(
              &c_interim[(threadWarpRow * TM + tid_m) * N +
                         threadWarpCol * TN + tid_n]) = result;
        }
      }
    }
  }
}

#endif // PIPELINING_KERNEL_H
