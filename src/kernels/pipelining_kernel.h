#ifndef PIPELINING_KERNEL_H
#define PIPELINING_KERNEL_H

#include <cuda/pipeline>

template <int BM, int BN, int BK, int WM, int WN>
__device__ void load_tile(float *a_t, float *b,
                          float (&As_stage)[BK][BM + 4],
                          float (&Bs_stage)[BK][BN], int kOffset, int by,
                          int bx, int M, int N,
                          cuda::pipeline<cuda::thread_scope_thread> &pipe) {
  constexpr int numThreads = ((BM / WM) * (BN / WN)) * 32;
  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;

  // A tile: BK rows × BM cols from K×M row-major a_t — symmetric to B.
  const int innerRowA = linearThreadId / (BM / 4);
  const int innerColA = linearThreadId % (BM / 4);
  constexpr int rowStrideA = numThreads / (BM / 4);
  // B tile: BK rows × BN cols.
  const int innerRowB = linearThreadId / (BN / 4);
  const int innerColB = linearThreadId % (BN / 4);
  constexpr int rowStrideB = numThreads / (BN / 4);

  // Load A tile with 16-byte async copies (a_t is K×M row-major).
  for (int offset = 0; offset + rowStrideA <= BK; offset += rowStrideA) {
    cuda::memcpy_async(
        &As_stage[innerRowA + offset][innerColA * 4],
        &a_t[(kOffset + innerRowA + offset) * M + BM * by + innerColA * 4],
        cuda::aligned_size_t<16>(16), pipe);
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
                blocktiling_2d_transpose_kernel(float *a_t, // A transposed: K×M row-major
                                                float *b, float *c, int M,
                                                int N, int K, float alpha,
                                                float beta) {
  const int NUM_STAGES = 2;

  // As[k][m+pad]: stride-1 along m for compute; +4 padding shifts bank pattern between k-steps
  __shared__ float As[NUM_STAGES][BK][BM + 4];
  __shared__ float Bs[NUM_STAGES][BK][BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int linearThreadId = ty * blockDim.x + tx;

  constexpr int warpSize = 32;
  const int warpId = linearThreadId / warpSize;
  const int laneId = linearThreadId % 32;
  constexpr int threadTiles = (WM * WN) / (TM * TN * warpSize);
  constexpr int WNITER = WN / (WSUBN * TN);
  constexpr int WMITER = threadTiles / WNITER;
  constexpr int WSUBM = warpSize / WSUBN;
  const int warpRow = (warpId * WN) / BN;
  const int warpCol = warpId % (BN / WN);
  const int threadWarpCol = laneId % WSUBN;
  const int threadWarpRow = laneId / WSUBN;

  c += (by * BM + warpRow * WM) * N + bx * BN + warpCol * WN;

  float threadSum[TM * TN * WNITER * WMITER] = {0.0f};
  float regM[TM * WMITER] = {0.0f};
  float regN[TN * WNITER] = {0.0f};

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  load_tile<BM, BN, BK, WM, WN>(a_t, b, As[0], Bs[0], 0, by, bx, M, N, pipe);
  pipe.producer_commit();

  int compute_stage = 0;

  for (int i = 0; i < K; i += BK) {
    const int load_stage = compute_stage ^ 1;

    if (i + BK < K) {
      pipe.producer_acquire();
      load_tile<BM, BN, BK, WM, WN>(a_t, b, As[load_stage], Bs[load_stage],
                                      i + BK, by, bx, M, N, pipe);
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

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
