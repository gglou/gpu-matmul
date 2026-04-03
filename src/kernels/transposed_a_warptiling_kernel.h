#ifndef TRANSPOSED_A_WARPTILING_KERNEL_H
#define TRANSPOSED_A_WARPTILING_KERNEL_H

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
__global__ void __launch_bounds__(((BM / WM) * (BN / WN)) * 32)
                transposed_a_warptiling_kernel(float * __restrict__ a_t,
                                               float * __restrict__ b,
                                               float * __restrict__ c,
                                               int M, int N, int K,
                                               float alpha, float beta) {
  struct alignas(16) Tiles {
    float As[BK][BM + 4];
    float Bs[BK][BN];
  };
  __shared__ Tiles smem;
  auto &As = smem.As;
  auto &Bs = smem.Bs;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  constexpr int numThreads = ((BM / WM) * (BN / WN)) * 32;
  const int linearThreadId = ty * blockDim.x + tx;

  const int innerRowA = linearThreadId / (BM / 4);
  const int innerColA = linearThreadId % (BM / 4);
  constexpr int rowStrideA = numThreads / (BM / 4);
  const int innerRowB = linearThreadId / (BN / 4);
  const int innerColB = linearThreadId % (BN / 4);
  constexpr int rowStrideB = numThreads / (BN / 4);

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

  for (int i = 0; i < K; i += BK) {
    for (int offset = 0; offset + rowStrideA <= BK; offset += rowStrideA) {
      *reinterpret_cast<float4 *>(&As[innerRowA + offset][innerColA * 4]) =
          *reinterpret_cast<const float4 *>(
              &a_t[(i + innerRowA + offset) * M + BM * by + innerColA * 4]);
    }

    for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      *reinterpret_cast<float4 *>(&Bs[innerRowB + offset][innerColB * 4]) =
          *reinterpret_cast<const float4 *>(
              &b[(i + innerRowB + offset) * N + BN * bx + innerColB * 4]);
    }

    __syncthreads();

    for (int j = 0; j < BK; j++) {
      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int tid_m = 0; tid_m < TM; tid_m++) {
          const int aRow =
              warpRow * WM + wSubRow * WSUBM * TM + threadWarpRow * TM + tid_m;
          regM[wSubRow * TM + tid_m] = As[j][aRow];
        }
      }

      for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
        for (int tid_n = 0; tid_n < TN; tid_n++) {
          const int bCol =
              warpCol * WN + wSubCol * WSUBN * TN + threadWarpCol * TN + tid_n;
          regN[wSubCol * TN + tid_n] = Bs[j][bCol];
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

#endif // TRANSPOSED_A_WARPTILING_KERNEL_H
