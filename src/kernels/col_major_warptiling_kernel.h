#ifndef COL_MAJOR_WARP_TILING_KERNEL_H
#define COL_MAJOR_WARP_TILING_KERNEL_H

// A: row-major, B: col-major, C: col-major.
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
__global__ void __launch_bounds__(((BM / WM) * (BN / WN)) * 32)
                col_major_warptiling_kernel(float * __restrict__ a,
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

  const int innerRowA = linearThreadId / (BK / 4);
  const int innerColA = linearThreadId % (BK / 4);
  constexpr int rowStrideA = (numThreads * 4) / BK;
  const int innerRowB = linearThreadId / (BK / 4);
  const int innerColB = linearThreadId % (BK / 4);
  constexpr int rowStrideB = numThreads / (BK / 4);

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

  // Advance c to the warp's tile: col-major C[m][n] = c[n*M + m].
  c += (bx * BN + warpCol * WN) * M + by * BM + warpRow * WM;

  float threadSum[TM * TN * WNITER * WMITER] = {0.0f};
  float regM[TM * WMITER] = {0.0f};
  float regN[TN * WNITER] = {0.0f};

  for (int i = 0; i < K; i += BK) {

    // Load A tile (row-major), transposing on-the-fly into As[k][m].
    for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 val = *reinterpret_cast<const float4 *>(
          &a[(BM * by + innerRowA + offset) * K + innerColA * 4]);
      As[innerColA * 4 + 0][innerRowA + offset] = val.x;
      As[innerColA * 4 + 1][innerRowA + offset] = val.y;
      As[innerColA * 4 + 2][innerRowA + offset] = val.z;
      As[innerColA * 4 + 3][innerRowA + offset] = val.w;
    }

    // Load col-major B: b[n*K + k]. Float4 along k (contiguous), scatter into Bs[k][n].
    for (int offset = 0; offset + rowStrideB <= BN; offset += rowStrideB) {
      float4 val = *reinterpret_cast<const float4 *>(
          &b[(BN * bx + innerRowB + offset) * K + innerColB * 4]);
      Bs[innerColB * 4 + 0][innerRowB + offset] = val.x;
      Bs[innerColB * 4 + 1][innerRowB + offset] = val.y;
      Bs[innerColB * 4 + 2][innerRowB + offset] = val.z;
      Bs[innerColB * 4 + 3][innerRowB + offset] = val.w;
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

    a += BK;
    b += BK;

    __syncthreads();
  }

  // Write col-major C via float4 along M (TM must be a multiple of 4).
  for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
    for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
      float *c_interim = c + (wSubCol * WSUBN * TN) * M + wSubRow * WSUBM * TM;
      for (int tid_n = 0; tid_n < TN; tid_n++) {
        for (int tid_m = 0; tid_m < TM; tid_m += 4) {
          const int resIdx =
              (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;

          float4 c_reg = *reinterpret_cast<const float4 *>(
              &c_interim[(threadWarpCol * TN + tid_n) * M +
                         threadWarpRow * TM + tid_m]);

          float4 result = make_float4(
              alpha * threadSum[resIdx + 0 * WNITER * TN] + beta * c_reg.x,
              alpha * threadSum[resIdx + 1 * WNITER * TN] + beta * c_reg.y,
              alpha * threadSum[resIdx + 2 * WNITER * TN] + beta * c_reg.z,
              alpha * threadSum[resIdx + 3 * WNITER * TN] + beta * c_reg.w);

          *reinterpret_cast<float4 *>(
              &c_interim[(threadWarpCol * TN + tid_n) * M +
                         threadWarpRow * TM + tid_m]) = result;
        }
      }
    }
  }
}

#endif // COL_MAJOR_WARP_TILING_KERNEL_H
