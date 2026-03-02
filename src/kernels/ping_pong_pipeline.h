#ifndef PING_PONG_PIPELINE_KERNEL_H
#define PING_PONG_PIPELINE_KERNEL_H

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
__global__ void blocktiling_2d_transpose_kernel(float *a, // A: M×K row-major
                                                float *b, float *c, int M,
                                                int N, int K, float alpha,
                                                float beta) {
  // Pipelining stages.
  const int NUM_STAGES = 2;
  // As stored transposed: As[k][m] — stride-1 column reads during compute
  __shared__ float As[NUM_STAGES][BK][BM + 1];
  __shared__ float Bs[NUM_STAGES][BK][BN];
  // thread "coordinates"
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // block "coordinates"
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int numThreads = blockDim.x * blockDim.y;
  const int linearThreadId = ty * blockDim.x + tx;

  const int loadPerThreadA = (BM * BK) / (numThreads * 4);
  const int loadPerThreadB = (BK * BN) / (numThreads * 4);

  // Warp specific calculations. Useful for warp-tiling.
  const int warpSize = 32;
  const int warpId = linearThreadId / warpSize;
  const int laneId = linearThreadId % 32;
  // How many TM x TN subtiles per thread.
  const int threadTiles = (WM * WN) / (TM * TN * warpSize);
  // WMITER and WNITER are the "dimensions" of the thread tiles.
  const int WNITER = WN / (WSUBN * TN);
  const int WMITER = threadTiles / WNITER;
  // WSUBN * WSUBM == 32.
  const int WSUBM = warpSize / WSUBN;
  // This is the warpColumn and warpRow inside the block.
  const int warpRow = (warpId * WN) / BN;
  const int warpCol = warpId % (BN / WN);
  // Thread row / column inside the warp.
  // Reminder: WSUBN is the how many threads are on the N dimension
  // of the warp tile.
  const int threadWarpCol = laneId % WSUBN;
  const int threadWarpRow = laneId / WSUBN;

  const int cRow = BM * by + warpRow * WM + threadWarpRow * TM;
  const int cCol = BN * bx + warpCol * WN + threadWarpCol * TN;

  // 2D block tiling on register file.
  float threadSum[TM * TN * WNITER * WMITER] = {0.0f};
  float regM[TM * WMITER] = {0.0f};
  float regN[TN * WNITER] = {0.0f};

  for (int i = 0; i < K; i += BK) {

    const int stage = (i / BK) & 1; // Alternate stages.

    // Load A tile, transposing on-the-fly into As[k][m].
    for (int la = 0; la < loadPerThreadA; la++) {
      int idx = (linearThreadId + la * numThreads) * 4;
      int aRow = idx / BK;
      int aCol = idx % BK;

      float4 val = *reinterpret_cast<const float4 *>(
          &a[(BM * by + aRow) * K + i + aCol]);
      As[stage][aCol + 0][aRow] = val.x;
      As[stage][aCol + 1][aRow] = val.y;
      As[stage][aCol + 2][aRow] = val.z;
      As[stage][aCol + 3][aRow] = val.w;
    }

    // Load Bs from B (row-major, coalesced).
    for (int lb = 0; lb < loadPerThreadB; lb++) {
      int idx = (linearThreadId + lb * numThreads) * 4;
      int bRow = idx / BN;
      int bCol = idx % BN;

      float4 val = *reinterpret_cast<const float4 *>(
          &b[(i + bRow) * N + BN * bx + bCol]);

      Bs[stage][bRow][bCol + 0] = val.x;
      Bs[stage][bRow][bCol + 1] = val.y;
      Bs[stage][bRow][bCol + 2] = val.z;
      Bs[stage][bRow][bCol + 3] = val.w;
    }

    __syncthreads();

    for (int j = 0; j < BK; j++) {
      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int tid_m = 0; tid_m < TM; tid_m++) {
          // warpRow * WM to go to the correct warp starting row of the warp
          // tile. wSubRow * WSUBM * TM to go to the correct "row" of the
          // (WMITER * TM) x (WNITER * TN) tile. threadWarpRow * TM + tid_in to
          // go to the correct value for the TM x TN tile
          const int aRow =
              warpRow * WM + wSubRow * WSUBM * TM + threadWarpRow * TM + tid_m;
          regM[wSubRow * TM + tid_m] = As[stage][j][aRow];
        }
      }

      for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
        for (int tid_n = 0; tid_n < TN; tid_n++) {
          // Similar logic to above.
          const int bCol =
              warpCol * WN + wSubCol * WSUBN * TN + threadWarpCol * TN + tid_n;
          regN[wSubCol * TN + tid_n] = Bs[stage][j][bCol];
        }
      }

      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
          for (int tid_m = 0; tid_m < TM; tid_m++) {
            for (int tid_n = 0; tid_n < TN; tid_n++) {
              const int regMIdx = wSubRow * TM + tid_m;
              const int regNIdx = wSubCol * TN + tid_n;
              // wSubRow * TM + tid_m -- row
              // wSubCol * TN + tid_n -- column
              // threadResults has a WMITER * TM x WNITER * WN shape.
              // (Compared to the TM x TN shape that it used to have).
              const int resIdx =
                  (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;
              threadSum[resIdx] += regM[regMIdx] * regN[regNIdx];
            }
          }
        }
      }
    }
  }

  // C = alpha * (A*B) + beta * C  — float4 stores (TN must be a multiple of 4).
  for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
    for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
      for (int tid_m = 0; tid_m < TM; tid_m++) {
        for (int tid_n = 0; tid_n < TN; tid_n += 4) {
          const int resIdx =
              (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;

          const int base = (cRow + wSubRow * WSUBM * TM + tid_m) * N + cCol +
                           wSubCol * WSUBN * TN + tid_n;

          // Load c values first into a register.
          // Keeping TM, TN as 4 is super convenient for vectorized stores.
          float4 c_reg = *reinterpret_cast<const float4 *>(&c[base]);

          // Do math in registers
          float4 result =
              make_float4(alpha * threadSum[resIdx + 0] + beta * c_reg.x,
                          alpha * threadSum[resIdx + 1] + beta * c_reg.y,
                          alpha * threadSum[resIdx + 2] + beta * c_reg.z,
                          alpha * threadSum[resIdx + 3] + beta * c_reg.w);

          // Assign back the result
          *reinterpret_cast<float4 *>(&c[base]) = result;
        }
      }
    }
  }
}

#endif // PING_PONG_PIPELINE
