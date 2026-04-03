#ifndef PIPELINING_KERNEL_H
#define PIPELINING_KERNEL_H

// Emit a 16-byte async copy that bypasses L1 (cache-global).
__device__ __forceinline__ void cp_async_cg(void *dst, const void *src) {
  unsigned dst_smem =
      static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(dst_smem),
         "l"(reinterpret_cast<unsigned long long>(src)));
}

// Close the current async-copy group.
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most N groups are still pending.  N must be a compile-time
// constant so ptxas can encode it as a literal in the instruction.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

template <int BM, int BN, int BK, int WM, int WN>
__device__ void load_tile(float *a_t, float *b,
                          float (&As_stage)[BK][BM + 4],
                          float (&Bs_stage)[BK][BN], int kOffset, int by,
                          int bx, int M, int N) {
  constexpr int numThreads = ((BM / WM) * (BN / WN)) * 32;
  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;

  // A tile: BK rows × BM cols from K×M row-major a_t.
  const int innerRowA = linearThreadId / (BM / 4);
  const int innerColA = linearThreadId % (BM / 4);
  constexpr int rowStrideA = numThreads / (BM / 4);
  // B tile: BK rows × BN cols.
  const int innerRowB = linearThreadId / (BN / 4);
  const int innerColB = linearThreadId % (BN / 4);
  constexpr int rowStrideB = numThreads / (BN / 4);

  for (int offset = 0; offset + rowStrideA <= BK; offset += rowStrideA) {
    cp_async_cg(
        &As_stage[innerRowA + offset][innerColA * 4],
        &a_t[(kOffset + innerRowA + offset) * M + BM * by + innerColA * 4]);
  }

  for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cp_async_cg(
        &Bs_stage[innerRowB + offset][innerColB * 4],
        &b[(kOffset + innerRowB + offset) * N + BN * bx + innerColB * 4]);
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WSUBN>
__global__ void __launch_bounds__(((BM / WM) * (BN / WN)) * 32)
                pipelining_kernel(float *a_t, // A transposed: K×M row-major
                                  float *b, float *c, int M,
                                  int N, int K, float alpha,
                                  float beta) {
  struct alignas(128) Tiles {
    float As[2][BK][BM + 4];
    float Bs[2][BK][BN];
  };
  __shared__ Tiles smem;
  auto &As = smem.As;
  auto &Bs = smem.Bs;

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
  float regM[2][TM * WMITER] = {};
  float regN[2][TN * WNITER] = {};

  // Prologue: issue the first tile load and close its group.
  load_tile<BM, BN, BK, WM, WN>(a_t, b, As[0], Bs[0], 0, by, bx, M, N);
  cp_async_commit();

  int compute_stage = 0;

  for (int i = 0; i < K; i += BK) {
    const int load_stage = compute_stage ^ 1;

    if (i + BK < K) {
      // Prefetch the next tile into the alternate buffer.
      load_tile<BM, BN, BK, WM, WN>(a_t, b, As[load_stage], Bs[load_stage],
                                     i + BK, by, bx, M, N);
      cp_async_commit();
      // Allow the newly committed group to stay in-flight while we wait
      // for the older group (the one we are about to compute on).
      cp_async_wait<1>();
    } else {
      // Last k-step: no prefetch — drain all pending groups.
      cp_async_wait<0>();
    }
    __syncthreads();

    // Prologue: load j=0 into register buffer 0.
    for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
      for (int tid_m = 0; tid_m < TM; tid_m++) {
        const int aRow =
            warpRow * WM + wSubRow * WSUBM * TM + threadWarpRow * TM + tid_m;
        regM[0][wSubRow * TM + tid_m] = As[compute_stage][0][aRow];
      }
    }
    for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
      for (int tid_n = 0; tid_n < TN; tid_n++) {
        const int bCol =
            warpCol * WN + wSubCol * WSUBN * TN + threadWarpCol * TN + tid_n;
        regN[0][wSubCol * TN + tid_n] = Bs[compute_stage][0][bCol];
      }
    }

    for (int j = 0; j < BK; j++) {
      const int cur = j & 1;
      const int nxt = 1 - cur;

      // Prefetch j+1 into alternate buffer (overlaps with FMAs below).
      if (j + 1 < BK) {
        for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
          for (int tid_m = 0; tid_m < TM; tid_m++) {
            const int aRow =
                warpRow * WM + wSubRow * WSUBM * TM + threadWarpRow * TM + tid_m;
            regM[nxt][wSubRow * TM + tid_m] = As[compute_stage][j + 1][aRow];
          }
        }
        for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
          for (int tid_n = 0; tid_n < TN; tid_n++) {
            const int bCol =
                warpCol * WN + wSubCol * WSUBN * TN + threadWarpCol * TN + tid_n;
            regN[nxt][wSubCol * TN + tid_n] = Bs[compute_stage][j + 1][bCol];
          }
        }
      }

      // Compute using current buffer.
      for (int wSubRow = 0; wSubRow < WMITER; wSubRow++) {
        for (int wSubCol = 0; wSubCol < WNITER; wSubCol++) {
          for (int tid_m = 0; tid_m < TM; tid_m++) {
            for (int tid_n = 0; tid_n < TN; tid_n++) {
              const int regMIdx = wSubRow * TM + tid_m;
              const int regNIdx = wSubCol * TN + tid_n;
              const int resIdx =
                  (wSubRow * TM + tid_m) * (WNITER * TN) + wSubCol * TN + tid_n;
              threadSum[resIdx] += regM[cur][regMIdx] * regN[cur][regNIdx];
            }
          }
        }
      }
    }

    __syncthreads();
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
