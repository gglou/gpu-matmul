#include <cuda.h>
#include <cuda_bf16.h>
#include "test_harness.h"
#include "tensor_kernels/tensor_kernel_1.cu"

__global__ void convert_fp32_to_bf16(const float* in, __nv_bfloat16* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

static __nv_bfloat16* to_bf16(float* d_fp32, int n) {
  __nv_bfloat16* d_bf16;
  cudaMalloc(&d_bf16, sizeof(__nv_bfloat16) * n);
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  convert_fp32_to_bf16<<<blocks, threads>>>(d_fp32, d_bf16, n);
  cudaDeviceSynchronize();
  return d_bf16;
}

// Transpose B from K x N (row-major) to N x K (row-major) using cublasSgeam.
// d_b is K x N row-major = N x K col-major (lda=N).
// We want d_b_t as N x K row-major = K x N col-major (ldc=K).
static float* transpose_b(float* d_b, int K, int N) {
  float* d_b_t;
  cudaMalloc(&d_b_t, sizeof(float) * K * N);
  float one = 1.0f, zero = 0.0f;
  cublasSgeam(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
              K, N, &one, d_b, N, &zero, d_b_t, K, d_b_t, K);
  cudaDeviceSynchronize();
  return d_b_t;
}

int main(int argc, char** argv) {
  auto ctx = setup_test("Tensor Kernel 1 (TMA + mma.sync m16n8k16 bf16, sm_120)",
                        parse_mode(argc, argv));
  int M = ctx.dims.M, N = ctx.dims.N, K = ctx.dims.K;

  std::cout << "Tile: BM=" << BM << " BN=" << BN
            << " MMA=" << MMA_M << "x" << MMA_N << "x" << MMA_K
            << " threads=" << NUM_THREADS
            << " regs/thread=" << RESULTS_PER_THREAD << "\n\n";

  // convert A to bf16 (M x K, K contiguous)
  __nv_bfloat16* d_a_bf16 = to_bf16(ctx.d_a, M * K);

  // transpose B from K x N to N x K, then convert to bf16
  float* d_b_t = transpose_b(ctx.d_b, K, N);
  __nv_bfloat16* d_b_bf16 = to_bf16(d_b_t, K * N);
  cudaFree(d_b_t);

  // A: M x K bf16 row-major, tile BM x BK
  CUtensorMap tma_a;
  create_tma_descriptor_2d<BM, BK>(tma_a, d_a_bf16, M, K);

  // B: N x K bf16 row-major (transposed), tile BN x BK
  CUtensorMap tma_b;
  create_tma_descriptor_2d<BN, BK>(tma_b, d_b_bf16, N, K);

  dim3 threads(NUM_THREADS);
  dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
  size_t smem_size = (BM * BK + BN * BK) * sizeof(__nv_bfloat16);

  BenchmarkResult result = run_kernel_custom(ctx, "Tensor Kernel 1", [&]() {
    tensor_kernel_1<<<blocks, threads, smem_size>>>(
      tma_a, tma_b, ctx.d_c, M, N, K);
  });

  // bf16 inputs accumulate larger rounding errors than fp32
  cudaMemcpy(ctx.h_c, ctx.d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  float max_err = 0, max_rel = 0;
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    float err = std::abs(ctx.h_c[i] - ctx.h_ref[i]);
    float rel = err / (std::abs(ctx.h_ref[i]) + 1e-6f);
    if (err > max_err) max_err = err;
    if (rel > max_rel) max_rel = rel;
    if (rel > 0.02f) err_count++;
  }
  printf("Max abs error: %.4f  Max rel error: %.6f  Elements >2%% rel error: %d / %d\n",
         max_err, max_rel, err_count, M * N);

  if (max_rel < 0.02f)
    printf("PASSED (within bf16 tolerance)\n");
  else
    printf("WARNING: %d elements exceed 2%% relative error\n", err_count);

  std::cout << "Speed: " << ctx.cublas_result.min_time / result.min_time
            << "x cuBLAS  (" << result.gflops << " vs "
            << ctx.cublas_result.gflops << " GFLOPS)\n";

  cudaFree(d_a_bf16);
  cudaFree(d_b_bf16);
  cleanup_test(ctx);
  return 0;
}
