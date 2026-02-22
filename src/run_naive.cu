#include "test_harness.h"
#include "kernels/naive_kernel.h"

int main(int argc, char** argv) {
    auto ctx = setup_test("Naive Kernel", parse_mode(argc, argv));

    dim3 threads(16, 16);
    BenchmarkResult result = run_kernel(ctx, naive_kernel_matmul, "Naive Kernel", threads);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
