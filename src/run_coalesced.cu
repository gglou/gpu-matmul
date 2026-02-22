#include "test_harness.h"
#include "kernels/coalesced_kernel.h"

int main(int argc, char** argv) {
    auto ctx = setup_test("Coalesced Kernel", parse_mode(argc, argv));

    dim3 threads(16, 16);
    BenchmarkResult result = run_kernel(ctx, coalesced_kernel_matmul, "Coalesced Kernel", threads);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
