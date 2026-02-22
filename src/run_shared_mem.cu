#include "test_harness.h"
#include "kernels/shared_mem_kernel.h"

int main(int argc, char** argv) {
    constexpr int BLOCKSIZE = 16;

    auto ctx = setup_test("Shared Memory Kernel", parse_mode(argc, argv));
    std::cout << "Tile:   BLOCKSIZE=" << BLOCKSIZE << "\n\n";

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    BenchmarkResult result = run_kernel(
        ctx, shared_mem_kernel<BLOCKSIZE>, "Shared Memory Kernel", threads);

    verify_and_report(ctx, result);
    cleanup_test(ctx);
    return 0;
}
