# CUDA kernel build & run recipes (all native, requires local CUDA install):
#   just run naive
#   just build 2d_blocktiling_vectorized
#   just profile reduce_shared_memory_banks
#   just check naive
#   just sass 2d_blocktiling_vectorized
#   just ptx  2d_blocktiling_vectorized

sm       := "120"
src      := "src"
kernels  := "src/kernels"
libs     := "-lcublas"
includes := "-I./src"
shared   := src / "benchmark.cu" + " " + src / "utils.cu" + " " + kernels / "cublas_matmul.cu"

# List all available recipes
default:
    @just --list

# ── Native ────────────────────────────────────────────────────────────────────

# Compile-check (no output binary)  (e.g. just check naive)
check kernel="naive":
    nvcc {{includes}} -arch=sm_{{sm}} {{src}}/run_{{kernel}}.cu {{shared}} {{libs}} -o /dev/null 2>&1 \
        | grep -E "error|warning" || echo "✓ No errors found"

# Build a kernel                     (e.g. just build naive)
build kernel="naive":
    nvcc {{includes}} -arch=sm_{{sm}} {{src}}/run_{{kernel}}.cu {{shared}} {{libs}} -o run_{{kernel}}
    @echo "✓ Built: run_{{kernel}}"

# Build + run a kernel               (e.g. just run naive)
run kernel="naive": (build kernel)
    ./run_{{kernel}}

# Build + autotune a kernel          (e.g. just autotune reduce_shared_memory_banks)
autotune kernel: (build kernel)
    ./run_{{kernel}} --autotune

# Build + ncu profile                (e.g. just profile naive)
profile kernel="naive": (build kernel)
    ncu --set full ./run_{{kernel}} --profile

# Dump SASS assembly                 (e.g. just sass 2d_blocktiling_vectorized)
sass kernel="naive":
    mkdir -p inspect
    nvcc {{includes}} -arch=sm_{{sm}} {{src}}/run_{{kernel}}.cu {{shared}} {{libs}} --cubin \
        -o inspect/run_{{kernel}}.sm{{sm}}.cubin
    cuobjdump --dump-sass inspect/run_{{kernel}}.sm{{sm}}.cubin \
        > inspect/run_{{kernel}}.sm{{sm}}.sass
    @echo "✓ SASS written to inspect/run_{{kernel}}.sm{{sm}}.sass"

# Dump PTX IR                        (e.g. just ptx 2d_blocktiling_vectorized)
ptx kernel="naive":
    mkdir -p inspect
    nvcc {{includes}} -arch=sm_{{sm}} {{src}}/run_{{kernel}}.cu {{shared}} {{libs}} --ptx \
        -o inspect/run_{{kernel}}.sm{{sm}}.ptx
    @echo "✓ PTX written to inspect/run_{{kernel}}.sm{{sm}}.ptx"

# Build + run every kernel sequentially
run-all:
    @for k in naive coalesced shared_mem 1d_blocktiling 2d_blocktiling \
               2d_blocktiling_vectorized 2d_blocktiling_transpose \
               transpose_swizzle reduce_shared_memory_banks; do \
        echo ""; \
        echo "══════════════════════════════════════════"; \
        echo "  Running: $k"; \
        echo "══════════════════════════════════════════"; \
        just run $k; \
    done

# ── Utility ───────────────────────────────────────────────────────────────────

# Remove all built binaries and inspect output
clean:
    rm -f run_naive run_coalesced run_shared_mem run_1d_blocktiling \
          run_2d_blocktiling run_2d_blocktiling_vectorized \
          run_2d_blocktiling_transpose run_transpose_swizzle \
          run_autotune run_reduce_shared_memory_banks \
          run_reference_vectorized
    rm -rf inspect
