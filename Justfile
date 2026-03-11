# CUDA kernel build & run recipes (all native, requires local CUDA install):
#   just run naive
#   just build 2d_blocktiling_vectorized
#   just check naive
#   just sass 2d_blocktiling_vectorized
#   just ptx  2d_blocktiling_vectorized

sm       := "120"
src      := "src"
kernels  := "src/kernels"
libs     := "-lcublas"
includes := "-I./src"
shared   := src / "benchmark.cu" + " " + src / "utils.cu" + " " + kernels / "cublas_matmul.cu"

# Default warp-tiling template params: BM,BN,BK,TM,TN,WM,WN,WSUBN
warp_params := "128,128,16,4,4,64,64,8"

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
sass kernel="naive": (build kernel)
    mkdir -p inspect
    cuobjdump --dump-sass run_{{kernel}} \
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
               warptiling pipelining; do \
        echo ""; \
        echo "══════════════════════════════════════════"; \
        echo "  Running: $k"; \
        echo "══════════════════════════════════════════"; \
        just run $k; \
    done

# ── Profiler ─────────────────────────────────────────────────────────────────

# Build + ncu dashboard (6-panel PNG)  (e.g. just plot warptiling)
plot kernel="naive": (build kernel)
    #!/usr/bin/env bash
    stamp=$(date +%Y%m%d_%H%M%S)
    python3 profiler/ncu_dashboard.py ./run_{{kernel}} \
        --save-csv "profiler/output/{{kernel}}_${stamp}.csv" \
        -o "profiler/output/{{kernel}}_${stamp}.png"

# Dump raw ncu CSV only              (e.g. just ncu-csv warptiling)
ncu-csv kernel="naive": (build kernel)
    @mkdir -p profiler/output
    ncu --csv --metrics \
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_bit_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_control_pred_on.sum,smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__sass_thread_inst_executed_op_misc_pred_on.sum,smsp__sass_thread_inst_executed_op_uniform_pred_on.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,gpu__time_duration.sum,smsp__inst_executed.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active" \
        ./run_{{kernel}} --profile > profiler/output/{{kernel}}.csv 2>/dev/null
    @echo "✓ CSV saved: profiler/output/{{kernel}}.csv"

# Plot from existing CSV              (e.g. just plot-csv profiler/output/warptiling.csv)
plot-csv csv_path:
    python3 profiler/ncu_dashboard.py --csv {{csv_path}}

# SASS assembly analysis report       (e.g. just sass-report warptiling)
sass-report kernel="naive" params=warp_params: (build kernel)
    python3 profiler/sass_report.py ./run_{{kernel}} \
        --params {{params}} \
        -o profiler/output/{{kernel}}_sass.png

# SASS overview (all template instantiations)  (e.g. just sass-overview warptiling)
sass-overview kernel="naive": (build kernel)
    python3 profiler/sass_report.py ./run_{{kernel}} --overview \
        -o profiler/output/{{kernel}}_sass_overview.png

# SASS comparison of two kernels     (e.g. just sass-compare warptiling reference_warptiling)
sass-compare kernel1 kernel2 params=warp_params: (build kernel1) (build kernel2)
    python3 profiler/sass_report.py ./run_{{kernel1}} \
        --compare ./run_{{kernel2}} \
        --params {{params}} \
        -o profiler/output/{{kernel1}}_vs_{{kernel2}}_sass.png

# ── Utility ───────────────────────────────────────────────────────────────────

# Remove all built binaries and inspect output
clean:
    rm -f run_naive run_coalesced run_shared_mem run_1d_blocktiling \
          run_2d_blocktiling run_2d_blocktiling_vectorized \
          run_2d_blocktiling_transpose \
          run_warptiling run_pipelining run_ping_pong_pipeline \
          run_reference_warptiling run_cublas
    rm -rf inspect profiler/output
