#!/usr/bin/env python3
"""
ncu_dashboard.py — Nsight Compute profiling dashboard for GPU matmul kernels.

Collects ~40 hardware metrics via `ncu --csv` and produces a 6-panel visual
report covering throughput, instruction mix, warp stalls, pipeline utilization,
memory hierarchy efficiency, and roofline analysis.

Usage:
    # Via Justfile (recommended):
    just plot warptiling                            # build + profile + plot
    just ncu-csv warptiling                         # dump raw CSV only
    just plot-csv profiler/output/warptiling.csv    # plot from existing CSV

    # Direct invocation:
    python3 profiler/ncu_dashboard.py ./run_warptiling
    python3 profiler/ncu_dashboard.py ./run_warptiling -o report.png
    python3 profiler/ncu_dashboard.py --csv profiler/output/warptiling.csv

Requires: matplotlib numpy pandas
    pip install matplotlib numpy pandas
"""

import argparse
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Metric catalogue — verified against ncu 2025.4 / SM 12.0 (Blackwell)
#
# Keys are short names used internally; values are full ncu metric identifiers.
# Every metric here has been tested to produce real values on this GPU.
# ═══════════════════════════════════════════════════════════════════════════════

METRICS: dict[str, str] = {}

# ── Panel 1: Hardware throughput (% of peak sustained) ────────────────────────
METRICS |= {
    "sm_throughput":   "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput":  "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput":    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput":    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
}

# ── Panel 2: Instruction mix (thread-level op counts, summed) ─────────────────
METRICS |= {
    "inst_ffma":       "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "inst_fadd":       "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "inst_fmul":       "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "inst_fp16":       "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "inst_fp64":       "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
    "inst_integer":    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "inst_bit":        "smsp__sass_thread_inst_executed_op_bit_pred_on.sum",
    "inst_memory":     "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "inst_control":    "smsp__sass_thread_inst_executed_op_control_pred_on.sum",
    "inst_conversion": "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "inst_misc":       "smsp__sass_thread_inst_executed_op_misc_pred_on.sum",
    "inst_uniform":    "smsp__sass_thread_inst_executed_op_uniform_pred_on.sum",
}

# ── Panel 3: Warp stall reasons (% of active warp cycles) ────────────────────
METRICS |= {
    "stall_long_sb":    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_short_sb":   "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_mio":        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "stall_math":       "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "stall_barrier":    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_membar":     "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "stall_not_sel":    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "stall_selected":   "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "stall_wait":       "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_tex":        "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    "stall_lg":         "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "stall_drain":      "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "stall_dispatch":   "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
    "stall_sleeping":   "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
    "stall_no_inst":    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "stall_branch":     "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "stall_misc":       "smsp__warp_issue_stalled_misc_per_warp_active.pct",
}

# ── Panel 4: Pipeline utilization (% of peak cycles active) ──────────────────
METRICS |= {
    "pipe_fma":        "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_alu":        "smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_fp64":       "smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_tensor":     "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_tma":        "smsp__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_fmalite":    "smsp__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active",
    "pipe_fmaheavy":   "smsp__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active",
}

# ── Panel 5: Memory hierarchy efficiency ─────────────────────────────────────
METRICS |= {
    "l1_hit_rate":               "l1tex__t_sector_hit_rate.pct",
    "l2_hit_rate":               "lts__t_sector_hit_rate.pct",
    "smem_bank_conflicts_st":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smem_bank_conflicts_ld":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_wavefronts_total":     "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "global_ld_sectors_per_req": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "global_st_sectors_per_req": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
}

# ── Register & launch configuration ───────────────────────────────────────────
METRICS |= {
    "regs_per_thread":           "launch__registers_per_thread",
    "regs_per_thread_allocated": "launch__registers_per_thread_allocated",
    "occupancy_limit_regs":      "launch__occupancy_limit_registers",
    "block_size":                "launch__block_size",
    "smem_per_block":            "launch__shared_mem_per_block",
}

# ── Panel 6: Roofline ingredients + summary ──────────────────────────────────
METRICS |= {
    "dram_bytes_read":   "dram__bytes_op_read.sum",
    "dram_bytes_write":  "dram__bytes_op_write.sum",
    "dram_bytes_total":  "dram__bytes.sum",
    "duration_ns":       "gpu__time_duration.sum",
    "ipc_active":        "smsp__inst_executed.avg.per_cycle_active",
    "achieved_occ":      "sm__warps_active.avg.pct_of_peak_sustained_active",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ncu runner + CSV parser
# ═══════════════════════════════════════════════════════════════════════════════

def run_ncu(binary: str, kernel_args: list[str]) -> pd.DataFrame:
    """Launch ncu with --csv on the given binary and return a metric DataFrame."""
    metrics_str = ",".join(METRICS.values())
    cmd = ["ncu", "--csv", "--metrics", metrics_str, binary] + kernel_args
    print(f"▶  {' '.join(cmd[:6])} ... ({len(METRICS)} metrics)", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        print("ERROR — ncu produced no CSV output.", file=sys.stderr)
        if result.stderr:
            print(result.stderr[:2000], file=sys.stderr)
        sys.exit(1)
    return parse_csv(result.stdout)


def parse_csv(raw: str) -> pd.DataFrame:
    """
    Parse ncu --csv output into a tidy DataFrame.

    Handles two formats:
      1. Raw ncu output — may contain ==PROF== lines; CSV rows start with '"'
      2. Pandas-saved CSV — standard CSV with "Metric Name" / "Metric Value" columns
    """
    all_lines = raw.splitlines()

    # Try raw ncu format first: filter to lines starting with '"'
    quoted_lines = [line for line in all_lines if line.startswith('"')]
    if quoted_lines:
        df = pd.read_csv(io.StringIO("\n".join(quoted_lines)))
    elif "Metric Name" in raw:
        # Already a clean CSV (e.g. saved by pandas)
        df = pd.read_csv(io.StringIO(raw))
    else:
        raise ValueError("No CSV data found in ncu output. Did you use --csv?")

    # Clean numeric values — ncu may use locale commas in large numbers
    df["Metric Value"] = (
        df["Metric Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("n/a", "nan", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df


def get(df: pd.DataFrame, key: str, agg: str = "first") -> float:
    """
    Retrieve a single scalar for the metric identified by *key* (short name).
    Falls back to 0.0 if the metric is missing or NaN.
    """
    ncu_name = METRICS.get(key, key)
    rows = df.loc[df["Metric Name"] == ncu_name, "Metric Value"].dropna()
    if rows.empty:
        return 0.0
    if agg == "sum":
        return float(rows.sum())
    if agg == "mean":
        return float(rows.mean())
    return float(rows.iloc[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers — one function per dashboard panel
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette (Material-inspired)
C_BLUE    = "#1565C0"
C_LBLUE   = "#42A5F5"
C_CYAN    = "#0097A7"
C_GREEN   = "#2E7D32"
C_LGREEN  = "#66BB6A"
C_AMBER   = "#F9A825"
C_ORANGE  = "#E65100"
C_RED     = "#C62828"
C_PURPLE  = "#6A1B9A"
C_LPURPLE = "#AB47BC"
C_GRAY    = "#757575"
C_LGRAY   = "#BDBDBD"


def _annotate_hbar(ax, bars, values, fmt="{:.1f}%", fontsize=8):
    """Add value labels to the end of horizontal bars."""
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                val + ax.get_xlim()[1] * 0.015,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", fontsize=fontsize,
            )


# ── Panel 1: Hardware throughput ──────────────────────────────────────────────

def plot_throughput(ax: plt.Axes, df: pd.DataFrame):
    entries = [
        ("SM Compute",  "sm_throughput",  C_BLUE),
        ("DRAM",        "dram_throughput", C_RED),
        ("L1/TEX",      "l1_throughput",   C_GREEN),
        ("L2",          "l2_throughput",   C_PURPLE),
    ]
    labels  = [e[0] for e in entries]
    values  = [get(df, e[1]) for e in entries]
    colors  = [e[2] for e in entries]

    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of Peak Sustained")
    ax.set_title("① Hardware Throughput Utilization", fontweight="bold", fontsize=10)
    ax.axvline(80, color=C_RED, ls="--", lw=0.8, alpha=0.5, label="80% line")
    _annotate_hbar(ax, bars, values)
    ax.legend(fontsize=7, loc="lower right")
    ax.invert_yaxis()


# ── Panel 2: Instruction mix ─────────────────────────────────────────────────

def plot_instruction_mix(ax: plt.Axes, df: pd.DataFrame):
    entries = [
        ("FFMA (FP32 fused)",   "inst_ffma",       C_BLUE),
        ("FADD (FP32 add)",     "inst_fadd",        C_LBLUE),
        ("FMUL (FP32 mul)",     "inst_fmul",        C_CYAN),
        ("FP16",                "inst_fp16",         "#00BCD4"),
        ("FP64",                "inst_fp64",         "#1A237E"),
        ("Integer",             "inst_integer",      C_RED),
        ("Bitwise",             "inst_bit",          C_ORANGE),
        ("Memory (LD/ST)",      "inst_memory",       C_GREEN),
        ("Control / Branch",    "inst_control",      C_AMBER),
        ("Conversion",          "inst_conversion",   C_LPURPLE),
        ("Uniform",             "inst_uniform",      C_GRAY),
        ("Misc",                "inst_misc",         C_LGRAY),
    ]

    labels = [e[0] for e in entries]
    values = [get(df, e[1]) for e in entries]
    colors = [e[2] for e in entries]

    # Filter out zero-count categories
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not non_zero:
        ax.text(0.5, 0.5, "No instruction-mix data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("② Instruction Mix", fontweight="bold", fontsize=10)
        return

    labels_f, values_f, colors_f = zip(*non_zero)
    total = sum(values_f)

    wedges, texts, autotexts = ax.pie(
        values_f, labels=labels_f, colors=colors_f,
        autopct=lambda p: f"{p:.1f}%" if p > 2.5 else "",
        startangle=90, pctdistance=0.78,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for t in texts:
        t.set_fontsize(7)
    for t in autotexts:
        t.set_fontsize(7)
        t.set_fontweight("bold")

    ax.set_title("② Instruction Mix", fontweight="bold", fontsize=10)
    ax.text(0, -1.4, f"Total: {total/1e9:.2f} B thread-instructions",
            ha="center", fontsize=7, color=C_GRAY)


# ── Panel 3: Warp stall reasons ──────────────────────────────────────────────

STALL_LABELS = {
    "stall_long_sb":   "Long Scoreboard\n(global mem latency)",
    "stall_short_sb":  "Short Scoreboard\n(shared/local mem)",
    "stall_mio":       "MIO Throttle",
    "stall_math":      "Math Pipe Throttle",
    "stall_barrier":   "Barrier (__syncthreads)",
    "stall_membar":    "Memory Barrier",
    "stall_not_sel":   "Not Selected",
    "stall_selected":  "Selected (issuing)",
    "stall_wait":      "Wait (fixed latency)",
    "stall_tex":       "TEX Throttle",
    "stall_lg":        "LG Throttle",
    "stall_drain":     "Drain (post-EXIT)",
    "stall_dispatch":  "Dispatch Stall",
    "stall_sleeping":  "Sleeping (nanosleep)",
    "stall_no_inst":   "No Instruction",
    "stall_branch":    "Branch Resolving",
    "stall_misc":      "Misc",
}


def plot_stall_reasons(ax: plt.Axes, df: pd.DataFrame):
    pairs = [
        (label, get(df, key))
        for key, label in STALL_LABELS.items()
        if key != "stall_selected"  # "selected" means actively issuing, not a stall
    ]
    # Sort descending, drop near-zero
    pairs = sorted([(l, v) for l, v in pairs if v > 0.05], key=lambda x: -x[1])

    if not pairs:
        ax.text(0.5, 0.5, "No stall data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("③ Warp Stall Reasons", fontweight="bold", fontsize=10)
        return

    labels, values = zip(*pairs)
    n = len(values)
    cmap = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, max(n, 2)))
    bars = ax.barh(labels, values, color=cmap[:n], height=0.6, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("% of Active Warp Cycles")
    ax.set_title("③ Warp Stall Reasons", fontweight="bold", fontsize=10)
    ax.set_xlim(0, max(values) * 1.3 if values else 100)
    _annotate_hbar(ax, bars, values)
    ax.invert_yaxis()


# ── Panel 4: Pipeline utilization ─────────────────────────────────────────────

def plot_pipeline_utilization(ax: plt.Axes, df: pd.DataFrame):
    entries = [
        ("FMA\n(FP32 mul-add)",  "pipe_fma",      C_BLUE),
        ("FMA Lite",             "pipe_fmalite",   C_LBLUE),
        ("FMA Heavy",            "pipe_fmaheavy",  C_CYAN),
        ("ALU\n(integer)",       "pipe_alu",       C_RED),
        ("FP64",                 "pipe_fp64",      C_PURPLE),
        ("Tensor Core",          "pipe_tensor",    C_GREEN),
        ("TMA",                  "pipe_tma",       C_AMBER),
    ]

    labels = [e[0] for e in entries]
    values = [get(df, e[1]) for e in entries]
    colors = [e[2] for e in entries]

    # Filter out zero-utilization pipes
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0.01]
    if not non_zero:
        ax.text(0.5, 0.5, "No pipeline utilization data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("④ Pipeline Utilization", fontweight="bold", fontsize=10)
        return

    labels_f, values_f, colors_f = zip(*non_zero)
    bars = ax.barh(labels_f, values_f, color=colors_f, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, max(values_f) * 1.3)
    ax.set_xlabel("% of Peak Cycles Active")
    ax.set_title("④ Pipeline Utilization", fontweight="bold", fontsize=10)
    _annotate_hbar(ax, bars, values_f)
    ax.invert_yaxis()


# ── Panel 5: Memory hierarchy efficiency ──────────────────────────────────────

def plot_memory_hierarchy(ax: plt.Axes, df: pd.DataFrame):
    l1_hit    = get(df, "l1_hit_rate")
    l2_hit    = get(df, "l2_hit_rate")
    occ       = get(df, "achieved_occ")
    ipc       = get(df, "ipc_active")

    # Bank conflict rate
    bc_st = get(df, "smem_bank_conflicts_st")
    bc_ld = get(df, "smem_bank_conflicts_ld")
    wf    = get(df, "smem_wavefronts_total")
    bc_pct = 100.0 * (bc_st + bc_ld) / wf if wf > 0 else 0.0

    # Global coalescing (sectors per request; lower = better coalescing)
    gld_spr = get(df, "global_ld_sectors_per_req")
    gst_spr = get(df, "global_st_sectors_per_req")

    # ── Left side: hit rates + occupancy (percentage bars) ──
    pct_labels = ["L1 Hit Rate", "L2 Hit Rate", "Achieved Occupancy"]
    pct_values = [l1_hit, l2_hit, occ]
    pct_colors = [C_GREEN, C_LGREEN, C_BLUE]

    bars = ax.barh(pct_labels, pct_values, color=pct_colors, height=0.45,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% (hit rate or occupancy)")
    ax.set_title("⑤ Memory Hierarchy & Efficiency", fontweight="bold", fontsize=10)
    _annotate_hbar(ax, bars, pct_values)
    ax.invert_yaxis()

    # Register usage
    regs       = get(df, "regs_per_thread")
    regs_alloc = get(df, "regs_per_thread_allocated")
    occ_limit  = get(df, "occupancy_limit_regs")
    blk_size   = get(df, "block_size")
    smem_blk   = get(df, "smem_per_block")

    # ── Right side: text annotations ──
    text_lines = []
    text_lines.append(f"Registers/thread: {regs:.0f}  (allocated: {regs_alloc:.0f})")
    text_lines.append(f"Occupancy limit by regs: {occ_limit:.0f}%")
    text_lines.append(f"Block size: {blk_size:.0f} threads   SMEM/block: {smem_blk/1024:.1f} KB")
    text_lines.append(f"")
    text_lines.append(f"Shared Mem Bank Conflicts: {bc_pct:.1f}% of wavefronts")
    text_lines.append(f"  Store conflicts: {bc_st/1e6:.2f}M   Load conflicts: {bc_ld/1e6:.2f}M")
    text_lines.append(f"Global LD sectors/req: {gld_spr:.1f}   ST sectors/req: {gst_spr:.1f}")
    text_lines.append(f"IPC (active): {ipc:.2f} inst/cycle")

    ax.text(
        0.98, 0.02, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=7, fontfamily="monospace",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCCCCC", alpha=0.9),
    )


# ── Panel 6: Roofline ────────────────────────────────────────────────────────

def plot_roofline(ax: plt.Axes, df: pd.DataFrame, bench_gflops: float | None = None):
    """
    Empirical roofline plot.

    We back-calculate device peak FLOP/s and peak BW from the measured
    throughput percentages, avoiding the need to hardcode device specs.

    If *bench_gflops* is provided (from a native, un-instrumented benchmark),
    we use it as the achieved FLOP/s instead of deriving it from ncu's
    gpu__time_duration, which is inflated by profiling instrumentation.
    """
    ffma  = get(df, "inst_ffma")
    fadd  = get(df, "inst_fadd")
    fmul  = get(df, "inst_fmul")
    dram  = get(df, "dram_bytes_total")
    dur   = get(df, "duration_ns")      # nanoseconds
    sm_pct   = get(df, "sm_throughput") / 100.0
    dram_pct = get(df, "dram_throughput") / 100.0

    flops     = ffma * 2 + fadd + fmul      # each FFMA = 2 FLOPs
    dur_s     = dur * 1e-9

    missing = []
    if flops <= 0:   missing.append("FFMA/FADD/FMUL counts")
    if dram  <= 0:   missing.append("DRAM bytes")
    if dur_s <= 0:   missing.append("duration")
    if sm_pct <= 0:  missing.append("SM throughput %")

    if missing:
        ax.text(0.5, 0.5,
                "Insufficient data for roofline.\n"
                f"Missing: {', '.join(missing)}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color=C_GRAY)
        ax.set_title("⑥ Roofline", fontweight="bold", fontsize=10)
        return

    ai             = flops / dram                              # FLOP/byte

    # If a native benchmark GFLOP/s was given, use it for absolute positioning.
    # Otherwise fall back to ncu-instrumented timing (which is slower due to
    # profiling overhead — typically 2-5x).
    if bench_gflops is not None and bench_gflops > 0:
        achieved_flops = bench_gflops * 1e9                    # GFLOP/s → FLOP/s
        # Back-calculate achieved BW from the same native timing
        native_dur_s   = flops / achieved_flops
        achieved_bw    = dram / native_dur_s                   # byte/s
        timing_source  = "benchmark"
    else:
        achieved_flops = flops / dur_s                         # FLOP/s
        achieved_bw    = dram / dur_s                          # byte/s
        timing_source  = "ncu (instrumented)"

    peak_flops = achieved_flops / sm_pct   if sm_pct > 0   else achieved_flops * 2
    peak_bw    = achieved_bw    / dram_pct if dram_pct > 0  else achieved_bw * 20
    ridge      = peak_flops / peak_bw                          # FLOP/byte

    # Draw roofline envelope
    ai_range = np.logspace(-1, np.log10(max(ai * 10, ridge * 10, 1e4)), 500)
    mem_line    = peak_bw * ai_range         # memory-bound slope
    comp_line   = np.full_like(ai_range, peak_flops)
    roof        = np.minimum(mem_line, comp_line)

    ax.loglog(ai_range, roof / 1e12, color=C_BLUE, lw=2.5, label="Roofline", zorder=2)

    # Ridge point
    ax.axvline(ridge, color=C_GRAY, ls="--", lw=0.8, alpha=0.5)
    ax.text(ridge, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.01,
            f"  ridge={ridge:.1f}", fontsize=7, color=C_GRAY, va="bottom")

    # Kernel operating point
    ax.scatter([ai], [achieved_flops / 1e12], color=C_RED, s=140, zorder=5, marker="*",
               label=f"Kernel: AI={ai:.1f} FLOP/B,  {achieved_flops/1e12:.2f} TFLOP/s")

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    subtitle = "⑥ Roofline"
    if timing_source == "ncu (instrumented)":
        subtitle += "  (⚠ ncu-instrumented timing — use --benchmark-gflops for native)"
    ax.set_title(subtitle, fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)

    # Use plain scalar tick labels instead of powers-of-ten notation
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.get_major_formatter().set_scientific(False)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Bound annotation
    bound = "MEMORY-BOUND" if ai < ridge else "COMPUTE-BOUND"
    colour = C_RED if bound == "MEMORY-BOUND" else C_BLUE
    ax.text(0.03, 0.95, bound, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color=colour,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, alpha=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_dashboard(df: pd.DataFrame, title: str, output: str,
                    bench_gflops: float | None = None):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"NCU Profiling Dashboard  —  {title}",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.99, 0.005, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             fontsize=7, color="#999999", ha="right", va="bottom",
             fontfamily="monospace")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.40,
                           top=0.95, bottom=0.04, left=0.08, right=0.96)

    plot_throughput(           fig.add_subplot(gs[0, 0]), df)
    plot_instruction_mix(     fig.add_subplot(gs[0, 1]), df)
    plot_stall_reasons(       fig.add_subplot(gs[1, 0]), df)
    plot_pipeline_utilization(fig.add_subplot(gs[1, 1]), df)
    plot_memory_hierarchy(    fig.add_subplot(gs[2, 0]), df)
    plot_roofline(            fig.add_subplot(gs[2, 1]), df, bench_gflops)

    plt.savefig(output, bbox_inches="tight", dpi=150)
    print(f"✓  Saved: {output}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GPU kernel profiling dashboard via Nsight Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "binary",
        help="Kernel binary (e.g. ./run_warptiling) OR path to CSV file (with --csv)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (PNG/PDF/SVG). Default: profiler/output/<name>.png",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Treat BINARY as a pre-generated ncu --csv file instead of a binary to profile",
    )
    parser.add_argument(
        "--kernel-args", default="--profile",
        help="Arguments forwarded to the kernel binary (default: '--profile')",
    )
    parser.add_argument(
        "--save-csv", default=None,
        help="Also save the raw CSV to this path (only when profiling live)",
    )
    parser.add_argument(
        "--benchmark-gflops", type=float, default=None,
        help="Native (un-instrumented) GFLOP/s from your benchmark run. "
             "Used to anchor the roofline y-axis to real performance instead of "
             "ncu's inflated gpu__time_duration.",
    )
    args = parser.parse_args()

    if args.csv:
        raw = Path(args.binary).read_text()
        df = parse_csv(raw)
        title = Path(args.binary).stem
    else:
        df = run_ncu(args.binary, args.kernel_args.split())
        title = Path(args.binary).name
        if args.save_csv:
            # Re-export the parsed data
            df.to_csv(args.save_csv, index=False)
            print(f"✓  CSV saved: {args.save_csv}", file=sys.stderr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or f"profiler/output/{title}_{stamp}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    build_dashboard(df, title, out, bench_gflops=args.benchmark_gflops)


if __name__ == "__main__":
    main()
