#!/usr/bin/env python3
"""
ptx_report.py — PTX IR analysis dashboard for GPU matmul kernels.

Parses the PTX intermediate representation produced by nvcc --ptx and generates
a visual report covering:

  ① Kernel summary (register file, shared memory, instruction count)
  ② Instruction mix — detailed opcode categories
  ③ Instruction groups — high-level pie chart
  ④ Top opcodes by frequency
  ⑤ Code structure & loop detection (back-edge branches)
  ⑥ Efficiency metrics (FMA density, memory ratio, etc.)

PTX is one level above SASS — it's the virtual ISA that NVVM emits before the
hardware assembler runs. Comparing PTX vs SASS shows what the backend compiler
changed (e.g., register allocation, instruction scheduling, FMA merging).

Usage:
    just ptx-report warptiling
    python3 profiler/ptx_report.py inspect/run_warptiling.sm120.ptx
    python3 profiler/ptx_report.py inspect/run_warptiling.sm120.ptx -o out.png

Requires: matplotlib numpy
"""

import argparse
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# PTX instruction classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_ptx_opcode(opcode: str) -> str:
    """Classify a PTX opcode string into a broad category."""
    parts = opcode.split(".")
    base = parts[0]

    # Memory — check space qualifier before type
    if base in ("ld", "ldu"):
        if "global" in parts or "param" not in parts and "shared" not in parts:
            if "shared" in parts:   return "Load Shared"
            if "const" in parts:    return "Load Const"
            if "param" in parts:    return "Load Param"
            return "Load Global"
        if "shared" in parts:       return "Load Shared"
        if "const" in parts:        return "Load Const"
        if "param" in parts:        return "Load Param"
        return "Load Global"
    if base == "st":
        if "global" in parts:       return "Store Global"
        if "shared" in parts:       return "Store Shared"
        if "param" in parts:        return "Store Param"
        return "Store Global"
    if base == "prefetch":          return "Load Global"
    if base in ("cp",):
        if "async" in parts:        return "Async Copy"
        return "Load Global"

    # FP compute — check type qualifier
    if base == "fma":
        if "f64" in parts:          return "FP64"
        return "FP32 FMA"
    if base == "mul":
        if "f64" in parts:          return "FP64"
        if "f32" in parts:          return "FP32 Mul"
        return "Integer Arith"
    if base == "add":
        if "f64" in parts:          return "FP64"
        if "f32" in parts:          return "FP32 Add"
        return "Integer Arith"
    if base == "sub":
        if "f64" in parts:          return "FP64"
        if "f32" in parts:          return "FP32 Add"
        return "Integer Arith"
    if base in ("max", "min", "abs", "neg", "rcp", "rsqrt", "sqrt",
                "sin", "cos", "lg2", "ex2"):
        if "f64" in parts:          return "FP64"
        if "f32" in parts:          return "FP32 Other"
        return "Integer Arith"

    # Integer arithmetic
    if base in ("mad", "madc", "sad"):  return "Integer Arith"
    if base in ("div", "rem", "bfe", "bfi", "clz", "popc", "brev"):
        return "Integer Arith"
    if base == "mul" and "wide" in parts:   return "Integer Arith"

    # Bit / logic
    if base in ("and", "or", "xor", "not", "cnot"):    return "Bit / Logic"
    if base in ("shl", "shr", "shf"):                  return "Bit / Logic"

    # Address / LEA
    if base in ("mov",):
        # mov of u64 addresses is effectively LEA; count as Move
        return "Move / Select"
    if base in ("lea",):                return "Address / LEA"

    # Conversion
    if base == "cvt":                   return "Conversion"
    if base == "cvta":                  return "Conversion"

    # Move / select
    if base in ("selp", "slct"):        return "Move / Select"

    # Predicate / compare
    if base == "setp":                  return "Predicate"
    if base in ("set",):                return "Predicate"
    if base in ("testp",):              return "Predicate"

    # Control flow
    if base in ("bra", "brx", "call", "ret", "exit"):  return "Branch"

    # Barrier / sync
    if base in ("bar", "membar", "fence"):              return "Barrier / Sync"
    if base == "cp" and "async" in parts:               return "Async Copy"
    if base in ("cp",):                                 return "Barrier / Sync"

    # Special registers / misc
    if base in ("atom", "red"):         return "Atomic"
    if base in ("shfl", "vote", "match", "redux"):      return "Warp Ops"
    if base in ("tex", "tld4", "txq"): return "Texture"
    if base in ("suld", "sust"):        return "Surface"
    if base == "trap":                  return "Misc / NOP"

    return "Unknown"


SUMMARY_GROUPS = OrderedDict([
    ("FP32 Compute",  ["FP32 FMA", "FP32 Mul", "FP32 Add", "FP32 Other"]),
    ("FP64",          ["FP64"]),
    ("Integer",       ["Integer Arith", "Bit / Logic", "Address / LEA", "Conversion"]),
    ("Data Movement", ["Move / Select", "Load Param", "Load Const"]),
    ("Global Memory", ["Load Global", "Store Global"]),
    ("Shared Memory", ["Load Shared", "Store Shared"]),
    ("Async Copy",    ["Async Copy"]),
    ("Control Flow",  ["Branch", "Barrier / Sync", "Predicate"]),
    ("Misc / NOP",    ["Misc / NOP", "Atomic", "Warp Ops"]),
])


def _group_counts(detail: dict[str, int]) -> dict[str, int]:
    result = OrderedDict()
    for group, cats in SUMMARY_GROUPS.items():
        total = sum(detail.get(c, 0) for c in cats)
        if total > 0:
            result[group] = total
    unk = detail.get("Unknown", 0)
    if unk > 0:
        result["Unknown"] = unk
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegFile:
    pred:  int = 0   # .pred
    b16:   int = 0   # .b16
    b32:   int = 0   # .b32
    b64:   int = 0   # .b64
    f32:   int = 0   # .f32
    f64:   int = 0   # .f64

    @property
    def total_virtual(self) -> int:
        return self.pred + self.b16 + self.b32 + self.b64 + self.f32 + self.f64


@dataclass
class BranchInfo:
    line_idx: int
    target: str
    target_line: int   # -1 if not resolved
    is_back_edge: bool


@dataclass
class KernelPTX:
    mangled_name: str
    demangled_name: str
    instructions: list[tuple[int, str]]   # (line_idx, opcode)
    opcode_counts: Counter
    category_counts: dict[str, int]
    branches: list[BranchInfo]
    reg_file: RegFile
    shared_bytes: int = 0
    maxntid: int = 0

    @property
    def total_instructions(self) -> int:
        return len(self.instructions)

    @property
    def fma_count(self) -> int:
        return self.opcode_counts.get("fma.rn.f32", 0) + self.opcode_counts.get("fma.rz.f32", 0)

    @property
    def compute_density(self) -> float:
        if self.total_instructions == 0:
            return 0.0
        return self.fma_count / self.total_instructions

    @property
    def fp_compute_count(self) -> int:
        return sum(self.category_counts.get(c, 0)
                   for c in ["FP32 FMA", "FP32 Mul", "FP32 Add", "FP32 Other", "FP64"])

    @property
    def memory_count(self) -> int:
        return sum(self.category_counts.get(c, 0)
                   for c in ["Load Global", "Store Global", "Load Shared", "Store Shared",
                              "Async Copy"])

    @property
    def num_back_edges(self) -> int:
        return sum(1 for b in self.branches if b.is_back_edge)

    @property
    def num_forward_branches(self) -> int:
        return sum(1 for b in self.branches if not b.is_back_edge)

    @property
    def short_name(self) -> str:
        name = self.demangled_name
        m = re.search(r"(\w+)<(.+?)>", name)
        if m:
            func = m.group(1)
            nums = re.findall(r"\d+", m.group(2))
            return f"{func}<{','.join(nums)}>"
        m = re.match(r"[\w:]+?(\w+)\(", name)
        if m:
            return m.group(1)
        return name[:60]


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _demangle(name: str) -> str:
    try:
        r = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else name
    except Exception:
        return name


def parse_ptx(ptx_text: str) -> list[KernelPTX]:
    """Parse a PTX file into KernelPTX objects (one per .entry)."""
    lines = ptx_text.splitlines()
    kernels: list[KernelPTX] = []

    entry_re  = re.compile(r"\.(?:visible\s+)?\.entry\s+(\S+)\s*\(")
    reg_re    = re.compile(r"\.reg\s+(\.\w+)\s+%\w+<(\d+)>")
    shared_re = re.compile(r"\.shared\s+.*\.b8\s+\S+\[(\d+)\]")
    maxntid_re= re.compile(r"\.maxntid\s+(\d+)")
    inst_re   = re.compile(r"^\s+(?:@%p\S*\s+)?([a-z]\S*)\s")
    label_re  = re.compile(r"^\s*(\$\S+|BB\S+|[A-Za-z_]\w*):")
    bra_re    = re.compile(r"^\s+(?:@%p\S*\s+)?(bra\S*)\s+(\S+?);")

    i = 0
    while i < len(lines):
        m = entry_re.search(lines[i])
        if not m:
            i += 1
            continue

        mangled = m.group(1).rstrip("(").strip()
        demangled = _demangle(mangled)
        i += 1

        reg_file = RegFile()
        shared_bytes = 0
        maxntid = 0
        instructions: list[tuple[int, str]] = []
        opcode_counts: Counter = Counter()
        branches: list[BranchInfo] = []
        label_lines: dict[str, int] = {}   # label → line index

        depth = 0
        body_started = False

        while i < len(lines):
            line = lines[i]

            # Track brace depth to know when the function body ends
            depth += line.count("{") - line.count("}")
            if body_started and depth <= 0:
                i += 1
                break
            if "{" in line:
                body_started = True

            # Register declarations
            rm = reg_re.search(line)
            if rm:
                typ, count = rm.group(1), int(rm.group(2))
                if typ == ".pred":  reg_file.pred  += count
                elif typ == ".b16": reg_file.b16   += count
                elif typ == ".b32": reg_file.b32   += count
                elif typ == ".b64": reg_file.b64   += count
                elif typ == ".f32": reg_file.f32   += count
                elif typ == ".f64": reg_file.f64   += count

            # Shared memory
            sm = shared_re.search(line)
            if sm:
                shared_bytes += int(sm.group(1))

            # maxntid
            nt = maxntid_re.search(line)
            if nt:
                maxntid = int(nt.group(1))

            # Labels
            lm = label_re.match(line)
            if lm and ":" in line and not line.strip().startswith("//"):
                label_lines[lm.group(1)] = i

            # Branch instructions
            bm = bra_re.search(line)
            if bm:
                target = bm.group(2).rstrip(";").strip()
                branches.append(BranchInfo(
                    line_idx=i, target=target,
                    target_line=-1, is_back_edge=False,
                ))

            # Generic instructions
            im = inst_re.match(line)
            if im and not line.strip().startswith("//") and ":" not in line.split("//")[0]:
                opcode = im.group(1).rstrip(";").strip()
                # skip pseudo-directives
                if not opcode.startswith(".") and not opcode.startswith("//"):
                    instructions.append((i, opcode))
                    opcode_counts[opcode] += 1

            i += 1

        # Resolve branch back-edges (target label appears before branch)
        for br in branches:
            tgt_line = label_lines.get(br.target, -1)
            br.target_line = tgt_line
            br.is_back_edge = (tgt_line != -1 and tgt_line < br.line_idx)

        # Build category counts
        category_counts: dict[str, int] = {}
        for op, count in opcode_counts.items():
            cat = classify_ptx_opcode(op)
            category_counts[cat] = category_counts.get(cat, 0) + count

        kernels.append(KernelPTX(
            mangled_name=mangled,
            demangled_name=demangled,
            instructions=instructions,
            opcode_counts=opcode_counts,
            category_counts=category_counts,
            branches=branches,
            reg_file=reg_file,
            shared_bytes=shared_bytes,
            maxntid=maxntid,
        ))

    return kernels


def load_kernel(ptx_path: str) -> KernelPTX:
    """Parse PTX file and return the first kernel."""
    text = Path(ptx_path).read_text()
    kernels = parse_ptx(text)
    if not kernels:
        print(f"ERROR — no .entry found in {ptx_path}", file=sys.stderr)
        sys.exit(1)
    if len(kernels) > 1:
        print(f"   {len(kernels)} kernels found — using first: {kernels[0].short_name}",
              file=sys.stderr)
    return kernels[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════════════

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

GROUP_COLORS = {
    "FP32 Compute":  C_BLUE,
    "FP64":          "#0D47A1",
    "Integer":       C_RED,
    "Data Movement": C_AMBER,
    "Global Memory": C_ORANGE,
    "Shared Memory": C_LGREEN,
    "Async Copy":    C_CYAN,
    "Control Flow":  C_PURPLE,
    "Misc / NOP":    C_LGRAY,
    "Unknown":       C_GRAY,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate_hbar(ax, bars, values, fmt="{:.0f}", fontsize=8):
    xlim = ax.get_xlim()[1]
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(val + xlim * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=fontsize)


# ── Panel 1: Summary table ──────────────────────────────────────────────────

def plot_summary_table(ax, kernel: KernelPTX):
    ax.axis("off")
    ax.set_title("① Kernel Summary", fontweight="bold", fontsize=10, loc="left")

    rf = kernel.reg_file
    real_loops = sum(1 for b in kernel.branches
                     if b.is_back_edge and b.line_idx != b.target_line)

    rows = [
        ("Kernel",             kernel.short_name),
        ("Total Instructions", f"{kernel.total_instructions:,}"),
        ("Max Threads / Block",f"{kernel.maxntid}  (from .maxntid)"),
        ("Shared Memory",      f"{kernel.shared_bytes:,} bytes  "
                               f"({kernel.shared_bytes / 1024:.1f} KB)"),
        ("",                   ""),
        ("Register File",      "virtual count"),
        ("  .pred",            f"{rf.pred}"),
        ("  .b32  (int)",      f"{rf.b32}"),
        ("  .b64  (addr/int)", f"{rf.b64}"),
        ("  .f32  (fp32)",     f"{rf.f32}"),
        ("  .f64  (fp64)",     f"{rf.f64}"),
        ("  Total virtual",    f"{rf.total_virtual}"),
        ("",                   ""),
        ("FMA Count",          f"{kernel.fma_count:,}"),
        ("FP Compute",         f"{kernel.fp_compute_count:,}  "
                               f"({100 * kernel.fp_compute_count / max(kernel.total_instructions, 1):.1f}%)"),
        ("Memory Ops",         f"{kernel.memory_count:,}  "
                               f"({100 * kernel.memory_count / max(kernel.total_instructions, 1):.1f}%)"),
        ("FMA Density",        f"{kernel.compute_density:.1%}  (fma.f32 / total)"),
        ("",                   ""),
        ("Back-edge Branches", f"{real_loops}  "
                               f"({'⚠ loops present' if real_loops > 1 else '✓ likely fully unrolled'})"),
        ("Forward Branches",   f"{kernel.num_forward_branches}"),
    ]

    y = 0.97
    for label, value in rows:
        if not label and not value:
            y -= 0.025
            continue
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=7.5,
                fontfamily="monospace", fontweight="bold", va="top")
        ax.text(0.45, y, str(value), transform=ax.transAxes, fontsize=7.5,
                fontfamily="monospace", va="top")
        y -= 0.055


# ── Panel 2: Instruction mix (detailed bar) ──────────────────────────────────

def plot_instruction_mix_bar(ax, kernel: KernelPTX):
    cats = [(cat, kernel.category_counts.get(cat, 0))
            for cat in [
                "FP32 FMA", "FP32 Mul", "FP32 Add", "FP32 Other",
                "FP64", "Integer Arith", "Bit / Logic", "Address / LEA",
                "Conversion", "Move / Select", "Load Global", "Store Global",
                "Load Shared", "Store Shared", "Async Copy",
                "Load Param", "Load Const", "Branch", "Predicate",
                "Barrier / Sync", "Atomic", "Warp Ops", "Misc / NOP",
            ]]
    cats = [(c, v) for c, v in cats if v > 0]
    if not cats:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("② Instruction Mix (Detailed)", fontweight="bold", fontsize=10)
        return

    labels, values = zip(*reversed(cats))
    n = len(values)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n, 2)))
    bars = ax.barh(labels, values, color=cmap[:n], height=0.7,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Instruction Count")
    ax.set_title("② Instruction Mix (Detailed)", fontweight="bold", fontsize=10)
    _annotate_hbar(ax, bars, values, fontsize=7)


# ── Panel 3: Instruction groups (pie) ────────────────────────────────────────

def plot_instruction_mix_pie(ax, kernel: KernelPTX):
    grouped = _group_counts(kernel.category_counts)
    if not grouped:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("③ Instruction Groups", fontweight="bold", fontsize=10)
        return

    labels = list(grouped.keys())
    values = list(grouped.values())
    colors = [GROUP_COLORS.get(l, C_GRAY) for l in labels]

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, pctdistance=0.78,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for t in texts:     t.set_fontsize(7)
    for t in autotexts: t.set_fontsize(7); t.set_fontweight("bold")

    ax.set_title("③ Instruction Groups", fontweight="bold", fontsize=10)
    ax.text(0, -1.35, f"Total: {sum(values):,} instructions",
            ha="center", fontsize=7, color=C_GRAY)


# ── Panel 4: Top opcodes ─────────────────────────────────────────────────────

def plot_top_opcodes(ax, kernel: KernelPTX, top_n: int = 15):
    most_common = kernel.opcode_counts.most_common(top_n)
    if not most_common:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        ax.set_title("④ Top Opcodes", fontweight="bold", fontsize=10)
        return

    opcodes, counts = zip(*reversed(most_common))

    cat_colors = {
        "FP32 FMA": C_BLUE, "FP32 Mul": C_LBLUE, "FP32 Add": C_CYAN,
        "Load Global": C_ORANGE, "Store Global": C_RED,
        "Load Shared": C_LGREEN, "Store Shared": C_GREEN,
        "Integer Arith": "#EF5350", "Bit / Logic": "#FFAB91",
        "Move / Select": C_AMBER,  "Conversion": "#FFE082",
        "Branch": C_PURPLE, "Barrier / Sync": C_LPURPLE,
        "Predicate": "#CE93D8", "Async Copy": C_CYAN,
    }
    colors = [cat_colors.get(classify_ptx_opcode(op), C_LGRAY) for op in opcodes]

    bars = ax.barh(opcodes, counts, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title("④ Top Opcodes", fontweight="bold", fontsize=10)
    _annotate_hbar(ax, bars, counts, fontsize=7)


# ── Panel 5: Code structure & loop analysis ──────────────────────────────────

def plot_loop_analysis(ax, kernel: KernelPTX):
    ax.set_title("⑤ Code Structure & Loops", fontweight="bold", fontsize=10)

    if not kernel.instructions:
        ax.text(0.5, 0.5, "No instructions", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        return

    # Use line index as position proxy
    all_lines = [ln for ln, _ in kernel.instructions]
    max_line = max(all_lines) if all_lines else 1

    bucket_size = max(1, max_line // 200)   # ~200 buckets
    n_buckets = (max_line // bucket_size) + 1
    density      = np.zeros(n_buckets)
    fma_density  = np.zeros(n_buckets)
    mem_density  = np.zeros(n_buckets)

    for ln, op in kernel.instructions:
        b = ln // bucket_size
        if b < n_buckets:
            density[b] += 1
            cat = classify_ptx_opcode(op)
            if cat == "FP32 FMA":
                fma_density[b] += 1
            elif cat in ("Load Global", "Store Global", "Load Shared",
                         "Store Shared", "Async Copy"):
                mem_density[b] += 1

    x = np.arange(n_buckets)
    ax.fill_between(x, fma_density, alpha=0.7, color=C_BLUE, label="fma.f32", step="mid")
    ax.fill_between(x, mem_density, alpha=0.5, color=C_GREEN, label="Memory",  step="mid")
    ax.plot(x, density, color=C_GRAY, alpha=0.4, lw=0.5, label="All")

    for br in kernel.branches:
        if br.is_back_edge and br.target_line != -1:
            src_b = br.line_idx  // bucket_size
            dst_b = br.target_line // bucket_size
            if src_b < n_buckets and dst_b < n_buckets:
                ax.annotate("",
                    xy=(dst_b, density[dst_b]),
                    xytext=(src_b, density[src_b]),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
                )

    ax.set_xlabel("Code Position (line buckets)")
    ax.set_ylabel("Instructions per bucket")
    ax.legend(fontsize=7, loc="upper right")

    real_loops = sum(1 for b in kernel.branches
                     if b.is_back_edge and b.line_idx != b.target_line)
    note = (f"{real_loops} loop back-edge(s) detected"
            if real_loops > 0
            else "No loops — code is fully unrolled ✓")
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top",
            color=C_RED if real_loops > 1 else C_GREEN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_LGRAY, alpha=0.9))


# ── Panel 6: Efficiency metrics ───────────────────────────────────────────────

def plot_efficiency(ax, kernel: KernelPTX):
    ax.set_title("⑥ Efficiency Metrics", fontweight="bold", fontsize=10)

    total = max(kernel.total_instructions, 1)
    metrics = OrderedDict([
        ("FMA Density\n(fma.f32 / total)",
         100 * kernel.fma_count / total),
        ("FP Compute\n(all FP / total)",
         100 * kernel.fp_compute_count / total),
        ("Memory Ops\n(LD+ST / total)",
         100 * kernel.memory_count / total),
        ("Integer / Addr\n(arith+bit+cvt / total)",
         100 * sum(kernel.category_counts.get(c, 0)
                   for c in ["Integer Arith", "Bit / Logic", "Address / LEA",
                              "Conversion"]) / total),
        ("Control / Sync\n(bra+bar / total)",
         100 * sum(kernel.category_counts.get(c, 0)
                   for c in ["Branch", "Barrier / Sync", "Predicate"]) / total),
        ("Move / Data\n(mov+selp / total)",
         100 * sum(kernel.category_counts.get(c, 0)
                   for c in ["Move / Select", "Load Param", "Load Const"]) / total),
    ])

    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [C_BLUE, C_CYAN, C_GREEN, C_AMBER, C_PURPLE, C_LGRAY]

    bars = ax.barh(labels, values, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of Total Instructions")
    _annotate_hbar(ax, bars, values, fmt="{:.1f}%", fontsize=8)
    ax.invert_yaxis()


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_dashboard(kernel: KernelPTX, title: str, output: str):
    """6-panel PTX analysis dashboard for one kernel."""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"PTX IR Report  —  {title}",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.99, 0.005, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             fontsize=7, color="#999999", ha="right", va="bottom",
             fontfamily="monospace")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.40,
                           top=0.95, bottom=0.04, left=0.08, right=0.96)

    plot_summary_table(      fig.add_subplot(gs[0, 0]), kernel)
    plot_instruction_mix_pie(fig.add_subplot(gs[0, 1]), kernel)
    plot_instruction_mix_bar(fig.add_subplot(gs[1, 0]), kernel)
    plot_top_opcodes(        fig.add_subplot(gs[1, 1]), kernel)
    plot_loop_analysis(      fig.add_subplot(gs[2, 0]), kernel)
    plot_efficiency(         fig.add_subplot(gs[2, 1]), kernel)

    plt.savefig(output, bbox_inches="tight", dpi=150)
    print(f"✓  Saved: {output}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PTX IR analysis dashboard for GPU kernels",
        epilog="Example: python3 profiler/ptx_report.py inspect/run_warptiling.sm120.ptx",
    )
    parser.add_argument("ptx_file", help="PTX file to analyse (from just ptx <kernel>)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path (default: profiler/output/<name>_ptx.png)")
    args = parser.parse_args()

    ptx_path = Path(args.ptx_file)
    if not ptx_path.exists():
        print(f"ERROR — file not found: {ptx_path}", file=sys.stderr)
        sys.exit(1)

    # Derive a clean name: "run_warptiling.sm120.ptx" → "warptiling"
    name = ptx_path.stem          # run_warptiling.sm120
    name = re.sub(r"\.sm\d+$", "", name)   # run_warptiling
    name = name.removeprefix("run_")       # warptiling

    print(f"▶  Parsing {ptx_path}", file=sys.stderr)
    kernel = load_kernel(str(ptx_path))
    print(f"   Kernel: {kernel.short_name}  ({kernel.total_instructions} instructions)",
          file=sys.stderr)

    out_dir = Path("profiler/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or str(out_dir / f"{name}_ptx.png")
    build_dashboard(kernel, name, out)


if __name__ == "__main__":
    main()
