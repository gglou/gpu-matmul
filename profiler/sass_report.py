#!/usr/bin/env python3
"""
sass_report.py — SASS assembly analysis dashboard for GPU matmul kernels.

Extracts the native GPU assembly (SASS) from a compiled binary using
`cuobjdump`, then parses it to produce a visual report covering:

  ① Instruction count & code size per kernel
  ② Instruction mix (FP compute, integer/address, memory, control)
  ③ Loop detection (back-edge branches → unrolled vs. looped code)
  ④ Compute density & FFMA efficiency
  ⑤ Register and shared-memory usage

This directly answers the question "did constexpr / template parameters
help the compiler unroll my loops?" — fewer back-edge branches and more
straight-line FFMA sequences = better unrolling.

Usage:
    just sass-report warptiling
    python3 profiler/sass_report.py ./run_warptiling
    python3 profiler/sass_report.py ./run_warptiling -o profiler/output/warptiling_sass.png

Requires: matplotlib numpy
    pip install matplotlib numpy
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
import matplotlib.ticker as ticker
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SASS instruction classification
# ═══════════════════════════════════════════════════════════════════════════════

# Broad categories — every SASS opcode maps to one of these.
# Categorization based on NVIDIA SASS ISA documentation.
OPCODE_CATEGORIES = OrderedDict([
    ("FP32 FMA",        ["FFMA"]),
    ("FP32 Mul",        ["FMUL"]),
    ("FP32 Add",        ["FADD"]),
    ("FP32 Other",      ["FABS", "FNEG", "FCHK", "FMNMX", "FSEL", "FSET", "FSETP",
                          "FCMP", "FRND", "MUFU"]),
    ("FP16 / BF16",     ["HFMA2", "HMUL2", "HADD2", "HSET2", "HSETP2",
                          "HMNMX2", "HFMA2_MMA"]),
    ("FP64",            ["DFMA", "DMUL", "DADD", "DMNMX", "DSETP"]),
    ("Tensor Core",     ["HMMA", "IMMA", "DMMA", "MMA"]),
    ("Integer Arith",   ["IADD", "IADD3", "IADD.64", "IMAD", "IMAD.WIDE",
                          "IMAD.HI", "IMAD.IADD", "ISUB", "IMUL",
                          "ISETP", "ISET", "ICMP", "IMNMX",
                          "IABS", "INEG"]),
    ("Uniform Int",     ["UIADD3", "UIMAD", "UISETP", "UISETP.GE.AND",
                          "UISETP.NE.AND", "UISETP.LT.AND"]),
    ("Bit / Logic",     ["LOP3.LUT", "LOP", "SHF", "SHF.L.U32", "SHF.R.S32.HI",
                          "SHF.R.U32.HI", "SHR", "SHL", "BFI", "BFE", "FLO",
                          "POPC", "BMSK", "BREV",
                          "USHF.L.U32", "USHF.R.U32.HI"]),
    ("Address / LEA",   ["LEA", "LEA.HI", "LEA.HI.X", "ULEA"]),
    ("Conversion",      ["I2F", "F2I", "I2I", "F2F", "I2IP", "F2FP"]),
    ("Move / Select",   ["MOV", "MOVM", "SEL", "SHFL", "PRMT", "SGXT",
                          "UMOV", "UPRMT", "USGXT", "UBREV", "UPOPC"]),
    ("Load Global",     ["LDG", "LDG.E", "LDG.E.128", "LDG.E.64",
                          "LDG.E.SYS", "LDG.E.128.SYS"]),
    ("Store Global",    ["STG", "STG.E", "STG.E.128", "STG.E.64",
                          "STG.E.SYS", "STG.E.128.SYS"]),
    ("Load Shared",     ["LDS", "LDS.64", "LDS.128", "LDS.U", "LDS.U.128"]),
    ("Store Shared",    ["STS", "STS.64", "STS.128", "STS.U", "STS.U.128"]),
    ("Load Const",      ["LDC", "LDC.64", "LDCU", "LDCU.64"]),
    ("Load Special",    ["S2R", "CS2R", "S2UR"]),
    ("Branch",          ["BRA", "BRA.U", "BRX", "JMP", "JMX", "CALL", "RET",
                          "BREAK", "CONT", "SSY", "PBK"]),
    ("Barrier / Sync",  ["BAR", "BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING",
                          "BAR.RED", "BSYNC", "BSSY", "YIELD", "NANOSLEEP",
                          "WARPSYNC"]),
    ("Predicate",       ["PLOP3.LUT", "UPLOP3.LUT", "CSET", "CSETP",
                          "P2R", "R2P"]),
    ("Misc / NOP",      ["NOP", "EXIT", "DEPBAR", "ERRBAR", "MEMBAR",
                          "VOTEU", "VOTE", "REDUX"]),
])

# Build reverse lookup: opcode → category
_OPCODE_TO_CAT: dict[str, str] = {}
for cat, opcodes in OPCODE_CATEGORIES.items():
    for op in opcodes:
        _OPCODE_TO_CAT[op] = cat


def classify_opcode(opcode: str) -> str:
    """Map a SASS opcode to a broad category."""
    if opcode in _OPCODE_TO_CAT:
        return _OPCODE_TO_CAT[opcode]
    base = opcode.split(".")[0]
    if base in _OPCODE_TO_CAT:
        return _OPCODE_TO_CAT[base]
    for cat, opcodes in OPCODE_CATEGORIES.items():
        for op in opcodes:
            if opcode.startswith(op):
                return cat
    return "Unknown"


# Higher-level groupings for the summary pie chart
SUMMARY_GROUPS = OrderedDict([
    ("FP32 Compute",  ["FP32 FMA", "FP32 Mul", "FP32 Add", "FP32 Other"]),
    ("FP16 / BF16",   ["FP16 / BF16"]),
    ("FP64",          ["FP64"]),
    ("Tensor Core",   ["Tensor Core"]),
    ("Integer",       ["Integer Arith", "Uniform Int", "Bit / Logic",
                       "Address / LEA", "Conversion"]),
    ("Data Movement", ["Move / Select", "Load Const", "Load Special"]),
    ("Global Memory", ["Load Global", "Store Global"]),
    ("Shared Memory", ["Load Shared", "Store Shared"]),
    ("Control Flow",  ["Branch", "Barrier / Sync", "Predicate"]),
    ("Misc / NOP",    ["Misc / NOP"]),
])


def _group_counts(detail: dict[str, int]) -> dict[str, int]:
    """Roll up detailed category counts into summary groups."""
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
class BranchInfo:
    src_addr: int
    dst_addr: int
    opcode: str
    is_back_edge: bool   # dst <= src → likely a loop


@dataclass
class KernelSASS:
    """Parsed SASS for one kernel function."""
    mangled_name: str
    demangled_name: str
    instructions: list[tuple[int, str]]          # (addr, opcode)
    opcode_counts: Counter                        # opcode → count
    category_counts: dict[str, int]               # category → count
    branches: list[BranchInfo]
    # From --dump-resource-usage
    registers: int = 0
    stack: int = 0
    shared_mem: int = 0
    local_mem: int = 0
    constant_mem: int = 0

    @property
    def total_instructions(self) -> int:
        return len(self.instructions)

    @property
    def code_size_bytes(self) -> int:
        """Approximate code size from last instruction address."""
        if not self.instructions:
            return 0
        return self.instructions[-1][0] + 0x10  # each instruction is 16 bytes

    @property
    def num_back_edges(self) -> int:
        return sum(1 for b in self.branches if b.is_back_edge)

    @property
    def num_forward_branches(self) -> int:
        return sum(1 for b in self.branches if not b.is_back_edge)

    @property
    def ffma_count(self) -> int:
        return self.opcode_counts.get("FFMA", 0)

    @property
    def compute_density(self) -> float:
        """Fraction of instructions that are FP32 FMA (the workhorse op)."""
        if self.total_instructions == 0:
            return 0.0
        return self.ffma_count / self.total_instructions

    @property
    def fp_compute_count(self) -> int:
        return sum(self.category_counts.get(c, 0)
                   for c in ["FP32 FMA", "FP32 Mul", "FP32 Add", "FP32 Other",
                              "FP16 / BF16", "FP64", "Tensor Core"])

    @property
    def memory_count(self) -> int:
        return sum(self.category_counts.get(c, 0)
                   for c in ["Load Global", "Store Global",
                              "Load Shared", "Store Shared"])

    @property
    def short_name(self) -> str:
        """A readable short name derived from the demangled name."""
        name = self.demangled_name
        m = re.search(r"(\w+)<(.+?)>", name)
        if m:
            func = m.group(1)
            params = m.group(2)
            nums = re.findall(r"\d+", params)
            return f"{func}<{','.join(nums)}>"
        m = re.match(r"[\w:]+?(\w+)\(", name)
        if m:
            return m.group(1)
        return name[:60]


# ═══════════════════════════════════════════════════════════════════════════════
# SASS extraction & parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _demangle(name: str) -> str:
    """Demangle a C++ symbol using c++filt."""
    try:
        r = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else name
    except Exception:
        return name


def extract_sass(binary: str) -> str:
    """Run cuobjdump --dump-sass on a binary."""
    cmd = ["cuobjdump", "--dump-sass", binary]
    print(f"▶  {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR — cuobjdump failed:\n{r.stderr[:2000]}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def extract_resource_usage(binary: str) -> dict[str, dict[str, int]]:
    """Run cuobjdump --dump-resource-usage and parse per-function resources."""
    cmd = ["cuobjdump", "--dump-resource-usage", binary]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}

    resources: dict[str, dict[str, int]] = {}
    func_re = re.compile(r"Function (\S+):")
    res_re = re.compile(
        r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)\s+CONSTANT\[0\]:(\d+)"
    )

    current_func = None
    for line in r.stdout.splitlines():
        m = func_re.search(line)
        if m:
            current_func = m.group(1)
            continue
        m = res_re.search(line)
        if m and current_func:
            resources[current_func] = {
                "registers": int(m.group(1)),
                "stack": int(m.group(2)),
                "shared_mem": int(m.group(3)),
                "local_mem": int(m.group(4)),
                "constant_mem": int(m.group(5)),
            }
            current_func = None
    return resources


def parse_sass(sass_text: str, resource_map: dict[str, dict[str, int]] | None = None
               ) -> list[KernelSASS]:
    """Parse cuobjdump --dump-sass output into KernelSASS objects."""
    if resource_map is None:
        resource_map = {}

    kernels: list[KernelSASS] = []
    lines = sass_text.splitlines()

    func_re = re.compile(r"Function\s*:\s*(\S+)")
    inst_re = re.compile(r"/\*([0-9a-f]+)\*/\s+(@\S+\s+)?(\S+)")
    bra_re = re.compile(r"/\*([0-9a-f]+)\*/\s+(@\S+\s+)?(BRA\S*)\s+.*?0x([0-9a-f]+)")

    i = 0
    while i < len(lines):
        m = func_re.search(lines[i])
        if not m:
            i += 1
            continue

        mangled = m.group(1)
        demangled = _demangle(mangled)
        i += 1

        instructions: list[tuple[int, str]] = []
        opcode_counts: Counter = Counter()
        branches: list[BranchInfo] = []

        while i < len(lines):
            if func_re.search(lines[i]):
                break
            line = lines[i]
            i += 1

            im = inst_re.search(line)
            if not im:
                continue
            addr = int(im.group(1), 16)
            opcode = im.group(3)

            if opcode.startswith("0x") or opcode.startswith("/*"):
                continue

            instructions.append((addr, opcode))
            opcode_counts[opcode] += 1

            bm = bra_re.search(line)
            if bm:
                src = int(bm.group(1), 16)
                dst = int(bm.group(4), 16)
                branches.append(BranchInfo(
                    src_addr=src, dst_addr=dst,
                    opcode=bm.group(3),
                    is_back_edge=(dst <= src),
                ))

        category_counts: dict[str, int] = {}
        for op, count in opcode_counts.items():
            cat = classify_opcode(op)
            category_counts[cat] = category_counts.get(cat, 0) + count

        res = resource_map.get(mangled, {})

        kernels.append(KernelSASS(
            mangled_name=mangled,
            demangled_name=demangled,
            instructions=instructions,
            opcode_counts=opcode_counts,
            category_counts=category_counts,
            branches=branches,
            registers=res.get("registers", 0),
            stack=res.get("stack", 0),
            shared_mem=res.get("shared_mem", 0),
            local_mem=res.get("local_mem", 0),
            constant_mem=res.get("constant_mem", 0),
        ))

    return kernels


def load_kernel(binary: str) -> KernelSASS:
    """Full pipeline: extract SASS + resources, parse, return the first kernel."""
    sass_text = extract_sass(binary)
    resource_map = extract_resource_usage(binary)
    kernels = parse_sass(sass_text, resource_map)
    if not kernels:
        print("ERROR — no kernels found in binary.", file=sys.stderr)
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
    "FP16 / BF16":   C_CYAN,
    "FP64":          "#0D47A1",
    "Tensor Core":   C_GREEN,
    "Integer":       C_RED,
    "Data Movement": C_AMBER,
    "Global Memory": C_ORANGE,
    "Shared Memory": C_LGREEN,
    "Control Flow":  C_PURPLE,
    "Misc / NOP":    C_LGRAY,
    "Unknown":       C_GRAY,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate_hbar(ax, bars, values, fmt="{:.0f}", fontsize=8):
    """Add value labels at the end of horizontal bars."""
    xlim = ax.get_xlim()[1]
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                val + xlim * 0.015,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", fontsize=fontsize,
            )


# ── Panel 1: Summary table ──────────────────────────────────────────────────

def plot_summary_table(ax: plt.Axes, kernel: KernelSASS):
    """Key scalar metrics in a clean table layout."""
    ax.axis("off")
    ax.set_title("① Kernel Summary", fontweight="bold", fontsize=10, loc="left")

    rows = [
        ("Kernel",               kernel.short_name),
        ("Total Instructions",   f"{kernel.total_instructions:,}"),
        ("Code Size",            f"{kernel.code_size_bytes:,} bytes  "
                                 f"({kernel.code_size_bytes / 1024:.1f} KB)"),
        ("Registers / Thread",   f"{kernel.registers}"),
        ("Shared Memory",        f"{kernel.shared_mem:,} bytes  "
                                 f"({kernel.shared_mem / 1024:.1f} KB)"),
        ("Stack / Local",        f"{kernel.stack} / {kernel.local_mem} bytes"),
        ("",                     ""),
        ("FFMA Count",           f"{kernel.ffma_count:,}"),
        ("FP Compute Insts",     f"{kernel.fp_compute_count:,}  "
                                 f"({100 * kernel.fp_compute_count / max(kernel.total_instructions, 1):.1f}%)"),
        ("Memory Insts",         f"{kernel.memory_count:,}  "
                                 f"({100 * kernel.memory_count / max(kernel.total_instructions, 1):.1f}%)"),
        ("Compute Density",      f"{kernel.compute_density:.1%}  (FFMA / total)"),
        ("",                     ""),
        ("Back-edge Branches",   f"{kernel.num_back_edges}  "
                                 f"({'⚠ loops present' if kernel.num_back_edges > 1 else '✓ likely fully unrolled'})"),
        ("Forward Branches",     f"{kernel.num_forward_branches}"),
        ("Total Branches",       f"{len(kernel.branches)}"),
    ]

    y = 0.95
    for label, value in rows:
        if not label and not value:
            y -= 0.03
            continue
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=8,
                fontfamily="monospace", fontweight="bold", va="top")
        ax.text(0.42, y, str(value), transform=ax.transAxes, fontsize=8,
                fontfamily="monospace", va="top")
        y -= 0.065


# ── Panel 2: Instruction mix (detailed bar chart) ────────────────────────────

def plot_instruction_mix_bar(ax: plt.Axes, kernel: KernelSASS):
    """Horizontal bar chart of detailed instruction categories."""
    cats = [(cat, kernel.category_counts.get(cat, 0))
            for cat in OPCODE_CATEGORIES.keys()]
    cats = [(c, v) for c, v in cats if v > 0]
    if not cats:
        ax.text(0.5, 0.5, "No instruction data", ha="center", va="center",
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


# ── Panel 3: Instruction mix (grouped pie) ──────────────────────────────────

def plot_instruction_mix_pie(ax: plt.Axes, kernel: KernelSASS):
    """Pie chart of high-level instruction groups."""
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
    for t in texts:
        t.set_fontsize(7)
    for t in autotexts:
        t.set_fontsize(7)
        t.set_fontweight("bold")

    ax.set_title("③ Instruction Groups", fontweight="bold", fontsize=10)
    total = sum(values)
    ax.text(0, -1.35, f"Total: {total:,} instructions",
            ha="center", fontsize=7, color=C_GRAY)


# ── Panel 4: Top opcodes ────────────────────────────────────────────────────

def plot_top_opcodes(ax: plt.Axes, kernel: KernelSASS, top_n: int = 15):
    """Bar chart of the most frequent individual opcodes."""
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
        "Integer Arith": "#EF5350", "Uniform Int": "#EF9A9A",
        "Bit / Logic": "#FFAB91", "Address / LEA": "#FFE082",
        "Branch": C_PURPLE, "Barrier / Sync": C_LPURPLE,
        "Load Special": C_AMBER, "Load Const": "#FFE0B2",
    }
    colors = [cat_colors.get(classify_opcode(op), C_LGRAY) for op in opcodes]

    bars = ax.barh(opcodes, counts, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title("④ Top Opcodes", fontweight="bold", fontsize=10)
    _annotate_hbar(ax, bars, counts, fontsize=7)


# ── Panel 5: Loop / unroll analysis ──────────────────────────────────────────

def plot_loop_analysis(ax: plt.Axes, kernel: KernelSASS):
    """Visualize branch targets to show loop structure and unrolling."""
    ax.set_title("⑤ Code Structure & Loops", fontweight="bold", fontsize=10)

    if not kernel.instructions:
        ax.text(0.5, 0.5, "No instructions", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=C_GRAY)
        return

    addrs = [a for a, _ in kernel.instructions]
    max_addr = max(addrs) if addrs else 1

    bucket_size = 0x40  # 64 bytes = 4 instructions
    n_buckets = (max_addr // bucket_size) + 1
    density = np.zeros(n_buckets)
    ffma_density = np.zeros(n_buckets)
    mem_density = np.zeros(n_buckets)

    for addr, opcode in kernel.instructions:
        bucket = addr // bucket_size
        if bucket < n_buckets:
            density[bucket] += 1
            if opcode == "FFMA":
                ffma_density[bucket] += 1
            elif classify_opcode(opcode) in ("Load Global", "Store Global",
                                              "Load Shared", "Store Shared"):
                mem_density[bucket] += 1

    x = np.arange(n_buckets)
    ax.fill_between(x, ffma_density, alpha=0.7, color=C_BLUE, label="FFMA", step="mid")
    ax.fill_between(x, mem_density, alpha=0.5, color=C_GREEN, label="Memory", step="mid")
    ax.plot(x, density, color=C_GRAY, alpha=0.4, lw=0.5, label="All")

    for br in kernel.branches:
        if br.is_back_edge:
            src_bucket = br.src_addr // bucket_size
            dst_bucket = br.dst_addr // bucket_size
            if src_bucket < n_buckets and dst_bucket < n_buckets:
                ax.annotate("",
                    xy=(dst_bucket, density[dst_bucket] if dst_bucket < n_buckets else 0),
                    xytext=(src_bucket, density[src_bucket] if src_bucket < n_buckets else 0),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
                )

    ax.set_xlabel(f"Code Position (×{bucket_size} bytes)")
    ax.set_ylabel("Instructions per bucket")
    ax.legend(fontsize=7, loc="upper right")

    real_loops = sum(1 for b in kernel.branches
                     if b.is_back_edge and b.src_addr != b.dst_addr)
    note = (f"{real_loops} loop back-edge(s) detected"
            if real_loops > 0
            else "No loops — code is fully unrolled ✓")
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top",
            color=C_RED if real_loops > 1 else C_GREEN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_LGRAY, alpha=0.9))


# ── Panel 6: Efficiency metrics ──────────────────────────────────────────────

def plot_efficiency(ax: plt.Axes, kernel: KernelSASS):
    """Bar chart of key efficiency ratios."""
    ax.set_title("⑥ Efficiency Metrics", fontweight="bold", fontsize=10)

    total = max(kernel.total_instructions, 1)
    metrics = OrderedDict([
        ("FFMA Density\n(FFMA / total)",
         100 * kernel.ffma_count / total),
        ("FP Compute\n(all FP / total)",
         100 * kernel.fp_compute_count / total),
        ("Memory Ops\n(LD+ST / total)",
         100 * kernel.memory_count / total),
        ("Address Math\n(LEA+IMAD / total)",
         100 * sum(kernel.category_counts.get(c, 0)
                   for c in ["Address / LEA", "Integer Arith", "Uniform Int"]) / total),
        ("Control / Sync\n(BRA+BAR / total)",
         100 * sum(kernel.category_counts.get(c, 0)
                   for c in ["Branch", "Barrier / Sync"]) / total),
        ("NOP / Misc\n(NOP+EXIT / total)",
         100 * kernel.category_counts.get("Misc / NOP", 0) / total),
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

def build_dashboard(kernel: KernelSASS, title: str, output: str):
    """6-panel SASS analysis dashboard for one kernel."""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"SASS Assembly Report  —  {title}",
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
        description="SASS assembly analysis dashboard for GPU kernels",
        epilog="Example: python3 profiler/sass_report.py ./run_warptiling",
    )
    parser.add_argument("binary", help="Compiled CUDA binary to analyse")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path (default: profiler/output/<name>_sass.png)")
    args = parser.parse_args()

    name = Path(args.binary).stem.removeprefix("run_")
    kernel = load_kernel(args.binary)

    out_dir = Path("profiler/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or str(out_dir / f"{name}_sass.png")
    build_dashboard(kernel, name, out)


if __name__ == "__main__":
    main()
