"""
plot_comparison.py
==================
Compares RegNet measurements across batch sizes (32, 16, 8).
Run this AFTER plot_measurements.py has been run for each batch size.

Usage:
    python GPU_result/plot_comparison.py \
        --base_dir /home/slurm/comp597/students/zqiu6/regnet_measurements \
        --out_dir  ~/COMP597-starter-code-Eric/GPU_result/plots/comparison \
        --batch_sizes 32 16 8 \
        --num_runs 3 \
        --log_dirs  comp597-logs/bs32 comp597-logs/bs16 comp597-logs/bs8
"""

import argparse
import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


_STEP_LINE_RE = re.compile(
    r"step\s+(?P<step>[\d.]+)\s+--\s+"
    r"forward\s+(?P<fwd>[\d.]+)\s+--\s+"
    r"backward\s+(?P<bwd>[\d.]+)\s+--\s+"
    r"optimizer step\s+(?P<opt>[\d.]+)"
    r"(?:.*?gpu_util%\s+(?P<gpu_util>[\d.]+))?"
    r"(?:.*?gpu_mem\(MB\)\s+(?P<gpu_mem>[\d.]+))?"
    r"(?:.*?energy_step\(mJ\)\s+(?P<e_step>[\d.]+))?"
)


def parse_logs(log_files):
    rows = []
    for p in log_files:
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            for line in f:
                m = _STEP_LINE_RE.search(line)
                if m:
                    rows.append({k: float(v) for k, v in m.groupdict().items() if v is not None})
    return pd.DataFrame(rows)


def load_cc_full(base_dir, batch_size, num_runs, rank=0):
    rows = []
    d = os.path.join(base_dir, f"bs_{batch_size}")
    for run in range(num_runs):
        path = os.path.join(d, f"run_{run}_cc_full_rank_{rank}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if not df.empty:
                rows.append(df.iloc[-1])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",    required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[32, 16, 8])
    parser.add_argument("--num_runs",    type=int, default=3)
    parser.add_argument("--logs_bs32",   nargs="*", default=[],
                        help="Log files for batch size 32")
    parser.add_argument("--logs_bs16",   nargs="*", default=[],
                        help="Log files for batch size 16")
    parser.add_argument("--logs_bs8",    nargs="*", default=[],
                        help="Log files for batch size 8")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    batch_sizes = args.batch_sizes

    # Map batch size -> explicit log files
    logs_map = {32: args.logs_bs32, 16: args.logs_bs16, 8: args.logs_bs8}

    # ------------------------------------------------------------------ #
    # Load per-batch-size data                                            #
    # ------------------------------------------------------------------ #
    throughput_means, throughput_stds = [], []
    energy_per_sample_means, energy_per_sample_stds = [], []
    gpu_util_means, gpu_util_stds = [], []
    phase_means = {"forward": [], "backward": [], "optimizer": []}
    phase_stds  = {"forward": [], "backward": [], "optimizer": []}

    for bs in batch_sizes:
        log_files = logs_map.get(bs, [])
        if not log_files:
            print(f"  [warn] no log files provided for batch_size={bs}; skipping")
        df = parse_logs(log_files)

        if df.empty:
            throughput_means.append(0); throughput_stds.append(0)
            energy_per_sample_means.append(0); energy_per_sample_stds.append(0)
            gpu_util_means.append(0); gpu_util_stds.append(0)
            for p in phase_means: phase_means[p].append(0); phase_stds[p].append(0)
            continue

        # Throughput (samples/sec)
        if "step" in df.columns:
            tp = bs / (df["step"] / 1000.0).replace(0, np.nan)
            throughput_means.append(tp.mean())
            throughput_stds.append(tp.std())
        else:
            throughput_means.append(0); throughput_stds.append(0)

        # Energy per sample (mJ/sample)
        if "e_step" in df.columns:
            eps = df["e_step"] / bs
            energy_per_sample_means.append(eps.mean())
            energy_per_sample_stds.append(eps.std())
        else:
            energy_per_sample_means.append(0); energy_per_sample_stds.append(0)

        # GPU utilization
        if "gpu_util" in df.columns:
            gpu_util_means.append(df["gpu_util"].mean())
            gpu_util_stds.append(df["gpu_util"].std())
        else:
            gpu_util_means.append(0); gpu_util_stds.append(0)

        # Phase times
        for phase, col in [("forward","fwd"), ("backward","bwd"), ("optimizer","opt")]:
            if col in df.columns:
                phase_means[phase].append(df[col].mean())
                phase_stds[phase].append(df[col].std())
            else:
                phase_means[phase].append(0); phase_stds[phase].append(0)

    x = np.arange(len(batch_sizes))
    xlabels = [f"bs={b}" for b in batch_sizes]

    # ------------------------------------------------------------------ #
    # Throughput vs batch size                                            #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots()
    ax.bar(x, throughput_means, yerr=throughput_stds, capsize=5, color="#4e79a7")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Samples / sec")
    ax.set_title("Throughput vs Batch Size")
    save(fig, os.path.join(args.out_dir, "compare_throughput.png"))

    # ------------------------------------------------------------------ #
    # Energy per sample vs batch size                                     #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots()
    ax.bar(x, energy_per_sample_means, yerr=energy_per_sample_stds, capsize=5, color="#f28e2b")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("GPU Energy per Sample (mJ) — NVML")
    ax.set_title("Energy per Sample vs Batch Size\n(lower = more efficient)")
    save(fig, os.path.join(args.out_dir, "compare_energy_per_sample.png"))

    # ------------------------------------------------------------------ #
    # GPU utilization vs batch size                                       #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots()
    ax.bar(x, gpu_util_means, yerr=gpu_util_stds, capsize=5, color="#59a14f")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU Utilization vs Batch Size")
    ax.set_ylim(0, 100)
    save(fig, os.path.join(args.out_dir, "compare_gpu_util.png"))

    # ------------------------------------------------------------------ #
    # Phase time breakdown vs batch size (grouped bars)                  #
    # ------------------------------------------------------------------ #
    phases = ["forward", "backward", "optimizer"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    width  = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (phase, color) in enumerate(zip(phases, colors)):
        offset = (i - 1) * width
        ax.bar(x + offset, phase_means[phase], width, yerr=phase_stds[phase],
               capsize=4, label=phase, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Avg Time (ms)")
    ax.set_title("Phase Time vs Batch Size")
    ax.legend()
    save(fig, os.path.join(args.out_dir, "compare_phase_times.png"))

    # ------------------------------------------------------------------ #
    # CodeCarbon total energy vs batch size                               #
    # ------------------------------------------------------------------ #
    cc_energy_means, cc_energy_stds = [], []
    for bs in batch_sizes:
        df_cc = load_cc_full(args.base_dir, bs, args.num_runs)
        if not df_cc.empty and "energy_consumed" in df_cc.columns:
            vals = df_cc["energy_consumed"] * 3600  # kWh -> kJ
            cc_energy_means.append(vals.mean())
            cc_energy_stds.append(vals.std())
        else:
            cc_energy_means.append(0); cc_energy_stds.append(0)

    fig, ax = plt.subplots()
    ax.bar(x, cc_energy_means, yerr=cc_energy_stds, capsize=5, color="#e15759")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Total Energy (kJ) — CodeCarbon")
    ax.set_title("Total Training Energy vs Batch Size (5 min run)")
    save(fig, os.path.join(args.out_dir, "compare_total_energy.png"))

    print(f"\nAll comparison plots written to: {args.out_dir}")


if __name__ == "__main__":
    main()
