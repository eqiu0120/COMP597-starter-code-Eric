# Compares RegNet measurements across batch sizes (32, 16, 8).
# Usage: python GPU_result/plot_comparison.py --base_dir ... --out_dir ... --batch_sizes 32 16 8 --num_runs 3

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


PHASE_COLORS = {
    "forward":      "#4e79a7",
    "backward":     "#f28e2b",
    "optimizer":    "#59a14f",
    "data_loading": "#e15759",
}

_STEP_LINE_RE = re.compile(
    r"step\s+(?P<step>[\d.]+)\s+--\s+"
    r"forward\s+(?P<fwd>[\d.]+)\s+--\s+"
    r"backward\s+(?P<bwd>[\d.]+)\s+--\s+"
    r"optimizer step\s+(?P<opt>[\d.]+)"
    r"(?:.*?data_loading\s+(?P<data>[\d.]+))?"
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


def parse_gpu_csv(path, fmt='%Y/%m/%d %H:%M:%S.%f'):
    """Load nvidia-smi CSV, return (t_seconds, util_percent) arrays."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=fmt)
    t0 = df["timestamp"].iloc[0]
    df["t_s"] = (df["timestamp"] - t0).dt.total_seconds()
    col = " utilization.gpu [%]"
    df[col] = df[col].astype(str).str.replace("%", "").str.strip().astype(float)
    return df["t_s"].to_numpy(), df[col].to_numpy()


def plot_gpu_util_zoomed(gpu_csv, log_files, batch_size, zoom_center, out_dir):
    """
    Plot a one-epoch-wide zoomed window of GPU utilization with phase shading.
    zoom_center: time in seconds around which to centre the window.
    """
    # Load GPU util timeline
    t_gpu, util = parse_gpu_csv(gpu_csv)

    # Build phase timeline from log (first log file only)
    phase_bands = []   # (t_start, t_end, phase_label)
    if log_files:
        df_log = parse_logs([log_files[0]])
        if not df_log.empty and all(c in df_log.columns for c in ["fwd", "bwd", "opt"]):
            t_cursor = 0.0
            for _, row in df_log.iterrows():
                fwd  = row["fwd"]  / 1000.0
                bwd  = row["bwd"]  / 1000.0
                opt  = row["opt"]  / 1000.0
                dat  = (row["data"] / 1000.0) if "data" in df_log.columns and not pd.isna(row.get("data", float("nan"))) else 0.0
                phase_bands.append((t_cursor,               t_cursor + fwd,           "forward"))
                phase_bands.append((t_cursor + fwd,         t_cursor + fwd + bwd,     "backward"))
                phase_bands.append((t_cursor + fwd + bwd,   t_cursor + fwd + bwd + opt, "optimizer"))
                if dat > 0:
                    phase_bands.append((t_cursor + fwd + bwd + opt,
                                        t_cursor + fwd + bwd + opt + dat, "data_loading"))
                t_cursor += fwd + bwd + opt + dat

    # Determine zoom window: one epoch wide centred on zoom_center
    steps_per_epoch = 2000 / batch_size
    # Estimate epoch duration from log
    if log_files:
        df_log = parse_logs([log_files[0]])
        if not df_log.empty and "step" in df_log.columns:
            mean_step_s = df_log["step"].mean() / 1000.0
            epoch_s = steps_per_epoch * mean_step_s
        else:
            epoch_s = 30.0
    else:
        epoch_s = 30.0

    t_lo = zoom_center - epoch_s / 2
    t_hi = zoom_center + epoch_s / 2

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Phase background shading
    seen = set()
    for (ts, te, phase) in phase_bands:
        if te < t_lo or ts > t_hi:
            continue
        label = phase if phase not in seen else None
        ax.axvspan(max(ts, t_lo), min(te, t_hi),
                   color=PHASE_COLORS[phase], alpha=0.18, linewidth=0, label=label)
        seen.add(phase)

    # GPU util line
    mask = (t_gpu >= t_lo) & (t_gpu <= t_hi)
    ax.plot(t_gpu[mask], util[mask], color="black", linewidth=1.2,
            label="GPU util (%)", zorder=3)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title(f"GPU Utilization — bs={batch_size}, zoomed ~{t_lo:.0f}–{t_hi:.0f}s\n"
                 f"(phase shading from single run)")
    # Reorder legend: gpu util first, then phases
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc="lower right")
    fname = f"gpu_util_zoomed_bs{batch_size}.png"
    save(fig, os.path.join(out_dir, fname))


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
    parser.add_argument("--gpu_csv_bs32", default="",
                        help="Single nvidia-smi CSV for bs=32 zoomed plot")
    parser.add_argument("--gpu_csv_bs16", default="",
                        help="Single nvidia-smi CSV for bs=16 zoomed plot")
    parser.add_argument("--gpu_csv_bs8",  default="",
                        help="Single nvidia-smi CSV for bs=8 zoomed plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    batch_sizes = args.batch_sizes

    logs_map = {32: args.logs_bs32, 16: args.logs_bs16, 8: args.logs_bs8}

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

        if "step" in df.columns:
            tp = bs / (df["step"] / 1000.0).replace(0, np.nan)
            throughput_means.append(tp.mean())
            throughput_stds.append(tp.std())
        else:
            throughput_means.append(0); throughput_stds.append(0)

        if "e_step" in df.columns:
            eps = df["e_step"] / bs
            energy_per_sample_means.append(eps.mean())
            energy_per_sample_stds.append(eps.std())
        else:
            energy_per_sample_means.append(0); energy_per_sample_stds.append(0)

        if "gpu_util" in df.columns:
            gpu_util_means.append(df["gpu_util"].mean())
            gpu_util_stds.append(df["gpu_util"].std())
        else:
            gpu_util_means.append(0); gpu_util_stds.append(0)

        for phase, col in [("forward","fwd"), ("backward","bwd"), ("optimizer","opt")]:
            if col in df.columns:
                phase_means[phase].append(df[col].mean())
                phase_stds[phase].append(df[col].std())
            else:
                phase_means[phase].append(0); phase_stds[phase].append(0)

    x = np.arange(len(batch_sizes))
    xlabels = [f"bs={b}" for b in batch_sizes]

    fig, ax = plt.subplots()
    ax.bar(x, throughput_means, yerr=throughput_stds, capsize=5, color="#4e79a7")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Samples / sec")
    ax.set_title("Throughput vs Batch Size")
    save(fig, os.path.join(args.out_dir, "compare_throughput.png"))

    fig, ax = plt.subplots()
    ax.bar(x, energy_per_sample_means, yerr=energy_per_sample_stds, capsize=5, color="#f28e2b")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("GPU Energy per Sample (mJ) — NVML")
    ax.set_title("Energy per Sample vs Batch Size\n(lower = more efficient)")
    save(fig, os.path.join(args.out_dir, "compare_energy_per_sample.png"))

    fig, ax = plt.subplots()
    ax.bar(x, gpu_util_means, yerr=gpu_util_stds, capsize=5, color="#59a14f")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU Utilization vs Batch Size")
    ax.set_ylim(0, 100)
    save(fig, os.path.join(args.out_dir, "compare_gpu_util.png"))

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

    cc_energy_means, cc_energy_stds = [], []
    for bs in batch_sizes:
        df_cc = load_cc_full(args.base_dir, bs, args.num_runs)
        if not df_cc.empty and "energy_consumed" in df_cc.columns:
            vals = df_cc["energy_consumed"] * 3600
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

    # Zoomed GPU util plots centred on known big-dip regions
    zoom_cfg = [
        (32, args.gpu_csv_bs32, args.logs_bs32, 232.0),  # dips at t~220-245s
        (16, args.gpu_csv_bs16, args.logs_bs16,  85.0),  # dip at t~85s
        (8,  args.gpu_csv_bs8,  args.logs_bs8,   90.0),  # dip at t~90s
    ]
    for bs, gpu_csv, log_files, center in zoom_cfg:
        if gpu_csv and os.path.isfile(gpu_csv):
            plot_gpu_util_zoomed(gpu_csv, log_files, bs, center, args.out_dir)
        else:
            print(f"  [skip] no gpu_csv for bs={bs} zoomed plot")

    print(f"\nAll comparison plots written to: {args.out_dir}")


if __name__ == "__main__":
    main()
