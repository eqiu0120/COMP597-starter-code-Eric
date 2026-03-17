"""
plot_measurements.py
====================
Visualises and averages the combined GPU-resource and CodeCarbon measurements
produced by N independent RegNet training runs.

Output layout produced by the training job (job_config.sh):
  REGNET_OUT_DIR/
    run_<N>_cc_full_rank_0.csv          <- total-training CodeCarbon summary
    run_<N>_cc_step_rank_0-steps.csv    <- per-step CodeCarbon energy/emissions
    run_<N>_cc_substep_rank_0-substeps.csv
    losses/
      run_<N>_cc_loss_rank_0.csv

External nvidia-smi CSV (produced by job.sh when COMP597_LOG_GPU=1):
  /home/slurm/comp597/students/<USER>/gpu_measurements/gpu_<JOBID>.csv

Usage (on SLURM after all runs finish):
  python GPU_result/plot_measurements.py \\
      --cc_dir   /home/slurm/comp597/students/$USER/regnet_measurements \\
      --out_dir  /home/slurm/comp597/students/$USER/regnet_plots \\
      --num_runs 3 \\
      --gpu_csvs /home/.../gpu_<JOB0>.csv /home/.../gpu_<JOB1>.csv ...
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


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def _strip_unit(series: pd.Series, unit: str) -> pd.Series:
    return series.astype(str).str.replace(unit, "", regex=False).str.strip().astype(float)


def _load_step_csvs(cc_dir: str, num_runs: int, rank: int) -> list[pd.DataFrame]:
    """Load per-step CodeCarbon CSVs for all runs that exist."""
    dfs = []
    for run in range(num_runs):
        path = os.path.join(cc_dir, f"run_{run}_cc_step_rank_{rank}-steps.csv")
        if os.path.isfile(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"  [warn] step CSV not found: {path}")
    return dfs


def _load_substep_csvs(cc_dir: str, num_runs: int, rank: int) -> list[pd.DataFrame]:
    dfs = []
    for run in range(num_runs):
        path = os.path.join(cc_dir, f"run_{run}_cc_substep_rank_{rank}-substeps.csv")
        if os.path.isfile(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"  [warn] substep CSV not found: {path}")
    return dfs


def _load_loss_csvs(cc_dir: str, num_runs: int, rank: int) -> list[pd.DataFrame]:
    dfs = []
    for run in range(num_runs):
        path = os.path.join(cc_dir, "losses", f"run_{run}_cc_loss_rank_{rank}.csv")
        if os.path.isfile(path):
            dfs.append(pd.read_csv(path, header=None, names=["task_name", "loss"]))
        else:
            print(f"  [warn] loss CSV not found: {path}")
    return dfs


def _align_and_stack(dfs: list[pd.DataFrame], col: str) -> np.ndarray:
    """Stack a column from multiple run DataFrames into shape (num_runs, steps).

    Truncates to the shortest run so the array is rectangular.
    """
    arrays = [df[col].to_numpy() for df in dfs if col in df.columns]
    if not arrays:
        return np.empty((0,))
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)  # (runs, steps)


# ------------------------------------------------------------------ #
# nvidia-smi plots (one file per run, align by relative time)         #
# ------------------------------------------------------------------ #

def plot_nvidia_smi(gpu_csvs: list[str], out_dir: str):
    """Averaged time-series from one or more nvidia-smi logs."""
    if not gpu_csvs:
        print("No nvidia-smi CSVs provided; skipping GPU time-series plots.")
        return

    metrics = {
        " utilization.gpu [%]":    ("%",   "GPU Utilization (%)",   "gpu_util.png"),
        " memory.used [MiB]":      ("MiB", "GPU Memory Used (MiB)", "gpu_mem.png"),
        " power.draw [W]":         ("W",   "GPU Power Draw (W)",    "gpu_power.png"),
        " clocks.current.sm [MHz]":("MHz", "SM Clock (MHz)",        "gpu_clock.png"),
    }

    # Parse all runs; strip units first so values are numeric, then resample
    run_series: dict[str, list] = {k: [] for k in metrics}
    for csv_path in gpu_csvs:
        if not os.path.isfile(csv_path):
            print(f"  [warn] GPU CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["t_s"] = t.round(0).astype(int)
        # Strip units from string columns so groupby can average them
        for col, (unit, _, _) in metrics.items():
            if col in df.columns:
                df[col] = _strip_unit(df[col], unit)
        df_rs = df.groupby("t_s").mean(numeric_only=True)
        for col in metrics:
            if col in df_rs.columns:
                run_series[col].append(df_rs[col])

    t_label = "Time (s)"
    for col, (unit, ylabel, fname) in metrics.items():
        series_list = run_series[col]
        if not series_list:
            continue
        # Use actual elapsed seconds from index, align to shortest run
        min_len = min(len(s) for s in series_list)
        arr = np.stack([np.array(s.iloc[:min_len], dtype=float) for s in series_list], axis=0)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        # Use actual timestamps from the first run's index
        t_vals = series_list[0].index[:min_len]

        fig, ax = plt.subplots()
        ax.plot(t_vals, mean, label="mean")
        ax.fill_between(t_vals, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — averaged over {len(series_list)} run(s)")
        ax.legend()
        save(fig, os.path.join(out_dir, fname))


# ------------------------------------------------------------------ #
# CodeCarbon per-step plots                                           #
# ------------------------------------------------------------------ #

def plot_codecarbon_steps(cc_dir: str, num_runs: int, rank: int, out_dir: str):
    dfs = _load_step_csvs(cc_dir, num_runs, rank)
    if not dfs:
        print("No per-step CodeCarbon CSVs found; skipping.")
        return

    n_runs = len(dfs)

    # Energy per step (kWh -> mJ for readability)
    energy_arr = _align_and_stack(dfs, "energy_consumed") * 3_600_000  # kWh -> J *1e3 = mJ /1 = J
    if energy_arr.ndim == 2:
        mean_e = energy_arr.mean(axis=0)
        std_e  = energy_arr.std(axis=0)
        steps  = np.arange(len(mean_e))

        fig, ax = plt.subplots()
        ax.bar(steps, mean_e, label="mean energy")
        ax.errorbar(steps, mean_e, yerr=std_e, fmt="none", color="black",
                    capsize=2, label="±1 std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy Consumed (J)")
        ax.set_title(f"Energy per Training Step — averaged over {n_runs} run(s)")
        ax.legend()
        save(fig, os.path.join(out_dir, "cc_energy_per_step.png"))

    # CO2 per step (kg -> mg)
    co2_arr = _align_and_stack(dfs, "emissions") * 1e6  # kg -> mg
    if co2_arr.ndim == 2:
        mean_c = co2_arr.mean(axis=0)
        std_c  = co2_arr.std(axis=0)
        steps  = np.arange(len(mean_c))

        fig, ax = plt.subplots()
        ax.bar(steps, mean_c, label="mean CO2")
        ax.errorbar(steps, mean_c, yerr=std_c, fmt="none", color="black",
                    capsize=2, label="±1 std")
        ax.set_xlabel("Step")
        ax.set_ylabel("CO2 Emissions (mg)")
        ax.set_title(f"CO2 per Training Step — averaged over {n_runs} run(s)")
        ax.legend()
        save(fig, os.path.join(out_dir, "cc_co2_per_step.png"))

    # Cumulative energy curve
    if energy_arr.ndim == 2:
        cum_arr = np.cumsum(energy_arr, axis=1)
        mean_cum = cum_arr.mean(axis=0)
        std_cum  = cum_arr.std(axis=0)
        steps = np.arange(len(mean_cum))

        fig, ax = plt.subplots()
        ax.plot(steps, mean_cum, label="mean")
        ax.fill_between(steps, mean_cum - std_cum, mean_cum + std_cum,
                        alpha=0.3, label="±1 std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Energy (J)")
        ax.set_title(f"Cumulative Energy During Training — averaged over {n_runs} run(s)")
        ax.legend()
        save(fig, os.path.join(out_dir, "cc_energy_cumulative.png"))


# ------------------------------------------------------------------ #
# CodeCarbon per-substep plots                                        #
# ------------------------------------------------------------------ #

def plot_codecarbon_substeps(cc_dir: str, num_runs: int, rank: int, out_dir: str):
    dfs = _load_substep_csvs(cc_dir, num_runs, rank)
    if not dfs:
        print("No per-substep CodeCarbon CSVs found; skipping.")
        return

    n_runs = len(dfs)
    substep_labels = ["forward", "backward", "optimizer"]
    substep_patterns = ["Forward", "Backward", "Optim"]

    # Accumulate total energy per substep type across all runs
    totals_per_run = []
    for df in dfs:
        if "task_name" not in df.columns or "energy_consumed" not in df.columns:
            continue
        row = {}
        for label, pattern in zip(substep_labels, substep_patterns):
            mask = df["task_name"].str.contains(pattern, case=False)
            row[label] = df.loc[mask, "energy_consumed"].sum() * 3600  # kWh -> J
        totals_per_run.append(row)

    if not totals_per_run:
        return

    totals_df = pd.DataFrame(totals_per_run)  # shape (n_runs, 3)
    mean_totals = totals_df.mean()
    std_totals  = totals_df.std()

    fig, ax = plt.subplots()
    x = np.arange(len(substep_labels))
    ax.bar(x, mean_totals, yerr=std_totals, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(substep_labels)
    ax.set_ylabel("Total Energy (J)")
    ax.set_title(f"Energy Breakdown by Substep — averaged over {n_runs} run(s)")
    save(fig, os.path.join(out_dir, "cc_energy_substeps.png"))


# ------------------------------------------------------------------ #
# Loss curve                                                          #
# ------------------------------------------------------------------ #

def plot_losses(cc_dir: str, num_runs: int, rank: int, out_dir: str):
    dfs = _load_loss_csvs(cc_dir, num_runs, rank)
    if not dfs:
        print("No loss CSVs found; skipping.")
        return

    n_runs = len(dfs)
    loss_arr = _align_and_stack(dfs, "loss")
    if loss_arr.ndim < 2:
        return

    mean_loss = loss_arr.mean(axis=0)
    std_loss  = loss_arr.std(axis=0)
    steps = np.arange(len(mean_loss))

    fig, ax = plt.subplots()
    ax.plot(steps, mean_loss, label="mean loss")
    ax.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss,
                    alpha=0.3, label="±1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (RegNet) — averaged over {n_runs} run(s)")
    ax.legend()
    save(fig, os.path.join(out_dir, "training_loss.png"))


# ------------------------------------------------------------------ #
# SLURM stdout log parsing (ResourceTrainerStats per-step lines)      #
# ------------------------------------------------------------------ #

# Matches lines produced by ResourceTrainerStats.log_step(), e.g.:
#   step 450.1234 -- forward 120.56 -- backward 200.12 -- optimizer step 50.45 \
#   -- data_loading 79.00 -- gpu_util% 85.20 -- gpu_mem(MB) 12000.0 \
#   -- io_read(B) 1024 -- io_write(B) 0 \
#   -- energy_step(mJ) 500.0 -- energy_fwd(mJ) 150.0 ...
_STEP_LINE_RE = re.compile(
    r"step\s+(?P<step>[\d.]+)\s+--\s+"
    r"forward\s+(?P<fwd>[\d.]+)\s+--\s+"
    r"backward\s+(?P<bwd>[\d.]+)\s+--\s+"
    r"optimizer step\s+(?P<opt>[\d.]+)"
    r"(?:.*?data_loading\s+(?P<data>[\d.]+))?"
    r"(?:.*?gpu_util%\s+(?P<gpu_util>[\d.]+))?"
    r"(?:.*?gpu_mem\(MB\)\s+(?P<gpu_mem>[\d.]+))?"
    r"(?:.*?energy_step\(mJ\)\s+(?P<e_step>[\d.]+))?"
    r"(?:.*?energy_fwd\(mJ\)\s+(?P<e_fwd>[\d.]+))?"
    r"(?:.*?energy_bwd\(mJ\)\s+(?P<e_bwd>[\d.]+))?"
    r"(?:.*?energy_opt\(mJ\)\s+(?P<e_opt>[\d.]+))?"
)

def parse_stdout_log(log_path: str) -> pd.DataFrame:
    """Parse per-step ResourceTrainerStats lines from a SLURM log file."""
    rows = []
    with open(log_path) as f:
        for line in f:
            m = _STEP_LINE_RE.search(line)
            if m:
                rows.append({k: float(v) for k, v in m.groupdict().items() if v is not None})
    return pd.DataFrame(rows)


def _load_stdout_logs(log_files: list[str]) -> list[pd.DataFrame]:
    dfs = []
    for p in log_files:
        if os.path.isfile(p):
            df = parse_stdout_log(p)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"  [warn] no step lines found in: {p}")
        else:
            print(f"  [warn] log file not found: {p}")
    return dfs


def plot_time_breakdown(log_files: list[str], out_dir: str, batch_size: int = 0):
    """Per-phase bars (forward/backward/optimizer/data_loading) with mean ± std."""
    dfs = _load_stdout_logs(log_files)
    if not dfs:
        print("No stdout log files provided; skipping time-breakdown plots.")
        return

    n_runs = len(dfs)
    labels   = ["forward", "backward", "optimizer", "data_loading"]
    cols_map = ["fwd",     "bwd",      "opt",       "data"]

    # Collect per-step values across all runs
    all_vals = {lbl: [] for lbl in labels}
    for df in dfs:
        for lbl, col in zip(labels, cols_map):
            if col in df.columns:
                all_vals[lbl].extend(df[col].tolist())
            elif col == "data" and all(c in df.columns for c in ["step", "fwd", "bwd", "opt"]):
                derived = np.maximum(0, df["step"] - df["fwd"] - df["bwd"] - df["opt"])
                all_vals[lbl].extend(derived.tolist())

    means = [np.mean(all_vals[l]) if all_vals[l] else 0 for l in labels]
    stds  = [np.std(all_vals[l])  if all_vals[l] else 0 for l in labels]

    # Separate bars per phase with error bars
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (ms)")
    bs_str = f" — batch size {batch_size}" if batch_size else ""
    ax.set_title(f"Avg Time per Phase ± std — {n_runs} run(s){bs_str}\n"
                 f"(data_loading = GPU idle between batches)")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
    save(fig, os.path.join(out_dir, "time_breakdown.png"))

    # Also save stacked version for total step time view
    fig2, ax2 = plt.subplots(figsize=(5, 6))
    bottom = 0.0
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax2.bar(0, means[j], bottom=bottom, color=color, label=label, width=0.5)
        bottom += means[j]
    ax2.set_xticks([])
    ax2.set_ylabel("Time (ms)")
    ax2.set_title(f"Step Time Composition{bs_str}")
    ax2.legend(loc="upper right")
    save(fig2, os.path.join(out_dir, "time_breakdown_stacked.png"))

    # Line chart: data-loading overhead per step (early vs late steps)
    if all("data" in df.columns or ("step" in df.columns) for df in dfs):
        data_arrays = []
        for df in dfs:
            if "data" in df.columns:
                data_arrays.append(df["data"].to_numpy())
            elif all(c in df.columns for c in ["step", "fwd", "bwd", "opt"]):
                data_arrays.append(
                    np.maximum(0, df["step"].to_numpy() - df["fwd"].to_numpy()
                               - df["bwd"].to_numpy() - df["opt"].to_numpy()))
        if data_arrays:
            min_len = min(len(a) for a in data_arrays)
            arr = np.stack([a[:min_len] for a in data_arrays])
            mean_d = arr.mean(axis=0)
            std_d  = arr.std(axis=0)
            steps = np.arange(min_len)
            fig, ax = plt.subplots()
            ax.plot(steps, mean_d, label="mean data-loading time")
            ax.fill_between(steps, mean_d - std_d, mean_d + std_d, alpha=0.3)
            ax.set_xlabel("Step")
            ax.set_ylabel("Data Loading Time (ms)")
            ax.set_title(f"Data Loading Overhead per Step — {n_runs} run(s)")
            ax.legend()
            save(fig, os.path.join(out_dir, "data_loading_overhead.png"))


def plot_nvml_energy(log_files: list[str], out_dir: str):
    """Per-substep NVML GPU energy from stdout logs."""
    dfs = _load_stdout_logs(log_files)
    energy_cols = {"e_step": "step", "e_fwd": "forward", "e_bwd": "backward", "e_opt": "optimizer"}
    if not dfs or not any("e_step" in df.columns for df in dfs):
        print("No NVML energy data found in logs; skipping NVML energy plots.")
        return

    n_runs = len(dfs)

    # Substep energy breakdown bar (averaged over all steps and runs)
    substep_labels = ["forward", "backward", "optimizer"]
    substep_cols   = ["e_fwd",   "e_bwd",    "e_opt"]
    totals_per_run = []
    for df in dfs:
        row = {label: df[col].mean() if col in df.columns else 0.0
               for label, col in zip(substep_labels, substep_cols)}
        totals_per_run.append(row)

    totals_df  = pd.DataFrame(totals_per_run)
    mean_t     = totals_df.mean()
    std_t      = totals_df.std()

    fig, ax = plt.subplots()
    x = np.arange(len(substep_labels))
    ax.bar(x, mean_t, yerr=std_t, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(substep_labels)
    ax.set_ylabel("Avg GPU Energy per Step (mJ) — NVML direct")
    ax.set_title(f"GPU Energy Breakdown by Substep — {n_runs} run(s)")
    save(fig, os.path.join(out_dir, "nvml_energy_substeps.png"))

    # Per-step GPU energy trend
    e_step_arrays = [df["e_step"].to_numpy() for df in dfs if "e_step" in df.columns]
    if e_step_arrays:
        min_len = min(len(a) for a in e_step_arrays)
        arr = np.stack([a[:min_len] for a in e_step_arrays])
        mean_e = arr.mean(axis=0)
        std_e  = arr.std(axis=0)
        steps  = np.arange(min_len)
        fig, ax = plt.subplots()
        ax.plot(steps, mean_e, label="mean GPU energy/step")
        ax.fill_between(steps, mean_e - std_e, mean_e + std_e, alpha=0.3, label="±1 std")
        ax.set_xlabel("Step")
        ax.set_ylabel("GPU Energy (mJ) — NVML direct")
        ax.set_title(f"GPU Energy per Step — {n_runs} run(s)")
        ax.legend()
        save(fig, os.path.join(out_dir, "nvml_energy_per_step.png"))


# ------------------------------------------------------------------ #
# CPU utilization timeline                                            #
# ------------------------------------------------------------------ #

def plot_cpu_util(cpu_csvs: list[str], out_dir: str, batch_size: int = 0):
    """Averaged CPU utilization timeline from cpu_monitor.py logs."""
    if not cpu_csvs:
        print("No CPU CSVs provided; skipping CPU timeline plot.")
        return

    run_series = []
    for csv_path in cpu_csvs:
        if not os.path.isfile(csv_path):
            print(f"  [warn] CPU CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "cpu_percent" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["t_s"] = t.round(0).astype(int)
        df_rs = df.groupby("t_s")["cpu_percent"].mean()
        run_series.append(df_rs)

    if not run_series:
        print("No valid CPU data found; skipping.")
        return

    min_len = min(len(s) for s in run_series)
    arr = np.stack([s.iloc[:min_len].to_numpy() for s in run_series])
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    t_vals = np.arange(min_len)

    bs_str = f" — batch size {batch_size}" if batch_size else ""
    fig, ax = plt.subplots()
    ax.plot(t_vals, mean, label="mean CPU util")
    ax.fill_between(t_vals, mean - std, mean + std, alpha=0.3, label="±1 std")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title(f"CPU Utilization — averaged over {len(run_series)} run(s){bs_str}")
    ax.legend()
    save(fig, os.path.join(out_dir, "cpu_util.png"))


# ------------------------------------------------------------------ #
# Throughput                                                          #
# ------------------------------------------------------------------ #

def plot_throughput(log_files: list[str], out_dir: str, batch_size: int = 8):
    """Samples/sec throughput over time, derived from per-step timing logs."""
    dfs = _load_stdout_logs(log_files)
    if not dfs:
        return

    n_runs = len(dfs)
    throughput_arrays = []
    for df in dfs:
        if "step" in df.columns:
            # step time is in ms; throughput = batch_size / (step_time_s)
            step_s = df["step"] / 1000.0
            tp = batch_size / step_s.replace(0, np.nan)
            throughput_arrays.append(tp.dropna().to_numpy())

    if not throughput_arrays:
        return

    min_len = min(len(a) for a in throughput_arrays)
    arr = np.stack([a[:min_len] for a in throughput_arrays])
    mean_tp = arr.mean(axis=0)
    std_tp  = arr.std(axis=0)
    steps   = np.arange(min_len)

    fig, ax = plt.subplots()
    ax.plot(steps, mean_tp, label="mean throughput")
    ax.fill_between(steps, mean_tp - std_tp, mean_tp + std_tp, alpha=0.3, label="±1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title(f"Training Throughput — batch size {batch_size}, {n_runs} run(s)")
    ax.legend()
    save(fig, os.path.join(out_dir, "throughput.png"))


# ------------------------------------------------------------------ #
# Total summary table                                                 #
# ------------------------------------------------------------------ #

def print_total_summary(cc_dir: str, num_runs: int, rank: int):
    rows = []
    for run in range(num_runs):
        path = os.path.join(cc_dir, f"run_{run}_cc_full_rank_{rank}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if not df.empty:
                rows.append(df.iloc[-1])

    if not rows:
        print("No full-training CodeCarbon CSV found.")
        return

    summary = pd.DataFrame(rows)
    cols = ["duration", "energy_consumed", "emissions", "cpu_energy", "gpu_energy", "ram_energy"]
    print(f"\n===== Total Training Summary — {len(rows)} run(s) =====")
    for col in cols:
        if col not in summary.columns:
            continue
        mean_val = summary[col].mean()
        std_val  = summary[col].std()
        if col == "energy_consumed":
            print(f"  energy_consumed : {mean_val:.6f} ± {std_val:.6f} kWh"
                  f"  ({mean_val * 3600:.3f} ± {std_val * 3600:.3f} kJ)")
        elif col == "emissions":
            print(f"  emissions       : {mean_val * 1000:.6f} ± {std_val * 1000:.6f} g CO2eq")
        elif col == "duration":
            print(f"  duration        : {mean_val:.2f} ± {std_val:.2f} s")
        else:
            print(f"  {col:<16}: {mean_val:.6f} ± {std_val:.6f} kWh")
    print("=" * 51 + "\n")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Plot and average RegNet measurement results")
    parser.add_argument("--cc_dir",    required=True,
                        help="Directory containing CodeCarbon CSV files")
    parser.add_argument("--out_dir",   required=True,
                        help="Directory to save plots")
    parser.add_argument("--num_runs",  type=int, default=3,
                        help="Number of runs to average (default: 3)")
    parser.add_argument("--rank",      type=int, default=0,
                        help="GPU rank (default: 0)")
    parser.add_argument("--gpu_csvs",   nargs="*", default=[],
                        help="nvidia-smi CSV files, one per run (space-separated)")
    parser.add_argument("--cpu_csvs",   nargs="*", default=[],
                        help="cpu_monitor.py CSV files, one per run (space-separated)")
    parser.add_argument("--log_files",  nargs="*", default=[],
                        help="SLURM stdout log files (.log), one per run.")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size used (for plot titles and throughput calc)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nAveraging over {args.num_runs} run(s) from: {args.cc_dir}")

    # Auto-detect nvidia-smi CSVs if none provided
    gpu_csvs = args.gpu_csvs
    if not gpu_csvs:
        pattern = f"/home/slurm/comp597/students/{os.environ.get('USER', 'zqiu6')}/gpu_measurements/gpu_*.csv"
        gpu_csvs = sorted(glob.glob(pattern))
        if gpu_csvs:
            print(f"Auto-detected {len(gpu_csvs)} GPU CSV(s): {gpu_csvs}")

    # Auto-detect CPU CSVs if none provided
    cpu_csvs = args.cpu_csvs
    if not cpu_csvs:
        pattern = f"/home/slurm/comp597/students/{os.environ.get('USER', 'zqiu6')}/gpu_measurements/cpu_*.csv"
        cpu_csvs = sorted(glob.glob(pattern))
        if cpu_csvs:
            print(f"Auto-detected {len(cpu_csvs)} CPU CSV(s): {cpu_csvs}")

    # Auto-detect SLURM log files if none provided
    log_files = args.log_files
    if not log_files:
        log_files = sorted(glob.glob("comp597-regnet-run*-*.log"))
        if log_files:
            print(f"Auto-detected {len(log_files)} SLURM log file(s): {log_files}")

    bs = args.batch_size

    # --- Plots from nvidia-smi (GPU time-series) ---
    plot_nvidia_smi(gpu_csvs, args.out_dir)

    # --- CPU utilization timeline ---
    plot_cpu_util(cpu_csvs, args.out_dir, batch_size=bs)

    # --- Plots from CodeCarbon CSVs (energy / CO2 per step) ---
    plot_codecarbon_steps(args.cc_dir, args.num_runs, args.rank, args.out_dir)
    plot_codecarbon_substeps(args.cc_dir, args.num_runs, args.rank, args.out_dir)
    plot_losses(args.cc_dir, args.num_runs, args.rank, args.out_dir)

    # --- Plots from SLURM stdout logs ---
    plot_time_breakdown(log_files, args.out_dir, batch_size=bs)
    plot_nvml_energy(log_files, args.out_dir)
    plot_throughput(log_files, args.out_dir, batch_size=bs if bs else 8)

    # --- Summary table ---
    print_total_summary(args.cc_dir, args.num_runs, args.rank)

    print(f"\nAll plots written to: {args.out_dir}")
    print("\nPlots generated:")
    print("  gpu_util.png / gpu_mem.png / gpu_power.png  -> GPU timelines (entire run)")
    print("  cpu_util.png                                -> CPU utilization timeline")
    print("  time_breakdown.png                          -> separate bars per phase ± std")
    print("  time_breakdown_stacked.png                  -> stacked step composition")
    print("  throughput.png                              -> samples/sec over time")
    print("  nvml_energy_substeps.png                    -> GPU energy per phase")
    print("  cc_energy_per_step.png                      -> CodeCarbon energy per step")
    print("  cc_energy_substeps.png                      -> CodeCarbon energy by phase")


if __name__ == "__main__":
    main()
