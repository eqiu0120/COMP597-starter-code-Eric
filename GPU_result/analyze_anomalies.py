"""
Analyze anomalies: energy spikes, throughput dips, GPU util dips.
Correlates per-substep CodeCarbon data with nvidia-smi GPU timeline.

Usage:
    python GPU_result/analyze_anomalies.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "raw_csvs")
OUT  = os.path.join(os.path.dirname(__file__), "plots", "anomaly_analysis")
os.makedirs(OUT, exist_ok=True)

# Job IDs for combined (Exp3) runs
JOB_IDS = {
    32: [16142, 16145, 16204],
    16: [16148, 16150, 16151],
    8:  [16182, 16183, 16189],
}

# Which run to use for deep analysis (run 0 = most representative)
ANALYSIS_RUN = 0


def parse_ts(ts_str):
    """Parse CodeCarbon or nvidia-smi timestamp to datetime."""
    ts_str = str(ts_str).strip()
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"]:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def load_substep_csv(bs, run=0):
    """Load per-substep CodeCarbon CSV. Returns DataFrame with elapsed_s column."""
    path = os.path.join(BASE, f"bs_{bs}", f"run_{run}_cc_substep_rank_0-substeps.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    df = pd.read_csv(path)
    df["ts"] = df["timestamp"].apply(parse_ts)
    t0 = df["ts"].min()
    df["elapsed_s"] = (df["ts"] - t0).dt.total_seconds()
    df["energy_mj"] = df["energy_consumed"] * 3_600_000  # kWh -> mJ
    df["substep"] = df["task_name"].str.extract(r"(Forward|Backward|Optimizer)")
    return df


def load_step_csv(bs, run=0):
    """Load per-step CodeCarbon CSV."""
    path = os.path.join(BASE, f"bs_{bs}", f"run_{run}_cc_step_rank_0-steps.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["ts"] = df["timestamp"].apply(parse_ts)
    t0 = df["ts"].min()
    df["elapsed_s"] = (df["ts"] - t0).dt.total_seconds()
    df["energy_mj"] = df["energy_consumed"] * 3_600_000
    df["throughput"] = 0.0  # will fill below
    return df


def load_gpu_csv(bs, job_id):
    """Load nvidia-smi CSV."""
    path = os.path.join(BASE, f"bs_{bs}", f"gpu_{job_id}.csv")
    if not os.path.exists(path):
        print(f"  Missing GPU CSV: {path}")
        return None
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # Strip units from value columns
    for col in df.columns:
        if col != "timestamp":
            df[col] = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = df["timestamp"].apply(parse_ts)
    t0 = df["ts"].min()
    df["elapsed_s"] = (df["ts"] - t0).dt.total_seconds()
    return df


def find_anomalies(series, z_thresh=2.0, direction="high"):
    """Return indices where series deviates more than z_thresh std from mean."""
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Index([])
    z = (series - mean) / std
    if direction == "high":
        return series.index[z > z_thresh]
    else:
        return series.index[z < -z_thresh]


def analyze_batch(bs):
    run = ANALYSIS_RUN
    job_id = JOB_IDS[bs][run]
    print(f"\n{'='*60}")
    print(f"  Batch size {bs} — run {run} (job {job_id})")
    print(f"{'='*60}")

    sub = load_substep_csv(bs, run)
    gpu = load_gpu_csv(bs, job_id)

    if sub is None or gpu is None:
        print("  Skipping — missing data.")
        return

    # ── Substep duration anomalies ──────────────────────────────
    sub["duration_ms"] = sub["duration"] * 1000
    mean_dur = sub.groupby("substep")["duration_ms"].mean()
    print("\nMean substep duration (ms):")
    print(mean_dur.to_string())

    slow_idx = find_anomalies(sub["duration_ms"], z_thresh=2.5, direction="high")
    print(f"\nSlow substep anomalies (>{2.5}σ): {len(slow_idx)} events")
    if len(slow_idx):
        print(sub.loc[slow_idx, ["task_name", "elapsed_s", "duration_ms", "energy_mj"]].to_string(index=False))

    # ── Energy spike anomalies ──────────────────────────────────
    energy_spike_idx = find_anomalies(sub["energy_mj"], z_thresh=2.5, direction="high")
    print(f"\nEnergy spike anomalies (>{2.5}σ): {len(energy_spike_idx)} events")
    if len(energy_spike_idx):
        print(sub.loc[energy_spike_idx, ["task_name", "elapsed_s", "duration_ms", "energy_mj"]].to_string(index=False))

    # ── GPU util dips ───────────────────────────────────────────
    gpu_col = [c for c in gpu.columns if "utilization" in c.lower()][0]
    util_dip_idx = find_anomalies(gpu[gpu_col].dropna(), z_thresh=2.0, direction="low")
    print(f"\nGPU util dips (>{2.0}σ below mean): {len(util_dip_idx)} events")
    if len(util_dip_idx):
        dip_times = gpu.loc[util_dip_idx, ["elapsed_s", gpu_col]]
        print(dip_times.to_string(index=False))

    # ── Epoch boundary detection ────────────────────────────────
    # Steps per epoch = dataset_size / batch_size = 2000 / bs
    steps_per_epoch = 2000 // bs
    # cc_substep has 3 rows per step (forward, backward, optimizer)
    total_substeps = len(sub)
    total_steps = total_substeps // 3
    print(f"\nTotal steps: {total_steps}, steps per epoch: {steps_per_epoch}")
    epoch_step_indices = list(range(steps_per_epoch, total_steps, steps_per_epoch))
    epoch_substep_indices = [i * 3 for i in epoch_step_indices if i * 3 < len(sub)]
    epoch_times = sub.iloc[epoch_substep_indices]["elapsed_s"].values if epoch_substep_indices else []
    print(f"Epoch boundaries at ~elapsed seconds: {np.round(epoch_times, 1).tolist()}")

    # ── Correlate GPU dips with epoch boundaries ────────────────
    if len(util_dip_idx) and len(epoch_times):
        dip_ts = gpu.loc[util_dip_idx, "elapsed_s"].values
        for dip_t in dip_ts:
            nearest_epoch = min(epoch_times, key=lambda e: abs(e - dip_t))
            gap = abs(dip_t - nearest_epoch)
            if gap < 10:
                print(f"  GPU dip at t={dip_t:.1f}s is {gap:.1f}s from epoch boundary at t={nearest_epoch:.1f}s → likely epoch reshuffle")

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle(f"Anomaly Analysis — batch size {bs} — run {run}", fontsize=13)

    # 1. Substep duration over time
    ax = axes[0]
    colors = {"Forward": "steelblue", "Backward": "tomato", "Optimizer": "seagreen"}
    for name, grp in sub.groupby("substep"):
        ax.scatter(grp["elapsed_s"], grp["duration_ms"], s=2, alpha=0.4,
                   label=name, color=colors.get(name, "gray"))
    if len(slow_idx):
        ax.scatter(sub.loc[slow_idx, "elapsed_s"], sub.loc[slow_idx, "duration_ms"],
                   s=40, color="red", zorder=5, label="slow outlier")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Substep Duration Over Time (orange = epoch boundary)")
    ax.legend(markerscale=4, fontsize=8)

    # 2. Substep energy over time
    ax = axes[1]
    for name, grp in sub.groupby("substep"):
        ax.scatter(grp["elapsed_s"], grp["energy_mj"], s=2, alpha=0.4,
                   label=name, color=colors.get(name, "gray"))
    if len(energy_spike_idx):
        ax.scatter(sub.loc[energy_spike_idx, "elapsed_s"], sub.loc[energy_spike_idx, "energy_mj"],
                   s=40, color="red", zorder=5, label="energy spike")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Energy (mJ)")
    ax.set_title("Substep Energy Over Time")
    ax.legend(markerscale=4, fontsize=8)

    # 3. GPU power over time
    ax = axes[2]
    pow_col = [c for c in gpu.columns if "power" in c.lower()][0]
    ax.plot(gpu["elapsed_s"], gpu[pow_col], color="purple", linewidth=0.8, label="GPU power (W)")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("GPU Power (W)")
    ax.set_title("GPU Power Draw Over Time")
    ax.legend(fontsize=8)

    # 4. GPU utilization over time
    ax = axes[3]
    ax.plot(gpu["elapsed_s"], gpu[gpu_col], color="teal", linewidth=0.8, label="GPU util (%)")
    if len(util_dip_idx):
        ax.scatter(gpu.loc[util_dip_idx, "elapsed_s"], gpu.loc[util_dip_idx, gpu_col],
                   s=40, color="red", zorder=5, label="util dip")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_title("GPU Utilization Over Time")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT, f"anomaly_bs{bs}_run{run}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    for bs in [32, 16, 8]:
        analyze_batch(bs)
    print("\nDone.")
