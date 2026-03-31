# GPU utilization dip analysis: correlates nvidia-smi timeline with per-substep data.
# Usage: python GPU_result/analyze_gpu_util_dips.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), "raw_csvs")
OUT  = os.path.join(os.path.dirname(__file__), "plots", "anomaly_analysis")
os.makedirs(OUT, exist_ok=True)

JOB_IDS = {
    32: [16142, 16145, 16204],
    16: [16148, 16150, 16151],
    8:  [16182, 16183, 16189],
}
ANALYSIS_RUN = 0


def parse_ts(ts_str):
    ts_str = str(ts_str).strip()
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"]:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse: {ts_str}")


def load_gpu_csv(bs, job_id):
    path = os.path.join(BASE, f"bs_{bs}", f"gpu_{job_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col != "timestamp":
            df[col] = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = df["timestamp"].apply(parse_ts)
    t0 = df["ts"].min()
    df["elapsed_s"] = (df["ts"] - t0).dt.total_seconds()
    return df


def load_substep_csv(bs, run=0):
    path = os.path.join(BASE, f"bs_{bs}", f"run_{run}_cc_substep_rank_0-substeps.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["ts"] = df["timestamp"].apply(parse_ts)
    t0 = df["ts"].min()
    df["elapsed_s"] = (df["ts"] - t0).dt.total_seconds()
    df["duration_ms"] = df["duration"] * 1000
    df["substep"] = df["task_name"].str.extract(r"(Forward|Backward|Optim)")
    return df


def analyze_gpu_dips(bs):
    run = ANALYSIS_RUN
    job_id = JOB_IDS[bs][run]
    print(f"\n{'='*60}")
    print(f"  GPU Util Dip Analysis — batch size {bs} — run {run} (job {job_id})")
    print(f"{'='*60}")

    gpu = load_gpu_csv(bs, job_id)
    sub = load_substep_csv(bs, run)
    if gpu is None or sub is None:
        print("  Missing data, skipping.")
        return

    util_col = [c for c in gpu.columns if "utilization" in c.lower()][0]
    pow_col  = [c for c in gpu.columns if "power" in c.lower()][0]

    steady = gpu[(gpu["elapsed_s"] > 20) & (gpu["elapsed_s"] < 305)].copy()
    mean_util = steady[util_col].mean()
    std_util  = steady[util_col].std()

    print(f"\nSteady-state GPU util (t=20–305s):")
    print(f"  Mean:   {mean_util:.1f}%")
    print(f"  Std:    {std_util:.1f}%")
    print(f"  Min:    {steady[util_col].min():.1f}%")
    print(f"  Max:    {steady[util_col].max():.1f}%")

    dip_thresh = mean_util - std_util
    dips = steady[steady[util_col] < dip_thresh].copy()
    print(f"\nDip threshold: <{dip_thresh:.1f}% (mean - 1σ)")
    print(f"Total dip events: {len(dips)}")

    if len(dips) > 0:
        dips = dips.reset_index(drop=True)
        dips["gap"] = dips["elapsed_s"].diff().fillna(999)
        dip_groups = (dips["gap"] > 2).cumsum()
        dip_summary = dips.groupby(dip_groups).agg(
            t_start=("elapsed_s", "min"),
            t_end=("elapsed_s", "max"),
            min_util=(util_col, "min"),
            mean_util_dip=(util_col, "mean"),
            n_samples=(util_col, "count"),
        ).reset_index(drop=True)
        print(f"\nClustered dip events: {len(dip_summary)}")
        print(dip_summary.to_string(index=False))

        if len(dip_summary) > 1:
            intervals = dip_summary["t_start"].diff().dropna()
            print(f"\nInter-dip interval stats (s):")
            print(f"  Mean:   {intervals.mean():.1f}s")
            print(f"  Std:    {intervals.std():.1f}s")
            print(f"  Min:    {intervals.min():.1f}s")
            print(f"  Max:    {intervals.max():.1f}s")

        if sub is not None:
            print(f"\nCorrelating dip times with training phases...")
            steps_per_epoch = 2000 // bs
            epoch_times = []
            for i in range(steps_per_epoch, len(sub)//3, steps_per_epoch):
                idx = i * 3
                if idx < len(sub):
                    epoch_times.append(sub.iloc[idx]["elapsed_s"])

            near_epoch = 0
            near_optim = 0
            near_data  = 0

            optim_rows = sub[sub["task_name"].str.contains("Optim", na=False)]
            optim_times = optim_rows["elapsed_s"].values

            for _, dip in dip_summary.iterrows():
                t = dip["t_start"]
                # Check if near epoch boundary
                if epoch_times:
                    nearest_epoch = min(epoch_times, key=lambda e: abs(e - t))
                    if abs(t - nearest_epoch) < 5:
                        near_epoch += 1
                        continue
                # Check if near optimizer step
                if len(optim_times):
                    nearest_optim = min(optim_times, key=lambda o: abs(o - t))
                    if abs(t - nearest_optim) < 1:
                        near_optim += 1
                        continue
                near_data += 1

            print(f"  Near epoch boundary (±5s): {near_epoch}")
            print(f"  Near optimizer step (±1s): {near_optim}")
            print(f"  Other (data loading gap):  {near_data}")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"GPU Utilization Dip Analysis — batch size {bs}", fontsize=13)

    steps_per_epoch = 2000 // bs
    epoch_times = []
    for i in range(steps_per_epoch, len(sub)//3, steps_per_epoch):
        idx = i * 3
        if idx < len(sub):
            epoch_times.append(sub.iloc[idx]["elapsed_s"])

    ax = axes[0]
    ax.plot(gpu["elapsed_s"], gpu[util_col], color="teal", linewidth=0.7, label="GPU util (%)")
    ax.axhline(dip_thresh, color="red", linestyle="--", linewidth=1,
               label=f"dip threshold ({dip_thresh:.0f}%)")
    ax.axhline(mean_util, color="gray", linestyle=":", linewidth=1,
               label=f"mean ({mean_util:.0f}%)")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.4, linewidth=0.8)
    if len(dips) > 0:
        ax.scatter(dips["elapsed_s"], dips[util_col], s=8, color="red",
                   zorder=5, label="dip sample")
    ax.axvspan(0, 20, alpha=0.1, color="gray", label="warmup")
    ax.set_ylabel("GPU Util (%)")
    ax.set_title("GPU Utilization — all dips highlighted (orange = epoch boundary)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(-5, 105)

    ax = axes[1]
    zoom = gpu[gpu["elapsed_s"] <= 100]
    ax.plot(zoom["elapsed_s"], zoom[util_col], color="teal", linewidth=1, label="GPU util (%)")
    ax.axhline(dip_thresh, color="red", linestyle="--", linewidth=1)
    ax.axhline(mean_util, color="gray", linestyle=":", linewidth=1)
    for et in [e for e in epoch_times if e <= 100]:
        ax.axvline(et, color="orange", alpha=0.5, linewidth=1, label="epoch boundary")

    optim_zoom = sub[(sub["task_name"].str.contains("Optim", na=False)) &
                     (sub["elapsed_s"] <= 100)]
    for _, row in optim_zoom.iterrows():
        ax.axvspan(row["elapsed_s"], row["elapsed_s"] + row["duration_ms"]/1000,
                   alpha=0.15, color="purple")
    purple_patch = mpatches.Patch(color="purple", alpha=0.3, label="optimizer step")
    ax.legend(handles=[purple_patch], fontsize=7)
    ax.set_ylabel("GPU Util (%)")
    ax.set_title("Zoomed: first 100s — purple = optimizer step duration")
    ax.set_ylim(-5, 105)

    ax = axes[2]
    ax.plot(gpu["elapsed_s"], gpu[pow_col], color="purple", linewidth=0.7, label="GPU power (W)")
    for et in epoch_times:
        ax.axvline(et, color="orange", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("GPU Power (W)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_title("GPU Power Draw (orange = epoch boundary)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUT, f"gpu_util_dips_bs{bs}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    for bs in [32, 16, 8]:
        analyze_gpu_dips(bs)
    print("\nDone.")
