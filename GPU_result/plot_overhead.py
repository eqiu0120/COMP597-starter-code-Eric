# Bar chart comparing per-step time overhead across experiments and batch sizes.
# Usage: python GPU_result/plot_overhead.py --out_dir GPU_result/figures

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ms/step from log analysis (mean over 3 runs except Exp4 which is 1 run)
DATA = {
    "bs=32": {"Exp1 (noop)": 355, "Exp2 (cc_full)": 358, "Exp3 (combined)": 403, "Exp4 (per_step)": 357},
    "bs=16": {"Exp1 (noop)": 216, "Exp2 (cc_full)": 216, "Exp3 (combined)": 262, "Exp4 (per_step)": 221},
    "bs=8":  {"Exp1 (noop)": 148, "Exp2 (cc_full)": 147, "Exp3 (combined)": 185, "Exp4 (per_step)": 149},
}

OVERHEAD = {
    "bs=32": {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": 0.8, "Exp3 (combined)": 13.5, "Exp4 (per_step)": 0.6},
    "bs=16": {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": 0.0, "Exp3 (combined)": 21.3, "Exp4 (per_step)": 2.3},
    "bs=8":  {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": 0.0, "Exp3 (combined)": 25.0, "Exp4 (per_step)": 0.7},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="GPU_result/comparison")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    batch_sizes = list(DATA.keys())
    exps = list(next(iter(DATA.values())).keys())
    x = np.arange(len(batch_sizes))
    width = 0.18
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # Figure 1: absolute ms/step
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, exp in enumerate(exps):
        vals = [DATA[bs][exp] for bs in batch_sizes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=exp, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylabel("ms / step")
    ax.set_title("Per-Step Time by Experiment")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(v for d in DATA.values() for v in d.values()) * 1.22)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "overhead_perstep_time.png")
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"Saved: {out1}")

    # Figure 2: overhead %
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, exp in enumerate(exps):
        vals = [OVERHEAD[bs][exp] for bs in batch_sizes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=exp, color=colors[i])
        for bar, v in zip(bars, vals):
            if v >= 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"+{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.axhline(5, color="red", linestyle="--", linewidth=1, label="5% target")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylabel("Overhead vs. Exp1 (%)")
    ax.set_title("Time Overhead by Experiment")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(v for d in OVERHEAD.values() for v in d.values()) * 1.22)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "overhead_percent.png")
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
