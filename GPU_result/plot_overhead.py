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
    "bs=32": {"Exp1 (noop)": 355, "Exp2 (cc_full)": 358, "Exp4 (per_step)": 357, "Exp3 (combined)": 403},
    "bs=16": {"Exp1 (noop)": 216, "Exp2 (cc_full)": 216, "Exp4 (per_step)": 221, "Exp3 (combined)": 262},
    "bs=8":  {"Exp1 (noop)": 148, "Exp2 (cc_full)": 147, "Exp4 (per_step)": 149, "Exp3 (combined)": 185},
}

OVERHEAD = {
    "bs=32": {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": 0.8, "Exp4 (per_step)": 0.6, "Exp3 (combined)": 13.5},
    "bs=16": {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": 0.0, "Exp4 (per_step)": 2.3, "Exp3 (combined)": 21.3},
    "bs=8":  {"Exp1 (noop)": 0.0, "Exp2 (cc_full)": -0.7, "Exp4 (per_step)": 0.7, "Exp3 (combined)": 25.0},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="GPU_result/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    batch_sizes = list(DATA.keys())
    exps = list(next(iter(DATA.values())).keys())
    x = np.arange(len(batch_sizes))
    width = 0.18
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: absolute ms/step
    ax = axes[0]
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

    # Right panel: overhead %
    ax = axes[1]
    for i, exp in enumerate(exps):
        vals = [OVERHEAD[bs][exp] for bs in batch_sizes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=exp, color=colors[i])
        for bar, v in zip(bars, vals):
            if abs(v) >= 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:+.1f}%", ha="center", va="bottom", fontsize=7)
    ax.axhline(5, color="red", linestyle="--", linewidth=1, label="5% target")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylabel("Overhead vs. Exp1 (%)")
    ax.set_title("Time Overhead by Experiment")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, max(v for d in OVERHEAD.values() for v in d.values()) * 1.22)

    fig.tight_layout()
    out = os.path.join(args.out_dir, "overhead_comparison.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
