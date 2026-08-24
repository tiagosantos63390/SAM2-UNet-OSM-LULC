"""Render the test-set comparison chart from the per-seed metrics JSONs in
results/history/. Writes results/test_comparison.png.

Run from the repository root:
    python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HISTORY_DIR = RESULTS_DIR / "history"

CONFIGS = [
    ("Baseline SAM2-UNet", "baseline"),
    ("OSM Auxiliary DT (Roads Only)", "roads_only_L0p5"),
    ("OSM Auxiliary DT (Roads+Buildings)", "roads_bldg_L0p25"),
]
SEEDS = [42, 43, 44]
METRICS = [("miou", "mIoU"), ("f1", "F1"), ("bmiou", "Boundary mIoU")]


def main():
    means, stds = [], []
    labels = []
    for label, base in CONFIGS:
        values = {key: [] for key, _ in METRICS}
        for seed in SEEDS:
            path = HISTORY_DIR / f"{base}_seed{seed}_metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            test = json.load(open(path, encoding="utf-8"))["test"]
            for key, _ in METRICS:
                values[key].append(test[key])
        means.append([np.mean(values[key]) * 100 for key, _ in METRICS])
        stds.append([np.std(values[key]) * 100 for key, _ in METRICS])
        labels.append(label)

    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, color) in enumerate(zip(labels, ["#7f8fa6", "#2f6f4f", "#b8860b"])):
        ax.bar(x + (i - 1) * width, means[i], width, yerr=stds[i], capsize=4,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in METRICS])
    ax.set_ylim(50, 85)
    ax.set_ylabel("Score (%)")
    ax.set_title("OpenEarthMap test set (mean over seeds 42, 43, 44)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = RESULTS_DIR / "test_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
