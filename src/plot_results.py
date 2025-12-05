# src/plot_results.py

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "total_reward" not in df.columns:
        raise ValueError(f"CSV {path} does not look like an episode history file.")
    return df


def main():
    root = Path(".").resolve()
    train_path = root / "train_history.csv"
    baseline_path = root / "baseline_history.csv"
    eval_path = root / "eval_history.csv"

    print(f"Loading:\n  {train_path}\n  {baseline_path}\n  {eval_path}")

    train_df = load_csv_safely(str(train_path))
    baseline_df = load_csv_safely(str(baseline_path))
    eval_df = load_csv_safely(str(eval_path))

    # --- Basic stats ---
    print("\n=== Basic Stats ===")
    print("Train RL (last 5 episodes):")
    print(train_df.tail()[["episode", "total_reward", "avg_runtime", "avg_cost", "avg_quality"]])

    print("\nHeuristic baseline (mean over episodes):")
    print(baseline_df[["total_reward", "avg_runtime", "avg_cost", "avg_quality"]].mean())

    print("\nTrained RL eval (mean over episodes):")
    print(eval_df[["total_reward", "avg_runtime", "avg_cost", "avg_quality"]].mean())

    # --- Create figure with 4 subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DataFlowRL – Training & Performance Comparison", fontsize=14)

    # 1) Training total reward curve
    ax = axes[0, 0]
    ax.plot(train_df["episode"], train_df["total_reward"])
    ax.set_title("Training – Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)

    # Prepare aggregate stats for baseline vs eval
    metrics = ["avg_runtime", "avg_cost", "avg_quality"]
    metric_titles = {
        "avg_runtime": "Average Runtime per Episode",
        "avg_cost": "Average Cost per Episode",
        "avg_quality": "Average Quality per Episode",
    }

    # 2) Runtime comparison (baseline vs eval)
    ax = axes[0, 1]
    metric = "avg_runtime"
    baseline_mean = baseline_df[metric].mean()
    baseline_std = baseline_df[metric].std()
    eval_mean = eval_df[metric].mean()
    eval_std = eval_df[metric].std()

    x = [0, 1]
    means = [baseline_mean, eval_mean]
    stds = [baseline_std, eval_std]
    labels = ["Baseline", "RL (Eval)"]

    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric_titles[metric])
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    # 3) Cost comparison
    ax = axes[1, 0]
    metric = "avg_cost"
    baseline_mean = baseline_df[metric].mean()
    baseline_std = baseline_df[metric].std()
    eval_mean = eval_df[metric].mean()
    eval_std = eval_df[metric].std()

    x = [0, 1]
    means = [baseline_mean, eval_mean]
    stds = [baseline_std, eval_std]
    labels = ["Baseline", "RL (Eval)"]

    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric_titles[metric])
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    # 4) Quality comparison
    ax = axes[1, 1]
    metric = "avg_quality"
    baseline_mean = baseline_df[metric].mean()
    baseline_std = baseline_df[metric].std()
    eval_mean = eval_df[metric].mean()
    eval_std = eval_df[metric].std()

    x = [0, 1]
    means = [baseline_mean, eval_mean]
    stds = [baseline_std, eval_std]
    labels = ["Baseline", "RL (Eval)"]

    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric_titles[metric])
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = root / "results_summary.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved comparison figure to: {output_path}")


if __name__ == "__main__":
    main()
