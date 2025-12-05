# src/plot_experiments.py

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path("experiments")


def load_history(prefix: str, dataset_name: str, pipeline_name: str) -> pd.DataFrame:
    """
    Load one of: train_*, baseline_*, eval_* CSVs.
    """
    filename = f"{prefix}_{dataset_name}_{pipeline_name}.csv"
    path = EXPERIMENT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def plot_training_rewards(train_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """
    Line plot of total reward vs episode for a single scenario.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(train_df["episode"], train_df["total_reward"])
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_baseline_vs_rl_metrics(
    baseline_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    dataset_name: str,
    pipeline_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Bar chart comparing Baseline vs RL on avg_runtime, avg_cost, avg_quality.
    Returns a dict of summary metrics.
    """
    metrics = ["avg_runtime", "avg_cost", "avg_quality"]
    metric_labels = ["Runtime", "Cost", "Quality"]

    baseline_means = [baseline_df[m].mean() for m in metrics]
    baseline_stds = [baseline_df[m].std() for m in metrics]
    rl_means = [eval_df[m].mean() for m in metrics]
    rl_stds = [eval_df[m].std() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    # Matplotlib will choose default colors – we don't specify them
    plt.bar(x - width / 2, baseline_means, width, yerr=baseline_stds, capsize=4, label="Baseline")
    plt.bar(x + width / 2, rl_means, width, yerr=rl_stds, capsize=4, label="RL (Eval)")

    plt.xticks(x, metric_labels)
    plt.ylabel("Value")
    plt.title(f"Baseline vs RL – {dataset_name}, {pipeline_name}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    summary = {
        "dataset": dataset_name,
        "pipeline": pipeline_name,
        "baseline_avg_runtime": baseline_means[0],
        "baseline_avg_cost": baseline_means[1],
        "baseline_avg_quality": baseline_means[2],
        "rl_avg_runtime": rl_means[0],
        "rl_avg_cost": rl_means[1],
        "rl_avg_quality": rl_means[2],
    }
    return summary


def main():
    if not EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Directory '{EXPERIMENT_DIR}' not found. Run run_experiments.py first.")

    # You can edit these lists if you add more
    dataset_names = ["small_dataset", "medium_dataset", "large_dataset"]
    pipeline_names = ["pipeline_a", "pipeline_b", "pipeline_c"]

    all_summaries: List[Dict[str, Any]] = []

    for ds in dataset_names:
        for pipe in pipeline_names:
            base_name = f"{ds}_{pipe}"
            print(f"\n=== Visualizing: {base_name} ===")

            # Load histories
            train_df = load_history("train", ds, pipe)
            baseline_df = load_history("baseline", ds, pipe)
            eval_df = load_history("eval", ds, pipe)

            # 1) Training reward curve
            train_title = f"Training Reward – {ds}, {pipe}"
            train_png = EXPERIMENT_DIR / f"train_rewards_{base_name}.png"
            plot_training_rewards(train_df, train_title, train_png)
            print(f"  Saved training curve → {train_png}")

            # 2) Baseline vs RL metrics comparison
            metrics_png = EXPERIMENT_DIR / f"metrics_comparison_{base_name}.png"
            summary = plot_baseline_vs_rl_metrics(
                baseline_df=baseline_df,
                eval_df=eval_df,
                dataset_name=ds,
                pipeline_name=pipe,
                output_path=metrics_png,
            )
            all_summaries.append(summary)
            print(f"  Saved metrics comparison → {metrics_png}")

    # Save overall summary CSV
    summary_df = pd.DataFrame(all_summaries)
    summary_csv_path = EXPERIMENT_DIR / "summary_all_scenarios.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved cross-scenario summary to: {summary_csv_path}")


if __name__ == "__main__":
    main()
