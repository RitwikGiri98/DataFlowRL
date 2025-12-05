# src/analysis_summary.py

from pathlib import Path
import pandas as pd

SUMMARY_PATH = Path("experiments/summary_all_scenarios.csv")


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} not found. Run plot_experiments.py first.")

    df = pd.read_csv(SUMMARY_PATH)
    print("=== Per-scenario summary (baseline vs RL) ===")
    print(df)

    # Compute deltas: RL - Baseline
    df["delta_runtime"] = df["rl_avg_runtime"] - df["baseline_avg_runtime"]
    df["delta_cost"] = df["rl_avg_cost"] - df["baseline_avg_cost"]
    df["delta_quality"] = df["rl_avg_quality"] - df["baseline_avg_quality"]

    print("\n=== Average deltas across all datasets & pipelines (RL - Baseline) ===")
    avg = df[["delta_runtime", "delta_cost", "delta_quality"]].mean()
    print(avg)

    print("\n(negative delta_runtime / delta_cost = RL is better; values near 0 for quality = similar quality)")

    # Save expanded summary with deltas
    out_path = SUMMARY_PATH.parent / "summary_all_scenarios_with_deltas.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved extended summary with deltas to: {out_path}")


if __name__ == "__main__":
    main()
