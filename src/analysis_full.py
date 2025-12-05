# src/analysis_full.py

from pathlib import Path
import pandas as pd


def main():
    exp_dir = Path("experiments")
    summary_path = exp_dir / "summary_all_scenarios.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. "
            "Run `python -m src.plot_experiments` first to generate it."
        )

    # ---------- 1) Load base summary ----------
    df = pd.read_csv(summary_path)
    print("=== Loaded summary_all_scenarios.csv ===")
    print(df.head())

    required_cols = [
        "dataset",
        "pipeline",
        "baseline_avg_runtime",
        "baseline_avg_cost",
        "baseline_avg_quality",
        "rl_avg_runtime",
        "rl_avg_cost",
        "rl_avg_quality",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column in summary CSV: {col}")

    # ---------- 2) Compute raw deltas (RL - Baseline) ----------
    df["delta_runtime"] = df["rl_avg_runtime"] - df["baseline_avg_runtime"]
    df["delta_cost"] = df["rl_avg_cost"] - df["baseline_avg_cost"]
    df["delta_quality"] = df["rl_avg_quality"] - df["baseline_avg_quality"]

    # Save extended summary with deltas
    extended_path = exp_dir / "summary_all_scenarios_with_deltas.csv"
    df.to_csv(extended_path, index=False)
    print(f"\nSaved extended summary with deltas to: {extended_path}")

    # ---------- 3) Compute percentage improvements per scenario ----------
    # runtime/cost improvement: (baseline - rl) / baseline * 100 (higher is better)
    # quality change: (rl - baseline) * 100 (since it's already 0-1 scaled)
    improvements = []

    for _, row in df.iterrows():
        baseline_runtime = row["baseline_avg_runtime"]
        rl_runtime = row["rl_avg_runtime"]

        baseline_cost = row["baseline_avg_cost"]
        rl_cost = row["rl_avg_cost"]

        baseline_quality = row["baseline_avg_quality"]
        rl_quality = row["rl_avg_quality"]

        # Guard against division by zero
        if baseline_runtime == 0 or baseline_cost == 0:
            continue

        runtime_pct = ((baseline_runtime - rl_runtime) / baseline_runtime) * 100.0
        cost_pct = ((baseline_cost - rl_cost) / baseline_cost) * 100.0
        quality_pct = (rl_quality - baseline_quality) * 100.0

        improvements.append(
            {
                "dataset": row["dataset"],
                "pipeline": row["pipeline"],
                "runtime_pct_improvement": runtime_pct,
                "cost_pct_improvement": cost_pct,
                "quality_pct_change": quality_pct,
            }
        )

    imp_df = pd.DataFrame(improvements)

    pct_path = exp_dir / "percentage_improvements.csv"
    imp_df.to_csv(pct_path, index=False)
    print(f"Saved per-scenario percentage improvements to: {pct_path}")

    # ---------- 4) Print aggregate stats you can quote in the report ----------
    print("\n=== Per-scenario percentage improvements ===")
    print(imp_df)

    means = imp_df[["runtime_pct_improvement", "cost_pct_improvement", "quality_pct_change"]].mean()
    print("\n=== Average percentage improvements across all scenarios ===")
    print(means)

    print("\nInterpretation:")
    print(
        f"- Runtime: RL is faster by about {means['runtime_pct_improvement']:.2f}% on average."
    )
    print(
        f"- Cost: RL is cheaper by about {means['cost_pct_improvement']:.2f}% on average."
    )
    print(
        f"- Quality: RL changes quality by about {means['quality_pct_change']:.2f} percentage points on average."
    )
    print(
        "\nUse these three numbers directly in your report text "
        "(Experimental Results / Statistical Validation section)."
    )


if __name__ == "__main__":
    main()
