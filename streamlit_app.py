# streamlit_app.py

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

EXPERIMENT_DIR = Path("experiments")

DATASET_LABELS = {
    "small_dataset": "Small (10K rows)",
    "medium_dataset": "Medium (100K rows)",
    "large_dataset": "Large (1M rows)",
}

PIPELINE_LABELS = {
    "pipeline_a": "Pipeline A",
    "pipeline_b": "Pipeline B",
    "pipeline_c": "Pipeline C",
}


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_scenario_data(dataset: str, pipeline: str):
    base_name = f"{dataset}_{pipeline}"
    train_path = EXPERIMENT_DIR / f"train_{base_name}.csv"
    baseline_path = EXPERIMENT_DIR / f"baseline_{base_name}.csv"
    eval_path = EXPERIMENT_DIR / f"eval_{base_name}.csv"

    train_df = load_csv_safe(train_path)
    baseline_df = load_csv_safe(baseline_path)
    eval_df = load_csv_safe(eval_path)

    return train_df, baseline_df, eval_df


def load_percentage_improvements():
    path = EXPERIMENT_DIR / "percentage_improvements.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def pct_improvement(baseline: float, rl: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - rl) / baseline * 100.0


def main():
    st.set_page_config(
        page_title="DataFlowRL Dashboard",
        layout="wide",
    )

    st.title("📊 DataFlowRL – RL Results Dashboard")
    st.markdown(
        """
Interactive dashboard to explore the performance of **DataFlowRL** across
different datasets and pipeline configurations.

Use the sidebar to select a scenario; the metrics, charts, and summary
will update automatically.
        """
    )

    # ----- SIDEBAR -----
    st.sidebar.header("Scenario Selection")

    dataset = st.sidebar.selectbox(
        "Dataset",
        list(DATASET_LABELS.keys()),
        format_func=lambda k: DATASET_LABELS[k],
    )

    pipeline = st.sidebar.selectbox(
        "Pipeline",
        list(PIPELINE_LABELS.keys()),
        format_func=lambda k: PIPELINE_LABELS[k],
    )

    st.sidebar.info(
        f"Showing results for **{DATASET_LABELS[dataset]}**, "
        f"**{PIPELINE_LABELS[pipeline]}**"
    )

    # ----- LOAD DATA -----
    train_df, baseline_df, eval_df = load_scenario_data(dataset, pipeline)
    pct_df = load_percentage_improvements()

    base_name = f"{dataset}_{pipeline}"

    if train_df is None or baseline_df is None or eval_df is None:
        st.error(
            f"No experiment data found for `{base_name}`.\n\n"
            "Make sure you ran `python -m src.run_experiments` first."
        )
        return

    # ----- METRICS SECTION -----
    st.subheader("Key Metrics – Baseline vs RL")

    baseline_runtime = baseline_df["avg_runtime"].mean()
    baseline_cost = baseline_df["avg_cost"].mean()
    baseline_quality = baseline_df["avg_quality"].mean()

    rl_runtime = eval_df["avg_runtime"].mean()
    rl_cost = eval_df["avg_cost"].mean()
    rl_quality = eval_df["avg_quality"].mean()

    runtime_improve = pct_improvement(baseline_runtime, rl_runtime)
    cost_improve = pct_improvement(baseline_cost, rl_cost)
    quality_delta_pts = (rl_quality - baseline_quality) * 100.0  # in points

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏱ Avg Runtime (RL)",
            f"{rl_runtime:.2f}",
            f"{runtime_improve:.1f}% vs baseline",
        )
        st.caption(f"Baseline: {baseline_runtime:.2f}")

    with col2:
        st.metric(
            "💰 Avg Cost (RL)",
            f"{rl_cost:.2f}",
            f"{cost_improve:.1f}% vs baseline",
        )
        st.caption(f"Baseline: {baseline_cost:.2f}")

    with col3:
        st.metric(
            "✅ Avg Quality (RL)",
            f"{rl_quality:.3f}",
            f"{quality_delta_pts:+.2f} pts vs baseline",
        )
        st.caption(f"Baseline: {baseline_quality:.3f}")

    st.markdown("---")

    # ----- TRAINING CURVE -----
    st.subheader("📈 Training Reward Curve")

    if "episode" in train_df.columns and "total_reward" in train_df.columns:
        train_plot_df = train_df[["episode", "total_reward"]].set_index("episode")
        st.line_chart(train_plot_df)
    else:
        st.warning("Training CSV missing `episode` or `total_reward` columns.")

    st.markdown("---")

    # ----- GROUPED BAR WITH IMPROVEMENT LABELS -----
    st.subheader("🏁 Baseline vs RL — Runtime, Cost, Quality (with Improvements)")

    metrics_data = pd.DataFrame(
        {
            "Metric": ["Runtime", "Cost", "Quality"],
            "Baseline": [baseline_runtime, baseline_cost, baseline_quality],
            "RL": [rl_runtime, rl_cost, rl_quality],
        }
    )

    # improvement: runtime & cost as %; quality as delta points
    improvements = [runtime_improve, cost_improve, quality_delta_pts]

    # Melt to long form for grouped bars
    melted = metrics_data.melt(
        id_vars="Metric",
        value_vars=["Baseline", "RL"],
        var_name="Model",
        value_name="Value",
    )

    # Base grouped bar chart (xOffset = group by Model)
    bar_chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            xOffset="Model:N",
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(range=["#4a90e2", "#f5a623"]),
                legend=alt.Legend(title="Model"),
            ),
            tooltip=["Metric", "Model", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=320)
    )

    # Labels only for RL bars
    labels_df = pd.DataFrame(
        {
            "Metric": ["Runtime", "Cost", "Quality"],
            "Model": ["RL", "RL", "RL"],
            "Value": [rl_runtime, rl_cost, rl_quality],
            "Improvement": improvements,
        }
    )

    labels_chart = (
        alt.Chart(labels_df)
        .mark_text(
            dy=-8,
            fontSize=13,
            fontWeight="bold",
            color="#f5a623",
        )
        .encode(
            x="Metric:N",
            xOffset="Model:N",
            y="Value:Q",
            text=alt.Text("Improvement:Q", format="+.1f'%'"),
        )
    )

    final_chart = bar_chart + labels_chart
    st.altair_chart(final_chart, use_container_width=True)

    st.markdown("---")

    # ----- SCENARIO SUMMARY -----
    st.subheader("📊 Scenario Summary")

    if pct_df is not None:
        row = pct_df[
            (pct_df["dataset"] == dataset) & (pct_df["pipeline"] == pipeline)
        ]
        if not row.empty:
            st.write("Percentage improvements for this scenario:")
            st.dataframe(row.reset_index(drop=True))
        else:
            st.info("No percentage_improvements entry found for this scenario.")
    else:
        st.info(
            "percentage_improvements.csv not found. "
            "Run `python -m src.analysis_full` to generate it."
        )

    # ----- RAW DETAILS -----
    with st.expander("Show raw episode data (baseline / RL eval / training)"):
        st.write("**Baseline episodes**")
        st.dataframe(baseline_df.head())

        st.write("**RL evaluation episodes**")
        st.dataframe(eval_df.head())

        st.write("**Training episodes**")
        st.dataframe(train_df.head())


if __name__ == "__main__":
    main()
