# src/run_experiments.py

import pandas as pd
from pathlib import Path

from src.controller.controller import Controller
from src.rl.dqn_agent import DQNAgent
from src.baselines import run_heuristic_baseline
from src.datasets import get_small_dataset, get_medium_dataset, get_large_dataset
from src.pipelines import pipeline_a, pipeline_b, pipeline_c

OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)


def summarize(ep, mode, result):
    steps = result["steps"]
    if len(steps) == 0:
        avg_runtime = avg_cost = avg_quality = 0.0
    else:
        avg_runtime = sum(s["runtime"] for s in steps) / len(steps)
        avg_cost = sum(s["cost"] for s in steps) / len(steps)
        avg_quality = sum(s["quality"] for s in steps) / len(steps)

    return {
        "episode": ep,
        "mode": mode,
        "total_reward": result["total_reward"],
        "avg_runtime": avg_runtime,
        "avg_cost": avg_cost,
        "avg_quality": avg_quality,
    }


def run_single_experiment(dataset_meta, pipeline_def):
    print(f"\n=== Running experiment: {dataset_meta['name']} × {pipeline_def['name']} ===")

    state_dim = 10
    action_dim = 9

    rl_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10_000,
        target_update_freq=10,
    )

    controller = Controller(
        rl_agent=rl_agent,
        pipeline_def=pipeline_def,
        dataset_meta=dataset_meta,
        mode="hybrid",
    )

    ### TRAIN RL ###
    train_summaries = []
    for ep in range(1, 40):  # 40 episodes per experiment
        result = controller.run_episode(train=True)
        train_summaries.append(summarize(ep, "train_rl", result))

    ### BASELINE ###
    baseline_summaries = []
    baseline_results = run_heuristic_baseline(
        pipeline_def=pipeline_def,
        dataset_meta=dataset_meta,
        num_episodes=20,
    )
    for i, result in enumerate(baseline_results, 1):
        baseline_summaries.append(summarize(i, "baseline", result))

    ### EVAL RL ###
    eval_summaries = []
    for ep in range(1, 20):
        result = controller.run_episode(train=False)
        eval_summaries.append(summarize(ep, "eval_rl", result))

    #### SAVE FILES ####
    base_name = f"{dataset_meta['name']}_{pipeline_def['name']}"

    pd.DataFrame(train_summaries).to_csv(OUTPUT_DIR / f"train_{base_name}.csv", index=False)
    pd.DataFrame(baseline_summaries).to_csv(OUTPUT_DIR / f"baseline_{base_name}.csv", index=False)
    pd.DataFrame(eval_summaries).to_csv(OUTPUT_DIR / f"eval_{base_name}.csv", index=False)

    print(f"Saved CSVs for experiment: {base_name}")


def main():
    datasets = [get_small_dataset(), get_medium_dataset(), get_large_dataset()]
    pipelines = [pipeline_a(), pipeline_b(), pipeline_c()]

    for dataset in datasets:
        for pipe in pipelines:
            run_single_experiment(dataset, pipe)


if __name__ == "__main__":
    main()
