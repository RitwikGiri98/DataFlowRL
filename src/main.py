# src/main.py

from typing import List, Dict, Any

import pandas as pd

from src.controller.controller import Controller
from src.rl.dqn_agent import DQNAgent
from src.baselines import run_heuristic_baseline


def get_sample_pipeline():
    """
    Simple linear pipeline with 4 steps.
    """
    return {
        "name": "demo_pipeline",
        "steps": [
            {"name": "extract", "type": "extract", "complexity": "low"},
            {"name": "clean", "type": "transform", "complexity": "medium"},
            {"name": "aggregate", "type": "aggregate", "complexity": "high"},
            {"name": "load", "type": "load", "complexity": "low"},
        ],
    }


def get_dataset_meta():
    """
    Synthetic dataset metadata.
    """
    return {
        "name": "demo_dataset",
        "num_rows": 100_000,
        "num_features": 30,
    }


def summarize_episode(ep_idx: int, mode: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a flat summary dict for one episode: total reward, average runtime/cost/quality.
    """
    steps = result["steps"]
    if len(steps) == 0:
        avg_runtime = 0.0
        avg_cost = 0.0
        avg_quality = 0.0
    else:
        avg_runtime = sum(s["runtime"] for s in steps) / len(steps)
        avg_cost = sum(s["cost"] for s in steps) / len(steps)
        avg_quality = sum(s["quality"] for s in steps) / len(steps)

    return {
        "episode": ep_idx,
        "mode": mode,
        "total_reward": result["total_reward"],
        "avg_runtime": avg_runtime,
        "avg_cost": avg_cost,
        "avg_quality": avg_quality,
    }


def main():
    pipeline_def = get_sample_pipeline()
    dataset_meta = get_dataset_meta()

    state_dim = 10   # must match PipelineEnv.state_dim
    action_dim = 9   # 3 tools * 3 configs

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

    # Choose mode: "dqn", "ucb", or "hybrid"
    mode = "hybrid"

    controller = Controller(
        rl_agent=rl_agent,
        pipeline_def=pipeline_def,
        dataset_meta=dataset_meta,
        mode=mode,
    )

    # 1) TRAINING LOOP (RL)
    num_train_episodes = 50
    print(f"=== Training RL controller (mode='{mode}') for {num_train_episodes} episodes ===")

    train_summaries: List[Dict[str, Any]] = []

    for episode in range(1, num_train_episodes + 1):
        result = controller.run_episode(train=True)
        summary = summarize_episode(episode, f"train_{mode}", result)
        train_summaries.append(summary)
        print(
            f"[TRAIN] Episode {episode}/{num_train_episodes} | "
            f"Total reward: {summary['total_reward']:.3f} | "
            f"Avg runtime: {summary['avg_runtime']:.2f} | "
            f"Avg cost: {summary['avg_cost']:.2f} | "
            f"Avg quality: {summary['avg_quality']:.3f}"
        )

    # Save training history
    train_df = pd.DataFrame(train_summaries)
    train_df.to_csv("train_history.csv", index=False)
    print("\nSaved training history to 'train_history.csv'.")

    # 2) HEURISTIC BASELINE
    num_baseline_episodes = 20
    print(f"\n=== Running heuristic baseline for {num_baseline_episodes} episodes ===")
    baseline_results = run_heuristic_baseline(
        pipeline_def=pipeline_def,
        dataset_meta=dataset_meta,
        num_episodes=num_baseline_episodes,
    )

    baseline_summaries: List[Dict[str, Any]] = []
    for i, result in enumerate(baseline_results, start=1):
        summary = summarize_episode(i, "baseline_heuristic", result)
        baseline_summaries.append(summary)
        print(
            f"[BASELINE] Episode {i}/{num_baseline_episodes} | "
            f"Total reward: {summary['total_reward']:.3f} | "
            f"Avg runtime: {summary['avg_runtime']:.2f} | "
            f"Avg cost: {summary['avg_cost']:.2f} | "
            f"Avg quality: {summary['avg_quality']:.3f}"
        )

    baseline_df = pd.DataFrame(baseline_summaries)
    baseline_df.to_csv("baseline_history.csv", index=False)
    print("Saved baseline history to 'baseline_history.csv'.")

    # 3) EVALUATION OF TRAINED RL (no exploration)
    num_eval_episodes = 20
    print(f"\n=== Evaluating trained RL policy for {num_eval_episodes} episodes ===")

    eval_summaries: List[Dict[str, Any]] = []
    for episode in range(1, num_eval_episodes + 1):
        # train=False -> DQN greedy, no epsilon, bandit not used
        result = controller.run_episode(train=False)
        summary = summarize_episode(episode, f"eval_{mode}", result)
        eval_summaries.append(summary)
        print(
            f"[EVAL] Episode {episode}/{num_eval_episodes} | "
            f"Total reward: {summary['total_reward']:.3f} | "
            f"Avg runtime: {summary['avg_runtime']:.2f} | "
            f"Avg cost: {summary['avg_cost']:.2f} | "
            f"Avg quality: {summary['avg_quality']:.3f}"
        )

    eval_df = pd.DataFrame(eval_summaries)
    eval_df.to_csv("eval_history.csv", index=False)
    print("Saved evaluation history to 'eval_history.csv'.")


if __name__ == "__main__":
    main()
