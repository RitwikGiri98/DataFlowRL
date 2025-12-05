# src/baselines.py

from typing import Dict, Any, List
from src.env.pipeline_env import PipelineEnv


def _heuristic_action(env: PipelineEnv, step: Dict[str, Any], dataset_meta: Dict[str, Any]) -> int:
    """
    Simple rule-based policy:
    - If dataset is small -> PANDAS BALANCED
    - If medium -> DUCKDB BALANCED
    - If large or step is high complexity -> SPARK HIGH_PARALLELISM
    """
    num_rows = dataset_meta.get("num_rows", 100_000)
    complexity = step.get("complexity", "medium")

    # Decide tool
    if num_rows < 50_000:
        tool = "PANDAS"
    elif num_rows < 500_000:
        tool = "DUCKDB"
    else:
        tool = "SPARK"

    # If step is high complexity, bias toward SPARK
    if complexity == "high":
        tool = "SPARK"

    # Decide config
    if num_rows < 50_000:
        config = "LOW_RESOURCES"
    elif num_rows < 500_000:
        config = "BALANCED"
    else:
        config = "HIGH_PARALLELISM"

    # Map tool + config -> action index (same as env._decode_action inverse)
    tool_idx = env.TOOLS.index(tool)
    config_idx = env.CONFIGS.index(config)
    num_configs = len(env.CONFIGS)

    action = tool_idx * num_configs + config_idx
    return action


def run_heuristic_baseline(
    pipeline_def: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    num_episodes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run the pipeline using the heuristic policy for several episodes.
    Returns a list of episode results:
        [{ "total_reward": float, "steps": [info dicts per step] }, ...]
    """
    results = []

    for _ in range(num_episodes):
        env = PipelineEnv(pipeline_def=pipeline_def, dataset_meta=dataset_meta)
        state = env.reset()
        done = False
        total_reward = 0.0
        steps_log = []

        while not done:
            step = env.steps[env.current_step_idx]
            action = _heuristic_action(env, step, dataset_meta)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            steps_log.append(info)
            state = next_state

        results.append(
            {
                "total_reward": total_reward,
                "steps": steps_log,
            }
        )

    return results
