# src/tools/pipeline_simulator.py

from typing import Dict, Any
import random
import math


def _base_complexity_factor(step: Dict[str, Any]) -> float:
    """
    Rough complexity factor based on step type + complexity label.
    """
    step_type = step.get("type", "transform")
    complexity = step.get("complexity", "medium")

    type_factor = {
        "extract": 0.5,
        "transform": 1.0,
        "aggregate": 1.2,
        "join": 1.3,
        "load": 0.6,
    }.get(step_type, 1.0)

    complexity_factor = {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.4,
    }.get(complexity, 1.0)

    return type_factor * complexity_factor


def _tool_speed_factor(tool: str) -> float:
    """
    Lower is faster.
    """
    return {
        "PANDAS": 1.0,
        "SPARK": 0.7,   # faster on large
        "DUCKDB": 0.8,
    }.get(tool, 1.0)


def _config_factor(config: str) -> float:
    """
    Lower is more resources / faster, higher is slower.
    """
    return {
        "LOW_RESOURCES": 1.3,
        "BALANCED": 1.0,
        "HIGH_PARALLELISM": 0.7,
    }.get(config, 1.0)


def _compute_quality(tool: str, config: str, step_complexity: str, num_rows: int) -> float:
    """
    Option 1: Realistic but moderate quality variation.

    - Base quality slightly lower for very large datasets.
    - High parallelism can reduce quality a bit.
    - PANDAS on high-complexity steps can struggle.
    - DUCKDB on very large datasets can lose a bit of quality.
    """
    # Base quality: large datasets slightly harder to keep perfect
    if num_rows < 100_000:
        base_quality = 0.98
    elif num_rows < 500_000:
        base_quality = 0.96
    else:
        base_quality = 0.94

    # Penalty for very aggressive parallelism on complex steps
    if config == "HIGH_PARALLELISM" and step_complexity == "high":
        base_quality -= 0.03

    # PANDAS struggling with high-complexity operations (e.g., heavy aggregates)
    if tool == "PANDAS" and step_complexity == "high":
        base_quality -= 0.04

    # DUCKDB might degrade slightly on very large data
    if tool == "DUCKDB" and num_rows > 500_000:
        base_quality -= 0.02

    # Clamp base_quality to [0, 1]
    base_quality = max(0.0, min(1.0, base_quality))

    # Add small Gaussian noise
    noise = random.normalvariate(0.0, 0.005)
    quality = base_quality + noise
    quality = max(0.0, min(1.0, quality))

    return quality


def simulate_step(step: Dict[str, Any], dataset_meta: Dict[str, Any],
                  tool: str, config: str) -> Dict[str, Any]:
    """
    Simulate execution of one pipeline step and return metrics.

    Returns:
        {
            "runtime": float,
            "cost": float,
            "quality": float,   # 0..1
            "success": bool,
        }
    """
    num_rows = dataset_meta.get("num_rows", 10_000)
    num_features = dataset_meta.get("num_features", 20)

    base_complexity = _base_complexity_factor(step)
    tool_factor = _tool_speed_factor(tool)
    cfg_factor = _config_factor(config)

    # approximate data size factor
    size_factor = math.log10(max(1, num_rows)) * 0.5 + math.log10(max(1, num_features)) * 0.2

    # runtime ~ complexity * size / speed
    runtime = base_complexity * size_factor * (1.0 / tool_factor) * cfg_factor * 10.0

    # cost ~ runtime * resource usage (config-based)
    resource_factor = {
        "LOW_RESOURCES": 0.7,
        "BALANCED": 1.0,
        "HIGH_PARALLELISM": 1.3,
    }.get(config, 1.0)

    cost = runtime * resource_factor

    # --- QUALITY (new logic) ---
    step_complexity = step.get("complexity", "medium")
    quality = _compute_quality(tool, config, step_complexity, num_rows)

    # add small randomness to runtime / cost
    runtime *= random.uniform(0.9, 1.1)
    cost *= random.uniform(0.9, 1.1)

    # simulate failure condition: very low quality -> higher failure chance
    success = True
    if quality < 0.80:
        if random.random() < 0.3:
            success = False

    return {
        "runtime": float(runtime),
        "cost": float(cost),
        "quality": float(quality),
        "success": success,
    }
