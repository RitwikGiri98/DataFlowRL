# src/agents/execution_agent.py

from typing import Dict, Any
from src.tools.pipeline_simulator import simulate_step


class ExecutionAgent:
    """
    Thin wrapper around the simulator for now.
    In a more advanced version, this would call real engines.
    """

    def __init__(self, dataset_meta: Dict[str, Any]):
        self.dataset_meta = dataset_meta

    def execute(self, step: Dict[str, Any], tool: str, config: str) -> Dict[str, Any]:
        return simulate_step(step=step, dataset_meta=self.dataset_meta, tool=tool, config=config)
