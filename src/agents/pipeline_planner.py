# src/agents/pipeline_planner.py

from typing import Dict, Any, List


class PipelinePlannerAgent:
    """
    Placeholder for a planner that could transform
    a DAG into an ordered list of executable steps.
    """

    def __init__(self, pipeline_def: Dict[str, Any]):
        self.pipeline_def = pipeline_def

    def get_ordered_steps(self) -> List[Dict[str, Any]]:
        # For now, pipeline_def already contains "steps" in order.
        return self.pipeline_def.get("steps", [])
