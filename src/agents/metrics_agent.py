# src/agents/metrics_agent.py

from typing import Dict, Any, List


class MetricsAgent:
    """
    Collects and buffers metrics for analysis and plotting.
    """

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def log_episode(self, episode_result: Dict[str, Any]):
        self.episodes.append(episode_result)

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        return self.episodes
