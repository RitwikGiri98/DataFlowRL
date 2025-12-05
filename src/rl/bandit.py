# src/rl/bandit.py

from typing import Optional
import numpy as np


class UCBBandit:
    """
    Simple UCB1 multi-armed bandit.

    Each action index (0..action_dim-1) is treated as an arm.
    We keep track of:
      - n[a]: how many times we tried arm a
      - q[a]: average reward observed for arm a

    UCB formula:
      UCB[a] = q[a] + c * sqrt( ln(t) / n[a] )
    where:
      t = total pulls so far (sum of n[a])
    """

    def __init__(self, action_dim: int, c: float = 2.0):
        self.action_dim = action_dim
        self.c = c

        self.counts = np.zeros(action_dim, dtype=np.int64)   # n[a]
        self.values = np.zeros(action_dim, dtype=np.float32) # q[a]
        self.total_pulls = 0

    def select_action(self) -> int:
        """
        Selects an action index using UCB.
        If any action has not been tried yet, try those first.
        """
        # First explore all actions at least once
        for a in range(self.action_dim):
            if self.counts[a] == 0:
                return a

        self.total_pulls += 1

        # Compute UCB value for each action
        ucb_values = np.zeros(self.action_dim, dtype=np.float32)
        for a in range(self.action_dim):
            avg_reward = self.values[a]
            exploration_term = self.c * np.sqrt(
                np.log(self.total_pulls + 1) / self.counts[a]
            )
            ucb_values[a] = avg_reward + exploration_term

        # Pick best UCB
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float) -> None:
        """
        Update internal estimates after taking 'action' and observing 'reward'.
        """
        self.total_pulls += 1
        self.counts[action] += 1

        # Incremental update of average reward
        n = self.counts[action]
        old_q = self.values[action]
        new_q = old_q + (reward - old_q) / float(n)
        self.values[action] = new_q

    def get_stats(self) -> dict:
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_pulls": self.total_pulls,
        }
