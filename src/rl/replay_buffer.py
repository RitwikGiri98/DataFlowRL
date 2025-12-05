# src/rl/replay_buffer.py

from collections import deque
from typing import Deque, Tuple
import numpy as np


class ReplayBuffer:
    """
    Simple experience replay buffer for DQN.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxs:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
