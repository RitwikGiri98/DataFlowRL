# src/rl/dqn_agent.py

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Basic DQN with epsilon-greedy exploration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, train: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        """
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).cpu().item())
        return action

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition and perform a training step.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        self.train_steps += 1
        self._train_step()

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

        # Periodically update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def _train_step(self) -> None:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Q(s,a)
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]

        targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
