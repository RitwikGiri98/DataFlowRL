# src/controller/controller.py

from typing import Dict, Any, Literal
from src.env.pipeline_env import PipelineEnv
from src.rl.bandit import UCBBandit


ModeType = Literal["dqn", "ucb", "hybrid"]


class Controller:
    """
    Orchestrates pipeline execution and interacts with the RL agent + bandit.
    """

    def __init__(
        self,
        rl_agent,
        pipeline_def: Dict[str, Any],
        dataset_meta: Dict[str, Any],
        mode: ModeType = "dqn",
    ):
        """
        Args:
            rl_agent: DQNAgent
            pipeline_def: pipeline definition dict
            dataset_meta: dataset metadata dict
            mode: "dqn" -> epsilon-greedy DQN,
                  "ucb" -> UCB bandit for action choice,
                  "hybrid" -> mix of both (here: half UCB, half DQN)
        """
        self.rl_agent = rl_agent
        self.env = PipelineEnv(pipeline_def=pipeline_def, dataset_meta=dataset_meta)

        self.mode: ModeType = mode
        # Bandit over same action space as RL
        self.bandit = UCBBandit(action_dim=self.env.action_dim)

    def _select_action(self, state, train: bool) -> int:
        """
        Decide which exploration strategy to use.
        """
        if not train:
            # Evaluation mode: always use greedy DQN
            return self.rl_agent.select_action(state, train=False)

        if self.mode == "dqn":
            # pure epsilon-greedy DQN
            return self.rl_agent.select_action(state, train=True)

        elif self.mode == "ucb":
            # pure bandit selection
            return self.bandit.select_action()

        elif self.mode == "hybrid":
            # simple hybrid: 50% of the time UCB, 50% DQN
            import random

            if random.random() < 0.5:
                return self.bandit.select_action()
            else:
                return self.rl_agent.select_action(state, train=True)

        else:
            # fallback
            return self.rl_agent.select_action(state, train=True)

    def run_episode(self, train: bool = True) -> Dict[str, Any]:
        state = self.env.reset()
        done = False
        total_reward = 0.0
        steps_log = []

        while not done:
            action = self._select_action(state, train=train)
            next_state, reward, done, info = self.env.step(action)

            # Update DQN with experience
            if train:
                self.rl_agent.observe(state, action, reward, next_state, done)

            # Update bandit if we're using it
            if train and self.mode in ("ucb", "hybrid"):
                self.bandit.update(action, reward)

            total_reward += reward
            steps_log.append(
                {
                    "step_idx": info.get("step_idx"),
                    "step_name": info.get("step_name"),
                    "tool": info.get("tool"),
                    "config": info.get("config"),
                    "reward": reward,
                    "runtime": info.get("runtime"),
                    "cost": info.get("cost"),
                    "quality": info.get("quality"),
                    "success": info.get("success"),
                }
            )

            state = next_state

        return {
            "total_reward": total_reward,
            "steps": steps_log,
        }
