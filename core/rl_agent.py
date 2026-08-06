"""
RL betting agent for Apollo.

Primary : DQN via stable-baselines3 (requires pip install stable-baselines3)
Fallback: Epsilon-greedy Q-table agent (pure numpy, no extra deps)
"""

import ast
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Fallback Q-table agent ────────────────────────────────────────────

class QTableAgent:
    """
    Simple epsilon-greedy agent with discretised state.
    Used when stable-baselines3 is unavailable.
    """
    N_ACTIONS = 4

    def __init__(self, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q: dict = {}

    def _discretise(self, obs: np.ndarray) -> tuple:
        # Bucket key features: elo_delta, max_edge, bankroll
        elo_bucket  = int(np.clip(obs[2] * 2, -4, 4))    # elo_delta
        edge_bucket = int(np.clip(obs[12] * 10, -2, 5))  # max_edge
        bk_bucket   = int(np.clip(obs[13], 0, 3))         # bankroll
        return (elo_bucket, edge_bucket, bk_bucket)

    def _q(self, state: tuple) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(self.N_ACTIONS)
        return self.q[state]

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, None]:
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.N_ACTIONS)), None
        return int(np.argmax(self._q(self._discretise(obs)))), None

    def update(self, obs, action, reward, next_obs, done):
        s  = self._discretise(obs)
        s2 = self._discretise(next_obs)
        q_vals = self._q(s)
        target = reward + (0.0 if done else self.gamma * np.max(self._q(s2)))
        q_vals[action] += self.lr * (target - q_vals[action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        data = {
            "type": "qtable",
            "epsilon": self.epsilon,
            "q": {str(k): v.tolist() for k, v in self.q.items()},
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "QTableAgent":
        data = json.loads(Path(path).read_text())
        agent = cls()
        agent.epsilon = data["epsilon"]
        agent.q = {ast.literal_eval(k): np.array(v) for k, v in data["q"].items()}
        return agent


# ── SB3 DQN agent ─────────────────────────────────────────────────────

def make_dqn_agent(env, **kwargs):
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        default = dict(
            policy="MlpPolicy",
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=500,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0,
        )
        default.update(kwargs)
        return DQN(env=env, **default)
    except ImportError:
        logger.warning("stable-baselines3 not installed; using QTableAgent fallback.")
        return None


def make_agent(env, use_dqn: bool = True):
    """Return DQN if sb3 available, else QTableAgent."""
    if use_dqn:
        agent = make_dqn_agent(env)
        if agent is not None:
            return agent, "dqn"
    return QTableAgent(), "qtable"
