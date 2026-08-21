"""Episode containers and policy rollout helpers."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .observation import ACTION_DIM, OBS_DIM


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass
class Episode:
    """One complete attempt, stored as state-action training pairs."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success: bool
    seed: int | None = None
    usable_for_training: bool | None = None
    phases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        n = len(self.observations)
        if self.observations.shape != (n, OBS_DIM):
            raise ValueError(f"observations must have shape (T, {OBS_DIM})")
        if self.actions.shape != (n, ACTION_DIM):
            raise ValueError(f"actions must have shape (T, {ACTION_DIM})")
        if self.rewards.shape != (n,):
            raise ValueError("rewards must have shape (T,)")
        if self.phases and len(self.phases) != n:
            raise ValueError("phases must be empty or have one label per step")
        if self.usable_for_training is None:
            self.usable_for_training = self.success

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


def rollout_episode(env: Any, policy: Policy, seed: int | None = None) -> Episode:
    """Run a flat-observation policy until termination or timeout."""

    obs, _ = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    info: dict[str, Any] = {}

    while True:
        action = np.asarray(policy(obs), dtype=np.float32)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    return Episode(
        observations=np.stack(observations),
        actions=np.stack(actions),
        rewards=np.asarray(rewards),
        success=bool(info.get("is_success", False)),
        seed=seed,
    )
