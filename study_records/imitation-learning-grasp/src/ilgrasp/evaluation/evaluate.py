"""Evaluate policies on fresh simulator episodes."""

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from ilgrasp.sim import rollout_episode


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial success rate (normal z=1.96)."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return center - margin, center + margin


@dataclass
class EvaluationResult:
    successes: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    seeds: np.ndarray

    @property
    def n_episodes(self) -> int:
        return len(self.successes)

    @property
    def success_rate(self) -> float:
        return float(self.successes.mean())

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return wilson_interval(int(self.successes.sum()), self.n_episodes)

    def as_dict(self) -> dict[str, float | int | list[float]]:
        low, high = self.confidence_interval
        return {
            "episodes": self.n_episodes,
            "successes": int(self.successes.sum()),
            "success_rate": self.success_rate,
            "ci95": [low, high],
            "mean_return": float(self.episode_returns.mean()),
            "mean_episode_length": float(self.episode_lengths.mean()),
        }


def evaluate_policy(
    env: Any,
    policy: Any,
    n_episodes: int = 100,
    seed_start: int = 10_000,
) -> EvaluationResult:
    """Run on a disjoint deterministic seed range for reproducible evaluation."""

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    seeds = np.arange(seed_start, seed_start + n_episodes, dtype=np.int64)
    episodes = [rollout_episode(env, policy, seed=int(seed)) for seed in seeds]
    return EvaluationResult(
        successes=np.asarray([episode.success for episode in episodes], dtype=bool),
        episode_returns=np.asarray([episode.total_reward for episode in episodes]),
        episode_lengths=np.asarray([len(episode) for episode in episodes]),
        seeds=seeds,
    )
