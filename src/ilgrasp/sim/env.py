"""Thin, explicit wrapper around panda-gym's pick-and-place environment."""

from typing import Any

import numpy as np

from .observation import ACTION_DIM, OBS_DIM, flatten_obs


class GraspEnv:
    """Expose flat observations while retaining the expert's raw observation.

    Imports of gymnasium and panda-gym are delayed until construction so data
    analysis and unit tests can run on machines without the simulator installed.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        env_id: str = "PandaPickAndPlace-v3",
        max_episode_steps: int | None = 100,
    ) -> None:
        try:
            import gymnasium as gym
            import panda_gym  # noqa: F401 -- import registers Panda environments
        except ImportError as exc:
            raise ImportError(
                "The simulator is not installed. Run `pip install -e .` from the project root."
            ) from exc

        # panda-gym 3.0.x expects headless mode by omitting render_mode; passing
        # render_mode=None explicitly raises a ValueError.
        kwargs: dict[str, Any] = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        self._env = gym.make(env_id, **kwargs)
        self.env_id = env_id
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.action_space = self._env.action_space
        self.raw_obs: dict[str, np.ndarray] | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        self.raw_obs = obs
        return flatten_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape ({ACTION_DIM},), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.raw_obs = obs
        return flatten_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "GraspEnv":  # noqa: PYI034 -- Python 3.10 compatible
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
