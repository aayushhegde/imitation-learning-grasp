"""The grasp environment: a thin wrapper over panda-gym's PandaPickAndPlace-v3.

Wraps the dict observation into a flat 22-vector so the policy network sees a
single array, while keeping the raw dict available for the scripted expert.
"""

import numpy as np
import gymnasium as gym
import panda_gym  # noqa: F401 — import registers 'PandaPickAndPlace-v3' with gymnasium

from ilgrasp.constants import OBS_DIM, ACTION_DIM


def flatten_obs(obs_dict):
    parts = [obs_dict['observation'], obs_dict['desired_goal']]
    return np.concatenate(parts).astype(np.float32)


class GraspEnv:
    def __init__(self, render_mode="rgb_array"):
        # 'rgb_array' is the default because gymnasium 1.3.0 raises ValueError
        # on render_mode=None. Use 'human' only to watch (freezes on Apple Silicon).
        self._env = gym.make('PandaPickAndPlace-v3', render_mode=render_mode)
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.action_space = self._env.action_space
        self.raw_obs = None

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self.raw_obs = obs                      # keep the dict for the expert
        return flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.raw_obs = obs
        return flatten_obs(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()