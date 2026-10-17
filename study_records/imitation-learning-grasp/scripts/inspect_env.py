"""Resets the PandaPickAndPlace-v3 gym environment once and prints its observation keys/values and a sample action, to inspect the env's I/O shape."""

import gymnasium as gym
import panda_gym  # noqa: F401 -- import registers Panda environments

env = gym.make("PandaPickAndPlace-v3")
obs, info = env.reset()

print("OBSERVATION keys:", obs.keys())
for key, value in obs.items():
    print(f"  {key}: {len(value)} numbers → {value}")

print("\nACTION (one random sample):", env.action_space.sample())

env.close()
