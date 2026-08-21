"""Teaching-first imitation learning for simulated robotic grasping."""

__version__ = "0.1.0"

# Re-exported so notebooks/01_simulator_environment.ipynb keeps working after
# src/ilgrasp/env.py and src/ilgrasp/constants.py are removed. These are now
# thin aliases: there is exactly one GraspEnv and one flatten_obs in the tree.
from ilgrasp.sim import ACTION_DIM, OBS_DIM, GraspEnv, flatten_obs

__all__ = ["ACTION_DIM", "GraspEnv", "OBS_DIM", "flatten_obs"]
