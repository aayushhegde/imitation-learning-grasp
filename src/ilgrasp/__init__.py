"""ilgrasp — imitation learning for parallel-jaw grasping."""

from ilgrasp.env import GraspEnv, flatten_obs
from ilgrasp.constants import (
    OBS_DIM,
    ACTION_DIM,
    OBJ_POS_SLICE,
    MAX_EPISODE_STEPS,
)

__all__ = [
    "GraspEnv",
    "flatten_obs",
    "OBS_DIM",
    "ACTION_DIM",
    "OBJ_POS_SLICE",
    "MAX_EPISODE_STEPS",
]