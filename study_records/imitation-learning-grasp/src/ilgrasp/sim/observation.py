"""Observation contract for ``PandaPickAndPlace-v3``.

The policy sees the 19-value task observation followed by the 3-value desired
goal. ``achieved_goal`` is omitted because the object position is already in the
task observation at indices 7:10.
"""

from collections.abc import Mapping

import numpy as np

OBS_DIM = 22
ACTION_DIM = 4

FLAT_LABELS = (
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "fingers_width",
    "object_x",
    "object_y",
    "object_z",
    "object_roll",
    "object_pitch",
    "object_yaw",
    "object_vx",
    "object_vy",
    "object_vz",
    "object_wx",
    "object_wy",
    "object_wz",
    "goal_x",
    "goal_y",
    "goal_z",
)


def flatten_obs(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert a panda-gym goal observation to the policy's fixed 22-vector.

    A loud error is preferable to silently training on a changed observation
    order after a simulator upgrade.
    """

    missing = {"observation", "desired_goal"} - set(obs)
    if missing:
        raise KeyError(f"Observation is missing keys: {sorted(missing)}")

    flat = np.concatenate((obs["observation"], obs["desired_goal"])).astype(
        np.float32, copy=False
    )
    if flat.shape != (OBS_DIM,):
        raise ValueError(
            f"Expected a ({OBS_DIM},) flattened observation, got {flat.shape}. "
            "The panda-gym observation contract may have changed."
        )
    if not np.isfinite(flat).all():
        raise ValueError("Observation contains NaN or infinity")
    return flat

