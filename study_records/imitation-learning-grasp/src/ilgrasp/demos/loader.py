"""
Data-loading stage of the imitation-learning-grasp pipeline.

This is the first stage: everything downstream (retargeting onto the robot's
kinematics, augmentation, and behavior-cloning policy training) consumes the
`Demo` objects returned here instead of touching CSVs directly. That means
swapping in a real dataset later is just writing a new function that returns
`list[Demo]` — no other module needs to change.

Today's source is synthetic data from generate_trajectories.py, standing in
for a real dataset (e.g. DexYCB) until real demonstrations are wired in.
Any CSV with the same column schema loads the same way.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"t", "x", "y", "z", "roll", "pitch", "yaw"}


@dataclass
class Demo:
    """One grasp demonstration: a single robot pose per timestep."""
    id: int
    t: np.ndarray           # (T,)   time in seconds
    position: np.ndarray    # (T, 3) x, y, z in meters
    orientation: np.ndarray  # (T, 3) roll, pitch, yaw in degrees

    def __len__(self) -> int:
        return len(self.t)


def load_demos(csv_path: str) -> list[Demo]:
    """Load every trajectory in a CSV, one Demo per trajectory_id."""
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Single-trajectory CSVs (e.g. grasp_trajectory.csv) have no trajectory_id
    # column — treat the whole file as one demo so both formats load the same way.
    if "trajectory_id" not in df.columns:
        df["trajectory_id"] = 0

    demos = []
    for traj_id, group in df.groupby("trajectory_id"):
        group = group.sort_values("t")  # groupby doesn't guarantee row order within a group
        demos.append(Demo(
            id=int(traj_id),
            t=group["t"].to_numpy(),
            position=group[["x", "y", "z"]].to_numpy(),
            orientation=group[["roll", "pitch", "yaw"]].to_numpy(),
        ))
    return demos
