"""
Generate a batch of fake 6-DOF grasp trajectories, save them to a single CSV,
and plot one of them in 3D.

CSV columns: trajectory_id, t, x, y, z, roll, pitch, yaw
  - trajectory_id     : integer index identifying which demonstration a row belongs to
  - x, y, z           : end-effector position in meters
  - roll, pitch, yaw  : end-effector orientation in degrees
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "grasp_trajectories.csv"
N_TRAJECTORIES = 10
N_STEPS = 60


def generate_trajectory(traj_id: int, rng: np.random.Generator, n_steps: int = N_STEPS) -> pd.DataFrame:
    """
    Synthesize one pick-and-place grasp trajectory with randomized start/place
    poses, following the same three-phase shape as visualize_trajectory.py:
      phase 0-30  : pre-grasp approach (descend toward object)
      phase 30-45 : grasp & lift
      phase 45-60 : transport to place pose
    """
    t = np.linspace(0, 3.0, n_steps)

    # randomize object pickup location and place offset per trajectory
    start_x, start_y = rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.5)
    place_dx, place_dy = rng.uniform(0.15, 0.35), rng.uniform(-0.1, 0.15)
    approach_z, lift_z = rng.uniform(0.30, 0.40), rng.uniform(0.25, 0.35)

    x = np.concatenate([
        np.linspace(start_x, start_x, 30),
        np.linspace(start_x, start_x, 15),
        np.linspace(start_x, start_x + place_dx, 15),
    ])
    y = np.concatenate([
        np.linspace(start_y, start_y, 30),
        np.linspace(start_y, start_y, 15),
        np.linspace(start_y, start_y + place_dy, 15),
    ])
    z = np.concatenate([
        np.linspace(approach_z, 0.08, 30),
        np.linspace(0.08, lift_z, 15),
        np.linspace(lift_z, lift_z, 15),
    ])

    yaw_target = rng.uniform(-30, 30)
    roll = np.zeros(n_steps)
    pitch = np.concatenate([
        np.linspace(0, 15, 30),
        np.linspace(15, 0, 30),
    ])
    yaw = np.concatenate([
        np.zeros(30),
        np.linspace(0, yaw_target, 30),
    ])

    noise = lambda sigma: rng.normal(0, sigma, n_steps)
    x     += noise(0.002)
    y     += noise(0.002)
    z     += noise(0.001)
    roll  += noise(0.3)
    pitch += noise(0.5)
    yaw   += noise(0.5)

    return pd.DataFrame({
        "trajectory_id": traj_id,
        "t": t, "x": x, "y": y, "z": z,
        "roll": roll, "pitch": pitch, "yaw": yaw,
    })


def generate_dataset(n_trajectories: int = N_TRAJECTORIES, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trajectories = [generate_trajectory(i, rng) for i in range(n_trajectories)]
    return pd.concat(trajectories, ignore_index=True)


def plot_trajectory_3d(df: pd.DataFrame, traj_id: int) -> None:
    traj = df[df["trajectory_id"] == traj_id]
    t = traj["t"].values

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(traj["x"], traj["y"], traj["z"], c=np.linspace(0, 1, len(t)),
               cmap="viridis", s=18, depthshade=True)
    ax.plot(traj["x"], traj["y"], traj["z"], color="gray", lw=0.8, alpha=0.5)

    ax.scatter(*traj[["x", "y", "z"]].iloc[0],  color="green", s=80, zorder=5, label="start")
    ax.scatter(*traj[["x", "y", "z"]].iloc[-1], color="red",   s=80, zorder=5, label="end")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Grasp Trajectory {traj_id} (3-D path)")
    ax.legend(fontsize=8)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=t[0], vmax=t[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time (s)", shrink=0.6, pad=0.1)

    plt.tight_layout()
    out_path = f"trajectory_{traj_id}_3d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated {N_TRAJECTORIES} trajectories ({len(df)} rows) → {CSV_PATH}")

    plot_trajectory_3d(df, traj_id=0)
