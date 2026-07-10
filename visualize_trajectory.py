"""
Visualize a 6-DOF grasp trajectory from a CSV file.
If no CSV exists, a synthetic demonstration is generated and saved.

CSV columns: t, x, y, z, roll, pitch, yaw
  - x, y, z  : end-effector position in meters
  - roll, pitch, yaw : orientation in degrees
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "grasp_trajectory.csv"


def generate_fake_trajectory(n_steps: int = 60) -> pd.DataFrame:
    """
    Synthesize a plausible pick-and-place grasp trajectory:
      phase 0-30  : pre-grasp approach (descend toward object)
      phase 30-45 : grasp & lift
      phase 45-60 : transport to place pose
    """
    t = np.linspace(0, 3.0, n_steps)  # 3 seconds total

    # --- position ---
    x = np.concatenate([
        np.linspace(0.0,  0.0,  30),   # approach in place
        np.linspace(0.0,  0.0,  15),   # lift in place
        np.linspace(0.0,  0.25, 15),   # move sideways to place
    ])
    y = np.concatenate([
        np.linspace(0.4,  0.4,  30),
        np.linspace(0.4,  0.4,  15),
        np.linspace(0.4,  0.55, 15),
    ])
    z = np.concatenate([
        np.linspace(0.35, 0.08, 30),   # descend to object height
        np.linspace(0.08, 0.30, 15),   # lift
        np.linspace(0.30, 0.30, 15),   # keep height during transport
    ])

    # --- orientation (degrees) ---
    roll  = np.zeros(n_steps)
    pitch = np.concatenate([
        np.linspace(0, 15, 30),        # tilt toward object on approach
        np.linspace(15, 0, 30),        # level out while lifting / placing
    ])
    yaw   = np.concatenate([
        np.zeros(30),
        np.linspace(0, -20, 30),       # rotate during transport
    ])

    # small noise to look like real demonstrations
    rng = np.random.default_rng(42)
    noise = lambda sigma: rng.normal(0, sigma, n_steps)
    x     += noise(0.002)
    y     += noise(0.002)
    z     += noise(0.001)
    roll  += noise(0.3)
    pitch += noise(0.5)
    yaw   += noise(0.5)

    df = pd.DataFrame({"t": t, "x": x, "y": y, "z": z,
                       "roll": roll, "pitch": pitch, "yaw": yaw})
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated synthetic trajectory → {CSV_PATH}")
    return df


def load_trajectory(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"t", "x", "y", "z", "roll", "pitch", "yaw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def plot_trajectory(df: pd.DataFrame) -> None:
    t = df["t"].values
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(t)))

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("6-DOF Grasp Trajectory", fontsize=14, fontweight="bold")

    # --- 3-D path ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(df["x"], df["y"], df["z"], c=np.linspace(0, 1, len(t)),
                 cmap="viridis", s=18, depthshade=True)
    ax3d.plot(df["x"], df["y"], df["z"], color="gray", lw=0.8, alpha=0.5)

    # mark start / end
    ax3d.scatter(*df[["x", "y", "z"]].iloc[0],  color="green", s=80,
                 zorder=5, label="start")
    ax3d.scatter(*df[["x", "y", "z"]].iloc[-1], color="red",   s=80,
                 zorder=5, label="end")

    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("End-effector 3-D path\n(color = time)")
    ax3d.legend(fontsize=8)

    # --- time-series of all 6 DOF ---
    ax_ts = fig.add_subplot(1, 2, 2)
    dof_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for dof, col in zip(["x", "y", "z", "roll", "pitch", "yaw"], dof_colors):
        label = f"{dof} {'(m)' if dof in ('x','y','z') else '(°)'}"
        ax_ts.plot(t, df[dof], label=label, color=col, lw=1.5)

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_title("DOF vs. time")
    ax_ts.legend(fontsize=8, ncol=2)
    ax_ts.grid(True, alpha=0.3)

    # colorbar proxy for 3-D plot
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=t[0], vmax=t[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, label="Time (s)", shrink=0.5, pad=0.1)

    plt.tight_layout()
    plt.savefig("trajectory_plot.png", dpi=150, bbox_inches="tight")
    print("Plot saved → trajectory_plot.png")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        df = generate_fake_trajectory()
    else:
        df = load_trajectory(CSV_PATH)
        print(f"Loaded {len(df)} steps from {CSV_PATH}")

    print(df.describe().round(3).to_string())
    plot_trajectory(df)
