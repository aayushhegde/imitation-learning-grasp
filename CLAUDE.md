# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Early-stage research project on imitation learning for robotic grasping. Currently the repository
contains a single Python utility for visualizing a 6-DOF end-effector grasp trajectory — there is no
model training code, LaTeX paper, or test suite yet. The `.gitignore` is a generic LaTeX template
carried over from repo creation; no `.tex` sources exist yet, so don't assume a paper is in progress.

## Setup

```bash
python3 -m pip install numpy pandas matplotlib
```

## Running

```bash
python3 visualize_trajectory.py
```

- If `grasp_trajectory.csv` exists in the working directory, it loads and plots that data.
- Otherwise it generates a synthetic pick-and-place trajectory, saves it to `grasp_trajectory.csv`,
  and plots that instead.
- Output is written to `trajectory_plot.png` and also shown interactively via `plt.show()`.

## Data format

`grasp_trajectory.csv` has columns `t, x, y, z, roll, pitch, yaw`:
- `t`: time in seconds
- `x, y, z`: end-effector position in meters
- `roll, pitch, yaw`: end-effector orientation in degrees

`visualize_trajectory.py` validates on load that all seven columns are present
(`load_trajectory`) and raises `ValueError` if any are missing.
