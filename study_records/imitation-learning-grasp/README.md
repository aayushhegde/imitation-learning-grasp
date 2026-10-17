# Imitation Learning for Robotic Grasping

A complete, teaching-first behavior-cloning pipeline for `PandaPickAndPlace-v3`:

```text
simulator → scripted expert → SQLite demonstrations → MLP training → closed-loop evaluation
```

The repository is designed for a strong high-school student: each component is small enough to
read, but the experimental rules (reproducible seeds, episode-level validation, confidence
intervals, and repeated trials) are the same habits used in research.

## What is implemented

| Stage | Public interface | Purpose |
|---|---|---|
| Simulator | `ilgrasp.sim.GraspEnv` | Flat 22-value observation and bounded 4-value action |
| Expert | `ilgrasp.experts.ScriptedExpert` | Finite-state pick-and-place teacher |
| Verified task | `ilgrasp.sim.VerifiedPlacementEnv` | Physical barrier plus release/retreat/stability certificate |
| Verified expert | `ilgrasp.experts.VerifiedPickPlaceExpert` | Robust 13-phase obstacle-crossing teacher |
| Data | `ilgrasp.data.DemoDatabase` | Atomic SQLite episode storage and successful-demo queries |
| Learning | `ilgrasp.policies.GraspPolicy` | Normalized MLP trained with behavior cloning |
| Evaluation | `ilgrasp.evaluation.evaluate_policy` | Fresh-seed success rate and 95% Wilson interval |
| Visualization | `ilgrasp.visualization.record_verified_expert_gif` | Two-camera 1280×720 rollout and live completion certificate |

The original pose-trajectory CSV loader remains in `src/ilgrasp/demos/` for visualization and
legacy data. It is intentionally separate from the new state-action database: pose paths and
behavior-cloning samples are different data contracts.

## Setup

Python 3.11 or 3.12 is recommended.

### Apple Silicon Mac (recommended)

PyPI currently has no compatible prebuilt PyBullet wheel for some Apple-Silicon/Python
combinations, so a normal `pip install` may unexpectedly compile thousands of C++ files and fail.
Install the prebuilt conda-forge package, then let pip handle the pure-Python pieces:

```bash
conda create -y -p .sim-env -c conda-forge python=3.12 pybullet=3.25
conda activate "$PWD/.sim-env"
python -m pip install gymnasium scipy torch pandas matplotlib pillow jupyterlab nbformat pytest ruff
python -m pip install panda-gym --no-deps
python -m pip install -e . --no-deps
pytest
```

Use this `.sim-env` kernel for the notebooks. The `--no-deps` on the last two commands is
intentional: conda already supplied the working PyBullet binary.

### Linux, Windows, or a platform with a PyBullet wheel

```bash
cd imitation-learning-grasp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
pytest
```

If `uv` is installed, the equivalent faster setup is:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[notebook,dev]"
pytest
```

Run headless for collection/training. The PyBullet GUI can be unreliable on Apple Silicon and is
only enabled when a script receives `--render`.

## End-to-end commands

Start with a small smoke run:

```bash
python scripts/collect_demos.py --successful 10 --max-attempts 100
python scripts/train.py --episodes 10 --epochs 30
python scripts/evaluate.py --episodes 20
```

Then run the research-sized version:

```bash
python scripts/collect_demos.py --successful 200 --max-attempts 1000
python scripts/train.py --episodes 200 --epochs 80
python scripts/evaluate.py --episodes 100
python scripts/run_experiments.py
```

Generated databases, checkpoints, metrics, and plots go under `artifacts/` and are ignored by Git.

Record the verified use case independently with:

```bash
python scripts/record_verified_animation.py \
  --output notebooks/assets/verified_pick_cross_place.gif \
  --seed 7
```

## Notebook set

- `notebooks/04_verified_expert_pipeline_student_en.ipynb` — English student edition
- `notebooks/04_verified_expert_pipeline_solutions_en.ipynb` — English solutions, saved with a
  verified execution
- `notebooks/04_verified_expert_pipeline_student_zh.ipynb` — 中文学生版
- `notebooks/04_verified_expert_pipeline_solutions_zh.ipynb` — 中文答案版，已保存完整运行输出

These are the recommended notebooks after the simulator lab. All four have identical 46-cell
structure and 13 matched exercises. Students build state parsing, bounded control, phase targets,
transition guards, recovery, the complete expert, aligned trace collection, verified SQL queries,
episode splitting, hybrid BC loss, strict evaluation, and failure classification. The solutions
execute the real task end to end; no illustrative results are substituted for experiments.

The earlier `02_full_pipeline_*` and `03_pipeline_from_scratch_animated.ipynb` files remain as legacy
references, but they use Panda Gym's easier native success condition and are not the current lesson.

## Scientific protocol

- Training seeds and evaluation seeds are disjoint.
- Train/validation splitting happens by episode, not by individual time step.
- Native goal contact is not task completion. Verified success requires lift, barrier crossing,
  release, robot retreat, 12 stable steps, and no barrier penetration.
- Only strictly verified expert episodes train the policy; failed attempts remain useful for QA.
- Evaluation uses at least 100 fresh episodes for a headline result and reports uncertainty.
- Ablations run three seeds and display mean ± standard deviation.

## 中文快速说明

新的推荐课程从学生已经完成的 simulator 开始，重点构建 13-phase expert。任务要求方块越过实体
障碍物，在桌面目标处松爪，机械臂撤离，并连续稳定 12 步后才算完成。请使用四份
`04_verified_expert_pipeline_*` Notebook；旧版 02/03 仅保留作历史参考。
