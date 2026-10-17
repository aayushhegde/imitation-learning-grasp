#!/usr/bin/env python3
"""Build the complete, animated, from-scratch teaching notebook.

The notebook deliberately re-implements the project classes in dependency order.
It does not import the finished ``ilgrasp`` implementations, so students can see
how the interfaces connect instead of treating the package as a black box.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_pipeline_from_scratch_animated.ipynb"


def _source(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def md(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _source(text)}


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    md(
        r'''
        # Build an Imitation-Learning Robot — Completely from Scratch

        **One executable notebook, one complete pipeline, no hidden project imports.**

        You already know the Panda simulator, finite-state experts, and the idea of a neural
        network. This notebook now answers the engineering question that usually feels mysterious:

        > How do we turn those separate ideas into one reliable learning system?

        We will write every project class in dependency order, run the real simulator, save expert
        demonstrations in SQLite, train a behavior-cloning policy, evaluate it on unseen seeds, and
        animate both expert and learned behavior.

        **Audience:** advanced high-school students. A careful first pass takes about 2–3 hours.
        During a mentor meeting, use the suggested pauses marked **Mentor checkpoint**.

        <div style="padding:12px 16px;border-left:5px solid #3366cc;background:#eef4ff">
        <b>Important:</b> the notebook re-implements the classes for learning. The production
        versions live under <code>src/ilgrasp/</code>; the final section maps every teaching class
        back to its real file.
        </div>
        ''',
        "title",
    ),
    md(
        r'''
        ## 0. The complete architecture

        The arrows are *contracts*. Each box only needs to know the shape and meaning of what comes
        through its arrows.

        ```text
        raw simulator dict
                │ flatten_obs
                ▼
        GraspEnv ──22D state──▶ ScriptedExpert ──4D action──▶ Episode
             ▲                                                    │
             │                                                    ▼ atomic save
             │                                              DemoDatabase
             │                                                    │ TrainingData
             │                                                    ▼
             └────────── NumpyPolicy ◀── GraspPolicy ◀── train_behavior_cloning
                              │
                              └──────────▶ EvaluationResult + failure analysis
        ```

        | Class | Its one job | Important invariant |
        |---|---|---|
        | `GraspEnv` | make the simulator look like a normal vector environment | state `(22,)`, action `(4,)` |
        | `Episode` | hold one complete attempt | every array has the same time length `T` |
        | `ScriptedExpert` | label simulator states with actions | phase transitions are explicit |
        | `TrainingData` | hold many supervised samples | sample count `N` agrees across arrays |
        | `DemoDatabase` | preserve episodes and steps on disk | one episode is written atomically |
        | `GraspPolicy` | map state vectors to bounded actions | normalization is saved with the model |
        | `NumpyPolicy` | connect PyTorch to NumPy simulator code | inference only; output `(4,)` |
        | `TrainResult` | return model plus evidence about training | train/validation episode IDs are disjoint |
        | `EvaluationResult` | summarize fresh closed-loop trials | rate is accompanied by uncertainty |

        First, animate that flow. The highlighted box is the component currently doing work.
        ''',
        "architecture",
    ),
    code(
        r'''
        # Imports used throughout the notebook. No ilgrasp project classes are imported.
        from __future__ import annotations

        import json
        import os
        import random
        import sqlite3
        import sys
        import tempfile
        from collections import Counter
        from collections.abc import Sequence
        from copy import deepcopy
        from dataclasses import dataclass, field
        from enum import Enum
        from math import sqrt
        from pathlib import Path
        from typing import Any, Protocol

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ilgrasp-mpl"))

        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from IPython.display import HTML, Image as NotebookImage, display
        from matplotlib import animation
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        from PIL import Image as PILImage, ImageDraw, ImageFont
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        np.set_printoptions(precision=3, suppress=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})

        # This complete notebook defaults to the meaningful full run.
        # Change to True when demonstrating the mechanics in only a few minutes.
        FAST_MODE = False
        TARGET_DEMOS = 20 if FAST_MODE else 100
        TRAIN_EPOCHS = 45 if FAST_MODE else 100
        EVAL_EPISODES = 20 if FAST_MODE else 100

        # All tutorial artifacts live in a temporary directory and cannot pollute your real data.
        tutorial_dir = Path(tempfile.mkdtemp(prefix="ilgrasp_tutorial_"))
        print(f"Tutorial workspace: {tutorial_dir}")
        print(f"Mode: {'FAST classroom run' if FAST_MODE else 'FULL experiment run'}")
        ''',
        "imports",
    ),
    code(
        r'''
        def show_pipeline_animation() -> None:
            labels = ["Simulator", "Expert", "Database", "BC learner", "Evaluation"]
            x = np.linspace(0.08, 0.92, len(labels))
            fig, ax = plt.subplots(figsize=(10, 2.4))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            boxes = []
            for i, (xi, label) in enumerate(zip(x, labels)):
                box = FancyBboxPatch(
                    (xi - 0.085, 0.38), 0.17, 0.25,
                    boxstyle="round,pad=0.015", linewidth=2,
                    edgecolor="#203864", facecolor="#e8eef7"
                )
                ax.add_patch(box)
                ax.text(xi, 0.505, label, ha="center", va="center", fontsize=10, weight="bold")
                boxes.append(box)
                if i < len(labels) - 1:
                    ax.add_patch(FancyArrowPatch(
                        (xi + 0.087, 0.505), (x[i + 1] - 0.087, 0.505),
                        arrowstyle="-|>", mutation_scale=14, color="#6c757d"
                    ))
            status = ax.text(0.5, 0.15, "", ha="center", fontsize=11, color="#203864")
            messages = [
                "reset → 22-number observation", "state → expert action",
                "episode → atomic SQL transaction", "(state, action) → gradient update",
                "fresh seed → success or failure"
            ]

            def update(frame: int):
                active = frame % len(boxes)
                for i, box in enumerate(boxes):
                    box.set_facecolor("#ffcc66" if i == active else "#e8eef7")
                status.set_text(messages[active])
                return [*boxes, status]

            anim = animation.FuncAnimation(fig, update, frames=15, interval=700, blit=False)
            display(HTML(anim.to_jshtml()))
            plt.close(fig)

        show_pipeline_animation()
        ''',
        "pipeline-animation",
    ),
    md(
        r'''
        ## 1. Contract #1: convert simulator data into learning data

        Panda Gym returns a dictionary, but an MLP expects one fixed-size vector. We choose:

        $$o = [\texttt{observation}_{19},\;\texttt{desired\_goal}_{3}] \in \mathbb{R}^{22}.$$

        We intentionally do **not** append `achieved_goal`: object position is already inside the
        19-dimensional main observation. Duplicating it would change the model without adding new
        information.

        An action is

        $$a=[\Delta x,\Delta y,\Delta z,\text{gripper}]\in[-1,1]^4.$$

        These two constants are the most important agreement in the whole project.
        ''',
        "contract-one",
    ),
    code(
        r'''
        OBS_DIM = 22
        ACTION_DIM = 4

        EE_SLICE = slice(0, 3)
        OBJECT_SLICE = slice(7, 10)
        GOAL_SLICE = slice(19, 22)

        def flatten_obs(raw_obs: dict[str, np.ndarray]) -> np.ndarray:
            """Turn Panda Gym's dictionary observation into exactly 22 float32 values."""
            main = np.asarray(raw_obs["observation"], dtype=np.float32)
            goal = np.asarray(raw_obs["desired_goal"], dtype=np.float32)
            flat = np.concatenate([main, goal])
            if flat.shape != (OBS_DIM,):
                raise ValueError(f"Expected ({OBS_DIM},), got {flat.shape}")
            return flat

        def decode_positions(flat_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return end-effector, object, and goal positions from a flat observation."""
            flat_obs = np.asarray(flat_obs)
            return flat_obs[EE_SLICE], flat_obs[OBJECT_SLICE], flat_obs[GOAL_SLICE]
        ''',
        "observation-code",
    ),
    md(
        r'''
        ### Why these functions come before every class

        They define the vocabulary used by the rest of the code. If the simulator wrapper and
        neural network disagree about the order of the 22 numbers, Python will not throw an error—the
        robot will simply learn nonsense. Shape checks turn a silent scientific bug into a loud,
        local error.

        **Mentor checkpoint:** ask the student why `shape == (22,)` is stronger than `len(x) == 22`.
        ''',
        "observation-explain",
    ),
    md(
        r'''
        ## 2. Class 1 — `GraspEnv`: a thin boundary around the simulator

        `GraspEnv` is an **adapter**. It does not re-implement physics. It gives every downstream
        component a small, stable interface:

        - `reset(seed) -> (flat_observation, info)`
        - `step(action) -> (flat_observation, reward, terminated, truncated, info)`
        - `raw_obs` remains available because the privileged expert needs exact simulator fields.

        Notice the delayed imports. Database analysis and neural-network tests should still work on a
        computer without PyBullet.
        ''',
        "env-intro",
    ),
    code(
        r'''
        class GraspEnv:
            """Expose flat states while preserving raw observations for the expert."""

            def __init__(
                self,
                render_mode: str | None = None,
                env_id: str = "PandaPickAndPlace-v3",
                max_episode_steps: int = 100,
            ) -> None:
                import gymnasium as gym
                import panda_gym  # Registers Panda environments with Gymnasium.

                kwargs: dict[str, Any] = {"max_episode_steps": max_episode_steps}
                # panda-gym headless mode requires omitting the key, not passing None.
                if render_mode is not None:
                    kwargs["render_mode"] = render_mode

                self._env = gym.make(env_id, **kwargs)
                self.env_id = env_id
                self.obs_dim = OBS_DIM
                self.action_dim = ACTION_DIM
                self.action_space = self._env.action_space
                self.raw_obs: dict[str, np.ndarray] | None = None

            def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
                raw_obs, info = self._env.reset(seed=seed)
                self.raw_obs = raw_obs
                return flatten_obs(raw_obs), info

            def step(self, action: np.ndarray):
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (ACTION_DIM,):
                    raise ValueError(f"Action must have shape ({ACTION_DIM},), got {action.shape}")
                action = np.clip(action, self.action_space.low, self.action_space.high)
                raw_obs, reward, terminated, truncated, info = self._env.step(action)
                self.raw_obs = raw_obs
                return (
                    flatten_obs(raw_obs), float(reward),
                    bool(terminated), bool(truncated), info
                )

            def close(self) -> None:
                self._env.close()

            def render(self) -> np.ndarray:
                """Return the simulator's RGB camera frame in rgb_array mode."""
                return np.asarray(self._env.render())

            def __enter__(self) -> "GraspEnv":
                return self

            def __exit__(self, *_: object) -> None:
                self.close()
        ''',
        "env-class",
    ),
    code(
        r'''
        # Contract test: fail here before building anything else.
        env = GraspEnv()
        first_obs, first_info = env.reset(seed=0)
        ee, obj, goal = decode_positions(first_obs)
        print("state shape:", first_obs.shape)
        print("action space:", env.action_space)
        print("end effector:", ee, "object:", obj, "goal:", goal)
        assert first_obs.shape == (22,)
        assert env.action_space.shape == (4,)
        ''',
        "env-smoke-test",
    ),
    md(
        r'''
        ## 3. Class 2 — `Episode`: the unit of robot experience

        A supervised-learning sample is one `(observation, action)` pair. But a robotics attempt is a
        **sequence**, and sequence boundaries matter for validation and debugging. `Episode` protects
        that boundary.

        The `__post_init__` method is a gatekeeper: construction succeeds only if all arrays agree on
        `T`. The class also records `phases` for interpreting expert behavior and
        `usable_for_training` for excluding accidental one-step successes.
        ''',
        "episode-intro",
    ),
    code(
        r'''
        @dataclass
        class Episode:
            observations: np.ndarray
            actions: np.ndarray
            rewards: np.ndarray
            success: bool
            seed: int | None = None
            usable_for_training: bool | None = None
            phases: list[str] = field(default_factory=list)

            def __post_init__(self) -> None:
                self.observations = np.asarray(self.observations, dtype=np.float32)
                self.actions = np.asarray(self.actions, dtype=np.float32)
                self.rewards = np.asarray(self.rewards, dtype=np.float32)
                t = len(self.observations)
                if self.observations.shape != (t, OBS_DIM):
                    raise ValueError(f"observations must be (T, {OBS_DIM})")
                if self.actions.shape != (t, ACTION_DIM):
                    raise ValueError(f"actions must be (T, {ACTION_DIM})")
                if self.rewards.shape != (t,):
                    raise ValueError("rewards must be (T,)")
                if self.phases and len(self.phases) != t:
                    raise ValueError("phases must be empty or contain T labels")
                if self.usable_for_training is None:
                    self.usable_for_training = self.success

            def __len__(self) -> int:
                return len(self.observations)

            @property
            def total_reward(self) -> float:
                return float(self.rewards.sum())


        class Policy(Protocol):
            """Any callable object with this signature can drive the environment."""
            def __call__(self, observation: np.ndarray) -> np.ndarray: ...
        ''',
        "episode-class",
    ),
    md(
        r'''
        ## 4. Classes 3–4 — the expert's state machine

        The expert uses one proportional controller in seven different phases:

        $$a_{xyz}=\operatorname{clip}\big(k(p_{target}-p_{ee}),-1,1\big).$$

        The equation is easy. The intellectual work is choosing a target and deciding when a phase
        is finished. We separate named phases (`ExpertPhase`) from tunable numbers (`ExpertConfig`)
        and mutable control logic (`ScriptedExpert`). This makes each responsibility inspectable.

        ```text
        APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → HOLD
        ```
        ''',
        "expert-intro",
    ),
    code(
        r'''
        class ExpertPhase(str, Enum):
            APPROACH = "approach"
            DESCEND = "descend"
            GRASP = "grasp"
            LIFT = "lift"
            TRANSPORT = "transport"
            PLACE = "place"
            HOLD = "hold"


        @dataclass(frozen=True)
        class ExpertConfig:
            hover_height: float = 0.10
            transport_clearance: float = 0.12
            grasp_offset: float = 0.015
            position_tolerance: float = 0.018
            horizontal_tolerance: float = 0.020
            action_gain: float = 20.0
            grasp_steps: int = 10
            open_action: float = 1.0
            close_action: float = -1.0
        ''',
        "expert-types",
    ),
    code(
        r'''
        class ScriptedExpert:
            """Finite-state teacher with privileged object and goal positions."""

            def __init__(self, config: ExpertConfig | None = None) -> None:
                self.config = config or ExpertConfig()
                self.reset()

            def reset(self) -> None:
                self.phase = ExpertPhase.APPROACH
                self.phase_steps = 0
                self.grasp_offset = np.array([0.0, 0.0, self.config.grasp_offset])
                self.safe_z: float | None = None

            @staticmethod
            def positions(raw_obs: dict[str, np.ndarray]):
                main = np.asarray(raw_obs["observation"])
                return main[0:3], main[7:10], np.asarray(raw_obs["desired_goal"])

            def arrived(self, ee: np.ndarray, target: np.ndarray) -> bool:
                return bool(np.linalg.norm(ee - target) < self.config.position_tolerance)

            def move_action(
                self, ee: np.ndarray, target: np.ndarray, gripper: float
            ) -> np.ndarray:
                xyz = np.clip(self.config.action_gain * (target - ee), -1.0, 1.0)
                return np.r_[xyz, np.clip(gripper, -1.0, 1.0)].astype(np.float32)

            def act(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
                ee, obj, goal = self.positions(raw_obs)
                cfg = self.config
                self.phase_steps += 1

                if self.phase is ExpertPhase.APPROACH:
                    target = obj + np.array([0.0, 0.0, cfg.hover_height])
                    action = self.move_action(ee, target, cfg.open_action)
                    xy_close = np.linalg.norm(ee[:2] - obj[:2]) < cfg.horizontal_tolerance
                    if xy_close and abs(ee[2] - target[2]) < cfg.position_tolerance:
                        self.phase, self.phase_steps = ExpertPhase.DESCEND, 0

                elif self.phase is ExpertPhase.DESCEND:
                    target = obj + np.array([0.0, 0.0, cfg.grasp_offset])
                    action = self.move_action(ee, target, cfg.open_action)
                    if self.arrived(ee, target):
                        self.grasp_offset = ee - obj
                        self.phase, self.phase_steps = ExpertPhase.GRASP, 0

                elif self.phase is ExpertPhase.GRASP:
                    target = obj + self.grasp_offset
                    action = self.move_action(ee, target, cfg.close_action)
                    if self.phase_steps >= cfg.grasp_steps:
                        self.safe_z = max(float(obj[2]), float(goal[2])) + cfg.transport_clearance
                        self.phase, self.phase_steps = ExpertPhase.LIFT, 0

                elif self.phase is ExpertPhase.LIFT:
                    if self.safe_z is None:
                        raise RuntimeError("LIFT entered without a safe height")
                    # Freeze x/y at the current hand position; only z is commanded.
                    target = np.array([ee[0], ee[1], self.safe_z])
                    action = self.move_action(ee, target, cfg.close_action)
                    if abs(ee[2] - self.safe_z) < cfg.position_tolerance:
                        self.phase, self.phase_steps = ExpertPhase.TRANSPORT, 0

                elif self.phase is ExpertPhase.TRANSPORT:
                    target = goal + self.grasp_offset
                    target[2] = self.safe_z if self.safe_z is not None else ee[2]
                    action = self.move_action(ee, target, cfg.close_action)
                    if np.linalg.norm(obj[:2] - goal[:2]) < cfg.horizontal_tolerance:
                        self.phase, self.phase_steps = ExpertPhase.PLACE, 0

                elif self.phase is ExpertPhase.PLACE:
                    target = goal + self.grasp_offset
                    action = self.move_action(ee, target, cfg.close_action)
                    if np.linalg.norm(obj - goal) < cfg.position_tolerance:
                        self.phase, self.phase_steps = ExpertPhase.HOLD, 0

                else:  # HOLD
                    target = goal + self.grasp_offset
                    action = self.move_action(ee, target, cfg.close_action)

                return action
        ''',
        "expert-class",
    ),
    md(
        r'''
        ### Read `ScriptedExpert` as three layers

        1. **Memory:** `phase`, `phase_steps`, `grasp_offset`, and `safe_z` remember facts across time.
        2. **Reusable mechanics:** `positions`, `arrived`, and `move_action` avoid copying math.
        3. **Decision tree:** `act` contains one branch per phase. Each branch chooses a target, creates
           an action, and optionally changes the next phase.

        A subtle but important bug is avoided in `LIFT`: if the target were defined as
        `object_position + clearance` on every step, the target could rise with the grasped object and
        become impossible to reach. We compute an absolute `safe_z` once.

        **Mentor checkpoint:** cover the transition code and ask the student to state the geometric
        condition that should end each phase.
        ''',
        "expert-explain",
    ),
    md(
        r'''
        ## 5. Function 1 — collect one labeled episode

        The collection loop is the first place all previous contracts meet. Record the observation
        **before** applying its action; that alignment makes row `t` mean “in state `o_t`, the expert
        chose `a_t`."
        ''',
        "collect-intro",
    ),
    code(
        r'''
        def collect_expert_episode(
            env: GraspEnv, expert: ScriptedExpert, seed: int | None = None
        ) -> Episode:
            obs, _ = env.reset(seed=seed)
            expert.reset()
            observations, actions, rewards, phases = [], [], [], []
            info: dict[str, Any] = {}

            while True:
                if env.raw_obs is None:
                    raise RuntimeError("raw_obs is missing after reset")
                phase_before_action = expert.phase.value
                action = expert.act(env.raw_obs)

                observations.append(obs.copy())
                actions.append(action.copy())
                phases.append(phase_before_action)

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break

            reached_manipulation = expert.phase in {
                ExpertPhase.TRANSPORT, ExpertPhase.PLACE, ExpertPhase.HOLD
            }
            success = bool(info.get("is_success", False))
            return Episode(
                observations=np.stack(observations),
                actions=np.stack(actions),
                rewards=np.asarray(rewards),
                success=success,
                seed=seed,
                usable_for_training=success and reached_manipulation,
                phases=phases,
            )


        expert = ScriptedExpert()
        expert_episode = collect_expert_episode(env, expert, seed=3)
        print({
            "success": expert_episode.success,
            "steps": len(expert_episode),
            "return": expert_episode.total_reward,
            "phase_counts": dict(Counter(expert_episode.phases)),
        })
        ''',
        "collect-code",
    ),
    md(
        r'''
        ## 6. Animation 1 — diagnose the expert in state space

        A simulator video is visually appealing, but a state-space animation is often more useful:
        it shows the end effector, object, goal, active phase, and object-to-goal distance together.
        The left panel is the table viewed from above; the right panel shows height.
        ''',
        "expert-animation-intro",
    ),
    code(
        r'''
        PHASE_COLORS = {
            "approach": "#4c78a8", "descend": "#72b7b2", "grasp": "#f58518",
            "lift": "#e45756", "transport": "#54a24b", "place": "#b279a2",
            "hold": "#ff9da6", "learner": "#7f7f7f",
        }

        def animate_episode(ep: Episode, title: str, max_frames: int = 55) -> HTML:
            ee = ep.observations[:, EE_SLICE]
            obj = ep.observations[:, OBJECT_SLICE]
            goal = ep.observations[:, GOAL_SLICE]
            indices = np.unique(np.linspace(0, len(ep) - 1, min(max_frames, len(ep))).astype(int))
            all_xyz = np.vstack([ee, obj, goal])
            padding = 0.035

            fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(10, 4.1))
            fig.suptitle(title, fontsize=14, weight="bold")
            ax_xy.set(xlabel="x (m)", ylabel="y (m)", title="Top view (x-y)")
            ax_xz.set(xlabel="x (m)", ylabel="z (m)", title="Side view (x-z)")
            ax_xy.set_xlim(all_xyz[:, 0].min() - padding, all_xyz[:, 0].max() + padding)
            ax_xy.set_ylim(all_xyz[:, 1].min() - padding, all_xyz[:, 1].max() + padding)
            ax_xz.set_xlim(all_xyz[:, 0].min() - padding, all_xyz[:, 0].max() + padding)
            ax_xz.set_ylim(all_xyz[:, 2].min() - padding, all_xyz[:, 2].max() + padding)

            hand_xy, = ax_xy.plot([], [], "o", ms=9, color="#1f77b4", label="end effector")
            object_xy, = ax_xy.plot([], [], "s", ms=9, color="#d62728", label="object")
            goal_xy, = ax_xy.plot([], [], "*", ms=14, color="#2ca02c", label="goal")
            trail_xy, = ax_xy.plot([], [], "-", lw=2, alpha=0.65, color="#1f77b4")
            hand_xz, = ax_xz.plot([], [], "o", ms=9, color="#1f77b4")
            object_xz, = ax_xz.plot([], [], "s", ms=9, color="#d62728")
            goal_xz, = ax_xz.plot([], [], "*", ms=14, color="#2ca02c")
            trail_xz, = ax_xz.plot([], [], "-", lw=2, alpha=0.65, color="#1f77b4")
            ax_xy.legend(loc="best", fontsize=8)
            status = fig.text(0.5, 0.01, "", ha="center", fontsize=10)

            def update(frame_index: int):
                i = int(indices[frame_index])
                phase = ep.phases[i] if ep.phases else "learner"
                color = PHASE_COLORS.get(phase, "#7f7f7f")
                hand_xy.set_data([ee[i, 0]], [ee[i, 1]])
                object_xy.set_data([obj[i, 0]], [obj[i, 1]])
                goal_xy.set_data([goal[i, 0]], [goal[i, 1]])
                trail_xy.set_data(ee[:i + 1, 0], ee[:i + 1, 1])
                hand_xz.set_data([ee[i, 0]], [ee[i, 2]])
                object_xz.set_data([obj[i, 0]], [obj[i, 2]])
                goal_xz.set_data([goal[i, 0]], [goal[i, 2]])
                trail_xz.set_data(ee[:i + 1, 0], ee[:i + 1, 2])
                for trail in (trail_xy, trail_xz):
                    trail.set_color(color)
                distance = np.linalg.norm(obj[i] - goal[i])
                status.set_text(
                    f"step {i:03d} | controller: {phase.upper()} | "
                    f"object→goal: {distance:.3f} m"
                )
                return (
                    hand_xy, object_xy, goal_xy, trail_xy,
                    hand_xz, object_xz, goal_xz, trail_xz, status,
                )

            anim = animation.FuncAnimation(
                fig, update, frames=len(indices), interval=120, blit=False, repeat=True
            )
            html = HTML(anim.to_jshtml())
            plt.close(fig)
            return html

        display(animate_episode(expert_episode, "Scripted expert: seed 3"))
        ''',
        "expert-animation",
    ),
    md(
        r'''
        ## 6.1 Animation 2 — the real 3-D Panda robot

        The previous animation is a diagnostic view of state variables. Now we record the actual
        PyBullet camera. This is the use-case animation to show in a presentation: the Panda arm
        approaches the cube, descends, closes its gripper, lifts, transports, and finishes at the
        goal. The banner displays the controller phase and live object-to-goal distance.

        The important connection is that **the video is not a separate demo**. Each frame comes from
        the same `GraspEnv.step(action)` loop that produced the learning data.
        ''',
        "real-3d-intro",
    ),
    code(
        r'''
        REAL_3D_PHASE_COLORS = {
            "approach": "#4C78A8", "descend": "#72B7B2", "grasp": "#F58518",
            "lift": "#E45756", "transport": "#54A24B", "place": "#B279A2",
            "hold": "#FF9DA6",
        }

        def tutorial_font(size: int):
            try:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except OSError:
                return ImageFont.load_default()


        def annotate_3d_frame(
            frame: np.ndarray,
            phase: str,
            step: int,
            distance: float,
            success: bool,
            width: int = 600,
        ) -> PILImage.Image:
            """Resize one simulator frame and add a readable teaching overlay."""
            image = PILImage.fromarray(frame).convert("RGB")
            height = round(image.height * width / image.width)
            image = image.resize((width, height), PILImage.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rounded_rectangle((12, 12, width - 12, 70), 10, fill=(15, 24, 39, 205))
            draw.rounded_rectangle(
                (25, 24, 145, 57), 8,
                fill=REAL_3D_PHASE_COLORS.get(phase, "#777777")
            )
            draw.text(
                (85, 40), phase.upper(), anchor="mm", fill="white",
                font=tutorial_font(15)
            )
            draw.text(
                (160, 27), f"Step {step:02d}   object → goal: {distance:.3f} m",
                fill="white", font=tutorial_font(15)
            )
            if success:
                draw.rounded_rectangle(
                    (width // 2 - 105, height - 62, width // 2 + 105, height - 18),
                    10, fill=(25, 135, 84, 225)
                )
                draw.text(
                    (width // 2, height - 40), "TASK SUCCESS", anchor="mm",
                    fill="white", font=tutorial_font(20)
                )
            return image


        def record_real_3d_expert_gif(
            output: Path, seed: int = 3, duration_ms: int = 170
        ) -> dict[str, Any]:
            """Run the real renderer, annotate frames, and save a looping GIF."""
            render_env = GraspEnv(render_mode="rgb_array")
            render_expert = ScriptedExpert()
            frames: list[PILImage.Image] = []
            success = False
            step = 0
            try:
                render_env.reset(seed=seed)
                render_expert.reset()
                while True:
                    if render_env.raw_obs is None:
                        raise RuntimeError("Missing simulator observation")
                    phase = render_expert.phase.value
                    action = render_expert.act(render_env.raw_obs)
                    _, _, terminated, truncated, info = render_env.step(action)
                    step += 1

                    main = np.asarray(render_env.raw_obs["observation"])
                    object_position = main[7:10]
                    goal_position = np.asarray(render_env.raw_obs["desired_goal"])
                    distance = float(np.linalg.norm(object_position - goal_position))
                    success = bool(info.get("is_success", False))
                    frames.append(annotate_3d_frame(
                        render_env.render(), phase, step, distance, success
                    ))
                    if terminated or truncated:
                        break
            finally:
                render_env.close()

            if success:
                frames.extend([frames[-1].copy() for _ in range(7)])
            frames[0].save(
                output, save_all=True, append_images=frames[1:], duration=duration_ms,
                loop=0, optimize=True, disposal=2
            )
            return {"seed": seed, "steps": step, "success": success, "output": str(output)}


        real_3d_gif = tutorial_dir / "expert_grasp_real_3d.gif"
        recording = record_real_3d_expert_gif(real_3d_gif, seed=3)
        print(recording)
        display(NotebookImage(filename=str(real_3d_gif)))
        ''',
        "real-3d-animation",
    ),
    code(
        r'''
        # A static diagnostic complements the animation.
        ee = expert_episode.observations[:, EE_SLICE]
        obj = expert_episode.observations[:, OBJECT_SLICE]
        goal = expert_episode.observations[:, GOAL_SLICE]
        t = np.arange(len(expert_episode))

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].plot(t, np.linalg.norm(ee - obj, axis=1), label="hand → object")
        axes[0].plot(t, np.linalg.norm(obj - goal, axis=1), label="object → goal")
        axes[0].set(xlabel="step", ylabel="distance (m)", title="Task distances")
        axes[0].legend()
        counts = Counter(expert_episode.phases)
        axes[1].bar(counts.keys(), counts.values(), color=[PHASE_COLORS[k] for k in counts])
        axes[1].set(xlabel="phase", ylabel="number of steps", title="Phase balance")
        axes[1].tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.show()
        ''',
        "expert-diagnostics",
    ),
    md(
        r'''
        ## 7. Class 5 — `TrainingData`: arrays after the database

        `Episode` validates one trajectory with length `T`. `TrainingData` validates the combined
        supervised dataset with length `N`. It also keeps `episode_ids`, which will prevent leakage
        during validation, and `phases`, which will let us correct phase imbalance.
        ''',
        "training-data-intro",
    ),
    code(
        r'''
        @dataclass
        class TrainingData:
            observations: np.ndarray
            actions: np.ndarray
            episode_ids: np.ndarray
            phases: np.ndarray

            def __post_init__(self) -> None:
                n = len(self.observations)
                if self.observations.shape != (n, OBS_DIM):
                    raise ValueError(f"observations must be (N, {OBS_DIM})")
                if self.actions.shape != (n, ACTION_DIM):
                    raise ValueError(f"actions must be (N, {ACTION_DIM})")
                if self.episode_ids.shape != (n,) or self.phases.shape != (n,):
                    raise ValueError("episode_ids and phases must each contain N values")

            def __len__(self) -> int:
                return len(self.observations)
        ''',
        "training-data-class",
    ),
    md(
        r'''
        ## 8. Class 6 — `DemoDatabase`: durable, queryable demonstrations

        We use two tables because episode facts and timestep facts have different natural units.

        ```text
        episodes (one row per attempt)       steps (one row per action)
        ┌────────────┬──────┬─────────┐      ┌────────────┬──────┬─────────────┐
        │ episode_id │ seed │ success │ 1──∞ │ episode_id │ step │ observation │
        └────────────┴──────┴─────────┘      └────────────┴──────┴─────────────┘
        ```

        SQLite transactions make `save_episode` atomic: an interruption cannot leave half an episode
        pretending to be valid training data. JSON is not the fastest array encoding, but it is
        readable and perfectly adequate for this teaching-scale dataset.
        ''',
        "database-intro",
    ),
    code(
        r'''
        class DemoDatabase:
            def __init__(self, path: str | Path) -> None:
                self.path = Path(path)
                self.connection = sqlite3.connect(self.path)
                self.connection.execute("PRAGMA foreign_keys = ON")
                self._create_schema()

            def _create_schema(self) -> None:
                self.connection.executescript("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        seed INTEGER,
                        success INTEGER NOT NULL CHECK(success IN (0, 1)),
                        usable INTEGER NOT NULL CHECK(usable IN (0, 1)),
                        n_steps INTEGER NOT NULL,
                        total_reward REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS steps (
                        episode_id INTEGER NOT NULL,
                        step INTEGER NOT NULL,
                        observation TEXT NOT NULL,
                        action TEXT NOT NULL,
                        reward REAL NOT NULL,
                        phase TEXT NOT NULL,
                        PRIMARY KEY (episode_id, step),
                        FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_success ON episodes(success, usable);
                    CREATE INDEX IF NOT EXISTS idx_phase ON steps(phase);
                """)
                self.connection.commit()

            def save_episode(self, episode: Episode) -> int:
                phases = episode.phases or [""] * len(episode)
                # The context manager begins a transaction and rolls back on error.
                with self.connection:
                    cursor = self.connection.execute("""
                        INSERT INTO episodes(seed, success, usable, n_steps, total_reward)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        episode.seed, int(episode.success),
                        int(bool(episode.usable_for_training)),
                        len(episode), episode.total_reward,
                    ))
                    episode_id = int(cursor.lastrowid)
                    rows = [
                        (
                            episode_id, step,
                            json.dumps(obs.tolist(), separators=(",", ":")),
                            json.dumps(action.tolist(), separators=(",", ":")),
                            float(reward), phase,
                        )
                        for step, (obs, action, reward, phase) in enumerate(zip(
                            episode.observations, episode.actions, episode.rewards, phases
                        ))
                    ]
                    self.connection.executemany("""
                        INSERT INTO steps(episode_id, step, observation, action, reward, phase)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
                return episode_id

            def load_training_data(
                self, successful_only: bool = True, max_episodes: int | None = None
            ) -> TrainingData:
                where = "WHERE success = 1 AND usable = 1" if successful_only else ""
                limit = "LIMIT ?" if max_episodes is not None else ""
                params = (max_episodes,) if max_episodes is not None else ()
                rows = self.connection.execute(f"""
                    WITH selected AS (
                        SELECT episode_id FROM episodes {where}
                        ORDER BY episode_id {limit}
                    )
                    SELECT s.episode_id, s.observation, s.action, s.phase
                    FROM steps s JOIN selected x USING(episode_id)
                    ORDER BY s.episode_id, s.step
                """, params).fetchall()
                if not rows:
                    raise ValueError("No matching demonstration steps")
                return TrainingData(
                    observations=np.asarray([json.loads(r[1]) for r in rows], dtype=np.float32),
                    actions=np.asarray([json.loads(r[2]) for r in rows], dtype=np.float32),
                    episode_ids=np.asarray([r[0] for r in rows], dtype=np.int64),
                    phases=np.asarray([r[3] for r in rows], dtype=str),
                )

            def summary(self) -> dict[str, int | float]:
                episodes, wins, usable, steps = self.connection.execute("""
                    SELECT COUNT(*), COALESCE(SUM(success), 0),
                           COALESCE(SUM(success * usable), 0), COALESCE(SUM(n_steps), 0)
                    FROM episodes
                """).fetchone()
                return {
                    "episodes": int(episodes), "successes": int(wins),
                    "usable_demos": int(usable), "steps": int(steps),
                    "expert_success_rate": float(wins / episodes) if episodes else 0.0,
                }

            def close(self) -> None:
                self.connection.close()
        ''',
        "database-class",
    ),
    md(
        r'''
        ## 9. Collect a real dataset and inspect it before training

        We keep attempting new seeds until the database contains the target number of **successful,
        usable** demonstrations. Failed expert episodes are still saved: they are valuable diagnostic
        evidence even though basic behavior cloning does not train on them.

        In fast mode this is a classroom smoke test. Change `FAST_MODE = False` at the top for a
        100-demonstration run.
        ''',
        "dataset-intro",
    ),
    code(
        r'''
        database = DemoDatabase(tutorial_dir / "demonstrations.sqlite")
        expert = ScriptedExpert()
        attempts = 0
        usable = 0
        max_attempts = max(60, TARGET_DEMOS * 3)

        while usable < TARGET_DEMOS and attempts < max_attempts:
            episode = collect_expert_episode(env, expert, seed=attempts)
            database.save_episode(episode)
            usable += int(bool(episode.usable_for_training))
            attempts += 1
            if attempts % 10 == 0 or usable == TARGET_DEMOS:
                print(f"attempts={attempts:3d}  usable demonstrations={usable:3d}/{TARGET_DEMOS}")

        summary = database.summary()
        print(summary)
        if usable < 5:
            raise RuntimeError("Too few usable expert episodes; inspect the expert before training")

        data = database.load_training_data(successful_only=True, max_episodes=TARGET_DEMOS)
        print("observations:", data.observations.shape)
        print("actions:     ", data.actions.shape)
        print("episode IDs: ", np.unique(data.episode_ids).shape)
        ''',
        "dataset-collect",
    ),
    code(
        r'''
        # Dataset dashboard: phase imbalance, action distributions, and state coverage.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        phase_counts = Counter(data.phases)
        axes[0, 0].bar(
            phase_counts.keys(), phase_counts.values(),
            color=[PHASE_COLORS.get(k, "#777777") for k in phase_counts]
        )
        axes[0, 0].set(title="Samples by expert phase", ylabel="samples")
        axes[0, 0].tick_params(axis="x", rotation=30)

        action_labels = ["dx", "dy", "dz", "gripper"]
        for j, label in enumerate(action_labels):
            axes[0, 1].hist(data.actions[:, j], bins=25, alpha=0.55, label=label)
        axes[0, 1].set(title="Expert action distribution", xlabel="action", ylabel="count")
        axes[0, 1].legend()

        sc = axes[1, 0].scatter(
            data.observations[:, 7], data.observations[:, 8],
            c=data.observations[:, 9], s=7, alpha=0.45, cmap="viridis"
        )
        axes[1, 0].set(title="Object-state coverage", xlabel="object x (m)", ylabel="object y (m)")
        fig.colorbar(sc, ax=axes[1, 0], label="object z (m)")

        ids, lengths = np.unique(data.episode_ids, return_counts=True)
        axes[1, 1].hist(lengths, bins=min(15, len(np.unique(lengths)) + 2), color="#4c78a8")
        axes[1, 1].set(title="Successful episode lengths", xlabel="steps", ylabel="episodes")
        plt.tight_layout()
        plt.show()
        ''',
        "dataset-dashboard",
    ),
    md(
        r'''
        **Mentor checkpoint:** before training, ask for three predictions:

        1. Which phase dominates the data, and why?
        2. Why is the gripper histogram likely bimodal?
        3. What kind of test state is missing from the object-position scatter plot?

        These questions turn “we collected data” into “we understand what the learner is allowed to
        learn from.”
        ''',
        "dataset-checkpoint",
    ),
    md(
        r'''
        ## 10. Classes 7–8 — `GraspPolicy` and `NumpyPolicy`

        The neural network is deliberately small: `22 → 128 → 128 → 4`. The key systems decisions
        are more important than the exact width:

        - training-set mean/std are registered as model **buffers**, so they move with the device and
          are included in checkpoints;
        - `tanh` enforces the action contract `[-1, 1]`;
        - `NumpyPolicy` adapts a batched PyTorch module to the simulator's single NumPy observation.

        The adapter means evaluation does not need to know PyTorch exists.
        ''',
        "policy-intro",
    ),
    code(
        r'''
        class GraspPolicy(nn.Module):
            def __init__(
                self,
                obs_dim: int = OBS_DIM,
                action_dim: int = ACTION_DIM,
                hidden_sizes: Sequence[int] = (128, 128),
            ) -> None:
                super().__init__()
                if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
                    raise ValueError("hidden_sizes must contain positive widths")
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.hidden_sizes = tuple(hidden_sizes)

                layers: list[nn.Module] = []
                input_width = obs_dim
                for width in hidden_sizes:
                    layers.extend([nn.Linear(input_width, width), nn.ReLU()])
                    input_width = width
                layers.extend([nn.Linear(input_width, action_dim), nn.Tanh()])
                self.network = nn.Sequential(*layers)

                self.register_buffer("obs_mean", torch.zeros(obs_dim))
                self.register_buffer("obs_std", torch.ones(obs_dim))

            def set_normalization(self, observations: np.ndarray) -> None:
                values = torch.as_tensor(
                    observations, dtype=torch.float32, device=self.obs_mean.device
                )
                self.obs_mean.copy_(values.mean(dim=0))
                self.obs_std.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6))

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                normalized = (observations - self.obs_mean) / self.obs_std
                return self.network(normalized)

            def save(self, path: str | Path) -> None:
                torch.save({
                    "state_dict": self.state_dict(),
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_sizes": self.hidden_sizes,
                }, path)

            @classmethod
            def load(cls, path: str | Path, device: str = "cpu") -> "GraspPolicy":
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                model = cls(
                    checkpoint["obs_dim"], checkpoint["action_dim"],
                    checkpoint["hidden_sizes"]
                )
                model.load_state_dict(checkpoint["state_dict"])
                return model.to(device).eval()


        class NumpyPolicy:
            def __init__(self, model: GraspPolicy, device: str = "cpu") -> None:
                self.model = model.to(device).eval()
                self.device = device

            def __call__(self, observation: np.ndarray) -> np.ndarray:
                x = torch.as_tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.inference_mode():
                    action = self.model(x).squeeze(0)
                return action.cpu().numpy().astype(np.float32)


        untrained = GraspPolicy()
        n_parameters = sum(p.numel() for p in untrained.parameters() if p.requires_grad)
        print(untrained)
        print(f"Trainable parameters: {n_parameters:,}")
        ''',
        "policy-classes",
    ),
    md(
        r'''
        ## 11. Class 9 and Function 2 — train behavior cloning correctly

        The objective is mean squared expert-action error:

        $$\mathcal L(\theta)=\frac{1}{N}\sum_i\|\pi_\theta(o_i)-a_i^*\|_2^2.$$

        But two details make this a robotics-quality training function:

        **Episode-level split.** Adjacent simulator states are nearly identical. Splitting individual
        timesteps would place near-duplicates from one trajectory in both training and validation.

        **Phase balancing.** Long phases otherwise dominate ordinary MSE. Bounded inverse-frequency
        weights let rare grasp/place decisions contribute without creating extreme gradients.
        ''',
        "training-intro",
    ),
    code(
        r'''
        @dataclass
        class TrainResult:
            model: GraspPolicy
            train_loss: list[float]
            validation_loss: list[float]
            train_episode_ids: np.ndarray
            validation_episode_ids: np.ndarray


        def phase_weights(phases: np.ndarray) -> np.ndarray:
            if len(phases) == 0 or np.all(phases == ""):
                return np.ones(len(phases), dtype=np.float32)
            labels, counts = np.unique(phases, return_counts=True)
            inverse = {
                label: len(phases) / (len(labels) * count)
                for label, count in zip(labels, counts)
            }
            weights = np.asarray([inverse[label] for label in phases], dtype=np.float32)
            return np.clip(weights, 0.25, 4.0)


        def train_behavior_cloning(
            data: TrainingData,
            hidden_sizes: Sequence[int] = (128, 128),
            epochs: int = 80,
            batch_size: int = 256,
            learning_rate: float = 1e-3,
            validation_fraction: float = 0.2,
            seed: int = 0,
            balance_phases: bool = True,
            device: str = "cpu",
        ) -> TrainResult:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # 1. Split whole episode IDs.
            unique_ids = np.unique(data.episode_ids)
            shuffled = np.random.default_rng(seed).permutation(unique_ids)
            n_val = max(1, round(len(shuffled) * validation_fraction)) if len(shuffled) >= 2 else 0
            n_val = min(n_val, max(0, len(shuffled) - 1))
            val_ids = np.sort(shuffled[:n_val])
            train_ids = np.sort(shuffled[n_val:])
            train_mask = np.isin(data.episode_ids, train_ids)
            val_mask = np.isin(data.episode_ids, val_ids)

            # 2. Construct and normalize the model using training states only.
            model = GraspPolicy(hidden_sizes=hidden_sizes).to(device)
            model.set_normalization(data.observations[train_mask])

            # 3. Build shuffled mini-batches with an optional weight per sample.
            weights = (
                phase_weights(data.phases[train_mask])
                if balance_phases else np.ones(train_mask.sum(), dtype=np.float32)
            )
            dataset = TensorDataset(
                torch.as_tensor(data.observations[train_mask], dtype=torch.float32),
                torch.as_tensor(data.actions[train_mask], dtype=torch.float32),
                torch.as_tensor(weights, dtype=torch.float32),
            )
            loader = DataLoader(
                dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
                generator=torch.Generator().manual_seed(seed)
            )
            x_val = torch.as_tensor(data.observations[val_mask], dtype=torch.float32, device=device)
            y_val = torch.as_tensor(data.actions[val_mask], dtype=torch.float32, device=device)

            # 4. Optimize, measure validation, and remember the best checkpoint.
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_history, val_history = [], []
            best_state = deepcopy(model.state_dict())
            best_val = float("inf")

            for _ in range(epochs):
                model.train()
                weighted_error, total_weight = 0.0, 0.0
                for x_batch, y_batch, w_batch in loader:
                    x_batch, y_batch, w_batch = (
                        x_batch.to(device), y_batch.to(device), w_batch.to(device)
                    )
                    optimizer.zero_grad()
                    per_sample = (model(x_batch) - y_batch).square().mean(dim=1)
                    loss = (per_sample * w_batch).sum() / w_batch.sum()
                    loss.backward()
                    optimizer.step()
                    weighted_error += float((per_sample.detach() * w_batch).sum())
                    total_weight += float(w_batch.sum())
                train_history.append(weighted_error / total_weight)

                model.eval()
                with torch.inference_mode():
                    val_loss = float((model(x_val) - y_val).square().mean()) if n_val else np.nan
                val_history.append(val_loss)
                if not n_val or val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            return TrainResult(model.eval(), train_history, val_history, train_ids, val_ids)
        ''',
        "training-code",
    ),
    code(
        r'''
        result = train_behavior_cloning(data, epochs=TRAIN_EPOCHS, seed=7)
        assert set(result.train_episode_ids).isdisjoint(result.validation_episode_ids)
        print("train episodes:", len(result.train_episode_ids))
        print("validation episodes:", len(result.validation_episode_ids))
        print("best validation MSE:", np.nanmin(result.validation_loss))

        checkpoint_path = tutorial_dir / "behavior_cloning.pt"
        result.model.save(checkpoint_path)
        reloaded_model = GraspPolicy.load(checkpoint_path)
        learned_policy = NumpyPolicy(reloaded_model)
        print("checkpoint round-trip:", checkpoint_path)
        ''',
        "train-run",
    ),
    code(
        r'''
        # Learning curves plus per-action prediction diagnostics.
        val_mask = np.isin(data.episode_ids, result.validation_episode_ids)
        x_val = torch.as_tensor(data.observations[val_mask], dtype=torch.float32)
        with torch.inference_mode():
            predictions = result.model(x_val).cpu().numpy()
        targets = data.actions[val_mask]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(result.train_loss, label="train")
        axes[0].plot(result.validation_loss, label="validation")
        axes[0].set(xlabel="epoch", ylabel="MSE", title="Behavior-cloning loss")
        axes[0].set_yscale("log")
        axes[0].legend()

        for j, label in enumerate(["dx", "dy", "dz"]):
            axes[1].scatter(targets[:, j], predictions[:, j], s=8, alpha=0.28, label=label)
        axes[1].plot([-1, 1], [-1, 1], "k--", lw=1)
        axes[1].set(
            xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
            xlabel="expert action", ylabel="predicted action", title="Motion predictions"
        )
        axes[1].legend()

        axes[2].scatter(targets[:, 3], predictions[:, 3], s=9, alpha=0.35, color="#f58518")
        axes[2].plot([-1, 1], [-1, 1], "k--", lw=1)
        axes[2].set(
            xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
            xlabel="expert gripper", ylabel="predicted gripper", title="Gripper predictions"
        )
        plt.tight_layout()
        plt.show()
        ''',
        "training-dashboard",
    ),
    md(
        r'''
        ### What the loss can—and cannot—prove

        A falling validation loss shows that the network imitates expert actions on **expert-visited
        states** from held-out episodes. It does not show that the robot succeeds when its own earlier
        mistakes determine the next state. That is a different experiment: closed-loop evaluation.

        **Mentor checkpoint:** point at a gripper prediction near zero and ask what physical command
        it represents. Regression can average `open = +1` and `close = -1` into indecision.
        ''',
        "loss-meaning",
    ),
    md(
        r'''
        ## 12. Function 3 — a policy-agnostic rollout

        Because both expert adapters and learned policies can satisfy the `Policy` protocol, one
        rollout function can evaluate any flat-observation policy. This is polymorphism doing useful
        scientific work: the experimental loop stays identical when the model changes.
        ''',
        "rollout-intro",
    ),
    code(
        r'''
        def rollout_episode(env: GraspEnv, policy: Policy, seed: int | None = None) -> Episode:
            obs, _ = env.reset(seed=seed)
            observations, actions, rewards = [], [], []
            info: dict[str, Any] = {}

            while True:
                action = np.asarray(policy(obs), dtype=np.float32)
                observations.append(obs.copy())
                actions.append(action.copy())
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break

            return Episode(
                observations=np.stack(observations),
                actions=np.stack(actions),
                rewards=np.asarray(rewards),
                success=bool(info.get("is_success", False)),
                seed=seed,
            )
        ''',
        "rollout-code",
    ),
    md(
        r'''
        ## 13. Class 10 — `EvaluationResult`: conclusions need uncertainty

        Success is binomial: each episode either succeeds or fails. `8/10` and `80/100` have the same
        point estimate but very different certainty. The Wilson interval behaves better than the
        naive normal interval, especially with few trials or rates near 0% and 100%.
        ''',
        "evaluation-intro",
    ),
    code(
        r'''
        def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
            if trials <= 0 or not 0 <= successes <= trials:
                raise ValueError("Require 0 <= successes <= trials and trials > 0")
            p = successes / trials
            denominator = 1 + z**2 / trials
            center = (p + z**2 / (2 * trials)) / denominator
            margin = z * sqrt(
                p * (1 - p) / trials + z**2 / (4 * trials**2)
            ) / denominator
            return center - margin, center + margin


        @dataclass
        class EvaluationResult:
            successes: np.ndarray
            episode_returns: np.ndarray
            episode_lengths: np.ndarray
            seeds: np.ndarray

            @property
            def n_episodes(self) -> int:
                return len(self.successes)

            @property
            def success_rate(self) -> float:
                return float(self.successes.mean())

            @property
            def confidence_interval(self) -> tuple[float, float]:
                return wilson_interval(int(self.successes.sum()), self.n_episodes)

            def as_dict(self) -> dict[str, Any]:
                return {
                    "episodes": self.n_episodes,
                    "successes": int(self.successes.sum()),
                    "success_rate": self.success_rate,
                    "ci95": list(self.confidence_interval),
                    "mean_return": float(self.episode_returns.mean()),
                    "mean_length": float(self.episode_lengths.mean()),
                }


        def evaluate_policy(
            env: GraspEnv,
            policy: Policy,
            n_episodes: int = 100,
            seed_start: int = 10_000,
        ) -> EvaluationResult:
            seeds = np.arange(seed_start, seed_start + n_episodes, dtype=np.int64)
            episodes = [rollout_episode(env, policy, seed=int(seed)) for seed in seeds]
            return EvaluationResult(
                successes=np.asarray([ep.success for ep in episodes], dtype=bool),
                episode_returns=np.asarray([ep.total_reward for ep in episodes]),
                episode_lengths=np.asarray([len(ep) for ep in episodes]),
                seeds=seeds,
            )
        ''',
        "evaluation-code",
    ),
    code(
        r'''
        evaluation = evaluate_policy(
            env, learned_policy, n_episodes=EVAL_EPISODES, seed_start=10_000
        )
        metrics = evaluation.as_dict()
        print(json.dumps(metrics, indent=2))

        rate = metrics["success_rate"]
        low, high = metrics["ci95"]
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.bar(["learned policy"], [rate], color="#4c78a8", width=0.5)
        ax.errorbar(
            [0], [rate], yerr=[[rate - low], [high - rate]],
            fmt="none", ecolor="#203864", capsize=8, lw=2
        )
        ax.set(ylim=(0, 1), ylabel="closed-loop success rate", title="Fresh evaluation seeds")
        ax.text(0, min(rate + 0.08, 0.93), f"{metrics['successes']}/{metrics['episodes']}", ha="center")
        plt.show()
        ''',
        "evaluation-run",
    ),
    md(
        r'''
        ## 14. Animation 2 — let the learner reveal distribution shift

        The learner receives no phase label and no corrections from the expert. Its state at step
        `t+1` is caused by its own action at step `t`. Watch for the **first** meaningful departure:
        overshooting the object, closing too early, failing to lift, or drifting during transport.

        A failure is not just a bad score. It is a trajectory that can support a testable diagnosis.
        ''',
        "learner-animation-intro",
    ),
    code(
        r'''
        # Prefer a failed seed because it is often more educational; otherwise use the first seed.
        failed = np.flatnonzero(~evaluation.successes)
        chosen_index = int(failed[0]) if len(failed) else 0
        chosen_seed = int(evaluation.seeds[chosen_index])
        learner_episode = rollout_episode(env, learned_policy, seed=chosen_seed)
        display(animate_episode(
            learner_episode,
            f"Learned policy: seed {chosen_seed} | success={learner_episode.success}"
        ))
        ''',
        "learner-animation",
    ),
    code(
        r'''
        # Compare expert and learner on exactly the same initial condition.
        same_seed_expert = collect_expert_episode(env, ScriptedExpert(), seed=chosen_seed)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
        for ax, ep, name, color in [
            (axes[0], same_seed_expert, "expert", "#2ca02c"),
            (axes[1], learner_episode, "learner", "#d62728"),
        ]:
            hand = ep.observations[:, EE_SLICE]
            obj = ep.observations[:, OBJECT_SLICE]
            goal = ep.observations[:, GOAL_SLICE]
            ax.plot(hand[:, 0], hand[:, 1], color=color, lw=2, label="hand path")
            ax.scatter(obj[0, 0], obj[0, 1], marker="s", s=60, label="object start")
            ax.scatter(goal[0, 0], goal[0, 1], marker="*", s=130, label="goal")
            ax.set(
                xlabel="x (m)", ylabel="y (m)",
                title=f"{name}: success={ep.success}, steps={len(ep)}"
            )
            ax.legend(fontsize=8)
        fig.suptitle(f"Controlled comparison on seed {chosen_seed}", weight="bold")
        plt.tight_layout()
        plt.show()
        ''',
        "comparison",
    ),
    md(
        r'''
        ## 15. The learning mechanism: why behavior cloning eventually fails

        The database contains states visited by the expert, approximately
        $o\sim d_{\pi^*}$. Closed-loop deployment creates states visited by the learner,
        $o\sim d_{\pi_\theta}$. These distributions need not match.

        ```text
        small action error
              ↓
        slightly unfamiliar next state
              ↓
        larger prediction error
              ↓
        even less familiar state  →  compounding error
        ```

        This is why validation MSE and closed-loop success answer different questions.

        A natural extension is **DAgger**:

        1. roll out the current learner;
        2. ask the expert to label the learner-visited states;
        3. aggregate those new `(state, expert action)` pairs;
        4. retrain and repeat.

        DAgger changes the training distribution, not merely the neural-network architecture.
        ''',
        "learning-mechanism",
    ),
    md(
        r'''
        ## 16. Turn the pipeline into an experiment

        A full scientific result repeats the pipeline while changing **one independent variable**.
        The most informative first experiment is number of successful expert demonstrations:

        ```python
        demo_counts = [10, 25, 50, 100, 200]
        training_seeds = [0, 1, 2]

        for n_demos in demo_counts:
            subset = database.load_training_data(max_episodes=n_demos)
            for seed in training_seeds:
                trained = train_behavior_cloning(subset, seed=seed)
                score = evaluate_policy(env, NumpyPolicy(trained.model),
                                        n_episodes=100, seed_start=10_000)
                record(n_demos, seed, score.success_rate)
        ```

        Keep the evaluation seeds fixed across conditions. That is a *paired comparison*: every model
        faces the same initial scenes. Report mean ± standard deviation across training seeds, plus
        the raw success counts.

        Do not launch the full loop during a 45-minute lesson. The production command is:

        ```bash
        .sim-env/bin/python scripts/run_experiments.py \
          --database artifacts/demonstrations.sqlite \
          --output artifacts/experiments
        ```
        ''',
        "experiments",
    ),
    md(
        r'''
        ## 17. From notebook classes to production files

        The notebook used one linear story. A real project separates code by responsibility so it can
        be tested and reused from command-line scripts.

        | What you built here | Production file | Why it belongs there |
        |---|---|---|
        | `flatten_obs`, shape constants | `src/ilgrasp/sim/observation.py` | one observation contract |
        | `GraspEnv` | `src/ilgrasp/sim/env.py` | simulator boundary only |
        | `Episode`, `rollout_episode` | `src/ilgrasp/sim/rollout.py` | trajectory representation and execution |
        | `ExpertPhase`, `ExpertConfig`, `ScriptedExpert` | `src/ilgrasp/experts/scripted.py` | privileged teacher logic |
        | `TrainingData`, `DemoDatabase` | `src/ilgrasp/data/database.py` | persistent data contract |
        | `GraspPolicy`, `NumpyPolicy` | `src/ilgrasp/policies/mlp.py` | model architecture and inference adapter |
        | `TrainResult`, `train_behavior_cloning` | `src/ilgrasp/policies/training.py` | optimization and split rules |
        | `EvaluationResult`, `evaluate_policy` | `src/ilgrasp/evaluation/evaluate.py` | closed-loop measurement |
        | data/learning-curve loops | `scripts/run_experiments.py` | reproducible experiment orchestration |

        Supporting entry points each perform one operation:

        - `scripts/inspect_env.py`: verify simulator shapes before expensive work;
        - `scripts/collect_demos.py`: expert → SQLite;
        - `scripts/train.py`: SQLite → model checkpoint;
        - `scripts/evaluate.py`: checkpoint → fresh-seed metrics;
        - `scripts/run_experiments.py`: repeated controlled comparisons;
        - `tests/`: protect data shapes, database behavior, training split, and statistics.
        ''',
        "file-map",
    ),
    md(
        r'''
        ## 18. Student challenge sequence

        Do these in order; each challenge changes only one layer.

        1. **Observation audit (10 min):** print labels for all 22 state values and explain which are
           privileged versus available to the learner.
        2. **Expert ablation (15 min):** halve `transport_clearance`; predict and measure which phase
           fails more often.
        3. **Data experiment (25 min):** compare 10 and 50 demos using identical evaluation seeds.
        4. **Model experiment (20 min):** compare `(64, 64)` and `(256, 256)` with the same data.
        5. **Failure report (20 min):** animate one failed seed, identify the first meaningful error,
           and propose one falsifiable fix.
        6. **DAgger design (advanced):** write pseudocode that labels learner-visited states without
           accidentally storing the learner's own incorrect actions as expert targets.

        A good final presentation has four panels:

        - expert trajectory and phase timing;
        - dataset coverage and action distribution;
        - train/validation loss plus action-prediction scatter;
        - closed-loop success with uncertainty and one animated failure case.
        ''',
        "challenges",
    ),
    code(
        r'''
        # Clean shutdown. Run this only after finishing the animations and experiments.
        database.close()
        env.close()
        print("Closed simulator and SQLite connection.")
        print("Temporary artifacts remain available until the operating system cleans:", tutorial_dir)
        ''',
        "cleanup",
    ),
    md(
        r'''
        ## Exit ticket

        In one minute, explain the complete project without naming a library:

        > The simulator produces states. A scripted expert labels those states with actions. We keep
        > complete episodes in a database, split by episode, train a normalized policy to imitate the
        > actions, and then test that policy in closed loop on fresh initial conditions. Validation
        > loss measures imitation on expert states; success rate measures actual task performance.

        If you can explain why every arrow in that sentence needs a shape check, seed rule, or
        boundary, you understand the pipeline—not just the neural network inside it.
        ''',
        "exit-ticket",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ilgrasp)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} ({len(cells)} cells)")
