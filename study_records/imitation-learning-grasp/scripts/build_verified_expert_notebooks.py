#!/usr/bin/env python3
"""Build four synchronized expert-first notebooks in English and Chinese."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


@dataclass(frozen=True)
class LessonCell:
    cell_id: str
    kind: str
    en: str = ""
    zh: str = ""
    starter: str | None = None
    solution: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def render(self, language: str, solutions: bool) -> dict:
        if self.kind == "markdown":
            source = self.en if language == "en" else self.zh
            return {
                "cell_type": "markdown",
                "id": self.cell_id,
                "metadata": {"tags": list(self.tags)} if self.tags else {},
                "source": lines(source),
            }
        if self.kind != "code":
            raise ValueError(self.kind)
        if self.starter is not None:
            source = self.solution if solutions else self.starter
        else:
            source = self.en
        return {
            "cell_type": "code",
            "id": self.cell_id,
            "metadata": {"tags": list(self.tags)} if self.tags else {},
            "execution_count": None,
            "outputs": [],
            "source": lines(source or ""),
        }


CELLS: list[LessonCell] = []


def m(cell_id: str, en: str, zh: str, *tags: str) -> None:
    CELLS.append(LessonCell(cell_id, "markdown", en=en, zh=zh, tags=tags))


def c(cell_id: str, source: str, *tags: str) -> None:
    CELLS.append(LessonCell(cell_id, "code", en=source, tags=tags))


def ex(cell_id: str, starter: str, solution: str) -> None:
    CELLS.append(
        LessonCell(
            cell_id,
            "code",
            starter=starter,
            solution=solution,
            tags=("exercise", "solution-replaced"),
        )
    )


m(
    "title",
    r'''
    # Verified Pick–Cross–Place: Build the Expert, Then Teach a Learner

    **Expert-first lab · student/solutions editions share exactly the same cell order**

    You already built the simulator. We will not repeat it. We will extend its task contract, then
    build the expert from geometric primitives, transition guards, memory, recovery, data collection,
    a demonstration database, hybrid behavior cloning, and strict closed-loop evaluation.

    The mission is intentionally harder than ordinary `PandaPickAndPlace-v3`: carry an orange cube
    over a physical blue barrier, place it on the opposite side, open the fingers, retreat the arm,
    and prove that the released cube remains stable for 12 consecutive steps.
    ''',
    r'''
    # 严格抓取—越障—放置：先构建 Expert，再训练 Learner

    **以 Expert 为主线 · 学生版与答案版拥有完全相同的单元顺序**

    你已经完成 simulator，本实验不再重复它。我们先扩展任务完成条件，再从几何控制、状态转移、
    控制器记忆与失败恢复开始构建 expert，随后连接数据收集、示范数据库、混合行为克隆和严格闭环评估。

    任务比普通 `PandaPickAndPlace-v3` 更复杂：把橙色方块抬过真实蓝色障碍物，在另一侧放下，
    打开夹爪，机械臂撤离，并证明释放后的方块连续 12 步保持稳定。
    ''',
    "overview",
)

c(
    "setup",
    r'''
    from __future__ import annotations

    import json
    import random
    import sqlite3
    import sys
    import tempfile
    from collections import Counter
    from copy import deepcopy
    from dataclasses import dataclass, field
    from enum import Enum
    from math import sqrt
    from pathlib import Path
    from typing import Any

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from IPython.display import Image as NotebookImage, display
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Locate the repository whether Jupyter starts in the root or notebooks/.
    candidates = [Path.cwd(), Path.cwd().parent]
    PROJECT_ROOT = next(path for path in candidates if (path / "src" / "ilgrasp").exists())
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from ilgrasp.data import TrainingData
    from ilgrasp.sim import Episode, VerifiedPlacementEnv
    from ilgrasp.visualization import record_verified_expert_gif

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})
    np.set_printoptions(precision=3, suppress=True)

    QUICK_MODE = True
    EXPERT_BENCHMARK_EPISODES = 12 if QUICK_MODE else 100
    TARGET_DEMOS = 24 if QUICK_MODE else 100
    TRAIN_EPOCHS = 45 if QUICK_MODE else 100
    EVAL_EPISODES = 12 if QUICK_MODE else 100
    tutorial_dir = Path(tempfile.mkdtemp(prefix="verified_expert_lab_"))
    print("project:", PROJECT_ROOT)
    print("temporary run directory:", tutorial_dir)
    ''',
    "setup",
)

m(
    "mission",
    r'''
    ## 0. Begin with the finished use case

    The animation below is generated from the real PyBullet simulator—not a hand-drawn robot. It has
    two cameras, the complete 13-phase timeline, a live completion certificate, native versus strict
    success indicators, and physical telemetry. Watch what happens **after** `NATIVE HIT` becomes
    yellow. The expert must still open, retreat, and verify stability before `VERIFIED` turns green.
    ''',
    r'''
    ## 0. 先观察真正完成的 use case

    下面不是手绘机器人，而是直接来自 PyBullet 的真实仿真画面。动画包含双镜头、13 个 phase 的完整
    时间轴、实时完成证书，以及 native 与 strict success 的对比。注意 `NATIVE HIT` 变黄之后任务并未
    结束；expert 还必须松爪、撤离并验证稳定，最后 `VERIFIED` 才会变绿。
    ''',
)

c(
    "show-reference-animation",
    r'''
    animation_path = PROJECT_ROOT / "notebooks" / "assets" / "verified_pick_cross_place.gif"
    if not animation_path.exists():
        record_verified_expert_gif(animation_path, seed=7)
    display(NotebookImage(filename=str(animation_path)))
    ''',
    "visualization",
)

m(
    "task-contract",
    r'''
    ## 1. The provided task adapter: same simulator, stricter meaning of success

    Panda Gym's native task stops when the cube first enters a 5 cm sphere around the goal. It may
    still be held in a closed gripper—and 70% of native goals can be suspended in mid-air. That metric
    is not a completed placement.

    `VerifiedPlacementEnv` is a thin adapter around the simulator you already know. It preserves the
    exact observation `(22,)` and action `(4,)` contracts. It fixes both object and goal on the table,
    places them on opposite sides of a fixed barrier, disables native early termination, and exposes
    an independent physical certificate:

    ```text
    lifted ∧ crossed ∧ released ∧ retreated ∧ stable_12 ∧ collision_free
    ```

    A fixed obstacle keeps the challenge observable without changing the 22-dimensional state. A
    randomly moving obstacle would have to be added to the learner's observation.
    ''',
    r'''
    ## 1. 已提供的任务 adapter：simulator 相同，success 的含义更严格

    Panda Gym 原任务在方块第一次进入目标 5 cm 半径时立即结束。此时方块可能仍被闭合的夹爪抓在空中；
    并且原生目标约 70% 是悬空位置，因此它并不代表“已经放置完成”。

    `VerifiedPlacementEnv` 是对你已有 simulator 的轻量扩展，仍保持 observation `(22,)` 和 action
    `(4,)`。它把物体与目标固定在桌面两侧，加入固定障碍物，关闭原生提前结束，并使用独立物理证书：

    ```text
    已抬起 ∧ 已越障 ∧ 已释放 ∧ 已撤离 ∧ 连续稳定12步 ∧ 无碰撞
    ```

    障碍物必须固定，才能继续使用 22 维状态；如果障碍物随机移动，就必须把它的位置加入 observation。
    ''',
)

c(
    "native-counterexample",
    r'''
    counter_env = VerifiedPlacementEnv()
    obs, info = counter_env.reset(seed=2)
    goal = np.asarray(counter_env.raw_obs["desired_goal"])
    base = counter_env.unwrapped
    base.sim.set_base_pose("object", goal, np.array([0.0, 0.0, 0.0, 1.0]))
    object_id = base.sim._bodies_idx["object"]
    base.sim.physics_client.resetBaseVelocity(
        object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
    )
    _, _, terminated, _, info = counter_env.step(np.zeros(4, dtype=np.float32))
    print({key: info[key] for key in [
        "native_success", "strict_success", "ever_lifted",
        "ever_released", "retreated", "stable_steps"
    ]})
    assert info["native_success"] and not terminated and not info["strict_success"]
    counter_env.close()
    ''',
    "task-test",
)

m(
    "state-section",
    r'''
    ## 2. E01 — Parse simulator state into concepts

    Neural networks like vectors; experts need concepts. `RobotState` gives names to the slices used by
    geometry. Return copies so later simulator updates cannot silently mutate saved states.

    Main observation layout: `ee 0:3`, `finger width 6`, `object 7:10`, object linear velocity
    `13:16`, angular velocity `16:19`; the desired goal is a separate three-vector.
    ''',
    r'''
    ## 2. E01 — 把 simulator state 解析成有含义的概念

    神经网络适合接收向量，而 expert 需要有名字的几何概念。`RobotState` 为切片命名，并且必须返回
    copy，避免 simulator 后续更新悄悄修改已经保存的状态。

    主 observation：`ee 0:3`、`finger width 6`、`object 7:10`、物体线速度 `13:16`、
    角速度 `16:19`；desired goal 是独立的三维向量。
    ''',
)

ex(
    "e01-robot-state",
    r'''
    @dataclass(frozen=True)
    class RobotState:
        ee: np.ndarray
        gripper_width: float
        obj: np.ndarray
        obj_velocity: np.ndarray
        obj_angular_velocity: np.ndarray
        goal: np.ndarray

        @classmethod
        def from_raw_obs(cls, raw_obs: dict[str, np.ndarray]) -> "RobotState":
            """E01: validate and decode one Panda observation."""
            # TODO: extract all six fields, copy arrays, reject wrong/non-finite input.
            raise NotImplementedError("Complete E01")
    ''',
    r'''
    @dataclass(frozen=True)
    class RobotState:
        ee: np.ndarray
        gripper_width: float
        obj: np.ndarray
        obj_velocity: np.ndarray
        obj_angular_velocity: np.ndarray
        goal: np.ndarray

        @classmethod
        def from_raw_obs(cls, raw_obs: dict[str, np.ndarray]) -> "RobotState":
            """Validate and decode one Panda observation."""
            main = np.asarray(raw_obs["observation"], dtype=np.float32)
            goal = np.asarray(raw_obs["desired_goal"], dtype=np.float32)
            if main.shape != (19,) or goal.shape != (3,):
                raise ValueError(f"Expected main (19,) and goal (3,), got {main.shape}, {goal.shape}")
            if not np.isfinite(main).all() or not np.isfinite(goal).all():
                raise ValueError("Observation contains non-finite values")
            return cls(
                ee=main[0:3].copy(),
                gripper_width=float(main[6]),
                obj=main[7:10].copy(),
                obj_velocity=main[13:16].copy(),
                obj_angular_velocity=main[16:19].copy(),
                goal=goal.copy(),
            )
    ''',
)

c(
    "e01-tests",
    r'''
    test_env = VerifiedPlacementEnv()
    test_env.reset(seed=5)
    decoded = RobotState.from_raw_obs(test_env.raw_obs)
    assert decoded.ee.shape == decoded.obj.shape == decoded.goal.shape == (3,)
    original = test_env.raw_obs["observation"][0]
    decoded.ee[0] += 10
    assert test_env.raw_obs["observation"][0] == original, "E01 must return copies"
    test_env.close()
    print("E01 checks passed")
    ''',
    "public-test",
)

m(
    "controller-section",
    r'''
    ## 3. E02 — One bounded geometric controller

    Every movement phase uses the same proportional controller:

    $$a_{xyz}=\mathrm{clip}(k(p_{target}-p_{ee}),-1,1).$$

    A large error saturates at full speed; a small error produces a gentle correction. The fourth
    action is the gripper command: `+1` opens and `-1` closes.
    ''',
    r'''
    ## 3. E02 — 一个有界几何控制器

    所有移动 phase 都复用同一个比例控制器：

    $$a_{xyz}=\mathrm{clip}(k(p_{target}-p_{ee}),-1,1).$$

    误差大时动作饱和为全速，误差小时产生细小修正。第四维是夹爪：`+1` 打开，`-1` 闭合。
    ''',
)

ex(
    "e02-bounded-controller",
    r'''
    def bounded_move_action(
        ee: np.ndarray, target: np.ndarray, gripper: float, gain: float = 20.0
    ) -> np.ndarray:
        """E02: return [dx, dy, dz, gripper] as clipped float32."""
        # TODO
        raise NotImplementedError("Complete E02")
    ''',
    r'''
    def bounded_move_action(
        ee: np.ndarray, target: np.ndarray, gripper: float, gain: float = 20.0
    ) -> np.ndarray:
        """Return [dx, dy, dz, gripper] as clipped float32."""
        ee = np.asarray(ee, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if ee.shape != (3,) or target.shape != (3,):
            raise ValueError("ee and target must both have shape (3,)")
        xyz = np.clip(gain * (target - ee), -1.0, 1.0)
        return np.r_[xyz, np.clip(gripper, -1.0, 1.0)].astype(np.float32)
    ''',
)

c(
    "e02-tests",
    r'''
    a = bounded_move_action(np.zeros(3), np.array([1.0, -0.01, 0.02]), -2.0)
    np.testing.assert_allclose(a, [1.0, -0.2, 0.4, -1.0], atol=1e-6)
    assert a.dtype == np.float32 and a.shape == (4,)
    print("E02 checks passed")
    ''',
    "public-test",
)

m(
    "fsm-types-intro",
    r'''
    ## 4. Build a 13-phase expert from decisions and guards

    We separate four ideas:

    - `ExpertPhase`: named discrete modes;
    - `ExpertConfig`: tunable numbers, never hidden magic constants;
    - `ExpertMemory`: facts remembered across time;
    - `ControlDecision`: the current target and gripper command.

    The controller decision answers “what should I do now?” A transition guard separately answers
    “when may I move to the next phase?” This separation makes the expert testable.
    ''',
    r'''
    ## 4. 用 decision 与 guard 构建 13-phase expert

    我们把四类概念分开：`ExpertPhase` 是离散模式，`ExpertConfig` 保存可调参数，
    `ExpertMemory` 保存跨 timestep 的事实，`ControlDecision` 描述当前目标与夹爪命令。

    controller decision 回答“现在做什么”，transition guard 单独回答“什么时候可以进入下一阶段”。
    这种分离方式让 expert 更容易测试和定位错误。
    ''',
)

c(
    "fsm-types",
    r'''
    class ExpertPhase(str, Enum):
        APPROACH = "approach"
        DESCEND = "descend"
        CLOSE = "close"
        TEST_GRASP = "test_grasp"
        LIFT_CLEAR = "lift_clear"
        PRE_CROSS = "pre_cross"
        CROSS_BARRIER = "cross_barrier"
        ABOVE_GOAL = "above_goal"
        LOWER = "lower"
        OPEN = "open"
        RETREAT_UP = "retreat_up"
        RETREAT_HOME = "retreat_home"
        VERIFY = "verify"


    @dataclass(frozen=True)
    class ExpertConfig:
        hover_height: float = 0.12
        grasp_offset: float = 0.015
        test_lift: float = 0.045
        safe_z: float = 0.17
        pre_cross_y: float = -0.075
        post_cross_y: float = 0.075
        position_tolerance: float = 0.018
        place_tolerance: float = 0.035
        gain: float = 20.0
        close_steps: int = 10
        open_steps: int = 12
        test_timeout: int = 18
        home: tuple[float, float, float] = (0.04, 0.0, 0.20)


    @dataclass
    class ExpertMemory:
        phase: ExpertPhase = ExpertPhase.APPROACH
        phase_steps: int = 0
        retries: int = 0
        grasp_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.015]))
        grasp_target: np.ndarray | None = None
        release_target: np.ndarray | None = None


    @dataclass(frozen=True)
    class ControlDecision:
        target: np.ndarray
        gripper: float
    ''',
)

m(
    "decision-exercises-intro",
    r'''
    ### E03–E05 — Write the phase targets

    These three functions cover pick, transport, and finish. They contain no transition logic. A
    frozen absolute `grasp_target` is essential while closing: following the moving cube can push it
    diagonally instead of grasping it.
    ''',
    r'''
    ### E03–E05 — 编写每个 phase 的 target

    三个函数分别负责 pick、transport 和 finish，但不处理状态转移。闭合夹爪时必须使用冻结的绝对
    `grasp_target`；如果 target 持续跟随移动中的方块，夹爪可能把方块斜着推走。
    ''',
)

ex(
    "e03-pick-decisions",
    r'''
    def pick_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        """E03: APPROACH, DESCEND, CLOSE, TEST_GRASP."""
        # TODO
        raise NotImplementedError("Complete E03")
    ''',
    r'''
    def pick_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        if phase is ExpertPhase.APPROACH:
            return ControlDecision(state.obj + [0, 0, cfg.hover_height], +1.0)
        if phase is ExpertPhase.DESCEND:
            return ControlDecision(state.obj + [0, 0, cfg.grasp_offset], +1.0)
        if phase is ExpertPhase.CLOSE:
            if memory.grasp_target is None:
                raise RuntimeError("CLOSE requires a frozen grasp_target")
            return ControlDecision(memory.grasp_target, -1.0)
        if phase is ExpertPhase.TEST_GRASP:
            if memory.grasp_target is None:
                raise RuntimeError("TEST_GRASP requires grasp_target")
            return ControlDecision(memory.grasp_target + [0, 0, cfg.test_lift], -1.0)
        raise ValueError(f"Not a pick phase: {phase}")
    ''',
)

ex(
    "e04-transport-decisions",
    r'''
    def transport_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        """E04: LIFT_CLEAR, PRE_CROSS, CROSS_BARRIER, ABOVE_GOAL."""
        # TODO
        raise NotImplementedError("Complete E04")
    ''',
    r'''
    def transport_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        if phase is ExpertPhase.LIFT_CLEAR:
            target = np.array([state.ee[0], state.ee[1], cfg.safe_z])
        elif phase is ExpertPhase.PRE_CROSS:
            target = np.array([state.goal[0], cfg.pre_cross_y, cfg.safe_z])
        elif phase is ExpertPhase.CROSS_BARRIER:
            target = np.array([state.goal[0], cfg.post_cross_y, cfg.safe_z])
        elif phase is ExpertPhase.ABOVE_GOAL:
            target = np.array([state.goal[0], state.goal[1], cfg.safe_z])
        else:
            raise ValueError(f"Not a transport phase: {phase}")
        return ControlDecision(target, -1.0)
    ''',
)

ex(
    "e05-finish-decisions",
    r'''
    def finish_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        """E05: LOWER, OPEN, RETREAT_UP, RETREAT_HOME, VERIFY."""
        # TODO
        raise NotImplementedError("Complete E05")
    ''',
    r'''
    def finish_decision(
        phase: ExpertPhase, state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        if phase is ExpertPhase.LOWER:
            return ControlDecision(state.goal + memory.grasp_offset, -1.0)
        if phase is ExpertPhase.OPEN:
            target = memory.release_target
            if target is None:
                raise RuntimeError("OPEN requires release_target")
            return ControlDecision(target, +1.0)
        if phase is ExpertPhase.RETREAT_UP:
            return ControlDecision(np.array([state.ee[0], state.ee[1], cfg.safe_z]), +1.0)
        if phase in {ExpertPhase.RETREAT_HOME, ExpertPhase.VERIFY}:
            return ControlDecision(np.asarray(cfg.home), +1.0)
        raise ValueError(f"Not a finish phase: {phase}")
    ''',
)

c(
    "decision-dispatch",
    r'''
    PICK_PHASES = {
        ExpertPhase.APPROACH, ExpertPhase.DESCEND,
        ExpertPhase.CLOSE, ExpertPhase.TEST_GRASP,
    }
    TRANSPORT_PHASES = {
        ExpertPhase.LIFT_CLEAR, ExpertPhase.PRE_CROSS,
        ExpertPhase.CROSS_BARRIER, ExpertPhase.ABOVE_GOAL,
    }

    def controller_decision(
        state: RobotState, memory: ExpertMemory, cfg: ExpertConfig
    ) -> ControlDecision:
        if memory.phase in PICK_PHASES:
            return pick_decision(memory.phase, state, memory, cfg)
        if memory.phase in TRANSPORT_PHASES:
            return transport_decision(memory.phase, state, memory, cfg)
        return finish_decision(memory.phase, state, memory, cfg)

    def arrived(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(a - b) < tolerance)
    ''',
)

m(
    "transition-intro",
    r'''
    ### E06 — Transition guards and recovery

    A phase changes only after a measurable guard succeeds. `TEST_GRASP` is the recovery branch: if
    the cube does not rise with the hand before timeout, retry from `APPROACH` instead of blindly
    carrying an empty gripper.
    ''',
    r'''
    ### E06 — Transition guard 与恢复逻辑

    每次 phase 转移都必须由可测量条件触发。`TEST_GRASP` 是恢复分支：如果超时前方块没有跟随夹爪
    上升，就回到 `APPROACH` 重试，而不是拿着空夹爪继续搬运。
    ''',
)

ex(
    "e06-transition-guards",
    r'''
    def advance_phase(state: RobotState, memory: ExpertMemory, cfg: ExpertConfig) -> None:
        """E06: mutate memory only when the current phase guard is satisfied."""
        # TODO: implement all 13 transitions, including TEST_GRASP recovery.
        raise NotImplementedError("Complete E06")
    ''',
    r'''
    def advance_phase(state: RobotState, memory: ExpertMemory, cfg: ExpertConfig) -> None:
        p = memory.phase

        def switch(next_phase: ExpertPhase) -> None:
            memory.phase = next_phase
            memory.phase_steps = 0

        if p is ExpertPhase.APPROACH:
            target = state.obj + [0, 0, cfg.hover_height]
            if arrived(state.ee, target, cfg.position_tolerance):
                switch(ExpertPhase.DESCEND)
        elif p is ExpertPhase.DESCEND:
            target = state.obj + [0, 0, cfg.grasp_offset]
            if arrived(state.ee, target, cfg.position_tolerance):
                memory.grasp_offset = state.ee - state.obj
                memory.grasp_target = state.ee.copy()
                switch(ExpertPhase.CLOSE)
        elif p is ExpertPhase.CLOSE and memory.phase_steps >= cfg.close_steps:
            switch(ExpertPhase.TEST_GRASP)
        elif p is ExpertPhase.TEST_GRASP:
            if state.obj[2] > 0.05:
                switch(ExpertPhase.LIFT_CLEAR)
            elif memory.phase_steps >= cfg.test_timeout:
                memory.retries += 1
                memory.grasp_target = None
                switch(ExpertPhase.APPROACH)
        elif p is ExpertPhase.LIFT_CLEAR and abs(state.ee[2] - cfg.safe_z) < cfg.position_tolerance:
            switch(ExpertPhase.PRE_CROSS)
        elif p is ExpertPhase.PRE_CROSS:
            target = np.array([state.goal[0], cfg.pre_cross_y, cfg.safe_z])
            if arrived(state.ee, target, cfg.position_tolerance):
                switch(ExpertPhase.CROSS_BARRIER)
        elif p is ExpertPhase.CROSS_BARRIER:
            target = np.array([state.goal[0], cfg.post_cross_y, cfg.safe_z])
            if arrived(state.ee, target, cfg.position_tolerance):
                switch(ExpertPhase.ABOVE_GOAL)
        elif p is ExpertPhase.ABOVE_GOAL:
            target = np.array([state.goal[0], state.goal[1], cfg.safe_z])
            if arrived(state.ee, target, cfg.position_tolerance):
                switch(ExpertPhase.LOWER)
        elif p is ExpertPhase.LOWER and np.linalg.norm(state.obj - state.goal) < cfg.place_tolerance:
            memory.release_target = state.goal + memory.grasp_offset
            switch(ExpertPhase.OPEN)
        elif p is ExpertPhase.OPEN and memory.phase_steps >= cfg.open_steps:
            switch(ExpertPhase.RETREAT_UP)
        elif p is ExpertPhase.RETREAT_UP and abs(state.ee[2] - cfg.safe_z) < cfg.position_tolerance:
            switch(ExpertPhase.RETREAT_HOME)
        elif p is ExpertPhase.RETREAT_HOME and arrived(state.ee, np.asarray(cfg.home), 0.025):
            switch(ExpertPhase.VERIFY)
    ''',
)

m(
    "assembly-intro",
    r'''
    ### E07 — Assemble the expert

    `act` should be boring: decode state, advance a timestep counter, ask for one decision, convert it
    to an action, then evaluate transition guards. Small orchestration methods are easier to trust.
    ''',
    r'''
    ### E07 — 组装完整 expert

    `act` 应该很简单：解析 state、增加 phase 计数、获得一个 decision、转成 action、最后检查
    transition guard。越简单的 orchestration 越容易验证。
    ''',
)

ex(
    "e07-expert-class",
    r'''
    class BarrierPickPlaceExpert:
        def __init__(self, config: ExpertConfig | None = None) -> None:
            self.config = config or ExpertConfig()
            self.reset()

        def reset(self) -> None:
            self.memory = ExpertMemory()
            self.last_target: np.ndarray | None = None

        @property
        def phase(self) -> ExpertPhase:
            return self.memory.phase

        def act(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
            """E07: assemble parser, decision, controller, and transition."""
            # TODO
            raise NotImplementedError("Complete E07")
    ''',
    r'''
    class BarrierPickPlaceExpert:
        def __init__(self, config: ExpertConfig | None = None) -> None:
            self.config = config or ExpertConfig()
            self.reset()

        def reset(self) -> None:
            self.memory = ExpertMemory()
            self.last_target: np.ndarray | None = None

        @property
        def phase(self) -> ExpertPhase:
            return self.memory.phase

        def act(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
            state = RobotState.from_raw_obs(raw_obs)
            self.memory.phase_steps += 1
            decision = controller_decision(state, self.memory, self.config)
            self.last_target = np.asarray(decision.target).copy()
            action = bounded_move_action(
                state.ee, decision.target, decision.gripper, self.config.gain
            )
            advance_phase(state, self.memory, self.config)
            return action
    ''',
)

c(
    "expert-one-rollout-test",
    r'''
    expert_env = VerifiedPlacementEnv()
    expert = BarrierPickPlaceExpert()
    obs, _ = expert_env.reset(seed=7)
    expert.reset()
    phases, final_info = [], {}
    for _ in range(expert_env.config.max_episode_steps):
        phases.append(expert.phase.value)
        action = expert.act(expert_env.raw_obs)
        obs, _, terminated, truncated, final_info = expert_env.step(action)
        if terminated or truncated:
            break
    required_phases = {phase.value for phase in ExpertPhase}
    print({
        "strict_success": final_info["strict_success"],
        "steps": final_info["step"],
        "retries": expert.memory.retries,
        "phases_seen": len(set(phases)),
        "certificate": {k: final_info[k] for k in [
            "ever_lifted", "ever_crossed", "ever_released",
            "retreated", "stable_steps", "collision_free"
        ]},
    })
    assert final_info["strict_success"]
    assert required_phases.issubset(set(phases))
    expert_env.close()
    ''',
    "integration-test",
)

m(
    "collector-intro",
    r'''
    ## 5. E08 — Collect an aligned expert trace

    Row `t` must mean: “from observation `o_t`, the expert selected action `a_t` in phase `p_t`."
    Record all three **before** stepping. Store task information after the step as the outcome of that
    action. This alignment rule prevents a subtle one-step label shift.
    ''',
    r'''
    ## 5. E08 — 收集时间对齐的 expert trace

    第 `t` 行必须表示：“在 observation `o_t` 下，expert 在 phase `p_t` 选择 action `a_t`”。
    因此三者都要在 step 前记录，而 task info 是该 action 执行后的结果。这个规则可以避免标签整体错位一步。
    ''',
)

c(
    "trace-types",
    r'''
    @dataclass
    class ExpertTrace:
        observations: np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        phases: list[str]
        infos: list[dict[str, Any]]
        seed: int

        @property
        def success(self) -> bool:
            return bool(self.infos[-1]["strict_success"])

        def as_episode(self) -> Episode:
            return Episode(
                self.observations,
                self.actions,
                self.rewards,
                self.success,
                seed=self.seed,
                usable_for_training=self.success and self.infos[-1]["collision_free"],
                phases=self.phases,
            )
    ''',
)

ex(
    "e08-collector",
    r'''
    def collect_trace(env: VerifiedPlacementEnv, expert: BarrierPickPlaceExpert, seed: int) -> ExpertTrace:
        """E08: collect one complete strict task trace without an off-by-one error."""
        # TODO
        raise NotImplementedError("Complete E08")
    ''',
    r'''
    def collect_trace(env: VerifiedPlacementEnv, expert: BarrierPickPlaceExpert, seed: int) -> ExpertTrace:
        obs, _ = env.reset(seed=seed)
        expert.reset()
        observations, actions, rewards, phases, infos = [], [], [], [], []
        while True:
            observations.append(obs.copy())
            phases.append(expert.phase.value)
            action = expert.act(env.raw_obs)
            actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            infos.append(dict(info))
            if terminated or truncated:
                break
        return ExpertTrace(
            np.stack(observations).astype(np.float32),
            np.stack(actions).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            phases, infos, seed,
        )
    ''',
)

c(
    "expert-benchmark",
    r'''
    benchmark_env = VerifiedPlacementEnv()
    benchmark_traces = [
        collect_trace(benchmark_env, BarrierPickPlaceExpert(), seed)
        for seed in range(EXPERT_BENCHMARK_EPISODES)
    ]
    benchmark_env.close()
    benchmark_successes = sum(trace.success for trace in benchmark_traces)
    lengths = np.array([len(trace.actions) for trace in benchmark_traces])
    clearances = [trace.infos[-1]["min_crossing_clearance"] for trace in benchmark_traces]
    print({
        "strict_successes": f"{benchmark_successes}/{len(benchmark_traces)}",
        "steps_min_mean_max": [int(lengths.min()), float(lengths.mean()), int(lengths.max())],
        "minimum_crossing_clearance_m": float(min(clearances)),
    })

    phase_counts = Counter(p for trace in benchmark_traces for p in trace.phases)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(phase_counts.keys(), phase_counts.values(), color="#4C78A8")
    axes[0].set(title="Expert phase coverage", ylabel="timesteps")
    axes[0].tick_params(axis="x", rotation=50)
    axes[1].hist(lengths, bins=max(5, len(np.unique(lengths))), color="#2A9D8F")
    axes[1].set(title="Episode length", xlabel="steps", ylabel="episodes")
    plt.tight_layout()
    plt.show()
    ''',
    "benchmark",
)

m(
    "database-intro",
    r'''
    ## 6. E09 — Store strict evidence, not just a success bit

    The database keeps failed traces for diagnosis but the training query selects only
    `strict_success=1 AND collision_free=1`. `native_success_seen` is stored separately so nobody can
    accidentally substitute the easier benchmark metric later.
    ''',
    r'''
    ## 6. E09 — 保存严格证据，而不只是一个 success bit

    数据库保留失败 trace 用于诊断，但训练 query 只选择 `strict_success=1 AND collision_free=1`。
    `native_success_seen` 单独保存，避免未来误把更容易的原生指标当成真正完成。
    ''',
)

c(
    "database-class",
    r'''
    class VerifiedDemoStore:
        def __init__(self, path: Path) -> None:
            self.connection = sqlite3.connect(path)
            self.connection.executescript("""
                CREATE TABLE episodes (
                    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seed INTEGER NOT NULL,
                    strict_success INTEGER NOT NULL,
                    native_success_seen INTEGER NOT NULL,
                    collision_free INTEGER NOT NULL,
                    stable_steps INTEGER NOT NULL,
                    n_steps INTEGER NOT NULL
                );
                CREATE TABLE steps (
                    episode_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    observation TEXT NOT NULL,
                    action TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    PRIMARY KEY (episode_id, step),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
            """)

        def save(self, trace: ExpertTrace) -> int:
            final = trace.infos[-1]
            with self.connection:
                cursor = self.connection.execute("""
                    INSERT INTO episodes(
                        seed, strict_success, native_success_seen,
                        collision_free, stable_steps, n_steps
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trace.seed, int(trace.success), int(final["native_success_seen"]),
                    int(final["collision_free"]), int(final["stable_steps"]), len(trace.actions)
                ))
                episode_id = int(cursor.lastrowid)
                self.connection.executemany("""
                    INSERT INTO steps(episode_id, step, observation, action, phase)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (
                        episode_id, i,
                        json.dumps(obs.tolist(), separators=(",", ":")),
                        json.dumps(action.tolist(), separators=(",", ":")),
                        phase,
                    )
                    for i, (obs, action, phase) in enumerate(zip(
                        trace.observations, trace.actions, trace.phases
                    ))
                ])
            return episode_id

        def close(self) -> None:
            self.connection.close()
    ''',
)

ex(
    "e09-verified-query",
    r'''
    def load_verified_training_data(
        connection: sqlite3.Connection, max_episodes: int
    ) -> TrainingData:
        """E09: load steps from at most N strict, collision-free episodes."""
        # TODO: use a CTE to select episodes, then join ordered steps.
        raise NotImplementedError("Complete E09")
    ''',
    r'''
    def load_verified_training_data(
        connection: sqlite3.Connection, max_episodes: int
    ) -> TrainingData:
        rows = connection.execute("""
            WITH selected AS (
                SELECT episode_id FROM episodes
                WHERE strict_success = 1 AND collision_free = 1
                ORDER BY episode_id LIMIT ?
            )
            SELECT s.episode_id, s.observation, s.action, s.phase
            FROM steps s JOIN selected x USING(episode_id)
            ORDER BY s.episode_id, s.step
        """, (max_episodes,)).fetchall()
        if not rows:
            raise ValueError("No verified demonstrations")
        return TrainingData(
            observations=np.asarray([json.loads(row[1]) for row in rows], dtype=np.float32),
            actions=np.asarray([json.loads(row[2]) for row in rows], dtype=np.float32),
            episode_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
            phases=np.asarray([row[3] for row in rows], dtype=str),
        )
    ''',
)

c(
    "collect-database",
    r'''
    store = VerifiedDemoStore(tutorial_dir / "verified_demos.sqlite")
    data_env = VerifiedPlacementEnv()
    for seed in range(TARGET_DEMOS):
        trace = collect_trace(data_env, BarrierPickPlaceExpert(), seed)
        store.save(trace)
    data_env.close()
    data = load_verified_training_data(store.connection, TARGET_DEMOS)
    episode_rows = store.connection.execute("""
        SELECT COUNT(*), SUM(strict_success), SUM(native_success_seen), SUM(collision_free)
        FROM episodes
    """).fetchone()
    print({
        "episodes / strict / native-hit / collision-free": episode_rows,
        "observations": data.observations.shape,
        "actions": data.actions.shape,
    })
    ''',
    "data-collection",
)

c(
    "data-dashboard",
    r'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    counts = Counter(data.phases)
    axes[0].bar(counts.keys(), counts.values(), color="#4C78A8")
    axes[0].set(title="Training samples by phase", ylabel="samples")
    axes[0].tick_params(axis="x", rotation=50)
    for j, name in enumerate(["dx", "dy", "dz", "gripper"]):
        axes[1].hist(data.actions[:, j], bins=25, alpha=0.5, label=name)
    axes[1].set(title="Expert action distribution", xlabel="action")
    axes[1].legend()
    axes[2].scatter(
        data.observations[:, 7], data.observations[:, 8],
        c=data.observations[:, 9], s=6, alpha=0.35, cmap="viridis"
    )
    axes[2].axhspan(-0.018, 0.018, color="#2455C3", alpha=0.25, label="barrier")
    axes[2].set(title="Object-state coverage", xlabel="object x", ylabel="object y")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    ''',
    "data-visualization",
)

m(
    "learning-intro",
    r'''
    ## 7. E10–E11 — Learning mechanism: preserve episodes and gripper modes

    The student already knows MLPs, so our focus is the interface. Adjacent timesteps from one
    trajectory must never be split across train and validation. For the output, ordinary four-channel
    MSE can average `open=+1` and `close=-1` toward a useless zero. We use a hybrid head:

    - xyz motion: bounded regression with Smooth L1;
    - gripper: binary open/close classification with BCE.
    ''',
    r'''
    ## 7. E10–E11 — 学习机制：保留 episode 边界与夹爪模式

    学生已经理解 MLP，因此这里关注数据接口。同一 trajectory 的相邻 timestep 不能同时进入 train 和
    validation。对于输出，普通四维 MSE 可能把 `open=+1` 与 `close=-1` 平均成无用的 0。我们改用：

    - xyz：有界 regression，使用 Smooth L1；
    - gripper：open/close 二分类，使用 BCE。
    ''',
)

ex(
    "e10-episode-split",
    r'''
    def split_episode_ids(
        episode_ids: np.ndarray, validation_fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """E10: return train_mask, val_mask, train_ids, val_ids."""
        # TODO: shuffle unique IDs, never individual timesteps.
        raise NotImplementedError("Complete E10")
    ''',
    r'''
    def split_episode_ids(
        episode_ids: np.ndarray, validation_fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        unique = np.unique(episode_ids)
        shuffled = np.random.default_rng(seed).permutation(unique)
        n_val = max(1, round(len(unique) * validation_fraction)) if len(unique) > 1 else 0
        n_val = min(n_val, max(0, len(unique) - 1))
        val_ids = np.sort(shuffled[:n_val])
        train_ids = np.sort(shuffled[n_val:])
        train_mask = np.isin(episode_ids, train_ids)
        val_mask = np.isin(episode_ids, val_ids)
        assert set(train_ids).isdisjoint(val_ids)
        return train_mask, val_mask, train_ids, val_ids
    ''',
)

ex(
    "e11-hybrid-loss",
    r'''
    def hybrid_bc_loss(
        motion_prediction: torch.Tensor,
        gripper_logit: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        """E11: SmoothL1(xyz) + 0.35 * BCE(open/close)."""
        # TODO
        raise NotImplementedError("Complete E11")
    ''',
    r'''
    def hybrid_bc_loss(
        motion_prediction: torch.Tensor,
        gripper_logit: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        motion = F.smooth_l1_loss(motion_prediction, expert_actions[:, :3])
        open_target = (expert_actions[:, 3] > 0).float().unsqueeze(1)
        gripper = F.binary_cross_entropy_with_logits(gripper_logit, open_target)
        return motion + 0.35 * gripper
    ''',
)

c(
    "hybrid-model-training",
    r'''
    class HybridPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(22, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            self.motion_head = nn.Sequential(nn.Linear(128, 3), nn.Tanh())
            self.gripper_head = nn.Linear(128, 1)
            self.register_buffer("obs_mean", torch.zeros(22))
            self.register_buffer("obs_std", torch.ones(22))

        def set_normalization(self, observations: np.ndarray) -> None:
            x = torch.as_tensor(observations, dtype=torch.float32)
            self.obs_mean.copy_(x.mean(0))
            self.obs_std.copy_(x.std(0, unbiased=False).clamp_min(1e-6))

        def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.trunk((observations - self.obs_mean) / self.obs_std)
            return self.motion_head(hidden), self.gripper_head(hidden)


    class HybridNumpyPolicy:
        def __init__(self, model: HybridPolicy) -> None:
            self.model = model.eval()

        def __call__(self, observation: np.ndarray) -> np.ndarray:
            x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                motion, logit = self.model(x)
            grip = torch.where(logit >= 0, 1.0, -1.0)
            return torch.cat([motion, grip], dim=1).squeeze(0).numpy().astype(np.float32)


    @dataclass
    class HybridTrainResult:
        model: HybridPolicy
        train_loss: list[float]
        validation_loss: list[float]
        train_ids: np.ndarray
        validation_ids: np.ndarray


    def train_hybrid(data: TrainingData, epochs: int, seed: int = 0) -> HybridTrainResult:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train_mask, val_mask, train_ids, val_ids = split_episode_ids(
            data.episode_ids, 0.2, seed
        )
        model = HybridPolicy()
        model.set_normalization(data.observations[train_mask])
        train_set = TensorDataset(
            torch.as_tensor(data.observations[train_mask], dtype=torch.float32),
            torch.as_tensor(data.actions[train_mask], dtype=torch.float32),
        )
        loader = DataLoader(
            train_set, batch_size=min(256, len(train_set)), shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )
        x_val = torch.as_tensor(data.observations[val_mask], dtype=torch.float32)
        y_val = torch.as_tensor(data.actions[val_mask], dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_history, val_history = [], []
        best_state, best_val = deepcopy(model.state_dict()), float("inf")
        for _ in range(epochs):
            model.train(); epoch_sum = 0.0; n = 0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                motion, logit = model(x_batch)
                loss = hybrid_bc_loss(motion, logit, y_batch)
                loss.backward(); optimizer.step()
                epoch_sum += float(loss.detach()) * len(x_batch); n += len(x_batch)
            train_history.append(epoch_sum / n)
            model.eval()
            with torch.inference_mode():
                motion, logit = model(x_val)
                val = float(hybrid_bc_loss(motion, logit, y_val))
            val_history.append(val)
            if val < best_val:
                best_val, best_state = val, deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return HybridTrainResult(model.eval(), train_history, val_history, train_ids, val_ids)


    trained = train_hybrid(data, TRAIN_EPOCHS, seed=7)
    learned_policy = HybridNumpyPolicy(trained.model)
    assert set(trained.train_ids).isdisjoint(trained.validation_ids)
    plt.figure(figsize=(7, 3.5))
    plt.plot(trained.train_loss, label="train")
    plt.plot(trained.validation_loss, label="validation")
    plt.xlabel("epoch"); plt.ylabel("hybrid loss"); plt.title("Behavior cloning")
    plt.legend(); plt.show()
    ''',
    "learning",
)

m(
    "evaluation-intro",
    r'''
    ## 8. E12–E13 — Strict closed-loop evaluation and failure diagnosis

    Evaluation uses fresh seeds beginning at 10,000. Report native hit rate and strict completion
    separately. A learner that reaches the target while still holding the cube may score well under
    the old metric and zero under the real task.
    ''',
    r'''
    ## 8. E12–E13 — 严格闭环评估与失败诊断

    评估使用从 10,000 开始的新 seed，并分别报告 native hit rate 与 strict completion。
    如果 learner 抓着方块到达目标但没有松手，旧指标可能很高，而真实任务得分应为零。
    ''',
)

ex(
    "e12-strict-evaluator",
    r'''
    @dataclass
    class StrictEvaluation:
        strict_successes: np.ndarray
        native_hits: np.ndarray
        final_infos: list[dict[str, Any]]
        seeds: np.ndarray

    def evaluate_strict(policy: Any, n_episodes: int, seed_start: int = 10_000) -> StrictEvaluation:
        """E12: run a flat policy until strict termination or timeout on fresh seeds."""
        # TODO
        raise NotImplementedError("Complete E12")
    ''',
    r'''
    @dataclass
    class StrictEvaluation:
        strict_successes: np.ndarray
        native_hits: np.ndarray
        final_infos: list[dict[str, Any]]
        seeds: np.ndarray

    def evaluate_strict(policy: Any, n_episodes: int, seed_start: int = 10_000) -> StrictEvaluation:
        env = VerifiedPlacementEnv()
        seeds = np.arange(seed_start, seed_start + n_episodes)
        strict, native, infos = [], [], []
        try:
            for seed in seeds:
                obs, _ = env.reset(seed=int(seed))
                while True:
                    action = np.asarray(policy(obs), dtype=np.float32)
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                strict.append(bool(info["strict_success"]))
                native.append(bool(info["native_success_seen"]))
                infos.append(dict(info))
        finally:
            env.close()
        return StrictEvaluation(
            np.asarray(strict, dtype=bool), np.asarray(native, dtype=bool), infos, seeds
        )
    ''',
)

ex(
    "e13-failure-classifier",
    r'''
    def classify_failure(info: dict[str, Any]) -> str:
        """E13: classify the earliest missing lifecycle milestone."""
        # TODO: barrier_collision, grasp_failed, dropped_before_barrier,
        # not_released, no_retreat, unstable_placement, or success.
        raise NotImplementedError("Complete E13")
    ''',
    r'''
    def classify_failure(info: dict[str, Any]) -> str:
        if info["strict_success"]:
            return "success"
        if info["barrier_violation"]:
            return "barrier_collision"
        if not info["ever_lifted"]:
            return "grasp_failed"
        if not info["ever_crossed"]:
            return "dropped_before_barrier"
        if not info["ever_released"]:
            return "not_released"
        if not info["retreated"]:
            return "no_retreat"
        return "unstable_placement"
    ''',
)

c(
    "evaluation-run",
    r'''
    evaluation = evaluate_strict(learned_policy, EVAL_EPISODES)
    strict_rate = float(evaluation.strict_successes.mean())
    native_rate = float(evaluation.native_hits.mean())

    def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
        p = successes / trials
        d = 1 + z*z/trials
        center = (p + z*z/(2*trials)) / d
        margin = z * sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / d
        return center - margin, center + margin

    strict_ci = wilson(int(evaluation.strict_successes.sum()), EVAL_EPISODES)
    failures = Counter(classify_failure(info) for info in evaluation.final_infos)
    print({
        "native_hits": f"{evaluation.native_hits.sum()}/{EVAL_EPISODES}",
        "strict_successes": f"{evaluation.strict_successes.sum()}/{EVAL_EPISODES}",
        "strict_ci95": strict_ci,
        "failure_types": dict(failures),
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(["native hit", "strict complete"], [native_rate, strict_rate],
                color=["#F2B84B", "#35C48D"])
    axes[0].set(ylim=(0, 1), ylabel="rate", title="Fresh-seed closed loop")
    axes[1].bar(failures.keys(), failures.values(), color="#E45756")
    axes[1].set(title="Failure diagnosis", ylabel="episodes")
    axes[1].tick_params(axis="x", rotation=35)
    plt.tight_layout(); plt.show()
    ''',
    "evaluation",
)

m(
    "dagger",
    r'''
    ## 9. DAgger: change the visited states, not merely the network

    Behavior cloning trains on expert-visited states. The learner creates different states after its
    first small error. For one failed seed:

    1. roll out the learner and save every learner-visited observation;
    2. restore or replay each state, then ask the expert for the correct action;
    3. append **expert labels**, never the learner's incorrect action;
    4. retrain and reevaluate on the unchanged fresh-seed suite.

    Use `classify_failure` to choose which state distribution to collect first. If most failures are
    `not_released`, prioritize states near the goal rather than simply collecting more approaches.
    ''',
    r'''
    ## 9. DAgger：改变 learner 访问的状态，而不只是换网络

    Behavior cloning 只在 expert 访问过的 state 上训练；learner 第一次产生小误差后会进入不同状态。
    对一条失败 seed：保存 learner 访问的 observation，恢复或 replay 状态，让 expert 给出正确 action，
    只追加 expert label，然后在不变的新 seed 集上重新评估。

    用 `classify_failure` 决定优先收集哪类状态。如果多数失败是 `not_released`，应优先补充目标附近的
    状态，而不是继续收集更多 approach 数据。
    ''',
)

m(
    "experiments",
    r'''
    ## 10. Research extensions

    Run each comparison with at least three training seeds and the same evaluation seeds:

    - demonstrations: `[10, 25, 50, 100, 200]`;
    - four-channel MSE versus hybrid regression/classification;
    - phase balancing on/off;
    - with/without `TEST_GRASP` recovery;
    - barrier height curriculum: 4 cm, 6 cm, 8 cm.

    Report expert completion, collision rate, minimum crossing clearance, native hit, strict success,
    Wilson intervals, and categorized failures. Never report an illustrative number as a result.
    ''',
    r'''
    ## 10. 研究扩展

    每个比较至少使用三个 training seed，并保持相同 evaluation seeds：demo 数量
    `[10,25,50,100,200]`、四维 MSE 与 hybrid loss、phase balancing、是否使用 `TEST_GRASP`
    恢复，以及 4/6/8 cm 障碍高度 curriculum。

    报告 expert 完成率、碰撞率、最小越障净空、native hit、strict success、Wilson 区间与失败分类。
    不要把 illustrative 数字当作实验结果。
    ''',
)

m(
    "file-map",
    r'''
    ## 11. Where every piece belongs in the project

    | Notebook concept | Production file |
    |---|---|
    | strict lifecycle and barrier scene | `src/ilgrasp/sim/verified_task.py` |
    | reference 13-phase expert | `src/ilgrasp/experts/verified.py` |
    | high-fidelity two-camera recorder | `src/ilgrasp/visualization/recording.py` |
    | original observation contract | `src/ilgrasp/sim/observation.py` |
    | episode container/rollout | `src/ilgrasp/sim/rollout.py` |
    | reusable database layer | `src/ilgrasp/data/database.py` |
    | baseline MLP and training | `src/ilgrasp/policies/mlp.py`, `training.py` |
    | strict-task tests | `tests/test_verified_task.py` |
    | animation CLI | `scripts/record_verified_animation.py` |

    The environment monitor never reads the expert's phase. The expert never decides whether the task
    succeeded. Keeping those responsibilities independent prevents the teacher from grading itself.
    ''',
    r'''
    ## 11. 每一个部分在项目中的位置

    | Notebook 概念 | Production file |
    |---|---|
    | 严格 lifecycle 与障碍场景 | `src/ilgrasp/sim/verified_task.py` |
    | 13-phase reference expert | `src/ilgrasp/experts/verified.py` |
    | 双镜头高质量动画 | `src/ilgrasp/visualization/recording.py` |
    | 原 observation contract | `src/ilgrasp/sim/observation.py` |
    | episode 与 rollout | `src/ilgrasp/sim/rollout.py` |
    | database | `src/ilgrasp/data/database.py` |
    | baseline MLP 与训练 | `src/ilgrasp/policies/mlp.py`, `training.py` |
    | 严格任务测试 | `tests/test_verified_task.py` |
    | 动画命令 | `scripts/record_verified_animation.py` |

    环境 monitor 绝不读取 expert phase，expert 也不能自己决定是否成功。让两者独立，才能避免“老师给自己打分”。
    ''',
)

c(
    "cleanup",
    r'''
    store.close()
    print("Closed the SQLite connection. Temporary artifacts:", tutorial_dir)
    ''',
)

m(
    "exit-ticket",
    r'''
    ## Exit ticket

    Explain why these are three different claims:

    1. “the cube entered the native 5 cm goal radius”;
    2. “the expert completed a collision-free release, retreat, and stability certificate”;
    3. “a learned policy repeats that strict task on fresh seeds.”

    A complete robotics pipeline defines and measures claim 3. It does not rename claim 1 as success.
    ''',
    r'''
    ## 离场问题

    请解释为什么下面是三个不同命题：方块进入原生 5 cm 目标半径；expert 无碰撞地完成释放、撤离与
    稳定证书；learned policy 在新 seed 上重复完成严格任务。

    完整机器人 pipeline 应定义并测量第三个命题，而不是把第一个命题重新命名为 success。
    ''',
)


def build(language: str, solutions: bool) -> dict:
    rendered = [cell.render(language, solutions) for cell in CELLS]
    edition = {
        ("en", False): "Student edition",
        ("en", True): "Solutions edition",
        ("zh", False): "学生版",
        ("zh", True): "中文答案版",
    }[(language, solutions)]
    rendered[0]["source"].insert(1, f"\n**{edition}**\n")
    return {
        "cells": rendered,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ilgrasp)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_structure(notebooks: dict[str, dict]) -> None:
    reference = [
        (cell["id"], cell["cell_type"], tuple(cell.get("metadata", {}).get("tags", [])))
        for cell in next(iter(notebooks.values()))["cells"]
    ]
    for name, notebook in notebooks.items():
        structure = [
            (cell["id"], cell["cell_type"], tuple(cell.get("metadata", {}).get("tags", [])))
            for cell in notebook["cells"]
        ]
        if structure != reference:
            raise AssertionError(f"Notebook structure drifted: {name}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec")
    for name, notebook in notebooks.items():
        if "solutions" in name:
            source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            for forbidden in ("TODO", "NotImplementedError"):
                if forbidden in source:
                    raise AssertionError(f"{forbidden} remains in {name}")


def main() -> None:
    targets = {
        "04_verified_expert_pipeline_student_en.ipynb": build("en", False),
        "04_verified_expert_pipeline_solutions_en.ipynb": build("en", True),
        "04_verified_expert_pipeline_student_zh.ipynb": build("zh", False),
        "04_verified_expert_pipeline_solutions_zh.ipynb": build("zh", True),
    }
    validate_structure(targets)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, notebook in targets.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
