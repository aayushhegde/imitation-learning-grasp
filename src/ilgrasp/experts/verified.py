"""Robust expert for the verified obstacle pick-and-place task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ilgrasp.sim.rollout import Episode


class VerifiedExpertPhase(str, Enum):
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
class VerifiedExpertConfig:
    hover_height: float = 0.12
    grasp_offset: float = 0.015
    test_lift: float = 0.045
    safe_z: float = 0.17
    pre_cross_y: float = -0.075
    post_cross_y: float = 0.075
    position_tolerance: float = 0.018
    place_tolerance: float = 0.035
    action_gain: float = 20.0
    close_steps: int = 10
    open_steps: int = 12
    test_timeout: int = 18
    open_action: float = 1.0
    close_action: float = -1.0
    home: tuple[float, float, float] = (0.04, 0.0, 0.20)


class VerifiedPickPlaceExpert:
    """Thirteen-phase teacher that releases and verifies the placed cube."""

    def __init__(self, config: VerifiedExpertConfig | None = None) -> None:
        self.config = config or VerifiedExpertConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = VerifiedExpertPhase.APPROACH
        self.phase_steps = 0
        self.grasp_offset = np.array([0.0, 0.0, self.config.grasp_offset])
        self.grasp_target: np.ndarray | None = None
        self.release_target: np.ndarray | None = None

    @staticmethod
    def positions(
        raw_obs: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        main = np.asarray(raw_obs["observation"])
        return main[0:3], main[7:10], np.asarray(raw_obs["desired_goal"])

    def _arrived(self, position: np.ndarray, target: np.ndarray, tolerance: float | None = None) -> bool:
        tol = self.config.position_tolerance if tolerance is None else tolerance
        return bool(np.linalg.norm(position - target) < tol)

    def _move(self, ee: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        xyz = np.clip(self.config.action_gain * (target - ee), -1.0, 1.0)
        return np.r_[xyz, np.clip(gripper, -1.0, 1.0)].astype(np.float32)

    def act(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        ee, obj, goal = self.positions(raw_obs)
        cfg = self.config
        self.phase_steps += 1

        if self.phase is VerifiedExpertPhase.APPROACH:
            target = obj + np.array([0.0, 0.0, cfg.hover_height])
            action = self._move(ee, target, cfg.open_action)
            if self._arrived(ee, target):
                self.phase, self.phase_steps = VerifiedExpertPhase.DESCEND, 0

        elif self.phase is VerifiedExpertPhase.DESCEND:
            target = obj + np.array([0.0, 0.0, cfg.grasp_offset])
            action = self._move(ee, target, cfg.open_action)
            if self._arrived(ee, target):
                self.grasp_offset = ee - obj
                # Freeze the absolute hand pose while closing.  Recomputing it
                # from a moving cube can make the controller chase/push the cube.
                self.grasp_target = ee.copy()
                self.phase, self.phase_steps = VerifiedExpertPhase.CLOSE, 0

        elif self.phase is VerifiedExpertPhase.CLOSE:
            if self.grasp_target is None:
                raise RuntimeError("CLOSE entered without a frozen grasp target")
            action = self._move(ee, self.grasp_target, cfg.close_action)
            if self.phase_steps >= cfg.close_steps:
                self.phase, self.phase_steps = VerifiedExpertPhase.TEST_GRASP, 0

        elif self.phase is VerifiedExpertPhase.TEST_GRASP:
            if self.grasp_target is None:
                raise RuntimeError("TEST_GRASP entered without a grasp target")
            target = self.grasp_target + np.array([0.0, 0.0, cfg.test_lift])
            action = self._move(ee, target, cfg.close_action)
            if obj[2] > 0.05:
                self.phase, self.phase_steps = VerifiedExpertPhase.LIFT_CLEAR, 0
            elif self.phase_steps >= cfg.test_timeout:
                self.phase, self.phase_steps = VerifiedExpertPhase.APPROACH, 0

        elif self.phase is VerifiedExpertPhase.LIFT_CLEAR:
            target = np.array([ee[0], ee[1], cfg.safe_z])
            action = self._move(ee, target, cfg.close_action)
            if abs(ee[2] - cfg.safe_z) < cfg.position_tolerance:
                self.phase, self.phase_steps = VerifiedExpertPhase.PRE_CROSS, 0

        elif self.phase is VerifiedExpertPhase.PRE_CROSS:
            target = np.array([goal[0], cfg.pre_cross_y, cfg.safe_z])
            action = self._move(ee, target, cfg.close_action)
            if self._arrived(ee, target):
                self.phase, self.phase_steps = VerifiedExpertPhase.CROSS_BARRIER, 0

        elif self.phase is VerifiedExpertPhase.CROSS_BARRIER:
            target = np.array([goal[0], cfg.post_cross_y, cfg.safe_z])
            action = self._move(ee, target, cfg.close_action)
            if self._arrived(ee, target):
                self.phase, self.phase_steps = VerifiedExpertPhase.ABOVE_GOAL, 0

        elif self.phase is VerifiedExpertPhase.ABOVE_GOAL:
            target = np.array([goal[0], goal[1], cfg.safe_z])
            action = self._move(ee, target, cfg.close_action)
            if self._arrived(ee, target):
                self.phase, self.phase_steps = VerifiedExpertPhase.LOWER, 0

        elif self.phase is VerifiedExpertPhase.LOWER:
            target = goal + self.grasp_offset
            action = self._move(ee, target, cfg.close_action)
            if np.linalg.norm(obj - goal) < cfg.place_tolerance:
                self.release_target = target.copy()
                self.phase, self.phase_steps = VerifiedExpertPhase.OPEN, 0

        elif self.phase is VerifiedExpertPhase.OPEN:
            target = self.release_target if self.release_target is not None else goal + self.grasp_offset
            action = self._move(ee, target, cfg.open_action)
            if self.phase_steps >= cfg.open_steps:
                self.phase, self.phase_steps = VerifiedExpertPhase.RETREAT_UP, 0

        elif self.phase is VerifiedExpertPhase.RETREAT_UP:
            target = np.array([ee[0], ee[1], cfg.safe_z])
            action = self._move(ee, target, cfg.open_action)
            if abs(ee[2] - cfg.safe_z) < cfg.position_tolerance:
                self.phase, self.phase_steps = VerifiedExpertPhase.RETREAT_HOME, 0

        elif self.phase is VerifiedExpertPhase.RETREAT_HOME:
            target = np.asarray(cfg.home)
            action = self._move(ee, target, cfg.open_action)
            if self._arrived(ee, target, tolerance=0.025):
                self.phase, self.phase_steps = VerifiedExpertPhase.VERIFY, 0

        else:  # VERIFY: stay away while the task monitor counts stable steps.
            action = self._move(ee, np.asarray(cfg.home), cfg.open_action)

        return action


def collect_verified_episode(
    env: Any,
    expert: VerifiedPickPlaceExpert,
    seed: int | None = None,
) -> Episode:
    """Collect one strict episode and retain all thirteen phase labels."""

    obs, _ = env.reset(seed=seed)
    expert.reset()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    phases: list[str] = []
    info: dict[str, Any] = {}

    while True:
        if env.raw_obs is None:
            raise RuntimeError("raw_obs is unavailable after reset")
        phases.append(expert.phase.value)
        action = expert.act(env.raw_obs)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    expected_phases = {phase.value for phase in VerifiedExpertPhase}
    covers_full_lifecycle = expected_phases.issubset(set(phases))
    success = bool(info.get("strict_success", False))
    return Episode(
        observations=np.stack(observations),
        actions=np.stack(actions),
        rewards=np.asarray(rewards),
        success=success,
        seed=seed,
        usable_for_training=success and covers_full_lifecycle,
        phases=phases,
    )


__all__ = [
    "VerifiedExpertConfig",
    "VerifiedExpertPhase",
    "VerifiedPickPlaceExpert",
    "collect_verified_episode",
]
