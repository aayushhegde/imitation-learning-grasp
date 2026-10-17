"""A transparent finite-state expert for Panda pick-and-place.

The controller intentionally uses privileged simulator state (true object and
goal positions). That is acceptable because it is a teacher used to label data;
the learned policy receives only the standard 22-value observation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ilgrasp.sim.rollout import Episode


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
    """Distances are metres; actions are normalized to panda-gym's [-1, 1]."""

    hover_height: float = 0.10
    transport_clearance: float = 0.12
    grasp_offset: float = 0.015
    position_tolerance: float = 0.018
    horizontal_tolerance: float = 0.020
    action_gain: float = 20.0
    grasp_steps: int = 10
    hold_steps: int = 5
    open_action: float = 1.0
    close_action: float = -1.0


class ScriptedExpert:
    """State machine: approach → descend → grasp → lift → transport → place."""

    def __init__(self, config: ExpertConfig | None = None) -> None:
        self.config = config or ExpertConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = ExpertPhase.APPROACH
        self._phase_steps = 0
        self._grasp_offset = np.array([0.0, 0.0, self.config.grasp_offset])
        self._safe_z: float | None = None

    @staticmethod
    def positions(raw_obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        main = np.asarray(raw_obs["observation"])
        if main.shape[0] < 10:
            raise ValueError("Expected panda-gym task observation with at least 10 values")
        ee = main[0:3]
        obj = main[7:10]
        goal = np.asarray(raw_obs["desired_goal"])
        return ee, obj, goal

    def _arrived(self, ee: np.ndarray, target: np.ndarray, tolerance: float | None = None) -> bool:
        tol = self.config.position_tolerance if tolerance is None else tolerance
        return bool(np.linalg.norm(ee - target) < tol)

    def _move_action(self, ee: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        move = np.clip(self.config.action_gain * (target - ee), -1.0, 1.0)
        return np.r_[move, np.clip(gripper, -1.0, 1.0)].astype(np.float32)

    def act(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return one action and advance the state machine when targets are met."""

        ee, obj, goal = self.positions(raw_obs)
        cfg = self.config
        self._phase_steps += 1

        if self.phase is ExpertPhase.APPROACH:
            target = obj + np.array([0.0, 0.0, cfg.hover_height])
            action = self._move_action(ee, target, cfg.open_action)
            xy_close = np.linalg.norm(ee[:2] - obj[:2]) < cfg.horizontal_tolerance
            if xy_close and abs(ee[2] - target[2]) < cfg.position_tolerance:
                self.phase, self._phase_steps = ExpertPhase.DESCEND, 0

        elif self.phase is ExpertPhase.DESCEND:
            target = obj + np.array([0.0, 0.0, cfg.grasp_offset])
            action = self._move_action(ee, target, cfg.open_action)
            if self._arrived(ee, target):
                self._grasp_offset = ee - obj
                self.phase, self._phase_steps = ExpertPhase.GRASP, 0

        elif self.phase is ExpertPhase.GRASP:
            target = obj + self._grasp_offset
            action = self._move_action(ee, target, cfg.close_action)
            if self._phase_steps >= cfg.grasp_steps:
                self._safe_z = max(float(obj[2]), float(goal[2])) + cfg.transport_clearance
                self.phase, self._phase_steps = ExpertPhase.LIFT, 0

        elif self.phase is ExpertPhase.LIFT:
            if self._safe_z is None:
                raise RuntimeError("Expert entered LIFT without a safe transport height")
            target = np.array([ee[0], ee[1], self._safe_z])
            action = self._move_action(ee, target, cfg.close_action)
            if abs(ee[2] - self._safe_z) < cfg.position_tolerance:
                self.phase, self._phase_steps = ExpertPhase.TRANSPORT, 0

        elif self.phase is ExpertPhase.TRANSPORT:
            target = goal + self._grasp_offset
            target[2] = self._safe_z if self._safe_z is not None else ee[2]
            action = self._move_action(ee, target, cfg.close_action)
            if np.linalg.norm(obj[:2] - goal[:2]) < cfg.horizontal_tolerance:
                self.phase, self._phase_steps = ExpertPhase.PLACE, 0

        elif self.phase is ExpertPhase.PLACE:
            target = goal + self._grasp_offset
            action = self._move_action(ee, target, cfg.close_action)
            if np.linalg.norm(obj - goal) < cfg.position_tolerance:
                self.phase, self._phase_steps = ExpertPhase.HOLD, 0

        else:  # HOLD: keep the object at the target until success is registered.
            target = goal + self._grasp_offset
            action = self._move_action(ee, target, cfg.close_action)

        return action


def collect_expert_episode(env: Any, expert: ScriptedExpert, seed: int | None = None) -> Episode:
    """Run the expert and retain the phase label for every training example."""

    obs, _ = env.reset(seed=seed)
    expert.reset()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    phases: list[str] = []
    info: dict[str, Any] = {}

    while True:
        if env.raw_obs is None:
            raise RuntimeError("GraspEnv.raw_obs is unavailable after reset")
        phase = expert.phase.value
        action = expert.act(env.raw_obs)
        observations.append(obs.copy())
        actions.append(action.copy())
        phases.append(phase)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    # PandaPickAndPlace can occasionally sample a goal already within its
    # success threshold. Such one-step task wins contain no grasp behavior, so
    # we count them in expert success rate but mark them unusable for BC.
    reached_manipulation_phase = expert.phase in {
        ExpertPhase.TRANSPORT,
        ExpertPhase.PLACE,
        ExpertPhase.HOLD,
    }
    return Episode(
        observations=np.stack(observations),
        actions=np.stack(actions),
        rewards=np.asarray(rewards),
        success=bool(info.get("is_success", False)),
        seed=seed,
        usable_for_training=bool(info.get("is_success", False) and reached_manipulation_phase),
        phases=phases,
    )
