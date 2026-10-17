"""A history-synchronized, abstaining oracle for learner-visited states.

An action query is read-only. Only ``observe_transition`` advances memory,
using the action actually applied by ``VerifiedPlacementEnv.step``. The
controller supports ordinary grasp/transport/release deviations; it abstains
after an irreversible certificate violation or outside its recovery domain.
This is an oracle implementation, not evidence of a DAgger performance gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .verified import VerifiedExpertConfig
from .verified import VerifiedExpertPhase as Phase


@dataclass(frozen=True)
class OracleLabel:
    action: np.ndarray | None
    phase: str
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.action is not None


class HistoryAwareRecoveryOracle:
    """Privileged simulator teacher with explicit query/observe semantics.

    Usage: reset the environment, construct/reset the oracle, query a label,
    execute either the learner or teacher action, then call
    ``observe_transition()`` exactly once. Do not aggregate invalid labels.
    """

    def __init__(self, env: Any, config: VerifiedExpertConfig | None = None) -> None:
        self.env = env
        self.config = config or VerifiedExpertConfig()
        self.reset()

    def reset(self) -> None:
        if self.env.raw_obs is None or self.env.last_info.get("step") != 0:
            raise RuntimeError("Reset the environment before resetting the oracle")
        self.phase = Phase.APPROACH
        self.phase_steps = 0
        self.close_steps = 0
        self.open_steps = 0
        self.grasp_offset = np.array([0.0, 0.0, self.config.grasp_offset])
        self.grasp_target: np.ndarray | None = None
        self.release_target: np.ndarray | None = None
        self._step = 0
        self._reset_count = self.env._reset_count
        self._previous_main = np.asarray(self.env.raw_obs["observation"]).copy()

    def _state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        raw = self.env.raw_obs
        main = np.asarray(raw["observation"])
        return main[:3], main[7:10], np.asarray(raw["desired_goal"]), float(main[6])

    def _held(self) -> bool:
        """Require actual object contacts on both Panda finger links."""
        sim = self.env.unwrapped.sim
        contacts = sim.physics_client.getContactPoints(
            bodyA=sim._bodies_idx["panda"], bodyB=sim._bodies_idx["object"]
        )
        links = {int(contact[3]) for contact in contacts if float(contact[8]) < 0.002}
        return {9, 10}.issubset(links)

    def _arrived(self, position: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(position - target) < self.config.position_tolerance)

    def _set_phase(self, phase: Phase) -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_steps = 0

    def _target(self) -> tuple[np.ndarray, float]:
        ee, obj, goal, _ = self._state()
        cfg = self.config
        phase = self.phase
        if phase is Phase.APPROACH:
            # Lift before a lateral pickup recovery when the hand is low.
            if np.linalg.norm(ee[:2] - obj[:2]) > 0.03 and ee[2] < 0.10:
                return np.r_[ee[:2], cfg.safe_z], cfg.open_action
            return obj + np.array([0.0, 0.0, cfg.hover_height]), cfg.open_action
        if phase is Phase.DESCEND:
            return obj + np.array([0.0, 0.0, cfg.grasp_offset]), cfg.open_action
        if phase is Phase.CLOSE:
            return self.grasp_target.copy(), cfg.close_action
        if phase is Phase.TEST_GRASP:
            return self.grasp_target + np.array([0.0, 0.0, cfg.test_lift]), cfg.close_action
        if phase is Phase.LIFT_CLEAR:
            return np.r_[ee[:2], cfg.safe_z], cfg.close_action
        if phase is Phase.PRE_CROSS:
            return np.array([goal[0], cfg.pre_cross_y, cfg.safe_z]), cfg.close_action
        if phase is Phase.CROSS_BARRIER:
            return np.array([goal[0], cfg.post_cross_y, cfg.safe_z]), cfg.close_action
        if phase is Phase.ABOVE_GOAL:
            return np.array([goal[0], goal[1], cfg.safe_z]), cfg.close_action
        if phase is Phase.LOWER:
            return goal + self.grasp_offset, cfg.close_action
        if phase is Phase.OPEN:
            return self.release_target.copy(), cfg.open_action
        if phase is Phase.RETREAT_UP:
            return np.r_[ee[:2], cfg.safe_z], cfg.open_action
        return np.asarray(cfg.home), cfg.open_action

    def query(self) -> OracleLabel:
        """Return a label without advancing timers, phases, or the simulator."""
        if self.env._reset_count != self._reset_count:
            raise RuntimeError("Reset the oracle after every environment reset")
        if self.env.last_info["step"] != self._step:
            raise RuntimeError("Call observe_transition() after every executed action")
        ee, obj, _, _ = self._state()
        info = self.env.last_info
        reason = None
        if info["barrier_violation"]:
            reason = "irreversible_barrier_violation"
        elif not np.isfinite(np.r_[ee, obj]).all():
            reason = "nonfinite_state"
        elif abs(obj[0]) > 0.25 or abs(obj[1]) > 0.30 or obj[2] < 0:
            reason = "object_outside_recovery_workspace"
        elif obj[2] < 0.055 and abs(obj[1]) < 0.065 and not self._held():
            reason = "ungrasped_object_near_barrier"
        elif (
            obj[2] > 0.065
            and not self._held()
            and self.phase not in {Phase.OPEN, Phase.RETREAT_UP, Phase.RETREAT_HOME, Phase.VERIFY}
        ):
            reason = "airborne_object_without_verified_grasp"
        if reason is not None:
            return OracleLabel(None, self.phase.value, reason)
        target, gripper = self._target()
        xyz = np.clip(self.config.action_gain * (target - ee), -1, 1)
        return OracleLabel(np.r_[xyz, gripper].astype(np.float32), self.phase.value)

    def observe_transition(self) -> None:
        """Update from the real post-step state and recorded executed action.

        In particular, querying CLOSE ten times does not count as closing the
        fingers, and a learner OPEN action resets the consecutive close count.
        """
        if self.env._reset_count != self._reset_count:
            raise RuntimeError("Reset the oracle after every environment reset")
        if self.env.last_info["step"] != self._step + 1:
            raise RuntimeError("Observe each environment transition exactly once")
        actual = self.env.last_action
        if actual is None:
            raise RuntimeError("Environment has no executed action")
        self._step += 1
        self.phase_steps += 1
        self.close_steps = self.close_steps + 1 if actual[3] < 0 else 0
        self.open_steps = self.open_steps + 1 if actual[3] > 0 else 0
        previous_ee = self._previous_main[:3]
        previous_obj = self._previous_main[7:10]
        self._previous_main = np.asarray(self.env.raw_obs["observation"]).copy()
        ee, obj, goal, width = self._state()
        cfg = self.config
        info = self.env.last_info
        held = self._held()
        in_goal = np.linalg.norm(obj - goal) < cfg.place_tolerance
        finishing = {Phase.OPEN, Phase.RETREAT_UP, Phase.RETREAT_HOME, Phase.VERIFY}
        transporting = {
            Phase.LIFT_CLEAR, Phase.PRE_CROSS, Phase.CROSS_BARRIER, Phase.ABOVE_GOAL, Phase.LOWER
        }

        # Reconcile the mode with what the learner actually accomplished.
        if info["ever_crossed"] and in_goal and width > self.env.config.open_gripper_width:
            if self.phase not in finishing:
                self.release_target = ee.copy()
                self._set_phase(Phase.OPEN)
        elif held and obj[2] > 0.05 and self.phase in {Phase.APPROACH, Phase.DESCEND, Phase.CLOSE}:
            self.grasp_offset = ee - obj
            self._set_phase(Phase.LIFT_CLEAR)
        elif not held and obj[2] < 0.04 and (
            (self.phase in transporting and not in_goal)
            or (self.phase in finishing and not in_goal)
        ):
            self.grasp_target = None
            self.release_target = None
            self._set_phase(Phase.APPROACH)
        elif (self.phase in transporting and held and ee[2] < cfg.safe_z - 0.04
              and not in_goal and self.phase is not Phase.LOWER):
            # A low learner state must lift before translating across the wall.
            self._set_phase(Phase.LIFT_CLEAR)

        phase = self.phase
        if phase is Phase.APPROACH:
            if self._arrived(previous_ee, previous_obj + np.array([0.0, 0.0, cfg.hover_height])):
                self._set_phase(Phase.DESCEND)
        elif phase is Phase.DESCEND:
            if self._arrived(previous_ee, previous_obj + np.array([0.0, 0.0, cfg.grasp_offset])):
                self.grasp_offset = previous_ee - previous_obj
                self.grasp_target = previous_ee.copy()
                self.close_steps = 0
                self._set_phase(Phase.CLOSE)
        elif phase is Phase.CLOSE:
            if self.close_steps >= cfg.close_steps:
                self._set_phase(Phase.TEST_GRASP)
        elif phase is Phase.TEST_GRASP:
            if held and previous_obj[2] > 0.05:
                self._set_phase(Phase.LIFT_CLEAR)
            elif self.phase_steps >= cfg.test_timeout:
                self._set_phase(Phase.APPROACH)
        elif phase is Phase.LIFT_CLEAR:
            if abs(previous_ee[2] - cfg.safe_z) < cfg.position_tolerance:
                self._set_phase(Phase.ABOVE_GOAL if obj[1] > cfg.post_cross_y else Phase.PRE_CROSS)
        elif phase in {Phase.PRE_CROSS, Phase.CROSS_BARRIER, Phase.ABOVE_GOAL}:
            target, _ = self._target()
            if self._arrived(previous_ee, target):
                following = {
                    Phase.PRE_CROSS: Phase.CROSS_BARRIER,
                    Phase.CROSS_BARRIER: Phase.ABOVE_GOAL,
                    Phase.ABOVE_GOAL: Phase.LOWER,
                }
                self._set_phase(following[phase])
        elif phase is Phase.LOWER:
            if np.linalg.norm(previous_obj - goal) < cfg.place_tolerance:
                # Open at the measured arrival pose. Continuing downward to
                # an old grasp offset can press the fingers into the table
                # and knock a correctly positioned cube away during release.
                self.release_target = ee.copy()
                self.open_steps = 0
                self._set_phase(Phase.OPEN)
            elif np.linalg.norm(obj[:2] - goal[:2]) > 0.06:
                self._set_phase(Phase.LIFT_CLEAR)
        elif phase is Phase.OPEN:
            if self.open_steps >= cfg.open_steps:
                self._set_phase(Phase.RETREAT_UP)
        elif phase is Phase.RETREAT_UP:
            if abs(previous_ee[2] - cfg.safe_z) < cfg.position_tolerance:
                self._set_phase(Phase.RETREAT_HOME)
        elif phase is Phase.RETREAT_HOME and np.linalg.norm(previous_ee - np.asarray(cfg.home)) < 0.025:
            self._set_phase(Phase.VERIFY)


__all__ = ["HistoryAwareRecoveryOracle", "OracleLabel"]
