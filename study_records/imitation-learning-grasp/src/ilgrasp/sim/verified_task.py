"""A strict pick--cross--place task built on the completed Panda simulator.

Panda Gym's native pick-and-place task terminates as soon as the cube enters a
5 cm ball around the goal.  That is useful for reinforcement-learning
benchmarks, but it is not a verified placement: the gripper may still be closed
and the cube may still be in the air.  This adapter keeps the same 22-value
observation and 4-value action while requiring a complete manipulation cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from .observation import ACTION_DIM, OBS_DIM, flatten_obs


class TaskStage(str, Enum):
    PICK_AND_LIFT = "pick_and_lift"
    CROSS_BARRIER = "cross_barrier"
    PLACE = "place"
    RELEASE_AND_RETREAT = "release_and_retreat"
    VERIFY_STABILITY = "verify_stability"
    COMPLETE = "complete"


@dataclass(frozen=True)
class VerifiedTaskConfig:
    """Geometry and certificate thresholds for verified placement."""

    max_episode_steps: int = 180
    object_x_range: tuple[float, float] = (-0.09, 0.09)
    object_y_range: tuple[float, float] = (-0.20, -0.15)
    goal_x_range: tuple[float, float] = (-0.09, 0.09)
    goal_y_range: tuple[float, float] = (0.15, 0.20)
    table_object_z: float = 0.02
    barrier_half_extents: tuple[float, float, float] = (0.20, 0.018, 0.04)
    barrier_color: tuple[float, float, float, float] = (0.12, 0.32, 0.82, 1.0)
    lift_height: float = 0.10
    crossed_y: float = 0.06
    goal_xy_tolerance: float = 0.035
    goal_z_tolerance: float = 0.025
    open_gripper_width: float = 0.055
    retreat_distance: float = 0.12
    retreat_height: float = 0.12
    stable_speed: float = 0.025
    stable_spin: float = 1.0
    required_stable_steps: int = 12

    def __post_init__(self) -> None:
        if self.max_episode_steps <= 0 or self.required_stable_steps <= 0:
            raise ValueError("step limits must be positive")


class _CompletedPickPlaceTask:
    """Raw-dictionary task adapter; ``VerifiedPlacementEnv`` adds flattening."""

    def __init__(self, env: Any, config: VerifiedTaskConfig) -> None:
        self.env = env
        self.config = config
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped

        task = self.unwrapped.task
        if not hasattr(task, "distance_threshold"):
            raise RuntimeError("Verified task requires panda-gym PickAndPlace")
        self.native_distance_threshold = float(task.distance_threshold)
        # The outer task owns termination.  A negative threshold prevents the
        # native environment from declaring a terminal state mid-manipulation.
        task.distance_threshold = -1.0

        half = np.asarray(config.barrier_half_extents, dtype=float)
        with self.unwrapped.sim.no_rendering():
            self.unwrapped.sim.create_box(
                body_name="verified_barrier",
                half_extents=half,
                mass=0.0,
                position=np.array([0.0, 0.0, half[2]]),
                rgba_color=np.asarray(config.barrier_color),
            )
            # Make the three task roles visually unmistakable in recordings.
            bodies = self.unwrapped.sim._bodies_idx
            client = self.unwrapped.sim.physics_client
            client.changeVisualShape(
                bodies["object"], -1, rgbaColor=[0.98, 0.36, 0.07, 1.0]
            )
            client.changeVisualShape(
                bodies["target"], -1, rgbaColor=[0.05, 0.78, 0.63, 0.48]
            )
        self._step_count = 0
        self._stable_steps = 0
        self._ever_lifted = False
        self._ever_crossed = False
        self._ever_released = False
        self._native_success_seen = False
        self._barrier_violation = False
        self._min_crossing_clearance = float("inf")
        self._first_barrier_contact: dict[str, Any] | None = None

    def _sample_scene(self, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        cfg = self.config
        obj = np.array(
            [
                rng.uniform(*cfg.object_x_range),
                rng.uniform(*cfg.object_y_range),
                cfg.table_object_z,
            ]
        )
        goal = np.array(
            [
                rng.uniform(*cfg.goal_x_range),
                rng.uniform(*cfg.goal_y_range),
                cfg.table_object_z,
            ]
        )
        return obj, goal

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.env.reset(seed=seed, options=options)
        base = self.unwrapped
        obj, goal = self._sample_scene(seed)
        base.task.goal = goal
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        base.sim.set_base_pose("object", obj, quaternion)
        base.sim.set_base_pose("target", goal, quaternion)

        # panda-gym 3.0.7 exposes no public base-velocity reset helper.
        if not hasattr(base.sim, "_bodies_idx") or "object" not in base.sim._bodies_idx:
            raise RuntimeError("Unsupported panda-gym internals: object body not found")
        object_id = base.sim._bodies_idx["object"]
        base.sim.physics_client.resetBaseVelocity(
            object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
        )

        self._step_count = 0
        self._stable_steps = 0
        self._ever_lifted = False
        self._ever_crossed = False
        self._ever_released = False
        self._native_success_seen = False
        self._barrier_violation = False
        self._min_crossing_clearance = float("inf")
        self._first_barrier_contact = None
        obs = base._get_obs()
        return obs, self._make_info(obs, native_success=False, update=False)

    def _make_info(
        self,
        obs: dict[str, np.ndarray],
        *,
        native_success: bool,
        update: bool = True,
    ) -> dict[str, Any]:
        main = np.asarray(obs["observation"])
        ee = main[0:3]
        gripper_width = float(main[6])
        obj = main[7:10]
        velocity = main[13:16]
        angular_velocity = main[16:19]
        goal = np.asarray(obs["desired_goal"])

        xy_error = float(np.linalg.norm(obj[:2] - goal[:2]))
        z_error = float(abs(obj[2] - goal[2]))
        goal_error = float(np.linalg.norm(obj - goal))
        speed = float(np.linalg.norm(velocity))
        spin = float(np.linalg.norm(angular_velocity))
        ee_object_distance = float(np.linalg.norm(ee - obj))
        in_goal = xy_error < self.config.goal_xy_tolerance and z_error < self.config.goal_z_tolerance
        gripper_open = gripper_width > self.config.open_gripper_width
        geometrically_retreated = (
            ee_object_distance > self.config.retreat_distance
            and ee[2] > self.config.retreat_height
        )

        if update:
            self._ever_lifted |= bool(obj[2] > self.config.lift_height)
            self._ever_crossed |= bool(self._ever_lifted and obj[1] > self.config.crossed_y)
            self._ever_released |= bool(self._ever_crossed and in_goal and gripper_open)
            self._native_success_seen |= native_success

            retreated = self._ever_released and geometrically_retreated
            stable_now = (
                self._ever_released
                and in_goal
                and speed < self.config.stable_speed
                and spin < self.config.stable_spin
                and retreated
            )
            self._stable_steps = self._stable_steps + 1 if stable_now else 0
        else:
            retreated = self._ever_released and geometrically_retreated

        penetration = False
        sim = self.unwrapped.sim
        barrier_id = sim._bodies_idx["verified_barrier"]
        for contact in sim.physics_client.getContactPoints(bodyB=barrier_id):
            # PyBullet also reports close proximity with positive separation.
            if float(contact[8]) < -1e-4:
                penetration = True
                if update and self._first_barrier_contact is None:
                    if not self._ever_crossed:
                        collision_context = "before_crossing"
                    elif obj[1] < 0:
                        collision_context = "after_crossing_returned_near_side"
                    else:
                        collision_context = "after_crossing_far_side"
                    body_names = {body_id: name for name, body_id in sim._bodies_idx.items()}
                    self._first_barrier_contact = {
                        "step": self._step_count,
                        "body": body_names.get(int(contact[1]), str(contact[1])),
                        "link": int(contact[3]),
                        "penetration_depth": -float(contact[8]),
                        "object_position": obj.tolist(),
                        "ee_position": ee.tolist(),
                        "ever_crossed": bool(self._ever_crossed),
                        "context": collision_context,
                    }
                break
        if update:
            self._barrier_violation |= penetration
            half = np.asarray(self.config.barrier_half_extents)
            near_barrier = abs(float(obj[1])) < half[1] + 0.04
            if self._ever_lifted and near_barrier:
                cube_bottom = float(obj[2] - self.config.table_object_z)
                barrier_top = float(2 * half[2])
                self._min_crossing_clearance = min(
                    self._min_crossing_clearance, cube_bottom - barrier_top
                )

        complete = (
            self._stable_steps >= self.config.required_stable_steps
            and not self._barrier_violation
        )
        if complete:
            stage = TaskStage.COMPLETE
        elif self._ever_released:
            stage = TaskStage.VERIFY_STABILITY
        elif self._ever_crossed and in_goal:
            stage = TaskStage.RELEASE_AND_RETREAT
        elif self._ever_crossed:
            stage = TaskStage.PLACE
        elif self._ever_lifted:
            stage = TaskStage.CROSS_BARRIER
        else:
            stage = TaskStage.PICK_AND_LIFT

        return {
            "is_success": complete,
            "strict_success": complete,
            "native_success": native_success,
            "native_success_seen": self._native_success_seen,
            "task_stage": stage.value,
            "ever_lifted": self._ever_lifted,
            "ever_crossed": self._ever_crossed,
            "ever_released": self._ever_released,
            "retreated": retreated,
            "stable_steps": self._stable_steps,
            "required_stable_steps": self.config.required_stable_steps,
            "goal_error": goal_error,
            "xy_error": xy_error,
            "z_error": z_error,
            "object_speed": speed,
            "object_spin": spin,
            "gripper_width": gripper_width,
            "ee_object_distance": ee_object_distance,
            "barrier_penetration": penetration,
            "barrier_violation": self._barrier_violation,
            "first_barrier_contact": (
                None if self._first_barrier_contact is None else dict(self._first_barrier_contact)
            ),
            "collision_free": not self._barrier_violation,
            "min_crossing_clearance": (
                None
                if not np.isfinite(self._min_crossing_clearance)
                else self._min_crossing_clearance
            ),
            "step": self._step_count,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, _, _, native_truncated, _ = self.env.step(action)
        self._step_count += 1
        native_success = bool(
            np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            < self.native_distance_threshold
        )
        info = self._make_info(obs, native_success=native_success)
        terminated = bool(info["strict_success"])
        truncated = bool(self._step_count >= self.config.max_episode_steps or native_truncated)
        reward = -0.01 - float(info["goal_error"])
        if terminated:
            reward += 5.0
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class VerifiedPlacementEnv:
    """Flat-observation interface for the completed obstacle placement task."""

    def __init__(
        self,
        render_mode: str | None = None,
        config: VerifiedTaskConfig | None = None,
        env_id: str = "PandaPickAndPlace-v3",
        reset_mode: Literal["canonical", "independent"] = "canonical",
    ) -> None:
        if reset_mode not in {"canonical", "independent"}:
            raise ValueError("reset_mode must be 'canonical' or 'independent'")
        self.config = config or VerifiedTaskConfig()
        self.reset_mode = reset_mode
        self._render_mode = render_mode
        self._env_id = env_id
        self._reset_count = 0
        self._task = self._create_task()
        self.action_space = self._task.action_space
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.raw_obs: dict[str, np.ndarray] | None = None
        self.last_info: dict[str, Any] = {}
        self.last_action: np.ndarray | None = None

    def _create_task(self) -> _CompletedPickPlaceTask:
        try:
            import gymnasium as gym
            import panda_gym  # noqa: F401 -- registers Panda environments
        except ImportError as exc:
            raise ImportError("Install the simulator dependencies from README.md") from exc

        kwargs: dict[str, Any] = {
            "max_episode_steps": self.config.max_episode_steps + 20,
        }
        if self._render_mode is not None:
            kwargs["render_mode"] = self._render_mode
        base_env = gym.make(self._env_id, **kwargs)
        return _CompletedPickPlaceTask(base_env, self.config)

    @property
    def unwrapped(self) -> Any:
        return self._task.unwrapped

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the scene under an explicit, recorded physics protocol.

        ``canonical`` retains the historical simulator client, so evaluation
        must use a new environment for each condition and the identical seed
        sequence. ``independent`` rebuilds the Bullet world between episodes,
        removing solver/contact history; it is a different evaluation protocol.
        """
        if self.reset_mode == "independent" and self._reset_count:
            self._task.close()
            self._task = self._create_task()
            self.action_space = self._task.action_space
        raw_obs, info = self._task.reset(seed=seed)
        self._reset_count += 1
        info["reset_mode"] = self.reset_mode
        self.raw_obs = raw_obs
        self.last_info = info
        self.last_action = None
        return flatten_obs(raw_obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape ({ACTION_DIM},), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        raw_obs, reward, terminated, truncated, info = self._task.step(action)
        self.last_action = action.copy()
        info["reset_mode"] = self.reset_mode
        self.raw_obs = raw_obs
        self.last_info = info
        return flatten_obs(raw_obs), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        frame = self._task.render()
        if frame is None:
            raise RuntimeError("Construct with render_mode='rgb_array' to record frames")
        return np.asarray(frame)

    def render_view(
        self,
        *,
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray | None = None,
        distance: float = 1.2,
        yaw: float = 55,
        pitch: float = -25,
        roll: float = 0,
    ) -> np.ndarray:
        frame = self.unwrapped.sim.render(
            width=width,
            height=height,
            target_position=(
                np.array([-0.12, 0.0, 0.14])
                if target_position is None
                else np.asarray(target_position)
            ),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
        if frame is None:
            raise RuntimeError("Construct with render_mode='rgb_array' to record frames")
        return np.asarray(frame)

    def close(self) -> None:
        self._task.close()

    def __enter__(self) -> "VerifiedPlacementEnv":  # noqa: PYI034, UP037
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["TaskStage", "VerifiedPlacementEnv", "VerifiedTaskConfig"]
