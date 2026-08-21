from __future__ import annotations

import numpy as np

from ilgrasp.experts import (
    VerifiedExpertPhase,
    VerifiedPickPlaceExpert,
    collect_verified_episode,
)
from ilgrasp.sim import VerifiedPlacementEnv


def test_verified_scene_is_reproducible_and_spans_barrier() -> None:
    env = VerifiedPlacementEnv()
    try:
        first, _ = env.reset(seed=17)
        second, _ = env.reset(seed=17)
        np.testing.assert_allclose(first, second)
        assert first[8] < -0.15  # object y
        assert first[20] > 0.15  # goal y in the appended goal slots
    finally:
        env.close()


def test_native_goal_hit_is_not_verified_completion() -> None:
    env = VerifiedPlacementEnv()
    try:
        _, _ = env.reset(seed=2)
        goal = np.asarray(env.raw_obs["desired_goal"])
        base = env.unwrapped
        base.sim.set_base_pose("object", goal, np.array([0.0, 0.0, 0.0, 1.0]))
        object_id = base.sim._bodies_idx["object"]
        base.sim.physics_client.resetBaseVelocity(
            object_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
        )
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["native_success"]
        assert not info["strict_success"]
        assert not terminated
        assert not truncated
        assert not info["ever_lifted"]
        assert not info["ever_released"]
    finally:
        env.close()


def test_verified_expert_completes_every_phase_and_certificate() -> None:
    required = {phase.value for phase in VerifiedExpertPhase}
    env = VerifiedPlacementEnv()
    try:
        for seed in range(20):
            episode = collect_verified_episode(env, VerifiedPickPlaceExpert(), seed=seed)
            assert episode.success
            assert episode.usable_for_training
            assert required.issubset(set(episode.phases))
            assert 60 <= len(episode) < env.config.max_episode_steps
            assert env.last_info["ever_lifted"]
            assert env.last_info["ever_crossed"]
            assert env.last_info["ever_released"]
            assert env.last_info["retreated"]
            assert env.last_info["stable_steps"] >= env.config.required_stable_steps
            assert env.last_info["collision_free"]
            assert env.last_info["native_success_seen"]
    finally:
        env.close()
