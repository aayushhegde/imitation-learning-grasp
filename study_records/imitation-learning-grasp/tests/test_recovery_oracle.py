from __future__ import annotations

import numpy as np
import pytest

from ilgrasp.experts import HistoryAwareRecoveryOracle, VerifiedExpertPhase
from ilgrasp.sim import VerifiedPlacementEnv


def test_oracle_queries_do_not_advance_and_only_actual_actions_count() -> None:
    with VerifiedPlacementEnv() as env:
        env.reset(seed=7)
        oracle = HistoryAwareRecoveryOracle(env)
        for _ in range(40):
            if oracle.phase is VerifiedExpertPhase.CLOSE:
                break
            env.step(oracle.query().action)
            oracle.observe_transition()
        assert oracle.phase is VerifiedExpertPhase.CLOSE
        before = (oracle.phase_steps, oracle.close_steps)
        first = oracle.query().action
        for _ in range(30):
            np.testing.assert_array_equal(oracle.query().action, first)
        assert (oracle.phase_steps, oracle.close_steps) == before
        for _ in range(3):
            env.step(oracle.query().action)
            oracle.observe_transition()
        assert oracle.close_steps == 3
        env.step(np.array([0.0, 0.0, 0.0, 1.0]))
        oracle.observe_transition()
        assert oracle.close_steps == 0
        assert oracle.phase is VerifiedExpertPhase.CLOSE


def test_oracle_rejects_missing_or_duplicate_observations_and_stale_reset() -> None:
    with VerifiedPlacementEnv() as env:
        env.reset(seed=7)
        oracle = HistoryAwareRecoveryOracle(env)
        env.step(oracle.query().action)
        with pytest.raises(RuntimeError, match="observe_transition"):
            oracle.query()
        oracle.observe_transition()
        with pytest.raises(RuntimeError, match="exactly once"):
            oracle.observe_transition()
        env.reset(seed=8)
        with pytest.raises(RuntimeError, match="Reset the oracle"):
            oracle.query()


def test_oracle_abstains_after_irreversible_collision() -> None:
    with VerifiedPlacementEnv() as env:
        env.reset(seed=7)
        oracle = HistoryAwareRecoveryOracle(env)
        env.last_info["barrier_violation"] = True
        label = oracle.query()
        assert not label.valid
        assert label.action is None
        assert label.reason == "irreversible_barrier_violation"


@pytest.mark.parametrize("seed", [3, 7, 13])
def test_oracle_recovers_from_executed_high_transport_perturbation(seed: int) -> None:
    with VerifiedPlacementEnv() as env:
        env.reset(seed=seed)
        oracle = HistoryAwareRecoveryOracle(env)
        perturbations = 0
        for _ in range(env.config.max_episode_steps):
            label = oracle.query()
            assert label.valid, label.reason
            action = label.action
            if oracle.phase is VerifiedExpertPhase.CROSS_BARRIER and perturbations < 3:
                action = np.array([0.3, -0.2, 0.0, -1.0], dtype=np.float32)
                perturbations += 1
            _, _, done, truncated, info = env.step(action)
            oracle.observe_transition()
            if done or truncated:
                break
        assert perturbations == 3
        assert info["strict_success"]
