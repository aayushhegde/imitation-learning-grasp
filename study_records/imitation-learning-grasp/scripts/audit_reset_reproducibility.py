"""Compare identical actions and seeds with/without retained Bullet history."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ilgrasp.experts import VerifiedPickPlaceExpert
from ilgrasp.sim import VerifiedPlacementEnv


def expert_episode(env, seed):
    initial, _ = env.reset(seed=seed)
    states, actions = [initial], []
    expert = VerifiedPickPlaceExpert()
    for _ in range(env.config.max_episode_steps):
        action = expert.act(env.raw_obs)
        state, _, done, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        if done or truncated:
            break
    return np.asarray(states), np.asarray(actions)


def replay(env, seed, actions):
    initial, _ = env.reset(seed=seed)
    states = [initial]
    # Fixed-action replay intentionally has the reference episode's horizon.
    for action in actions:
        states.append(env.step(action)[0])
    return np.asarray(states)


def compare(reference, other):
    difference = np.abs(reference - other)
    return {
        "exactly_equal": bool(np.array_equal(reference, other)),
        "reset_max_abs": float(difference[0].max()),
        "first_step_max_abs": float(difference[1].max()),
        "trajectory_max_abs": float(difference.max()),
        "object_position_max_abs_m": float(difference[:, 7:10].max()),
    }


def main():
    output = ROOT / "artifacts/revision_study"
    output.mkdir(parents=True, exist_ok=True)
    with VerifiedPlacementEnv() as env:
        reference, actions = expert_episode(env, 10000)
        expert_episode(env, 10001)
        repeated = replay(env, 10000, actions)
    with VerifiedPlacementEnv() as env:
        fresh = replay(env, 10000, actions)
    with VerifiedPlacementEnv(reset_mode="independent") as env:
        independent_first = replay(env, 10000, actions)
        expert_episode(env, 10001)
        independent_repeated = replay(env, 10000, actions)
    results = {
        "seed": 10000, "intervening_seed": 10001, "action_count": len(actions),
        "same_actions_replayed": True,
        "canonical_fresh_clients": compare(reference, fresh),
        "canonical_retained_history": compare(reference, repeated),
        "independent_repeated": compare(independent_first, independent_repeated),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pybullet", "panda-gym", "gymnasium")},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "src/ilgrasp/sim/verified_task.py")},
        "interpretation": "Scene reset preserves Bullet world history; rebuilding removes the measured order effect. This audit does not identify an individual internal solver cache.",
    }
    np.savez_compressed(output / "reset_audit_traces.npz", actions=actions, reference=reference,
                        fresh=fresh, repeated=repeated, independent_first=independent_first,
                        independent_repeated=independent_repeated)
    (output / "reset_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    assert results["canonical_fresh_clients"]["exactly_equal"]
    assert results["independent_repeated"]["exactly_equal"]


if __name__ == "__main__":
    main()
