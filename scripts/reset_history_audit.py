"""Reset-history audit: does an earlier rollout change a later replay of the same scene?

For each scene seed and each reference trajectory (a strict expert success and a deliberate
pre-crossing collision failure), the script

1. records the command sequence and the observation trajectory in a freshly constructed
   simulator (the reference);
2. replays exactly those commands open loop after a chosen intervening history, either in
   the same simulator instance ("reused") or in a newly constructed one ("rebuilt");
3. re-runs the closed-loop controller under the same two lifecycles, to see whether the
   episode outcome or length changes.

Two trajectories are called identical only if every float32 observation value and every
monitor flag matches exactly (tolerance 0). Otherwise the maximum absolute deviation is
reported. "Rebuilt" means: close the PyBullet client, construct a new VerifiedPlacementEnv
(gym.make -> new DIRECT client, resetSimulation, timestep 1/500 s, 20 substeps, gravity
-9.81 m/s^2, plane, table, Panda URDF, cube, ghost target, barrier), then reset(seed).

Usage (from the repository root):
    python scripts/reset_history_audit.py --scenes 10000 10001 10002 10003 10004 \
        --out output/reset_audit
    python scripts/reset_history_audit.py --original-case --out output/reset_audit

The second command repeats the configuration of the first audit (a 73-command expert
episode at scene 10000 replayed after an intervening expert rollout at scene 10001).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_predicate_checks import LowCrossing, Scenario  # noqa: E402

from ilgrasp.sim.verified_task import VerifiedPlacementEnv  # noqa: E402

FLAGS = ("ever_lifted", "ever_crossed", "ever_released", "retreated", "stable_steps",
         "barrier_violation", "strict_success")
PHYSICS_KEYS = ("fixedTimeStep", "numSubSteps", "numSolverIterations", "gravityAccelerationX",
                "gravityAccelerationY", "gravityAccelerationZ", "contactBreakingThreshold",
                "erp", "contactERP", "frictionERP", "enableConeFriction", "deterministicOverlappingPairs",
                "allowedCcdPenetration", "useSplitImpulse", "splitImpulsePenetrationThreshold")


def physics_params(env: VerifiedPlacementEnv) -> dict[str, Any]:
    client = env.unwrapped.sim.physics_client
    params = client.getPhysicsEngineParameters()
    out = {k: params[k] for k in PHYSICS_KEYS if k in params}
    out["panda_gym_substeps"] = env.unwrapped.sim.n_substeps
    out["panda_gym_timestep"] = env.unwrapped.sim.timestep
    return out


def closed_loop(env: VerifiedPlacementEnv, make: Callable[[], Scenario], seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    controller = make()
    observations, commands, flags = [obs.copy()], [], []
    for _ in range(env.config.max_episode_steps):
        command = controller.act(env.raw_obs, info)
        obs, _, terminated, truncated, info = env.step(command)
        observations.append(obs.copy())
        commands.append(np.asarray(command, dtype=np.float32).copy())
        flags.append([info[k] for k in FLAGS])
        if terminated or truncated:
            break
    return {"obs": np.stack(observations), "cmd": np.stack(commands),
            "flags": np.asarray(flags, dtype=float), "strict": bool(info["strict_success"])}


def open_loop(env: VerifiedPlacementEnv, commands: np.ndarray, seed: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    observations, flags = [obs.copy()], []
    info: dict[str, Any] = {}
    for command in commands:
        obs, _, _, _, info = env.step(command)
        observations.append(obs.copy())
        flags.append([info[k] for k in FLAGS])
    return {"obs": np.stack(observations), "flags": np.asarray(flags, dtype=float),
            "strict": bool(info.get("strict_success", False))}


def random_rollout(env: VerifiedPlacementEnv, seed: int, rng_seed: int) -> None:
    env.reset(seed=seed)
    rng = np.random.default_rng(rng_seed)
    for _ in range(env.config.max_episode_steps):
        env.step(rng.uniform(-1.0, 1.0, size=4).astype(np.float32))


def make_history(name: str, scene: int) -> Callable[[VerifiedPlacementEnv], None]:
    """Intervening histories; each uses scenes outside the audited set."""
    other = scene + 50
    if name == "none":
        return lambda env: None
    if name == "expert_success":
        return lambda env: closed_loop(env, Scenario, other)
    if name == "collision_failure":
        return lambda env: closed_loop(env, LowCrossing, other)
    if name == "random_commands":
        return lambda env: random_rollout(env, other, rng_seed=scene)
    if name == "two_expert_rollouts":
        return lambda env: (closed_loop(env, Scenario, other), closed_loop(env, Scenario, other + 1))
    raise ValueError(name)


HISTORIES = ("none", "expert_success", "collision_failure", "random_commands", "two_expert_rollouts")
TRAJECTORIES: dict[str, Callable[[], Scenario]] = {"expert_success": Scenario,
                                                   "collision_failure": LowCrossing}


def compare(ref: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    n = min(len(ref["obs"]), len(run["obs"]))
    diff = np.abs(ref["obs"][:n].astype(np.float64) - run["obs"][:n].astype(np.float64))
    per_step = diff.max(axis=1)
    divergent = np.nonzero(per_step > 0)[0]
    same_len = len(ref["obs"]) == len(run["obs"])
    flags_equal = same_len and np.array_equal(ref["flags"], run["flags"])
    return {
        "reset_identical": bool(np.array_equal(ref["obs"][0], run["obs"][0])),
        "identical": bool(same_len and np.array_equal(ref["obs"], run["obs"]) and flags_equal),
        "first_divergent_step": int(divergent[0]) if divergent.size else None,
        "max_abs_obs_diff": float(diff.max()),
        "max_abs_object_pos_diff_m": float(diff[:, 7:10].max()),
        "length_ref": len(ref["obs"]) - 1,
        "length_run": len(run["obs"]) - 1,
        "strict_ref": ref["strict"],
        "strict_run": run["strict"],
        "per_step": per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, nargs="+", default=[10000, 10001, 10002, 10003, 10004])
    parser.add_argument("--out", type=Path, default=Path("output/reset_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    traces_dir = args.out / "traces"
    traces_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    physics: dict[str, Any] = {}
    for scene in args.scenes:
        for traj_name, make in TRAJECTORIES.items():
            env = VerifiedPlacementEnv()
            ref = closed_loop(env, make, scene)
            physics.setdefault("reference", physics_params(env))
            env.close()
            np.savez_compressed(traces_dir / f"ref_{traj_name}_scene{scene}.npz",
                                obs=ref["obs"], cmd=ref["cmd"], flags=ref["flags"])
            for hist in HISTORIES:
                for lifecycle in ("reused", "rebuilt"):
                    if hist == "none" and lifecycle == "rebuilt":
                        continue  # identical to reused with no history: a fresh client
                    for loop in ("open_loop_replay", "closed_loop_rerun"):
                        env = VerifiedPlacementEnv()
                        make_history(hist, scene)(env)
                        if lifecycle == "rebuilt":
                            env.close()
                            env = VerifiedPlacementEnv()
                        params = physics_params(env)
                        run = (open_loop(env, ref["cmd"], scene) if loop == "open_loop_replay"
                               else closed_loop(env, make, scene))
                        env.close()
                        result = compare(ref, run)
                        per_step = result.pop("per_step")
                        if not result["identical"]:
                            np.savez_compressed(
                                traces_dir / f"{traj_name}_scene{scene}_{hist}_{lifecycle}_{loop}.npz",
                                ref_obs=ref["obs"], run_obs=run["obs"], max_abs_diff_per_step=per_step)
                        rows.append({"scene": scene, "trajectory": traj_name, "history": hist,
                                     "lifecycle": lifecycle, "mode": loop,
                                     "physics_params_equal": params == physics["reference"],
                                     **result})
                        print(f"scene={scene} traj={traj_name:17s} hist={hist:19s} {lifecycle:7s} "
                              f"{loop:17s} identical={int(result['identical'])} "
                              f"maxdiff={result['max_abs_obs_diff']:.3g} "
                              f"objdiff={result['max_abs_object_pos_diff_m']:.3g} m "
                              f"len {result['length_ref']}->{result['length_run']} "
                              f"strict {int(result['strict_ref'])}->{int(result['strict_run'])}",
                              flush=True)

    versions = {pkg: metadata.version(pkg) for pkg in ("gymnasium", "panda-gym", "pybullet", "numpy")}
    with open(args.out / "reset_audit_summary.json", "w") as f:
        json.dump({"platform": platform.platform(), "machine": platform.machine(),
                   "python": platform.python_version(), "versions": versions,
                   "physics_reference": physics["reference"], "tolerance": 0.0, "rows": rows},
                  f, indent=2, default=str)
    with open(args.out / "reset_audit_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_case(out: Path) -> None:
    """The configuration of the first audit: expert at 10000, intervening rollout at 10001."""
    traces_dir = out / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    env = VerifiedPlacementEnv()
    ref = closed_loop(env, Scenario, 10000)
    env.close()
    results = {"reference_commands": len(ref["cmd"])}
    for lifecycle in ("reused", "rebuilt"):
        env = VerifiedPlacementEnv()
        closed_loop(env, Scenario, 10001)
        if lifecycle == "rebuilt":
            env.close()
            env = VerifiedPlacementEnv()
        run = open_loop(env, ref["cmd"], 10000)
        env.close()
        result = compare(ref, run)
        per_step = result.pop("per_step")
        np.savez_compressed(traces_dir / f"original_case_scene10000_after10001_{lifecycle}.npz",
                            ref_obs=ref["obs"], run_obs=run["obs"], max_abs_diff_per_step=per_step)
        results[lifecycle] = result
    fresh = []
    for _ in range(2):
        env = VerifiedPlacementEnv()
        fresh.append(open_loop(env, ref["cmd"], 10000))
        env.close()
    results["two_fresh_clients_identical"] = all(
        np.array_equal(r["obs"], ref["obs"]) for r in fresh)
    with open(out / "original_case.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    if "--original-case" in sys.argv:
        sys.argv.remove("--original-case")
        out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(
            "output/reset_audit")
        original_case(out_dir)
    else:
        main()
