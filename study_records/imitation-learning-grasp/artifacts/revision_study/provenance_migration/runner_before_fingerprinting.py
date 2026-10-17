"""Reproducible, persistent matched study for the verified student notebook.

Run with the documented simulator interpreter; no model selection on test episodes.
All episode results, normalization statistics, split IDs and checkpoints are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ilgrasp.data import TrainingData
from ilgrasp.experts.verified import VerifiedPickPlaceExpert
from ilgrasp.policies.revision import (
    HybridNumpyPolicy,
    HybridPolicy,
    RecurrentHybridPolicy,
    RecurrentNumpyPolicy,
    train_hybrid,
    train_recurrent_hybrid,
)
from ilgrasp.sim import VerifiedPlacementEnv


def save_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported JSON type: {type(item)}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n")
    temporary.replace(path)


def collect(output, count):
    path = output / "demos.npz"
    if path.exists():
        data = np.load(path)
        if len(np.unique(data["episode_ids"])) != count:
            raise ValueError("Existing dataset size differs; use a new output directory")
        return TrainingData(**{k: data[k] for k in data.files})
    observations, actions, phases, ids, attempts = [], [], [], [], []
    successes = 0
    env = VerifiedPlacementEnv(reset_mode="independent")
    try:
        for seed in range(count * 5):
            obs, _ = env.reset(seed=seed)
            expert = VerifiedPickPlaceExpert()
            xs, ys, ps = [], [], []
            for _ in range(env.config.max_episode_steps):
                phase = expert.phase.value
                action = expert.act(env.raw_obs)
                xs.append(obs.copy())
                ys.append(action.copy())
                ps.append(phase)
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            attempts.append({"seed": seed, "n_steps": len(xs), **info})
            if info["strict_success"] and info["collision_free"]:
                observations.extend(xs)
                actions.extend(ys)
                phases.extend(ps)
                ids.extend([seed] * len(xs))
                successes += 1
            if successes % 25 == 0:
                print(f"collected {successes}/{count}", flush=True)
            if successes == count:
                break
    finally:
        env.close()
    save_json(output / "collection.json", attempts)
    if successes != count:
        raise RuntimeError(f"Only {successes}/{count} successful demonstrations")
    data = TrainingData(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(ids),
        np.asarray(phases),
    )
    np.savez_compressed(path, **asdict(data))
    return data


def failure(info):
    if info["strict_success"]:
        return "success"
    if info["barrier_violation"]:
        return "barrier_collision"
    for key, label in (
        ("ever_lifted", "grasp_failed"),
        ("ever_crossed", "dropped_before_barrier"),
        ("ever_released", "not_released"),
        ("retreated", "no_retreat"),
    ):
        if not info[key]:
            return label
    return "unstable_placement"


def evaluate(policy, seeds, reset_mode="independent", trajectories=None):
    results = []
    env = VerifiedPlacementEnv(reset_mode=reset_mode)
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            policy.reset() if hasattr(policy, "reset") else None
            xs, ys = [], []
            for _ in range(env.config.max_episode_steps):
                action = (
                    policy.act(env.raw_obs)
                    if isinstance(policy, VerifiedPickPlaceExpert)
                    else policy(obs)
                )
                xs.append(obs.copy())
                ys.append(action.copy())
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            results.append({"seed": int(seed), "failure": failure(info), **info})
            if trajectories is not None:
                np.savez_compressed(
                    trajectories / f"{seed}.npz",
                    observations=np.array(xs),
                    actions=np.array(ys),
                    final_observation=obs,
                )
    finally:
        env.close()
    return results


def summarize(results):
    n = len(results)
    k = sum(r["strict_success"] for r in results)
    p = k / n
    z = 1.95996398454
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {
        "n": n,
        "strict": k,
        "native": sum(r["native_success_seen"] for r in results),
        "wilson95": [c - m, c + m],
        "failures": dict(Counter(r["failure"] for r in results)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/revision_study")
    parser.add_argument("--demos", type=int, default=200)
    parser.add_argument("--conditions", nargs="+", default=["mlp22", "gru33", "gru33_floor"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--eval-start", type=int, default=40000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = collect(args.output, args.demos)
    provenance = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "panda-gym", "gymnasium", "pybullet")
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reset_mode": "independent",
        "transitions": len(data),
        "split_seed": args.split_seed,
        "dataset_sha256": hashlib.sha256((args.output / "demos.npz").read_bytes()).hexdigest(),
    }
    stamp = f"{args.eval_start}_{args.eval_n}"
    invocation = "_".join(args.conditions) + "_s" + "-".join(map(str, args.train_seeds))
    save_json(args.output / f"manifest_{stamp}_{invocation}.json", provenance)
    for condition in args.conditions:
        for seed in [0] if condition == "expert" else args.train_seeds:
            label = f"{condition}_s{seed}"
            result_path = args.output / f"{label}_eval_{stamp}.json"
            if result_path.exists():
                print(label, summarize(json.loads(result_path.read_text())), flush=True)
                continue
            if condition == "expert":
                policy = VerifiedPickPlaceExpert()
            else:
                floor = args.floor if "floor" in condition else 1e-6
                recurrent = condition.startswith("gru")
                input_dim = 33 if "33" in condition else 22
                checkpoint = args.output / f"{label}.pt"
                if checkpoint.exists():
                    saved = torch.load(checkpoint, weights_only=False)
                    if saved["std_floor"] != floor or saved["epochs"] != args.epochs:
                        raise ValueError(
                            "Checkpoint configuration differs; choose another directory/condition"
                        )
                    model = (
                        RecurrentHybridPolicy(std_floor=floor)
                        if recurrent
                        else HybridPolicy(input_dim=input_dim, std_floor=floor)
                    )
                    model.load_state_dict(saved["model"])
                else:
                    start = time.time()
                    print(f"training {label} {args.epochs} epochs", flush=True)
                    if recurrent:
                        trained = train_recurrent_hybrid(
                            data,
                            epochs=args.epochs,
                            seed=seed,
                            split_seed=args.split_seed,
                            std_floor=floor,
                        )
                    else:
                        trained = train_hybrid(
                            data,
                            epochs=args.epochs,
                            seed=seed,
                            split_seed=args.split_seed,
                            std_floor=floor,
                            input_dim=input_dim,
                        )
                    model = trained.model
                    metadata = {
                        "train_ids": trained.train_ids.tolist(),
                        "validation_ids": trained.validation_ids.tolist(),
                        "train_loss": trained.train_loss,
                        "validation_loss": trained.validation_loss,
                        "std_floor": floor,
                        "epochs": args.epochs,
                        "seed": seed,
                        "elapsed_seconds": time.time() - start,
                        "best_epoch": int(np.argmin(trained.validation_loss)) + 1,
                    }
                    save_json(args.output / f"{label}_training.json", metadata)
                    torch.save({"model": model.state_dict(), **metadata}, checkpoint)
                    print(f"trained {label} in {metadata['elapsed_seconds']:.1f}s", flush=True)
                policy = RecurrentNumpyPolicy(model) if recurrent else HybridNumpyPolicy(model)
            trace_dir = args.output / f"{label}_traces_{stamp}"
            trace_dir.mkdir(exist_ok=True)
            results = evaluate(
                policy,
                range(args.eval_start, args.eval_start + args.eval_n),
                trajectories=trace_dir,
            )
            save_json(result_path, results)
            print(label, summarize(results), flush=True)


if __name__ == "__main__":
    main()
