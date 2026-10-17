"""Reproducible, persistent matched study for the verified student notebook.

Run with the documented simulator interpreter; no model selection on test episodes.
All episode results, normalization statistics, split IDs and checkpoints are retained.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import uuid
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
    relative_features,
    split_episode_ids,
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

    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
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


CONDITIONS = ("mlp22", "mlp22_floor", "mlp33", "mlp33_floor", "gru33", "gru33_floor", "expert")
KNOWN_DATASET_SHA256 = "14f4d294cdd82d251f5a7922839ba66ef965174a0782abdb0097849ecd08df7b"
KNOWN_CONDITIONS = {"mlp22", "mlp33_floor", "gru33", "gru33_floor", "expert"}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scientific_sources():
    paths = [
        "src/ilgrasp/policies/revision.py",
        "src/ilgrasp/experts/verified.py",
        "src/ilgrasp/sim/verified_task.py",
        "src/ilgrasp/sim/observation.py",
    ]
    result = {path: file_hash(ROOT / path) for path in paths}
    # Fingerprint numerical runner functions, excluding provenance/plumbing-only edits.
    tree = ast.parse(Path(__file__).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"collect", "evaluate", "failure", "summarize"}
    ]
    result["runner_numerical_ast"] = fingerprint(ast.dump(ast.Module(body=nodes, type_ignores=[])))
    return result


def training_contract(args, data, condition, seed):
    _, _, train_ids, val_ids = split_episode_ids(data.episode_ids, 0.2, args.split_seed)
    return {
        "schema": 1,
        "condition": condition,
        "seed": seed,
        "dataset_sha256": file_hash(args.output / "demos.npz"),
        "demos": args.demos,
        "split_seed": args.split_seed,
        "train_ids": train_ids.tolist(),
        "validation_ids": val_ids.tolist(),
        "epochs": args.epochs,
        "std_floor": args.floor if "floor" in condition else 1e-6,
        "input_dim": 33 if "33" in condition else 22,
        "reset_mode": "independent",
        "torch_threads": 1,
        "deterministic_algorithms": True,
        "sources": scientific_sources(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "panda-gym", "gymnasium", "pybullet")
        },
    }


def result_contract(training, checkpoint, seeds):
    return {
        "schema": 1,
        "training_contract": training,
        "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        "ordered_eval_seeds": list(seeds),
        "reset_mode": "independent",
    }


def provenance_path(path):
    return path.with_name(path.name + ".provenance.json")


def validate_provenance(path, contract):
    sidecar = provenance_path(path)
    if not sidecar.exists():
        raise ValueError(
            f"Missing provenance for {path.name}; do not reuse arbitrary legacy outputs. "
            "For this audited September revision only, use --migrate-known-revision first."
        )
    saved = json.loads(sidecar.read_text())
    if saved.get("contract") != contract or saved.get("fingerprint") != fingerprint(contract):
        raise ValueError(
            f"Configuration/source mismatch for {path.name}; use a new output directory"
        )
    if saved.get("artifact_sha256") != file_hash(path):
        raise ValueError(
            f"Artifact hash mismatch for {path.name}; refusing stale or modified output"
        )
    return saved


def write_provenance(path, contract, migration=None):
    if provenance_path(path).exists():
        validate_provenance(path, contract)
        return
    save_json(
        provenance_path(path),
        {
            "contract": contract,
            "fingerprint": fingerprint(contract),
            "artifact_sha256": file_hash(path),
            "origin": "explicit_known_run_migration" if migration else "recorded_at_creation",
            "migration_evidence": migration,
        },
    )


def validate_results(results, seeds):
    if not isinstance(results, list) or [row.get("seed") for row in results] != list(seeds):
        raise ValueError("Cached evaluation has a different ordered seed list or episode count")
    for row in results:
        if not isinstance(row.get("strict_success"), bool) or not isinstance(
            row.get("native_success_seen"), bool
        ):
            raise TypeError("Evaluation success fields must be boolean")


def atomic_save_checkpoint(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_known_revision(args, data):
    """Explicit, constrained attestation of this one previously executed session.

    This preserves checkpoint/result bytes and does NOT pretend provenance was recorded
    at creation. Missing old metadata is reconstructed only for the known dataset/run.
    """
    if (
        file_hash(args.output / "demos.npz") != KNOWN_DATASET_SHA256
        or args.demos != 200
        or args.epochs != 80
        or args.split_seed != 7
        or args.floor != 0.05
    ):
        raise ValueError(
            "Migration is restricted to the audited 200-demo,80-epoch,split7,floor.05 dataset"
        )
    if set(args.conditions) - KNOWN_CONDITIONS or set(args.train_seeds) - {0, 1, 2}:
        raise ValueError("Unknown condition/seed is outside the explicit migration authorization")
    manifests = list(args.output.glob("manifest_*.json"))
    evidence_manifests = []
    for path in manifests:
        value = json.loads(path.read_text())
        config = value.get("config", {})
        if (
            value.get("dataset_sha256") == KNOWN_DATASET_SHA256
            and config.get("demos") == 200
            and config.get("epochs") == 80
            and config.get("split_seed") == 7
            and value.get("reset_mode") == "independent"
        ):
            evidence_manifests.append({"file": path.name, "sha256": file_hash(path)})
    if not evidence_manifests:
        raise ValueError("No matching contemporaneous manifest supports known-run migration")
    logs = [(path, path.read_text(errors="replace")) for path in args.output.glob("*.log")]
    migrated = []
    for condition in args.conditions:
        for seed in [0] if condition == "expert" else args.train_seeds:
            label = f"{condition}_s{seed}"
            checkpoint = None if condition == "expert" else args.output / f"{label}.pt"
            if checkpoint is not None and not checkpoint.exists():
                continue  # A still-running old process may not have completed this model yet.
            contract = training_contract(args, data, condition, seed)
            matching_logs = [
                {"file": path.name, "sha256": file_hash(path)}
                for path, content in logs
                if re.search(rf"\b{re.escape(label)}\b", content)
            ]
            if not matching_logs:
                # No artifact for an unexecuted condition is not an error or a claim of completion.
                if condition == "expert" and not list(args.output.glob(f"{label}_eval_*.json")):
                    continue
                raise ValueError(f"No execution log corroborates {label}")
            evidence = {
                "authorization": "Explicit migration requested for this current revision session",
                "limitations": "Sources/config attribution reconstructed during audit, not metadata recorded by the legacy process",
                "manifests": evidence_manifests,
                "logs": matching_logs,
                "dataset_identity": "Exact audited dataset SHA256",
                "checks": [],
            }
            if checkpoint is not None:
                saved = torch.load(checkpoint, weights_only=True)
                for key in ("train_ids", "validation_ids", "std_floor", "epochs", "seed"):
                    if saved.get(key) != contract[key]:
                        raise ValueError(f"Legacy checkpoint {label} has conflicting {key}")
                metadata_path = args.output / f"{label}_training.json"
                if not metadata_path.exists():
                    raise ValueError(f"Training metadata missing for {label}")
                metadata = json.loads(metadata_path.read_text())
                for key in (
                    "train_ids",
                    "validation_ids",
                    "std_floor",
                    "epochs",
                    "seed",
                    "train_loss",
                    "validation_loss",
                ):
                    if metadata.get(key) != saved.get(key):
                        raise ValueError(f"Training JSON and checkpoint disagree for {label}/{key}")
                mask = np.isin(data.episode_ids, contract["train_ids"])
                x = (
                    relative_features(data.observations[mask])
                    if "33" in condition
                    else data.observations[mask]
                )
                tensor = torch.as_tensor(x, dtype=torch.float32)
                prefix = "feature" if condition.startswith("gru") else "obs"
                expected_std = tensor.std(0, unbiased=False).clamp_min(contract["std_floor"])
                if not torch.equal(saved["model"][prefix + "_mean"], tensor.mean(0)):
                    raise ValueError(
                        f"Normalization mean does not identify expected training data for {label}"
                    )
                if not torch.equal(saved["model"][prefix + "_std"], expected_std):
                    raise ValueError(
                        f"Normalization scale does not match expected training data/floor for {label}"
                    )
                evidence["checks"] = [
                    "checkpoint/JSON config and loss agreement",
                    "exact split IDs",
                    "exact training-only normalization buffers",
                    "dataset SHA and execution log",
                ]
                write_provenance(checkpoint, contract, evidence)
            for path in sorted(args.output.glob(f"{label}_eval_*.json")):
                match = re.fullmatch(rf"{re.escape(label)}_eval_(\d+)_(\d+)\.json", path.name)
                if not match:
                    continue
                start, count = map(int, match.groups())
                if (start, count) not in {(30000, 24), (40000, 100)}:
                    raise ValueError(f"Unknown historical suite {start}/{count}; migration refused")
                results = json.loads(path.read_text())
                seeds = list(range(start, start + count))
                validate_results(results, seeds)
                trace_dir = args.output / f"{label}_traces_{start}_{count}"
                if any(not (trace_dir / f"{scene}.npz").is_file() for scene in seeds):
                    raise ValueError(f"Missing saved trajectory evidence for {path.name}")
                write_provenance(path, result_contract(contract, checkpoint, seeds), evidence)
                migrated.append(path.name)
    print(
        json.dumps(
            {"migration": "explicit_known_revision", "completed_evaluations_enrolled": migrated}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/revision_study")
    parser.add_argument("--demos", type=int, default=200)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=["mlp22", "gru33", "gru33_floor"]
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--eval-start", type=int, default=40000)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument(
        "--migrate-known-revision",
        action="store_true",
        help="Only enroll verified artifacts from the explicitly audited current revision; never trains/evaluates",
    )
    args = parser.parse_args()
    if (
        args.demos < 2
        or args.epochs < 1
        or args.eval_n < 1
        or not np.isfinite(args.floor)
        or args.floor <= 0
    ):
        parser.error("Require demos>=2, epochs>=1, eval-n>=1 and a positive finite floor")
    if len(args.conditions) != len(set(args.conditions)) or len(args.train_seeds) != len(
        set(args.train_seeds)
    ):
        parser.error("Duplicate conditions or training seeds are not allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.migrate_known_revision and not (args.output / "demos.npz").exists():
        parser.error("Migration never collects new data; existing audited demos are required")
    data = collect(args.output, args.demos)
    if args.migrate_known_revision:
        migrate_known_revision(args, data)
        return
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
        "dataset_sha256": file_hash(args.output / "demos.npz"),
        "sources": scientific_sources(),
    }
    stamp = f"{args.eval_start}_{args.eval_n}"
    invocation = "_".join(args.conditions) + "_s" + "-".join(map(str, args.train_seeds))
    manifest = args.output / f"manifest_{stamp}_{invocation}_{fingerprint(provenance)[:16]}.json"
    if not manifest.exists():
        save_json(manifest, provenance)
    seeds = list(range(args.eval_start, args.eval_start + args.eval_n))
    for condition in args.conditions:
        for seed in [0] if condition == "expert" else args.train_seeds:
            label = f"{condition}_s{seed}"
            contract = training_contract(args, data, condition, seed)
            checkpoint = None if condition == "expert" else args.output / f"{label}.pt"
            result_path = args.output / f"{label}_eval_{stamp}.json"
            if checkpoint is not None and checkpoint.exists():
                validate_provenance(checkpoint, contract)
            if result_path.exists():
                if checkpoint is not None and not checkpoint.exists():
                    raise ValueError(f"Checkpoint missing for cached evaluation {label}")
                validate_provenance(result_path, result_contract(contract, checkpoint, seeds))
                results = json.loads(result_path.read_text())
                validate_results(results, seeds)
                print(label, summarize(results), flush=True)
                continue
            if condition == "expert":
                policy = VerifiedPickPlaceExpert()
            else:
                recurrent = condition.startswith("gru")
                if checkpoint.exists():
                    saved = torch.load(checkpoint, weights_only=True)
                    model = (
                        RecurrentHybridPolicy(std_floor=contract["std_floor"])
                        if recurrent
                        else HybridPolicy(
                            input_dim=contract["input_dim"], std_floor=contract["std_floor"]
                        )
                    )
                    model.load_state_dict(saved["model"])
                else:
                    start = time.time()
                    print(f"training {label} {args.epochs} epochs", flush=True)
                    common = {
                        "epochs": args.epochs,
                        "seed": seed,
                        "split_seed": args.split_seed,
                        "std_floor": contract["std_floor"],
                    }
                    trained = (
                        train_recurrent_hybrid(data, **common)
                        if recurrent
                        else train_hybrid(data, input_dim=contract["input_dim"], **common)
                    )
                    model = trained.model
                    metadata = {
                        "train_ids": trained.train_ids.tolist(),
                        "validation_ids": trained.validation_ids.tolist(),
                        "train_loss": trained.train_loss,
                        "validation_loss": trained.validation_loss,
                        "std_floor": contract["std_floor"],
                        "epochs": args.epochs,
                        "seed": seed,
                        "elapsed_seconds": time.time() - start,
                        "best_epoch": int(np.argmin(trained.validation_loss)) + 1,
                    }
                    save_json(args.output / f"{label}_training.json", metadata)
                    atomic_save_checkpoint(checkpoint, {"model": model.state_dict(), **metadata})
                    write_provenance(checkpoint, contract)
                    print(f"trained {label} in {metadata['elapsed_seconds']:.1f}s", flush=True)
                policy = RecurrentNumpyPolicy(model) if recurrent else HybridNumpyPolicy(model)
            trace_dir = args.output / f"{label}_traces_{stamp}"
            trace_dir.mkdir(exist_ok=True)
            results = evaluate(policy, seeds, trajectories=trace_dir)
            validate_results(results, seeds)
            save_json(result_path, results)
            write_provenance(result_path, result_contract(contract, checkpoint, seeds))
            print(label, summarize(results), flush=True)


if __name__ == "__main__":
    main()
